"""Streaming zeta reduction and HDF5 output (ZetaWriter).

What has to hold:

* the streamed zeta is *exactly* what calculate_all_zetas produces from the
  same map-sets stacked -- reducing one map-set at a time must not change
  the estimator (it cannot: zeta averages over patches, not over maps);
* the three input routes -- reduce on device, reduce in the writer thread
  (multi-GPU / CPU), and no reduction at all -- agree bitwise;
* a MultiDeviceIntegrated3PCF works unchanged, on the CPU-reduction route;
* the flushed prefix survives a crash, and `resume` continues from it;
* failures in the writer thread reach the caller instead of being swallowed.
"""

import os
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from unittest.mock import patch

import h5py
import healpy as hp
import numpy as np

from CosmoFuse import Integrated3PCF, ZetaWriter, calculate_all_zetas
from CosmoFuse.multi_device import MultiDeviceIntegrated3PCF

NSIDE = 16
NPIX = hp.nside2npix(NSIDE)


def make_corr(devices=None, n_patches=4):
    theta_pix, _ = hp.pix2ang(NSIDE, np.arange(NPIX))
    mask = (theta_pix < 1.4).astype(float)
    rng = np.random.default_rng(11)
    phi = rng.uniform(0.2, 1.1, size=n_patches)
    theta = rng.uniform(0.7, 1.05, size=n_patches)
    corr = Integrated3PCF(
        NSIDE, phi, theta,
        nbins=3, theta_min=100, theta_max=400, patch_size=250, theta_Q=250,
        mask=mask, map_precision="float64", rotation_precision="float64",
        device="cpu" if devices is None else devices,
    )
    corr.preprocess()
    return corr


def make_maps(corr, nmaps=5, seed=3):
    rng = np.random.default_rng(seed)
    n = corr.n_active
    return [
        (rng.normal(size=(2, 2, n)), rng.uniform(0.5, 2.0, size=(2, n)))
        for _ in range(nmaps)
    ]


def fake_cupy():
    """A stand-in for cupy so the device branch of _prepare is exercised.

    Only what the reduction touches: the arrays must *look* like they live
    on a device (``type(x).__module__ == "cupy"``), and the module must
    offer the handful of functions `_xp` dispatches to.
    """

    class _DeviceArray(np.ndarray):
        pass

    _DeviceArray.__module__ = "cupy"

    def _wrap(a, **kw):
        arr = np.asarray(a, **kw)
        return arr if isinstance(arr, _DeviceArray) else arr.view(_DeviceArray)

    mod = types.ModuleType("cupy")
    mod.asarray = _wrap
    mod.zeros = lambda *a, **k: _wrap(np.zeros(*a, **k))
    mod.mean = lambda *a, **k: _wrap(np.mean(*a, **k))
    mod.result_type = np.result_type
    # the reduction concatenates the centres and the annuli once so that all
    # eight estimators share one gather
    mod.concatenate = lambda arrays, **kw: _wrap(np.concatenate(arrays, **kw))
    mod.asnumpy = lambda a: np.asarray(a).view(np.ndarray).copy()
    return mod, _DeviceArray


def read_zetas(path, **kw):
    with h5py.File(path, "r", **kw) as f:
        return {k: f["zeta"][k][:] for k in f["zeta"]}, dict(f.attrs), int(
            f["n_flushed"][()]
        )


def _on_device(result, DeviceArray):
    """The same result with every array wrapped as a (fake) device array."""
    import dataclasses

    return dataclasses.replace(
        result, **{k: np.asarray(v).view(DeviceArray) for k, v in result.fields().items()}
    )


class ZetaWriterBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corr = make_corr()
        cls.maps = make_maps(cls.corr)
        cls.results = [cls.corr.gg(g, w, return_device=False) for g, w in cls.maps]
        # the reference: every map-set stacked, reduced in one go
        cls.reference = calculate_all_zetas(
            M_a=np.stack([np.asarray(r.M_a) for r in cls.results]),
            xi_p=np.stack([np.asarray(r.xi_p) for r in cls.results]),
            xi_m=np.stack([np.asarray(r.xi_m) for r in cls.results]),
        )

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "zeta.h5")

    def tearDown(self):
        self._tmp.cleanup()

    def assert_matches_reference(self, got, ref=None):
        ref = self.reference if ref is None else ref
        self.assertEqual(set(got), set(ref))
        for key in ref:
            self.assertTrue(np.array_equal(got[key], np.asarray(ref[key])), key)


class TestStreamedReduction(ZetaWriterBase):
    def test_streamed_zeta_equals_a_single_batch_reduction(self):
        with ZetaWriter(self.path, self.corr, flush_every=2) as out:
            for r in self.results:
                out.submit(r)
            self.assertEqual(out.n_submitted, len(self.results))
        got, attrs, n_flushed = read_zetas(self.path)
        self.assert_matches_reference(got)
        self.assertEqual(n_flushed, len(self.results))
        self.assertEqual(
            next(iter(got.values())).shape[0], len(self.results)
        )

    def test_device_route_matches_host_route(self):
        """Reducing on the device (before the copy) and reducing in the
        writer thread must give the same numbers."""
        mod, DeviceArray = fake_cupy()
        dev_path = os.path.join(self._tmp.name, "dev.h5")
        with patch.dict(sys.modules, {"cupy": mod}):
            with ZetaWriter(dev_path, self.corr) as out:
                for r in self.results:
                    out.submit(_on_device(r, DeviceArray))
        device_got, _, _ = read_zetas(dev_path)
        self.assert_matches_reference(device_got)

        with ZetaWriter(self.path, self.corr) as out:
            for r in self.results:
                out.submit(r)
        host_got, _, _ = read_zetas(self.path)
        for key in host_got:
            self.assertTrue(np.array_equal(device_got[key], host_got[key]), key)

    def test_device_route_reduces_before_copying(self):
        """The whole point: what crosses PCIe is the 9 kB data vector, not
        the ~1 MB per-patch arrays."""
        mod, DeviceArray = fake_cupy()
        copied = []
        mod.asnumpy = lambda a, _c=copied: (
            _c.append(np.asarray(a).size), np.asarray(a).view(np.ndarray).copy()
        )[1]
        with patch.dict(sys.modules, {"cupy": mod}):
            with ZetaWriter(self.path, self.corr) as out:
                out.submit(_on_device(self.results[0], DeviceArray))
        raw_size = sum(np.asarray(a).size for a in self.results[0].fields().values())
        self.assertGreater(raw_size, 0)
        self.assertLess(sum(copied), raw_size)

    def test_raw_mode_stores_the_per_patch_arrays(self):
        with ZetaWriter(self.path, self.corr, reduce="none") as out:
            for r in self.results:
                out.submit(r)
        with h5py.File(self.path, "r") as f:
            self.assertNotIn("zeta", f)
            for name in ("M_a", "xi_p", "xi_m"):
                stored = f["raw"][name][:]
                want = np.stack([np.asarray(getattr(r, name)) for r in self.results])
                self.assertTrue(np.array_equal(stored, want), name)
        # ... and zeta is still recoverable from them
        with h5py.File(self.path, "r") as f:
            got = calculate_all_zetas(
                M_a=f["raw"]["M_a"][:],
                xi_p=f["raw"]["xi_p"][:],
                xi_m=f["raw"]["xi_m"][:],
            )
        self.assert_matches_reference(got)

    def test_provenance_is_written(self):
        with ZetaWriter(self.path, self.corr) as out:
            out.submit(self.results[0])
        _, attrs, _ = read_zetas(self.path)
        for key in ("nside", "nbins", "n_patches", "theta_min", "theta_max",
                    "level_table", "row_pix_hash", "cosmofuse_version"):
            self.assertIn(key, attrs)
        self.assertEqual(int(attrs["nside"]), NSIDE)
        self.assertEqual(int(attrs["n_patches"]), self.corr.n_patches)


class TestMultiDevice(ZetaWriterBase):
    """A patch-parallel group returns host arrays, so the reduction runs on
    the CPU in the writer thread -- the route the group must work on."""

    def test_group_matches_single_device(self):
        multi = make_corr(devices=["cpu", "cpu"])
        self.assertIsInstance(multi, MultiDeviceIntegrated3PCF)
        with ZetaWriter(self.path, multi) as out:
            for g, w in self.maps:
                out.submit(multi.gg(g, w))
        got, attrs, _ = read_zetas(self.path)
        self.assert_matches_reference(got)
        # provenance survives the __getattr__ forwarding of the group
        self.assertEqual(int(attrs["n_patches"]), multi.n_patches)
        self.assertIn("level_table", attrs)

    def test_group_input_is_reduced_in_the_writer_thread(self):
        multi = make_corr(devices=["cpu", "cpu"])
        with ZetaWriter(self.path, multi) as out:
            item = out._prepare({"M_a": np.zeros((2, self.corr.n_patches))})
            self.assertIn("raw", item)      # not reduced on the calling thread
            out.submit(multi.gg(*self.maps[0]))


class TestDurability(ZetaWriterBase):
    def test_n_flushed_tracks_the_durable_prefix(self):
        with ZetaWriter(self.path, self.corr, flush_every=2) as out:
            for r in self.results[:3]:
                out.submit(r)
            out._queue.join()
            self.assertEqual(out.n_written, 3)
        _, _, n_flushed = read_zetas(self.path)
        self.assertEqual(n_flushed, 3)     # close() flushes the remainder

    def test_survives_a_hard_kill_and_resumes(self):
        prog = textwrap.dedent(f"""
            import os, sys
            sys.path.insert(0, {os.path.join(os.getcwd(), 'src')!r})
            sys.path.insert(0, {os.getcwd()!r})
            import numpy as np
            from tests.test_zeta_writer import make_corr, make_maps
            from CosmoFuse import ZetaWriter
            corr = make_corr()
            out = ZetaWriter({self.path!r}, corr, flush_every=2)
            for g, w in make_maps(corr):
                out.submit(corr.gg(g, w, return_device=False))
            out._queue.join()
            os._exit(1)                      # no close(), no flush of the tail
        """)
        env = dict(os.environ, NUMBA_DISABLE_JIT="1")
        subprocess.run([sys.executable, "-c", prog], check=False, env=env)

        # the flushed prefix is intact and readable by an ordinary reader
        got, _, n_flushed = read_zetas(self.path)
        self.assertGreaterEqual(n_flushed, 2)
        self.assertEqual(n_flushed % 2, 0)
        for key in got:
            self.assertTrue(np.all(np.isfinite(got[key][:n_flushed])))

        # resuming truncates anything past the last flush and continues
        with ZetaWriter(self.path, self.corr, flush_every=2, resume=True) as out:
            for r in self.results[n_flushed:]:
                out.submit(r)
        got, _, _ = read_zetas(self.path)
        self.assert_matches_reference(got)


class TestErrorsAndBackpressure(ZetaWriterBase):
    def test_backpressure_does_not_deadlock(self):
        with ZetaWriter(self.path, self.corr, depth=1, flush_every=1) as out:
            for r in self.results * 3:
                out.submit(r)
        got, _, n = read_zetas(self.path)
        self.assertEqual(n, 3 * len(self.results))

    def test_writer_failure_reaches_the_caller(self):
        """A map-set with a different number of tomographic bins reduces
        fine but cannot be appended: 3 bins give 10 zeta triplets where the
        file holds 4.  The writer thread must not swallow that."""
        npatch, nbins = self.corr.n_patches, self.corr.nbins
        other = dict(
            M_a=np.zeros((3, npatch)),             # 3 tomographic bins
            xi_p=np.zeros((6, npatch, nbins)),
            xi_m=np.zeros((6, npatch, nbins)),
        )
        writer = ZetaWriter(self.path, self.corr)
        writer.submit(self.results[0])
        writer._queue.join()
        writer.submit(**other)               # fails in the thread
        writer._queue.join()
        with self.assertRaisesRegex(RuntimeError, "thread failed"):
            writer.submit(self.results[1])
        with self.assertRaisesRegex(RuntimeError, "thread failed"):
            writer.close()
        self.assertIsNone(writer._file)            # handle released anyway

    def test_rejects_incomplete_or_unknown_fields(self):
        with ZetaWriter(self.path, self.corr) as out:
            with self.assertRaisesRegex(ValueError, "unknown field"):
                out.submit(M_a=np.zeros((2, 4)), nonsense=1)
            with self.assertRaisesRegex(ValueError, "central field"):
                out.submit(xi_p=np.zeros((3, 4, 3)))
            with self.assertRaisesRegex(ValueError, "annular field"):
                out.submit(M_a=np.zeros((2, 4)))
            out.submit(self.results[0])

    def test_result_or_fields_not_both(self):
        with ZetaWriter(self.path, self.corr) as out:
            with self.assertRaisesRegex(ValueError, "not both"):
                out.submit(self.results[0], M_a=self.results[0].M_a)
            out.submit(**self.results[0].fields())      # keyword form still works
            out.submit(self.results[0])


class TestSWMR(ZetaWriterBase):
    def test_reader_follows_a_growing_file(self):
        with ZetaWriter(self.path, self.corr, flush_every=1, swmr=True) as out:
            out.submit(self.results[0])
            out._queue.join()
            with h5py.File(self.path, "r", swmr=True, libver="latest") as f:
                self.assertEqual(int(f["n_flushed"][()]), 1)
            out.submit(self.results[1])
            out._queue.join()
            with h5py.File(self.path, "r", swmr=True, libver="latest") as f:
                f["n_flushed"].refresh()
                self.assertEqual(int(f["n_flushed"][()]), 2)
        got, _, _ = read_zetas(self.path, swmr=True, libver="latest")
        self.assertEqual(next(iter(got.values())).shape[0], 2)

    def test_default_is_off(self):
        with ZetaWriter(self.path, self.corr) as out:
            out.submit(self.results[0])
            self.assertFalse(out.swmr)
        # a plain reader opens it with no special flags
        with h5py.File(self.path, "r") as f:
            self.assertEqual(int(f["n_flushed"][()]), 1)


if __name__ == "__main__":
    unittest.main()
