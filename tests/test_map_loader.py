"""T12 -- ring-buffer map loader: identical to the serial loop, in order,
under randomised thread timing; archive/geometry mismatch is caught."""

import threading
import time
import unittest

import healpy as hp
import numpy as np

from CosmoFuse import Correlation, MapFileLoader

NSIDE = 32
NPIX = hp.nside2npix(NSIDE)


def make_corr():
    theta, phi = hp.pix2ang(NSIDE, np.arange(NPIX))
    mask = (np.degrees(phi) < 140.0) & (np.abs(90 - np.degrees(theta)) < 40)
    corr = Correlation(
        NSIDE, np.radians([40.0, 70.0]), np.radians([90.0, 100.0]), nbins=3,
        theta_min=150, theta_max=900, patch_size=500, theta_Q=150, mask=mask,
        device="cpu", rotation_precision="float64", resolution_factor=2.0,
    )
    corr.preprocess()
    return corr, mask


class TestMapFileLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corr, cls.mask = make_corr()
        rng = np.random.default_rng(5)
        cls.nmaps = 23
        cls.archive = rng.normal(size=(cls.nmaps, 2, 2, cls.corr.n_active)) * 0.05
        cls.w = np.ones((2, cls.corr.n_active))
        cls.serial = [
            cls.corr.get_full_tomo_shear(m, cls.w, return_device=False) for m in cls.archive
        ]

    def run_loader(self, n_slots, n_readers, jitter):
        rng = np.random.default_rng(n_slots * 10 + n_readers)
        lock = threading.Lock()
        active = {"now": 0, "max": 0}

        def read(source, out):
            with lock:
                active["now"] += 1
                active["max"] = max(active["max"], active["now"])
                delay = float(rng.uniform(0, jitter))
            time.sleep(delay)
            out["shear"][...] = self.archive[source]
            with lock:
                active["now"] -= 1

        loader = MapFileLoader(
            self.corr, {"shear": (2, 2, self.corr.n_active)}, sources=range(self.nmaps),
            read_fn=read, n_slots=n_slots, n_readers=n_readers,
            row_pix_hash=self.corr.row_pix_hash,
        )
        self.assertEqual(len(loader), self.nmaps)
        order, results = [], []
        for k, dev in loader:
            order.append(k)
            if k % 5 == 0:
                time.sleep(jitter)  # a slow consumer must not lose maps
            results.append(self.corr.get_full_tomo_shear(dev["shear"], self.w, return_device=False))
        self.assertEqual(order, list(range(self.nmaps)))
        for got, want in zip(results, self.serial):
            for a, b in zip(got, want):
                self.assertTrue(np.array_equal(a, b))
        self.assertLessEqual(active["max"], min(n_readers, n_slots - 1))
        return active["max"]

    def test_identical_to_serial_loop_under_random_timing(self):
        for n_slots, n_readers in ((2, 1), (3, 4), (4, 2), (8, 3)):
            with self.subTest(n_slots=n_slots, n_readers=n_readers):
                self.run_loader(n_slots, n_readers, jitter=0.004)

    def test_readers_run_concurrently(self):
        self.assertGreater(self.run_loader(6, 4, jitter=0.01), 1)

    def test_asynchronous_streams_worst_case_lag(self):
        """CUDA-stream simulator: nothing runs until something synchronises,
        and every copy reads its source when it executes.  Catches host
        buffers refilled before their upload ran, device buffers overwritten
        before the kernels that use them ran, and missing dependencies."""
        from types import SimpleNamespace

        from . import cuda_stream_sim as sim

        nmaps = 40
        rng = np.random.default_rng(9)
        archive = rng.normal(size=(nmaps, 3, 50))
        for n_slots, n_readers, sched_seed in ((2, 1, 0), (3, 2, 1), (4, 4, 2), (3, 1, 3), (6, 2, 4)):
            with self.subTest(n_slots=n_slots, n_readers=n_readers):
                sim.seed(sched_seed)
                corr = SimpleNamespace(
                    backend=sim.SimBackend(), map_dtype=np.dtype(np.float64), row_pix_hash="x")
                jitter = np.random.default_rng(n_slots)

                def read(source, out):
                    time.sleep(float(jitter.uniform(0, 0.002)))
                    out["m"][...] = archive[source]

                loader = MapFileLoader(
                    corr, {"m": (3, 50)}, range(nmaps), read, n_slots=n_slots, n_readers=n_readers)
                seen = []
                for _k, dev in loader:
                    sim.enqueue_kernel(dev["m"], seen)  # runs (much) later
                sim.synchronize_all()
                self.assertEqual(len(seen), nmaps)
                for got, want in zip(seen, archive):
                    self.assertTrue(np.array_equal(got, want))

    def test_map_loader_worst_case_lag(self):
        """Same hazard in the two-slot MapLoader."""
        from types import SimpleNamespace

        from CosmoFuse import MapLoader

        from . import cuda_stream_sim as sim

        rng = np.random.default_rng(4)
        archive = rng.normal(size=(12, 3, 20))
        for sched_seed in range(6):
            with self.subTest(sched_seed=sched_seed):
                sim.seed(sched_seed)
                corr = SimpleNamespace(backend=sim.SimBackend(), map_dtype=np.dtype(np.float64))
                pipe = MapLoader(corr, {"m": (3, 20)})
                seen = []
                dev = pipe.wait(pipe.stage({"m": archive[0]}))
                for k in range(len(archive)):
                    nxt = pipe.stage({"m": archive[k + 1]}) if k + 1 < len(archive) else None
                    sim.enqueue_kernel(dev["m"], seen)
                    dev = pipe.wait(nxt)
                sim.synchronize_all()
                for got, want in zip(seen, archive):
                    self.assertTrue(np.array_equal(got, want))

    def test_reader_errors_propagate_and_threads_stop(self):
        def read(source, out):
            if source == 5:
                raise OSError("corrupt map file")
            out["shear"][...] = self.archive[source]

        before = threading.active_count()
        loader = MapFileLoader(
            self.corr, {"shear": (2, 2, self.corr.n_active)}, sources=range(self.nmaps),
            read_fn=read, n_slots=3, n_readers=2)
        with self.assertRaisesRegex(OSError, "corrupt map file"):
            for _k, _dev in loader:
                pass
        self.assertEqual(threading.active_count(), before)

    def test_early_exit_stops_threads(self):
        before = threading.active_count()
        loader = MapFileLoader(
            self.corr, {"shear": (2, 2, self.corr.n_active)}, sources=range(self.nmaps),
            read_fn=lambda s, out: out["shear"].__setitem__(Ellipsis, self.archive[s]),
            n_slots=4, n_readers=3)
        for k, _dev in loader:
            if k == 3:
                break
        del loader
        time.sleep(0.2)
        self.assertEqual(threading.active_count(), before)

    def test_hash_mismatch_raises(self):
        other = Correlation(NSIDE, np.radians([40.0]), np.radians([90.0]), device="cpu")
        self.assertNotEqual(other.row_pix_hash, self.corr.row_pix_hash)
        with self.assertRaisesRegex(ValueError, "row_pix hash mismatch"):
            MapFileLoader(
                self.corr, {"shear": (2, 2, self.corr.n_active)}, sources=[0],
                read_fn=lambda s, o: None, row_pix_hash=other.row_pix_hash)
        with self.assertRaises(ValueError):
            MapFileLoader(self.corr, {}, sources=[], read_fn=None, n_slots=1)

    def test_deprecated_aliases_still_work(self):
        """The 5.0 names keep working, with a DeprecationWarning."""
        from types import SimpleNamespace

        from CosmoFuse import (
            MapFileLoader as NewFile,
            MapLoader as New,
            PinnedMapPipeline,
            RowSpaceMapLoader,
        )

        self.assertTrue(issubclass(PinnedMapPipeline, New))
        self.assertTrue(issubclass(RowSpaceMapLoader, NewFile))
        corr = SimpleNamespace(
            backend=self.corr.backend, map_dtype=np.dtype(np.float64)
        )
        with self.assertWarnsRegex(DeprecationWarning, "PinnedMapPipeline"):
            PinnedMapPipeline(corr, {"m": (3, 20)})
        with self.assertWarnsRegex(DeprecationWarning, "RowSpaceMapLoader"):
            RowSpaceMapLoader(
                self.corr, {"m": (3, 20)}, sources=[], read_fn=lambda s, o: None
            )

    def test_to_row_space(self):
        full = np.zeros((2, 2, NPIX))
        full[..., self.corr.row_pix] = self.archive[0]
        rows = self.corr.to_row_space(full, dtype=np.float32)
        self.assertEqual(rows.dtype, np.float32)
        self.assertTrue(rows.flags.c_contiguous)
        self.assertTrue(np.array_equal(rows, self.archive[0].astype(np.float32)))


if __name__ == "__main__":
    unittest.main()
