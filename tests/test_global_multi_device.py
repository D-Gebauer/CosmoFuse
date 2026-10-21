"""Multi-device gates of the footprint-wide estimators.

M1  Global2PCF on 2 and 3 "devices" == one device, bitwise per tile, for
    gg/kk/kg/three_by_two_pt, at full resolution, with the static treecode and
    with packed pairs; the tile ranges cover the tiles in order.
M2  Global3PCF on 2 and 3 "devices" == one device to summation order
    (<= 1e-13), every component, numerators and normalisations.
M3  pair files: written by a group, read by a group or one device, bitwise.
M4  the split itself: balanced, contiguous, non-empty; errors.
M5  the GPU orchestration of every engine via the numpy twins (Global2PCF).

Exercised with CPU "devices" so it runs in CI; on a multi-GPU box the only
difference is which backend each part holds (TestRealGPU).
"""

import contextlib
import os
import tempfile
import unittest
import warnings

import numpy as np
import pytest

from CosmoFuse import Global2PCF, Global3PCF
from CosmoFuse.multi_device import balanced_ranges, require_host_arrays

from . import test_global_2pcf as g2
from . import test_global_3pcf as g3
from .test_gpu_kernel_emulation import emulated_gpu


def _global2(mask, devices, **extra):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = Global2PCF(g2.NSIDE, mask=mask, **g2._kw(device=devices, **extra))
    g.preprocess()
    return g


def _global3(mask, devices, **kw):
    kw.setdefault("tile_nside", 2)
    kw.setdefault("n_max", g3.N_MAX)
    est = Global3PCF(
        g3.NSIDE,
        mask=mask,
        nbins=3,
        theta_min=240,
        theta_max=900,
        device=devices,
        **kw,
    )
    est.preprocess()
    return est


def _all_2pcf(g, shear, w, dens):
    out = dict(zip(("gg", "kk", "kg"), g2._measure_all(g, shear, w, dens)))
    for key, res in g.three_by_two_pt(
        shear, dens, w, w, flip_g1=True, return_device=False
    ).items():
        out["3x2pt_" + key] = res
    return out


class TestGlobal2PCF(unittest.TestCase):
    """M1"""

    @classmethod
    def setUpClass(cls):
        cls.mask, cls.shear, cls.w, cls.dens = g2._setup()

    def assertSameResult(self, a, b):
        self.assertEqual(a.statistic, b.statistic)
        self.assertEqual(a.combinations, b.combinations)
        np.testing.assert_array_equal(a.binedges, b.binedges)
        self.assertEqual(a.names, b.names)
        for name in a.names:
            np.testing.assert_array_equal(a.tile_num[name], b.tile_num[name])
        np.testing.assert_array_equal(a.tile_weight, b.tile_weight)

    def _check(self, **extra):
        single = _global2(self.mask, "cpu", tile_nside=8, **extra)
        ref = _all_2pcf(single, self.shear, self.w, self.dens)
        for devices in (["cpu", "cpu"], ["cpu", "cpu", "cpu"]):
            with self.subTest(n_devices=len(devices), **extra):
                multi = _global2(self.mask, devices, tile_nside=8, **extra)
                self.assertEqual(len(multi._engines), len(devices))
                self.assertEqual(multi.ntotpairs, single.ntotpairs)
                ranges = multi.tile_ranges
                self.assertEqual(ranges[0][0], 0)
                self.assertEqual(ranges[-1][1], multi.n_tiles)
                for (_, b), (a, _) in zip(ranges[:-1], ranges[1:]):
                    self.assertEqual(a, b)
                got = _all_2pcf(multi, self.shear, self.w, self.dens)
                for key in ref:
                    self.assertIsInstance(got[key].tile_weight, np.ndarray)
                    self.assertSameResult(got[key], ref[key])

    def test_full_resolution(self):
        self._check()

    def test_static_treecode(self):
        self._check(resolution_factor=2.0)

    def test_packed_pairs(self):
        self._check(pack_pairs=True, resolution_factor=2.0)

    def test_one_element_list_is_one_device(self):
        g = _global2(self.mask, ["cpu"], tile_nside=8)
        self.assertEqual(len(g._engines), 1)
        self.assertEqual(g.tile_ranges, [(0, g.n_tiles)])


class TestGlobal3PCF(unittest.TestCase):
    """M2"""

    @classmethod
    def setUpClass(cls):
        cls.mask, cls.g, cls.w, cls.d = g3._setup()
        cls.single = _global3(cls.mask, "cpu", centre_chunk=17)

    def _measure(self, est):
        return {
            "ggg": est.ggg(self.g, self.w, flip_g1=True),
            "kkk": est.kkk(self.d, self.w),
            "kgg": est.kgg(self.d, self.g, self.w, self.w),
            "kkg": est.kkg(self.d, self.g, self.w, self.w),
        }

    def test_matches_one_device(self):
        ref = self._measure(self.single)
        for devices in (["cpu", "cpu"], ["cpu", "cpu", "cpu"]):
            with self.subTest(n_devices=len(devices)):
                multi = _global3(self.mask, devices, centre_chunk=17)
                self.assertEqual(multi.ntotpairs, self.single.ntotpairs)
                self.assertEqual(
                    sum(p.ntotpairs for p in multi._parts), self.single.ntotpairs
                )
                ranges = multi.centre_ranges
                self.assertEqual(ranges[0][0], 0)
                self.assertEqual(ranges[-1][1], self.single.centre_rows.size)
                got = self._measure(multi)
                for key, res in ref.items():
                    for label in res.num:
                        self.assertLess(
                            g3._rel(got[key].num[label], res.num[label]), 1e-13
                        )
                        self.assertLess(
                            g3._rel(got[key].norm[label], res.norm[label]), 1e-13
                        )

    def test_row_space_input(self):
        multi = _global3(self.mask, ["cpu", "cpu"])
        full = multi.ggg(self.g, self.w)
        rows = multi.ggg(multi.to_row_space(self.g), multi.to_row_space(self.w))
        for label in full.num:
            np.testing.assert_array_equal(full.num[label], rows.num[label])


class TestPairFiles(unittest.TestCase):
    """M3"""

    def test_global2pcf(self):
        mask, shear, w, dens = g2._setup()
        multi = _global2(mask, ["cpu", "cpu"], tile_nside=8, resolution_factor=2.0)
        ref = multi.gg(shear, w, return_device=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pairs.h5")
            multi.save_pairs(path)
            for devices in ("cpu", ["cpu", "cpu", "cpu"]):
                with self.subTest(devices=devices):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        other = Global2PCF(
                            g2.NSIDE,
                            mask=mask,
                            **g2._kw(
                                device=devices, tile_nside=8, resolution_factor=2.0
                            ),
                        )
                    other.load_pairs(path)
                    got = other.gg(shear, w, return_device=False)
                    for name in ref.names:
                        np.testing.assert_array_equal(
                            got.tile_num[name], ref.tile_num[name]
                        )

    def test_global3pcf(self):
        mask, g, w, _d = g3._setup()
        single = _global3(mask, "cpu")
        ref = single.ggg(g, w)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pairs.h5")
            _global3(mask, ["cpu", "cpu"]).save_pairs(path)
            multi = Global3PCF(
                g3.NSIDE,
                mask=mask,
                nbins=3,
                theta_min=240,
                theta_max=900,
                n_max=g3.N_MAX,
                tile_nside=2,
                device=["cpu", "cpu"],
            )
            multi.load_pairs(path)
            got = multi.ggg(g, w)
            for label in ref.num:
                self.assertLess(g3._rel(got.num[label], ref.num[label]), 1e-13)


class TestSplit(unittest.TestCase):
    """M4"""

    def test_balanced_ranges(self):
        self.assertEqual(balanced_ranges([1, 1, 1, 1], 2), [(0, 2), (2, 4)])
        self.assertEqual(balanced_ranges([10, 1, 1, 1], 2), [(0, 1), (1, 4)])
        self.assertEqual(balanced_ranges([1, 1, 1, 10], 2), [(0, 3), (3, 4)])
        # never empty, even when the work is concentrated
        self.assertEqual(balanced_ranges([0, 0, 100], 3), [(0, 1), (1, 2), (2, 3)])
        self.assertEqual(balanced_ranges([0, 0, 0, 0], 2), [(0, 2), (2, 4)])
        rng = np.random.default_rng(0)
        work = rng.uniform(0, 1, 1000)
        ranges = balanced_ranges(work, 4)
        shares = [work[a:b].sum() / work.sum() for a, b in ranges]
        self.assertLess(max(abs(s - 0.25) for s in shares), 0.01)
        with self.assertRaisesRegex(ValueError, "cannot be split"):
            balanced_ranges([1.0], 2)

    def test_errors(self):
        mask = g2._setup()[0]
        with self.assertRaisesRegex(ValueError, "at least one device"):
            Global2PCF(g2.NSIDE, mask=mask, **g2._kw(device=[]))
        with self.assertRaisesRegex(ValueError, "duplicate device ids"):
            Global3PCF(g3.NSIDE, device=[0, 0])
        with self.assertRaisesRegex(ValueError, "cannot be split"):
            Global2PCF(g2.NSIDE, mask=mask, **g2._kw(device=["cpu"] * 4, tile_nside=1))

    def test_device_arrays_are_refused(self):
        class FakeDeviceArray:
            __cuda_array_interface__ = {}

        with self.assertRaisesRegex(TypeError, "host arrays"):
            require_host_arrays("gg", [np.zeros(3), FakeDeviceArray()])
        mask, shear, w, _dens = g2._setup()
        multi = _global2(mask, ["cpu", "cpu"], tile_nside=8)
        with self.assertRaisesRegex(TypeError, "host arrays"):
            multi.gg(shear, FakeDeviceArray())


class TestEmulatedGPU(unittest.TestCase):
    """M5"""

    def test_every_engine_runs_the_gpu_path(self):
        mask, shear, w, dens = g2._setup()
        kw = dict(
            tile_nside=8,
            map_precision="float32",
            rotation_precision="float32",
            accumulation_precision="float64",
            pack_pairs=True,
            resolution_factor=2.0,
        )
        multi = _global2(mask, ["cpu", "cpu"], **kw)
        cpu = g2._measure_all(multi, shear, w, dens)
        with contextlib.ExitStack() as stack:
            counters = [stack.enter_context(emulated_gpu(e)) for e in multi._engines]
            gpu = g2._measure_all(multi, shear, w, dens)
        for calls in counters:
            self.assertTrue(any(v for k, v in calls.items() if k != "_built"))
        for a, b in zip(g2._estimates(gpu), g2._estimates(cpu)):
            self.assertLess(g2._rel(a, b), 1e-5)


def _n_gpus():
    try:
        import cupy

        return int(cupy.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


@pytest.mark.gpu
class TestRealGPU(unittest.TestCase):
    def setUp(self):
        if _n_gpus() < 2:
            self.skipTest("needs two GPUs")

    def test_global2pcf(self):
        mask, shear, w, dens = g2._setup()
        for extra in ({}, {"pack_pairs": True, "resolution_factor": 2.0}):
            with self.subTest(**extra):
                one = _global2(mask, 0, **extra)
                two = _global2(mask, [0, 1], **extra)
                for a, b in zip(
                    g2._estimates(g2._measure_all(two, shear, w, dens)),
                    g2._estimates(g2._measure_all(one, shear, w, dens)),
                ):
                    np.testing.assert_array_equal(a, b)

    def test_global3pcf(self):
        mask, g, w, _d = g3._setup()
        one = _global3(mask, 0)
        two = _global3(mask, [0, 1])
        a, b = two.ggg(g, w), one.ggg(g, w)
        for label in a.num:
            self.assertLess(g3._rel(a.num[label], b.num[label]), 1e-13)
            self.assertLess(g3._rel(a.norm[label], b.norm[label]), 1e-13)
