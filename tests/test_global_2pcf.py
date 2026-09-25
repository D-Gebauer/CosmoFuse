"""Gates of the footprint-wide two-point estimator (Global2PCF).

The reference is the patch estimator with ONE patch that contains the whole
footprint: it sees every footprint pair exactly once, so it measures the
global 2PCF by construction -- through an independent pair search (the patch
finder) and one group instead of many tiles.

G1  == one-patch Integrated3PCF for xi+- (gg), w(theta) (kk), gamma_t (kg),
    for several tilings (tile invariance), at full resolution, with the static
    treecode (global cells == the one patch's cells) and with packed pairs.
G2  the result's per-tile sums, jackknife, row labels and ``+``;
    three_by_two_pt == its parts.
G3  row-space input == full-sky input (bitwise).
G4  pair files round-trip bitwise and refuse a different estimator.
G5  GPU orchestration (tiled and packed kernels) via the numpy twins == CPU.
"""

import os
import tempfile
import unittest
import warnings

import healpy as hp
import numpy as np
import pytest

from CosmoFuse import Global2PCF, Integrated3PCF, TwoPointResult

from .test_gpu_kernel_emulation import emulated_gpu

NSIDE = 64
CENTRE = (np.radians(60.0), np.radians(30.0))  # (theta, phi)
RADIUS = 8.0  # degrees
BINS = dict(nbins=5, theta_min=60, theta_max=480)


def _setup():
    npix = hp.nside2npix(NSIDE)
    disc = hp.query_disc(NSIDE, hp.ang2vec(*CENTRE), np.radians(RADIUS))
    mask = np.zeros(npix, bool)
    mask[disc] = True
    rng = np.random.default_rng(1)
    shear = rng.normal(size=(2, 2, npix)) * 0.3
    w = rng.uniform(0.5, 1.5, size=(2, npix))
    dens = rng.normal(size=(2, npix))
    return mask, shear, w, dens


def _kw(**extra):
    kw = dict(device="cpu", map_precision="float64", rotation_precision="float64", **BINS)
    kw.update(extra)
    return kw


def _global(mask, **extra):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = Global2PCF(NSIDE, mask=mask, **_kw(**extra))
    g.preprocess()
    return g


def _one_patch(mask, **extra):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = Integrated3PCF(
            NSIDE, np.array([CENTRE[1]]), np.array([CENTRE[0]]),
            patch_size=(RADIUS + 0.5) * 60, theta_Q=10, mask=mask, **_kw(**extra),
        )
        p.find_pairs()
        p.prepare()
    return p


def _rel(a, b):
    b = np.asarray(b)
    return float(np.abs(np.asarray(a) - b).max() / np.abs(b).max())


def _measure_all(g, shear, w, dens):
    return (
        g.gg(shear, w, flip_g1=True, return_device=False),
        g.kk(dens, w, return_device=False),
        g.kg(dens, shear, w, w, flip_g1=True, return_device=False),
    )


class TestOnePatchReference(unittest.TestCase):
    """G1"""

    @classmethod
    def setUpClass(cls):
        cls.mask, cls.shear, cls.w, cls.dens = _setup()

    def _compare(self, g, p, tol=1e-12):
        self.assertEqual(g.ntotpairs, p.ntotpairs)
        gg, kk, kg = _measure_all(g, self.shear, self.w, self.dens)
        ref_gg = p.gg(self.shear, self.w, apertures=False, flip_g1=True, return_device=False)
        ref_kk = p.kk(self.dens, self.w, apertures=False, return_device=False)
        ref_kg = p.kg(self.dens, self.shear, self.w, self.w, flip_g1=True, return_device=False)
        self.assertEqual(gg.xi_p.shape, (3, BINS["nbins"]))
        self.assertEqual(gg.combinations, ref_gg.gg_combinations)
        self.assertEqual(kk.combinations, ref_kk.kk_combinations)
        self.assertEqual(kg.combinations, ref_kg.kg_combinations)
        self.assertLess(_rel(gg.xi_p, ref_gg.xi_p[:, 0]), tol)
        self.assertLess(_rel(gg.xi_m, ref_gg.xi_m[:, 0]), tol)
        self.assertLess(_rel(kk.xi, ref_kk.xi_g[:, 0]), tol)
        self.assertLess(_rel(kg.xi, ref_kg.xi_t[:, 0]), tol)

    def test_full_resolution_any_tiling(self):
        p = _one_patch(self.mask)
        for tile_nside in (2, 4, 8):
            with self.subTest(tile_nside=tile_nside):
                self._compare(_global(self.mask, tile_nside=tile_nside), p)

    def test_static_treecode(self):
        g = _global(self.mask, resolution_factor=2.0)
        p = _one_patch(self.mask, resolution_factor=2.0)
        self.assertGreater(len(set(g.level_table["nside"].tolist())), 1)
        np.testing.assert_array_equal(g.level_table["nside"], p.level_table["nside"])
        self._compare(g, p)

    def test_packed_pairs(self):
        self._compare(_global(self.mask, pack_pairs=True), _one_patch(self.mask, pack_pairs=True))

    def test_tile_nside_finer_than_coarsest_level_raises(self):
        with self.assertRaises(ValueError), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Global2PCF(NSIDE, mask=self.mask, tile_nside=64, **_kw(resolution_factor=2.0))


class TestResultsAndInputs(unittest.TestCase):
    """G2 + G3"""

    @classmethod
    def setUpClass(cls):
        cls.mask, cls.shear, cls.w, cls.dens = _setup()
        cls.g = _global(cls.mask, tile_nside=8)

    def test_result_sums(self):
        res = self.g.gg(self.shear, self.w, return_device=False)
        self.assertIsInstance(res, TwoPointResult)
        self.assertEqual(res.statistic, "gg")
        self.assertEqual(res.names, ("xi_p", "xi_m"))
        self.assertEqual(res.combinations, ((0, 0), (0, 1), (1, 1)))
        self.assertEqual(res.tile_num["xi_p"].shape, (3, self.g.n_tiles, BINS["nbins"]))
        np.testing.assert_allclose(res.theta, self.g.bincenters, rtol=1e-12)
        ok = res.weight > 0
        np.testing.assert_allclose(
            (res.tile_num["xi_p"].sum(1) / np.where(ok, res.weight, 1))[ok], res.xi_p[ok], rtol=1e-12
        )
        with self.assertRaises(AttributeError):
            res.xi

    def test_jackknife(self):
        res = self.g.kk(self.dens, self.w, return_device=False)
        jk = res.jackknife()["xi"]
        self.assertEqual(jk.shape, (self.g.n_tiles, 3, BINS["nbins"]))
        t = 2
        num = res.tile_num["xi"].sum(1) - res.tile_num["xi"][:, t]
        den = res.tile_weight.sum(1) - res.tile_weight[:, t]
        ok = den > 0
        np.testing.assert_allclose(jk[t][ok], (num / np.where(ok, den, 1))[ok], rtol=1e-12)

    def test_add_joins_tiles(self):
        res = self.g.gg(self.shear, self.w, return_device=False)
        both = res + res
        self.assertEqual(both.n_tiles, 2 * res.n_tiles)
        np.testing.assert_allclose(both.xi_p, res.xi_p, rtol=1e-13)

    def test_three_by_two_pt(self):
        out = self.g.three_by_two_pt(self.shear, self.dens, self.w, self.w, flip_g1=True,
                                     return_device=False)
        ref = dict(zip(("gg", "kk", "kg"), _measure_all(self.g, self.shear, self.w, self.dens)))
        self.assertEqual(set(out), {"gg", "kk", "kg"})
        for key, res in out.items():
            for name in res.names:
                np.testing.assert_array_equal(res.tile_num[name], ref[key].tile_num[name])

    def test_row_space_input(self):
        g = self.g
        full = g.gg(self.shear, self.w, return_device=False)
        rows = g.gg(g.to_row_space(self.shear), g.to_row_space(self.w), return_device=False)
        for name in full.names:
            np.testing.assert_array_equal(full.tile_num[name], rows.tile_num[name])
        np.testing.assert_array_equal(full.tile_weight, rows.tile_weight)


class TestPairFiles(unittest.TestCase):
    """G4"""

    def test_round_trip(self):
        mask, shear, w, _ = _setup()
        for k in (None, 2.0):
            with self.subTest(resolution_factor=k):
                g = _global(mask, resolution_factor=k)
                ref = g.gg(shear, w, return_device=False)
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "g2.h5")
                    g.save_pairs(path)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        other = Global2PCF(NSIDE, mask=mask, **_kw(resolution_factor=k))
                    other.load_pairs(path)
                    res = other.gg(shear, w, return_device=False)
                    np.testing.assert_array_equal(res.xi_p, ref.xi_p)
                    np.testing.assert_array_equal(res.xi_m, ref.xi_m)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        wrong = Global2PCF(NSIDE, mask=mask, **_kw(resolution_factor=3.0 if k else 2.0))
                    with self.assertRaises(ValueError):
                        wrong.load_pairs(path)


def _estimates(results):
    out = []
    for res in results:
        out.extend(np.asarray(res.estimate(n)) for n in res.names)
    return out


class TestEmulatedGPU(unittest.TestCase):
    """G5"""

    def test_tiled_and_packed(self):
        mask, shear, w, dens = _setup()
        for extra in ({}, {"pack_pairs": True}, {"resolution_factor": 2.0}):
            with self.subTest(**extra):
                g = _global(mask, map_precision="float32", rotation_precision="float32",
                            accumulation_precision="float64", **extra)
                cpu = _estimates(_measure_all(g, shear, w, dens))
                with emulated_gpu(g._engine) as calls:
                    gpu = _estimates(_measure_all(g, shear, w, dens))
                self.assertTrue(any(v for k, v in calls.items() if k != "_built"))
                for a, b in zip(gpu, cpu):
                    self.assertLess(_rel(a, b), 1e-5)


@pytest.mark.gpu
class TestRealGPU(unittest.TestCase):
    def test_gpu_matches_cpu(self):
        mask, shear, w, dens = _setup()
        for extra in ({}, {"pack_pairs": True}, {"resolution_factor": 2.0}):
            with self.subTest(**extra):
                cpu = _global(mask, **extra)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gpu = Global2PCF(NSIDE, mask=mask, **_kw(device="gpu", **extra))
                gpu.preprocess()
                for a, b in zip(_estimates(_measure_all(gpu, shear, w, dens)),
                                _estimates(_measure_all(cpu, shear, w, dens))):
                    self.assertLess(_rel(a, b), 1e-12)
