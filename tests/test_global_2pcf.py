"""Gates of the footprint-wide two-point estimator (Global2PCF).

The reference is the patch estimator with ONE patch that contains the whole
footprint: it sees every footprint pair exactly once, so it measures the
global 2PCF by construction -- through an independent pair search (the patch
finder) and one group instead of many tiles.

G1  == one-patch PatchCorrelation for xi+-, w(theta), gamma_t, for several
    tilings (tile invariance), at full resolution, with the static treecode
    (global cells == the one patch's cells) and with packed pairs.
G2  per-tile sums add up to the estimate; three_by_two_pt == its parts.
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

from CosmoFuse import Global2PCF, PatchCorrelation

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
        p = PatchCorrelation(
            NSIDE, np.array([CENTRE[1]]), np.array([CENTRE[0]]),
            patch_size=(RADIUS + 0.5) * 60, theta_Q=10, mask=mask, **_kw(**extra),
        )
        p.find_pairs()
        p.prepare()
    return p


def _rel(a, b):
    b = np.asarray(b)
    return float(np.abs(np.asarray(a) - b).max() / np.abs(b).max())


class TestOnePatchReference(unittest.TestCase):
    """G1"""

    @classmethod
    def setUpClass(cls):
        cls.mask, cls.shear, cls.w, cls.dens = _setup()

    def _compare(self, g, p, tol=1e-12):
        self.assertEqual(g.ntotpairs, p.ntotpairs)
        xp, xm = g.shear(self.shear, self.w, flip_g1=True, return_device=False)
        rp, rm = p.shear_shear(self.shear, self.w, flip_g1=True, return_device=False)
        self.assertEqual(xp.shape, (3, BINS["nbins"]))
        self.assertLess(_rel(xp, rp[:, 0]), tol)
        self.assertLess(_rel(xm, rm[:, 0]), tol)
        xg = g.density(self.dens, self.w, return_device=False)
        self.assertLess(_rel(xg, p.density_density(self.dens, self.w, return_device=False)[:, 0]), tol)
        xt = g.ggl(self.dens, self.shear, self.w, self.w, flip_g1=True, return_device=False)
        rt = p.density_shear(self.dens, self.shear, self.w, self.w, flip_g1=True, return_device=False)
        self.assertLess(_rel(xt, rt[:, 0]), tol)

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


class TestSumsAndInputs(unittest.TestCase):
    """G2 + G3"""

    @classmethod
    def setUpClass(cls):
        cls.mask, cls.shear, cls.w, cls.dens = _setup()
        cls.g = _global(cls.mask, tile_nside=8)

    def test_per_tile_sums(self):
        g = self.g
        xp, xm = g.shear(self.shear, self.w, return_device=False)
        num_p, num_m, den = g.shear(self.shear, self.w, per_tile=True, return_device=False)
        self.assertEqual(num_p.shape, (3, g.n_tiles, BINS["nbins"]))
        tot = den.sum(1)
        ok = tot > 0
        np.testing.assert_allclose(num_p.sum(1)[ok] / tot[ok], xp[ok], rtol=1e-12)
        np.testing.assert_allclose(num_m.sum(1)[ok] / tot[ok], xm[ok], rtol=1e-12)
        num, den = g.density(self.dens, self.w, per_tile=True, return_device=False)
        xg = g.density(self.dens, self.w, return_device=False)
        np.testing.assert_allclose(num.sum(1)[den.sum(1) > 0] / den.sum(1)[den.sum(1) > 0],
                                   xg[den.sum(1) > 0], rtol=1e-12)

    def test_three_by_two_pt(self):
        g = self.g
        out = g.three_by_two_pt(self.shear, self.dens, self.w, self.w, flip_g1=True,
                                return_device=False)
        ref = (
            *g.shear(self.shear, self.w, flip_g1=True, return_device=False),
            g.density(self.dens, self.w, return_device=False),
            g.ggl(self.dens, self.shear, self.w, self.w, flip_g1=True, return_device=False),
        )
        for a, b in zip(out, ref):
            np.testing.assert_array_equal(a, b)

    def test_row_space_input(self):
        g = self.g
        full = g.shear(self.shear, self.w, return_device=False)
        rows = g.shear(g.to_row_space(self.shear), g.to_row_space(self.w), return_device=False)
        for a, b in zip(full, rows):
            np.testing.assert_array_equal(a, b)


class TestPairFiles(unittest.TestCase):
    """G4"""

    def test_round_trip(self):
        mask, shear, w, _ = _setup()
        for k in (None, 2.0):
            with self.subTest(resolution_factor=k):
                g = _global(mask, resolution_factor=k)
                ref = g.shear(shear, w, return_device=False)
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "g2.h5")
                    g.save_pairs(path)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        other = Global2PCF(NSIDE, mask=mask, **_kw(resolution_factor=k))
                    other.load_pairs(path)
                    for a, b in zip(ref, other.shear(shear, w, return_device=False)):
                        np.testing.assert_array_equal(a, b)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        wrong = Global2PCF(NSIDE, mask=mask, **_kw(resolution_factor=3.0 if k else 2.0))
                    with self.assertRaises(ValueError):
                        wrong.load_pairs(path)


class TestEmulatedGPU(unittest.TestCase):
    """G5"""

    def test_tiled_and_packed(self):
        mask, shear, w, dens = _setup()
        for extra in ({}, {"pack_pairs": True}, {"resolution_factor": 2.0}):
            with self.subTest(**extra):
                g = _global(mask, map_precision="float32", rotation_precision="float32",
                            accumulation_precision="float64", **extra)
                cpu = (*g.shear(shear, w, return_device=False),
                       g.density(dens, w, return_device=False),
                       g.ggl(dens, shear, w, w, return_device=False))
                with emulated_gpu(g._engine) as calls:
                    gpu = (*g.shear(shear, w, return_device=False),
                           g.density(dens, w, return_device=False),
                           g.ggl(dens, shear, w, w, return_device=False))
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
                for fn, args in (("shear", (shear, w)), ("density", (dens, w)),
                                 ("ggl", (dens, shear, w, w))):
                    a = getattr(cpu, fn)(*args, return_device=False)
                    b = getattr(gpu, fn)(*args, return_device=False)
                    a = a if isinstance(a, tuple) else (a,)
                    b = b if isinstance(b, tuple) else (b,)
                    for x, y in zip(b, a):
                        self.assertLess(_rel(x, y), 1e-12)
