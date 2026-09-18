"""The degraded estimator against TreeCorr.

The static treecode measures a *different, windowed* estimator: each
angular bin is a 2PCF of coarse cells, every cell a single point at its
binary-mask centroid carrying the weight sum and the weighted-mean shear
of its members.  That is exactly what the degrade kernel produces, so the
honest independent check is:

    CosmoFuse(resolution_factor=k)  ==  TreeCorr run on those same cells.

Comparing a treecode measurement against TreeCorr at *full* resolution
would be comparing two different estimators (T10 measured 20-40 %
suppression at k = 1.5), so it would prove nothing.

This gate therefore validates the whole chain at once: the cell weights
and weighted means the kernel writes into the virtual rows, the centroid
positions, the pair search over those cells, and the xi+- normalisation.
"""

import unittest
from unittest.mock import patch

import healpy as hp
import numpy as np
import pytest
import treecorr

from CosmoFuse import Correlation
from CosmoFuse.utils import pixel2RaDec

from .test_gpu_kernel_emulation import emulated_gpu

# nside 128 with k = 2 over 100'-400' assigns the three bins to nside
# 128 / 64 / 32: a base-resolution bin plus the full two-level chain.
NSIDE = 128
NPIX = hp.nside2npix(NSIDE)
THETA_MIN, THETA_MAX, NBINS = 100.0, 400.0, 3
PATCH_SIZE = 400.0
RESOLUTION_FACTOR = 2.0


def build():
    theta, phi = hp.pix2ang(NSIDE, np.arange(NPIX))
    mask = (np.degrees(phi) < 60.0) & (np.abs(90 - np.degrees(theta)) < 25)
    rng = np.random.default_rng(4)
    mask[rng.choice(NPIX, NPIX // 40, replace=False)] = False
    phi_c = np.array([0.30, 0.52])
    theta_c = np.array([1.48, 1.62])
    corr = Correlation(
        NSIDE, phi_c, theta_c,
        nbins=NBINS, theta_min=THETA_MIN, theta_max=THETA_MAX,
        patch_size=PATCH_SIZE, theta_Q=PATCH_SIZE / 2,
        mask=mask.astype(float), device="cpu",
        map_precision="float64", rotation_precision="float64",
        pair_search_precision="float64",
        resolution_factor=RESOLUTION_FACTOR,
    )
    corr.preprocess()
    return corr


@pytest.mark.slow
class TestDegradedEstimatorAgainstTreeCorr(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corr = build()
        from CosmoFuse.treecode import level_groups

        cls.groups = level_groups(cls.corr.level_nside)
        coarse = [g for g in cls.groups if g[0] != NSIDE]
        if len(coarse) < 2:
            raise unittest.SkipTest(
                "expected a multi-level chain, got nside per bin "
                f"{np.asarray(cls.corr.level_nside).tolist()}"
            )
        rng = np.random.default_rng(9)
        n = cls.corr.n_active
        cls.shear = rng.normal(size=(1, 2, n)) * 0.3
        cls.w = rng.uniform(0.2, 2.0, size=(1, n))

    def _cells(self, fused):
        """Cell positions, weights and weighted-mean shear, as the degrade
        writes them into the virtual rows."""
        corr = self.corr
        call = lambda c: c._expand_rows(
            (self.shear[:, 0], self.shear[:, 1]), self.w, "pairs")
        if fused:
            with emulated_gpu(corr):
                with patch.object(Correlation, "_use_fused_degrade",
                                  lambda self, w: True):
                    (g1, g2), w_rows = call(corr)
            corr.compute_context.Q_inds_dev = None
            corr.compute_context.degrade_csr = None
        else:
            corr.compute_context.degrade_ops = None
            (g1, g2), w_rows = call(corr)
        return np.asarray(g1), np.asarray(g2), np.asarray(w_rows)

    def _treecorr_xipm(self, g1, g2, w_rows):
        """TreeCorr on whatever CosmoFuse pairs for each bin: the pixel rows
        for a base-resolution group, that level's cells for a coarse one."""
        corr = self.corr
        tree = corr._treecode
        starts = tree.level_starts(first=corr.n_aperture_cells)
        xip = np.full((corr.n_patches, NBINS), np.nan)
        xim = np.full((corr.n_patches, NBINS), np.nan)

        for nside_b, b0, b1 in self.groups:
            gg = treecorr.GGCorrelation(
                nbins=NBINS, min_sep=THETA_MIN, max_sep=THETA_MAX,
                sep_units="arcmin", brute=True, metric="Arc",
                bin_slop=0.0, angle_slop=0.0,
            )
            for p in range(corr.n_patches):
                if nside_b == NSIDE:
                    vec = hp.ang2vec(corr.theta_center[p], corr.phi_center[p])
                    disc = hp.query_disc(
                        NSIDE, vec=vec, radius=np.radians(PATCH_SIZE / 60))
                    pix = np.intersect1d(disc, corr.map_inds)
                    rows = np.searchsorted(corr.map_inds, pix)
                    ra, dec = pixel2RaDec(pix, NSIDE)
                else:
                    level = tree.level_index(nside_b)
                    lo = int(tree.cell_offsets[level, p])
                    hi = int(tree.cell_offsets[level, p + 1])
                    self.assertGreater(
                        hi - lo, 15, f"too few cells at nside {nside_b}")
                    rows = corr.n_active + int(starts[level]) + np.arange(lo, hi)
                    ra = np.asarray(tree.cell_ra[level][lo:hi])
                    dec = np.asarray(tree.cell_dec[level][lo:hi])
                keep = w_rows[0, rows] > 0
                cat = treecorr.Catalog(
                    ra=ra[keep], dec=dec[keep],
                    g1=g1[0, rows][keep], g2=g2[0, rows][keep],
                    w=w_rows[0, rows][keep],
                    ra_units="rad", dec_units="rad", flip_g1=True,
                )
                gg.process(cat)
                # only the bins this level is responsible for
                xip[p, b0:b1] = gg.xip[b0:b1]
                xim[p, b0:b1] = gg.xim[b0:b1]
        self.assertFalse(np.any(np.isnan(xip)))
        return xip, xim

    def test_fused_kernel_matches_treecorr_on_the_cells(self):
        g1, g2, w_rows = self._cells(fused=True)
        xip_tc, xim_tc = self._treecorr_xipm(g1, g2, w_rows)

        with emulated_gpu(self.corr):
            self.corr.compute_context.Q_inds_dev = None
            with patch.object(Correlation, "_use_fused_degrade",
                              lambda self, w: True):
                xip, xim = self.corr.vectorized_shear_shear(
                    self.shear, self.w, flip_g1=True, return_device=False)
        self.corr.compute_context.Q_inds_dev = None
        self.corr.compute_context.degrade_csr = None

        xip, xim = np.asarray(xip[0]), np.asarray(xim[0])
        self.assertGreater(np.max(np.abs(xip_tc)), 0)
        # Same tolerance the full-resolution TreeCorr gate uses
        # (tests/test_end_to_end.py): the two codes evaluate the spherical
        # separations and the pair rotations independently, which shows up
        # at the 1e-7 level.  The kernel itself is pinned far tighter
        # against the sparse chain in tests/test_degrade_kernel.py.
        for got, want, name in ((xip, xip_tc, "xi+"), (xim, xim_tc, "xi-")):
            self.assertLess(np.abs(1 - got / want).max(), 1e-6, name)
            self.assertLess(np.abs(got - want).max(), 1e-10, name)

    def test_fused_and_sparse_cells_agree(self):
        """Both degrade implementations must hand TreeCorr the same cells."""
        a = self._cells(fused=True)
        b = self._cells(fused=False)
        for x, y, name in zip(a, b, ("g1", "g2", "w")):
            scale = np.max(np.abs(y))
            np.testing.assert_allclose(
                x, y, rtol=0, atol=1e-13 * scale, err_msg=name)

if __name__ == "__main__":
    unittest.main()
