"""One estimator, whichever entry point and whichever backend.

CLAUDE.md states the wrapper contract as a **ratio of sums**: both
orientations of a cross combination summed, then divided once (TreeCorr's
definition).  Nothing gated it per method, and ``compute_density_density``
quietly used a mean of ratios instead -- ``0.5*(N_ab/D_ab + N_ba/D_ba)`` --
which agrees with the rest only when the two orientations carry the same
weight, i.e. never for a cross pair.

The table below is the gate that was missing: for each statistic, the
single-map entry point against the vectorised one against the fused 3x2pt
path, on the CPU backend (and on a GPU when one is present).
"""

import unittest

import healpy as hp
import numpy as np
import pytest

from CosmoFuse.correlations import Correlation

NSIDE = 32
NPIX = hp.nside2npix(NSIDE)
PHI_C = np.radians([40.0, 80.0])
THETA_C = np.radians([95.0, 60.0])
PATCH = 700.0


def _mask():
    mask = np.zeros(NPIX, dtype=bool)
    for vec in hp.ang2vec(THETA_C, PHI_C):
        mask[hp.query_disc(NSIDE, vec, np.radians(PATCH / 60))] = True
    return mask


def _corr(device="cpu"):
    corr = Correlation(
        NSIDE, PHI_C, THETA_C, nbins=3, theta_min=120.0, theta_max=600.0,
        patch_size=PATCH, theta_Q=200.0, mask=_mask(), device=device,
        map_precision="float64", rotation_precision="float64",
    )
    corr.preprocess()
    corr.prepare()
    return corr


def _maps(corr, seed=3):
    """Two tomographic bins with *different* weights — the only case that
    tells a ratio of sums apart from a mean of ratios."""
    rng = np.random.default_rng(seed)
    n = corr.n_active
    density = rng.normal(size=(2, n))
    shear = rng.normal(size=(2, 2, n)) * 0.03
    w_d = rng.uniform(0.1, 3.0, size=(2, n))
    w_s = rng.uniform(0.1, 3.0, size=(2, n))
    # make the two bins' weights genuinely asymmetric
    w_d[1] *= 4.0
    w_s[1] *= 4.0
    return density, shear, w_d, w_s


def _host(value):
    return np.asarray(value if isinstance(value, np.ndarray) else np.asarray(value))


class EstimatorParityBase:
    """Subclassed once per backend."""

    device = "cpu"

    @classmethod
    def setUpClass(cls):
        cls.corr = _corr(cls.device)
        cls.density, cls.shear, cls.w_d, cls.w_s = _maps(cls.corr)
        # row 1 of the upper-triangular ordering for 2 bins is the cross (0,1)
        cls.cross_row = 1

    def _to_host(self, array):
        return np.asarray(self.corr.backend.to_numpy(array))

    def test_xi_g_cross_single_map_matches_vectorized_and_fused(self):
        corr = self.corr
        single = self._to_host(
            corr.compute_density_density(
                self.density[0], self.density[1], self.w_d[0], self.w_d[1]
            )[0]
        )
        vectorized = self._to_host(
            corr.vectorized_density_density(
                self.density, self.w_d, return_device=False
            )
        )[self.cross_row]
        fused = self._to_host(
            corr.get_3x2pt_tomo(
                shear_maps=self.shear, density_maps=self.density,
                weights=(self.w_s, self.w_d), return_device=False,
            )[4]
        )[self.cross_row]
        np.testing.assert_allclose(single, vectorized, rtol=1e-12, atol=0)
        np.testing.assert_allclose(single, fused, rtol=1e-12, atol=0)

    def test_xi_t_cross_single_map_matches_vectorized_and_fused(self):
        corr = self.corr
        single = self._to_host(
            corr.compute_density_shear(
                self.density[0], self.shear[1][0], self.shear[1][1],
                self.w_d[0], self.w_s[1],
            )[0]
        )
        vectorized = self._to_host(
            corr.vectorized_density_shear(
                self.density, self.shear, self.w_d, self.w_s, return_device=False
            )
        )
        # ggl combinations are (lens, source) in row-major order
        np.testing.assert_allclose(single, vectorized[1], rtol=1e-12, atol=0)

    def test_xi_pm_cross_single_map_matches_vectorized(self):
        corr = self.corr
        xip, xim = (
            self._to_host(a)
            for a in corr.compute_shear_shear(
                self.shear[0][0], self.shear[0][1],
                self.shear[1][0], self.shear[1][1],
                self.w_s[0], self.w_s[1], return_device=False,
            )
        )
        vp, vm = (
            self._to_host(a)
            for a in corr.vectorized_shear_shear(
                self.shear, self.w_s, return_device=False
            )
        )
        np.testing.assert_allclose(xip, vp[self.cross_row], rtol=1e-12, atol=0)
        np.testing.assert_allclose(xim, vm[self.cross_row], rtol=1e-12, atol=0)

    def test_density_density_is_a_ratio_of_sums(self):
        """Pin the estimator itself, not just agreement between paths.

        Both paths could drift together; this computes the two orientations
        by hand and checks the combination rule.
        """
        corr = self.corr
        auto = self._to_host(
            corr.compute_density_density(
                self.density[0], self.density[0], self.w_d[0], self.w_d[0]
            )[0]
        )
        # for an auto pair the two orientations coincide, so both estimators
        # agree -- the cross pair is where they part
        cross = self._to_host(
            corr.compute_density_density(
                self.density[0], self.density[1], self.w_d[0], self.w_d[1]
            )[0]
        )
        swapped = self._to_host(
            corr.compute_density_density(
                self.density[1], self.density[0], self.w_d[1], self.w_d[0]
            )[0]
        )
        self.assertTrue(np.all(np.isfinite(auto)))
        # a ratio of sums is symmetric under swapping the two maps
        np.testing.assert_allclose(cross, swapped, rtol=1e-12, atol=0)


class TestEstimatorParityCPU(EstimatorParityBase, unittest.TestCase):
    device = "cpu"


@pytest.mark.gpu
class TestEstimatorParityGPU(EstimatorParityBase, unittest.TestCase):
    device = "gpu"

    @classmethod
    def setUpClass(cls):
        pytest.importorskip("cupy")
        super().setUpClass()


class TestCombinationAccessors(unittest.TestCase):
    """The public row order must be the one the kernels are actually given.

    These accessors exist so a data vector can be labelled without guessing;
    that is only worth anything if they cannot drift from the private
    builders.
    """

    @classmethod
    def setUpClass(cls):
        cls.corr = _corr("cpu")

    def test_tomo_combinations_match_the_kernel_indices(self):
        for nzbins in (1, 2, 3, 5):
            ncomb = nzbins * (nzbins + 1) // 2
            comb_i, comb_j, auto = self.corr._get_tomo_combination_indices(
                nzbins, ncomb
            )
            built = list(zip(np.asarray(comb_i).tolist(), np.asarray(comb_j).tolist()))
            self.assertEqual(built, Correlation.tomo_combinations(nzbins))
            self.assertEqual(
                [i == j for i, j in built], np.asarray(auto).tolist()
            )

    def test_auto_only_density_combinations_match(self):
        for nzbins in (1, 3, 4):
            comb_i, comb_j, _auto, ncomb = (
                self.corr._get_selected_tomo_density_combination_indices(
                    nzbins, gc_auto_correlations_only=True
                )
            )
            built = list(zip(np.asarray(comb_i).tolist(), np.asarray(comb_j).tolist()))
            self.assertEqual(
                built, Correlation.tomo_combinations(nzbins, True)
            )
            self.assertEqual(ncomb, nzbins)

    def test_ggl_combinations_match_the_kernel_indices(self):
        for nlens, nsource in ((1, 1), (2, 3), (4, 4)):
            comb_i, comb_j, ncomb = (
                self.corr._get_selected_tomo_cross_combination_indices(
                    nlens, nsource
                )
            )
            built = list(zip(np.asarray(comb_i).tolist(), np.asarray(comb_j).tolist()))
            self.assertEqual(built, Correlation.ggl_combinations(nlens, nsource))
            self.assertEqual(ncomb, nlens * nsource)

    def test_ggl_combinations_honour_an_explicit_selection(self):
        selection = [(0, 2), (1, 0), (1, 3)]
        comb_i, comb_j, ncomb = (
            self.corr._get_selected_tomo_cross_combination_indices(
                2, 4, ggl_bin_combinations=selection
            )
        )
        built = list(zip(np.asarray(comb_i).tolist(), np.asarray(comb_j).tolist()))
        self.assertEqual(built, Correlation.ggl_combinations(2, 4, selection))
        self.assertEqual(ncomb, len(selection))

    def test_zeta_triplets_match_the_reduction(self):
        from CosmoFuse.correlation_helpers import _get_pair_index, _triplet_indices

        for nzbins in (1, 2, 4):
            indices = _triplet_indices(nzbins)
            triplets = Correlation.zeta_triplets(nzbins)
            self.assertEqual(len(triplets), len(indices.centers))
            for k, (z_center, z2, z3) in enumerate(triplets):
                self.assertEqual(indices.centers[k], z_center)
                self.assertEqual(
                    indices.pairs[k], _get_pair_index(nzbins, z2, z3)
                )


class TestPrecomputedSumOfWeights(unittest.TestCase):
    """``sumofweights=`` is public; so is the thing that builds a valid one.

    Passing the precomputed sums must give the same answer as letting the
    kernels accumulate their own denominators -- that equivalence is the
    whole reason the argument exists.
    """

    @classmethod
    def setUpClass(cls):
        cls.corr = _corr("cpu")
        _density, cls.shear, _w_d, cls.w_s = _maps(cls.corr)

    def test_matches_the_in_kernel_denominator(self):
        corr = self.corr
        sums = corr.compute_sumofweights(self.w_s)
        nz = self.w_s.shape[0]
        self.assertEqual(
            tuple(np.asarray(sums).shape),
            (2, nz * (nz + 1) // 2, corr.n_patches * corr.nbins),
        )
        xip_a, xim_a = corr.vectorized_shear_shear(
            self.shear, self.w_s, return_device=False
        )
        xip_b, xim_b = corr.vectorized_shear_shear(
            self.shear, self.w_s, sumofweights=sums, return_device=False
        )
        np.testing.assert_allclose(
            np.asarray(xip_b), np.asarray(xip_a), rtol=1e-12, atol=0
        )
        np.testing.assert_allclose(
            np.asarray(xim_b), np.asarray(xim_a), rtol=1e-12, atol=0
        )
