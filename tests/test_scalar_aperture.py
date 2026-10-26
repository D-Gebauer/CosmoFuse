"""The scalar aperture ``M_g`` has its own filter.

``M_a`` (shear) keeps ``Q_crittenden``; ``M_g`` (any spin-0 field) defaults
to a top hat of radius theta_Q (``U_tophat``) and takes ``U_crittenden`` /
``U_schneider`` for the compensated aperture mass of a convergence map.  The
two filters are independent everywhere both apertures are measured
(``kg``, ``three_by_two_pt``), and the scalar values are evaluated from the
stored aperture geometry, so changing them needs no new disc search and
works on loaded pair files.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import healpy as hp
import numpy as np

from CosmoFuse import (
    Q_crittenden,
    Q_schneider,
    U_crittenden,
    U_schneider,
    U_tophat,
)
from CosmoFuse.correlations import Integrated3PCF

NSIDE = 64
THETA_Q = 120.0


def make_corr(**kwargs):
    rng = np.random.default_rng(5)
    ncent = 3
    theta = np.arccos(rng.uniform(-0.6, 0.6, ncent))
    phi = rng.uniform(0, 2 * np.pi, ncent)
    corr = Integrated3PCF(
        NSIDE,
        phi,
        theta,
        nbins=2,
        theta_min=60,
        theta_max=120,
        patch_size=120,
        theta_Q=THETA_Q,
        device="cpu",
        rotation_precision="float64",
        **kwargs,
    )
    return corr


def random_maps(seed=11, nz=2):
    rng = np.random.default_rng(seed)
    npix = hp.nside2npix(NSIDE)
    return (
        rng.normal(size=(nz, 2, npix)) * 0.1,
        rng.normal(size=(nz, npix)),
        0.5 + rng.random((nz, npix)),
        0.5 + rng.random((nz, npix)),
    )


class TestFilters(unittest.TestCase):
    def test_u_tophat_is_normalised_over_the_cap(self):
        theta_q = np.radians(THETA_Q / 60)
        theta = np.linspace(0, 2 * theta_q, 200001)
        integral = (
            2 * np.pi * np.trapezoid(U_tophat(theta, THETA_Q) * np.sin(theta), theta)
        )
        self.assertAlmostEqual(integral, 1.0, places=4)
        self.assertEqual(U_tophat(1.01 * theta_q, THETA_Q), 0.0)

    def test_u_filters_are_compensated(self):
        theta_q = np.radians(THETA_Q / 60)
        theta = np.linspace(0, 10 * theta_q, 400001)
        for fn in (U_crittenden, U_schneider):
            with self.subTest(fn=fn.__name__):
                peak = np.trapezoid(np.abs(fn(theta, THETA_Q)) * theta, theta)
                net = np.trapezoid(fn(theta, THETA_Q) * theta, theta)
                self.assertLess(abs(net), 1e-6 * peak)


class TestScalarApertureFilter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corr = make_corr()
        cls.corr.preprocess()
        cls.shear, cls.k, cls.w_shear, cls.w_k = random_maps()

    def m_g(self, aperture_filter=None):
        return np.asarray(
            self.corr.m_g(
                self.k, self.w_k, aperture_filter=aperture_filter, return_device=False
            )
        )

    def m_a(self, aperture_filter=None):
        return np.asarray(
            self.corr.m_a(
                self.shear,
                self.w_shear,
                aperture_filter=aperture_filter,
                return_device=False,
            )
        )

    def test_default_is_the_top_hat(self):
        np.testing.assert_array_equal(self.m_g(), self.m_g(U_tophat))
        self.assertFalse(np.allclose(self.m_g(), self.m_g(Q_crittenden)))
        np.testing.assert_array_equal(
            self.corr.kk(self.k, self.w_k, return_device=False).M_g, self.m_g()
        )

    def test_scalar_values_reproduce_the_shear_filter_geometry(self):
        """Evaluated from the stored geometry, Q at the reconstructed
        distances is bitwise the Q evaluated during the disc search -- so
        ``m_g(..., aperture_filter=Q_crittenden)`` is the 7.6.0 default."""
        self.corr._ensure_aperture_pairs(Q_crittenden)
        np.testing.assert_array_equal(
            self.corr._k_aperture_values(Q_crittenden), self.corr.Q_val_flat
        )

    def test_changing_the_scalar_filter_keeps_the_geometry(self):
        m_a = self.m_a()
        geometry = self.corr.Q_inds_flat
        with patch.object(
            self.corr, "find_aperture_pairs", wraps=self.corr.find_aperture_pairs
        ) as spy:
            for fn in (U_crittenden, U_schneider, None, U_crittenden):
                self.m_g(fn)
        self.assertEqual(spy.call_count, 0)
        self.assertIs(self.corr.Q_inds_flat, geometry)
        np.testing.assert_array_equal(self.m_a(), m_a)

    def test_three_by_two_pt_uses_both_filters(self):
        for g_filter, k_filter in (
            (None, None),
            (Q_schneider, U_crittenden),
            (None, U_schneider),
        ):
            with self.subTest(g=getattr(g_filter, "__name__", None), k=k_filter):
                res = self.corr.three_by_two_pt(
                    self.shear,
                    self.k,
                    weights=(self.w_shear, self.w_k),
                    aperture_filter=g_filter,
                    k_aperture_filter=k_filter,
                    return_device=False,
                )
                np.testing.assert_allclose(
                    res.M_a, self.m_a(g_filter), rtol=1e-12, atol=1e-15
                )
                np.testing.assert_allclose(
                    res.M_g, self.m_g(k_filter), rtol=1e-12, atol=1e-15
                )

    def test_kg_uses_both_filters(self):
        res = self.corr.kg(
            self.k,
            self.shear,
            self.w_k,
            self.w_shear,
            m_g=True,
            m_a=True,
            aperture_filter=Q_schneider,
            k_aperture_filter=U_crittenden,
            return_device=False,
        )
        np.testing.assert_array_equal(res.M_a, self.m_a(Q_schneider))
        np.testing.assert_array_equal(res.M_g, self.m_g(U_crittenden))

    def test_loaded_pairs_evaluate_the_scalar_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pairs.h5")
            self.corr.save_pairs(path)
            loaded = make_corr()
            loaded.load_pairs(path)
        for fn in (None, U_crittenden):
            with self.subTest(filter=fn):
                np.testing.assert_array_equal(
                    np.asarray(
                        loaded.m_g(
                            self.k, self.w_k, aperture_filter=fn, return_device=False
                        )
                    ),
                    self.m_g(fn),
                )


class TestCoarseApertureScalarFilter(unittest.TestCase):
    def test_coarse_level_distances(self):
        """aperture_nside: ids are virtual rows of the coarse level; the
        reconstructed distances are those of the coarse pixels."""
        corr = make_corr(aperture_nside=NSIDE // 2)
        corr.find_aperture_pairs()
        np.testing.assert_array_equal(
            corr._k_aperture_values(Q_crittenden), corr.Q_val_flat
        )
        self.assertTrue(np.any(corr._k_aperture_values() > 0))


class TestConvergenceApertureMass(unittest.TestCase):
    """M_ap = ∫ U κ dΩ = ∫ Q γ_t dΩ: the compensated scalar aperture on a
    κ map equals the shear aperture on its (E-mode) shear field, up to
    pixelisation (it converges with θ_Q / pixel: slope 1.015 at 6.6
    pixels, 1.0015 at 13)."""

    def test_kappa_and_shear_agree(self):
        nside, theta_q = 256, 90.0
        lmax = 3 * nside - 1
        ell = np.arange(lmax + 1)
        cl = np.zeros(lmax + 1)
        cl[2:] = 1.0 / ell[2:] ** 2
        cl *= hp.gauss_beam(np.radians(20 / 60), lmax=lmax) ** 2
        np.random.seed(2)
        klm = hp.synalm(cl, lmax=lmax)
        kappa = hp.alm2map(klm, nside, lmax=lmax)
        fac = np.zeros(lmax + 1)
        fac[2:] = np.sqrt((ell[2:] + 2) * (ell[2:] - 1) / (ell[2:] * (ell[2:] + 1)))
        g1, g2 = hp.alm2map_spin([hp.almxfl(klm, -fac), 0 * klm], nside, 2, lmax)
        g2 = -g2  # healpy's polarisation convention -> lensing shear

        rng = np.random.default_rng(3)
        ncent = 20
        corr = Integrated3PCF(
            nside,
            rng.uniform(0, 2 * np.pi, ncent),
            np.arccos(rng.uniform(-0.9, 0.9, ncent)),
            nbins=2,
            theta_min=10,
            theta_max=60,
            patch_size=60,
            theta_Q=theta_q,
            device="cpu",
            rotation_precision="float64",
        )
        corr.find_aperture_pairs()
        w = np.ones((1, kappa.size))
        m_kappa = np.asarray(
            corr.m_g(kappa[None], w, aperture_filter=U_crittenden, return_device=False)
        )[0]
        m_shear = np.asarray(
            corr.m_a(np.stack([g1, g2])[None], w, return_device=False)
        )[0]
        rms = m_kappa.std()
        self.assertGreater(np.corrcoef(m_kappa, m_shear)[0, 1], 0.9995)
        self.assertLess(abs(np.polyfit(m_kappa, m_shear, 1)[0] - 1), 0.03)
        self.assertLess(np.abs(m_shear - m_kappa).max(), 0.1 * rms)


if __name__ == "__main__":
    unittest.main()
