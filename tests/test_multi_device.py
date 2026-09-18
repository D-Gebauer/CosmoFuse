"""Patch-parallel multi-device group (idea #6).

The split must be invisible: every public measurement method has to return
exactly what a single-device run returns, bitwise.  Exercised with two CPU
"devices" so it runs in CI; on a real multi-GPU box the only difference is
which backend each part holds.
"""

import unittest

import healpy as hp
import numpy as np

from CosmoFuse import Correlation
from CosmoFuse.multi_device import MultiDeviceCorrelation

NSIDE = 16
NPIX = hp.nside2npix(NSIDE)


def make_setup(n_patches=4, devices=None):
    theta_pix, _ = hp.pix2ang(NSIDE, np.arange(NPIX))
    mask = (theta_pix < 1.4).astype(float)
    rng = np.random.default_rng(11)
    phi = rng.uniform(0.2, 1.1, size=n_patches)
    theta = rng.uniform(0.7, 1.05, size=n_patches)
    kwargs = dict(
        nbins=3,
        theta_min=100,
        theta_max=400,
        patch_size=250,
        theta_Q=250,
        mask=mask,
        map_precision="float64",
        rotation_precision="float64",
    )
    device = devices if devices is not None else "cpu"
    corr = Correlation(NSIDE, phi, theta, device=device, **kwargs)
    corr.preprocess()
    return corr


class TestMultiDevice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.single = make_setup()
        cls.multi = make_setup(devices=["cpu", "cpu"])
        rng = np.random.default_rng(3)
        n = cls.single.n_active
        cls.shear = rng.normal(size=(2, 2, n))
        cls.dens = rng.normal(size=(2, n))
        cls.w = rng.uniform(0.5, 2.0, size=(2, n))

    def _same(self, a, b):
        a = a if isinstance(a, tuple) else (a,)
        b = b if isinstance(b, tuple) else (b,)
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            x, y = np.asarray(x), np.asarray(y)
            self.assertEqual(x.shape, y.shape)
            self.assertTrue(np.array_equal(x, y))

    def test_dispatch(self):
        self.assertIsInstance(self.multi, MultiDeviceCorrelation)
        self.assertEqual(len(self.multi), 2)
        self.assertEqual(self.multi.n_patches, self.single.n_patches)
        self.assertEqual(self.multi.patch_ranges, [(0, 2), (2, 4)])
        self.assertEqual(self.multi.ntotpairs, self.single.ntotpairs)
        self.assertEqual(self.multi.nbins, self.single.nbins)

    def test_single_element_list_is_a_plain_correlation(self):
        corr = Correlation(NSIDE, np.array([0.5]), np.array([0.9]), device=["cpu"])
        self.assertIsInstance(corr, Correlation)

    def test_measurements_match_single_device(self):
        s, m, w = self.single, self.multi, self.w
        for name, args in (
            ("get_full_tomo_shear", (self.shear, w)),
            ("vectorized_shear_shear", (self.shear, w)),
            ("get_full_tomo_density", (self.dens, w)),
            ("vectorized_density_density", (self.dens, w)),
            ("get_full_tomo_ggl", (self.dens, self.shear, w, w)),
            ("get_3x2pt_tomo", (self.shear, self.dens, w)),
        ):
            with self.subTest(method=name):
                ref = getattr(s, name)(*args, return_device=False)
                self._same(ref, getattr(m, name)(*args))

    def test_aperture_matches_single_device(self):
        ref = self.single.get_aperture_shear(
            self.shear[0, 0], self.shear[0, 1], self.w[0]
        )
        got = self.multi.get_aperture_shear(
            self.shear[0, 0], self.shear[0, 1], self.w[0]
        )
        self._same(np.asarray(ref), got)

    def test_uneven_split(self):
        single = make_setup(n_patches=5)
        multi = make_setup(n_patches=5, devices=["cpu", "cpu"])
        self.assertEqual(multi.patch_ranges, [(0, 2), (2, 5)])
        rng = np.random.default_rng(5)
        n = single.n_active
        shear = rng.normal(size=(2, 2, n))
        w = rng.uniform(0.5, 2.0, size=(2, n))
        self._same(
            single.get_full_tomo_shear(shear, w, return_device=False),
            multi.get_full_tomo_shear(shear, w),
        )

    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "at least one device"):
            MultiDeviceCorrelation(NSIDE, np.array([0.5]), np.array([0.9]), devices=[])
        with self.assertRaisesRegex(ValueError, "duplicate device ids"):
            MultiDeviceCorrelation(
                NSIDE, np.array([0.5, 0.6]), np.array([0.9, 0.9]), devices=[0, 0]
            )
        with self.assertRaisesRegex(ValueError, "cannot be split"):
            MultiDeviceCorrelation(
                NSIDE, np.array([0.5]), np.array([0.9]), devices=["cpu", "cpu"]
            )
        with self.assertRaises(NotImplementedError):
            self.multi.save_pairs("/tmp/should-not-be-written.h5")

    def test_load_pairs_slices_per_device(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pairs.h5")
            self.single.save_pairs(path)
            multi = Correlation(
                NSIDE,
                self.single.phi_center,
                self.single.theta_center,
                nbins=3,
                theta_min=100,
                theta_max=400,
                patch_size=250,
                theta_Q=250,
                mask=self.single.map_mask.astype(float),
                map_precision="float64",
                rotation_precision="float64",
                device=["cpu", "cpu"],
            )
            multi.load_pairs(path)
            self.assertEqual(multi.ntotpairs, self.single.ntotpairs)
            self._same(
                self.single.get_full_tomo_shear(self.shear, self.w, return_device=False),
                multi.get_full_tomo_shear(self.shear, self.w),
            )
            with self.assertRaisesRegex(ValueError, "holds"):
                multi.load_pairs(path, start_ind=1)


if __name__ == "__main__":
    unittest.main()
