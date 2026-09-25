"""Patch-parallel multi-device group (idea #6).

The split must be invisible: every public measurement method has to return
exactly what a single-device run returns, bitwise.  Exercised with two CPU
"devices" so it runs in CI; on a real multi-GPU box the only difference is
which backend each part holds.
"""

import unittest

import healpy as hp
import numpy as np

from CosmoFuse import Integrated3PCF
from CosmoFuse.multi_device import MultiDeviceIntegrated3PCF
from ._results import fields_of

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
    corr = Integrated3PCF(NSIDE, phi, theta, device=device, **kwargs)
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
        if hasattr(a, "fields"):
            self.assertEqual(a.gg_combinations, b.gg_combinations)
            self.assertEqual(a.kk_combinations, b.kk_combinations)
            self.assertEqual(a.kg_combinations, b.kg_combinations)
            a, b = a.fields(), b.fields()
            self.assertEqual(list(a), list(b))
            a, b = tuple(a.values()), tuple(b.values())
        a = a if isinstance(a, tuple) else (a,)
        b = b if isinstance(b, tuple) else (b,)
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            x, y = np.asarray(x), np.asarray(y)
            self.assertEqual(x.shape, y.shape)
            self.assertTrue(np.array_equal(x, y))

    def test_dispatch(self):
        self.assertIsInstance(self.multi, MultiDeviceIntegrated3PCF)
        self.assertEqual(len(self.multi), 2)
        self.assertEqual(self.multi.n_patches, self.single.n_patches)
        self.assertEqual(self.multi.patch_ranges, [(0, 2), (2, 4)])
        self.assertEqual(self.multi.ntotpairs, self.single.ntotpairs)
        self.assertEqual(self.multi.nbins, self.single.nbins)

    def test_single_element_list_is_a_plain_correlation(self):
        corr = Integrated3PCF(NSIDE, np.array([0.5]), np.array([0.9]), device=["cpu"])
        self.assertIsInstance(corr, Integrated3PCF)

    def test_measurements_match_single_device(self):
        s, m, w = self.single, self.multi, self.w
        for name, args, kw in (
            ("gg", (self.shear, w), {}),
            ("gg", (self.shear, w), {"apertures": False}),
            ("kk", (self.dens, w), {}),
            ("kk", (self.dens, w), {"apertures": False}),
            ("kg", (self.dens, self.shear, w, w), {}),
            ("kg", (self.dens, self.shear, w, w), {"m_g": True, "m_a": True}),
            ("three_by_two_pt", (self.shear, self.dens, w), {}),
            ("m_a", (self.shear, w), {}),
            ("m_g", (self.dens, w), {}),
        ):
            with self.subTest(method=name, **kw):
                ref = getattr(s, name)(*args, return_device=False, **kw)
                self._same(ref, getattr(m, name)(*args, **kw))

    def test_uneven_split(self):
        single = make_setup(n_patches=5)
        multi = make_setup(n_patches=5, devices=["cpu", "cpu"])
        self.assertEqual(multi.patch_ranges, [(0, 2), (2, 5)])
        rng = np.random.default_rng(5)
        n = single.n_active
        shear = rng.normal(size=(2, 2, n))
        w = rng.uniform(0.5, 2.0, size=(2, n))
        self._same(
            fields_of(single.gg(shear, w, return_device=False), "M_a", "xi_p", "xi_m"),
            fields_of(multi.gg(shear, w), "M_a", "xi_p", "xi_m"),
        )

    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "at least one device"):
            MultiDeviceIntegrated3PCF(NSIDE, np.array([0.5]), np.array([0.9]), devices=[])
        with self.assertRaisesRegex(ValueError, "duplicate device ids"):
            MultiDeviceIntegrated3PCF(
                NSIDE, np.array([0.5, 0.6]), np.array([0.9, 0.9]), devices=[0, 0]
            )
        with self.assertRaisesRegex(ValueError, "cannot be split"):
            MultiDeviceIntegrated3PCF(
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
            multi = Integrated3PCF(
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
                fields_of(self.single.gg(self.shear, self.w, return_device=False), "M_a", "xi_p", "xi_m"),
                fields_of(multi.gg(self.shear, self.w), "M_a", "xi_p", "xi_m"),
            )
            with self.assertRaisesRegex(ValueError, "holds"):
                multi.load_pairs(path, start_ind=1)


if __name__ == "__main__":
    unittest.main()
