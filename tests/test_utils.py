"""Tests for utility functions."""

import sys
import unittest
from pathlib import Path

import healpy as hp
import numpy as np

# Add src to path for testing
sys.path.insert(1, str(Path(__file__).parent.parent / "src"))

from CosmoFuse.correlations import Correlation
from CosmoFuse.utils import pixel2RaDec, select_patch_centers


class TestUtils(unittest.TestCase):
    """Test utility functions."""

    def test_pixel2RaDec_single_pixel(self):
        """Test pixel2RaDec with a single pixel."""
        nside = 64
        pixel = 0
        ra, dec = pixel2RaDec(pixel, nside)
        
        # For single pixel, healpy returns scalar values, not arrays
        self.assertIsInstance(ra, (np.ndarray, np.floating, float))
        self.assertIsInstance(dec, (np.ndarray, np.floating, float))

    def test_pixel2RaDec_array(self):
        """Test pixel2RaDec with an array of pixels."""
        nside = 64
        pixels = np.array([0, 1, 2, 3])
        ra, dec = pixel2RaDec(pixels, nside)

        self.assertIsInstance(ra, np.ndarray)
        self.assertIsInstance(dec, np.ndarray)
        self.assertEqual(ra.shape, pixels.shape)
        self.assertEqual(dec.shape, pixels.shape)

    def test_pixel2RaDec_uint64_array(self):
        """Test pixel2RaDec with uint64 pixel indices."""
        nside = 64
        pixels = np.array([0, 1, 2, 3], dtype=np.uint64)
        ra, dec = pixel2RaDec(pixels, nside)

        self.assertIsInstance(ra, np.ndarray)
        self.assertIsInstance(dec, np.ndarray)
        self.assertEqual(ra.shape, pixels.shape)
        self.assertEqual(dec.shape, pixels.shape)

    def test_pixel2RaDec_uint64_overflow_raises(self):
        """Test pixel2RaDec rejects uint64 values beyond int64 range."""
        nside = 64
        pixels = np.array([np.iinfo(np.int64).max + 1], dtype=np.uint64)

        with self.assertRaises(ValueError):
            pixel2RaDec(pixels, nside)


class TestSelectPatchCenters(unittest.TestCase):
    """Tests for mask-based patch-centre selection."""

    NSIDE_MASK = 64
    CAP_RADIUS_DEG = 30.0

    @classmethod
    def setUpClass(cls):
        """Footprint: a 30-degree polar cap (nonzero = observed)."""
        npix = hp.nside2npix(cls.NSIDE_MASK)
        theta_pix, _ = hp.pix2ang(cls.NSIDE_MASK, np.arange(npix))
        cls.mask = (theta_pix < np.radians(cls.CAP_RADIUS_DEG)).astype(np.float64)

    def test_accepted_centers_respect_mask_fractions(self):
        patch_size = 90.0  # arcmin -> filter support disc = 7.5 deg
        f_mask = 0.05
        phi_c, theta_c = select_patch_centers(
            self.mask, nside_centers=8, patch_size=patch_size, f_mask=f_mask
        )

        self.assertGreater(phi_c.size, 0)
        self.assertEqual(phi_c.shape, theta_c.shape)

        # Recompute both masked fractions for every accepted centre.
        unmasked = self.mask != 0
        vecs = hp.ang2vec(theta_c, phi_c)
        for i in range(phi_c.size):
            for radius in (np.radians(patch_size / 60), 5 * np.radians(patch_size / 60)):
                disc = hp.query_disc(self.NSIDE_MASK, vecs[i], radius)
                frac = 1.0 - np.count_nonzero(unmasked[disc]) / disc.size
                self.assertLessEqual(frac, f_mask)

    def test_return_order_matches_constructor(self):
        """Second return value is the colatitude: for a polar-cap footprint
        every accepted centre must lie near the pole (theta small), while
        phi spans the full 2*pi longitude range."""
        phi_c, theta_c = select_patch_centers(
            self.mask, nside_centers=8, patch_size=90.0, f_mask=0.05
        )
        self.assertLess(np.max(theta_c), np.radians(self.CAP_RADIUS_DEG))
        self.assertGreater(np.max(phi_c), np.pi)

    def test_looser_f_mask_accepts_more_centers(self):
        strict = select_patch_centers(self.mask, 8, patch_size=90.0, f_mask=0.01)
        loose = select_patch_centers(self.mask, 8, patch_size=90.0, f_mask=0.5)
        self.assertGreater(loose[0].size, strict[0].size)

    def test_separate_filter_threshold(self):
        """A tight filter-disc threshold must not be satisfied by centres
        that only pass the (smaller) patch disc."""
        both_loose = select_patch_centers(
            self.mask, 8, patch_size=90.0, f_mask=0.5, f_mask_filter=0.5
        )
        filter_tight = select_patch_centers(
            self.mask, 8, patch_size=90.0, f_mask=0.5, f_mask_filter=0.01
        )
        self.assertLess(filter_tight[0].size, both_loose[0].size)

    def test_empty_footprint_returns_empty(self):
        phi_c, theta_c = select_patch_centers(
            np.zeros_like(self.mask), nside_centers=8, patch_size=90.0
        )
        self.assertEqual(phi_c.size, 0)
        self.assertEqual(theta_c.size, 0)

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            select_patch_centers(self.mask, 8, patch_size=-1.0)
        with self.assertRaises(ValueError):
            select_patch_centers(self.mask, 8, f_mask=1.5)
        with self.assertRaises(ValueError):
            select_patch_centers(np.zeros((2, 12)), 8)

    def test_from_mask_builds_matching_correlation(self):
        phi_c, theta_c = select_patch_centers(
            self.mask, nside_centers=8, patch_size=90.0, f_mask=0.05
        )
        corr = Correlation.from_mask(
            self.NSIDE_MASK,
            self.mask,
            nside_centers=8,
            patch_size=90.0,
            theta_Q=90.0,
            f_mask=0.05,
            nbins=4,
            theta_min=10,
            theta_max=60,
            device="cpu",
        )
        self.assertEqual(corr.n_patches, phi_c.size)
        np.testing.assert_allclose(corr.phi_center, phi_c)
        np.testing.assert_allclose(corr.theta_center, theta_c)
        np.testing.assert_array_equal(corr.map_inds, np.flatnonzero(self.mask))

    def test_from_mask_regrades_mask_resolution(self):
        """Selection may run on a finer mask than the measurement nside."""
        corr = Correlation.from_mask(
            32,
            self.mask,  # nside 64 footprint
            nside_centers=8,
            patch_size=90.0,
            f_mask=0.05,
            nbins=4,
            theta_min=10,
            theta_max=60,
            device="cpu",
        )
        self.assertEqual(corr.map_mask.size, hp.nside2npix(32))
        self.assertGreater(corr.n_patches, 0)

    def test_from_mask_raises_when_nothing_accepted(self):
        with self.assertRaises(ValueError):
            Correlation.from_mask(
                self.NSIDE_MASK,
                np.zeros_like(self.mask),
                nside_centers=8,
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
