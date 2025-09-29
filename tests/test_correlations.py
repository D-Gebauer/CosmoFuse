"""Tests for the Correlation class."""

import sys
import unittest
from pathlib import Path

import numpy as np

# Add src to path for testing
sys.path.insert(1, str(Path(__file__).parent.parent / "src"))

from CosmoFuse.correlations import Correlation


class TestCorrelation(unittest.TestCase):
    """Test the Correlation class."""

    def setUp(self):
        """Set up test fixtures."""
        self.nside = 64
        self.phi_center = np.array([0.0, np.pi / 2, np.pi])
        self.theta_center = np.array([np.pi / 4, np.pi / 3, np.pi / 2])
        self.nbins = 5
        self.theta_min = 10.0
        self.theta_max = 100.0
        self.patch_size = 60.0
        self.theta_Q = 30.0

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        corr = Correlation(
            nside=self.nside,
            phi_center=self.phi_center,
            theta_center=self.theta_center,
            nbins=self.nbins,
            theta_min=self.theta_min,
            theta_max=self.theta_max,
            patch_size=self.patch_size,
            theta_Q=self.theta_Q,
        )

        self.assertEqual(corr.nside, self.nside)
        self.assertEqual(corr.nbins, self.nbins)
        self.assertEqual(corr.n_patches, len(self.phi_center))

    def test_init_invalid_nside(self):
        """Test initialization with invalid nside."""
        with self.assertRaises(ValueError):
            Correlation(
                nside=0, phi_center=self.phi_center, theta_center=self.theta_center
            )

    def test_init_invalid_nbins(self):
        """Test initialization with invalid nbins."""
        with self.assertRaises(ValueError):
            Correlation(
                nside=self.nside,
                phi_center=self.phi_center,
                theta_center=self.theta_center,
                nbins=0,
            )

    def test_init_invalid_theta_range(self):
        """Test initialization with invalid theta range."""
        with self.assertRaises(ValueError):
            Correlation(
                nside=self.nside,
                phi_center=self.phi_center,
                theta_center=self.theta_center,
                theta_min=100.0,
                theta_max=10.0,
            )

    def test_init_invalid_patch_size(self):
        """Test initialization with invalid patch_size."""
        with self.assertRaises(ValueError):
            Correlation(
                nside=self.nside,
                phi_center=self.phi_center,
                theta_center=self.theta_center,
                patch_size=0,
            )

    def test_init_invalid_theta_Q(self):
        """Test initialization with invalid theta_Q."""
        with self.assertRaises(ValueError):
            Correlation(
                nside=self.nside,
                phi_center=self.phi_center,
                theta_center=self.theta_center,
                theta_Q=0,
            )

    def test_init_mismatched_centers(self):
        """Test initialization with mismatched center arrays."""
        with self.assertRaises(ValueError):
            Correlation(
                nside=self.nside,
                phi_center=self.phi_center,
                theta_center=np.array([0.0, np.pi / 2]),  # Different length
            )

    def test_init_with_mask(self):
        """Test initialization with a mask."""
        mask = np.ones(12 * self.nside**2, dtype=bool)
        mask[::2] = False  # Set every other pixel to False

        corr = Correlation(
            nside=self.nside,
            phi_center=self.phi_center,
            theta_center=self.theta_center,
            mask=mask,
        )

        self.assertLess(len(corr.map_inds), 12 * self.nside**2)

    def test_init_invalid_mask_length(self):
        """Test initialization with invalid mask length."""
        mask = np.ones(100, dtype=bool)  # Wrong length

        with self.assertRaises(ValueError):
            Correlation(
                nside=self.nside,
                phi_center=self.phi_center,
                theta_center=self.theta_center,
                mask=mask,
            )

    def test_get_pairs_patch(self):
        """Test get_pairs_patch method."""
        corr = Correlation(
            nside=self.nside,
            phi_center=self.phi_center,
            theta_center=self.theta_center,
            nbins=self.nbins,
        )

        # Create test data
        patch_inds = np.array([0, 1, 2, 3])
        ra = np.array([0.0, 0.1, 0.2, 0.3])
        dec = np.array([0.0, 0.1, 0.2, 0.3])

        all_inds, exp2phi = corr.get_pairs_patch(patch_inds, ra, dec)

        self.assertIsInstance(all_inds, list)
        self.assertIsInstance(exp2phi, np.ndarray)


if __name__ == "__main__":
    unittest.main()
