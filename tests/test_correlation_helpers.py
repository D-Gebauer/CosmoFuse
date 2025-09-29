"""Tests for correlation helper functions."""

import sys
import unittest
from pathlib import Path

import numpy as np

# Add src to path for testing
sys.path.insert(1, str(Path(__file__).parent.parent / "src"))

from CosmoFuse.correlation_helpers import (
    Q_T,
    M_a_patch,
    getAngle,
    radec_to_xyz,
    xipm_patch,
    zeta,
)


class TestCorrelationHelpers(unittest.TestCase):
    """Test correlation helper functions."""

    def test_M_a_patch(self):
        """Test M_a_patch function."""
        # Create test data
        Q_inds = np.array([0, 1, 2])
        Q_cos = np.array([1.0, 0.5, 0.0])
        Q_sin = np.array([0.0, 0.5, 1.0])
        Q_val = np.array([0.1, 0.2, 0.3])
        g1 = np.array([0.01, 0.02, 0.03, 0.04])
        g2 = np.array([0.005, 0.015, 0.025, 0.035])
        Q_w = np.array([1.0, 1.0, 1.0, 1.0])
        Q_patch_area = 1.0

        result = M_a_patch(Q_inds, Q_cos, Q_sin, Q_val, g1, g2, Q_w, Q_patch_area)

        self.assertIsInstance(result, float)

    def test_xipm_patch(self):
        """Test xipm_patch function."""
        # Create test data
        inds = np.array([[0, 1], [2, 3]])
        exp2theta = np.array([1.0 + 0.0j, 1.0 + 0.0j])
        bin_inds = np.array([2])
        g11 = np.array([0.01, 0.02, 0.03, 0.04])
        g21 = np.array([0.005, 0.015, 0.025, 0.035])
        g12 = np.array([0.02, 0.03, 0.04, 0.05])
        g22 = np.array([0.01, 0.02, 0.03, 0.04])
        nbins = 1

        xip, xim = xipm_patch(inds, exp2theta, bin_inds, g11, g21, g12, g22, nbins)

        self.assertIsInstance(xip, np.ndarray)
        self.assertIsInstance(xim, np.ndarray)
        self.assertEqual(len(xip), nbins)
        self.assertEqual(len(xim), nbins)

    def test_radec_to_xyz(self):
        """Test radec_to_xyz function."""
        ra = 0.0
        dec = 0.0
        x, y, z = radec_to_xyz(ra, dec)

        self.assertAlmostEqual(x, 1.0, places=10)
        self.assertAlmostEqual(y, 0.0, places=10)
        self.assertAlmostEqual(z, 0.0, places=10)

    def test_getAngle(self):
        """Test getAngle function."""
        ra1, dec1 = 0.0, 0.0
        ra2, dec2 = np.pi / 2, 0.0
        ra3, dec3 = 0.0, np.pi / 2

        angle = getAngle(ra1, dec1, ra2, dec2, ra3, dec3)

        self.assertIsInstance(angle, float)
        self.assertGreaterEqual(angle, 0.0)
        self.assertLessEqual(angle, np.pi)

    def test_Q_T(self):
        """Test Q_T function."""
        theta = 1.0
        theta_Q = 90.0

        result = Q_T(theta, theta_Q)

        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)

    def test_zeta(self):
        """Test zeta function."""
        # Create test data
        nmaps = 1
        zbins = 2
        n_patches = 3
        nbins = 4

        M_ap = np.random.rand(nmaps, zbins, n_patches)
        xip = np.random.rand(nmaps, zbins, n_patches, nbins)
        xim = np.random.rand(nmaps, zbins, n_patches, nbins)

        zetap, zetam = zeta(M_ap, xip, xim)

        self.assertIsInstance(zetap, np.ndarray)
        self.assertIsInstance(zetam, np.ndarray)
        self.assertEqual(zetap.shape[0], nmaps)
        self.assertEqual(zetam.shape[0], nmaps)


if __name__ == "__main__":
    unittest.main()
