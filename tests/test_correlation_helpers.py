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

    def test_radec_to_xyz(self):
        """Test radec_to_xyz function."""
        ra = 0.0
        dec = 0.0
        x, y, z = radec_to_xyz(ra, dec)

        self.assertAlmostEqual(x, 1.0, places=10)
        self.assertAlmostEqual(y, 0.0, places=10)
        self.assertAlmostEqual(z, 0.0, places=10)

    def test_getAngle_known_geometry(self):
        """Test getAngle on a known right-angle spherical configuration."""
        # C at (0, 0), A at (pi/2, 0) (east), B at (0, pi/2) (north).
        ra1, dec1 = 0.0, 0.0
        ra2, dec2 = np.pi / 2, 0.0
        ra3, dec3 = 0.0, np.pi / 2

        angle = getAngle(ra1, dec1, ra2, dec2, ra3, dec3)

        np.testing.assert_allclose(angle, np.pi / 2, rtol=0.0, atol=1e-7)

    def test_getAngle_tangent_plane(self):
        """Test getAngle against a tangent-plane angle calculation."""
        ra1, dec1 = 0.3, 0.4
        ra2, dec2 = 1.1, -0.2
        ra3, dec3 = -0.7, 0.5

        angle = getAngle(ra1, dec1, ra2, dec2, ra3, dec3)

        c = np.array(radec_to_xyz(ra1, dec1))
        a = np.array(radec_to_xyz(ra2, dec2))
        b = np.array(radec_to_xyz(ra3, dec3))

        # Project directions onto the tangent plane at C.
        u = a - np.dot(a, c) * c
        v = b - np.dot(b, c) * c
        u /= np.linalg.norm(u)
        v /= np.linalg.norm(v)

        sin_term = np.dot(c, np.cross(u, v))
        cos_term = np.dot(u, v)
        expected = np.arctan2(sin_term, cos_term)

        np.testing.assert_allclose(angle, expected, rtol=0.0, atol=1e-7)

    def test_getAngle_right_angle(self):
        """Test getAngle against an independent vector-based formula."""
        ra1, dec1 = 0.0, 0.0
        ra2, dec2 = 0.0, np.pi / 2
        ra3, dec3 = np.pi / 2, 0.0

        angle = getAngle(ra1, dec1, ra2, dec2, ra3, dec3)

        c1 = np.array(radec_to_xyz(ra1, dec1))
        c2 = np.array(radec_to_xyz(ra2, dec2))
        c3 = np.array(radec_to_xyz(ra3, dec3))
        sin_term = np.dot(c1, np.cross(c2, c3))
        cos_term = np.dot(np.cross(c1, c2), np.cross(c1, c3))
        expected = np.arctan2(sin_term, cos_term)

        self.assertIsInstance(angle, float)
        np.testing.assert_allclose(angle, expected, rtol=0.0, atol=1e-7)

    def test_Q_T(self):
        """Test Q_T function."""
        theta = 1.0
        theta_Q = 90.0

        result = Q_T(theta, theta_Q)

        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)

    def test_zeta(self):
        """Test zeta function."""
        nmaps = 1
        zbins = 2
        n_patches = 3
        nbins = 4
        n_correlations = zbins * (zbins + 1) // 2
        n_zeta_combs = zbins * (zbins + 1) * (zbins + 2) // 6

        rng = np.random.default_rng(42)
        M_ap = rng.random((nmaps, zbins, n_patches))
        xip = rng.random((nmaps, n_correlations, n_patches, nbins))
        xim = rng.random((nmaps, n_correlations, n_patches, nbins))

        zetap, zetam = zeta(M_ap, xip, xim)
        self.assertIsInstance(zetap, np.ndarray)
        self.assertIsInstance(zetam, np.ndarray)
        self.assertEqual(zetap.shape, (nmaps, n_zeta_combs, nbins))
        self.assertEqual(zetam.shape, (nmaps, n_zeta_combs, nbins))


if __name__ == "__main__":
    unittest.main()
