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
