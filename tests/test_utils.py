"""Tests for utility functions."""

import sys
import unittest
from pathlib import Path

import numpy as np

# Add src to path for testing
sys.path.insert(1, str(Path(__file__).parent.parent / "src"))

from CosmoFuse.utils import pixel2RaDec


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



if __name__ == "__main__":
    unittest.main()
