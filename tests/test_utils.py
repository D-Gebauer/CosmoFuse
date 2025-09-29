"""Tests for utility functions."""

import sys
import unittest
from pathlib import Path

import numpy as np

# Add src to path for testing
sys.path.insert(1, str(Path(__file__).parent.parent / "src"))

from CosmoFuse.utils import eval_func_tuple, pixel2RaDec, set_mpl_params


class TestUtils(unittest.TestCase):
    """Test utility functions."""

    def test_pixel2RaDec_single_pixel(self):
        """Test pixel2RaDec with a single pixel."""
        nside = 64
        pixel = 0
        ra, dec = pixel2RaDec(pixel, nside)

        self.assertIsInstance(ra, np.ndarray)
        self.assertIsInstance(dec, np.ndarray)
        self.assertEqual(ra.shape, ())
        self.assertEqual(dec.shape, ())

    def test_pixel2RaDec_array(self):
        """Test pixel2RaDec with an array of pixels."""
        nside = 64
        pixels = np.array([0, 1, 2, 3])
        ra, dec = pixel2RaDec(pixels, nside)

        self.assertIsInstance(ra, np.ndarray)
        self.assertIsInstance(dec, np.ndarray)
        self.assertEqual(ra.shape, pixels.shape)
        self.assertEqual(dec.shape, pixels.shape)

    def test_set_mpl_params(self):
        """Test that set_mpl_params runs without error."""
        # This test just ensures the function doesn't crash
        set_mpl_params()

    def test_eval_func_tuple(self):
        """Test eval_func_tuple with a simple function."""

        def add(a, b):
            return a + b

        result = eval_func_tuple((add, 2, 3))
        self.assertEqual(result, 5)

    def test_eval_func_tuple_no_args(self):
        """Test eval_func_tuple with a function that takes no arguments."""

        def get_five():
            return 5

        result = eval_func_tuple((get_five,))
        self.assertEqual(result, 5)


if __name__ == "__main__":
    unittest.main()
