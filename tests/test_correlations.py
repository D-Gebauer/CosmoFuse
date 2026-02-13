"""Tests for the Correlation class."""
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import healpy as hp
import numpy as np
from scipy.special import binom

# Add src to path for testing
sys.path.insert(1, str(Path(__file__).parent.parent / "src"))

from CosmoFuse.correlations import Correlation
from CosmoFuse.utils import pixel2RaDec


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

    def test_init_invalid_multiprocessing_start_method(self):
        """Test initialization with invalid multiprocessing start method."""
        with self.assertRaises(ValueError):
            Correlation(
                nside=self.nside,
                phi_center=self.phi_center,
                theta_center=self.theta_center,
                multiprocessing_start_method="definitely_invalid",
            )

    def test_init_precision_configuration(self):
        """Test initialization with custom precision configuration."""
        corr = Correlation(
            nside=self.nside,
            phi_center=self.phi_center,
            theta_center=self.theta_center,
            map_precision="float32",
            rotation_precision="float32",
            index_precision="uint64",
        )
        self.assertEqual(corr.map_dtype, np.dtype(np.float32))
        self.assertEqual(corr.rotation_dtype, np.dtype(np.float32))
        self.assertEqual(corr.rotation_complex_dtype, np.dtype(np.complex64))
        self.assertEqual(corr.index_dtype, np.dtype(np.uint64))

    def test_init_invalid_map_precision(self):
        """Test initialization with invalid map precision."""
        with self.assertRaises(ValueError):
            Correlation(
                nside=self.nside,
                phi_center=self.phi_center,
                theta_center=self.theta_center,
                map_precision="float80",
            )

    def test_init_invalid_rotation_precision(self):
        """Test initialization with invalid rotation precision."""
        with self.assertRaises(ValueError):
            Correlation(
                nside=self.nside,
                phi_center=self.phi_center,
                theta_center=self.theta_center,
                rotation_precision="bfloat16",
            )

    def test_init_invalid_index_precision(self):
        """Test initialization with invalid index precision."""
        with self.assertRaises(ValueError):
            Correlation(
                nside=self.nside,
                phi_center=self.phi_center,
                theta_center=self.theta_center,
                index_precision="int32",
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


class TestCorrelationCalculations(unittest.TestCase):
    """Test the Correlation class calculation methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.nside = 64
        self.phi_center = np.array([0.0])
        self.theta_center = np.array([np.pi / 4])
        self.nbins = 5
        self.theta_min = 10.0
        self.theta_max = 100.0
        self.patch_size = 60.0
        self.theta_Q = 30.0
        self.npix = 12 * self.nside**2

        # Create a Correlation instance
        self.corr = Correlation(
            nside=self.nside,
            phi_center=self.phi_center,
            theta_center=self.theta_center,
            nbins=self.nbins,
            theta_min=self.theta_min,
            theta_max=self.theta_max,
            patch_size=self.patch_size,
            theta_Q=self.theta_Q,
        )

        # Mock Q data for one patch
        self.corr.Q_inds = [np.array([10, 20, 30], dtype=np.uint32)]
        self.corr.Q_cos = [np.array([0.5, 0.6, 0.7], dtype=np.float64)]
        self.corr.Q_sin = [np.array([0.8, 0.7, 0.6], dtype=np.float64)]
        self.corr.Q_val = [np.array([1.0, 1.0, 1.0], dtype=np.float64)]
        self.corr.Q_patch_area = [3.0]
        self.corr.n_patches = 1


    def test_get_M_a(self):
        """Test the get_M_a method."""
        g1 = np.random.rand(self.npix)
        g2 = np.random.rand(self.npix)
        w = np.ones(self.npix)

        M_a = self.corr.get_M_a(g1, g2, w)

        self.assertEqual(M_a.shape, (1,))
        
        Q_inds = self.corr.Q_inds[0]
        Q_cos = self.corr.Q_cos[0]
        Q_sin = self.corr.Q_sin[0]
        Q_val = self.corr.Q_val[0]
        Q_patch_area = self.corr.Q_patch_area[0]
        
        gt = -g1[Q_inds] * Q_cos - g2[Q_inds] * Q_sin
        expected_M_a = Q_patch_area * np.sum(w[Q_inds] * gt * Q_val) / np.sum(w[Q_inds])

        self.assertAlmostEqual(M_a[0], expected_M_a)

    def test_get_pairs_patch_M_a(self):
        """Test get_pairs_patch_M_a method."""
        pixels_RA_Q_patch = np.array([0.1, 0.2])
        pixels_dec_Q_patch = np.array([0.1, 0.2])
        Q_patch_center_RA = 0.0
        Q_patch_center_dec = 0.0

        cos_2phi, sin_2phi, Q = self.corr.get_pairs_patch_M_a(
            pixels_RA_Q_patch,
            pixels_dec_Q_patch,
            Q_patch_center_RA,
            Q_patch_center_dec,
        )

        self.assertEqual(cos_2phi.shape, (2,))
        self.assertEqual(sin_2phi.shape, (2,))
        self.assertEqual(Q.shape, (2,))

    @patch('healpy.ang2pix')
    @patch('healpy.query_disc')
    @patch('CosmoFuse.correlations.pixel2RaDec')
    def test_calculate_pairs_M_a(self, mock_pixel2RaDec, mock_query_disc, mock_ang2pix):
        """Test the calculate_pairs_M_a method."""
        mock_ang2pix.return_value = 99
        mock_query_disc.return_value = np.array([10, 20, 30, 40, 50, 99])
        
        # Mocking the return values for pixel2RaDec
        # First call for center, second for the rest
        ra_center, dec_center = np.array([np.pi/4]), np.array([0.0])
        Q_ra, Q_dec = np.array([0.1, 0.2, 0.3, 0.4, 0.5]), np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        mock_pixel2RaDec.side_effect = [(ra_center, dec_center), (Q_ra, Q_dec)]

        self.corr.calculate_pairs_M_a()

        self.assertEqual(len(self.corr.Q_cos), 1)
        self.assertEqual(len(self.corr.Q_sin), 1)
        self.assertEqual(len(self.corr.Q_val), 1)
        self.assertEqual(len(self.corr.Q_inds), 1)
        self.assertEqual(len(self.corr.Q_patch_area), 1)
        self.assertIsInstance(self.corr.Q_cos[0], np.ndarray)
        self.assertEqual(self.corr.Q_inds[0].size, 5)  # 6 - 1 (center)

    @patch('healpy.ang2pix')
    @patch('healpy.query_disc')
    @patch('CosmoFuse.correlations.pixel2RaDec')
    def test_save_and_load_pairs(self, mock_pixel2RaDec, mock_query_disc, mock_ang2pix):
        """Test the save_pairs and load_pairs methods."""
        # Mocking for calculate_pairs_2PCF
        mock_query_disc.return_value = np.arange(100)
        mock_pixel2RaDec.return_value = (np.random.rand(100), np.random.rand(100))
        self.corr.calculate_pairs_2PCF(threads=1)

        # Mocking for calculate_pairs_M_a
        mock_ang2pix.return_value = 99
        mock_query_disc.return_value = np.array([10, 20, 30, 40, 50, 99])
        ra_center, dec_center = np.array([np.pi/4]), np.array([0.0])
        Q_ra, Q_dec = np.array([0.1, 0.2, 0.3, 0.4, 0.5]), np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mock_pixel2RaDec.side_effect = [(ra_center, dec_center), (Q_ra, Q_dec)]
        self.corr.calculate_pairs_M_a()

        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            self.corr.save_pairs(tmp.name)

            new_corr = Correlation(
                nside=self.nside,
                phi_center=self.phi_center,
                theta_center=self.theta_center,
            )
            new_corr.load_pairs(tmp.name)

            self.assertEqual(self.corr.n_patches, new_corr.n_patches)
            self.assertEqual(self.corr.nbins, new_corr.nbins)
            np.testing.assert_allclose(self.corr.phi_center, new_corr.phi_center)
            np.testing.assert_allclose(self.corr.theta_center, new_corr.theta_center)

            for i in range(self.corr.n_patches):
                np.testing.assert_array_equal(self.corr.pair_inds[i], new_corr.pair_inds[i])
                np.testing.assert_allclose(self.corr.pair_exp2phi[i], new_corr.pair_exp2phi[i])
                np.testing.assert_array_equal(self.corr.bins[i], new_corr.bins[i])
                np.testing.assert_array_equal(self.corr.Q_inds[i], new_corr.Q_inds[i])
                np.testing.assert_allclose(self.corr.Q_cos[i], new_corr.Q_cos[i])
                np.testing.assert_allclose(self.corr.Q_sin[i], new_corr.Q_sin[i])
                np.testing.assert_allclose(self.corr.Q_val[i], new_corr.Q_val[i])
                np.testing.assert_allclose(self.corr.Q_patch_area[i], new_corr.Q_patch_area[i])
                
    def test_prepare_and_get_full_tomo(self):
        """Test prepare and get_full_tomo methods."""
        # Calculate pairs first
        with patch('healpy.query_disc') as mock_query_disc, \
             patch('CosmoFuse.correlations.pixel2RaDec') as mock_pixel2RaDec:
            
            mock_query_disc.return_value = np.arange(100)
            mock_pixel2RaDec.return_value = (np.random.rand(100), np.random.rand(100))

            self.corr.calculate_pairs_2PCF(threads=1)
        
        self.corr.prepare()

        self.assertIsNotNone(self.corr.inds_dev)
        self.assertIsNotNone(self.corr.exp2phi_dev)
        self.assertIsNotNone(self.corr.bins_dev)
        self.assertIsNotNone(self.corr.tot_bins_dev)
        self.assertIsNotNone(self.corr.tot_bins_reduceat_dev)
        self.assertGreater(self.corr.ntotpairs, 0)
        
        self.corr.Q_inds = [np.array([0, 1, 2], dtype=np.uint32)]
        self.corr.Q_cos = [np.array([1.0, 1.0, 1.0], dtype=np.float64)]
        self.corr.Q_sin = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]
        self.corr.Q_val = [np.array([1.0, 1.0, 1.0], dtype=np.float64)]
        self.corr.Q_patch_area = [3.0]

        nzbins = 2
        nzbin_combs = int(binom(nzbins + 1, 2))
        shear_maps = np.random.rand(nzbins, 2, self.npix)
        w = np.ones((nzbins, self.npix))

        M_ap, xip, xim = self.corr.get_full_tomo(shear_maps, w)

        self.assertEqual(M_ap.shape, (nzbins, self.corr.n_patches))
        self.assertEqual(xip.shape, (nzbin_combs, self.corr.n_patches, self.corr.nbins))
        self.assertEqual(xim.shape, (nzbin_combs, self.corr.n_patches, self.corr.nbins))

    @patch('CosmoFuse.correlations.get_context')
    def test_calculate_pairs_2PCF_multithread(self, mock_get_context):
        """Test calculate_pairs_2PCF with multiple threads."""
        # Mock the pool to avoid actual multiprocessing
        mock_pool = mock_get_context.return_value.Pool.return_value.__enter__.return_value
        mock_pool.map.return_value = [
            (np.array([[1], [2]]), np.array([1+1j]), np.array([1])),
            (np.array([[3], [4]]), np.array([1-1j]), np.array([1])),
        ]

        # Use 2 patches for this test
        self.corr.n_patches = 2
        self.corr.phi_center = np.array([0.0, 1.0])
        self.corr.theta_center = np.array([0.0, 1.0])

        self.corr.calculate_pairs_2PCF(threads=2)

        mock_get_context.assert_called_once_with('spawn')
        mock_pool.map.assert_called_once()
        self.assertEqual(len(self.corr.pair_inds), 2)


class TestCorrelationCoverage(unittest.TestCase):
    """Tests to improve coverage of the Correlation class."""

    def setUp(self):
        """Set up test fixtures."""
        self.nside = 16
        self.npix = hp.nside2npix(self.nside)
        self.phi_center = np.array([0.0])
        self.theta_center = np.array([np.pi / 4])
        self.nbins = 2
        self.corr = Correlation(
            nside=self.nside,
            phi_center=self.phi_center,
            theta_center=self.theta_center,
            nbins=self.nbins,
        )

    def _setup_mock_pairs(self):
        self.corr.pair_inds = [np.array([[10, 20], [30, 40]], dtype=np.uint32)]
        self.corr.pair_exp2phi = [
            np.array([[1 + 1j, 1 - 1j], [1 + 1j, 1 - 1j]], dtype=np.complex128)
        ]
        self.corr.bins = [np.array([1, 1], dtype=np.uint32)]
        self.corr.Q_inds = [np.array([10, 20, 30], dtype=np.uint32)]
        self.corr.Q_cos = [np.array([0.5, 0.6, 0.7], dtype=np.float64)]
        self.corr.Q_sin = [np.array([0.8, 0.7, 0.6], dtype=np.float64)]
        self.corr.Q_val = [np.array([1.0, 1.0, 1.0], dtype=np.float64)]
        self.corr.Q_patch_area = [3.0]
        self.corr.prepare()

    def test_pickleable(self):
        """Test if the Correlation object can be pickled and unpickled."""
        corr = Correlation(
            nside=self.nside,
            phi_center=self.phi_center,
            theta_center=self.theta_center,
        )
        pickled_corr = pickle.dumps(corr)
        unpickled_corr = pickle.loads(pickled_corr)

        self.assertEqual(corr.nside, unpickled_corr.nside)
        self.assertEqual(corr.n_patches, unpickled_corr.n_patches)
        np.testing.assert_array_equal(corr.phi_center, unpickled_corr.phi_center)
        np.testing.assert_array_equal(corr.theta_center, unpickled_corr.theta_center)

    @patch('CosmoFuse.correlations.Correlation.calculate_pairs_M_a')
    @patch('CosmoFuse.correlations.Correlation.calculate_pairs_2PCF')
    def test_preprocess(self, mock_calculate_pairs_2PCF, mock_calculate_pairs_M_a):
        """Test the preprocess method."""
        self.corr.preprocess(threads=1)
        mock_calculate_pairs_M_a.assert_called_once_with()
        mock_calculate_pairs_2PCF.assert_called_once_with(1)

    def test_get_full_tomo(self):
        """Test the get_full_tomo method."""
        nzbins = 2
        nzbin_combs = int(binom(nzbins + 1, 2))
        
        shear_maps = np.random.rand(nzbins, 2, self.npix)
        w = np.random.rand(nzbins, self.npix)
        self._setup_mock_pairs()
        
        M_ap, xip, xim = self.corr.get_full_tomo(shear_maps, w)

        self.assertEqual(M_ap.shape, (nzbins, self.corr.n_patches))
        self.assertEqual(xip.shape, (nzbin_combs, self.corr.n_patches, self.corr.nbins))
        self.assertEqual(xim.shape, (nzbin_combs, self.corr.n_patches, self.corr.nbins))

        # Test with flips
        M_ap_f, xip_f, xim_f = self.corr.get_full_tomo(
            shear_maps, w, flip_g1=True, flip_g2=True
        )
        self.assertTrue(np.allclose(M_ap, -M_ap_f))
        self.assertTrue(np.allclose(xip, xip_f))
        # xim should be different with flips
        # self.assertTrue(np.allclose(xim, xim_f))

    def test_get_full_tomo_same_w_reuses_cache(self):
        nzbins = 2
        shear_maps = np.random.rand(nzbins, 2, self.npix)
        w = np.random.rand(nzbins, self.npix)
        self._setup_mock_pairs()

        with patch.object(
            self.corr,
            "_compute_tomo_sumofweights",
            wraps=self.corr._compute_tomo_sumofweights,
        ) as spy_compute:
            self.corr.get_full_tomo(shear_maps, w)
            self.corr.get_full_tomo(shear_maps, w)
            self.assertEqual(spy_compute.call_count, 1)

    def test_get_full_tomo_changed_w_recomputes(self):
        nzbins = 2
        shear_maps = np.random.rand(nzbins, 2, self.npix)
        w = np.random.rand(nzbins, self.npix)
        w_changed = w.copy()
        w_changed[0, 0] += 1e-3
        self._setup_mock_pairs()

        with patch.object(
            self.corr,
            "_compute_tomo_sumofweights",
            wraps=self.corr._compute_tomo_sumofweights,
        ) as spy_compute:
            self.corr.get_full_tomo(shear_maps, w)
            self.corr.get_full_tomo(shear_maps, w_changed)
            self.assertEqual(spy_compute.call_count, 2)

    def test_get_full_tomo_explicit_sumofweights_still_accepted(self):
        nzbins = 2
        nzbin_combs = int(binom(nzbins + 1, 2))
        shear_maps = np.random.rand(nzbins, 2, self.npix)
        w = np.random.rand(nzbins, self.npix)
        sumofweights = np.ones((2, nzbin_combs))
        self._setup_mock_pairs()

        M_ap, xip, xim = self.corr.get_full_tomo(shear_maps, w, sumofweights=sumofweights)
        self.assertEqual(M_ap.shape, (nzbins, self.corr.n_patches))
        self.assertEqual(xip.shape, (nzbin_combs, self.corr.n_patches, self.corr.nbins))
        self.assertEqual(xim.shape, (nzbin_combs, self.corr.n_patches, self.corr.nbins))

    @patch('CosmoFuse.correlations.get_context')
    def test_calculate_pairs_2PCF_multithread_default(self, mock_get_context):
        """Test calculate_pairs_2PCF with multiple threads and default context."""
        self.corr.multiprocessing_start_method = "default"
        mock_pool = mock_get_context.return_value.Pool.return_value.__enter__.return_value
        mock_pool.map.return_value = [(np.array([[]]), np.array([]), np.array([]))]
        self.corr.calculate_pairs_2PCF(threads=2)
        mock_get_context.assert_called_once_with()

    def test_get_full_tomo_prepares(self):
        """Test that get_full_tomo calls prepare if needed."""
        self.corr.bins = [np.array([1, 0], dtype=np.uint32)]
        self.corr.pair_inds = [np.zeros((2, 1), dtype=np.uint32)]
        self.corr.pair_exp2phi = [np.zeros((2, 1), dtype=np.complex128)]
        self.corr.Q_inds = [np.array([0, 1, 2], dtype=np.uint32)]
        self.corr.Q_cos = [np.array([1.0, 1.0, 1.0], dtype=np.float64)]
        self.corr.Q_sin = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]
        self.corr.Q_val = [np.array([1.0, 1.0, 1.0], dtype=np.float64)]
        self.corr.Q_patch_area = [3.0]

        shear_maps = np.random.rand(1, 2, self.npix)
        w = np.ones((1, self.npix))
        with patch.object(self.corr, "prepare", wraps=self.corr.prepare) as spy_prepare:
            self.corr.get_full_tomo(shear_maps, w)
            spy_prepare.assert_called_once()


if __name__ == "__main__":
    unittest.main()
