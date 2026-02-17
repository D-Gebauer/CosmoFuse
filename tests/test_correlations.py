"""Tests for the Correlation class."""
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import healpy as hp
import numpy as np
from scipy.special import binom

# Add src to path for testing
sys.path.insert(1, str(Path(__file__).parent.parent / "src"))

import CosmoFuse.correlations as correlations_module
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

    def test_get_pairs_patch_single_pixel_returns_empty(self):
        """Test get_pairs_patch early return when fewer than 2 pixels are provided."""
        corr = Correlation(
            nside=self.nside,
            phi_center=self.phi_center,
            theta_center=self.theta_center,
            nbins=self.nbins,
        )

        patch_inds = np.array([5], dtype=np.uint32)
        ra = np.array([0.0])
        dec = np.array([0.0])

        all_inds, exp2phi = corr.get_pairs_patch(patch_inds, ra, dec)

        self.assertEqual(len(all_inds), corr.nbins)
        for inds in all_inds:
            self.assertEqual(inds.shape, (2, 0))
        self.assertEqual(exp2phi.shape, (2, 0))

    def test_get_pairs_patch_respects_fastmath_flag(self):
        """fastmath=False should use the precise pair kernel."""
        corr = Correlation(
            nside=self.nside,
            phi_center=self.phi_center,
            theta_center=self.theta_center,
            nbins=self.nbins,
            fastmath=False,
        )
        patch_inds = np.array([0, 1], dtype=np.uint32)
        ra = np.array([0.0, 0.1], dtype=np.float64)
        dec = np.array([0.0, 0.1], dtype=np.float64)

        empty_out = (
            np.array([], dtype=np.uint32),
            np.array([], dtype=np.uint32),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

        with patch.object(correlations_module, "_compute_pairs_numba", return_value=empty_out) as fast_mock, \
             patch.object(correlations_module, "_compute_pairs_numba_precise", return_value=empty_out) as precise_mock:
            corr.get_pairs_patch(patch_inds, ra, dec)

        fast_mock.assert_not_called()
        precise_mock.assert_called_once()

    def test_compute_pairs_numba_pyfunc_bin_gap_skips_pairs(self):
        """Test py_func path where bin edges do not admit any pair assignment."""
        patch_inds = np.array([0, 1], dtype=np.uint32)
        ra = np.array([0.0, 0.3], dtype=np.float64)
        dec = np.array([0.0, 0.0], dtype=np.float64)
        binedges = np.array([0.0, np.nan, 2.0], dtype=np.float64)
        kernel_fn = getattr(
            correlations_module._compute_pairs_numba,
            "py_func",
            correlations_module._compute_pairs_numba,
        )

        (
            inds_a,
            inds_b,
            bin_indices,
            exp2phi1_real,
            exp2phi1_imag,
            exp2phi2_real,
            exp2phi2_imag,
        ) = kernel_fn(
            patch_inds,
            ra,
            dec,
            binedges,
        )

        self.assertEqual(inds_a.size, 0)
        self.assertEqual(inds_b.size, 0)
        self.assertEqual(bin_indices.size, 0)
        self.assertEqual(exp2phi1_real.size, 0)
        self.assertEqual(exp2phi1_imag.size, 0)
        self.assertEqual(exp2phi2_real.size, 0)
        self.assertEqual(exp2phi2_imag.size, 0)

    def test_compute_pairs_numba_pyfunc_clamps_upper(self):
        """Test py_func upper clamp branch for cos(theta) > 1."""
        patch_inds = np.array([0, 1], dtype=np.uint32)
        ra = np.array([0.1, 0.2], dtype=np.float64)
        dec = np.array([0.1, 0.2], dtype=np.float64)
        binedges = np.array([-1.0, 1.0, 4.0], dtype=np.float64)
        kernel_fn = getattr(
            correlations_module._compute_pairs_numba,
            "py_func",
            correlations_module._compute_pairs_numba,
        )

        original_cos = correlations_module.np.cos

        def fake_cos(x):
            if np.isscalar(x):
                return 2.0
            return original_cos(x)

        with patch.object(correlations_module.np, "cos", side_effect=fake_cos):
            outputs = kernel_fn(
                patch_inds,
                ra,
                dec,
                binedges,
            )

        self.assertEqual(len(outputs), 7)

    def test_compute_pairs_numba_pyfunc_clamps_lower(self):
        """Test py_func lower clamp branch for cos(theta) < -1."""
        patch_inds = np.array([0, 1], dtype=np.uint32)
        ra = np.array([0.1, 0.2], dtype=np.float64)
        dec = np.array([0.1, 0.2], dtype=np.float64)
        binedges = np.array([-1.0, 1.0, 4.0], dtype=np.float64)
        kernel_fn = getattr(
            correlations_module._compute_pairs_numba,
            "py_func",
            correlations_module._compute_pairs_numba,
        )

        original_cos = correlations_module.np.cos

        def fake_cos(x):
            if np.isscalar(x):
                return -2.0
            return original_cos(x)

        with patch.object(correlations_module.np, "cos", side_effect=fake_cos):
            outputs = kernel_fn(
                patch_inds,
                ra,
                dec,
                binedges,
            )

        self.assertEqual(len(outputs), 7)


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
        self.corr.calculate_pairs_2PCF()

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
            self.assertIsNotNone(new_corr.inds_dev)

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

            self.corr.calculate_pairs_2PCF()
        
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

    def test_backend_import_with_fake_cupy(self):
        """Ensure the CuPy-backed kernel setup path is exercised."""
        import importlib.util

        module_path = Path(__file__).parent.parent / "src" / "CosmoFuse" / "backend.py"
        module_name = "CosmoFuse.backend_fake_cupy"
        fake_cupy = ModuleType("cupy")

        def elementwise_kernel(*_args, **_kwargs):
            def _kernel(*_kargs, **_kkwargs):
                return None

            return _kernel

        class _FakeRuntime:
            @staticmethod
            def getDeviceCount():
                return 1

        class _FakeCuda:
            runtime = _FakeRuntime()

            class Device:
                def __init__(self, _device_id):
                    self._device_id = _device_id

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

        fake_cupy.ElementwiseKernel = elementwise_kernel
        def elementwise_kernel(*_args, **_kwargs):
            def _kernel(*_kargs, **_kkwargs):
                return None

            return _kernel

        fake_cupy.ElementwiseKernel = elementwise_kernel
        fake_cupy.asarray = np.asarray
        fake_cupy.asnumpy = np.asarray
        fake_cupy.zeros = np.zeros
        fake_cupy.ones = np.ones
        fake_cupy.sum = np.sum
        fake_cupy.mean = np.mean
        fake_cupy.conjugate = np.conjugate
        fake_cupy.add = np.add
        fake_cupy.float32 = np.float32
        fake_cupy.float64 = np.float64
        fake_cupy.complex64 = np.complex64
        fake_cupy.complex128 = np.complex128
        fake_cupy.uint32 = np.uint32
        fake_cupy.int32 = np.int32
        fake_cupy.cuda = _FakeCuda()

        sys.modules["cupy"] = fake_cupy
        try:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                self.fail("Could not load backend module for fake CuPy import")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            backend = module.get_backend("gpu")
            self.assertIsNotNone(backend.fused_cross_corr_kernel)
        finally:
            sys.modules.pop("cupy", None)
            sys.modules.pop(module_name, None)

    def test_calculate_pairs_2PCF(self):
        """Test calculate_pairs_2PCF sequential aggregation."""
        self.corr.n_patches = 2
        with patch.object(
            self.corr,
            "__get_pairs_helper__",
            side_effect=[
                (np.array([[1], [2]]), np.array([1 + 1j]), np.array([1])),
                (np.array([[3], [4]]), np.array([1 - 1j]), np.array([1])),
            ],
        ) as mock_helper:
            self.corr.calculate_pairs_2PCF()

        self.assertEqual(mock_helper.call_count, 2)
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

    def test_setstate_rebuilds_missing_fields(self):
        """Ensure __setstate__ restores optional cached attributes."""
        corr = Correlation(
            nside=self.nside,
            phi_center=self.phi_center,
            theta_center=self.theta_center,
        )
        state = corr.__getstate__()
        for key in [
            "M_A_all_patches",
            "_prepare_version",
            "_tomo_sumofweights_cache",
            "_tomo_sumofweights_cache_w_fingerprint",
            "_tomo_sumofweights_cache_prepare_version",
            "_xipm_sumofweights_cache",
            "_xipm_sumofweights_cache_w_fingerprint",
            "_xipm_sumofweights_cache_prepare_version",
            "Q_inds_flat",
            "Q_cos_flat",
            "Q_sin_flat",
            "Q_val_flat",
            "Q_offsets",
            "Q_patch_area_flat",
        ]:
            state.pop(key, None)

        restored = Correlation.__new__(Correlation)
        restored.__setstate__(state)

        self.assertIsNotNone(restored.M_A_all_patches)
        self.assertEqual(restored._prepare_version, 0)
        self.assertIsNone(restored._tomo_sumofweights_cache)
        self.assertIsNone(restored._xipm_sumofweights_cache)
        self.assertIsNone(restored.Q_inds_flat)
        self.assertIsNone(restored.Q_offsets)

    def test_prepare_can_release_pair_inds(self):
        corr = Correlation(
            nside=1,
            phi_center=np.array([0.0]),
            theta_center=np.array([0.0]),
            nbins=1,
            theta_min=1.0,
            theta_max=2.0,
            patch_size=1.0,
            theta_Q=1.0,
            device="cpu",
        )
        corr.pair_inds = [np.array([[0, 1], [1, 2]], dtype=np.uint32)]
        corr.pair_exp2phi = [np.ones((2, 2), dtype=np.complex128)]
        corr.bins = [np.array([2], dtype=np.uint32)]

        corr.prepare(release_host_pairs=True)

        self.assertIsNone(corr.pair_inds)
        self.assertIsNone(corr.pair_exp2phi)
        self.assertIsNone(corr.bins)
        self.assertIsNotNone(corr.inds_dev)
        self.assertEqual(corr.ntotpairs, 2)

    def test_prepare_raises_when_host_pairs_released_and_not_prepared(self):
        corr = Correlation(
            nside=1,
            phi_center=np.array([0.0]),
            theta_center=np.array([0.0]),
            nbins=1,
            theta_min=1.0,
            theta_max=2.0,
            patch_size=1.0,
            theta_Q=1.0,
            device="cpu",
        )
        corr.pair_inds = None

        with self.assertRaisesRegex(RuntimeError, "Host pair arrays were released"):
            corr.prepare()

    def test_prepare_returns_when_host_pairs_released_but_device_ready(self):
        corr = Correlation(
            nside=1,
            phi_center=np.array([0.0]),
            theta_center=np.array([0.0]),
            nbins=1,
            theta_min=1.0,
            theta_max=2.0,
            patch_size=1.0,
            theta_Q=1.0,
            device="cpu",
        )
        corr.pair_inds = None
        corr.pair_exp2phi = None
        corr.bins = None

        corr.inds_dev = np.zeros((2, 1), dtype=np.uint32)
        corr.exp2phi_dev = np.zeros((2, 1), dtype=np.complex128)
        corr.bins_dev = np.zeros(1, dtype=np.uint32)
        corr.tot_bins_reduceat_dev = np.zeros(1, dtype=np.int64)

        prepare_version_before = corr._prepare_version
        corr.prepare()
        self.assertEqual(corr._prepare_version, prepare_version_before)

    def test_save_pairs_warns_when_host_pair_arrays_released(self):
        corr = Correlation(
            nside=1,
            phi_center=np.array([0.0]),
            theta_center=np.array([0.0]),
            nbins=1,
            theta_min=1.0,
            theta_max=2.0,
            patch_size=1.0,
            theta_Q=1.0,
            device="cpu",
        )
        corr.pair_inds = None
        corr.pair_exp2phi = None
        corr.bins = None

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = Path(tmpdir) / "pairs.h5"
            with self.assertWarnsRegex(RuntimeWarning, "Cannot save pairs"):
                corr.save_pairs(str(outpath))
            self.assertFalse(outpath.exists())

    def test_get_full_tomo_fused_cpu_assignments(self):
        """Cover fused-kernel assignment lines for CPU backend."""
        corr = Correlation(
            nside=1,
            phi_center=np.array([0.0]),
            theta_center=np.array([0.0]),
            nbins=1,
            theta_min=1.0,
            theta_max=2.0,
            patch_size=1.0,
            theta_Q=1.0,
            device="cpu",
        )

        corr.inds_dev = np.array([[0, 1], [1, 2]], dtype=np.uint32)
        corr.exp2phi_dev = np.ones((2, 2), dtype=np.complex128)
        corr.bins_dev = np.array([2], dtype=np.uint32)
        corr.tot_bins_reduceat_dev = np.array([0, 2], dtype=np.int64)
        corr.ntotpairs = 2

        corr.Q_inds = [np.array([0], dtype=np.uint32)]
        corr.Q_cos = [np.array([1.0], dtype=np.float64)]
        corr.Q_sin = [np.array([0.0], dtype=np.float64)]
        corr.Q_val = [np.array([1.0], dtype=np.float64)]
        corr.Q_patch_area = [1.0]

        shear_maps = np.zeros((2, 2, 3), dtype=np.float64)
        shear_maps[0, 0] = np.array([1.0, 2.0, 3.0])
        shear_maps[1, 0] = np.array([4.0, 5.0, 6.0])
        w = np.ones((2, 3), dtype=np.float64)
        sumofweights = np.full((2, 3, 1), 2.0, dtype=np.float64)

        called = {"value": False}

        def fused_kernel(
            g1a,
            g2a,
            g1b,
            g2b,
            wa,
            wb,
            ind_i,
            ind_j,
            exp_i,
            exp_j,
            out_ab_p,
            out_ab_m,
            out_ba_p,
            out_ba_m,
        ):
            called["value"] = True
            for idx in range(ind_i.shape[0]):
                i = ind_i[idx]
                j = ind_j[idx]
                ga_i = g1a[i] + 1j * g2a[i]
                gb_i = g1b[i] + 1j * g2b[i]
                ga_j = g1a[j] + 1j * g2a[j]
                gb_j = g1b[j] + 1j * g2b[j]

                ga_i_rot = wa[i] * ga_i * exp_i[idx]
                gb_i_rot = wb[i] * gb_i * exp_i[idx]
                ga_j_rot = wa[j] * ga_j * exp_j[idx]
                gb_j_rot = wb[j] * gb_j * exp_j[idx]

                out_ab_p[idx] = gb_j_rot * np.conjugate(ga_i_rot)
                out_ab_m[idx] = gb_j_rot * ga_i_rot
                out_ba_p[idx] = ga_j_rot * np.conjugate(gb_i_rot)
                out_ba_m[idx] = ga_j_rot * gb_i_rot

        corr.backend.fused_cross_corr_kernel = fused_kernel

        _, xip, xim = corr.get_full_tomo(shear_maps, w, sumofweights=sumofweights)

        self.assertTrue(called["value"])

        pairs = [(0, 1), (1, 2)]
        g_a = shear_maps[0, 0]
        g_b = shear_maps[1, 0]
        ab_vals = [g_b[j] * g_a[i] for i, j in pairs]
        ba_vals = [g_a[j] * g_b[i] for i, j in pairs]
        expected = ((np.sum(ab_vals) / 2.0) + (np.sum(ba_vals) / 2.0)) / 2.0

        self.assertAlmostEqual(xip[1, 0, 0], expected)
        self.assertAlmostEqual(xim[1, 0, 0], expected)

    @patch('CosmoFuse.correlations.Correlation.calculate_pairs_M_a')
    @patch('CosmoFuse.correlations.Correlation.calculate_pairs_2PCF')
    @patch('CosmoFuse.correlations.Correlation.prepare')
    def test_preprocess(
        self, mock_prepare, mock_calculate_pairs_2PCF, mock_calculate_pairs_M_a
    ):
        """Test the preprocess method."""
        self.corr.preprocess()
        mock_calculate_pairs_M_a.assert_called_once_with()
        mock_calculate_pairs_2PCF.assert_called_once_with()
        mock_prepare.assert_called_once_with()

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

    def test_prepare_aperture_flat_empty(self):
        """Ensure empty aperture inputs clear flat buffers."""
        self.corr.Q_inds = []
        self.corr.Q_cos = []
        self.corr.Q_sin = []
        self.corr.Q_val = []
        self.corr.Q_patch_area = []

        self.corr._prepare_aperture_flat()

        self.assertIsNone(self.corr.Q_inds_flat)
        self.assertIsNone(self.corr.Q_offsets)

    def test_get_full_tomo_missing_fused_kernel(self):
        """get_full_tomo should raise if fused kernel is missing."""
        corr = Correlation(
            nside=1,
            phi_center=np.array([0.0]),
            theta_center=np.array([0.0]),
            nbins=1,
            theta_min=1.0,
            theta_max=2.0,
            patch_size=1.0,
            theta_Q=1.0,
            device="cpu",
        )

        corr.pair_inds = [np.array([[0, 1], [1, 2]], dtype=np.uint32)]
        corr.pair_exp2phi = [np.ones((2, 2), dtype=np.complex128)]
        corr.bins = [np.array([2], dtype=np.uint32)]
        corr.prepare()

        corr.Q_inds = [np.array([0], dtype=np.uint32)]
        corr.Q_cos = [np.array([1.0], dtype=np.float64)]
        corr.Q_sin = [np.array([0.0], dtype=np.float64)]
        corr.Q_val = [np.array([1.0], dtype=np.float64)]
        corr.Q_patch_area = [1.0]

        corr.backend.fused_cross_corr_kernel = None

        shear_maps = np.ones((2, 2, 12), dtype=np.float64)
        w = np.ones((2, 12), dtype=np.float64)

        with self.assertRaises(RuntimeError):
            corr.get_full_tomo(shear_maps, w)

    def test_xipm_missing_kernel_raises(self):
        """xipm should raise if kernel is missing."""
        corr = Correlation(
            nside=1,
            phi_center=np.array([0.0]),
            theta_center=np.array([0.0]),
            nbins=1,
            theta_min=1.0,
            theta_max=2.0,
            patch_size=1.0,
            theta_Q=1.0,
            device="cpu",
        )

        corr.pair_inds = [np.array([[0, 1], [1, 2]], dtype=np.uint32)]
        corr.pair_exp2phi = [np.ones((2, 2), dtype=np.complex128)]
        corr.bins = [np.array([2], dtype=np.uint32)]
        corr.prepare()

        corr.backend.xipm_kernel = None

        g11 = np.ones(12, dtype=np.float64)
        g21 = np.ones(12, dtype=np.float64)
        g12 = g11
        g22 = g21
        w1 = np.ones(12, dtype=np.float64)
        w2 = w1

        with self.assertRaises(RuntimeError):
            corr.xipm(g11, g21, g12, g22, w1, w2)

    def test_rotation_precision_affects_kernel_dtype(self):
        """Rotation precision should drive complex kernel dtype choices."""
        corr = Correlation(
            nside=1,
            phi_center=np.array([0.0]),
            theta_center=np.array([0.0]),
            nbins=1,
            theta_min=1.0,
            theta_max=2.0,
            patch_size=1.0,
            theta_Q=1.0,
            device="cpu",
            rotation_precision="float32",
        )

        corr.pair_inds = [np.array([[0, 1], [1, 2]], dtype=np.uint32)]
        corr.pair_exp2phi = [
            np.ones((2, 2), dtype=corr.rotation_complex_dtype)
        ]
        corr.bins = [np.array([2], dtype=np.uint32)]
        corr.prepare()

        g11 = np.ones(12, dtype=np.float64)
        g21 = np.ones(12, dtype=np.float64)
        g12 = np.ones(12, dtype=np.float64)
        g22 = np.ones(12, dtype=np.float64)
        w1 = np.ones(12, dtype=np.float64)
        w2 = np.ones(12, dtype=np.float64)

        xip, xim = corr.xipm(g11, g21, g12, g22, w1, w2)

        self.assertEqual(corr.rotation_complex_dtype, np.dtype(np.complex64))
        self.assertEqual(xip.dtype, np.float32)
        self.assertEqual(xim.dtype, np.float32)

    def test_load_pairs_downcasts_to_instance_precisions(self):
        """Loading high-precision pairs into low-precision instance should cast dtypes."""
        high = Correlation(
            nside=1,
            phi_center=np.array([0.0]),
            theta_center=np.array([0.0]),
            nbins=1,
            theta_min=1.0,
            theta_max=2.0,
            patch_size=1.0,
            theta_Q=1.0,
            device="cpu",
            map_precision="float64",
            rotation_precision="float64",
            index_precision="uint64",
        )
        high.pair_inds = [np.array([[0, 1], [1, 2]], dtype=np.uint64)]
        high.pair_exp2phi = [np.ones((2, 2), dtype=np.complex128)]
        high.bins = [np.array([2], dtype=np.uint64)]
        high.Q_inds = [np.array([0, 1], dtype=np.uint64)]
        high.Q_cos = [np.array([1.0, 1.0], dtype=np.float64)]
        high.Q_sin = [np.array([0.0, 0.0], dtype=np.float64)]
        high.Q_val = [np.array([1.0, 1.0], dtype=np.float64)]
        high.Q_patch_area = [np.float64(2.0)]

        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            high.save_pairs(tmp.name)

            low = Correlation(
                nside=1,
                phi_center=np.array([0.0]),
                theta_center=np.array([0.0]),
                nbins=1,
                theta_min=1.0,
                theta_max=2.0,
                patch_size=1.0,
                theta_Q=1.0,
                device="cpu",
                map_precision="float32",
                rotation_precision="float32",
                index_precision="uint32",
            )
            low.load_pairs(tmp.name)

            self.assertEqual(low.pair_inds[0].dtype, np.uint32)
            self.assertEqual(low.bins[0].dtype, np.uint32)
            self.assertEqual(low.pair_exp2phi[0].dtype, np.complex64)
            self.assertEqual(low.Q_inds[0].dtype, np.uint32)
            self.assertEqual(low.Q_cos[0].dtype, np.float32)
            self.assertEqual(low.Q_sin[0].dtype, np.float32)
            self.assertEqual(low.Q_val[0].dtype, np.float32)
            self.assertEqual(np.asarray(low.Q_patch_area).dtype, np.float32)
            self.assertEqual(low.exp2phi_dev.dtype, np.complex64)

            g11 = np.ones(12, dtype=np.float64)
            g21 = np.ones(12, dtype=np.float64)
            g12 = np.ones(12, dtype=np.float64)
            g22 = np.ones(12, dtype=np.float64)
            w1 = np.ones(12, dtype=np.float64)
            w2 = np.ones(12, dtype=np.float64)
            xip, xim = low.xipm(g11, g21, g12, g22, w1, w2)
            self.assertEqual(xip.dtype, np.float32)
            self.assertEqual(xim.dtype, np.float32)

    def test_prepare_aperture_flat_populates(self):
        """Ensure aperture inputs are flattened with correct offsets."""
        self.corr.Q_inds = [
            np.array([1, 2], dtype=np.uint32),
            np.array([3, 4, 5], dtype=np.uint32),
        ]
        self.corr.Q_cos = [
            np.array([0.1, 0.2], dtype=np.float64),
            np.array([0.3, 0.4, 0.5], dtype=np.float64),
        ]
        self.corr.Q_sin = [
            np.array([0.2, 0.1], dtype=np.float64),
            np.array([0.6, 0.7, 0.8], dtype=np.float64),
        ]
        self.corr.Q_val = [
            np.array([1.0, 1.0], dtype=np.float64),
            np.array([2.0, 2.0, 2.0], dtype=np.float64),
        ]
        self.corr.Q_patch_area = [2.0, 3.0]

        self.corr._prepare_aperture_flat()

        np.testing.assert_array_equal(self.corr.Q_offsets, np.array([0, 2, 5]))
        np.testing.assert_array_equal(
            self.corr.Q_inds_flat, np.array([1, 2, 3, 4, 5], dtype=np.uint32)
        )

    def test_xipm_auto_sumofweights_matches_explicit(self):
        self._setup_mock_pairs()
        g11 = np.random.rand(self.npix)
        g21 = np.random.rand(self.npix)
        g12 = g11
        g22 = g21
        w1 = np.random.rand(self.npix)
        w2 = w1

        w1_dev = self.corr.backend.to_device(w1)
        w2_dev = self.corr.backend.to_device(w2)
        sumofweights = self.corr._compute_xipm_sumofweights(
            w1_dev,
            w2_dev,
        )
        xip_auto, xim_auto = self.corr.xipm(g11, g21, g12, g22, w1, w2)
        xip_explicit, xim_explicit = self.corr.xipm(
            g11, g21, g12, g22, w1, w2, sumofweights=sumofweights
        )

        np.testing.assert_allclose(
            self.corr.backend.to_numpy(xip_auto),
            self.corr.backend.to_numpy(xip_explicit),
            rtol=1e-12,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            self.corr.backend.to_numpy(xim_auto),
            self.corr.backend.to_numpy(xim_explicit),
            rtol=1e-12,
            atol=1e-14,
        )

    def test_xipm_auto_sumofweights_reuses_cache(self):
        self._setup_mock_pairs()
        g11 = np.random.rand(self.npix)
        g21 = np.random.rand(self.npix)
        g12 = g11
        g22 = g21
        w1 = np.random.rand(self.npix)
        w2 = w1

        with patch.object(
            self.corr,
            "_compute_xipm_sumofweights",
            wraps=self.corr._compute_xipm_sumofweights,
        ) as spy_compute:
            self.corr.xipm(g11, g21, g12, g22, w1, w2)
            self.corr.xipm(g11, g21, g12, g22, w1, w2)
            self.assertEqual(spy_compute.call_count, 1)

        def test_xipm_gpu_fallback_path_with_fake_cupy(self):
            """Cover the non-CPU xipm path with a fake CuPy backend."""
            import importlib.util
            import sys
            from types import ModuleType
            from pathlib import Path

            module_path = Path(__file__).parent.parent / "src" / "CosmoFuse" / "backend.py"
            module_name = "CosmoFuse.backend_fake_cupy_xipm"
            fake_cupy = ModuleType("cupy")

            class _FakeRuntime:
                @staticmethod
                def getDeviceCount():
                    return 1

            class _FakeCuda:
                runtime = _FakeRuntime()

                class Device:
                    def __init__(self, _device_id):
                        self._device_id = _device_id

                    def __enter__(self):
                        return self

                    def __exit__(self, exc_type, exc, tb):
                        return False

            fake_cupy.asarray = np.asarray
            fake_cupy.asnumpy = np.asarray
            fake_cupy.zeros = np.zeros
            fake_cupy.ones = np.ones
            fake_cupy.sum = np.sum
            fake_cupy.mean = np.mean
            fake_cupy.conjugate = np.conjugate
            fake_cupy.add = np.add
            fake_cupy.float32 = np.float32
            fake_cupy.float64 = np.float64
            fake_cupy.complex64 = np.complex64
            fake_cupy.complex128 = np.complex128
            fake_cupy.uint32 = np.uint32
            fake_cupy.int32 = np.int32
            fake_cupy.cuda = _FakeCuda()

            sys.modules["cupy"] = fake_cupy
            try:
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is None or spec.loader is None:
                    self.fail("Could not load backend module for fake CuPy import")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                corr = Correlation(
                    nside=self.nside,
                    phi_center=self.phi_center,
                    theta_center=self.theta_center,
                    nbins=self.nbins,
                    device="gpu",
                )
                corr.backend = module.get_backend("gpu")

                n_pairs = 4
                corr.pair_inds = [np.zeros((2, n_pairs), dtype=np.uint32)]
                corr.pair_exp2phi = [np.ones((2, n_pairs), dtype=np.complex128)]
                corr.bins = [np.array([2, 2], dtype=np.uint32)]
                corr.prepare()

                npix = 12 * self.nside**2
                g11 = np.ones(npix, dtype=np.float64)
                g21 = np.ones(npix, dtype=np.float64)
                g12 = np.ones(npix, dtype=np.float64)
                g22 = np.ones(npix, dtype=np.float64)
                w1 = np.ones(npix, dtype=np.float64)
                w2 = np.ones(npix, dtype=np.float64)

                xip, xim = corr.xipm(g11, g21, g12, g22, w1, w2)

                self.assertEqual(xip.shape, (self.n_patches, self.nbins))
                self.assertEqual(xim.shape, (self.n_patches, self.nbins))
            finally:
                sys.modules.pop("cupy", None)
                sys.modules.pop(module_name, None)
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
