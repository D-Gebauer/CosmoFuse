"""Tests for the Correlation class."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
from scipy.special import binom

# Add src to path for testing
sys.path.insert(1, str(Path(__file__).parent.parent / "src"))

from CosmoFuse.correlations import Correlation


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


    def test_get_aperture_shear(self):
        """Test the get_aperture_shear method."""
        g1 = np.random.rand(self.npix)
        g2 = np.random.rand(self.npix)
        w = np.ones(self.npix)

        M_a = self.corr.get_aperture_shear(g1, g2, w)

        self.assertEqual(M_a.shape, (1,))
        
        Q_inds = self.corr.Q_inds[0]
        Q_cos = self.corr.Q_cos[0]
        Q_sin = self.corr.Q_sin[0]
        Q_val = self.corr.Q_val[0]
        Q_patch_area = self.corr.Q_patch_area[0]
        
        gt = -g1[Q_inds] * Q_cos - g2[Q_inds] * Q_sin
        expected_M_a = Q_patch_area * np.sum(w[Q_inds] * gt * Q_val) / np.sum(w[Q_inds])

        # to_numpy: on a GPU backend the public getter returns a device array
        self.assertAlmostEqual(
            float(self.corr.backend.to_numpy(M_a)[0]), expected_M_a
        )

    def test_get_aperture_shear_non_numpy_backend_path(self):
        g1 = np.random.rand(self.npix)
        g2 = np.random.rand(self.npix)
        w = np.ones(self.npix)

        class FakeBackend:
            name = "cupy"
            module = np
            add = np.add

            @staticmethod
            def to_device(array):
                return np.asarray(array)

            @staticmethod
            def to_numpy(array):
                return np.asarray(array)

            @staticmethod
            def zeros(shape, dtype):
                return np.zeros(shape, dtype=dtype)

            @staticmethod
            def aperture_shear_kernel(
                Q_inds,
                Q_cos,
                Q_sin,
                Q_val,
                g1_vals,
                g2_vals,
                weights,
                out_num,
                out_den,
            ):
                gt = -g1_vals[Q_inds] * Q_cos - g2_vals[Q_inds] * Q_sin
                out_num[:] = weights[Q_inds] * gt * Q_val
                out_den[:] = weights[Q_inds]

        self.corr.backend = FakeBackend()
        M_a = self.corr.get_aperture_shear(g1, g2, w)

        Q_inds = self.corr.Q_inds[0]
        Q_cos = self.corr.Q_cos[0]
        Q_sin = self.corr.Q_sin[0]
        Q_val = self.corr.Q_val[0]
        Q_patch_area = self.corr.Q_patch_area[0]
        gt = -g1[Q_inds] * Q_cos - g2[Q_inds] * Q_sin
        expected_M_a = Q_patch_area * np.sum(w[Q_inds] * gt * Q_val) / np.sum(w[Q_inds])
        self.assertAlmostEqual(M_a[0], expected_M_a)

    def test_get_aperture_shear_missing_backend_kernel_raises(self):
        class IncompleteBackend:
            name = "numpy"

        self.corr.backend = IncompleteBackend()
        with self.assertRaisesRegex(RuntimeError, "aperture-shear kernel"):
            self.corr.get_aperture_shear(
                np.random.rand(self.npix),
                np.random.rand(self.npix),
                np.ones(self.npix),
            )

    def test_get_aperture_density(self):
        """Test the get_aperture_density method for spin-0 fields."""
        delta = np.random.rand(self.npix)
        w = np.ones(self.npix)

        aperture_density = self.corr.get_aperture_density(delta, w)

        self.assertEqual(aperture_density.shape, (1,))

        Q_inds = self.corr.Q_inds[0]
        Q_val = self.corr.Q_val[0]
        Q_patch_area = self.corr.Q_patch_area[0]
        expected = Q_patch_area * np.sum(w[Q_inds] * delta[Q_inds] * Q_val) / np.sum(
            w[Q_inds]
        )

        # to_numpy: on a GPU backend the public getter returns a device array
        self.assertAlmostEqual(
            float(self.corr.backend.to_numpy(aperture_density)[0]), expected
        )

    def test_get_aperture_density_non_numpy_backend_path(self):
        """Covers non-numpy backend execution path for scalar aperture."""
        delta = np.random.rand(self.npix)
        w = np.ones(self.npix)

        class FakeBackend:
            name = "cupy"
            module = np
            add = np.add

            @staticmethod
            def to_device(array):
                return np.asarray(array)

            @staticmethod
            def to_numpy(array):
                return np.asarray(array)

            @staticmethod
            def zeros(shape, dtype):
                return np.zeros(shape, dtype=dtype)

            @staticmethod
            def aperture_density_kernel(Q_inds, Q_val, map_values, weights, out_num, out_den):
                out_num[:] = weights[Q_inds] * map_values[Q_inds] * Q_val
                out_den[:] = weights[Q_inds]

        self.corr.backend = FakeBackend()

        aperture_density = self.corr.get_aperture_density(delta, w)
        Q_inds = self.corr.Q_inds[0]
        Q_val = self.corr.Q_val[0]
        Q_patch_area = self.corr.Q_patch_area[0]
        expected = Q_patch_area * np.sum(w[Q_inds] * delta[Q_inds] * Q_val) / np.sum(
            w[Q_inds]
        )
        self.assertAlmostEqual(aperture_density[0], expected)

    def test_get_aperture_density_missing_backend_kernel_raises(self):
        """Missing scalar aperture kernel should raise a clear runtime error."""

        class IncompleteBackend:
            name = "numpy"

        self.corr.backend = IncompleteBackend()
        with self.assertRaisesRegex(RuntimeError, "aperture-density kernel"):
            self.corr.get_aperture_density(np.random.rand(self.npix), np.ones(self.npix))

    def test_get_aperture_density_numpy_backend_path(self):
        """Explicitly cover numpy backend path for scalar aperture."""

        class NumpyBackend:
            name = "numpy"

            @staticmethod
            def aperture_density_kernel(
                Q_inds,
                Q_val,
                Q_offsets,
                map_values,
                weights,
                Q_patch_area,
                out_aperture,
            ):
                for patch_idx in range(Q_offsets.shape[0] - 1):
                    start = Q_offsets[patch_idx]
                    stop = Q_offsets[patch_idx + 1]
                    patch_inds = Q_inds[start:stop]
                    patch_q = Q_val[start:stop]
                    out_aperture[patch_idx] = (
                        Q_patch_area[patch_idx]
                        * np.sum(weights[patch_inds] * map_values[patch_inds] * patch_q)
                        / np.sum(weights[patch_inds])
                    )

        self.corr.backend = NumpyBackend()
        delta = np.random.rand(self.npix)
        w = np.ones(self.npix)
        result = self.corr.get_aperture_density(delta, w)
        self.assertEqual(result.shape, (1,))

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

    def test_get_pairs_patch_M_a_custom_filter(self):
        pixels_RA_Q_patch = np.array([0.1, 0.2])
        pixels_dec_Q_patch = np.array([0.1, 0.2])

        def custom_filter(theta, _theta_Q):
            return np.full_like(theta, 2.5)

        _cos_2phi, _sin_2phi, Q = self.corr.get_pairs_patch_M_a(
            pixels_RA_Q_patch,
            pixels_dec_Q_patch,
            0.0,
            0.0,
            aperture_filter=custom_filter,
        )
        np.testing.assert_allclose(Q, 2.5)

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
                
    def test_prepare_and_get_full_tomo_shear(self):
        """Test prepare and get_full_tomo_shear methods."""
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

        M_ap, xip, xim = self.corr.get_full_tomo_shear(shear_maps, w)

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
            self.assertIsNotNone(backend.xipm_cross_corr_kernel)
            self.assertIsNotNone(backend.aperture_density_kernel)
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
