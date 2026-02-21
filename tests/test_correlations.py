"""Tests for the Correlation class."""
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import healpy as hp
import numpy as np
from scipy.special import binom

# Add src to path for testing
sys.path.insert(1, str(Path(__file__).parent.parent / "src"))

import CosmoFuse.correlations as correlations_module
from CosmoFuse.correlation_helpers import (
    zeta_a_g as helper_zeta_a_g,
    zeta_a_minus as helper_zeta_a_minus,
    zeta_a_plus as helper_zeta_a_plus,
    zeta_a_t as helper_zeta_a_t,
    zeta_g_g as helper_zeta_g_g,
    zeta_g_minus as helper_zeta_g_minus,
    zeta_g_plus as helper_zeta_g_plus,
    zeta_g_t as helper_zeta_g_t,
)
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
        with patch.object(
            correlations_module,
            "_get_pairs_numba_kernel",
            wraps=correlations_module._get_pairs_numba_kernel,
        ) as spy_getter:
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

        with patch.object(corr, "_compute_pairs_kernel", return_value=empty_out) as kernel_mock:
            corr.get_pairs_patch(patch_inds, ra, dec)

        kernel_mock.assert_called_once()
        spy_getter.assert_called_with(False)

    def test_get_pairs_patch_invalid_angle_method_raises(self):
        corr = Correlation(
            nside=self.nside,
            phi_center=self.phi_center,
            theta_center=self.theta_center,
            nbins=self.nbins,
        )
        patch_inds = np.array([0, 1], dtype=np.uint32)
        ra = np.array([0.0, 0.1], dtype=np.float64)
        dec = np.array([0.0, 0.1], dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "angle_method must be one of"):
            corr.get_pairs_patch(patch_inds, ra, dec, angle_method="invalid")

    def test_get_pairs_numba_kernel_cache_hit(self):
        correlations_module._compute_pairs_kernel_cache.clear()
        k1 = correlations_module._get_pairs_numba_kernel(True)
        k2 = correlations_module._get_pairs_numba_kernel(True)
        self.assertIs(k1, k2)

    def test_compute_pairs_numba_pyfunc_bin_gap_skips_pairs(self):
        """Test py_func path where bin edges do not admit any pair assignment."""
        patch_inds = np.array([0, 1], dtype=np.uint32)
        ra = np.array([0.0, 0.3], dtype=np.float64)
        dec = np.array([0.0, 0.0], dtype=np.float64)
        binedges = np.array([0.0, np.nan, 2.0], dtype=np.float64)
        kernel_fn = correlations_module._compute_pairs_impl

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
            2,
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
        kernel_fn = correlations_module._compute_pairs_impl

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
                0,
            )

        self.assertEqual(len(outputs), 7)

    def test_compute_pairs_numba_pyfunc_clamps_lower(self):
        """Test py_func lower clamp branch for cos(theta) < -1."""
        patch_inds = np.array([0, 1], dtype=np.uint32)
        ra = np.array([0.1, 0.2], dtype=np.float64)
        dec = np.array([0.1, 0.2], dtype=np.float64)
        binedges = np.array([-1.0, 1.0, 4.0], dtype=np.float64)
        kernel_fn = correlations_module._compute_pairs_impl

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
                0,
            )

        self.assertEqual(len(outputs), 7)

    def test_compute_pairs_numba_pyfunc_r2_c1_zero_fallback(self):
        """Trigger fallback branch where R2_C1 == 0 and default exp(2i phi1) is used."""
        patch_inds = np.array([0, 1], dtype=np.uint32)
        ra = np.array([0.0, 0.0], dtype=np.float64)
        dec = np.array([np.pi / 2, 0.0], dtype=np.float64)
        binedges = np.array([0.1, 1.0, 2.0], dtype=np.float64)
        kernel_fn = correlations_module._compute_pairs_impl
        original_cos = correlations_module.np.cos

        def fake_cos(x):
            if isinstance(x, np.ndarray) and x.shape == dec.shape and np.allclose(x, dec):
                out = original_cos(x).copy()
                out[np.isclose(x, np.pi / 2)] = 0.0
                return out
            if np.isscalar(x) and np.isclose(x, np.pi / 2):
                return 0.0
            return original_cos(x)

        with patch.object(correlations_module.np, "cos", side_effect=fake_cos):
            (
                inds_a,
                inds_b,
                _bin_indices,
                exp2phi1_real,
                exp2phi1_imag,
                _exp2phi2_real,
                _exp2phi2_imag,
            ) = kernel_fn(
                patch_inds,
                ra,
                dec,
                binedges,
                2,
            )

        self.assertEqual(inds_a.size, 1)
        self.assertEqual(inds_b.size, 1)
        self.assertEqual(exp2phi1_real[0], -1.0)
        self.assertEqual(exp2phi1_imag[0], 0.0)

    def test_compute_pairs_numba_pyfunc_r2_c2_zero_fallback(self):
        """Trigger fallback branch where R2_C2 == 0 and default exp(2i phi2) is used."""
        patch_inds = np.array([0, 1], dtype=np.uint32)
        ra = np.array([0.0, 0.0], dtype=np.float64)
        dec = np.array([0.0, np.pi / 2], dtype=np.float64)
        binedges = np.array([0.1, 1.0, 2.0], dtype=np.float64)
        kernel_fn = correlations_module._compute_pairs_impl
        original_cos = correlations_module.np.cos

        def fake_cos(x):
            if isinstance(x, np.ndarray) and x.shape == dec.shape and np.allclose(x, dec):
                out = original_cos(x).copy()
                out[np.isclose(x, np.pi / 2)] = 0.0
                return out
            if np.isscalar(x) and np.isclose(x, np.pi / 2):
                return 0.0
            return original_cos(x)

        with patch.object(correlations_module.np, "cos", side_effect=fake_cos):
            (
                inds_a,
                inds_b,
                _bin_indices,
                _exp2phi1_real,
                _exp2phi1_imag,
                exp2phi2_real,
                exp2phi2_imag,
            ) = kernel_fn(
                patch_inds,
                ra,
                dec,
                binedges,
                2,
            )

        self.assertEqual(inds_a.size, 1)
        self.assertEqual(inds_b.size, 1)
        self.assertEqual(exp2phi2_real[0], -1.0)
        self.assertEqual(exp2phi2_imag[0], 0.0)

    def test_compute_pairs_numba_pyfunc_out_of_range_continues_for_arccos_and_law(self):
        """Cover out-of-range continue branches for arccos and law angle methods."""
        patch_inds = np.array([0, 1], dtype=np.uint32)
        ra = np.array([0.0, 0.3], dtype=np.float64)
        dec = np.array([0.0, 0.0], dtype=np.float64)
        binedges = np.array([0.4, 0.5], dtype=np.float64)
        kernel_fn = correlations_module._compute_pairs_impl

        outputs_arccos = kernel_fn(patch_inds, ra, dec, binedges, 0)
        outputs_law = kernel_fn(patch_inds, ra, dec, binedges, 2)

        self.assertEqual(outputs_arccos[0].size, 0)
        self.assertEqual(outputs_law[0].size, 0)

    def test_to_backend_array_bypasses_transfer_for_backend_native_arrays(self):
        corr = Correlation(
            nside=self.nside,
            phi_center=self.phi_center,
            theta_center=self.theta_center,
            nbins=self.nbins,
        )

        class FakeGPUArray:
            def __init__(self, values, *, device_id=0):
                if isinstance(values, FakeGPUArray):
                    values = values._values
                self._values = np.asarray(values)
                self.shape = self._values.shape
                self.dtype = self._values.dtype
                self.device = SimpleNamespace(id=device_id)
                self.data = SimpleNamespace(ptr=id(self._values))

            def astype(self, dtype, copy=False):
                return FakeGPUArray(self._values.astype(dtype, copy=copy), device_id=self.device.id)

            def __array__(self, *args, **kwargs):
                raise AssertionError("backend-native arrays should not be coerced to numpy")

        calls = {"to_device": 0}

        def fake_to_device(values):
            calls["to_device"] += 1
            return FakeGPUArray(values)

        corr.backend = SimpleNamespace(
            name="cupy",
            module=SimpleNamespace(ndarray=FakeGPUArray),
            device_id=0,
            to_device=fake_to_device,
            to_numpy=lambda values: values._values if isinstance(values, FakeGPUArray) else np.asarray(values),
        )

        native = FakeGPUArray([1.0, 2.0, 3.0], device_id=0)
        _ = corr._to_backend_array(native, dtype=np.float64)
        self.assertEqual(calls["to_device"], 0)

        host = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        _ = corr._to_backend_array(host, dtype=np.float64)
        self.assertEqual(calls["to_device"], 1)

    def test_vectorized_shear_shear_accepts_backend_native_arrays_without_transfer(self):
        corr = Correlation(
            nside=self.nside,
            phi_center=self.phi_center,
            theta_center=self.theta_center,
            nbins=self.nbins,
        )
        corr.inds_dev = np.array([0], dtype=np.int64)

        class FakeGPUArray:
            def __init__(self, values, *, device_id=0):
                if isinstance(values, FakeGPUArray):
                    values = values._values
                self._values = np.asarray(values)
                self.shape = self._values.shape
                self.dtype = self._values.dtype
                self.ndim = self._values.ndim
                self.device = SimpleNamespace(id=device_id)
                self.data = SimpleNamespace(ptr=id(self._values))

            def astype(self, dtype, copy=False):
                return FakeGPUArray(self._values.astype(dtype, copy=copy), device_id=self.device.id)

            def __array__(self, *args, **kwargs):
                raise AssertionError("backend-native arrays should not be coerced to numpy")

        calls = {"to_device": 0}

        def fake_to_device(values):
            calls["to_device"] += 1
            return FakeGPUArray(values)

        corr.backend = SimpleNamespace(
            name="cupy",
            module=SimpleNamespace(ndarray=FakeGPUArray),
            device_id=0,
            to_device=fake_to_device,
            to_numpy=lambda values: values._values if isinstance(values, FakeGPUArray) else np.asarray(values),
        )

        nzbins = 2
        nzbin_combs = int(binom(nzbins + 1, 2))
        shear_maps = FakeGPUArray(np.ones((nzbins, 2, 8), dtype=np.float64), device_id=0)
        weights = FakeGPUArray(np.ones((nzbins, 8), dtype=np.float64), device_id=0)
        sumofweights = FakeGPUArray(np.ones((2, nzbin_combs), dtype=np.float64), device_id=0)
        xip_dev = FakeGPUArray(
            np.ones((nzbin_combs, corr.n_patches, corr.nbins), dtype=np.float64),
            device_id=0,
        )
        xim_dev = FakeGPUArray(
            np.ones((nzbin_combs, corr.n_patches, corr.nbins), dtype=np.float64),
            device_id=0,
        )

        with patch.object(corr, "_xipm_tomo_vectorized", return_value=(xip_dev, xim_dev)):
            xip, xim = corr.vectorized_shear_shear(
                shear_maps,
                weights,
                sumofweights=sumofweights,
            )

        self.assertEqual(calls["to_device"], 0)
        self.assertEqual(xip.shape, (nzbin_combs, corr.n_patches, corr.nbins))
        self.assertEqual(xim.shape, (nzbin_combs, corr.n_patches, corr.nbins))

    def test_backend_native_array_device_checks_and_native_fingerprint(self):
        corr = Correlation(
            nside=self.nside,
            phi_center=self.phi_center,
            theta_center=self.theta_center,
            nbins=self.nbins,
        )

        class FakeGPUArray:
            def __init__(self, values, *, device_id=0, with_ptr=True):
                self._values = np.asarray(values)
                self.shape = self._values.shape
                self.dtype = self._values.dtype
                self.device = SimpleNamespace(id=device_id)
                self.data = SimpleNamespace(ptr=id(self._values)) if with_ptr else SimpleNamespace()

            def astype(self, dtype, copy=False):
                return FakeGPUArray(
                    self._values.astype(dtype, copy=copy),
                    device_id=self.device.id,
                    with_ptr=hasattr(self.data, "ptr"),
                )

        class FakeNoDevice(FakeGPUArray):
            def __init__(self, values):
                self._values = np.asarray(values)
                self.shape = self._values.shape
                self.dtype = self._values.dtype

        corr.backend = SimpleNamespace(
            name="cupy",
            module=SimpleNamespace(ndarray=FakeGPUArray),
            device_id=0,
            to_device=lambda values: FakeGPUArray(values),
            to_numpy=lambda values: values._values if hasattr(values, "_values") else np.asarray(values),
        )

        self.assertFalse(corr._is_backend_native_array(FakeNoDevice([1.0, 2.0])))
        self.assertFalse(corr._is_backend_native_array(FakeGPUArray([1.0, 2.0], device_id=1)))

        fp = corr._fingerprint_weights(FakeGPUArray([1.0, 2.0], with_ptr=False))
        self.assertEqual(fp[0], (2,))
        self.assertEqual(fp[1], np.dtype(np.float64).str)
        self.assertIn("device:0;ptr:", fp[2])

    def test_get_aperture_methods_accept_backend_native_arrays_without_input_transfer(self):
        corr = Correlation(
            nside=self.nside,
            phi_center=self.phi_center,
            theta_center=self.theta_center,
            nbins=self.nbins,
        )

        class FakeGPUArray:
            def __init__(self, values, *, device_id=0):
                if isinstance(values, FakeGPUArray):
                    values = values._values
                self._values = np.asarray(values)
                self.shape = self._values.shape
                self.dtype = self._values.dtype
                self.ndim = self._values.ndim
                self.device = SimpleNamespace(id=device_id)
                self.data = SimpleNamespace(ptr=id(self._values))

            def astype(self, dtype, copy=False):
                return FakeGPUArray(self._values.astype(dtype, copy=copy), device_id=self.device.id)

            def __array__(self, *args, **kwargs):
                raise AssertionError("backend-native arrays should not be coerced to numpy")

            def __getitem__(self, key):
                out = self._values[key]
                if np.isscalar(out):
                    return out
                return FakeGPUArray(out, device_id=self.device.id)

            def __mul__(self, other):
                other_v = other._values if isinstance(other, FakeGPUArray) else other
                return FakeGPUArray(self._values * other_v, device_id=self.device.id)

            def __truediv__(self, other):
                other_v = other._values if isinstance(other, FakeGPUArray) else other
                return FakeGPUArray(self._values / other_v, device_id=self.device.id)

            def __neg__(self):
                return FakeGPUArray(-self._values, device_id=self.device.id)

        class FakeCupyModule:
            ndarray = FakeGPUArray
            float64 = np.float64

            @staticmethod
            def ascontiguousarray(array):
                return array

            class add:
                @staticmethod
                def reduceat(array, starts):
                    arr = array._values if isinstance(array, FakeGPUArray) else np.asarray(array)
                    st = starts._values if isinstance(starts, FakeGPUArray) else np.asarray(starts)
                    reduced = np.add.reduceat(arr, st.astype(np.int64, copy=False))
                    return FakeGPUArray(reduced)

        calls = {"native_input_to_device": 0}

        def fake_to_device(values):
            if isinstance(values, FakeGPUArray):
                calls["native_input_to_device"] += 1
                return values
            return FakeGPUArray(values)

        def fake_to_numpy(values):
            return values._values if isinstance(values, FakeGPUArray) else np.asarray(values)

        def aperture_shear_kernel(Q_inds, Q_cos, Q_sin, Q_val, g1, g2, w, out_num, out_den):
            idx = Q_inds._values.astype(np.int64, copy=False)
            gt = -(g1._values[idx] * Q_cos._values) - (g2._values[idx] * Q_sin._values)
            out_num._values[:] = w._values[idx] * gt * Q_val._values
            out_den._values[:] = w._values[idx]

        def aperture_density_kernel(Q_inds, Q_val, map_values, w, out_num, out_den):
            idx = Q_inds._values.astype(np.int64, copy=False)
            out_num._values[:] = w._values[idx] * map_values._values[idx] * Q_val._values
            out_den._values[:] = w._values[idx]

        corr.backend = SimpleNamespace(
            name="cupy",
            module=FakeCupyModule,
            device_id=0,
            to_device=fake_to_device,
            to_numpy=fake_to_numpy,
            zeros=lambda shape, dtype: FakeGPUArray(np.zeros(shape, dtype=dtype)),
            aperture_shear_kernel=aperture_shear_kernel,
            aperture_density_kernel=aperture_density_kernel,
        )

        corr.Q_inds_flat = np.array([0, 1, 2], dtype=np.uint32)
        corr.Q_cos_flat = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        corr.Q_sin_flat = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        corr.Q_val_flat = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        corr.Q_offsets = np.array([0, 3], dtype=np.int64)
        corr.Q_patch_area_flat = np.array([1.0], dtype=np.float64)

        g1 = FakeGPUArray(np.array([1.0, 1.0, 1.0], dtype=np.float64), device_id=0)
        g2 = FakeGPUArray(np.array([0.0, 0.0, 0.0], dtype=np.float64), device_id=0)
        w = FakeGPUArray(np.array([1.0, 1.0, 1.0], dtype=np.float64), device_id=0)
        density = FakeGPUArray(np.array([2.0, 2.0, 2.0], dtype=np.float64), device_id=0)

        shear_out = corr.get_aperture_shear(g1, g2, w, aperture_filter=None)
        density_out = corr.get_aperture_density(density, w, aperture_filter=None)

        self.assertEqual(calls["native_input_to_device"], 0)
        self.assertEqual(shear_out.shape, (corr.Q_patch_area_flat.shape[0],))
        self.assertEqual(density_out.shape, (corr.Q_patch_area_flat.shape[0],))


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

        self.assertAlmostEqual(M_a[0], expected_M_a)

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

        self.assertAlmostEqual(aperture_density[0], expected)

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

    def _make_small_cpu_corr(self) -> Correlation:
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
        corr.Q_inds = [np.array([0], dtype=np.uint32)]
        corr.Q_cos = [np.array([1.0], dtype=np.float64)]
        corr.Q_sin = [np.array([0.0], dtype=np.float64)]
        corr.Q_val = [np.array([1.0], dtype=np.float64)]
        corr.Q_patch_area = [1.0]
        corr.prepare()
        return corr

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
            "aperture_shear_all_patches",
            "_prepare_version",
            "_tomo_sumofweights_cache",
            "_tomo_sumofweights_cache_w_fingerprint",
            "_tomo_sumofweights_cache_prepare_version",
            "_xipm_sumofweights_cache",
            "_xipm_sumofweights_cache_w_fingerprint",
            "_xipm_sumofweights_cache_prepare_version",
            "_tomo_combination_cache",
            "Q_inds_flat",
            "Q_cos_flat",
            "Q_sin_flat",
            "Q_val_flat",
            "Q_offsets",
            "Q_patch_area_flat",
            "_aperture_filter_active_key",
        ]:
            state.pop(key, None)

        restored = Correlation.__new__(Correlation)
        restored.__setstate__(state)

        self.assertIsNotNone(restored.aperture_shear_all_patches)
        self.assertEqual(restored._prepare_version, 0)
        self.assertIsNone(restored._tomo_sumofweights_cache)
        self.assertIsNone(restored._xipm_sumofweights_cache)
        self.assertEqual(restored._tomo_combination_cache, {})
        self.assertEqual(restored._aperture_filter_active_key, "Q_T")
        self.assertIsNone(restored.Q_inds_flat)
        self.assertIsNone(restored.Q_offsets)

    def test_aperture_filter_helpers_and_ensure_pairs_branches(self):
        corr = self._make_small_cpu_corr()

        self.assertEqual(corr._aperture_filter_key(correlations_module.Q_T), "Q_T")

        def custom_filter(theta):
            return np.full_like(theta, 2.0)

        self.assertIsInstance(corr._aperture_filter_key(custom_filter), int)
        vals = corr._evaluate_aperture_filter(custom_filter, np.array([1.0], dtype=np.float64))
        np.testing.assert_allclose(vals, np.array([2.0], dtype=np.float64))

        corr.Q_inds_flat = None
        corr.Q_inds = [np.array([0], dtype=np.uint32)]
        corr._aperture_filter_active_key = "different"
        with patch.object(corr, "calculate_pairs_M_a") as spy_calc, patch.object(
            corr, "_prepare_aperture_flat"
        ) as spy_flat:
            corr._ensure_aperture_pairs(aperture_filter=custom_filter)
        spy_calc.assert_called_once()
        spy_flat.assert_not_called()

        corr.Q_inds_flat = np.array([0], dtype=np.uint32)
        corr._aperture_filter_active_key = "stale"
        with patch.object(corr, "calculate_pairs_M_a") as spy_calc:
            corr._ensure_aperture_pairs(aperture_filter=custom_filter)
        spy_calc.assert_called_once()

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

    def test_reduce_pairs_supports_2d_pair_arrays(self):
        self._setup_mock_pairs()

        values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        reduced = self.corr.backend.to_numpy(self.corr._reduce_pairs(values))

        expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        np.testing.assert_allclose(reduced, expected)

    def test_vectorized_shear_shear_prefers_vectorized_path_when_available(self):
        self._setup_mock_pairs()

        nzbins = 2
        nzbin_combs = int(binom(nzbins + 1, 2))
        shear_maps = np.random.rand(nzbins, 2, self.npix)
        w = np.random.rand(nzbins, self.npix)
        xip_mock = np.ones((nzbin_combs, self.corr.n_patches, self.corr.nbins))
        xim_mock = np.full((nzbin_combs, self.corr.n_patches, self.corr.nbins), 2.0)

        with patch.object(
            self.corr,
            "_xipm_tomo_vectorized",
            return_value=(
                self.corr.backend.to_device(xip_mock),
                self.corr.backend.to_device(xim_mock),
            ),
        ) as spy_vectorized, patch.object(
            self.corr,
            "_xipm_auto",
            side_effect=AssertionError("_xipm_auto should not be called"),
        ), patch.object(
            self.corr,
            "_xipm_cross",
            side_effect=AssertionError("_xipm_cross should not be called"),
        ):
            xip, xim = self.corr.vectorized_shear_shear(shear_maps, w)

        spy_vectorized.assert_called_once()
        np.testing.assert_allclose(xip, xip_mock)
        np.testing.assert_allclose(xim, xim_mock)

    def test_xipm_auto_return_numpy_false(self):
        corr = self._make_small_cpu_corr()
        g1 = np.ones(12, dtype=np.float64)
        g2 = np.ones(12, dtype=np.float64)
        w = np.ones(12, dtype=np.float64)

        xip_dev, xim_dev = corr._xipm_auto(
            g1,
            g2,
            w,
            sumofweights=1.0,
            return_numpy=False,
        )

        self.assertEqual(xip_dev.shape, (corr.n_patches, corr.nbins))
        self.assertEqual(xim_dev.shape, (corr.n_patches, corr.nbins))

    def test_xipm_cross_return_numpy_false(self):
        corr = self._make_small_cpu_corr()
        g11 = np.ones(12, dtype=np.float64)
        g21 = np.ones(12, dtype=np.float64)
        g12 = np.full(12, 2.0, dtype=np.float64)
        g22 = np.full(12, 3.0, dtype=np.float64)
        w1 = np.ones(12, dtype=np.float64)
        w2 = np.ones(12, dtype=np.float64)

        xip_dev, xim_dev = corr._xipm_cross(
            g11,
            g21,
            g12,
            g22,
            w1,
            w2,
            sumofweights_ab=1.0,
            sumofweights_ba=1.0,
            return_numpy=False,
        )

        self.assertEqual(xip_dev.shape, (corr.n_patches, corr.nbins))
        self.assertEqual(xim_dev.shape, (corr.n_patches, corr.nbins))

    def test_compute_shear_shear_public_api(self):
        corr = self._make_small_cpu_corr()
        self.assertFalse(hasattr(corr, "xipm"))

        g11 = np.ones(12, dtype=np.float64)
        g21 = np.ones(12, dtype=np.float64)
        g12 = np.ones(12, dtype=np.float64)
        g22 = np.ones(12, dtype=np.float64)
        w1 = np.ones(12, dtype=np.float64)
        w2 = np.ones(12, dtype=np.float64)

        xip, xim = corr.compute_shear_shear(g11, g21, g12, g22, w1, w2, sumofweights=1.0)
        self.assertEqual(xip.shape, (corr.n_patches, corr.nbins))
        self.assertEqual(xim.shape, (corr.n_patches, corr.nbins))

    def test_compute_density_density(self):
        corr = self._make_small_cpu_corr()
        d1 = np.zeros(12, dtype=np.float64)
        d2 = np.zeros(12, dtype=np.float64)
        d1[:3] = np.array([1.0, 2.0, 3.0])
        d2[:3] = np.array([4.0, 5.0, 6.0])
        w1 = np.ones(12, dtype=np.float64)
        w2 = np.ones(12, dtype=np.float64)

        (wtheta,) = corr.compute_density_density(d1, d2, w1, w2, sumofweights=1.0)
        expected_ab = d1[0] * d2[1] + d1[1] * d2[2]
        expected_ba = d1[1] * d2[0] + d1[2] * d2[1]
        expected = 0.5 * (expected_ab + expected_ba)
        self.assertAlmostEqual(wtheta[0, 0], expected)

    def test_compute_density_density_missing_kernel_raises(self):
        corr = self._make_small_cpu_corr()
        corr.backend.kernel_density_density = None
        with self.assertRaisesRegex(RuntimeError, "density-density kernel"):
            corr.compute_density_density(
                np.ones(12, dtype=np.float64),
                np.ones(12, dtype=np.float64),
                np.ones(12, dtype=np.float64),
                np.ones(12, dtype=np.float64),
            )

    def test_compute_density_shear(self):
        corr = self._make_small_cpu_corr()
        d_lens = np.zeros(12, dtype=np.float64)
        g1 = np.zeros(12, dtype=np.float64)
        g2 = np.zeros(12, dtype=np.float64)
        d_lens[:3] = np.array([1.0, 2.0, 3.0])
        g1[:3] = np.array([0.1, 0.2, 0.3])
        g2[:3] = np.array([0.4, 0.5, 0.6])
        w_lens = np.ones(12, dtype=np.float64)
        w_source = np.ones(12, dtype=np.float64)

        (gamma_t,) = corr.compute_density_shear(
            d_lens,
            g1,
            g2,
            w_lens,
            w_source,
            sumofweights=1.0,
        )
        expected = (
            -d_lens[0] * g1[1]
            -d_lens[1] * g1[2]
            -d_lens[1] * g1[0]
            -d_lens[2] * g1[1]
        )
        self.assertAlmostEqual(gamma_t[0, 0], expected)

    def test_compute_density_shear_missing_kernel_raises(self):
        corr = self._make_small_cpu_corr()
        corr.backend.kernel_density_shear = None
        with self.assertRaisesRegex(RuntimeError, "density-shear kernel"):
            corr.compute_density_shear(
                np.ones(12, dtype=np.float64),
                np.ones(12, dtype=np.float64),
                np.ones(12, dtype=np.float64),
                np.ones(12, dtype=np.float64),
                np.ones(12, dtype=np.float64),
            )

    def test_compute_density_methods_prepare_on_demand(self):
        corr = self._make_small_cpu_corr()
        corr.inds_dev = None

        with patch.object(corr, "prepare", wraps=corr.prepare) as spy_prepare:
            corr.compute_density_density(
                np.ones(12, dtype=np.float64),
                np.ones(12, dtype=np.float64),
                np.ones(12, dtype=np.float64),
                np.ones(12, dtype=np.float64),
                sumofweights=1.0,
            )
            spy_prepare.assert_called_once()

        corr.inds_dev = None
        with patch.object(corr, "prepare", wraps=corr.prepare) as spy_prepare:
            corr.compute_density_shear(
                np.ones(12, dtype=np.float64),
                np.ones(12, dtype=np.float64),
                np.zeros(12, dtype=np.float64),
                np.ones(12, dtype=np.float64),
                np.ones(12, dtype=np.float64),
                sumofweights=1.0,
            )
            spy_prepare.assert_called_once()

    def test_normalize_scalar_pairs_array_sumofweights(self):
        corr = self._make_small_cpu_corr()
        num = np.array([2.0], dtype=np.float64)
        sumofweights = np.array([4.0], dtype=np.float64)

        normalized = corr._normalize_scalar_pairs(num, sumofweights)
        self.assertEqual(normalized.shape, (corr.n_patches, corr.nbins))
        self.assertAlmostEqual(normalized[0, 0], 0.5)

    def test_vectorized_shear_shear_returns_2pcf_outputs(self):
        corr = self._make_small_cpu_corr()
        shear_maps = np.ones((2, 2, 12), dtype=np.float64)
        w = np.ones((2, 12), dtype=np.float64)
        xip = np.full((3, corr.n_patches, corr.nbins), 1.5, dtype=np.float64)
        xim = np.full((3, corr.n_patches, corr.nbins), 2.5, dtype=np.float64)

        with patch.object(
            corr,
            "_xipm_tomo_vectorized",
            return_value=(corr.backend.to_device(xip), corr.backend.to_device(xim)),
        ) as spy_vectorized:
            out_xip, out_xim = corr.vectorized_shear_shear(
                shear_maps,
                w,
                sumofweights=np.ones((2, 3, corr.n_patches * corr.nbins), dtype=np.float64),
                flip_g1=True,
                flip_g2=False,
            )

        spy_vectorized.assert_called_once()
        np.testing.assert_allclose(out_xip, xip)
        np.testing.assert_allclose(out_xim, xim)

    def test_vectorized_density_methods_delegate_to_tomo_helpers(self):
        corr = self._make_small_cpu_corr()
        density_maps = np.arange(24, dtype=np.float64).reshape(2, 12)
        shear_maps = np.zeros((2, 2, 12), dtype=np.float64)
        density_w = np.ones((2, 12), dtype=np.float64)
        shear_w = np.ones((2, 12), dtype=np.float64)
        dd_out = np.full((3, corr.n_patches, corr.nbins), 7.0, dtype=np.float64)
        ds_out = np.full((4, corr.n_patches, corr.nbins), 9.0, dtype=np.float64)
        dd_sum = np.arange(3, dtype=np.float64)
        ds_sum = np.arange(4, dtype=np.float64)

        with patch.object(
            corr,
            "_density_density_tomo_vectorized",
            return_value=corr.backend.to_device(dd_out),
        ) as spy_dd, patch.object(
            corr,
            "_density_shear_tomo_vectorized",
            return_value=corr.backend.to_device(ds_out),
        ) as spy_ds, patch.object(
            corr,
            "compute_density_density",
            side_effect=AssertionError("legacy density-density loop path should not run"),
        ), patch.object(
            corr,
            "compute_density_shear",
            side_effect=AssertionError("legacy density-shear loop path should not run"),
        ):
            out_dd_none = corr.vectorized_density_density(density_maps, density_w)
            out_ds_none = corr.vectorized_density_shear(
                density_maps,
                shear_maps,
                density_w,
                shear_w,
            )

            out_dd_explicit = corr.vectorized_density_density(
                density_maps,
                density_w,
                sumofweights=dd_sum,
            )
            out_ds_explicit = corr.vectorized_density_shear(
                density_maps,
                shear_maps,
                density_w,
                shear_w,
                sumofweights=ds_sum,
            )

        self.assertEqual(out_dd_none.shape, (3, corr.n_patches, corr.nbins))
        self.assertEqual(out_ds_none.shape, (4, corr.n_patches, corr.nbins))
        self.assertEqual(out_dd_explicit.shape, (3, corr.n_patches, corr.nbins))
        self.assertEqual(out_ds_explicit.shape, (4, corr.n_patches, corr.nbins))
        np.testing.assert_allclose(out_dd_none, dd_out)
        np.testing.assert_allclose(out_ds_none, ds_out)
        np.testing.assert_allclose(out_dd_explicit, dd_out)
        np.testing.assert_allclose(out_ds_explicit, ds_out)
        self.assertEqual(spy_dd.call_count, 2)
        self.assertEqual(spy_ds.call_count, 2)
        self.assertIsNone(spy_dd.call_args_list[0].args[2])
        self.assertIsNone(spy_ds.call_args_list[0].args[4])
        np.testing.assert_array_equal(spy_dd.call_args_list[1].args[2], dd_sum)
        np.testing.assert_array_equal(spy_ds.call_args_list[1].args[4], ds_sum)

    def test_vectorized_density_density_auto_pairs_use_unique_term(self):
        corr = self._make_small_cpu_corr()
        density_maps = np.arange(24, dtype=np.float64).reshape(2, 12)
        density_w = np.linspace(1.0, 2.0, 24, dtype=np.float64).reshape(2, 12)

        out = corr.vectorized_density_density(density_maps, density_w)
        self.assertEqual(out.shape, (3, corr.n_patches, corr.nbins))

        (auto_00,) = corr.compute_density_density(
            density_maps[0],
            density_maps[0],
            density_w[0],
            density_w[0],
        )
        (auto_11,) = corr.compute_density_density(
            density_maps[1],
            density_maps[1],
            density_w[1],
            density_w[1],
        )
        np.testing.assert_allclose(out[0], auto_00)
        np.testing.assert_allclose(out[2], auto_11)

    def test_vectorized_density_density_auto_only_returns_only_auto_correlations(self):
        corr = self._make_small_cpu_corr()
        density_maps = np.arange(24, dtype=np.float64).reshape(2, 12)
        density_w = np.linspace(1.0, 2.0, 24, dtype=np.float64).reshape(2, 12)

        out = corr.vectorized_density_density(
            density_maps,
            density_w,
            gc_auto_correlations_only=True,
        )
        self.assertEqual(out.shape, (2, corr.n_patches, corr.nbins))

        (auto_00,) = corr.compute_density_density(
            density_maps[0],
            density_maps[0],
            density_w[0],
            density_w[0],
        )
        (auto_11,) = corr.compute_density_density(
            density_maps[1],
            density_maps[1],
            density_w[1],
            density_w[1],
        )
        np.testing.assert_allclose(out[0], auto_00)
        np.testing.assert_allclose(out[1], auto_11)

    def test_get_full_tomo_density_returns_aperture_and_wtheta(self):
        corr = self._make_small_cpu_corr()
        density_maps = np.zeros((2, 12), dtype=np.float64)
        density_maps[0, 0] = 2.0
        density_maps[1, 0] = 3.0
        density_w = np.ones((2, 12), dtype=np.float64)
        wtheta_mock = np.full((3, corr.n_patches, corr.nbins), 4.0, dtype=np.float64)

        with patch.object(
            corr,
            "vectorized_density_density",
            return_value=wtheta_mock,
        ) as spy_wtheta:
            N_ap, wtheta = corr.get_full_tomo_density(density_maps, density_w)

        spy_wtheta.assert_called_once()
        np.testing.assert_allclose(N_ap[:, 0], np.array([2.0, 3.0], dtype=np.float64))
        np.testing.assert_allclose(wtheta, wtheta_mock)

    def test_get_full_tomo_shear_forwards_aperture_filter(self):
        corr = self._make_small_cpu_corr()
        shear_maps = np.ones((2, 2, 12), dtype=np.float64)
        shear_w = np.ones((2, 12), dtype=np.float64)

        def custom_filter(theta, _theta_Q):
            return np.ones_like(theta)

        with patch.object(
            corr,
            "_compute_tomo_aperture_shear",
            return_value=np.full((2, corr.n_patches), 1.0, dtype=np.float64),
        ) as spy_map, patch.object(
            corr,
            "vectorized_shear_shear",
            return_value=(
                np.full((3, corr.n_patches, corr.nbins), 2.0, dtype=np.float64),
                np.full((3, corr.n_patches, corr.nbins), 3.0, dtype=np.float64),
            ),
        ):
            _ = corr.get_full_tomo_shear(
                shear_maps,
                shear_w,
                aperture_filter=custom_filter,
            )

        spy_map.assert_called_once()
        self.assertIs(spy_map.call_args.kwargs["aperture_filter"], custom_filter)

    def test_get_full_tomo_density_forwards_gc_auto_only_flag(self):
        corr = self._make_small_cpu_corr()
        density_maps = np.ones((2, 12), dtype=np.float64)
        density_w = np.ones((2, 12), dtype=np.float64)

        with patch.object(
            corr,
            "vectorized_density_density",
            return_value=np.full((2, corr.n_patches, corr.nbins), 4.0, dtype=np.float64),
        ) as spy_wtheta:
            _ = corr.get_full_tomo_density(
                density_maps,
                density_w,
                gc_auto_correlations_only=True,
            )

        spy_wtheta.assert_called_once()
        self.assertTrue(spy_wtheta.call_args.kwargs["gc_auto_correlations_only"])

    def test_get_full_tomo_density_forwards_aperture_filter(self):
        corr = self._make_small_cpu_corr()
        density_maps = np.ones((2, 12), dtype=np.float64)
        density_w = np.ones((2, 12), dtype=np.float64)

        def custom_filter(theta, _theta_Q):
            return np.ones_like(theta)

        with patch.object(
            corr,
            "_compute_tomo_aperture_density",
            return_value=np.full((2, corr.n_patches), 4.0, dtype=np.float64),
        ) as spy_nap, patch.object(
            corr,
            "vectorized_density_density",
            return_value=np.full((3, corr.n_patches, corr.nbins), 5.0, dtype=np.float64),
        ):
            _ = corr.get_full_tomo_density(
                density_maps,
                density_w,
                aperture_filter=custom_filter,
            )

        spy_nap.assert_called_once()
        self.assertIs(spy_nap.call_args.kwargs["aperture_filter"], custom_filter)

    def test_get_full_tomo_ggl_optional_outputs(self):
        corr = self._make_small_cpu_corr()
        density_maps = np.ones((2, 12), dtype=np.float64)
        shear_maps = np.ones((2, 2, 12), dtype=np.float64)
        density_w = np.ones((2, 12), dtype=np.float64)
        shear_w = np.ones((2, 12), dtype=np.float64)

        gammat_mock = np.full((3, corr.n_patches, corr.nbins), 8.0, dtype=np.float64)
        N_ap_mock = np.full((2, corr.n_patches), 6.0, dtype=np.float64)
        M_ap_mock = np.full((2, corr.n_patches), 5.0, dtype=np.float64)

        with patch.object(
            corr,
            "vectorized_density_shear",
            return_value=gammat_mock,
        ) as spy_gammat, patch.object(
            corr,
            "_compute_tomo_aperture_density",
            return_value=N_ap_mock,
        ) as spy_nap, patch.object(
            corr,
            "_compute_tomo_aperture_shear",
            return_value=M_ap_mock,
        ) as spy_map:
            out_default = corr.get_full_tomo_ggl(
                density_maps,
                shear_maps,
                density_w,
                shear_w,
            )
            out_nap = corr.get_full_tomo_ggl(
                density_maps,
                shear_maps,
                density_w,
                shear_w,
                return_N_ap=True,
            )
            out_map = corr.get_full_tomo_ggl(
                density_maps,
                shear_maps,
                density_w,
                shear_w,
                return_M_ap=True,
            )
            out_both = corr.get_full_tomo_ggl(
                density_maps,
                shear_maps,
                density_w,
                shear_w,
                return_N_ap=True,
                return_M_ap=True,
            )

        self.assertEqual(spy_gammat.call_count, 4)
        self.assertEqual(spy_nap.call_count, 2)
        self.assertEqual(spy_map.call_count, 2)
        np.testing.assert_allclose(out_default, gammat_mock)
        np.testing.assert_allclose(out_nap[0], gammat_mock)
        np.testing.assert_allclose(out_nap[1], N_ap_mock)
        np.testing.assert_allclose(out_map[0], gammat_mock)
        np.testing.assert_allclose(out_map[1], M_ap_mock)
        np.testing.assert_allclose(out_both[0], gammat_mock)
        np.testing.assert_allclose(out_both[1], N_ap_mock)
        np.testing.assert_allclose(out_both[2], M_ap_mock)

    def test_get_full_tomo_ggl_forwards_flip_flags(self):
        corr = self._make_small_cpu_corr()
        density_maps = np.ones((2, 12), dtype=np.float64)
        shear_maps = np.ones((2, 2, 12), dtype=np.float64)
        density_w = np.ones((2, 12), dtype=np.float64)
        shear_w = np.ones((2, 12), dtype=np.float64)

        gammat_mock = np.full((3, corr.n_patches, corr.nbins), 8.0, dtype=np.float64)
        M_ap_mock = np.full((2, corr.n_patches), 5.0, dtype=np.float64)

        with patch.object(
            corr,
            "vectorized_density_shear",
            return_value=gammat_mock,
        ) as spy_gammat, patch.object(
            corr,
            "_compute_tomo_aperture_shear",
            return_value=M_ap_mock,
        ) as spy_map:
            out = corr.get_full_tomo_ggl(
                density_maps,
                shear_maps,
                density_w,
                shear_w,
                return_M_ap=True,
                flip_g1=True,
                flip_g2=True,
            )

        np.testing.assert_allclose(out[0], gammat_mock)
        np.testing.assert_allclose(out[1], M_ap_mock)
        spy_gammat.assert_called_once()
        self.assertTrue(spy_gammat.call_args.kwargs["flip_g1"])
        self.assertTrue(spy_gammat.call_args.kwargs["flip_g2"])
        spy_map.assert_called_once()
        self.assertTrue(spy_map.call_args.kwargs["flip_g1"])
        self.assertTrue(spy_map.call_args.kwargs["flip_g2"])

    def test_get_full_tomo_ggl_forwards_selected_bin_combinations(self):
        corr = self._make_small_cpu_corr()
        density_maps = np.ones((2, 12), dtype=np.float64)
        shear_maps = np.ones((2, 2, 12), dtype=np.float64)
        density_w = np.ones((2, 12), dtype=np.float64)
        shear_w = np.ones((2, 12), dtype=np.float64)
        selected = [(1, 0)]

        with patch.object(
            corr,
            "vectorized_density_shear",
            return_value=np.full((1, corr.n_patches, corr.nbins), 8.0, dtype=np.float64),
        ) as spy_gammat:
            _ = corr.get_full_tomo_ggl(
                density_maps,
                shear_maps,
                density_w,
                shear_w,
                ggl_bin_combinations=selected,
            )

        spy_gammat.assert_called_once()
        self.assertEqual(spy_gammat.call_args.kwargs["ggl_bin_combinations"], selected)

    def test_get_full_tomo_ggl_forwards_aperture_filter_to_optional_outputs(self):
        corr = self._make_small_cpu_corr()
        density_maps = np.ones((2, 12), dtype=np.float64)
        shear_maps = np.ones((2, 2, 12), dtype=np.float64)
        density_w = np.ones((2, 12), dtype=np.float64)
        shear_w = np.ones((2, 12), dtype=np.float64)

        def custom_filter(theta, _theta_Q):
            return np.ones_like(theta)

        with patch.object(
            corr,
            "vectorized_density_shear",
            return_value=np.full((4, corr.n_patches, corr.nbins), 1.0, dtype=np.float64),
        ), patch.object(
            corr,
            "_compute_tomo_aperture_density",
            return_value=np.full((2, corr.n_patches), 2.0, dtype=np.float64),
        ) as spy_nap, patch.object(
            corr,
            "_compute_tomo_aperture_shear",
            return_value=np.full((2, corr.n_patches), 3.0, dtype=np.float64),
        ) as spy_map:
            _ = corr.get_full_tomo_ggl(
                density_maps,
                shear_maps,
                density_w,
                shear_w,
                aperture_filter=custom_filter,
                return_N_ap=True,
                return_M_ap=True,
            )

        self.assertIs(spy_nap.call_args.kwargs["aperture_filter"], custom_filter)
        self.assertIs(spy_map.call_args.kwargs["aperture_filter"], custom_filter)

    def test_vectorized_density_shear_uses_directional_lens_source_pairs(self):
        corr = self._make_small_cpu_corr()
        density_maps = np.array(
            [
                np.linspace(0.2, 1.3, 12, dtype=np.float64),
                np.linspace(1.4, 0.3, 12, dtype=np.float64),
            ]
        )
        shear_maps = np.zeros((2, 2, 12), dtype=np.float64)
        shear_maps[0, 0] = np.linspace(-0.4, 0.8, 12, dtype=np.float64)
        shear_maps[0, 1] = np.linspace(0.1, -0.2, 12, dtype=np.float64)
        shear_maps[1, 0] = np.linspace(0.7, -0.5, 12, dtype=np.float64)
        shear_maps[1, 1] = np.linspace(-0.3, 0.6, 12, dtype=np.float64)
        density_w = np.linspace(0.8, 1.6, 24, dtype=np.float64).reshape(2, 12)
        shear_w = np.linspace(1.1, 2.3, 24, dtype=np.float64).reshape(2, 12)

        out = corr.vectorized_density_shear(density_maps, shear_maps, density_w, shear_w)
        self.assertEqual(out.shape, (4, corr.n_patches, corr.nbins))

        combos = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for k, (i, j) in enumerate(combos):
            (num_ab,) = corr.compute_density_shear(
                density_maps[i],
                shear_maps[j, 0],
                shear_maps[j, 1],
                density_w[i],
                shear_w[j],
                sumofweights=1.0,
            )
            sum_ab = corr.backend.to_numpy(
                corr._get_xipm_sumofweights(
                    corr.backend.to_device(density_w[i]),
                    corr.backend.to_device(shear_w[j]),
                )
            ).reshape(corr.n_patches, corr.nbins)
            sum_ba = corr.backend.to_numpy(
                corr._get_xipm_sumofweights(
                    corr.backend.to_device(shear_w[j]),
                    corr.backend.to_device(density_w[i]),
                )
            ).reshape(corr.n_patches, corr.nbins)

            expected = np.zeros_like(num_ab)
            sum_total = sum_ab + sum_ba
            nonzero = sum_total != 0
            expected[nonzero] = num_ab[nonzero] / sum_total[nonzero]

            np.testing.assert_allclose(out[k], expected)

    def test_vectorized_density_shear_supports_selected_bin_subset_in_order(self):
        corr = self._make_small_cpu_corr()
        density_maps = np.array(
            [
                np.linspace(0.2, 1.3, 12, dtype=np.float64),
                np.linspace(1.4, 0.3, 12, dtype=np.float64),
            ]
        )
        shear_maps = np.zeros((3, 2, 12), dtype=np.float64)
        shear_maps[0, 0] = np.linspace(-0.4, 0.8, 12, dtype=np.float64)
        shear_maps[0, 1] = np.linspace(0.1, -0.2, 12, dtype=np.float64)
        shear_maps[1, 0] = np.linspace(0.7, -0.5, 12, dtype=np.float64)
        shear_maps[1, 1] = np.linspace(-0.3, 0.6, 12, dtype=np.float64)
        shear_maps[2, 0] = np.linspace(0.2, -0.7, 12, dtype=np.float64)
        shear_maps[2, 1] = np.linspace(0.9, -0.1, 12, dtype=np.float64)
        density_w = np.linspace(0.8, 1.6, 24, dtype=np.float64).reshape(2, 12)
        shear_w = np.linspace(1.1, 2.9, 36, dtype=np.float64).reshape(3, 12)

        selected = [(1, 2), (0, 1)]
        out = corr.vectorized_density_shear(
            density_maps,
            shear_maps,
            density_w,
            shear_w,
            ggl_bin_combinations=selected,
        )
        self.assertEqual(out.shape, (2, corr.n_patches, corr.nbins))

        for k, (i, j) in enumerate(selected):
            (num_ab,) = corr.compute_density_shear(
                density_maps[i],
                shear_maps[j, 0],
                shear_maps[j, 1],
                density_w[i],
                shear_w[j],
                sumofweights=1.0,
            )
            sum_ab = corr.backend.to_numpy(
                corr._get_xipm_sumofweights(
                    corr.backend.to_device(density_w[i]),
                    corr.backend.to_device(shear_w[j]),
                )
            ).reshape(corr.n_patches, corr.nbins)
            sum_ba = corr.backend.to_numpy(
                corr._get_xipm_sumofweights(
                    corr.backend.to_device(shear_w[j]),
                    corr.backend.to_device(density_w[i]),
                )
            ).reshape(corr.n_patches, corr.nbins)

            expected = np.zeros_like(num_ab)
            sum_total = sum_ab + sum_ba
            nonzero = sum_total != 0
            expected[nonzero] = num_ab[nonzero] / sum_total[nonzero]

            np.testing.assert_allclose(out[k], expected)

    def test_vectorized_density_shear_flip_flags_apply_to_shear_components(self):
        corr = self._make_small_cpu_corr()
        density_maps = np.ones((2, 12), dtype=np.float64)
        shear_maps = np.zeros((2, 2, 12), dtype=np.float64)
        shear_maps[:, 0] = 2.0
        shear_maps[:, 1] = -3.0
        density_w = np.ones((2, 12), dtype=np.float64)
        shear_w = np.ones((2, 12), dtype=np.float64)

        captured = {}

        def _fake_density_shear_tomo(
            density_arg,
            shear_arg,
            density_w_arg,
            shear_w_arg,
            sumofweights_arg,
            nlens_bins_arg,
            nsource_bins_arg,
            ggl_bin_combinations=None,
        ):
            captured["shear"] = np.array(shear_arg, copy=True)
            return np.zeros((4, corr.n_patches, corr.nbins), dtype=np.float64)

        with patch.object(
            corr,
            "_density_shear_tomo_vectorized",
            side_effect=_fake_density_shear_tomo,
        ):
            corr.vectorized_density_shear(
                density_maps,
                shear_maps,
                density_w,
                shear_w,
                flip_g1=True,
                flip_g2=True,
            )

        self.assertIn("shear", captured)
        np.testing.assert_allclose(captured["shear"][:, 0], -2.0)
        np.testing.assert_allclose(captured["shear"][:, 1], 3.0)

    def test_vectorized_density_shear_allows_distinct_lens_and_source_tomo_bins(self):
        corr = self._make_small_cpu_corr()
        density_maps = np.ones((2, 12), dtype=np.float64)
        shear_maps = np.ones((3, 2, 12), dtype=np.float64)
        density_w = np.ones((2, 12), dtype=np.float64)
        shear_w = np.ones((3, 12), dtype=np.float64)

        captured = {}

        def _fake_density_shear_tomo(
            density_arg,
            shear_arg,
            density_w_arg,
            shear_w_arg,
            sumofweights_arg,
            nlens_bins_arg,
            nsource_bins_arg,
            ggl_bin_combinations=None,
        ):
            captured["nlens"] = nlens_bins_arg
            captured["nsource"] = nsource_bins_arg
            return np.zeros((nlens_bins_arg * nsource_bins_arg, corr.n_patches, corr.nbins), dtype=np.float64)

        with patch.object(
            corr,
            "_density_shear_tomo_vectorized",
            side_effect=_fake_density_shear_tomo,
        ):
            out = corr.vectorized_density_shear(
                density_maps,
                shear_maps,
                density_w,
                shear_w,
            )

        self.assertEqual(captured["nlens"], 2)
        self.assertEqual(captured["nsource"], 3)
        self.assertEqual(out.shape, (6, corr.n_patches, corr.nbins))

    def test_vectorized_density_shear_des_y3_bin_count_has_20_correlations(self):
        corr = self._make_small_cpu_corr()
        density_maps = np.ones((5, 12), dtype=np.float64)
        shear_maps = np.ones((4, 2, 12), dtype=np.float64)
        density_w = np.ones((5, 12), dtype=np.float64)
        shear_w = np.ones((4, 12), dtype=np.float64)

        captured = {}

        def _fake_density_shear_tomo(
            density_arg,
            shear_arg,
            density_w_arg,
            shear_w_arg,
            sumofweights_arg,
            nlens_bins_arg,
            nsource_bins_arg,
            ggl_bin_combinations=None,
        ):
            captured["nlens"] = nlens_bins_arg
            captured["nsource"] = nsource_bins_arg
            return np.zeros(
                (nlens_bins_arg * nsource_bins_arg, corr.n_patches, corr.nbins),
                dtype=np.float64,
            )

        with patch.object(
            corr,
            "_density_shear_tomo_vectorized",
            side_effect=_fake_density_shear_tomo,
        ):
            out = corr.vectorized_density_shear(
                density_maps,
                shear_maps,
                density_w,
                shear_w,
            )

        self.assertEqual(captured["nlens"], 5)
        self.assertEqual(captured["nsource"], 4)
        self.assertEqual(out.shape, (20, corr.n_patches, corr.nbins))

    def test_vectorized_density_shear_invalid_shear_shape_raises(self):
        corr = self._make_small_cpu_corr()
        density_maps = np.ones((2, 12), dtype=np.float64)
        shear_maps = np.ones((2, 3, 12), dtype=np.float64)
        density_w = np.ones((2, 12), dtype=np.float64)
        shear_w = np.ones((2, 12), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "shear_maps must have shape"):
            corr.vectorized_density_shear(
                density_maps,
                shear_maps,
                density_w,
                shear_w,
            )

    def test_vectorized_density_shear_additional_validation_errors(self):
        corr = self._make_small_cpu_corr()
        density_maps = np.ones((2, 12), dtype=np.float64)
        shear_maps = np.ones((2, 2, 12), dtype=np.float64)
        density_w = np.ones((2, 12), dtype=np.float64)
        shear_w = np.ones((2, 12), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "density_maps must have shape"):
            corr.vectorized_density_shear(
                np.ones((12,), dtype=np.float64),
                shear_maps,
                density_w,
                shear_w,
            )

        with self.assertRaisesRegex(ValueError, "density_weights must match density_maps"):
            corr.vectorized_density_shear(
                density_maps,
                shear_maps,
                np.ones((3, 12), dtype=np.float64),
                shear_w,
            )

        with self.assertRaisesRegex(ValueError, "shear_weights must have shape"):
            corr.vectorized_density_shear(
                density_maps,
                shear_maps,
                density_w,
                np.ones((2, 1, 12), dtype=np.float64),
            )

        with self.assertRaisesRegex(ValueError, "shear_weights must match shear_maps"):
            corr.vectorized_density_shear(
                density_maps,
                shear_maps,
                density_w,
                np.ones((3, 12), dtype=np.float64),
            )

        with self.assertRaisesRegex(ValueError, "must have the same number of pixels"):
            corr.vectorized_density_shear(
                np.ones((2, 11), dtype=np.float64),
                np.ones((2, 2, 12), dtype=np.float64),
                np.ones((2, 11), dtype=np.float64),
                np.ones((2, 12), dtype=np.float64),
            )

    def test_get_selected_tomo_cross_combinations_validation_errors(self):
        corr = self._make_small_cpu_corr()

        with self.assertRaisesRegex(
            ValueError, r"iterable of \(lens_bin, source_bin\) pairs"
        ):
            corr._get_selected_tomo_cross_combination_indices(
                2,
                2,
                ggl_bin_combinations=[0],
            )

        with self.assertRaisesRegex(ValueError, "Lens tomographic bin index"):
            corr._get_selected_tomo_cross_combination_indices(
                2,
                2,
                ggl_bin_combinations=[(2, 0)],
            )

        with self.assertRaisesRegex(ValueError, "Source tomographic bin index"):
            corr._get_selected_tomo_cross_combination_indices(
                2,
                2,
                ggl_bin_combinations=[(0, 2)],
            )

    def test_normalize_tomo_sumofweights_per_comb_branches(self):
        corr = self._make_small_cpu_corr()
        nzbin_combs = 3

        out_scalar = corr._normalize_tomo_sumofweights_per_comb(2.0, nzbin_combs)
        self.assertEqual(corr.backend.to_numpy(out_scalar).shape, (3, 1))

        out_1d_combs = corr._normalize_tomo_sumofweights_per_comb(
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            nzbin_combs,
        )
        self.assertEqual(corr.backend.to_numpy(out_1d_combs).shape, (3, 1))

        out_1d_flat = corr._normalize_tomo_sumofweights_per_comb(
            np.array([1.0, 2.0, 3.0], dtype=np.float64).reshape(-1),
            nzbin_combs,
        )
        self.assertEqual(corr.backend.to_numpy(out_1d_flat).shape, (3, 1))

        out_2d = corr._normalize_tomo_sumofweights_per_comb(
            np.array([[1.0], [2.0], [3.0]], dtype=np.float64),
            nzbin_combs,
        )
        self.assertEqual(corr.backend.to_numpy(out_2d).shape, (3, 1))

        out_3d = corr._normalize_tomo_sumofweights_per_comb(
            np.array([[[1.0]], [[2.0]], [[3.0]]], dtype=np.float64),
            nzbin_combs,
        )
        self.assertEqual(corr.backend.to_numpy(out_3d).shape, (3, 1))

        corr_nbins2 = Correlation(
            nside=1,
            phi_center=np.array([0.0]),
            theta_center=np.array([0.0]),
            nbins=2,
            theta_min=1.0,
            theta_max=2.0,
            patch_size=1.0,
            theta_Q=1.0,
            device="cpu",
        )
        out_single_flat = corr_nbins2._normalize_tomo_sumofweights_per_comb(
            np.array([4.0, 5.0], dtype=np.float64),
            1,
        )
        self.assertEqual(corr_nbins2.backend.to_numpy(out_single_flat).shape, (1, 2))

        out_single_2d = corr_nbins2._normalize_tomo_sumofweights_per_comb(
            np.array([[6.0, 7.0]], dtype=np.float64),
            1,
        )
        self.assertEqual(corr_nbins2.backend.to_numpy(out_single_2d).shape, (1, 2))

        corr_two_patches = Correlation(
            nside=1,
            phi_center=np.array([0.0, 1.0]),
            theta_center=np.array([0.0, 0.5]),
            nbins=1,
            theta_min=1.0,
            theta_max=2.0,
            patch_size=1.0,
            theta_Q=1.0,
            device="cpu",
        )
        out_flat_full = corr_two_patches._normalize_tomo_sumofweights_per_comb(
            np.arange(6, dtype=np.float64),
            3,
        )
        self.assertEqual(corr_two_patches.backend.to_numpy(out_flat_full).shape, (3, 2))

        out_single_patchbin = corr_two_patches._normalize_tomo_sumofweights_per_comb(
            np.array([[8.0], [9.0]], dtype=np.float64),
            1,
        )
        self.assertEqual(corr_two_patches.backend.to_numpy(out_single_patchbin).shape, (1, 2))

        with self.assertRaisesRegex(ValueError, "sumofweights must be scalar"):
            corr_nbins2._normalize_tomo_sumofweights_per_comb(
                np.ones((2, 2), dtype=np.float64),
                1,
            )

    def test_normalize_tomo_sumofweights_directional_branches(self):
        corr = Correlation(
            nside=1,
            phi_center=np.array([0.0]),
            theta_center=np.array([0.0]),
            nbins=2,
            theta_min=1.0,
            theta_max=2.0,
            patch_size=1.0,
            theta_Q=1.0,
            device="cpu",
        )

        directional = np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float64)
        out_dir = corr._normalize_tomo_sumofweights_directional(directional, 1)
        self.assertEqual(corr.backend.to_numpy(out_dir).shape, (2, 1, 2))

        out_shared = corr._normalize_tomo_sumofweights_directional(5.0, 1)
        shared_np = corr.backend.to_numpy(out_shared)
        self.assertEqual(shared_np.shape, (2, 1, 2))
        np.testing.assert_allclose(shared_np[0], shared_np[1])

    def test_density_density_tomo_vectorized_prepare_and_missing_kernel(self):
        corr = self._make_small_cpu_corr()
        density_maps = np.ones((2, 12), dtype=np.float64)
        weights = np.ones((2, 12), dtype=np.float64)
        nzbins = 2
        nzbin_combs = int(binom(nzbins + 1, 2))

        corr.inds_dev = None
        with patch.object(corr, "prepare", wraps=corr.prepare) as spy_prepare:
            out = corr._density_density_tomo_vectorized(
                density_maps,
                weights,
                None,
                nzbins,
                False,
            )
            spy_prepare.assert_called_once()
        self.assertEqual(out.shape, (nzbin_combs, corr.n_patches, corr.nbins))

        corr.backend.kernel_density_density_tomo_vectorized = None
        with self.assertRaisesRegex(RuntimeError, "density-density tomography kernel"):
            corr._density_density_tomo_vectorized(
                density_maps,
                weights,
                None,
                nzbins,
                False,
            )

    def test_density_density_tomo_vectorized_non_numpy_launch_paths(self):
        corr = self._make_small_cpu_corr()
        density_maps = np.ones((2, 12), dtype=np.float64)
        weights = np.ones((2, 12), dtype=np.float64)
        nzbins = 2
        nzbin_combs = int(binom(nzbins + 1, 2))

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
            def kernel_density_density_tomo_vectorized(*_args):
                out_num = _args[-1]
                out_num[:] = 2.0
                return True

        corr.backend = FakeBackend()
        out = corr._density_density_tomo_vectorized(
            density_maps,
            weights,
            np.ones((2, nzbin_combs, corr.n_patches, corr.nbins), dtype=np.float64),
            nzbins,
            False,
        )
        self.assertEqual(out.shape, (nzbin_combs, corr.n_patches, corr.nbins))

        corr.backend.kernel_density_density_tomo_vectorized = staticmethod(lambda *_: False)
        with self.assertRaisesRegex(RuntimeError, "declined vectorized density-density"):
            corr._density_density_tomo_vectorized(
                density_maps,
                weights,
                np.ones((2, nzbin_combs, corr.n_patches, corr.nbins), dtype=np.float64),
                nzbins,
                False,
            )

    def test_density_density_tomo_vectorized_auto_only_shape(self):
        corr = self._make_small_cpu_corr()
        density_maps = np.ones((2, 12), dtype=np.float64)
        weights = np.ones((2, 12), dtype=np.float64)
        out = corr._density_density_tomo_vectorized(
            density_maps,
            weights,
            None,
            2,
            True,
        )
        self.assertEqual(out.shape, (2, corr.n_patches, corr.nbins))

    def test_density_shear_tomo_vectorized_prepare_missing_and_non_numpy_paths(self):
        corr = self._make_small_cpu_corr()
        density_maps = np.ones((2, 12), dtype=np.float64)
        shear_maps = np.ones((2, 2, 12), dtype=np.float64)
        density_w = np.ones((2, 12), dtype=np.float64)
        shear_w = np.ones((2, 12), dtype=np.float64)
        nlens_bins = 2
        nsource_bins = 2

        corr.inds_dev = None
        with patch.object(corr, "prepare", wraps=corr.prepare) as spy_prepare:
            out = corr._density_shear_tomo_vectorized(
                density_maps,
                shear_maps,
                density_w,
                shear_w,
                None,
                nlens_bins,
                nsource_bins,
            )
            spy_prepare.assert_called_once()
        self.assertEqual(out.shape, (4, corr.n_patches, corr.nbins))

        corr.backend.kernel_density_shear_tomo_vectorized = None
        with self.assertRaisesRegex(RuntimeError, "density-shear tomography kernel"):
            corr._density_shear_tomo_vectorized(
                density_maps,
                shear_maps,
                density_w,
                shear_w,
                None,
                nlens_bins,
                nsource_bins,
            )

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
            def kernel_density_shear_tomo_vectorized(*_args):
                out_num = _args[-1]
                out_num[:] = 3.0
                return True

        corr = self._make_small_cpu_corr()
        corr.backend = FakeBackend()
        out_dir = corr._density_shear_tomo_vectorized(
            density_maps,
            shear_maps,
            density_w,
            shear_w,
            np.ones((2, 4, corr.n_patches, corr.nbins), dtype=np.float64),
            nlens_bins,
            nsource_bins,
        )
        self.assertEqual(out_dir.shape, (4, corr.n_patches, corr.nbins))

        out_per_comb = corr._density_shear_tomo_vectorized(
            density_maps,
            shear_maps,
            density_w,
            shear_w,
            np.ones(4, dtype=np.float64),
            nlens_bins,
            nsource_bins,
        )
        self.assertEqual(out_per_comb.shape, (4, corr.n_patches, corr.nbins))

        corr.backend.kernel_density_shear_tomo_vectorized = staticmethod(lambda *_: False)
        with self.assertRaisesRegex(RuntimeError, "declined vectorized density-shear"):
            corr._density_shear_tomo_vectorized(
                density_maps,
                shear_maps,
                density_w,
                shear_w,
                None,
                nlens_bins,
                nsource_bins,
            )

    def test_vectorized_shear_shear_sumofweights_validation_errors(self):
        corr = self._make_small_cpu_corr()
        shear_maps = np.ones((2, 2, 12), dtype=np.float64)
        w = np.ones((2, 12), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "at least two dimensions"):
            corr.vectorized_shear_shear(
                shear_maps,
                w,
                sumofweights=np.array([1.0], dtype=np.float64),
            )

        with self.assertRaisesRegex(ValueError, "first dimensions"):
            corr.vectorized_shear_shear(
                shear_maps,
                w,
                sumofweights=np.ones((1, 3, 1), dtype=np.float64),
            )

    def test_get_3x2pt_tomo_raises_when_no_maps(self):
        corr = self._make_small_cpu_corr()
        with self.assertRaisesRegex(ValueError, "Both shear_maps and density_maps"):
            corr.get_3x2pt_tomo()

    def test_get_3x2pt_tomo_raises_when_only_one_map_is_provided(self):
        corr = self._make_small_cpu_corr()
        shear_maps = np.ones((1, 2, 12), dtype=np.float64)
        density_maps = np.ones((1, 12), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "Both shear_maps and density_maps"):
            corr.get_3x2pt_tomo(shear_maps=shear_maps)

        with self.assertRaisesRegex(ValueError, "Both shear_maps and density_maps"):
            corr.get_3x2pt_tomo(density_maps=density_maps)

    def test_get_3x2pt_tomo_both_maps_weights_variants(self):
        corr = self._make_small_cpu_corr()
        shear_maps = np.ones((2, 2, 12), dtype=np.float64)
        density_maps = np.ones((2, 12), dtype=np.float64)
        shear_w = np.full((2, 12), 2.0, dtype=np.float64)
        density_w = np.full((2, 12), 3.0, dtype=np.float64)
        fused_return = (
            np.full((2, corr.n_patches), 1.0, dtype=np.float64),
            np.full((2, corr.n_patches), 5.0, dtype=np.float64),
            np.full((3, corr.n_patches, corr.nbins), 2.0, dtype=np.float64),
            np.zeros((3, corr.n_patches, corr.nbins), dtype=np.float64),
            np.full((3, corr.n_patches, corr.nbins), 6.0, dtype=np.float64),
            np.full((4, corr.n_patches, corr.nbins), 7.0, dtype=np.float64),
        )

        with patch.object(
            corr,
            "_compute_3x2pt_tomo_fused",
            return_value=fused_return,
        ) as spy_fused:
            out_dict = corr.get_3x2pt_tomo(
                shear_maps=shear_maps,
                density_maps=density_maps,
                weights={"shear": shear_w, "density": density_w},
            )
            out_tuple = corr.get_3x2pt_tomo(
                shear_maps=shear_maps,
                density_maps=density_maps,
                weights=(shear_w, density_w),
            )
            out_shared = corr.get_3x2pt_tomo(
                shear_maps=shear_maps,
                density_maps=density_maps,
                weights=np.ones((2, 12), dtype=np.float64),
            )

        self.assertEqual(len(out_dict), 6)
        self.assertEqual(len(out_tuple), 6)
        self.assertEqual(len(out_shared), 6)
        self.assertEqual(spy_fused.call_count, 3)

        with self.assertRaisesRegex(
            ValueError,
            r"weights must be a dict or \(shear_weights, density_weights\)",
        ):
            corr.get_3x2pt_tomo(
                shear_maps=shear_maps,
                density_maps=density_maps,
                weights=np.ones((3, 12), dtype=np.float64),
            )

    def test_get_3x2pt_tomo_forwards_flip_flags_to_wl_and_ggl(self):
        corr = self._make_small_cpu_corr()
        shear_maps = np.ones((2, 2, 12), dtype=np.float64)
        density_maps = np.ones((2, 12), dtype=np.float64)
        shear_w = np.full((2, 12), 2.0, dtype=np.float64)
        density_w = np.full((2, 12), 3.0, dtype=np.float64)

        selected = [(1, 0)]

        with patch.object(
            corr,
            "_compute_3x2pt_tomo_fused",
            return_value=(
                np.zeros((2, corr.n_patches), dtype=np.float64),
                np.zeros((2, corr.n_patches), dtype=np.float64),
                np.zeros((3, corr.n_patches, corr.nbins), dtype=np.float64),
                np.zeros((3, corr.n_patches, corr.nbins), dtype=np.float64),
                np.zeros((3, corr.n_patches, corr.nbins), dtype=np.float64),
                np.zeros((3, corr.n_patches, corr.nbins), dtype=np.float64),
            ),
        ) as spy_fused:
            _ = corr.get_3x2pt_tomo(
                shear_maps=shear_maps,
                density_maps=density_maps,
                weights={"shear": shear_w, "density": density_w},
                ggl_bin_combinations=selected,
                flip_g1=True,
                flip_g2=True,
            )

        spy_fused.assert_called_once()
        self.assertTrue(spy_fused.call_args.kwargs["flip_g1"])
        self.assertTrue(spy_fused.call_args.kwargs["flip_g2"])
        self.assertEqual(spy_fused.call_args.kwargs["ggl_bin_combinations"], selected)

    def test_get_3x2pt_tomo_forwards_gc_auto_only_to_density(self):
        corr = self._make_small_cpu_corr()
        shear_maps = np.ones((2, 2, 12), dtype=np.float64)
        density_maps = np.ones((2, 12), dtype=np.float64)
        shear_w = np.full((2, 12), 2.0, dtype=np.float64)
        density_w = np.full((2, 12), 3.0, dtype=np.float64)

        with patch.object(
            corr,
            "_compute_3x2pt_tomo_fused",
            return_value=(
                np.full((2, corr.n_patches), 1.0, dtype=np.float64),
                np.full((2, corr.n_patches), 5.0, dtype=np.float64),
                np.full((3, corr.n_patches, corr.nbins), 2.0, dtype=np.float64),
                np.zeros((3, corr.n_patches, corr.nbins), dtype=np.float64),
                np.full((2, corr.n_patches, corr.nbins), 6.0, dtype=np.float64),
                np.full((4, corr.n_patches, corr.nbins), 7.0, dtype=np.float64),
            ),
        ) as spy_fused:
            _ = corr.get_3x2pt_tomo(
                shear_maps=shear_maps,
                density_maps=density_maps,
                weights={"shear": shear_w, "density": density_w},
                gc_auto_correlations_only=True,
            )

        spy_fused.assert_called_once()
        self.assertTrue(spy_fused.call_args.kwargs["gc_auto_correlations_only"])

    def test_get_3x2pt_tomo_forwards_aperture_filter(self):
        corr = self._make_small_cpu_corr()
        shear_maps = np.ones((2, 2, 12), dtype=np.float64)
        density_maps = np.ones((2, 12), dtype=np.float64)
        shear_w = np.full((2, 12), 2.0, dtype=np.float64)
        density_w = np.full((2, 12), 3.0, dtype=np.float64)

        def custom_filter(theta, _theta_Q):
            return np.ones_like(theta)

        with patch.object(
            corr,
            "_compute_3x2pt_tomo_fused",
            return_value=(
                np.full((2, corr.n_patches), 1.0, dtype=np.float64),
                np.full((2, corr.n_patches), 5.0, dtype=np.float64),
                np.full((3, corr.n_patches, corr.nbins), 2.0, dtype=np.float64),
                np.zeros((3, corr.n_patches, corr.nbins), dtype=np.float64),
                np.full((3, corr.n_patches, corr.nbins), 6.0, dtype=np.float64),
                np.full((4, corr.n_patches, corr.nbins), 7.0, dtype=np.float64),
            ),
        ) as spy_fused:
            _ = corr.get_3x2pt_tomo(
                shear_maps=shear_maps,
                density_maps=density_maps,
                weights={"shear": shear_w, "density": density_w},
                aperture_filter=custom_filter,
            )

        self.assertIs(spy_fused.call_args.kwargs["aperture_filter"], custom_filter)

    def test_get_3x2pt_tomo_both_maps_with_default_weights_none(self):
        corr = self._make_small_cpu_corr()
        shear_maps = np.ones((1, 2, 12), dtype=np.float64)
        density_maps = np.ones((1, 12), dtype=np.float64)

        with patch.object(
            corr,
            "_compute_3x2pt_tomo_fused",
            return_value=(
                np.full((1, corr.n_patches), 1.0, dtype=np.float64),
                np.full((1, corr.n_patches), 5.0, dtype=np.float64),
                np.full((1, corr.n_patches, corr.nbins), 2.0, dtype=np.float64),
                np.zeros((1, corr.n_patches, corr.nbins), dtype=np.float64),
                np.full((1, corr.n_patches, corr.nbins), 6.0, dtype=np.float64),
                np.full((1, corr.n_patches, corr.nbins), 7.0, dtype=np.float64),
            ),
        ) as spy_fused:
            outputs = corr.get_3x2pt_tomo(
                shear_maps=shear_maps,
                density_maps=density_maps,
                weights=None,
            )

        self.assertEqual(len(outputs), 6)
        self.assertTrue(np.all(spy_fused.call_args.kwargs["shear_weights"] == 1.0))
        self.assertTrue(np.all(spy_fused.call_args.kwargs["density_weights"] == 1.0))

    def test_get_3x2pt_tomo_dict_missing_keys_falls_back_to_default_weights(self):
        corr = self._make_small_cpu_corr()
        shear_maps = np.ones((1, 2, 12), dtype=np.float64)
        density_maps = np.ones((1, 12), dtype=np.float64)

        with patch.object(
            corr,
            "_compute_3x2pt_tomo_fused",
            return_value=(
                np.full((1, corr.n_patches), 1.0, dtype=np.float64),
                np.full((1, corr.n_patches), 5.0, dtype=np.float64),
                np.full((1, corr.n_patches, corr.nbins), 2.0, dtype=np.float64),
                np.zeros((1, corr.n_patches, corr.nbins), dtype=np.float64),
                np.full((1, corr.n_patches, corr.nbins), 6.0, dtype=np.float64),
                np.full((1, corr.n_patches, corr.nbins), 7.0, dtype=np.float64),
            ),
        ) as spy_fused:
            outputs = corr.get_3x2pt_tomo(
                shear_maps=shear_maps,
                density_maps=density_maps,
                weights={},
            )

        self.assertEqual(len(outputs), 6)
        self.assertTrue(np.all(spy_fused.call_args.kwargs["shear_weights"] == 1.0))
        self.assertTrue(np.all(spy_fused.call_args.kwargs["density_weights"] == 1.0))

    def test_get_3x2pt_tomo_fused_with_shared_weights_input(self):
        corr = self._make_small_cpu_corr()
        shear_maps = np.ones((2, 2, 12), dtype=np.float64)
        density_maps = np.ones((2, 12), dtype=np.float64)
        shared_w = np.full((2, 12), 9.0, dtype=np.float64)

        with patch.object(
            corr,
            "_compute_3x2pt_tomo_fused",
            return_value=(
                np.zeros((2, corr.n_patches), dtype=np.float64),
                np.zeros((2, corr.n_patches), dtype=np.float64),
                np.zeros((3, corr.n_patches, corr.nbins), dtype=np.float64),
                np.zeros((3, corr.n_patches, corr.nbins), dtype=np.float64),
                np.zeros((3, corr.n_patches, corr.nbins), dtype=np.float64),
                np.zeros((4, corr.n_patches, corr.nbins), dtype=np.float64),
            ),
        ) as spy_fused:
            _ = corr.get_3x2pt_tomo(
                shear_maps=shear_maps,
                density_maps=density_maps,
                weights=shared_w,
            )

        np.testing.assert_array_equal(spy_fused.call_args.kwargs["shear_weights"], shared_w)
        np.testing.assert_array_equal(spy_fused.call_args.kwargs["density_weights"], shared_w)

    def test_get_3x2pt_tomo_matches_component_methods_cpu(self):
        corr = self._make_small_cpu_corr()
        rng = np.random.default_rng(1234)

        shear_maps = rng.normal(size=(2, 2, 12)).astype(np.float64)
        density_maps = rng.normal(size=(2, 12)).astype(np.float64)
        shear_w = (1.0 + rng.random(size=(2, 12))).astype(np.float64)
        density_w = (1.0 + rng.random(size=(2, 12))).astype(np.float64)
        selected = [(0, 1), (1, 0)]

        M_a_ref, xi_p_ref, xi_m_ref = corr.get_full_tomo_shear(
            shear_maps,
            shear_w,
            flip_g1=True,
        )
        M_g_ref, xi_g_ref = corr.get_full_tomo_density(
            density_maps,
            density_w,
            gc_auto_correlations_only=True,
        )
        xi_t_ref = corr.vectorized_density_shear(
            density_maps,
            shear_maps,
            density_w,
            shear_w,
            ggl_bin_combinations=selected,
            flip_g1=True,
        )

        M_a, M_g, xi_p, xi_m, xi_g, xi_t = corr.get_3x2pt_tomo(
            shear_maps=shear_maps,
            density_maps=density_maps,
            weights={"shear": shear_w, "density": density_w},
            gc_auto_correlations_only=True,
            ggl_bin_combinations=selected,
            flip_g1=True,
        )

        np.testing.assert_allclose(M_a, M_a_ref, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(M_g, M_g_ref, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(xi_p, xi_p_ref, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(xi_m, xi_m_ref, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(xi_g, xi_g_ref, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(xi_t, xi_t_ref, rtol=1e-7, atol=1e-9)

    def test_compute_3x2pt_tomo_fused_missing_kernel_raises(self):
        corr = self._make_small_cpu_corr()
        corr.backend.kernel_3x2pt_tomo_fused = None

        with self.assertRaisesRegex(RuntimeError, "fused 3x2pt tomography kernel"):
            corr._compute_3x2pt_tomo_fused(
                shear_maps=np.ones((1, 2, 12), dtype=np.float64),
                density_maps=np.ones((1, 12), dtype=np.float64),
                shear_weights=np.ones((1, 12), dtype=np.float64),
                density_weights=np.ones((1, 12), dtype=np.float64),
            )

    def test_compute_3x2pt_tomo_fused_validation_errors(self):
        corr = self._make_small_cpu_corr()

        with self.assertRaisesRegex(ValueError, "density_maps must have shape"):
            corr._compute_3x2pt_tomo_fused(
                shear_maps=np.ones((1, 2, 12), dtype=np.float64),
                density_maps=np.ones((12,), dtype=np.float64),
                shear_weights=np.ones((1, 12), dtype=np.float64),
                density_weights=np.ones((1, 12), dtype=np.float64),
            )

        with self.assertRaisesRegex(ValueError, "shear_maps must have shape"):
            corr._compute_3x2pt_tomo_fused(
                shear_maps=np.ones((1, 12), dtype=np.float64),
                density_maps=np.ones((1, 12), dtype=np.float64),
                shear_weights=np.ones((1, 12), dtype=np.float64),
                density_weights=np.ones((1, 12), dtype=np.float64),
            )

        with self.assertRaisesRegex(ValueError, "density_weights must match density_maps"):
            corr._compute_3x2pt_tomo_fused(
                shear_maps=np.ones((1, 2, 12), dtype=np.float64),
                density_maps=np.ones((1, 12), dtype=np.float64),
                shear_weights=np.ones((1, 12), dtype=np.float64),
                density_weights=np.ones((2, 12), dtype=np.float64),
            )

        with self.assertRaisesRegex(ValueError, "shear_weights must have shape"):
            corr._compute_3x2pt_tomo_fused(
                shear_maps=np.ones((1, 2, 12), dtype=np.float64),
                density_maps=np.ones((1, 12), dtype=np.float64),
                shear_weights=np.ones((1, 1, 12), dtype=np.float64),
                density_weights=np.ones((1, 12), dtype=np.float64),
            )

        with self.assertRaisesRegex(ValueError, "shear_weights must match shear_maps"):
            corr._compute_3x2pt_tomo_fused(
                shear_maps=np.ones((1, 2, 12), dtype=np.float64),
                density_maps=np.ones((1, 12), dtype=np.float64),
                shear_weights=np.ones((2, 12), dtype=np.float64),
                density_weights=np.ones((1, 12), dtype=np.float64),
            )

        with self.assertRaisesRegex(ValueError, "must have the same number of pixels"):
            corr._compute_3x2pt_tomo_fused(
                shear_maps=np.ones((1, 2, 12), dtype=np.float64),
                density_maps=np.ones((1, 11), dtype=np.float64),
                shear_weights=np.ones((1, 12), dtype=np.float64),
                density_weights=np.ones((1, 11), dtype=np.float64),
            )

    def test_compute_3x2pt_tomo_fused_prepare_and_flip_g2_branch(self):
        corr = self._make_small_cpu_corr()
        corr.inds_dev = None
        captured = {}

        def fake_kernel(*args):
            captured["shear"] = np.array(args[1], copy=True)

        corr.backend.kernel_3x2pt_tomo_fused = fake_kernel

        shear_maps = np.zeros((1, 2, 12), dtype=np.float64)
        shear_maps[:, 0] = 2.0
        shear_maps[:, 1] = 3.0

        with patch.object(corr, "prepare", wraps=corr.prepare) as spy_prepare:
            corr._compute_3x2pt_tomo_fused(
                shear_maps=shear_maps,
                density_maps=np.ones((1, 12), dtype=np.float64),
                shear_weights=np.ones((1, 12), dtype=np.float64),
                density_weights=np.ones((1, 12), dtype=np.float64),
                flip_g2=True,
            )

        self.assertTrue(spy_prepare.called)
        self.assertIn("shear", captured)
        np.testing.assert_allclose(captured["shear"][:, 0, 1], -3.0)

    def test_compute_3x2pt_tomo_fused_xig_cross_numpy_and_non_numpy_paths(self):
        corr = self._make_small_cpu_corr()

        density_maps = np.array(
            [
                [2.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [3.0, 7.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        density_w = np.array(
            [
                [11.0, 13.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [17.0, 19.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        shear_maps = np.zeros((1, 2, 12), dtype=np.float64)
        shear_w = np.ones((1, 12), dtype=np.float64)

        out_np = corr._compute_3x2pt_tomo_fused(
            shear_maps=shear_maps,
            density_maps=density_maps,
            shear_weights=shear_w,
            density_weights=density_w,
            gc_auto_correlations_only=False,
        )

        xi_g_np = out_np[4]
        pairs = corr.pair_inds[0]
        ab_den = 0.0
        ba_den = 0.0
        ab_num = 0.0
        ba_num = 0.0
        for pix_a, pix_b in pairs:
            w_ab = density_w[0, pix_a] * density_w[1, pix_b]
            w_ba = density_w[1, pix_a] * density_w[0, pix_b]
            ab_den += w_ab
            ba_den += w_ba
            ab_num += w_ab * density_maps[0, pix_a] * density_maps[1, pix_b]
            ba_num += w_ba * density_maps[1, pix_a] * density_maps[0, pix_b]

        expected_numpy = (ab_num + ba_num) / (ab_den + ba_den)
        self.assertAlmostEqual(xi_g_np[1, 0, 0], expected_numpy)

        original_name = corr.backend.name
        original_kernel = corr.backend.kernel_3x2pt_tomo_fused
        original_to_device = corr.backend.to_device
        original_to_numpy = corr.backend.to_numpy
        try:
            corr.backend.name = "cupy"
            corr.backend.to_device = lambda arr: np.asarray(arr)
            corr.backend.to_numpy = lambda arr: np.asarray(arr)

            def launched_kernel(*args):
                original_kernel(*args)
                return True

            corr.backend.kernel_3x2pt_tomo_fused = launched_kernel
            out_fake_gpu = corr._compute_3x2pt_tomo_fused(
                shear_maps=shear_maps,
                density_maps=density_maps,
                shear_weights=shear_w,
                density_weights=density_w,
                gc_auto_correlations_only=False,
            )
        finally:
            corr.backend.name = original_name
            corr.backend.kernel_3x2pt_tomo_fused = original_kernel
            corr.backend.to_device = original_to_device
            corr.backend.to_numpy = original_to_numpy

        xi_g_fake_gpu = out_fake_gpu[4]
        expected_non_numpy = 0.5 * ((ab_num / ab_den) + (ba_num / ba_den))
        self.assertAlmostEqual(xi_g_fake_gpu[1, 0, 0], expected_non_numpy)

    def test_density_density_tomo_vectorized_gc_auto_cross_directional_sum_branch(self):
        corr = self._make_small_cpu_corr()
        density_maps = np.array(
            [
                np.linspace(1.0, 2.1, 12, dtype=np.float64),
                np.linspace(2.2, 3.3, 12, dtype=np.float64),
            ]
        )
        weights = np.array(
            [
                np.linspace(1.0, 1.5, 12, dtype=np.float64),
                np.linspace(1.6, 2.1, 12, dtype=np.float64),
            ]
        )

        comb_i = np.array([0], dtype=np.int32)
        comb_j = np.array([1], dtype=np.int32)
        auto = np.array([False], dtype=bool)
        with patch.object(
            corr,
            "_get_selected_tomo_density_combination_indices",
            return_value=(comb_i, comb_j, auto, 1),
        ), patch.object(corr, "_reduce_pairs", wraps=corr._reduce_pairs) as spy_reduce:
            out = corr._density_density_tomo_vectorized(
                density_maps,
                weights,
                sumofweights=None,
                nzbins=2,
                gc_auto_correlations_only=True,
            )

        self.assertEqual(out.shape, (1, corr.n_patches, corr.nbins))
        self.assertGreaterEqual(spy_reduce.call_count, 2)

    def test_compute_3x2pt_tomo_fused_non_numpy_launch_decline_raises(self):
        corr = self._make_small_cpu_corr()
        corr.backend.name = "cupy"
        corr.backend.to_device = lambda arr: np.asarray(arr)
        corr.backend.to_numpy = lambda arr: np.asarray(arr)
        corr.backend.kernel_3x2pt_tomo_fused = lambda *_args: False

        with self.assertRaisesRegex(RuntimeError, "declined fused 3x2pt tomography kernel launch"):
            corr._compute_3x2pt_tomo_fused(
                shear_maps=np.ones((1, 2, 12), dtype=np.float64),
                density_maps=np.ones((1, 12), dtype=np.float64),
                shear_weights=np.ones((1, 12), dtype=np.float64),
                density_weights=np.ones((1, 12), dtype=np.float64),
            )

    def test_compute_aperture_shear_all_patches_helper(self):
        vals = correlations_module._compute_aperture_shear_all_patches(
            np.array([0, 1], dtype=np.uint32),
            np.array([1.0, 1.0], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([2.0, 2.0], dtype=np.float64),
            np.array([0, 2], dtype=np.int64),
            np.array([1.0, 3.0], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([2.0, 4.0], dtype=np.float64),
            np.array([5.0], dtype=np.float64),
        )
        gt = np.array([-1.0, -3.0])
        expected = 5.0 * np.sum(np.array([2.0, 4.0]) * gt * 2.0) / np.sum(np.array([2.0, 4.0]))
        self.assertAlmostEqual(vals[0], expected)

    def test_compute_density_methods_non_numpy_backend_path(self):
        corr = self._make_small_cpu_corr()

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
            def kernel_density_density(d1, d2, w1, w2, ind_i, ind_j, out):
                out[:] = w1[ind_i] * w2[ind_j] * d1[ind_i] * d2[ind_j]

            @staticmethod
            def kernel_density_shear(d_l, g1_s, g2_s, w_l, w_s, ind_i, ind_j, exp_j, out):
                out[:] = (
                    w_l[ind_i]
                    * w_s[ind_j]
                    * d_l[ind_i]
                    * (g1_s[ind_j] * exp_j.real + g2_s[ind_j] * exp_j.imag)
                )

        corr.backend = FakeBackend()
        d1 = np.ones(12, dtype=np.float64)
        d2 = np.ones(12, dtype=np.float64)
        g1 = np.ones(12, dtype=np.float64)
        g2 = np.zeros(12, dtype=np.float64)
        w = np.ones(12, dtype=np.float64)

        (wtheta,) = corr.compute_density_density(d1, d2, w, w, sumofweights=1.0)
        (gamma_t,) = corr.compute_density_shear(d1, g1, g2, w, w, sumofweights=1.0)

        self.assertEqual(wtheta.shape, (corr.n_patches, corr.nbins))
        self.assertEqual(gamma_t.shape, (corr.n_patches, corr.nbins))

    def test_xipm_cross_non_numpy_backend_path(self):
        corr = self._make_small_cpu_corr()
        corr.backend.name = "other"

        def fake_kernel(
            _g11,
            _g21,
            _g12,
            _g22,
            _w1,
            _w2,
            _ind_i,
            _ind_j,
            _exp_i,
            _exp_j,
            out_ab_p,
            out_ab_m,
            out_ba_p,
            out_ba_m,
        ):
            out_ab_p[...] = 1.0 + 0.0j
            out_ab_m[...] = 2.0 + 0.0j
            out_ba_p[...] = 3.0 + 0.0j
            out_ba_m[...] = 4.0 + 0.0j

        corr.backend.xipm_cross_corr_kernel = fake_kernel
        g11 = np.ones(12, dtype=np.float64)
        g21 = np.ones(12, dtype=np.float64)
        g12 = np.ones(12, dtype=np.float64)
        g22 = np.ones(12, dtype=np.float64)
        w1 = np.ones(12, dtype=np.float64)
        w2 = np.ones(12, dtype=np.float64)

        with patch.object(corr, "_reduce_pairs", wraps=corr._reduce_pairs) as spy_reduce:
            xip, xim = corr._xipm_cross(
                g11,
                g21,
                g12,
                g22,
                w1,
                w2,
                sumofweights_ab=1.0,
                sumofweights_ba=1.0,
            )

        self.assertEqual(spy_reduce.call_count, 4)
        np.testing.assert_allclose(xip, np.array([[4.0]], dtype=np.float64))
        np.testing.assert_allclose(xim, np.array([[6.0]], dtype=np.float64))

    def test_xipm_cross_missing_kernel_raises(self):
        corr = self._make_small_cpu_corr()
        corr.backend.xipm_cross_corr_kernel = None

        g11 = np.ones(12, dtype=np.float64)
        g21 = np.ones(12, dtype=np.float64)
        g12 = np.ones(12, dtype=np.float64)
        g22 = np.ones(12, dtype=np.float64)
        w1 = np.ones(12, dtype=np.float64)
        w2 = np.ones(12, dtype=np.float64)

        with self.assertRaises(RuntimeError):
            corr._xipm_cross(g11, g21, g12, g22, w1, w2)

    def test_reduce_pairs_2d_axis_typeerror_fallback(self):
        corr = self._make_small_cpu_corr()
        original_add = corr.backend.add

        class _AddProxy:
            def __init__(self, add_ufunc):
                self._add = add_ufunc

            def reduceat(self, arr, starts, axis=None):
                if axis is not None:
                    raise TypeError("axis argument unsupported")
                return self._add.reduceat(arr, starts)

        corr.backend.add = _AddProxy(original_add)
        try:
            values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
            reduced = corr._reduce_pairs(values)
        finally:
            corr.backend.add = original_add

        expected = np.array([[3.0], [7.0]], dtype=np.float64)
        np.testing.assert_allclose(reduced, expected)

    def test_reduce_pairs_2d_partial_valid_axis_typeerror_fallback(self):
        corr = Correlation(
            nside=1,
            phi_center=np.array([0.0]),
            theta_center=np.array([0.0]),
            nbins=2,
            theta_min=1.0,
            theta_max=2.0,
            patch_size=1.0,
            theta_Q=1.0,
            device="cpu",
        )
        corr.pair_inds = [np.array([[0], [1]], dtype=np.uint32)]
        corr.pair_exp2phi = [np.ones((2, 1), dtype=np.complex128)]
        corr.bins = [np.array([1, 0], dtype=np.uint32)]
        corr.prepare()
        original_add = corr.backend.add

        class _AddProxy:
            def __init__(self, add_ufunc):
                self._add = add_ufunc

            def reduceat(self, arr, starts, axis=None):
                if axis is not None:
                    raise TypeError("axis argument unsupported")
                return self._add.reduceat(arr, starts)

        corr.backend.add = _AddProxy(original_add)
        try:
            values = np.array([[5.0], [7.0]], dtype=np.float64)
            reduced = corr.backend.to_numpy(corr._reduce_pairs(values))
        finally:
            corr.backend.add = original_add

        expected = np.array([[5.0, 0.0], [7.0, 0.0]], dtype=np.float64)
        np.testing.assert_allclose(reduced, expected)

    def test_xipm_tomo_vectorized_returns_none_when_backend_kernel_missing(self):
        corr = self._make_small_cpu_corr()
        corr.backend.xipm_tomo_vectorized_kernel = None
        result = corr._xipm_tomo_vectorized(None, None, None, 2, 3, 1, 1)
        self.assertIsNone(result)

    def test_xipm_tomo_vectorized_returns_none_when_backend_kernel_declines(self):
        corr = self._make_small_cpu_corr()
        corr.backend.name = "cupy"
        corr.backend.module = np
        corr.backend.xipm_tomo_vectorized_kernel = lambda *_args: False

        shear_maps_dev = np.ones((2, 2, 12), dtype=np.float64)
        w_dev = np.ones((2, 12), dtype=np.float64)
        sumofweights_dev = np.zeros((2, 3, 1), dtype=np.float64)
        result = corr._xipm_tomo_vectorized(
            shear_maps_dev,
            w_dev,
            sumofweights_dev,
            2,
            3,
            1,
            1,
        )
        self.assertIsNone(result)

    def test_xipm_tomo_vectorized_returns_none_for_unknown_backend_name(self):
        corr = self._make_small_cpu_corr()
        corr.backend.name = "other"
        corr.backend.xipm_tomo_vectorized_kernel = lambda *_args: True
        result = corr._xipm_tomo_vectorized(None, None, None, 2, 3, 1, 1)
        self.assertIsNone(result)

    def test_xipm_tomo_vectorized_cpu_raises_when_kernel_declines(self):
        corr = self._make_small_cpu_corr()
        corr.backend.xipm_tomo_vectorized_kernel = lambda *_args: False
        shear_maps_dev = np.ones((2, 2, 12), dtype=np.float64)
        w_dev = np.ones((2, 12), dtype=np.float64)
        sumofweights_dev = np.ones((2, 3, 1), dtype=np.float64)

        with self.assertRaises(RuntimeError):
            corr._xipm_tomo_vectorized(
                shear_maps_dev,
                w_dev,
                sumofweights_dev,
                2,
                3,
                1,
                1,
            )

    def test_vectorized_shear_shear_raises_when_vectorized_unavailable(self):
        corr = self._make_small_cpu_corr()
        shear_maps = np.ones((2, 2, 12), dtype=np.float64)
        w = np.ones((2, 12), dtype=np.float64)

        with patch.object(corr, "_xipm_tomo_vectorized", return_value=None):
            with self.assertRaises(RuntimeError):
                corr.vectorized_shear_shear(shear_maps, w)

    def test_vectorized_shear_shear_fused_cpu_assignments(self):
        """CPU vectorized_shear_shear should use vectorized tomography, not legacy cross kernel."""
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

        corr.backend.xipm_cross_corr_kernel = fused_kernel

        xip, xim = corr.vectorized_shear_shear(shear_maps, w, sumofweights=sumofweights)

        self.assertFalse(called["value"])

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
        mock_calculate_pairs_2PCF.assert_called_once_with(angle_method="haversine")
        mock_prepare.assert_called_once_with()

    @patch('CosmoFuse.correlations.Correlation.calculate_pairs_M_a')
    @patch('CosmoFuse.correlations.Correlation.calculate_pairs_2PCF')
    @patch('CosmoFuse.correlations.Correlation.prepare')
    def test_preprocess_with_aperture_filter(
        self, mock_prepare, mock_calculate_pairs_2PCF, mock_calculate_pairs_M_a
    ):
        def custom_filter(theta, theta_q):
            return np.ones_like(theta) * theta_q

        self.corr.preprocess(aperture_filter=custom_filter)
        mock_calculate_pairs_M_a.assert_called_once()
        self.assertIs(mock_calculate_pairs_M_a.call_args.kwargs["aperture_filter"], custom_filter)
        mock_calculate_pairs_2PCF.assert_called_once_with(angle_method="haversine")
        mock_prepare.assert_called_once_with()

    @patch('CosmoFuse.correlations.Correlation.calculate_pairs_M_a')
    @patch('CosmoFuse.correlations.Correlation.calculate_pairs_2PCF')
    @patch('CosmoFuse.correlations.Correlation.prepare')
    def test_preprocess_forwards_angle_method(
        self, mock_prepare, mock_calculate_pairs_2PCF, mock_calculate_pairs_M_a
    ):
        self.corr.preprocess(angle_method="law")
        mock_calculate_pairs_M_a.assert_called_once_with()
        mock_calculate_pairs_2PCF.assert_called_once_with(angle_method="law")
        mock_prepare.assert_called_once_with()

    @patch('CosmoFuse.correlations.Correlation.calculate_pairs_M_a')
    @patch('CosmoFuse.correlations.Correlation.calculate_pairs_2PCF')
    @patch('CosmoFuse.correlations.Correlation.prepare')
    def test_precompute_alias_forwards_angle_method(
        self, mock_prepare, mock_calculate_pairs_2PCF, mock_calculate_pairs_M_a
    ):
        self.corr.precompute(angle_method="arccos")
        mock_calculate_pairs_M_a.assert_called_once_with()
        mock_calculate_pairs_2PCF.assert_called_once_with(angle_method="arccos")
        mock_prepare.assert_called_once_with()

    def test_vectorized_shear_shear(self):
        """Test the vectorized_shear_shear method."""
        nzbins = 2
        nzbin_combs = int(binom(nzbins + 1, 2))
        
        shear_maps = np.random.rand(nzbins, 2, self.npix)
        w = np.random.rand(nzbins, self.npix)
        self._setup_mock_pairs()
        
        xip, xim = self.corr.vectorized_shear_shear(shear_maps, w)

        self.assertEqual(xip.shape, (nzbin_combs, self.corr.n_patches, self.corr.nbins))
        self.assertEqual(xim.shape, (nzbin_combs, self.corr.n_patches, self.corr.nbins))

        # Test with flips
        xip_f, xim_f = self.corr.vectorized_shear_shear(
            shear_maps, w, flip_g1=True, flip_g2=True
        )
        self.assertTrue(np.allclose(xip, xip_f))
        # xim should be different with flips
        # self.assertTrue(np.allclose(xim, xim_f))

        M_ap, _xip_full, _xim_full = self.corr.get_full_tomo_shear(shear_maps, w)
        M_ap_f, _xip_full_f, _xim_full_f = self.corr.get_full_tomo_shear(
            shear_maps, w, flip_g1=True, flip_g2=True
        )
        self.assertTrue(np.allclose(M_ap, -M_ap_f))

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

    def test_vectorized_shear_shear_missing_fused_kernel(self):
        """vectorized_shear_shear should not depend on legacy fused cross kernel."""
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

        corr.backend.xipm_cross_corr_kernel = None

        shear_maps = np.ones((2, 2, 12), dtype=np.float64)
        w = np.ones((2, 12), dtype=np.float64)

        xip, xim = corr.vectorized_shear_shear(shear_maps, w)
        self.assertEqual(xip.shape, (3, corr.n_patches, corr.nbins))
        self.assertEqual(xim.shape, (3, corr.n_patches, corr.nbins))

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

        corr.backend.xipm_auto_corr_kernel = None

        g11 = np.ones(12, dtype=np.float64)
        g21 = np.ones(12, dtype=np.float64)
        g12 = g11
        g22 = g21
        w1 = np.ones(12, dtype=np.float64)
        w2 = w1

        with self.assertRaises(RuntimeError):
            corr.compute_shear_shear(g11, g21, g12, g22, w1, w2)

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

        xip, xim = corr.compute_shear_shear(g11, g21, g12, g22, w1, w2)

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
            xip, xim = low.compute_shear_shear(g11, g21, g12, g22, w1, w2)
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
        xip_auto, xim_auto = self.corr.compute_shear_shear(g11, g21, g12, g22, w1, w2)
        xip_explicit, xim_explicit = self.corr.compute_shear_shear(
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
            self.corr.compute_shear_shear(g11, g21, g12, g22, w1, w2)
            self.corr.compute_shear_shear(g11, g21, g12, g22, w1, w2)
            self.assertEqual(spy_compute.call_count, 1)

    def test_xipm_cross_uses_sumofweights_getter(self):
        self._setup_mock_pairs()
        g11 = np.random.rand(self.npix)
        g21 = np.random.rand(self.npix)
        g12 = np.random.rand(self.npix)
        g22 = np.random.rand(self.npix)
        w1 = np.random.rand(self.npix)
        w2 = np.random.rand(self.npix)

        with patch.object(
            self.corr,
            "_get_xipm_sumofweights",
            wraps=self.corr._get_xipm_sumofweights,
        ) as spy_get:
            self.corr.compute_shear_shear(g11, g21, g12, g22, w1, w2)

        self.assertEqual(spy_get.call_count, 2)

    def test_xipm_cross_sumofweights_reuses_cache(self):
        self._setup_mock_pairs()
        g11 = np.random.rand(self.npix)
        g21 = np.random.rand(self.npix)
        g12 = np.random.rand(self.npix)
        g22 = np.random.rand(self.npix)
        w1 = np.random.rand(self.npix)
        w2 = np.random.rand(self.npix)

        with patch.object(
            self.corr,
            "_compute_xipm_sumofweights",
            wraps=self.corr._compute_xipm_sumofweights,
        ) as spy_compute:
            self.corr.compute_shear_shear(g11, g21, g12, g22, w1, w2)
            self.corr.compute_shear_shear(g11, g21, g12, g22, w1, w2)

        self.assertEqual(spy_compute.call_count, 2)
        self.assertIsInstance(self.corr._xipm_sumofweights_cache, dict)
        self.assertEqual(len(self.corr._xipm_sumofweights_cache), 2)

    def test_xipm_sumofweights_legacy_single_entry_cache_migrates(self):
        self._setup_mock_pairs()
        w1 = np.random.rand(self.npix)
        w2 = np.random.rand(self.npix)
        w1_dev = self.corr.backend.to_device(w1)
        w2_dev = self.corr.backend.to_device(w2)
        legacy_key = (
            self.corr._fingerprint_weights(np.asarray(w1)),
            self.corr._fingerprint_weights(np.asarray(w2)),
        )
        legacy_value = self.corr.backend.to_device(
            np.ones(self.corr.n_patches * self.corr.nbins, dtype=self.corr.map_dtype)
        )
        self.corr._xipm_sumofweights_cache = legacy_value
        self.corr._xipm_sumofweights_cache_w_fingerprint = legacy_key
        self.corr._xipm_sumofweights_cache_prepare_version = self.corr._prepare_version

        with patch.object(
            self.corr,
            "_compute_xipm_sumofweights",
            side_effect=AssertionError("legacy cache should satisfy lookup"),
        ):
            looked_up = self.corr._get_xipm_sumofweights(w1_dev, w2_dev)

        self.assertIs(looked_up, legacy_value)
        self.assertIsInstance(self.corr._xipm_sumofweights_cache, dict)
        self.assertIs(self.corr._xipm_sumofweights_cache[legacy_key], legacy_value)
        self.assertIsNone(self.corr._xipm_sumofweights_cache_w_fingerprint)

    def test_xipm_explicit_sumofweights_skips_fingerprint_checks(self):
        self._setup_mock_pairs()
        g11 = np.random.rand(self.npix)
        g21 = np.random.rand(self.npix)
        g12 = g11
        g22 = g21
        w1 = np.random.rand(self.npix)
        w2 = w1
        explicit_sum = np.ones((self.corr.n_patches, self.corr.nbins), dtype=np.float64)

        with patch.object(
            self.corr,
            "_fingerprint_weights",
            side_effect=AssertionError("fingerprint should be skipped"),
        ):
            xip, xim = self.corr.compute_shear_shear(
                g11,
                g21,
                g12,
                g22,
                w1,
                w2,
                sumofweights=explicit_sum,
            )

        self.assertEqual(xip.shape, (self.corr.n_patches, self.corr.nbins))
        self.assertEqual(xim.shape, (self.corr.n_patches, self.corr.nbins))

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

                xip, xim = corr.compute_shear_shear(g11, g21, g12, g22, w1, w2)

                self.assertEqual(xip.shape, (self.n_patches, self.nbins))
                self.assertEqual(xim.shape, (self.n_patches, self.nbins))
            finally:
                sys.modules.pop("cupy", None)
                sys.modules.pop(module_name, None)
    def test_vectorized_shear_shear_same_w_reuses_cache(self):
        nzbins = 2
        shear_maps = np.random.rand(nzbins, 2, self.npix)
        w = np.random.rand(nzbins, self.npix)
        self._setup_mock_pairs()

        with patch.object(
            self.corr,
            "_compute_tomo_sumofweights",
            wraps=self.corr._compute_tomo_sumofweights,
        ) as spy_compute:
            self.corr.vectorized_shear_shear(shear_maps, w)
            self.corr.vectorized_shear_shear(shear_maps, w)
            self.assertEqual(spy_compute.call_count, 1)

    def test_vectorized_shear_shear_changed_w_recomputes(self):
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
            self.corr.vectorized_shear_shear(shear_maps, w)
            self.corr.vectorized_shear_shear(shear_maps, w_changed)
            self.assertEqual(spy_compute.call_count, 2)

    def test_vectorized_shear_shear_explicit_sumofweights_still_accepted(self):
        nzbins = 2
        nzbin_combs = int(binom(nzbins + 1, 2))
        shear_maps = np.random.rand(nzbins, 2, self.npix)
        w = np.random.rand(nzbins, self.npix)
        sumofweights = np.ones((2, nzbin_combs))
        self._setup_mock_pairs()

        xip, xim = self.corr.vectorized_shear_shear(shear_maps, w, sumofweights=sumofweights)
        self.assertEqual(xip.shape, (nzbin_combs, self.corr.n_patches, self.corr.nbins))
        self.assertEqual(xim.shape, (nzbin_combs, self.corr.n_patches, self.corr.nbins))

    def test_vectorized_shear_shear_explicit_sumofweights_skips_fingerprint_checks(self):
        nzbins = 2
        nzbin_combs = int(binom(nzbins + 1, 2))
        shear_maps = np.random.rand(nzbins, 2, self.npix)
        w = np.random.rand(nzbins, self.npix)
        explicit_sum = np.ones((2, nzbin_combs), dtype=np.float64)
        self._setup_mock_pairs()

        cache_sentinel = np.array([123.0], dtype=np.float64)
        self.corr._tomo_sumofweights_cache = cache_sentinel
        self.corr._tomo_sumofweights_cache_w_fingerprint = ("sentinel",)
        self.corr._tomo_sumofweights_cache_prepare_version = -1

        with patch.object(
            self.corr,
            "_fingerprint_weights",
            side_effect=AssertionError("fingerprint should be skipped"),
        ):
            xip, xim = self.corr.vectorized_shear_shear(
                shear_maps, w, sumofweights=explicit_sum
            )

        self.assertEqual(xip.shape, (nzbin_combs, self.corr.n_patches, self.corr.nbins))
        self.assertEqual(xim.shape, (nzbin_combs, self.corr.n_patches, self.corr.nbins))
        self.assertIs(self.corr._tomo_sumofweights_cache, cache_sentinel)
        self.assertEqual(self.corr._tomo_sumofweights_cache_w_fingerprint, ("sentinel",))
        self.assertEqual(self.corr._tomo_sumofweights_cache_prepare_version, -1)

    def test_vectorized_shear_shear_prepares(self):
        """Test that vectorized_shear_shear calls prepare if needed."""
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
            self.corr.vectorized_shear_shear(shear_maps, w)
            spy_prepare.assert_called_once()

    def test_i3pcf_wrapper_methods_match_helpers(self):
        corr = self._make_small_cpu_corr()
        rng = np.random.default_rng(123)

        nmaps, nzbins, npatches, nbins = 2, 2, 4, 3
        npairs = nzbins * (nzbins + 1) // 2
        M_g = rng.normal(size=(nmaps, nzbins, npatches))
        M_a = rng.normal(size=(nmaps, nzbins, npatches))
        xi_p = rng.normal(size=(nmaps, npairs, npatches, nbins))
        xi_m = rng.normal(size=(nmaps, npairs, npatches, nbins))
        xi_g = rng.normal(size=(nmaps, npairs, npatches, nbins))
        xi_t = rng.normal(size=(nmaps, npairs, npatches, nbins))

        np.testing.assert_allclose(
            corr.zeta_g_plus(M_g, xi_p),
            helper_zeta_g_plus(M_g, xi_p),
        )
        np.testing.assert_allclose(
            corr.zeta_g_minus(M_g, xi_m),
            helper_zeta_g_minus(M_g, xi_m),
        )
        np.testing.assert_allclose(
            corr.zeta_a_plus(M_a, xi_p),
            helper_zeta_a_plus(M_a, xi_p),
        )
        np.testing.assert_allclose(
            corr.zeta_a_minus(M_a, xi_m),
            helper_zeta_a_minus(M_a, xi_m),
        )
        np.testing.assert_allclose(
            corr.zeta_g_g(M_g, xi_g),
            helper_zeta_g_g(M_g, xi_g),
        )
        np.testing.assert_allclose(
            corr.zeta_a_g(M_a, xi_g),
            helper_zeta_a_g(M_a, xi_g),
        )
        np.testing.assert_allclose(
            corr.zeta_g_t(M_g, xi_t),
            helper_zeta_g_t(M_g, xi_t),
        )
        np.testing.assert_allclose(
            corr.zeta_a_t(M_a, xi_t),
            helper_zeta_a_t(M_a, xi_t),
        )

    def test_calculate_all_zetas_api_uses_explicit_center_annulus_names(self):
        corr = self._make_small_cpu_corr()
        rng = np.random.default_rng(456)

        nmaps, nzbins, npatches, nbins = 2, 2, 4, 3
        npairs = nzbins * (nzbins + 1) // 2
        M_g = rng.normal(size=(nmaps, nzbins, npatches))
        M_a = rng.normal(size=(nmaps, nzbins, npatches))
        xi_p = rng.normal(size=(nmaps, npairs, npatches, nbins))
        xi_m = rng.normal(size=(nmaps, npairs, npatches, nbins))
        xi_g = rng.normal(size=(nmaps, npairs, npatches, nbins))
        xi_t = rng.normal(size=(nmaps, npairs, npatches, nbins))

        all_zetas = corr.calculate_all_zetas(
            M_g=M_g,
            M_a=M_a,
            xi_p=xi_p,
            xi_m=xi_m,
            xi_g=xi_g,
            xi_t=xi_t,
        )
        self.assertEqual(
            set(all_zetas.keys()),
            {
                "zeta_g_plus",
                "zeta_g_minus",
                "zeta_a_plus",
                "zeta_a_minus",
                "zeta_g_g",
                "zeta_a_g",
                "zeta_g_t",
                "zeta_a_t",
            },
        )


if __name__ == "__main__":
    unittest.main()
