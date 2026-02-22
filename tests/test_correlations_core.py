"""Tests for the Correlation class."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.special import binom

# Add src to path for testing
sys.path.insert(1, str(Path(__file__).parent.parent / "src"))

import CosmoFuse.correlations as correlations_module
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

    def test_init_precision_configuration(self):
        """Test initialization with custom precision configuration."""
        corr = Correlation(
            nside=self.nside,
            phi_center=self.phi_center,
            theta_center=self.theta_center,
            map_precision="float32",
            rotation_precision="float32",
        )
        self.assertEqual(corr.map_dtype, np.dtype(np.float32))
        self.assertEqual(corr.rotation_dtype, np.dtype(np.float32))
        self.assertEqual(corr.rotation_complex_dtype, np.dtype(np.complex64))
        self.assertEqual(corr.index_dtype, np.dtype(np.int64))

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

    def test_init_rejects_removed_index_precision_kwarg(self):
        """index_precision has been removed from the public Correlation API."""
        with self.assertRaises(TypeError):
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
