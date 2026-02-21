
import sys
import unittest
from unittest.mock import MagicMock, patch
import warnings
import importlib

import numpy as np

import CosmoFuse.backend
from CosmoFuse.backend import (
    Backend,
    _MAX_VECTOR_TOMO_BINS,
    _cpu_aperture_density_kernel,
    _build_cupy_density_density_tomo_vectorized_kernel,
    _build_cupy_density_shear_tomo_vectorized_kernel,
    _build_cupy_tomo_vectorized_kernel,
    _cpu_density_density_corr_kernel,
    _cpu_density_density_tomo_vectorized_kernel,
    _cpu_density_shear_corr_kernel,
    _cpu_density_shear_tomo_vectorized_kernel,
    _cpu_xipm_auto_corr_kernel,
    _cpu_xipm_cross_corr_kernel,
    get_backend,
)


class TestBackend(unittest.TestCase):
    def setUp(self):
        self.numpy_backend = Backend("numpy", np)

    def test_to_device_numpy(self):
        arr = np.array([1, 2, 3])
        dev_arr = self.numpy_backend.to_device(arr)
        np.testing.assert_array_equal(arr, dev_arr)
        self.assertIsInstance(dev_arr, np.ndarray)

    def test_to_numpy_numpy(self):
        arr = np.array([1, 2, 3])
        np_arr = self.numpy_backend.to_numpy(arr)
        np.testing.assert_array_equal(arr, np_arr)
        self.assertIsInstance(np_arr, np.ndarray)

    def test_get_memory_pool_numpy(self):
        self.assertIsNone(self.numpy_backend.get_memory_pool())

    def test_to_device_unknown_backend(self):
        backend = Backend("other", np)
        arr = np.array([1, 2, 3])
        dev_arr = backend.to_device(arr)
        self.assertIs(dev_arr, arr)

    def test_to_numpy_unknown_backend(self):
        backend = Backend("other", np)
        arr = np.array([1, 2, 3])
        np_arr = backend.to_numpy(arr)
        self.assertIsInstance(np_arr, np.ndarray)

    @patch.dict(sys.modules, {"cupy": MagicMock()})
    def test_cupy_backend_creation(self):
        import cupy
        cupy.cuda.runtime.getDeviceCount.return_value = 1
        backend = get_backend(0)
        self.assertEqual(backend.name, "cupy")
        self.assertEqual(backend.device_id, 0)

    @patch.dict(sys.modules, {"cupy": MagicMock()})
    def test_cupy_elementwise_kernels_use_fast_math(self):
        import cupy
        cupy.cuda.runtime.getDeviceCount.return_value = 1
        cupy.ElementwiseKernel.side_effect = [
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ]

        backend = get_backend(0)

        self.assertEqual(backend.name, "cupy")
        self.assertEqual(cupy.ElementwiseKernel.call_count, 5)
        for call in cupy.ElementwiseKernel.call_args_list:
            self.assertEqual(call.kwargs.get("options"), ("--use_fast_math",))

    @patch.dict(sys.modules, {"cupy": MagicMock()})
    def test_to_device_cupy(self):
        import cupy
        cupy.cuda.runtime.getDeviceCount.return_value = 1
        cupy.asarray.return_value = "cupy_array"
        
        backend = get_backend(0)
        arr = np.array([1, 2, 3])
        dev_arr = backend.to_device(arr)

        cupy.asarray.assert_called_once_with(arr)
        self.assertEqual(dev_arr, "cupy_array")

    @patch.dict(sys.modules, {"cupy": MagicMock()})
    def test_to_numpy_cupy(self):
        import cupy
        cupy.cuda.runtime.getDeviceCount.return_value = 1
        cupy.asnumpy.return_value = "numpy_array"
        
        backend = get_backend(0)
        arr = MagicMock() # mock a cupy array
        np_arr = backend.to_numpy(arr)

        cupy.asnumpy.assert_called_once_with(arr)
        self.assertEqual(np_arr, "numpy_array")


    @patch.dict(sys.modules, {"cupy": MagicMock()})
    def test_get_memory_pool_cupy(self):
        import cupy
        cupy.cuda.runtime.getDeviceCount.return_value = 1
        cupy.get_default_memory_pool.return_value = "cupy_pool"

        backend = get_backend(0)
        pool = backend.get_memory_pool()
        self.assertEqual(pool, "cupy_pool")

    def test_get_backend_unknown_device(self):
        with self.assertRaises(ValueError):
            get_backend("unknown")

    @patch.dict(sys.modules, {"cupy": MagicMock()})
    def test_get_backend_gpu_device_not_found(self):
        import cupy
        cupy.cuda.runtime.getDeviceCount.return_value = 1
        with self.assertRaises(ValueError):
            get_backend(1) # Request device 1 when only 0 is available

    @patch.dict(sys.modules, {"cupy": None})
    def test_get_backend_gpu_no_cupy(self):
        with self.assertRaises(ImportError):
            get_backend("gpu")

    def test_get_backend_cpu(self):
        backend = get_backend("cpu")
        self.assertEqual(backend.name, "numpy")
        
    def test_get_backend_auto_no_cupy(self):
        with patch.dict(sys.modules, {"cupy": None}):
            with patch("CosmoFuse.backend.warnings.warn") as mock_warn:
                importlib.reload(CosmoFuse.backend)
                backend = CosmoFuse.backend.get_backend('auto')
                self.assertEqual(backend.name, 'numpy')
                mock_warn.assert_called_once_with("Cupy not installed, falling back to CPU (numpy).")

    def test_get_backend_auto_import_error_in_gpu_block(self):
        real_import = __import__
        call_count = {"cupy": 0}

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "cupy":
                if call_count["cupy"] == 0:
                    call_count["cupy"] += 1
                    return MagicMock()
                raise ImportError("mocked cupy import failure")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            with patch("CosmoFuse.backend.warnings.warn") as mock_warn:
                backend = get_backend("auto")
                self.assertEqual(backend.name, "numpy")
                mock_warn.assert_called_once_with(
                    "Cupy not installed, falling back to CPU (numpy)."
                )

    @patch.dict(sys.modules, {"cupy": MagicMock()})
    def test_get_backend_auto_with_cupy(self):
        import cupy
        cupy.cuda.runtime.getDeviceCount.return_value = 1
        backend = get_backend("auto")
        self.assertEqual(backend.name, "cupy")

    def test_cpu_xipm_cross_corr_kernel_complex128_dispatch(self):
        g1a = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        g2a = np.array([0.4, 0.5, 0.6], dtype=np.float64)
        g1b = np.array([0.7, 0.8, 0.9], dtype=np.float64)
        g2b = np.array([1.0, 1.1, 1.2], dtype=np.float64)
        wa = np.array([1.0, 0.5, 2.0], dtype=np.float64)
        wb = np.array([1.5, 1.0, 0.25], dtype=np.float64)

        ind_i = np.array([0, 1], dtype=np.int64)
        ind_j = np.array([1, 2], dtype=np.int64)
        exp_i = np.array([1.0 + 0.5j, -0.2 + 0.3j], dtype=np.complex128)
        exp_j = np.array([0.7 - 0.1j, 0.4 + 0.8j], dtype=np.complex128)
        offsets = np.array([0, 1, 2], dtype=np.int64)

        out_ab_p = np.zeros(offsets.shape[0] - 1, dtype=np.complex128)
        out_ab_m = np.zeros(offsets.shape[0] - 1, dtype=np.complex128)
        out_ba_p = np.zeros(offsets.shape[0] - 1, dtype=np.complex128)
        out_ba_m = np.zeros(offsets.shape[0] - 1, dtype=np.complex128)

        _cpu_xipm_cross_corr_kernel(
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
            offsets,
            out_ab_p,
            out_ab_m,
            out_ba_p,
            out_ba_m,
        )

        exp_ab_p = np.zeros_like(out_ab_p)
        exp_ab_m = np.zeros_like(out_ab_m)
        exp_ba_p = np.zeros_like(out_ba_p)
        exp_ba_m = np.zeros_like(out_ba_m)

        for idx in range(ind_i.shape[0]):
            i = ind_i[idx]
            j = ind_j[idx]

            ga_i = np.complex128(g1a[i] + 1j * g2a[i])
            gb_i = np.complex128(g1b[i] + 1j * g2b[i])
            ga_j = np.complex128(g1a[j] + 1j * g2a[j])
            gb_j = np.complex128(g1b[j] + 1j * g2b[j])

            ga_i_rot = wa[i] * ga_i * exp_i[idx]
            gb_i_rot = wb[i] * gb_i * exp_i[idx]
            ga_j_rot = wa[j] * ga_j * exp_j[idx]
            gb_j_rot = wb[j] * gb_j * exp_j[idx]

            exp_ab_p[idx] = gb_j_rot * np.conjugate(ga_i_rot)
            exp_ab_m[idx] = gb_j_rot * ga_i_rot
            exp_ba_p[idx] = ga_j_rot * np.conjugate(gb_i_rot)
            exp_ba_m[idx] = ga_j_rot * gb_i_rot

        np.testing.assert_allclose(out_ab_p, exp_ab_p)
        np.testing.assert_allclose(out_ab_m, exp_ab_m)
        np.testing.assert_allclose(out_ba_p, exp_ba_p)
        np.testing.assert_allclose(out_ba_m, exp_ba_m)

    def test_cpu_xipm_auto_corr_kernel_complex64_dispatch(self):
        g11 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        g21 = np.array([0.4, 0.5, 0.6], dtype=np.float32)
        g12 = np.array([0.7, 0.8, 0.9], dtype=np.float32)
        g22 = np.array([1.0, 1.1, 1.2], dtype=np.float32)
        w1 = np.array([1.0, 0.5, 2.0], dtype=np.float32)
        w2 = np.array([1.5, 1.0, 0.25], dtype=np.float32)

        ind_i = np.array([0, 1], dtype=np.int64)
        ind_j = np.array([1, 2], dtype=np.int64)
        exp_i = np.array([1.0 + 0.5j, -0.2 + 0.3j], dtype=np.complex64)
        exp_j = np.array([0.7 - 0.1j, 0.4 + 0.8j], dtype=np.complex64)
        offsets = np.array([0, 1, 2], dtype=np.int64)

        out_p = np.zeros(offsets.shape[0] - 1, dtype=np.complex64)
        out_m = np.zeros(offsets.shape[0] - 1, dtype=np.complex64)

        _cpu_xipm_auto_corr_kernel(
            g11,
            g21,
            g12,
            g22,
            w1,
            w2,
            ind_i,
            ind_j,
            exp_i,
            exp_j,
            offsets,
            out_p,
            out_m,
        )

        exp_p = np.zeros_like(out_p)
        exp_m = np.zeros_like(out_m)
        for idx in range(ind_i.shape[0]):
            i = ind_i[idx]
            j = ind_j[idx]
            g2 = w1[i] * np.complex64(g11[i] + 1j * g21[i]) * exp_i[idx]
            g1 = w2[j] * np.complex64(g12[j] + 1j * g22[j]) * exp_j[idx]
            exp_p[idx] = g1 * np.conjugate(g2)
            exp_m[idx] = g1 * g2

        np.testing.assert_allclose(out_p, exp_p, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(out_m, exp_m, rtol=1e-6, atol=1e-7)

    def test_cpu_xipm_auto_corr_kernel_complex128_dispatch(self):
        g11 = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        g21 = np.array([0.4, 0.5, 0.6], dtype=np.float64)
        g12 = np.array([0.7, 0.8, 0.9], dtype=np.float64)
        g22 = np.array([1.0, 1.1, 1.2], dtype=np.float64)
        w1 = np.array([1.0, 0.5, 2.0], dtype=np.float64)
        w2 = np.array([1.5, 1.0, 0.25], dtype=np.float64)

        ind_i = np.array([0, 1], dtype=np.int64)
        ind_j = np.array([1, 2], dtype=np.int64)
        exp_i = np.array([1.0 + 0.5j, -0.2 + 0.3j], dtype=np.complex128)
        exp_j = np.array([0.7 - 0.1j, 0.4 + 0.8j], dtype=np.complex128)
        offsets = np.array([0, 1, 2], dtype=np.int64)

        out_p = np.zeros(offsets.shape[0] - 1, dtype=np.complex128)
        out_m = np.zeros(offsets.shape[0] - 1, dtype=np.complex128)

        _cpu_xipm_auto_corr_kernel(
            g11,
            g21,
            g12,
            g22,
            w1,
            w2,
            ind_i,
            ind_j,
            exp_i,
            exp_j,
            offsets,
            out_p,
            out_m,
        )

        exp_p = np.zeros_like(out_p)
        exp_m = np.zeros_like(out_m)
        for idx in range(ind_i.shape[0]):
            i = ind_i[idx]
            j = ind_j[idx]
            g2 = w1[i] * np.complex128(g11[i] + 1j * g21[i]) * exp_i[idx]
            g1 = w2[j] * np.complex128(g12[j] + 1j * g22[j]) * exp_j[idx]
            exp_p[idx] = g1 * np.conjugate(g2)
            exp_m[idx] = g1 * g2

        np.testing.assert_allclose(out_p, exp_p)
        np.testing.assert_allclose(out_m, exp_m)

    def test_cpu_backend_exposes_tomo_vectorized_kernel(self):
        backend = get_backend("cpu")
        self.assertIsNotNone(backend.xipm_tomo_vectorized_kernel)
        self.assertIsNotNone(backend.kernel_density_density)
        self.assertIsNotNone(backend.kernel_density_shear)
        self.assertIsNotNone(backend.kernel_density_density_tomo_vectorized)
        self.assertIsNotNone(backend.kernel_density_shear_tomo_vectorized)

        shear = np.array(
            [[[1.0, 0.0], [2.0, 0.0]], [[3.0, 0.0], [4.0, 0.0]]], dtype=np.float64
        )
        weights = np.ones((2, 2), dtype=np.float64)
        ind_i = np.array([0], dtype=np.int64)
        ind_j = np.array([1], dtype=np.int64)
        rot_i = np.array([1.0 + 0.0j], dtype=np.complex128)
        rot_j = np.array([1.0 + 0.0j], dtype=np.complex128)
        comb_i = np.array([0, 0, 1], dtype=np.int64)
        comb_j = np.array([0, 1, 1], dtype=np.int64)
        out_p = np.zeros((6, 1), dtype=np.complex128)
        out_m = np.zeros((6, 1), dtype=np.complex128)

        launched = backend.xipm_tomo_vectorized_kernel(
            shear,
            weights,
            ind_i,
            ind_j,
            rot_i,
            rot_j,
            np.array([0, 1], dtype=np.int64),
            comb_i,
            comb_j,
            out_p,
            out_m,
        )
        self.assertIsNone(launched)
        self.assertEqual(out_p.shape, (6, 1))
        self.assertEqual(out_m.shape, (6, 1))

    def test_cpu_density_density_corr_kernel(self):
        density_a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        density_b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        w_a = np.array([1.0, 0.5, 2.0], dtype=np.float64)
        w_b = np.array([1.0, 2.0, 0.25], dtype=np.float64)
        ind_i = np.array([0, 1, 2], dtype=np.int64)
        ind_j = np.array([1, 2, 0], dtype=np.int64)
        offsets = np.array([0, 2, 3], dtype=np.int64)
        out_w = np.zeros(2, dtype=np.float64)

        _cpu_density_density_corr_kernel(
            density_a,
            density_b,
            w_a,
            w_b,
            ind_i,
            ind_j,
            offsets,
            out_w,
        )

        expected = np.zeros_like(out_w)
        for b in range(offsets.shape[0] - 1):
            for idx in range(offsets[b], offsets[b + 1]):
                i = ind_i[idx]
                j = ind_j[idx]
                expected[b] += w_a[i] * w_b[j] * density_a[i] * density_b[j]

        np.testing.assert_allclose(out_w, expected)

    def test_cpu_density_shear_corr_kernel(self):
        density_lens = np.array([1.0, 2.0, 0.5], dtype=np.float64)
        g1_source = np.array([0.2, -0.1, 0.4], dtype=np.float64)
        g2_source = np.array([0.3, 0.5, -0.2], dtype=np.float64)
        w_lens = np.array([1.0, 0.5, 2.0], dtype=np.float64)
        w_source = np.array([1.0, 1.5, 0.25], dtype=np.float64)
        ind_i = np.array([0, 1, 2], dtype=np.int64)
        ind_j = np.array([1, 2, 0], dtype=np.int64)
        exp_j = np.array([0.6 + 0.8j, 1.0 + 0.0j, -0.5 + 0.5j], dtype=np.complex128)
        offsets = np.array([0, 2, 3], dtype=np.int64)
        out_gt = np.zeros(2, dtype=np.float64)

        _cpu_density_shear_corr_kernel(
            density_lens,
            g1_source,
            g2_source,
            w_lens,
            w_source,
            ind_i,
            ind_j,
            exp_j,
            offsets,
            out_gt,
        )

        expected = np.zeros_like(out_gt)
        for b in range(offsets.shape[0] - 1):
            for idx in range(offsets[b], offsets[b + 1]):
                i = ind_i[idx]
                j = ind_j[idx]
                gamma_t = -g1_source[j] * exp_j[idx].real + g2_source[j] * exp_j[idx].imag
                expected[b] += w_lens[i] * w_source[j] * density_lens[i] * gamma_t

        np.testing.assert_allclose(out_gt, expected)

    def test_cpu_density_density_tomo_vectorized_kernel(self):
        density = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        weights = np.array([[1.0, 0.5], [2.0, 1.0]], dtype=np.float64)
        ind_i = np.array([0], dtype=np.int64)
        ind_j = np.array([1], dtype=np.int64)
        offsets = np.array([0, 1], dtype=np.int64)
        comb_i = np.array([0, 0, 1], dtype=np.int32)
        comb_j = np.array([0, 1, 1], dtype=np.int32)
        out_num = np.zeros((3, 1), dtype=np.float64)

        _cpu_density_density_tomo_vectorized_kernel(
            density,
            weights,
            ind_i,
            ind_j,
            offsets,
            comb_i,
            comb_j,
            out_num,
        )

        self.assertEqual(out_num.shape, (3, 1))
        self.assertNotEqual(out_num[0, 0], 0.0)

    def test_cpu_density_shear_tomo_vectorized_kernel(self):
        density = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        shear = np.array(
            [
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.5, 0.6], [0.7, 0.8]],
            ],
            dtype=np.float64,
        )
        lens_weights = np.array([[1.0, 0.5], [2.0, 1.0]], dtype=np.float64)
        source_weights = np.array([[1.0, 1.5], [0.5, 2.0]], dtype=np.float64)
        ind_i = np.array([0], dtype=np.int64)
        ind_j = np.array([1], dtype=np.int64)
        rot_i = np.array([0.8 - 0.6j], dtype=np.complex128)
        rot_j = np.array([0.6 + 0.8j], dtype=np.complex128)
        offsets = np.array([0, 1], dtype=np.int64)
        comb_i = np.array([0, 1], dtype=np.int32)
        comb_j = np.array([0, 1], dtype=np.int32)
        out_num = np.zeros((2, 1), dtype=np.float64)

        _cpu_density_shear_tomo_vectorized_kernel(
            density,
            shear,
            lens_weights,
            source_weights,
            ind_i,
            ind_j,
            rot_i,
            rot_j,
            offsets,
            comb_i,
            comb_j,
            out_num,
        )

        self.assertEqual(out_num.shape, (2, 1))
        self.assertNotEqual(out_num[0, 0], 0.0)

    def test_cpu_aperture_density_kernel(self):
        Q_inds = np.array([0, 1, 2], dtype=np.int64)
        Q_val = np.array([0.2, 0.3, 0.5], dtype=np.float64)
        Q_offsets = np.array([0, 2, 3], dtype=np.int64)
        map_values = np.array([1.0, 2.0, 4.0], dtype=np.float64)
        weights = np.array([1.0, 0.5, 2.0], dtype=np.float64)
        Q_patch_area = np.array([1.0, 2.0], dtype=np.float64)
        out = np.zeros(2, dtype=np.float64)

        _cpu_aperture_density_kernel(
            Q_inds,
            Q_val,
            Q_offsets,
            map_values,
            weights,
            Q_patch_area,
            out,
        )

        expected0 = 1.0 * (
            (weights[0] * map_values[0] * Q_val[0])
            + (weights[1] * map_values[1] * Q_val[1])
        ) / (weights[0] + weights[1])
        expected1 = 2.0 * (weights[2] * map_values[2] * Q_val[2]) / weights[2]
        np.testing.assert_allclose(out, np.array([expected0, expected1]))

    def test_cupy_density_density_tomo_vectorized_kernel_missing_rawkernel_returns_false(self):
        class FakeModule:
            float32 = np.float32

        kernel = _build_cupy_density_density_tomo_vectorized_kernel(FakeModule)
        ok = kernel(
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.zeros((2, 1), dtype=np.float32),
        )
        self.assertFalse(ok)

    def test_cupy_density_density_tomo_vectorized_kernel_compile_failure_returns_false(self):
        class FakeModule:
            float32 = np.float32

            @staticmethod
            def RawKernel(*_args, **_kwargs):
                raise RuntimeError("compile failed")

        kernel = _build_cupy_density_density_tomo_vectorized_kernel(FakeModule)
        ok = kernel(
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.zeros((2, 1), dtype=np.float32),
        )
        self.assertFalse(ok)

    def test_cupy_density_density_tomo_vectorized_kernel_too_many_bins_returns_false(self):
        class FakeModule:
            float32 = np.float32
            RawKernel = staticmethod(lambda *_args, **_kwargs: MagicMock())

        kernel = _build_cupy_density_density_tomo_vectorized_kernel(FakeModule)
        ok = kernel(
            np.zeros((1, _MAX_VECTOR_TOMO_BINS + 1), dtype=np.float32),
            np.zeros((1, _MAX_VECTOR_TOMO_BINS + 1), dtype=np.float32),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.zeros((2, 1), dtype=np.float32),
        )
        self.assertFalse(ok)

    def test_cupy_density_density_tomo_vectorized_kernel_success_and_source(self):
        compiled_sources = []
        compile_calls = {"count": 0}

        class FakeKernel:
            def __call__(self, _grid, _block, _args):
                return None

        class FakeModule:
            float32 = np.float32

            @staticmethod
            def RawKernel(source, kernel_name, options=None):
                compile_calls["count"] += 1
                compiled_sources.append((source, kernel_name, options))
                return FakeKernel()

        kernel = _build_cupy_density_density_tomo_vectorized_kernel(FakeModule)
        ok = kernel(
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([0, 0, 1], dtype=np.int32),
            np.array([0, 1, 1], dtype=np.int32),
            np.zeros((6, 1), dtype=np.float32),
        )
        ok_cached = kernel(
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([0, 0, 1], dtype=np.int32),
            np.array([0, 1, 1], dtype=np.int32),
            np.zeros((6, 1), dtype=np.float32),
        )
        self.assertTrue(ok)
        self.assertTrue(ok_cached)
        self.assertEqual(len(compiled_sources), 1)
        self.assertEqual(compile_calls["count"], 1)
        self.assertNotIn("cuComplex", compiled_sources[0][0])
        self.assertEqual(compiled_sources[0][2], ("--use_fast_math",))

    def test_cupy_density_shear_tomo_vectorized_kernel_missing_rawkernel_returns_false(self):
        class FakeModule:
            float32 = np.float32
            complex64 = np.complex64

        kernel = _build_cupy_density_shear_tomo_vectorized_kernel(FakeModule)
        ok = kernel(
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1, 2), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.complex64),
            np.zeros(1, dtype=np.complex64),
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.zeros((1, 1), dtype=np.float32),
        )
        self.assertFalse(ok)

    def test_cupy_density_shear_tomo_vectorized_kernel_too_many_bins_returns_false(self):
        class FakeModule:
            float32 = np.float32
            complex64 = np.complex64
            RawKernel = staticmethod(lambda *_args, **_kwargs: MagicMock())

        kernel = _build_cupy_density_shear_tomo_vectorized_kernel(FakeModule)
        ok = kernel(
            np.zeros((1, _MAX_VECTOR_TOMO_BINS + 1), dtype=np.float32),
            np.zeros((1, _MAX_VECTOR_TOMO_BINS + 1, 2), dtype=np.float32),
            np.zeros((1, _MAX_VECTOR_TOMO_BINS + 1), dtype=np.float32),
            np.zeros((1, _MAX_VECTOR_TOMO_BINS + 1), dtype=np.float32),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.complex64),
            np.zeros(1, dtype=np.complex64),
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.zeros((1, 1), dtype=np.float32),
        )
        self.assertFalse(ok)

    def test_cupy_density_shear_tomo_vectorized_kernel_compile_failure_returns_false(self):
        class FakeModule:
            float32 = np.float32
            complex64 = np.complex64

            @staticmethod
            def RawKernel(*_args, **_kwargs):
                raise RuntimeError("compile failed")

        kernel = _build_cupy_density_shear_tomo_vectorized_kernel(FakeModule)
        ok = kernel(
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1, 2), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.complex64),
            np.zeros(1, dtype=np.complex64),
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.zeros((1, 1), dtype=np.float32),
        )
        self.assertFalse(ok)

    def test_cupy_density_shear_tomo_vectorized_kernel_success_and_source(self):
        compiled_sources = []

        class FakeKernel:
            def __call__(self, _grid, _block, _args):
                return None

        class FakeModule:
            float32 = np.float32
            complex64 = np.complex64

            @staticmethod
            def RawKernel(source, kernel_name, options=None):
                compiled_sources.append((source, kernel_name, options))
                return FakeKernel()

        kernel = _build_cupy_density_shear_tomo_vectorized_kernel(FakeModule)
        ok = kernel(
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.complex64),
            np.zeros(1, dtype=np.complex64),
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.zeros((1, 1), dtype=np.float32),
        )
        self.assertTrue(ok)
        self.assertEqual(len(compiled_sources), 1)
        self.assertIn("rot_i", compiled_sources[0][0])
        self.assertIn("rot_j", compiled_sources[0][0])
        self.assertEqual(compiled_sources[0][2], ("--use_fast_math",))

    def test_cupy_density_shear_tomo_vectorized_kernel_cache_reuse(self):
        compile_calls = {"count": 0}

        class FakeKernel:
            def __call__(self, _grid, _block, _args):
                return None

        class FakeModule:
            float32 = np.float32
            complex64 = np.complex64

            @staticmethod
            def RawKernel(_source, _kernel_name, options=None):
                compile_calls["count"] += 1
                return FakeKernel()

        kernel = _build_cupy_density_shear_tomo_vectorized_kernel(FakeModule)
        args = (
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.complex64),
            np.zeros(1, dtype=np.complex64),
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.zeros((1, 1), dtype=np.float32),
        )
        self.assertTrue(kernel(*args))
        self.assertTrue(kernel(*args))
        self.assertEqual(compile_calls["count"], 1)

    def test_cupy_density_shear_tomo_vectorized_kernel_float64_rotation_branch(self):
        class FakeKernel:
            def __call__(self, _grid, _block, _args):
                return None

        class FakeModule:
            float32 = np.float32
            complex64 = np.complex64

            @staticmethod
            def RawKernel(_source, _kernel_name, options=None):
                return FakeKernel()

        kernel = _build_cupy_density_shear_tomo_vectorized_kernel(FakeModule)
        ok = kernel(
            np.zeros((1, 1), dtype=np.float64),
            np.zeros((1, 1, 2), dtype=np.float64),
            np.zeros((1, 1), dtype=np.float64),
            np.zeros((1, 1), dtype=np.float64),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.complex128),
            np.zeros(1, dtype=np.complex128),
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.zeros((1, 1), dtype=np.float64),
        )
        self.assertTrue(ok)

    def test_cupy_tomo_vectorized_kernel_missing_rawkernel_returns_false(self):
        class FakeModule:
            float32 = np.float32
            complex64 = np.complex64

        kernel = _build_cupy_tomo_vectorized_kernel(FakeModule)
        ok = kernel(
            np.zeros((1, 1, 2), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.complex64),
            np.zeros(1, dtype=np.complex64),
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.zeros((2, 2, 1), dtype=np.complex64),
        )
        self.assertFalse(ok)

    def test_cupy_tomo_vectorized_kernel_too_many_bins_returns_false(self):
        class FakeModule:
            float32 = np.float32
            complex64 = np.complex64
            RawKernel = staticmethod(lambda *_args, **_kwargs: MagicMock())

        kernel = _build_cupy_tomo_vectorized_kernel(FakeModule)
        ok = kernel(
            np.zeros((1, _MAX_VECTOR_TOMO_BINS + 1, 2), dtype=np.float32),
            np.zeros((1, _MAX_VECTOR_TOMO_BINS + 1), dtype=np.float32),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.complex64),
            np.zeros(1, dtype=np.complex64),
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.zeros((2, 2, 1), dtype=np.complex64),
        )
        self.assertFalse(ok)

    def test_cupy_tomo_vectorized_kernel_compile_failure_returns_false(self):
        class FakeModule:
            float32 = np.float32
            complex64 = np.complex64

            @staticmethod
            def RawKernel(*_args, **_kwargs):
                raise RuntimeError("compile failed")

        kernel = _build_cupy_tomo_vectorized_kernel(FakeModule)
        ok = kernel(
            np.zeros((1, 1, 2), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.complex64),
            np.zeros(1, dtype=np.complex64),
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.zeros((2, 2, 1), dtype=np.complex64),
        )
        self.assertFalse(ok)

    def test_cupy_tomo_vectorized_kernel_success_and_cache(self):
        rawkernel_calls = {"count": 0}
        launches = []

        class FakeKernel:
            def __call__(self, grid, block, args):
                launches.append((grid, block, args))

        class FakeModule:
            float32 = np.float32
            complex64 = np.complex64

            @staticmethod
            def RawKernel(*_args, **_kwargs):
                rawkernel_calls["count"] += 1
                return FakeKernel()

        kernel = _build_cupy_tomo_vectorized_kernel(FakeModule)
        shear = np.zeros((1, 2, 2), dtype=np.float32)
        weights = np.zeros((1, 2), dtype=np.float32)
        ind_i = np.zeros(1, dtype=np.int64)
        ind_j = np.zeros(1, dtype=np.int64)
        rot_i = np.zeros(1, dtype=np.complex64)
        rot_j = np.zeros(1, dtype=np.complex64)
        bin_offsets = np.array([0, 1], dtype=np.int64)
        comb_i = np.array([0, 0, 1], dtype=np.int32)
        comb_j = np.array([0, 1, 1], dtype=np.int32)
        out_num = np.zeros((2, 6, 1), dtype=np.complex64)

        ok1 = kernel(
            shear,
            weights,
            ind_i,
            ind_j,
            rot_i,
            rot_j,
            bin_offsets,
            comb_i,
            comb_j,
            out_num,
        )
        ok2 = kernel(
            shear,
            weights,
            ind_i,
            ind_j,
            rot_i,
            rot_j,
            bin_offsets,
            comb_i,
            comb_j,
            out_num,
        )

        self.assertTrue(ok1)
        self.assertTrue(ok2)
        self.assertEqual(rawkernel_calls["count"], 1)
        self.assertEqual(len(launches), 2)

    def test_cupy_tomo_vectorized_kernel_complex128(self):
        class FakeKernel:
            def __call__(self, _grid, _block, _args):
                return None

        class FakeModule:
            float32 = np.float32
            complex64 = np.complex64
            # complex128 needs to be present for checks, though not used in fake
            complex128 = np.complex128

            @staticmethod
            def RawKernel(source, kernel_name, options=None):
                return FakeKernel()

        kernel = _build_cupy_tomo_vectorized_kernel(FakeModule)
        
        # Inputs with complex128 (default python complex is complex128)
        rot_i = np.zeros(1, dtype=np.complex128)
        rot_j = np.zeros(1, dtype=np.complex128)
        
        # Other inputs (types matter for map_c_type logic but here focused on suffix logic)
        shear = np.zeros((1, 1, 2), dtype=np.float64)
        weights = np.zeros((1, 1), dtype=np.float64)
        ind_i = np.zeros(1, dtype=np.int64)
        ind_j = np.zeros(1, dtype=np.int64)
        bin_offsets = np.array([0, 1], dtype=np.int64)
        comb_i = np.array([0], dtype=np.int32)
        comb_j = np.array([0], dtype=np.int32)
        out_num = np.zeros((2, 1), dtype=np.float64)

        ok = kernel(
            shear,
            weights,
            ind_i,
            ind_j,
            rot_i,
            rot_j,
            bin_offsets,
            comb_i,
            comb_j,
            out_num,
        )
        self.assertTrue(ok)

    def test_cupy_tomo_vectorized_kernel_templates_exact_bin_count(self):
        compiled_sources = []

        class FakeKernel:
            def __call__(self, _grid, _block, _args):
                return None

        class FakeModule:
            float32 = np.float32
            complex64 = np.complex64

            @staticmethod
            def RawKernel(source, kernel_name, options=None):
                compiled_sources.append((source, kernel_name, options))
                return FakeKernel()

        kernel = _build_cupy_tomo_vectorized_kernel(FakeModule)
        ind_i = np.zeros(1, dtype=np.int64)
        ind_j = np.zeros(1, dtype=np.int64)
        rot_i = np.zeros(1, dtype=np.complex64)
        rot_j = np.zeros(1, dtype=np.complex64)
        bin_offsets = np.array([0, 1], dtype=np.int64)

        shear_2 = np.zeros((1, 2, 2), dtype=np.float32)
        weights_2 = np.zeros((1, 2), dtype=np.float32)
        comb_i_2 = np.array([0, 0, 1], dtype=np.int32)
        comb_j_2 = np.array([0, 1, 1], dtype=np.int32)
        out_num_2 = np.zeros((2, 6, 1), dtype=np.complex64)

        shear_3 = np.zeros((1, 3, 2), dtype=np.float32)
        weights_3 = np.zeros((1, 3), dtype=np.float32)
        comb_i_3 = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
        comb_j_3 = np.array([0, 1, 2, 1, 2, 2], dtype=np.int32)
        out_num_3 = np.zeros((2, 12, 1), dtype=np.complex64)

        ok_2 = kernel(
            shear_2,
            weights_2,
            ind_i,
            ind_j,
            rot_i,
            rot_j,
            bin_offsets,
            comb_i_2,
            comb_j_2,
            out_num_2,
        )
        ok_3 = kernel(
            shear_3,
            weights_3,
            ind_i,
            ind_j,
            rot_i,
            rot_j,
            bin_offsets,
            comb_i_3,
            comb_j_3,
            out_num_3,
        )
        ok_2_cached = kernel(
            shear_2,
            weights_2,
            ind_i,
            ind_j,
            rot_i,
            rot_j,
            bin_offsets,
            comb_i_2,
            comb_j_2,
            out_num_2,
        )

        self.assertTrue(ok_2)
        self.assertTrue(ok_3)
        self.assertTrue(ok_2_cached)
        self.assertEqual(len(compiled_sources), 2)
        self.assertIn("#define TOMO_BINS 2", compiled_sources[0][0])
        self.assertIn("#define TOMO_BINS 3", compiled_sources[1][0])
        for source, _kernel_name, options in compiled_sources:
            self.assertNotIn("MAX_TOMO_BINS", source)
            self.assertEqual(options, ("--use_fast_math",))

if __name__ == "__main__":
    unittest.main()
