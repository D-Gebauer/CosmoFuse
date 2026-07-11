
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
    _compile_raw_cuda_kernel,
    _build_cupy_3x2pt_tomo_fused_kernel,
    _cpu_aperture_density_kernel,
    _cpu_aperture_shear_kernel,
    _cpu_3x2pt_tomo_fused_kernel,
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

    def test_to_device_with_stream_none_numpy(self):
        arr = np.array([1, 2, 3])
        dev_arr = self.numpy_backend.to_device(arr, stream=None)
        np.testing.assert_array_equal(arr, dev_arr)

    def test_to_numpy_with_stream_none_numpy(self):
        arr = np.array([1, 2, 3])
        np_arr = self.numpy_backend.to_numpy(arr, stream=None)
        np.testing.assert_array_equal(arr, np_arr)

    def test_create_stream_numpy_returns_none(self):
        self.assertIsNone(self.numpy_backend.create_stream())

    def test_synchronize_stream_numpy_noop(self):
        self.numpy_backend.synchronize_stream()
        self.numpy_backend.synchronize_stream(stream=None)

    def test_use_stream_numpy_returns_context_manager(self):
        ctx = self.numpy_backend.use_stream(None)
        with ctx:
            pass

    def test_warmup_numpy_runs_all_kernels(self):
        backend = get_backend("cpu")
        # exercises every CPU kernel with tiny inputs (compiles under JIT,
        # plain execution when JIT is disabled) - must not raise
        backend.warmup()
        backend.warmup(
            map_dtype=np.float32,
            rotation_dtype=np.float32,
            rotation_complex_dtype=np.complex64,
            index_dtype=np.int64,
        )

    def test_warmup_gpu_backend_is_noop(self):
        backend = Backend("cupy", MagicMock())
        backend.warmup()  # must not touch any kernels

    def test_alloc_pinned_numpy_returns_ndarray(self):
        arr = self.numpy_backend.alloc_pinned((3, 4), np.float64)
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (3, 4))
        self.assertEqual(arr.dtype, np.float64)

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

    def test_compile_raw_cuda_kernel_rawmodule_legacy_signature_fallback(self):
        source = "extern \"C\" __global__ void k(){}"
        kernel_name = "k"

        class LegacyRawModule:
            def __init__(self, *args, **kwargs):
                if "code" in kwargs:
                    raise TypeError("legacy RawModule signature")
                self.args = args
                self.kwargs = kwargs

            def get_function(self, name):
                return ("kernel", name, self.args, self.kwargs)

        class FakeModule:
            RawModule = LegacyRawModule

        compiled = _compile_raw_cuda_kernel(FakeModule, source, kernel_name)
        self.assertEqual(compiled[0], "kernel")
        self.assertEqual(compiled[1], kernel_name)
        self.assertEqual(compiled[2][0], source)

    def test_compile_raw_cuda_kernel_raises_without_raw_compiler(self):
        class FakeModule:
            pass

        with self.assertRaisesRegex(AttributeError, "RawModule or RawKernel"):
            _compile_raw_cuda_kernel(FakeModule, "code", "kernel")

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
            MagicMock(),
        ]

        backend = get_backend(0)

        self.assertEqual(backend.name, "cupy")
        self.assertEqual(cupy.ElementwiseKernel.call_count, 6)
        for call in cupy.ElementwiseKernel.call_args_list:
            self.assertEqual(call.kwargs.get("options"), ("--use_fast_math", "--std=c++14"))

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

        out_ab_p = np.zeros(offsets.shape[0] - 1, dtype=np.float64)
        out_ab_m = np.zeros(offsets.shape[0] - 1, dtype=np.float64)
        out_ba_p = np.zeros(offsets.shape[0] - 1, dtype=np.float64)
        out_ba_m = np.zeros(offsets.shape[0] - 1, dtype=np.float64)
        out_ab_w = np.zeros(offsets.shape[0] - 1, dtype=np.float64)
        out_ba_w = np.zeros(offsets.shape[0] - 1, dtype=np.float64)

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
            out_ab_w,
            out_ba_w,
        )

        exp_ab_p = np.zeros_like(out_ab_p)
        exp_ab_m = np.zeros_like(out_ab_m)
        exp_ba_p = np.zeros_like(out_ba_p)
        exp_ba_m = np.zeros_like(out_ba_m)
        exp_ab_w = np.zeros_like(out_ab_w)
        exp_ba_w = np.zeros_like(out_ba_w)

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

            exp_ab_p[idx] = np.real(gb_j_rot * np.conjugate(ga_i_rot))
            exp_ab_m[idx] = np.real(gb_j_rot * ga_i_rot)
            exp_ba_p[idx] = np.real(ga_j_rot * np.conjugate(gb_i_rot))
            exp_ba_m[idx] = np.real(ga_j_rot * gb_i_rot)
            exp_ab_w[idx] = wa[i] * wb[j]
            exp_ba_w[idx] = wb[i] * wa[j]

        np.testing.assert_allclose(out_ab_p, exp_ab_p)
        np.testing.assert_allclose(out_ab_m, exp_ab_m)
        np.testing.assert_allclose(out_ba_p, exp_ba_p)
        np.testing.assert_allclose(out_ba_m, exp_ba_m)
        np.testing.assert_allclose(out_ab_w, exp_ab_w)
        np.testing.assert_allclose(out_ba_w, exp_ba_w)

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

        out_p = np.zeros(offsets.shape[0] - 1, dtype=np.float32)
        out_m = np.zeros(offsets.shape[0] - 1, dtype=np.float32)
        out_w = np.zeros(offsets.shape[0] - 1, dtype=np.float32)

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
            out_w,
        )

        exp_p = np.zeros_like(out_p)
        exp_m = np.zeros_like(out_m)
        exp_w = np.zeros_like(out_w)
        for idx in range(ind_i.shape[0]):
            i = ind_i[idx]
            j = ind_j[idx]
            g2 = w1[i] * np.complex64(g11[i] + 1j * g21[i]) * exp_i[idx]
            g1 = w2[j] * np.complex64(g12[j] + 1j * g22[j]) * exp_j[idx]
            exp_p[idx] = np.real(g1 * np.conjugate(g2))
            exp_m[idx] = np.real(g1 * g2)
            exp_w[idx] = w1[i] * w2[j]

        np.testing.assert_allclose(out_p, exp_p, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(out_m, exp_m, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(out_w, exp_w, rtol=1e-6, atol=1e-7)

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

        out_p = np.zeros(offsets.shape[0] - 1, dtype=np.float64)
        out_m = np.zeros(offsets.shape[0] - 1, dtype=np.float64)
        out_w = np.zeros(offsets.shape[0] - 1, dtype=np.float64)

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
            out_w,
        )

        exp_p = np.zeros_like(out_p)
        exp_m = np.zeros_like(out_m)
        exp_w = np.zeros_like(out_w)
        for idx in range(ind_i.shape[0]):
            i = ind_i[idx]
            j = ind_j[idx]
            g2 = w1[i] * np.complex128(g11[i] + 1j * g21[i]) * exp_i[idx]
            g1 = w2[j] * np.complex128(g12[j] + 1j * g22[j]) * exp_j[idx]
            exp_p[idx] = np.real(g1 * np.conjugate(g2))
            exp_m[idx] = np.real(g1 * g2)
            exp_w[idx] = w1[i] * w2[j]

        np.testing.assert_allclose(out_p, exp_p)
        np.testing.assert_allclose(out_m, exp_m)
        np.testing.assert_allclose(out_w, exp_w)

    def test_cpu_backend_exposes_tomo_vectorized_kernel(self):
        backend = get_backend("cpu")
        self.assertIsNotNone(backend.xipm_tomo_vectorized_kernel)
        self.assertIsNotNone(backend.aperture_shear_kernel)
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
        out_p = np.zeros((6, 1), dtype=np.float64)
        out_m = np.zeros((6, 1), dtype=np.float64)
        out_w = np.zeros((6, 1), dtype=np.float64)

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
            out_w,
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
        out_ab = np.zeros(2, dtype=np.float64)
        out_ba = np.zeros(2, dtype=np.float64)
        out_ab_w = np.zeros(2, dtype=np.float64)
        out_ba_w = np.zeros(2, dtype=np.float64)

        _cpu_density_density_corr_kernel(
            density_a,
            density_b,
            w_a,
            w_b,
            ind_i,
            ind_j,
            offsets,
            out_ab,
            out_ba,
            out_ab_w,
            out_ba_w,
        )

        expected_ab = np.zeros_like(out_ab)
        expected_ba = np.zeros_like(out_ba)
        expected_ab_w = np.zeros_like(out_ab_w)
        expected_ba_w = np.zeros_like(out_ba_w)
        for b in range(offsets.shape[0] - 1):
            for idx in range(offsets[b], offsets[b + 1]):
                i = ind_i[idx]
                j = ind_j[idx]
                expected_ab[b] += w_a[i] * w_b[j] * density_a[i] * density_b[j]
                expected_ab_w[b] += w_a[i] * w_b[j]
                expected_ba[b] += w_a[j] * w_b[i] * density_a[j] * density_b[i]
                expected_ba_w[b] += w_a[j] * w_b[i]

        np.testing.assert_allclose(out_ab, expected_ab)
        np.testing.assert_allclose(out_ba, expected_ba)
        np.testing.assert_allclose(out_ab_w, expected_ab_w)
        np.testing.assert_allclose(out_ba_w, expected_ba_w)

    def test_cpu_density_shear_corr_kernel(self):
        density_lens = np.array([1.0, 2.0, 0.5], dtype=np.float64)
        g1_source = np.array([0.2, -0.1, 0.4], dtype=np.float64)
        g2_source = np.array([0.3, 0.5, -0.2], dtype=np.float64)
        w_lens = np.array([1.0, 0.5, 2.0], dtype=np.float64)
        w_source = np.array([1.0, 1.5, 0.25], dtype=np.float64)
        ind_i = np.array([0, 1, 2], dtype=np.int64)
        ind_j = np.array([1, 2, 0], dtype=np.int64)
        exp_i = np.array([0.9 - 0.3j, 0.5 + 0.5j, 0.2 + 0.7j], dtype=np.complex128)
        exp_j = np.array([0.6 + 0.8j, 1.0 + 0.0j, -0.5 + 0.5j], dtype=np.complex128)
        offsets = np.array([0, 2, 3], dtype=np.int64)
        out_ab = np.zeros(2, dtype=np.float64)
        out_ba = np.zeros(2, dtype=np.float64)
        out_ab_w = np.zeros(2, dtype=np.float64)
        out_ba_w = np.zeros(2, dtype=np.float64)

        _cpu_density_shear_corr_kernel(
            density_lens,
            g1_source,
            g2_source,
            w_lens,
            w_source,
            ind_i,
            ind_j,
            exp_i,
            exp_j,
            offsets,
            out_ab,
            out_ba,
            out_ab_w,
            out_ba_w,
        )

        expected_ab = np.zeros_like(out_ab)
        expected_ba = np.zeros_like(out_ba)
        expected_ab_w = np.zeros_like(out_ab_w)
        expected_ba_w = np.zeros_like(out_ba_w)
        for b in range(offsets.shape[0] - 1):
            for idx in range(offsets[b], offsets[b + 1]):
                i = ind_i[idx]
                j = ind_j[idx]
                gamma_t_ab = -g1_source[j] * exp_j[idx].real + g2_source[j] * exp_j[idx].imag
                expected_ab[b] += w_lens[i] * w_source[j] * density_lens[i] * gamma_t_ab
                expected_ab_w[b] += w_lens[i] * w_source[j]
                gamma_t_ba = -g1_source[i] * exp_i[idx].real + g2_source[i] * exp_i[idx].imag
                expected_ba[b] += w_lens[j] * w_source[i] * density_lens[j] * gamma_t_ba
                expected_ba_w[b] += w_lens[j] * w_source[i]

        np.testing.assert_allclose(out_ab, expected_ab)
        np.testing.assert_allclose(out_ba, expected_ba)
        np.testing.assert_allclose(out_ab_w, expected_ab_w)
        np.testing.assert_allclose(out_ba_w, expected_ba_w)

    def test_cpu_density_density_tomo_vectorized_kernel(self):
        density = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        weights = np.array([[1.0, 0.5], [2.0, 1.0]], dtype=np.float64)
        ind_i = np.array([0], dtype=np.int64)
        ind_j = np.array([1], dtype=np.int64)
        offsets = np.array([0, 1], dtype=np.int64)
        comb_i = np.array([0, 0, 1], dtype=np.int32)
        comb_j = np.array([0, 1, 1], dtype=np.int32)
        out_num = np.zeros((3, 1), dtype=np.float64)
        out_den = np.zeros((3, 1), dtype=np.float64)

        _cpu_density_density_tomo_vectorized_kernel(
            density,
            weights,
            ind_i,
            ind_j,
            offsets,
            comb_i,
            comb_j,
            out_num,
            out_den,
        )

        self.assertEqual(out_num.shape, (3, 1))
        self.assertEqual(out_den.shape, (3, 1))

        expected_num = np.zeros((3, 1), dtype=np.float64)
        expected_den = np.zeros((3, 1), dtype=np.float64)
        for k in range(3):
            i = comb_i[k]
            j = comb_j[k]
            for idx in range(1):
                pi, pj = ind_i[idx], ind_j[idx]
                w_ab = weights[pi, i] * weights[pj, j]
                ab = w_ab * density[pi, i] * density[pj, j]
                if i == j:
                    expected_num[k, 0] += ab
                    expected_den[k, 0] += w_ab
                else:
                    w_ba = weights[pi, j] * weights[pj, i]
                    ba = w_ba * density[pi, j] * density[pj, i]
                    expected_num[k, 0] += 0.5 * (ab + ba)
                    expected_den[k, 0] += 0.5 * (w_ab + w_ba)

        np.testing.assert_allclose(out_num, expected_num)
        np.testing.assert_allclose(out_den, expected_den)

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
        out_den = np.zeros((2, 1), dtype=np.float64)

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
            out_den,
        )

        self.assertEqual(out_num.shape, (2, 1))
        self.assertEqual(out_den.shape, (2, 1))
        self.assertNotEqual(out_num[0, 0], 0.0)

        expected_num = np.zeros((2, 1), dtype=np.float64)
        expected_den = np.zeros((2, 1), dtype=np.float64)
        for k in range(2):
            lens_bin = comb_i[k]
            source_bin = comb_j[k]
            for idx in range(1):
                pi, pj = ind_i[idx], ind_j[idx]
                gt_ab = (
                    -shear[pj, source_bin, 0] * rot_j[idx].real
                    + shear[pj, source_bin, 1] * rot_j[idx].imag
                )
                w_ab = lens_weights[pi, lens_bin] * source_weights[pj, source_bin]
                expected_num[k, 0] += w_ab * density[pi, lens_bin] * gt_ab
                expected_den[k, 0] += w_ab
                gt_ba = (
                    -shear[pi, source_bin, 0] * rot_i[idx].real
                    + shear[pi, source_bin, 1] * rot_i[idx].imag
                )
                w_ba = lens_weights[pj, lens_bin] * source_weights[pi, source_bin]
                expected_num[k, 0] += w_ba * density[pj, lens_bin] * gt_ba
                expected_den[k, 0] += w_ba

        np.testing.assert_allclose(out_num, expected_num)
        np.testing.assert_allclose(out_den, expected_den)

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

    def test_cpu_aperture_shear_kernel(self):
        Q_inds = np.array([0, 1, 2], dtype=np.int64)
        Q_cos = np.array([0.6, -0.2, 0.3], dtype=np.float64)
        Q_sin = np.array([0.8, 0.4, -0.5], dtype=np.float64)
        Q_val = np.array([0.2, 0.3, 0.5], dtype=np.float64)
        Q_offsets = np.array([0, 2, 3], dtype=np.int64)
        g1 = np.array([1.0, 2.0, 4.0], dtype=np.float64)
        g2 = np.array([0.5, -1.0, 0.25], dtype=np.float64)
        weights = np.array([1.0, 0.5, 2.0], dtype=np.float64)
        Q_patch_area = np.array([1.0, 2.0], dtype=np.float64)
        out = np.zeros(2, dtype=np.float64)

        _cpu_aperture_shear_kernel(
            Q_inds,
            Q_cos,
            Q_sin,
            Q_val,
            Q_offsets,
            g1,
            g2,
            weights,
            Q_patch_area,
            out,
        )

        gt0 = -g1[Q_inds[:2]] * Q_cos[:2] - g2[Q_inds[:2]] * Q_sin[:2]
        expected0 = (
            Q_patch_area[0]
            * np.sum(weights[Q_inds[:2]] * gt0 * Q_val[:2])
            / np.sum(weights[Q_inds[:2]])
        )
        gt1 = -g1[Q_inds[2:]] * Q_cos[2:] - g2[Q_inds[2:]] * Q_sin[2:]
        expected1 = (
            Q_patch_area[1]
            * np.sum(weights[Q_inds[2:]] * gt1 * Q_val[2:])
            / np.sum(weights[Q_inds[2:]])
        )
        np.testing.assert_allclose(out, np.array([expected0, expected1]))

    def test_cpu_3x2pt_tomo_fused_kernel(self):
        density_map = np.array(
            [
                [10.0],
                [20.0],
                [7.0],
            ],
            dtype=np.float64,
        )
        shear_map = np.array(
            [
                [[1.0, 0.0]],
                [[2.0, 0.0]],
                [[4.0, 0.0]],
            ],
            dtype=np.float64,
        )
        density_weights = np.array([[2.0], [3.0], [11.0]], dtype=np.float64)
        shear_weights = np.array([[3.0], [4.0], [5.0]], dtype=np.float64)

        ind_i = np.array([0], dtype=np.int64)
        ind_j = np.array([1], dtype=np.int64)
        rot_i = np.array([1.0 + 0.0j], dtype=np.complex128)
        rot_j = np.array([1.0 + 0.0j], dtype=np.complex128)
        pair_offsets = np.array([0, 1], dtype=np.int64)

        q_inds = np.array([2], dtype=np.uint32)
        q_cos = np.array([1.0], dtype=np.float64)
        q_sin = np.array([0.0], dtype=np.float64)
        q_val = np.array([2.0], dtype=np.float64)
        q_offsets = np.array([0, 1], dtype=np.int64)
        q_patch_area = np.array([3.0], dtype=np.float64)

        ss_comb_i = np.array([0], dtype=np.int32)
        ss_comb_j = np.array([0], dtype=np.int32)
        dd_comb_i = np.array([0], dtype=np.int32)
        dd_comb_j = np.array([0], dtype=np.int32)
        ds_comb_i = np.array([0], dtype=np.int32)
        ds_comb_j = np.array([0], dtype=np.int32)

        out_ma_num = np.zeros((1, 1), dtype=np.float64)
        out_ma_den = np.zeros((1, 1), dtype=np.float64)
        out_mg_num = np.zeros((1, 1), dtype=np.float64)
        out_mg_den = np.zeros((1, 1), dtype=np.float64)
        out_xip_num = np.zeros((2, 1), dtype=np.float64)
        out_xim_num = np.zeros((2, 1), dtype=np.float64)
        out_xipm_den = np.zeros((2, 1), dtype=np.float64)
        out_xig_num = np.zeros((2, 1), dtype=np.float64)
        out_xig_den = np.zeros((2, 1), dtype=np.float64)
        out_xit_num = np.zeros((1, 1), dtype=np.float64)
        out_xit_den = np.zeros((1, 1), dtype=np.float64)

        _cpu_3x2pt_tomo_fused_kernel(
            density_map,
            shear_map,
            density_weights,
            shear_weights,
            ind_i,
            ind_j,
            rot_i,
            rot_j,
            pair_offsets,
            q_inds,
            q_cos,
            q_sin,
            q_val,
            q_offsets,
            q_patch_area,
            ss_comb_i,
            ss_comb_j,
            dd_comb_i,
            dd_comb_j,
            ds_comb_i,
            ds_comb_j,
            out_ma_num,
            out_ma_den,
            out_mg_num,
            out_mg_den,
            out_xip_num,
            out_xim_num,
            out_xipm_den,
            out_xig_num,
            out_xig_den,
            out_xit_num,
            out_xit_den,
        )

        self.assertAlmostEqual(out_ma_num[0, 0], -120.0)
        self.assertAlmostEqual(out_ma_den[0, 0], 5.0)
        self.assertAlmostEqual(out_mg_num[0, 0], 462.0)
        self.assertAlmostEqual(out_mg_den[0, 0], 11.0)
        self.assertAlmostEqual(out_xip_num[0, 0], 24.0)
        self.assertAlmostEqual(out_xim_num[0, 0], 24.0)
        self.assertAlmostEqual(out_xipm_den[0, 0], 12.0)
        self.assertAlmostEqual(out_xig_num[0, 0], 1200.0)
        self.assertAlmostEqual(out_xig_den[0, 0], 6.0)
        self.assertAlmostEqual(out_xit_num[0, 0], -340.0)
        self.assertAlmostEqual(out_xit_den[0, 0], 17.0)

    def test_cpu_3x2pt_tomo_fused_kernel_density_cross_swap_orientation(self):
        density_map = np.array(
            [
                [2.0, 3.0],
                [5.0, 7.0],
            ],
            dtype=np.float64,
        )
        shear_map = np.zeros((2, 1, 2), dtype=np.float64)
        density_weights = np.array(
            [
                [11.0, 13.0],
                [17.0, 19.0],
            ],
            dtype=np.float64,
        )
        shear_weights = np.ones((2, 1), dtype=np.float64)

        ind_i = np.array([0], dtype=np.int64)
        ind_j = np.array([1], dtype=np.int64)
        rot_i = np.array([1.0 + 0.0j], dtype=np.complex128)
        rot_j = np.array([1.0 + 0.0j], dtype=np.complex128)
        pair_offsets = np.array([0, 1], dtype=np.int64)

        q_inds = np.array([], dtype=np.uint32)
        q_cos = np.array([], dtype=np.float64)
        q_sin = np.array([], dtype=np.float64)
        q_val = np.array([], dtype=np.float64)
        q_offsets = np.array([0, 0], dtype=np.int64)
        q_patch_area = np.array([1.0], dtype=np.float64)

        ss_comb_i = np.array([], dtype=np.int32)
        ss_comb_j = np.array([], dtype=np.int32)
        dd_comb_i = np.array([0, 0, 1], dtype=np.int32)
        dd_comb_j = np.array([0, 1, 1], dtype=np.int32)
        ds_comb_i = np.array([], dtype=np.int32)
        ds_comb_j = np.array([], dtype=np.int32)

        out_ma_num = np.zeros((1, 1), dtype=np.float64)
        out_ma_den = np.zeros((1, 1), dtype=np.float64)
        out_mg_num = np.zeros((2, 1), dtype=np.float64)
        out_mg_den = np.zeros((2, 1), dtype=np.float64)
        out_xip_num = np.zeros((0, 1), dtype=np.float64)
        out_xim_num = np.zeros((0, 1), dtype=np.float64)
        out_xipm_den = np.zeros((0, 1), dtype=np.float64)
        out_xig_num = np.zeros((6, 1), dtype=np.float64)
        out_xig_den = np.zeros((6, 1), dtype=np.float64)
        out_xit_num = np.zeros((0, 1), dtype=np.float64)
        out_xit_den = np.zeros((0, 1), dtype=np.float64)

        _cpu_3x2pt_tomo_fused_kernel(
            density_map,
            shear_map,
            density_weights,
            shear_weights,
            ind_i,
            ind_j,
            rot_i,
            rot_j,
            pair_offsets,
            q_inds,
            q_cos,
            q_sin,
            q_val,
            q_offsets,
            q_patch_area,
            ss_comb_i,
            ss_comb_j,
            dd_comb_i,
            dd_comb_j,
            ds_comb_i,
            ds_comb_j,
            out_ma_num,
            out_ma_den,
            out_mg_num,
            out_mg_den,
            out_xip_num,
            out_xim_num,
            out_xipm_den,
            out_xig_num,
            out_xig_den,
            out_xit_num,
            out_xit_den,
        )

        ab_idx = 2
        ba_idx = 3
        expected_den_ab = density_weights[0, 0] * density_weights[1, 1]
        expected_den_ba = density_weights[0, 1] * density_weights[1, 0]
        expected_num_ab = expected_den_ab * density_map[0, 0] * density_map[1, 1]
        expected_num_ba = expected_den_ba * density_map[0, 1] * density_map[1, 0]

        self.assertAlmostEqual(out_xig_den[ab_idx, 0], expected_den_ab)
        self.assertAlmostEqual(out_xig_den[ba_idx, 0], expected_den_ba)
        self.assertAlmostEqual(out_xig_num[ab_idx, 0], expected_num_ab)
        self.assertAlmostEqual(out_xig_num[ba_idx, 0], expected_num_ba)

    def test_cupy_3x2pt_tomo_fused_kernel_missing_rawkernel_returns_false(self):
        class FakeModule:
            float32 = np.float32
            int32 = np.int32
            complex64 = np.complex64

        kernel = _build_cupy_3x2pt_tomo_fused_kernel(FakeModule)
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
            np.zeros(1, dtype=np.uint32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.array([0, 1], dtype=np.int64),
            np.ones(1, dtype=np.float32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
        )
        self.assertFalse(ok)

    def test_cupy_3x2pt_tomo_fused_kernel_compile_failure_returns_false(self):
        compile_attempts = []

        class FakeModule:
            float32 = np.float32
            int32 = np.int32
            complex64 = np.complex64

            @staticmethod
            def RawKernel(*_args, **_kwargs):
                compile_attempts.append(1)
                raise RuntimeError("compile failed")

        kernel = _build_cupy_3x2pt_tomo_fused_kernel(FakeModule)
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
            np.zeros(1, dtype=np.uint32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.array([0, 1], dtype=np.int64),
            np.ones(1, dtype=np.float32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
        )
        self.assertFalse(ok)
        # Failed compilations are cached negatively: a second call must not
        # retry the (expensive) NVRTC compilation.
        args = (
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1, 2), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.complex64),
            np.zeros(1, dtype=np.complex64),
            np.array([0, 1], dtype=np.int64),
            np.zeros(1, dtype=np.uint32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.array([0, 1], dtype=np.int64),
            np.ones(1, dtype=np.float32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
        )
        attempts_after_first = len(compile_attempts)
        ok2 = kernel(*args)
        self.assertFalse(ok2)
        self.assertEqual(len(compile_attempts), attempts_after_first)

    @staticmethod
    def _make_fake_cuda_namespace():
        """Minimal stand-in for cupy.cuda: streams/events used by the
        per-section fused launches."""

        class FakeEvent:
            pass

        class FakeStream:
            def __init__(self, non_blocking=False):
                self.non_blocking = non_blocking

            def record(self):
                return FakeEvent()

            def wait_event(self, _event):
                return None

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        class FakeCuda:
            Stream = FakeStream

            @staticmethod
            def get_current_stream():
                return FakeStream()

        return FakeCuda

    def test_cupy_3x2pt_tomo_fused_kernel_success_and_cache(self):
        compile_calls = {"count": 0}
        launches = []

        class FakeKernel:
            def __call__(self, grid, block, args):
                launches.append((grid, block, args))

        class FakeModule:
            float32 = np.float32
            int32 = np.int32
            complex64 = np.complex64
            cuda = self._make_fake_cuda_namespace()

            @staticmethod
            def RawKernel(*_args, **_kwargs):
                compile_calls["count"] += 1
                return FakeKernel()

        kernel = _build_cupy_3x2pt_tomo_fused_kernel(FakeModule)
        args = (
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1, 2), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.complex64),
            np.zeros(1, dtype=np.complex64),
            np.array([0, 1], dtype=np.int64),
            np.zeros(1, dtype=np.uint32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.array([0, 1], dtype=np.int64),
            np.ones(1, dtype=np.float32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
        )
        self.assertTrue(kernel(*args))
        self.assertTrue(kernel(*args))
        self.assertEqual(compile_calls["count"], 1)
        # One launch per correlation section (5) for each of the two calls.
        self.assertEqual(len(launches), 10)
        # nbins_total=1, npatches=1, 1 tomo bin, 1 combination each:
        expected_grids = [
            (1, 1, 1),  # z=0 M_ap:  (npatches, n_shear_bins)
            (1, 1, 1),  # z=1 M_g:   (npatches, n_density_bins)
            (1, 2, 1),  # z=2 xi+/-: (nbins_total, 2 * n_ss_comb)
            (1, 2, 1),  # z=3 xi_g:  (nbins_total, 2 * n_dd_comb)
            (1, 1, 1),  # z=4 xi_t:  (nbins_total, n_ds_comb)
        ]
        for section, (launch, expected_grid) in enumerate(
            zip(launches[:5], expected_grids)
        ):
            grid, block, launch_args = launch
            self.assertEqual(grid, expected_grid)
            self.assertEqual(block, (256,))
            # The section selector is appended as the last kernel argument
            # (after the 32 wrapper args + 6 derived size scalars).
            self.assertEqual(int(launch_args[-1]), section)
            self.assertEqual(len(launch_args), len(args) + 7)

    def test_cupy_3x2pt_tomo_fused_kernel_complex128_branch(self):
        class FakeKernel:
            def __call__(self, _grid, _block, _args):
                return None

        class FakeModule:
            float32 = np.float32
            int32 = np.int32
            complex64 = np.complex64
            complex128 = np.complex128
            cuda = self._make_fake_cuda_namespace()

            @staticmethod
            def RawKernel(_source, _kernel_name, options=None):
                return FakeKernel()

        kernel = _build_cupy_3x2pt_tomo_fused_kernel(FakeModule)
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
            np.zeros(1, dtype=np.uint32),
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            np.array([0, 1], dtype=np.int64),
            np.ones(1, dtype=np.float64),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.zeros((1, 1), dtype=np.float64),
            np.zeros((1, 1), dtype=np.float64),
            np.zeros((1, 1), dtype=np.float64),
            np.zeros((1, 1), dtype=np.float64),
            np.zeros((2, 1), dtype=np.float64),
            np.zeros((2, 1), dtype=np.float64),
            np.zeros((2, 1), dtype=np.float64),
            np.zeros((2, 1), dtype=np.float64),
            np.zeros((2, 1), dtype=np.float64),
            np.zeros((1, 1), dtype=np.float64),
            np.zeros((1, 1), dtype=np.float64),
        )
        self.assertTrue(ok)

    def test_cupy_density_density_tomo_vectorized_kernel_failure_modes_return_false(self):
        cases = [
            {
                "name": "missing-rawkernel",
                "module": type("FakeModule", (), {"float32": np.float32}),
                "density_bins": 1,
            },
            {
                "name": "compile-failure",
                "module": type(
                    "FakeModule",
                    (),
                    {
                        "float32": np.float32,
                        "int32": np.int32,
                        "RawKernel": staticmethod(
                            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("compile failed"))
                        ),
                    },
                ),
                "density_bins": 1,
            },
            {
                "name": "too-many-bins",
                "module": type(
                    "FakeModule",
                    (),
                    {
                        "float32": np.float32,
                        "RawKernel": staticmethod(lambda *_args, **_kwargs: MagicMock()),
                    },
                ),
                "density_bins": _MAX_VECTOR_TOMO_BINS + 1,
            },
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                kernel = _build_cupy_density_density_tomo_vectorized_kernel(case["module"])
                ok = kernel(
                    np.zeros((1, case["density_bins"]), dtype=np.float32),
                    np.zeros((1, case["density_bins"]), dtype=np.float32),
                    np.zeros(1, dtype=np.int64),
                    np.zeros(1, dtype=np.int64),
                    np.array([0, 1], dtype=np.int64),
                    np.array([0], dtype=np.int32),
                    np.array([0], dtype=np.int32),
                    np.zeros((2, 1), dtype=np.float32),
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
            int32 = np.int32

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
            np.zeros((6, 1), dtype=np.float32),
        )
        self.assertTrue(ok)
        self.assertTrue(ok_cached)
        self.assertEqual(len(compiled_sources), 1)
        self.assertEqual(compile_calls["count"], 1)
        self.assertEqual(
            compiled_sources[0][1],
            "gpu_fused_tomo_reduce_dd<float, 2, long long>",
        )
        self.assertEqual(compiled_sources[0][2], ("--use_fast_math", "--std=c++14"))

    def test_cupy_density_shear_tomo_vectorized_kernel_failure_modes_return_false(self):
        cases = [
            {
                "name": "missing-rawkernel",
                "module": type(
                    "FakeModule", (), {"float32": np.float32, "complex64": np.complex64}
                ),
                "density_bins": 1,
            },
            {
                "name": "too-many-bins",
                "module": type(
                    "FakeModule",
                    (),
                    {
                        "float32": np.float32,
                        "complex64": np.complex64,
                        "RawKernel": staticmethod(lambda *_args, **_kwargs: MagicMock()),
                    },
                ),
                "density_bins": _MAX_VECTOR_TOMO_BINS + 1,
            },
            {
                "name": "compile-failure",
                "module": type(
                    "FakeModule",
                    (),
                    {
                        "float32": np.float32,
                        "int32": np.int32,
                        "complex64": np.complex64,
                        "RawKernel": staticmethod(
                            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("compile failed"))
                        ),
                    },
                ),
                "density_bins": 1,
            },
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                kernel = _build_cupy_density_shear_tomo_vectorized_kernel(case["module"])
                ok = kernel(
                    np.zeros((1, case["density_bins"]), dtype=np.float32),
                    np.zeros((1, case["density_bins"], 2), dtype=np.float32),
                    np.zeros((1, case["density_bins"]), dtype=np.float32),
                    np.zeros((1, case["density_bins"]), dtype=np.float32),
                    np.zeros(1, dtype=np.int64),
                    np.zeros(1, dtype=np.int64),
                    np.zeros(1, dtype=np.complex64),
                    np.zeros(1, dtype=np.complex64),
                    np.array([0, 1], dtype=np.int64),
                    np.array([0], dtype=np.int32),
                    np.array([0], dtype=np.int32),
                    np.zeros((1, 1), dtype=np.float32),
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
            int32 = np.int32
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
            np.zeros((1, 1), dtype=np.float32),
        )
        self.assertTrue(ok)
        self.assertEqual(len(compiled_sources), 1)
        self.assertEqual(
            compiled_sources[0][1],
            "gpu_fused_tomo_reduce_ds<float, cuFloatComplex, 2, 2, long long>",
        )
        self.assertEqual(compiled_sources[0][2], ("--use_fast_math", "--std=c++14"))

    def test_cupy_density_shear_tomo_vectorized_kernel_cache_reuse(self):
        compile_calls = {"count": 0}

        class FakeKernel:
            def __call__(self, _grid, _block, _args):
                return None

        class FakeModule:
            float32 = np.float32
            int32 = np.int32
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
            int32 = np.int32
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
            np.zeros((1, 1), dtype=np.float64),
        )
        self.assertTrue(ok)

    def test_cupy_tomo_vectorized_kernel_failure_modes_return_false(self):
        cases = [
            {
                "name": "missing-rawkernel",
                "module": type(
                    "FakeModule", (), {"float32": np.float32, "complex64": np.complex64}
                ),
                "shear_bins": 1,
            },
            {
                "name": "too-many-bins",
                "module": type(
                    "FakeModule",
                    (),
                    {
                        "float32": np.float32,
                        "complex64": np.complex64,
                        "RawKernel": staticmethod(lambda *_args, **_kwargs: MagicMock()),
                    },
                ),
                "shear_bins": _MAX_VECTOR_TOMO_BINS + 1,
            },
            {
                "name": "compile-failure",
                "module": type(
                    "FakeModule",
                    (),
                    {
                        "float32": np.float32,
                        "int32": np.int32,
                        "complex64": np.complex64,
                        "RawKernel": staticmethod(
                            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("compile failed"))
                        ),
                    },
                ),
                "shear_bins": 1,
            },
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                kernel = _build_cupy_tomo_vectorized_kernel(case["module"])
                ok = kernel(
                    np.zeros((1, case["shear_bins"], 2), dtype=np.float32),
                    np.zeros((1, case["shear_bins"]), dtype=np.float32),
                    np.zeros(1, dtype=np.int64),
                    np.zeros(1, dtype=np.int64),
                    np.zeros(1, dtype=np.complex64),
                    np.zeros(1, dtype=np.complex64),
                    np.array([0, 1], dtype=np.int64),
                    np.array([0], dtype=np.int32),
                    np.array([0], dtype=np.int32),
                    np.zeros((2, 2, 1), dtype=np.complex64),
                    np.zeros((2, 1), dtype=np.float32),
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
            int32 = np.int32
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
        out_den = np.zeros((6, 1), dtype=np.float32)

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
            out_den,
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
            out_den,
        )

        self.assertTrue(ok1)
        self.assertTrue(ok2)
        self.assertEqual(rawkernel_calls["count"], 1)
        self.assertEqual(len(launches), 2)
        # The launch tuple carries out_den directly after out_num.
        launch_args = launches[0][2]
        self.assertIs(launch_args[9], out_num)
        self.assertIs(launch_args[10], out_den)

    def test_cupy_tomo_vectorized_kernel_complex128(self):
        class FakeKernel:
            def __call__(self, _grid, _block, _args):
                return None

        class FakeModule:
            float32 = np.float32
            int32 = np.int32
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
        out_den = np.zeros((2, 1), dtype=np.float64)

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
            out_den,
        )
        self.assertTrue(ok)

    def test_cupy_tomo_vectorized_kernel_templates_exact_bin_count(self):
        compiled_sources = []

        class FakeKernel:
            def __call__(self, _grid, _block, _args):
                return None

        class FakeModule:
            float32 = np.float32
            int32 = np.int32
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
        out_den_2 = np.zeros((6, 1), dtype=np.float32)

        shear_3 = np.zeros((1, 3, 2), dtype=np.float32)
        weights_3 = np.zeros((1, 3), dtype=np.float32)
        comb_i_3 = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
        comb_j_3 = np.array([0, 1, 2, 1, 2, 2], dtype=np.int32)
        out_num_3 = np.zeros((2, 12, 1), dtype=np.complex64)
        out_den_3 = np.zeros((12, 1), dtype=np.float32)

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
            out_den_2,
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
            out_den_3,
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
            out_den_2,
        )

        self.assertTrue(ok_2)
        self.assertTrue(ok_3)
        self.assertTrue(ok_2_cached)
        self.assertEqual(len(compiled_sources), 2)
        self.assertEqual(
            compiled_sources[0][1],
            "gpu_fused_tomo_reduce_xipm<float, cuFloatComplex, 2, long long>",
        )
        self.assertEqual(
            compiled_sources[1][1],
            "gpu_fused_tomo_reduce_xipm<float, cuFloatComplex, 3, long long>",
        )
        for _source, _kernel_name, options in compiled_sources:
            self.assertEqual(options, ("--use_fast_math", "--std=c++14"))

if __name__ == "__main__":
    unittest.main()
