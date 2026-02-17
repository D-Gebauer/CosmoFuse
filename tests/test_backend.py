
import sys
import unittest
from unittest.mock import MagicMock, patch
import warnings
import importlib

import numpy as np

import CosmoFuse.backend
from CosmoFuse.backend import Backend, _cpu_fused_cross_corr_kernel, get_backend


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

    def test_cpu_fused_cross_corr_kernel_complex128_dispatch(self):
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

        out_ab_p = np.zeros(ind_i.shape[0], dtype=np.complex128)
        out_ab_m = np.zeros(ind_i.shape[0], dtype=np.complex128)
        out_ba_p = np.zeros(ind_i.shape[0], dtype=np.complex128)
        out_ba_m = np.zeros(ind_i.shape[0], dtype=np.complex128)

        _cpu_fused_cross_corr_kernel(
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

if __name__ == "__main__":
    unittest.main()
