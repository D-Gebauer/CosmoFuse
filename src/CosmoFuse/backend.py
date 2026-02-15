import logging
import warnings
from typing import Any, Optional, Union

import numpy as np
from numba import njit

logger = logging.getLogger(__name__)


@njit(fastmath=True)
def _cpu_fused_cross_corr_kernel_c64(
    g1a: np.ndarray,
    g2a: np.ndarray,
    g1b: np.ndarray,
    g2b: np.ndarray,
    wa: np.ndarray,
    wb: np.ndarray,
    ind_i: np.ndarray,
    ind_j: np.ndarray,
    exp_i: np.ndarray,
    exp_j: np.ndarray,
    out_ab_p: np.ndarray,
    out_ab_m: np.ndarray,
    out_ba_p: np.ndarray,
    out_ba_m: np.ndarray,
) -> None:
    for idx in range(ind_i.shape[0]):
        i = ind_i[idx]
        j = ind_j[idx]

        ga_i = np.complex64(g1a[i] + 1j * g2a[i])
        gb_i = np.complex64(g1b[i] + 1j * g2b[i])
        ga_j = np.complex64(g1a[j] + 1j * g2a[j])
        gb_j = np.complex64(g1b[j] + 1j * g2b[j])

        ga_i_rot = wa[i] * ga_i * exp_i[idx]
        gb_i_rot = wb[i] * gb_i * exp_i[idx]
        ga_j_rot = wa[j] * ga_j * exp_j[idx]
        gb_j_rot = wb[j] * gb_j * exp_j[idx]

        out_ab_p[idx] = gb_j_rot * np.conjugate(ga_i_rot)
        out_ab_m[idx] = gb_j_rot * ga_i_rot
        out_ba_p[idx] = ga_j_rot * np.conjugate(gb_i_rot)
        out_ba_m[idx] = ga_j_rot * gb_i_rot


@njit(fastmath=True)
def _cpu_fused_cross_corr_kernel_c128(
    g1a: np.ndarray,
    g2a: np.ndarray,
    g1b: np.ndarray,
    g2b: np.ndarray,
    wa: np.ndarray,
    wb: np.ndarray,
    ind_i: np.ndarray,
    ind_j: np.ndarray,
    exp_i: np.ndarray,
    exp_j: np.ndarray,
    out_ab_p: np.ndarray,
    out_ab_m: np.ndarray,
    out_ba_p: np.ndarray,
    out_ba_m: np.ndarray,
) -> None:
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

        out_ab_p[idx] = gb_j_rot * np.conjugate(ga_i_rot)
        out_ab_m[idx] = gb_j_rot * ga_i_rot
        out_ba_p[idx] = ga_j_rot * np.conjugate(gb_i_rot)
        out_ba_m[idx] = ga_j_rot * gb_i_rot


def _cpu_fused_cross_corr_kernel(
    g1a: np.ndarray,
    g2a: np.ndarray,
    g1b: np.ndarray,
    g2b: np.ndarray,
    wa: np.ndarray,
    wb: np.ndarray,
    ind_i: np.ndarray,
    ind_j: np.ndarray,
    exp_i: np.ndarray,
    exp_j: np.ndarray,
    out_ab_p: np.ndarray,
    out_ab_m: np.ndarray,
    out_ba_p: np.ndarray,
    out_ba_m: np.ndarray,
) -> None:
    if exp_i.dtype == np.complex64:
        _cpu_fused_cross_corr_kernel_c64(
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
    else:
        _cpu_fused_cross_corr_kernel_c128(
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


@njit(fastmath=True)
def _cpu_xipm_kernel_c64(
    g11: np.ndarray,
    g21: np.ndarray,
    g12: np.ndarray,
    g22: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    ind_i: np.ndarray,
    ind_j: np.ndarray,
    exp_i: np.ndarray,
    exp_j: np.ndarray,
    out_p: np.ndarray,
    out_m: np.ndarray,
) -> None:
    for idx in range(ind_i.shape[0]):
        i = ind_i[idx]
        j = ind_j[idx]

        g2 = w1[i] * np.complex64(g11[i] + 1j * g21[i]) * exp_i[idx]
        g1 = w2[j] * np.complex64(g12[j] + 1j * g22[j]) * exp_j[idx]

        out_p[idx] = g1 * np.conjugate(g2)
        out_m[idx] = g1 * g2


@njit(fastmath=True)
def _cpu_xipm_kernel_c128(
    g11: np.ndarray,
    g21: np.ndarray,
    g12: np.ndarray,
    g22: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    ind_i: np.ndarray,
    ind_j: np.ndarray,
    exp_i: np.ndarray,
    exp_j: np.ndarray,
    out_p: np.ndarray,
    out_m: np.ndarray,
) -> None:
    for idx in range(ind_i.shape[0]):
        i = ind_i[idx]
        j = ind_j[idx]

        g2 = w1[i] * np.complex128(g11[i] + 1j * g21[i]) * exp_i[idx]
        g1 = w2[j] * np.complex128(g12[j] + 1j * g22[j]) * exp_j[idx]

        out_p[idx] = g1 * np.conjugate(g2)
        out_m[idx] = g1 * g2


def _cpu_xipm_kernel(
    g11: np.ndarray,
    g21: np.ndarray,
    g12: np.ndarray,
    g22: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    ind_i: np.ndarray,
    ind_j: np.ndarray,
    exp_i: np.ndarray,
    exp_j: np.ndarray,
    out_p: np.ndarray,
    out_m: np.ndarray,
) -> None:
    if exp_i.dtype == np.complex64:
        _cpu_xipm_kernel_c64(
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
            out_p,
            out_m,
        )
    else:
        _cpu_xipm_kernel_c128(
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
            out_p,
            out_m,
        )


def _build_cupy_fused_cross_corr_kernel(module: Any) -> Any:
    return module.ElementwiseKernel(
        "raw T g1a, raw T g2a, raw T g1b, raw T g2b, raw T wa, raw T wb,"
        " raw I ind_i, raw I ind_j, raw C exp_i, raw C exp_j",
        "C out_ab_p, C out_ab_m, C out_ba_p, C out_ba_m",
        """
        const I idx_i = ind_i[i];
        const I idx_j = ind_j[i];

        C ga_i = C(g1a[idx_i], g2a[idx_i]);
        C gb_i = C(g1b[idx_i], g2b[idx_i]);
        C ga_j = C(g1a[idx_j], g2a[idx_j]);
        C gb_j = C(g1b[idx_j], g2b[idx_j]);

        C exp_i_val = exp_i[i];
        C exp_j_val = exp_j[i];

        C ga_i_rot = C(g1a[idx_i] * wa[idx_i], g2a[idx_i] * wa[idx_i]) * exp_i_val;
        C gb_i_rot = C(g1b[idx_i] * wb[idx_i], g2b[idx_i] * wb[idx_i]) * exp_i_val;
        C ga_j_rot = C(g1a[idx_j] * wa[idx_j], g2a[idx_j] * wa[idx_j]) * exp_j_val;
        C gb_j_rot = C(g1b[idx_j] * wb[idx_j], g2b[idx_j] * wb[idx_j]) * exp_j_val;

        out_ab_p = gb_j_rot * conj(ga_i_rot);
        out_ab_m = gb_j_rot * ga_i_rot;
        out_ba_p = ga_j_rot * conj(gb_i_rot);
        out_ba_m = ga_j_rot * gb_i_rot;
        """,
        "fused_cross_corr",
    )


def _build_cupy_xipm_kernel(module: Any) -> Any:
    return module.ElementwiseKernel(
        "raw T g11, raw T g21, raw T g12, raw T g22, raw T w1, raw T w2,"
        " raw I ind_i, raw I ind_j, raw C exp_i, raw C exp_j",
        "C out_p, C out_m",
        """
        const I idx_i = ind_i[i];
        const I idx_j = ind_j[i];

        C g2 = C(g11[idx_i] * w1[idx_i], g21[idx_i] * w1[idx_i]) * exp_i[i];
        C g1 = C(g12[idx_j] * w2[idx_j], g22[idx_j] * w2[idx_j]) * exp_j[i];

        out_p = g1 * conj(g2);
        out_m = g1 * g2;
        """,
        "xipm_kernel",
    )

class Backend:
    def __init__(
        self,
        name: str,
        module: Any,
        device_id: Optional[int] = None,
        fused_cross_corr_kernel: Optional[Any] = None,
        xipm_kernel: Optional[Any] = None,
    ) -> None:
        self.name = name
        self.module = module
        self.device_id = device_id
        self.fused_cross_corr_kernel = fused_cross_corr_kernel
        self.xipm_kernel = xipm_kernel

        self.asarray = module.asarray
        self.zeros = module.zeros
        self.ones = module.ones
        self.sum = module.sum
        self.mean = module.mean
        self.conjugate = module.conjugate
        self.add = module.add
        self.float32 = module.float32
        self.float64 = module.float64
        self.complex64 = module.complex64
        self.complex128 = module.complex128
        self.uint32 = module.uint32
        self.int32 = module.int32

    def to_device(self, array: Any) -> Any:
        """Move a numpy array to the backend device."""
        if self.name == 'numpy':
            return np.asarray(array)
        elif self.name == 'cupy':
            with self.module.cuda.Device(self.device_id):
                return self.module.asarray(array)
        return array

    def to_numpy(self, array: Any) -> np.ndarray:
        """Move an array from the backend device to numpy."""
        if self.name == 'numpy':
            return np.asarray(array)
        elif self.name == 'cupy':
            return self.module.asnumpy(array)
        return np.asarray(array)

    def get_memory_pool(self) -> Optional[Any]:
        if self.name == 'cupy':
            return self.module.get_default_memory_pool()
        return None

def get_backend(device: Union[str, int] = 'auto') -> "Backend":
    """
    Get the appropriate backend (numpy or cupy).

    Args:
        device (str or int): 'cpu', 'gpu', 'auto', or a GPU ID (int).
                             If 'gpu' is specified, it uses the first available GPU (ID 0).
                             If an int is provided, it uses that specific GPU ID.

    Returns:
        Backend: An object wrapping the numpy/cupy module with helper methods.
    """
    if isinstance(device, int):
        device_id = device
        device_type = 'gpu'
    elif device.lower() == 'gpu':
        device_id = 0
        device_type = 'gpu'
    elif device.lower() == 'cpu':
        device_id = None
        device_type = 'cpu'
    elif device.lower() == 'auto':
        try:
            import cupy
            device_id = 0
            device_type = 'gpu'
        except ImportError:
            warnings.warn("Cupy not installed, falling back to CPU (numpy).")
            device_id = None
            device_type = 'cpu'
    else:
        raise ValueError(f"Unknown device: {device}")

    if device_type == 'cpu':
        return Backend(
            'numpy',
            np,
            fused_cross_corr_kernel=_cpu_fused_cross_corr_kernel,
            xipm_kernel=_cpu_xipm_kernel,
        )

    elif device_type == 'gpu':
        try:
            import cupy
            # Check if the requested device is available
            if device_id >= cupy.cuda.runtime.getDeviceCount():
                raise ValueError(f"GPU ID {device_id} not found.")

            return Backend(
                'cupy',
                cupy,
                device_id,
                fused_cross_corr_kernel=_build_cupy_fused_cross_corr_kernel(cupy),
                xipm_kernel=_build_cupy_xipm_kernel(cupy),
            )
        except ImportError:
            if device == 'auto':
                warnings.warn("Cupy not installed, falling back to CPU (numpy).")
                return Backend('numpy', np)
            else:
                raise ImportError("Cupy not installed but GPU requested.")
