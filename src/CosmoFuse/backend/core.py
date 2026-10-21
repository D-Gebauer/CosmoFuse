"""The :class:`Backend` (array module + active kernel set) and
:func:`get_backend`."""

import warnings
from contextlib import nullcontext
from typing import Any, Optional, Union

import numpy as np

from .aperture import (
    _build_cupy_aperture_density_kernel,
    _build_cupy_aperture_shear_kernel,
    _build_cupy_aperture_tomo_density_kernel,
    _build_cupy_aperture_tomo_shear_kernel,
    _cpu_aperture_density_kernel,
    _cpu_aperture_shear_kernel,
)
from .degrade import _build_cupy_degrade_rows_kernel
from .density import (
    _build_cupy_density_density_corr_kernel,
    _build_cupy_density_density_tomo_packed_kernel,
    _build_cupy_density_density_tomo_vectorized_kernel,
    _build_cupy_density_shear_corr_kernel,
    _build_cupy_density_shear_tomo_packed_kernel,
    _build_cupy_density_shear_tomo_vectorized_kernel,
    _cpu_density_density_corr_kernel,
    _cpu_density_density_tomo_vectorized_kernel,
    _cpu_density_shear_corr_kernel,
    _cpu_density_shear_tomo_vectorized_kernel,
)
from .shear import (
    _build_cupy_tomo_packed_kernel,
    _build_cupy_tomo_vectorized_kernel,
    _build_cupy_xipm_auto_corr_kernel,
    _build_cupy_xipm_cross_corr_kernel,
    _cpu_vectorized_tomo_kernel,
    _cpu_xipm_auto_corr_kernel,
    _cpu_xipm_cross_corr_kernel,
)
from .three_by_two import (
    _build_cupy_3x2pt_tomo_aperture_kernel,
    _build_cupy_3x2pt_tomo_pairs_kernel,
    _cpu_3x2pt_tomo_fused_kernel,
)
from .threepoint import _cpu_3pcf_multipoles, build_cupy_3pcf_multipoles_kernel

_SAFE_DIVIDE_KERNEL: Optional[Any] = None


def _build_safe_divide_kernel(module: Any) -> Any:
    """``num / den``, zero where ``den == 0``, in one launch.

    The expression this replaces -- ``den != 0``, ``where``, ``divide``,
    ``astype``, ``*=`` -- is five kernels and two full-size temporaries per
    output array, and the calls that use it produce six output arrays.  At
    production shapes that is 0.78 ms of pure launch overhead on a ~4 ms call.

    Written out element by element it is one kernel and no temporary, and it
    is deliberately *the same expression* rather than a tidier one: the
    ``den == 0`` branch still multiplies the quotient by zero instead of
    assigning zero, so a negative numerator still yields ``-0.0`` and a
    non-finite one still yields ``NaN``.  That matters because these outputs
    are on the ``resolution_factor=None`` path, which must stay bit-for-bit
    identical to 4.20.0.
    """
    global _SAFE_DIVIDE_KERNEL
    if _SAFE_DIVIDE_KERNEL is None:
        _SAFE_DIVIDE_KERNEL = module.ElementwiseKernel(
            "T num, U den",
            "V out",
            """
            const U safe_den = (den != (U)0) ? den : (U)1;
            const V keep = (den != (U)0) ? (V)1 : (V)0;
            out = (V)(num / safe_den) * keep;
            """,
            "cosmofuse_safe_divide",
        )
    return _SAFE_DIVIDE_KERNEL


def _is_device_array(module: Any, array: Any) -> bool:
    return type(array).__module__.split(".")[0] == module.__name__.split(".")[0]


def safe_divide(
    module: Any,
    num: Any,
    den: Any,
    out: Optional[Any] = None,
    dtype: Optional[Any] = None,
) -> Any:
    """``num / den`` with zeros where ``den == 0``.

    One launch on a GPU (see :func:`_build_safe_divide_kernel`) instead of the
    five the equivalent array expression costs, and no full-size temporaries.
    Bit-for-bit what the expression produced, including the sign of the zeros
    and the propagation of non-finite numerators.

    ``den`` may be a scalar or broadcast against *num*; *out* may have a
    different dtype (the fused 3x2pt path divides float64 accumulators into
    float32 outputs).  Without *out* the result is allocated at the broadcast
    shape and at *dtype*, defaulting to ``num.dtype``.

    Takes the array module rather than a :class:`Backend` so that anything
    holding a numpy-backed stand-in keeps the host expression.
    """
    if (
        hasattr(module, "ElementwiseKernel")
        and _is_device_array(module, num)
        and _is_device_array(module, den)
    ):
        if out is None:
            shape = np.broadcast_shapes(num.shape, den.shape)
            out = module.empty(shape, dtype=num.dtype if dtype is None else dtype)
        _build_safe_divide_kernel(module)(num, den, out)
        return out

    mask = den != 0
    if out is None:
        safe_den = module.where(mask, den, 1)
        result_dtype = num.dtype if dtype is None else dtype
        return ((num / safe_den) * mask).astype(result_dtype, copy=False)
    safe_den = module.where(mask, den, 1)
    module.divide(num, safe_den, out=out)
    out *= module.asarray(mask).astype(out.dtype, copy=False)
    return out


class Backend:
    def __init__(
        self,
        name: str,
        module: Any,
        device_id: Optional[int] = None,
        xipm_cross_corr_kernel: Optional[Any] = None,
        xipm_auto_corr_kernel: Optional[Any] = None,
        xipm_tomo_vectorized_kernel: Optional[Any] = None,
        xipm_tomo_packed_kernel: Optional[Any] = None,
        aperture_density_kernel: Optional[Any] = None,
        aperture_shear_kernel: Optional[Any] = None,
        aperture_tomo_shear_kernel: Optional[Any] = None,
        aperture_tomo_density_kernel: Optional[Any] = None,
        kernel_density_density: Optional[Any] = None,
        kernel_density_shear: Optional[Any] = None,
        kernel_density_density_tomo_vectorized: Optional[Any] = None,
        kernel_density_shear_tomo_vectorized: Optional[Any] = None,
        kernel_density_density_tomo_packed: Optional[Any] = None,
        kernel_density_shear_tomo_packed: Optional[Any] = None,
        kernel_3x2pt_tomo_fused: Optional[Any] = None,
        kernel_3x2pt_tomo_aperture: Optional[Any] = None,
        kernel_3x2pt_tomo_pairs: Optional[Any] = None,
        degrade_rows_kernel: Optional[Any] = None,
        kernel_3pcf_multipoles: Optional[Any] = None,
    ) -> None:
        self.name = name
        self.module = module
        self.device_id = device_id
        self.xipm_cross_corr_kernel = xipm_cross_corr_kernel
        self.xipm_auto_corr_kernel = xipm_auto_corr_kernel
        self.xipm_tomo_vectorized_kernel = xipm_tomo_vectorized_kernel
        self.xipm_tomo_packed_kernel = xipm_tomo_packed_kernel
        self.aperture_density_kernel = aperture_density_kernel
        self.aperture_shear_kernel = aperture_shear_kernel
        self.aperture_tomo_shear_kernel = aperture_tomo_shear_kernel
        self.aperture_tomo_density_kernel = aperture_tomo_density_kernel
        self.kernel_density_density = kernel_density_density
        self.kernel_density_shear = kernel_density_shear
        self.kernel_density_density_tomo_vectorized = (
            kernel_density_density_tomo_vectorized
        )
        self.kernel_density_shear_tomo_vectorized = kernel_density_shear_tomo_vectorized
        self.kernel_density_density_tomo_packed = kernel_density_density_tomo_packed
        self.kernel_density_shear_tomo_packed = kernel_density_shear_tomo_packed
        # CPU: the Numba kernel computing all six 3x2pt outputs.  GPU: the
        # aperture sections; the pair statistics use the tiled pair kernels.
        self.kernel_3x2pt_tomo_fused = kernel_3x2pt_tomo_fused
        self.kernel_3x2pt_tomo_aperture = kernel_3x2pt_tomo_aperture
        self.kernel_3x2pt_tomo_pairs = kernel_3x2pt_tomo_pairs
        # Fused static-treecode row degrade (None on CPU / without a
        # raw compiler; the sparse chain is then used instead).
        self.degrade_rows_kernel = degrade_rows_kernel
        # Step 1 of the multipole three-point estimator (Global3PCF).
        self.kernel_3pcf_multipoles = kernel_3pcf_multipoles

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

    def to_device(self, array: Any, stream: Optional[Any] = None) -> Any:
        """Move a numpy array to the backend device.

        When *stream* is provided and the backend is CuPy, the transfer is
        scheduled on that CUDA stream.  If *array* is backed by pinned
        (page-locked) host memory the transfer is truly asynchronous —
        the call returns before the copy completes.
        """
        if self.name == "numpy":
            return np.asarray(array)
        elif self.name == "cupy":
            with self.module.cuda.Device(self.device_id):
                if stream is not None:
                    with stream:
                        return self.module.asarray(array)
                return self.module.asarray(array)
        return array

    def to_numpy(self, array: Any, stream: Optional[Any] = None) -> np.ndarray:
        """Move an array from the backend device to numpy.

        When *stream* is provided and the backend is CuPy, the transfer is
        scheduled on that CUDA stream.
        """
        if self.name == "numpy":
            return np.asarray(array)
        elif self.name == "cupy":
            if stream is not None:
                with stream:
                    return self.module.asnumpy(array)
            return self.module.asnumpy(array)
        return np.asarray(array)

    def get_memory_pool(self) -> Optional[Any]:
        if self.name == "cupy":
            return self.module.get_default_memory_pool()
        return None

    # ── CUDA stream management ──────────────────────────────────────────

    def create_stream(self, non_blocking: bool = True) -> Optional[Any]:
        """Create a CUDA stream for overlapping transfers and kernel execution.

        Returns ``None`` on CPU backends.
        """
        if self.name != "cupy":
            return None
        with self.module.cuda.Device(self.device_id):
            return self.module.cuda.Stream(non_blocking=non_blocking)

    def synchronize_stream(self, stream: Optional[Any] = None) -> None:
        """Wait for all operations on *stream* to complete.

        If *stream* is ``None``, synchronizes the current device.
        No-op on CPU backends.
        """
        if self.name != "cupy":
            return
        with self.module.cuda.Device(self.device_id):
            if stream is not None:
                stream.synchronize()
            else:
                self.module.cuda.runtime.deviceSynchronize()

    def use_stream(self, stream: Optional[Any] = None) -> Any:
        """Return a context manager that sets the active CUDA stream.

        All CuPy operations (kernel launches, memory copies) executed
        inside the context will be enqueued on *stream*.  On CPU backends
        or when *stream* is ``None`` this returns a no-op context.
        """
        if self.name == "cupy" and stream is not None:
            return stream
        return nullcontext()

    # ── Pinned (page-locked) host memory ────────────────────────────────

    def alloc_pinned(self, shape: Any, dtype: Any) -> np.ndarray:
        """Allocate page-locked (pinned) host memory.

        Returns a numpy array backed by pinned memory.  When this array
        is passed to :meth:`to_device` with a CUDA *stream*, the host-to-
        device copy can proceed asynchronously (DMA without CPU staging).

        On CPU backends returns a regular numpy array.
        """
        if self.name != "cupy":
            return np.empty(shape, dtype=dtype)
        dtype = np.dtype(dtype)
        count = int(np.prod(shape))
        nbytes = count * dtype.itemsize
        mem = self.module.cuda.alloc_pinned_memory(nbytes)
        return np.frombuffer(mem, dtype=dtype, count=count).reshape(shape)

    def warmup(
        self,
        map_dtype: Any = np.float64,
        rotation_dtype: Any = np.float32,
        rotation_complex_dtype: Any = np.complex64,
        index_dtype: Any = np.int32,
    ) -> None:
        """Eagerly JIT-compile the CPU measurement kernels.

        The Numba kernels compile lazily on first use, which otherwise
        serializes ~10 parallel-kernel compilations with the first map
        measurement.  Calling this once after constructing the backend
        (with the dtypes that will be used) moves that cost out of the
        measured path.  ``cache=True`` makes subsequent processes load
        the compiled kernels from disk, so this is cheap after the first
        run in a given environment.  No-op on GPU backends (CUDA kernels
        are compiled per tomographic-bin-count template on first launch).
        """
        if self.name != "numpy":
            return

        map_dtype = np.dtype(map_dtype)
        rotation_dtype = np.dtype(rotation_dtype)
        rotation_complex_dtype = np.dtype(rotation_complex_dtype)
        index_dtype = np.dtype(index_dtype)

        npix, nz = 4, 2
        m = np.zeros((npix, nz), dtype=map_dtype)
        shear = np.zeros((npix, nz, 2), dtype=map_dtype)
        v1 = np.zeros(npix, dtype=map_dtype)
        inds = np.zeros(2, dtype=index_dtype)
        rot = np.ones(2, dtype=rotation_complex_dtype)
        offsets = np.array([0, 2], dtype=np.int64)
        comb = np.array([0], dtype=np.int32)
        q_val = np.zeros(2, dtype=rotation_dtype)
        q_offsets = np.array([0, 2], dtype=np.int64)
        q_area = np.ones(1, dtype=rotation_dtype)
        ones1 = np.ones(npix, dtype=map_dtype)

        def outb(*shape: int) -> np.ndarray:
            return np.zeros(shape, dtype=map_dtype)

        self.aperture_density_kernel(inds, q_val, q_offsets, v1, ones1, q_area, outb(1))
        self.aperture_shear_kernel(
            inds, q_val, q_val, q_val, q_offsets, v1, v1, ones1, q_area, outb(1)
        )
        self.kernel_density_density(
            v1,
            v1,
            ones1,
            ones1,
            inds,
            inds,
            offsets,
            outb(1),
            outb(1),
            outb(1),
            outb(1),
        )
        self.kernel_density_shear(
            v1,
            v1,
            v1,
            ones1,
            ones1,
            inds,
            inds,
            rot,
            rot,
            offsets,
            outb(1),
            outb(1),
            outb(1),
            outb(1),
        )
        self.xipm_auto_corr_kernel(
            v1,
            v1,
            v1,
            v1,
            ones1,
            ones1,
            inds,
            inds,
            rot,
            rot,
            offsets,
            outb(1),
            outb(1),
            outb(1),
        )
        self.xipm_cross_corr_kernel(
            v1,
            v1,
            v1,
            v1,
            ones1,
            ones1,
            inds,
            inds,
            rot,
            rot,
            offsets,
            outb(1),
            outb(1),
            outb(1),
            outb(1),
            outb(1),
            outb(1),
        )
        self.kernel_density_density_tomo_vectorized(
            m, m, inds, inds, offsets, comb, comb, outb(1, 1), outb(1, 1)
        )
        self.kernel_density_shear_tomo_vectorized(
            m,
            shear,
            m,
            m,
            inds,
            inds,
            rot,
            rot,
            offsets,
            comb,
            comb,
            outb(1, 1),
            outb(1, 1),
        )
        self.xipm_tomo_vectorized_kernel(
            shear,
            m,
            inds,
            inds,
            rot,
            rot,
            offsets,
            comb,
            comb,
            outb(1, 1),
            outb(1, 1),
            outb(1, 1),
        )
        self.kernel_3x2pt_tomo_fused(
            m,
            shear,
            m,
            m,
            inds,
            inds,
            rot,
            rot,
            offsets,
            inds.astype(np.uint32),
            q_val.astype(map_dtype),
            q_val.astype(map_dtype),
            q_val.astype(map_dtype),
            q_offsets,
            q_area.astype(map_dtype),
            comb,
            comb,
            comb,
            comb,
            comb,
            comb,
            outb(nz, 1),
            outb(nz, 1),
            outb(nz, 1),
            outb(nz, 1),
            outb(1, 1),
            outb(1, 1),
            outb(1, 1),
            outb(1, 1),
            outb(1, 1),
            outb(1, 1),
            outb(1, 1),
        )


def get_backend(device: Union[str, int] = "auto") -> "Backend":
    """
    Get the appropriate backend (numpy or cupy).

    Args:
        device (str or int): 'cpu', 'gpu', 'auto', or a GPU ID (int).
                             If 'gpu' is specified, the first
                             available GPU (ID 0) is used.
                             If an int is provided, it uses that specific GPU ID.

    Returns:
        Backend: An object wrapping the numpy/cupy module with helper methods.
    """
    if isinstance(device, int):
        device_id = device
        device_type = "gpu"
    elif device.lower() == "gpu":
        device_id = 0
        device_type = "gpu"
    elif device.lower() == "cpu":
        device_id = None
        device_type = "cpu"
    elif device.lower() == "auto":
        # 'auto' promises a working backend, so probe for a usable *device*
        # and not merely for an importable cupy.  A driver/runtime mismatch,
        # CUDA_VISIBLE_DEVICES="", an unusable card -- all of these import
        # cleanly and then raise CUDARuntimeError on first use, which is a
        # crash where the caller asked for a fallback.  'gpu' and an explicit
        # device id stay strict: an explicit request must fail loudly.
        try:
            import cupy

            if cupy.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA device is visible")
            device_id = 0
            device_type = "gpu"
        except ImportError:
            warnings.warn("Cupy not installed, falling back to CPU (numpy).")
            device_id = None
            device_type = "cpu"
        except Exception as exc:
            warnings.warn(
                f"Cupy is installed but no usable CUDA device was found "
                f"({type(exc).__name__}: {exc}); falling back to CPU (numpy). "
                f"Pass device='gpu' or a device id to make this an error."
            )
            device_id = None
            device_type = "cpu"
    else:
        raise ValueError(f"Unknown device: {device}")

    if device_type == "cpu":
        return Backend(
            "numpy",
            np,
            xipm_cross_corr_kernel=_cpu_xipm_cross_corr_kernel,
            xipm_auto_corr_kernel=_cpu_xipm_auto_corr_kernel,
            xipm_tomo_vectorized_kernel=_cpu_vectorized_tomo_kernel,
            aperture_density_kernel=_cpu_aperture_density_kernel,
            aperture_shear_kernel=_cpu_aperture_shear_kernel,
            kernel_density_density=_cpu_density_density_corr_kernel,
            kernel_density_shear=_cpu_density_shear_corr_kernel,
            kernel_density_density_tomo_vectorized=_cpu_density_density_tomo_vectorized_kernel,  # noqa: E501
            kernel_density_shear_tomo_vectorized=_cpu_density_shear_tomo_vectorized_kernel,  # noqa: E501
            kernel_3x2pt_tomo_fused=_cpu_3x2pt_tomo_fused_kernel,
            kernel_3pcf_multipoles=_cpu_3pcf_multipoles,
        )

    elif device_type == "gpu":
        try:
            import cupy

            # Check if the requested device is available
            if device_id >= cupy.cuda.runtime.getDeviceCount():
                raise ValueError(f"GPU ID {device_id} not found.")

            return Backend(
                "cupy",
                cupy,
                device_id,
                xipm_cross_corr_kernel=_build_cupy_xipm_cross_corr_kernel(cupy),
                xipm_auto_corr_kernel=_build_cupy_xipm_auto_corr_kernel(cupy),
                xipm_tomo_vectorized_kernel=_build_cupy_tomo_vectorized_kernel(cupy),
                xipm_tomo_packed_kernel=_build_cupy_tomo_packed_kernel(cupy),
                aperture_density_kernel=_build_cupy_aperture_density_kernel(cupy),
                aperture_shear_kernel=_build_cupy_aperture_shear_kernel(cupy),
                aperture_tomo_shear_kernel=_build_cupy_aperture_tomo_shear_kernel(cupy),
                aperture_tomo_density_kernel=_build_cupy_aperture_tomo_density_kernel(
                    cupy
                ),
                kernel_density_density=_build_cupy_density_density_corr_kernel(cupy),
                kernel_density_shear=_build_cupy_density_shear_corr_kernel(cupy),
                kernel_density_density_tomo_vectorized=_build_cupy_density_density_tomo_vectorized_kernel(  # noqa: E501
                    cupy
                ),
                kernel_density_shear_tomo_vectorized=_build_cupy_density_shear_tomo_vectorized_kernel(  # noqa: E501
                    cupy
                ),
                kernel_density_density_tomo_packed=_build_cupy_density_density_tomo_packed_kernel(  # noqa: E501
                    cupy
                ),
                kernel_density_shear_tomo_packed=_build_cupy_density_shear_tomo_packed_kernel(  # noqa: E501
                    cupy
                ),
                kernel_3x2pt_tomo_aperture=_build_cupy_3x2pt_tomo_aperture_kernel(cupy),
                kernel_3x2pt_tomo_pairs=_build_cupy_3x2pt_tomo_pairs_kernel(cupy),
                degrade_rows_kernel=_build_cupy_degrade_rows_kernel(cupy),
                kernel_3pcf_multipoles=build_cupy_3pcf_multipoles_kernel(cupy),
            )
        except ImportError:
            if device == "auto":
                warnings.warn("Cupy not installed, falling back to CPU (numpy).")
                return Backend("numpy", np)
            else:
                raise ImportError("Cupy not installed but GPU requested.")
