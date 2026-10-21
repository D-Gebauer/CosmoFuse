"""Aperture statistics M_ap (shear) and M_g (density) per patch.

CPU: Numba kernels.  GPU: per-pixel ``ElementwiseKernel`` contributions and
the block-reduced tomographic kernels of ``cuda/aperture_tomo.cu``.
"""

from typing import Any, Optional, Tuple

import numpy as np
from numba import njit, prange

from .raw import _CUPY_FASTMATH_OPTIONS, KernelFile, _has_raw_cuda_compiler, c_real

# Beyond this many tomographic bins the fused aperture kernels (one block per
# patch, the bins looped inside it) hold too many per-thread accumulators --
# 2*NZ each -- and the per-(patch, bin) kernel is launched instead.
_MAX_FUSED_APERTURE_BINS = 16
# Map arguments the aperture kernels address through explicit strides
# (planar, strided and AoS views are all passed without a copy).
_STRIDED = ("g1", "g2", "weights", "values")


# ── Aperture statistics kernels ──────────────────────────────────────────
# Compute the aperture-filtered field for each sky patch:
#   M = A_patch · Σ(w · field · Q(θ)) / Σ(w)
# where Q(θ) is the compensated filter function.


@njit(fastmath=True, parallel=True, cache=True)
def _cpu_aperture_density_kernel(
    Q_inds: np.ndarray,
    Q_val: np.ndarray,
    Q_offsets: np.ndarray,
    map_values: np.ndarray,
    weights: np.ndarray,
    Q_patch_area: np.ndarray,
    out_aperture: np.ndarray,
) -> None:
    """Galaxy mean density M_g within an aperture: weighted δ_g convolved
    with the compensated filter Q(θ)."""
    n_patches = Q_offsets.shape[0] - 1
    # float64 explicitly, not ``x[0] * 0.0``.  That expression types as
    # float64 under numba (python-float promotion) but as float32 under
    # NUMBA_DISABLE_JIT=1, which pytest-env forces -- so the suite was
    # measuring a float32 accumulation of a kernel that ships accumulating in
    # float64.  Pinning it keeps the shipped numbers and makes the tests
    # exercise them.  (It is deliberately not ``acc_dtype``: these four
    # kernels reduce over a whole aperture disc or angular bin, where a
    # float32 accumulator loses precision for nothing.)
    zero = np.float64(0)

    for patch_idx in prange(n_patches):
        start = Q_offsets[patch_idx]
        stop = Q_offsets[patch_idx + 1]
        sum_w = zero
        sum_wdelta_q = zero
        for i in range(start, stop):
            pix_idx = Q_inds[i]
            weight = weights[pix_idx]
            sum_w += weight
            sum_wdelta_q += weight * map_values[pix_idx] * Q_val[i]
        out_aperture[patch_idx] = Q_patch_area[patch_idx] * sum_wdelta_q / sum_w


@njit(fastmath=True, parallel=True, cache=True)
def _cpu_aperture_shear_kernel(
    Q_inds: np.ndarray,
    Q_cos: np.ndarray,
    Q_sin: np.ndarray,
    Q_val: np.ndarray,
    Q_offsets: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    weights: np.ndarray,
    Q_patch_area: np.ndarray,
    out_aperture: np.ndarray,
) -> None:
    """Aperture mass M_ap: tangential shear γ_t convolved with Q(θ).

    γ_t = -γ₁·cos(2φ) - γ₂·sin(2φ) is the tangential shear component
    relative to the patch centre.
    """
    n_patches = Q_offsets.shape[0] - 1
    # float64 explicitly, not ``x[0] * 0.0``.  That expression types as
    # float64 under numba (python-float promotion) but as float32 under
    # NUMBA_DISABLE_JIT=1, which pytest-env forces -- so the suite was
    # measuring a float32 accumulation of a kernel that ships accumulating in
    # float64.  Pinning it keeps the shipped numbers and makes the tests
    # exercise them.  (It is deliberately not ``acc_dtype``: these four
    # kernels reduce over a whole aperture disc or angular bin, where a
    # float32 accumulator loses precision for nothing.)
    zero = np.float64(0)

    for patch_idx in prange(n_patches):
        start = Q_offsets[patch_idx]
        stop = Q_offsets[patch_idx + 1]
        sum_w = zero
        sum_wgt_q = zero
        for i in range(start, stop):
            pix_idx = Q_inds[i]
            weight = weights[pix_idx]
            # Tangential shear projection
            gt = -g1[pix_idx] * Q_cos[i] - g2[pix_idx] * Q_sin[i]
            sum_w += weight
            sum_wgt_q += weight * gt * Q_val[i]
        out_aperture[patch_idx] = Q_patch_area[patch_idx] * sum_wgt_q / sum_w


# ── CuPy ElementwiseKernel builders (GPU per-element operations) ──────────


def _build_cupy_aperture_density_kernel(module: Any) -> Any:
    """GPU kernel: per-pixel contribution to galaxy mean density M_g.

    The filter geometry arrays (type letter ``Q``) may be narrower than the
    map type ``T``; the ``(T)`` promotion at use is exact for float32 →
    float64.
    """
    return module.ElementwiseKernel(
        "raw I Q_inds, raw Q Q_val, raw T map_values, raw T weights",
        "T out_num, T out_den",
        """
        const I idx = Q_inds[i];
        const T w = weights[idx];
        out_num = w * map_values[idx] * (T)Q_val[i];
        out_den = w;
        """,
        "gpu_aperture_density_kernel",
        options=_CUPY_FASTMATH_OPTIONS,
    )


def _build_cupy_aperture_shear_kernel(module: Any) -> Any:
    """GPU kernel: per-pixel contribution to aperture mass M_ap via γ_t·Q(θ).

    The filter geometry arrays (type letter ``Q``) may be narrower than the
    map type ``T``; the ``(T)`` promotion at use is exact for float32 →
    float64.
    """
    return module.ElementwiseKernel(
        "raw I Q_inds, raw Q Q_cos, raw Q Q_sin, raw Q Q_val,"
        " raw T g1, raw T g2, raw T weights",
        "T out_num, T out_den",
        """
        const I idx = Q_inds[i];
        const T w = weights[idx];
        const T gt = -g1[idx] * (T)Q_cos[i] - g2[idx] * (T)Q_sin[i];
        out_num = w * gt * (T)Q_val[i];
        out_den = w;
        """,
        "gpu_aperture_shear_kernel",
        options=_CUPY_FASTMATH_OPTIONS,
    )


# ── Block-reduced aperture RawKernels (all tomo bins in one launch) ───────
# Replace the per-pixel ElementwiseKernel + add.reduceat path with a single
# launch over (patch, tomo bin) blocks; the wrappers fall back (return
# False) when no raw compiler is available so callers can use the legacy
# ElementwiseKernel path.


def _aperture_tomo_prepare_planar(module: Any, arr: Any) -> Tuple[Any, int, int]:
    """Return (array, bin stride, row stride) in elements for a 2D view.

    The kernel takes both strides, so nothing is ever copied: planar
    ``(nz, npix)`` arrays, strided views such as ``shear[:, 0]`` of an
    ``(nz, 2, npix)`` array, and the transposed AoS buffers
    ``shear_aos[:, :, 0].T`` of an ``(npix, nz, 2)`` array all work.
    """
    del module  # no copy is needed for any layout
    return (
        arr,
        int(arr.strides[0] // arr.itemsize),
        int(arr.strides[1] // arr.itemsize),
    )


# (runtime id, device) -> (runtime, SM count).  The runtime object is kept
# alive by the value so its id cannot be reused by a different one.
_SM_COUNT_CACHE: dict[Tuple[int, int], Tuple[Any, Optional[int]]] = {}


def _multiprocessor_count(module: Any) -> Optional[int]:
    """SM count of the active device, or None when it cannot be queried.

    Only used to size a launch grid, so a stand-in module without
    ``cuda.runtime`` (the emulated cupy of the tests) simply imposes no
    constraint.
    """
    runtime = getattr(getattr(module, "cuda", None), "runtime", None)
    if runtime is None:
        return None
    try:
        device = int(runtime.getDevice())
    except Exception:  # pragma: no cover - no device behind the module
        return None
    key = (id(runtime), device)
    if key not in _SM_COUNT_CACHE:
        try:
            props = runtime.getDeviceProperties(device)
            count: Optional[int] = int(props["multiProcessorCount"])
        except Exception:  # pragma: no cover - older runtime without the field
            count = None
        _SM_COUNT_CACHE[key] = (runtime, count)
    return _SM_COUNT_CACHE[key][1]


def _use_fused_aperture(module: Any, ntomo: int, npatches: int) -> bool:
    """Whether to launch the one-block-per-patch aperture kernel.

    It reads the aperture disc geometry once per pixel instead of once per
    tomographic bin: 1.72x on the kernel at nside 512 and 1.83x measured in
    situ inside ``gg`` (the result is bitwise identical
    either way).  Two things take it back to the
    per-(patch, bin) kernel: a bin set wide enough for the per-thread
    accumulators to spill, and a patch set small enough that one block per
    patch underfills the device -- the fused grid is ``ntomo`` times
    smaller, which is the point of it.
    """
    if ntomo <= 0 or ntomo > _MAX_FUSED_APERTURE_BINS:
        return False
    n_sm = _multiprocessor_count(module)
    if n_sm is not None and npatches < 2 * n_sm:
        return False
    return True


def _build_cupy_aperture_tomo_shear_kernel(module: Any) -> Any:
    """Builder for the GPU block-reduced tomographic aperture-mass kernel.

    Two kernels behind one launch contract (see ``aperture_tomo.cu``): the
    fused one-block-per-patch kernel is the default and reads the disc
    geometry once per pixel instead of once per tomographic bin; the
    per-(patch, bin) kernel is the fallback for wide bin sets and small
    patch counts.  The two are bitwise identical, so which one ran is never
    observable in the result.  ``ntomo`` is a template argument of the
    fused kernel and therefore part of the cache key; it is small and
    stable, so compiling on demand per value is fine.
    """
    kernels = KernelFile(
        module, "aperture_tomo.cu", "Aperture-shear tomo", strided=_STRIDED
    )

    def _cupy_aperture_tomo_shear_kernel(
        g1: Any,
        g2: Any,
        weights: Any,
        q_inds: Any,
        q_cos: Any,
        q_sin: Any,
        q_val: Any,
        q_offsets: Any,
        q_patch_area: Any,
        out_num: Any,
        out_den: Any,
    ) -> bool:
        if not _has_raw_cuda_compiler(module):
            return False
        if g1.ndim != 2 or g2.ndim != 2 or weights.ndim != 2:
            return False

        targs: Tuple[Any, ...] = (c_real(weights.dtype), c_real(q_cos.dtype))
        npatches = int(q_offsets.shape[0] - 1)
        ntomo = int(g1.shape[0])
        fused = _use_fused_aperture(module, ntomo, npatches)
        raw_kernel = (
            kernels("gpu_aperture_shear_tomo_fused", targs + (ntomo,))
            if fused
            else kernels("gpu_aperture_shear_tomo", targs)
        )
        if raw_kernel is None:
            return False

        g1, g1_stride, g1_elem = _aperture_tomo_prepare_planar(module, g1)
        g2, g2_stride, g2_elem = _aperture_tomo_prepare_planar(module, g2)
        if (g1_stride, g1_elem) != (g2_stride, g2_elem):
            g1 = module.ascontiguousarray(g1)
            g2 = module.ascontiguousarray(g2)
            g1_stride = g2_stride = int(g1.shape[1])
            g1_elem = g2_elem = 1
        weights, w_stride, w_elem = _aperture_tomo_prepare_planar(module, weights)

        blocks = (
            (max(1, npatches), 1, 1) if fused else (max(1, npatches), max(1, ntomo), 1)
        )
        # The fused kernel knows NZ at compile time and takes no `ntomo`.
        tail: Tuple[int, ...] = (npatches,) if fused else (npatches, ntomo)
        raw_kernel(
            blocks,
            (256,),
            (
                g1,
                g2,
                g1_stride,
                g1_elem,
                weights,
                w_stride,
                w_elem,
                q_inds,
                q_cos,
                q_sin,
                q_val,
                q_offsets,
                q_patch_area,
                out_num,
                out_den,
                *tail,
            ),
        )
        return True

    return _cupy_aperture_tomo_shear_kernel


def _build_cupy_aperture_tomo_density_kernel(module: Any) -> Any:
    """Builder for the GPU block-reduced tomographic aperture-density kernel.

    Same two-kernel contract as the aperture-mass builder above: the fused
    one-block-per-patch kernel by default, the per-(patch, bin) kernel as
    the fallback, bitwise identical results either way.
    """
    kernels = KernelFile(
        module, "aperture_tomo.cu", "Aperture-density tomo", strided=_STRIDED
    )

    def _cupy_aperture_tomo_density_kernel(
        values: Any,
        weights: Any,
        q_inds: Any,
        q_val: Any,
        q_offsets: Any,
        q_patch_area: Any,
        out_num: Any,
        out_den: Any,
    ) -> bool:
        if not _has_raw_cuda_compiler(module):
            return False
        if values.ndim != 2 or weights.ndim != 2:
            return False

        targs: Tuple[Any, ...] = (c_real(weights.dtype), c_real(q_val.dtype))
        npatches = int(q_offsets.shape[0] - 1)
        ntomo = int(values.shape[0])
        fused = _use_fused_aperture(module, ntomo, npatches)
        raw_kernel = (
            kernels("gpu_aperture_density_tomo_fused", targs + (ntomo,))
            if fused
            else kernels("gpu_aperture_density_tomo", targs)
        )
        if raw_kernel is None:
            return False

        values, v_stride, v_elem = _aperture_tomo_prepare_planar(module, values)
        weights, w_stride, w_elem = _aperture_tomo_prepare_planar(module, weights)

        blocks = (
            (max(1, npatches), 1, 1) if fused else (max(1, npatches), max(1, ntomo), 1)
        )
        # The fused kernel knows NZ at compile time and takes no `ntomo`.
        tail: Tuple[int, ...] = (npatches,) if fused else (npatches, ntomo)
        raw_kernel(
            blocks,
            (256,),
            (
                values,
                v_stride,
                v_elem,
                weights,
                w_stride,
                w_elem,
                q_inds,
                q_val,
                q_offsets,
                q_patch_area,
                out_num,
                out_den,
                *tail,
            ),
        )
        return True

    return _cupy_aperture_tomo_density_kernel
