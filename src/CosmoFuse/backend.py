"""
Backend abstraction layer for CPU (NumPy/Numba) and GPU (CuPy/CUDA) execution.

All correlation measurement kernels exist in two forms:
  - **CPU**: Numba @njit functions with parallel=True for multi-core execution
  - **GPU**: CuPy ElementwiseKernels for simple per-pair operations, and
    compiled CUDA RawKernels (.cu files in cuda/) for fused/vectorised
    tomographic operations

The Backend class holds references to the active kernel set and provides
to_device()/to_numpy() for transparent data movement between host and device.
"""

import logging
from contextlib import nullcontext
from pathlib import Path
import warnings
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
from numba import njit, prange

from .utils import live_object_serial

logger = logging.getLogger(__name__)

# --std is mandatory: NVRTC refuses to instantiate name_expressions
# (C++ templates) without an explicit standard. Keep in sync with
# scripts/compile_check_nvrtc.py.
_CUPY_FASTMATH_OPTIONS = ("--use_fast_math", "--std=c++14")
_MAX_VECTOR_TOMO_BINS = 64
# Beyond this many tomographic bins the fused aperture kernels (one block per
# patch, the bins looped inside it) hold too many per-thread accumulators --
# 2*NZ each -- and the per-(patch, bin) kernel is launched instead.
_MAX_FUSED_APERTURE_BINS = 16
_CUDA_DIR = Path(__file__).with_name("cuda")
_CUDA_SOURCE_CACHE: dict[str, str] = {}
# Sentinel distinguishing "never attempted" from "compilation failed" in the
# per-builder kernel caches, so a failing NVRTC compilation is not retried
# (and re-logged) on every subsequent call.
_KERNEL_CACHE_MISS = object()

def _load_cuda_source_file(filename: str) -> str:
    cached = _CUDA_SOURCE_CACHE.get(filename)
    if cached is not None:
        return cached
    source_path = _CUDA_DIR / filename
    source = source_path.read_text(encoding="utf-8")
    _CUDA_SOURCE_CACHE[filename] = source
    return source


_COMMON_CUDA_SOURCE = _load_cuda_source_file("common.cuh")
_PAIR_TILES_CUDA_SOURCE = _load_cuda_source_file("pair_tiles.cuh")

# Combination-tiled pair kernels keep every accumulator in a register; the
# tiles are used only while the per-thread accumulator count (numerators +
# weight sums over all combinations) stays within this budget.  Beyond it
# the per-(bin, row) kernels are launched instead (unpacked geometry only).
_MAX_TILED_ACCUMULATORS = 64
# Budget of the single-pass 3x2pt tile (xi+- + xi_g + xi_t accumulators).
# Measured on an A100 up to 120 (4 source x 6 lens bins, all combinations),
# where it is still ahead of three separate tiles; untested beyond.
_MAX_TILED_3X2PT_ACCUMULATORS = 120


def _prepare_cuda_source(filename: str) -> str:
    source = _load_cuda_source_file(filename)
    source = source.replace("__COMMON_CUDA_SOURCE__", _COMMON_CUDA_SOURCE)
    return source.replace("__PAIR_TILES_CUDA_SOURCE__", _PAIR_TILES_CUDA_SOURCE)


def _has_raw_cuda_compiler(module: Any) -> bool:
    return (
        getattr(module, "RawModule", None) is not None
        or getattr(module, "RawKernel", None) is not None
    )


def _compile_raw_cuda_kernel(module: Any, source: str, name_expression: str) -> Any:
    raw_module_ctor = getattr(module, "RawModule", None)
    if raw_module_ctor is not None:
        try:
            raw_module = raw_module_ctor(
                code=source,
                options=_CUPY_FASTMATH_OPTIONS,
                name_expressions=(name_expression,),
            )
        except TypeError:
            raw_module = raw_module_ctor(
                source,
                options=_CUPY_FASTMATH_OPTIONS,
                name_expressions=(name_expression,),
            )

        get_function = getattr(raw_module, "get_function", None)
        if get_function is not None:
            return get_function(name_expression)

    raw_kernel_ctor = getattr(module, "RawKernel", None)
    if raw_kernel_ctor is None:
        raise AttributeError("Backend module does not provide RawModule or RawKernel")

    return raw_kernel_ctor(source, name_expression, options=_CUPY_FASTMATH_OPTIONS)


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
    situ inside ``get_full_tomo_shear`` (the result is bitwise identical
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


def _build_cupy_degrade_rows_kernel(module: Any) -> Any:
    """Builder for the fused treecode row-degrade kernels.

    Two entry points behind one object: ``level()`` accumulates one CSR
    level into the accumulation-dtype scratch, ``finalize()`` normalises
    and scatters a range of appended rows into the map-dtype row buffers.
    Both return ``False`` when no raw compiler is available, so the caller
    can fall back to the sparse path.

    Every buffer is passed as a descriptor
    ``(array, base, lead_stride, comp_stride, row_stride)`` of element
    offsets, so the kernels write the SoA, interleaved or AoS layout
    directly -- see the header of ``degrade_rows.cu``.  ``comp_stride`` is
    unused for the weights.
    """
    build = _make_raw_kernel_builder(module, "degrade_rows.cu", "Treecode degrade")

    def _c_type(dtype: Any) -> str:
        return "float" if dtype == module.float32 else "double"

    def _w_args(desc: Tuple[Any, int, int, int, int]) -> Tuple[Any, ...]:
        arr, base, lead, _comp, row = desc
        return (arr, np.int64(base), np.int64(lead), np.int64(row))

    def _v_args(desc: Tuple[Any, int, int, int, int]) -> Tuple[Any, ...]:
        arr, base, lead, comp, row = desc
        return (arr, np.int64(base), np.int64(lead), np.int64(comp), np.int64(row))

    class _DegradeKernels:
        available = True

        @staticmethod
        def level(
            indptr: Any,
            indices: Any,
            w_src: Tuple[Any, int, int, int, int],
            v_src: Tuple[Any, int, int, int, int],
            w_dst: Tuple[Any, int, int, int, int],
            v_dst: Tuple[Any, int, int, int, int],
            n_cells: int,
            n_lead: int,
            n_val: int,
            weighted: bool,
        ) -> bool:
            if not _has_raw_cuda_compiler(module) or n_cells <= 0:
                return n_cells <= 0
            src_t = _c_type(w_src[0].dtype)
            acc_t = _c_type(w_dst[0].dtype)
            # Lanes per cell: the next power of two at or above the mean
            # number of children, capped at a warp.  A treecode level has
            # 4 children per cell, so a full warp would idle 87 % of its
            # lanes; a coarse aperture level has many more.
            nnz = int(indices.size)
            mean_children = max(1.0, nnz / max(1, int(n_cells)))
            seg = 1
            while seg < 32 and seg < mean_children:
                seg *= 2
            kernel = build(
                "gpu_degrade_level",
                (src_t, acc_t, int(n_val), "true" if weighted else "false", seg),
            )
            if kernel is None:
                return False
            # v_src/v_dst may be unused (n_val == 0) but must still be valid
            # pointers; the caller passes the weight arrays in that case.
            threads = 256
            groups = threads // seg
            total = int(n_cells) * int(n_lead)
            blocks = (total + groups - 1) // groups
            kernel(
                (max(1, blocks),),
                (threads,),
                (
                    indptr,
                    indices,
                    *_w_args(w_src),
                    *_v_args(v_src),
                    *_w_args(w_dst),
                    *_v_args(v_dst),
                    np.int32(n_cells),
                    np.int32(n_lead),
                ),
            )
            return True

        @staticmethod
        def finalize(
            w_app: Any,
            v_app: Any,
            n_appended: int,
            w_rows: Tuple[Any, int, int, int, int],
            v_rows: Tuple[Any, int, int, int, int],
            n_lead: int,
            lo: int,
            hi: int,
            n_val: int,
            row_inner: bool = True,
        ) -> bool:
            if not _has_raw_cuda_compiler(module):
                return False
            if hi <= lo:
                return True
            kernel = build(
                "gpu_degrade_finalize",
                (_c_type(w_rows[0].dtype), _c_type(w_app.dtype), int(n_val)),
            )
            if kernel is None:
                return False
            threads = 256
            total = int(hi - lo) * int(n_lead)
            blocks = min(65535, (total + threads - 1) // threads)
            # The scratch is always SoA with a unit row stride.
            kernel(
                (max(1, blocks),),
                (threads,),
                (
                    w_app,
                    np.int64(n_appended),
                    v_app,
                    np.int64(n_appended),
                    np.int64(int(n_lead) * int(n_appended)),
                    *_w_args(w_rows),
                    *_v_args(v_rows),
                    np.int32(n_lead),
                    np.int64(lo),
                    np.int64(hi),
                    np.int32(1 if row_inner else 0),
                ),
            )
            return True

    return _DegradeKernels()


def _build_cupy_aperture_tomo_shear_kernel(module: Any) -> Any:
    """Builder for the GPU block-reduced tomographic aperture-mass kernel.

    Two kernels behind one launch contract (see ``aperture_tomo.cu``): the
    fused one-block-per-patch kernel is the default and reads the disc
    geometry once per pixel instead of once per tomographic bin; the
    per-(patch, bin) kernel is the fallback for wide bin sets and small
    patch counts.  The two are bitwise identical, so which one ran is never
    observable in the result.  ``ntomo`` is a template argument of the
    fused kernel and therefore part of the cache key (``None`` = fallback);
    it is small and stable, so compiling on demand per value is fine.
    """
    kernel_cache: dict[tuple[str, str, Optional[int]], Any] = {}

    def _get_or_build_raw_kernel(
        map_c_type: str, q_c_type: str, ntomo: Optional[int]
    ) -> Optional[Any]:
        key = (map_c_type, q_c_type, ntomo)
        cached = kernel_cache.get(key, _KERNEL_CACHE_MISS)
        if cached is not _KERNEL_CACHE_MISS:
            # May be None: a previously failed compilation is cached negatively
            # so it is not retried (and re-logged) on every call.
            return cached

        if ntomo is None:
            name_expression = f"gpu_aperture_shear_tomo<{map_c_type}, {q_c_type}>"
        else:
            name_expression = (
                f"gpu_aperture_shear_tomo_fused<{map_c_type}, {q_c_type}, {ntomo}>"
            )
        source = _prepare_cuda_source("aperture_tomo.cu")

        try:
            kernel = _compile_raw_cuda_kernel(module, source, name_expression)
        except Exception as exc:
            logger.warning(
                "Aperture-shear tomo RawKernel compilation failed; using elementwise path: %s",
                exc,
            )
            kernel_cache[key] = None
            return None

        kernel_cache[key] = kernel
        return kernel

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

        map_c_type = "float" if weights.dtype == module.float32 else "double"
        q_c_type = "float" if q_cos.dtype == module.float32 else "double"
        npatches = int(q_offsets.shape[0] - 1)
        ntomo = int(g1.shape[0])
        fused = _use_fused_aperture(module, ntomo, npatches)
        raw_kernel = _get_or_build_raw_kernel(
            map_c_type, q_c_type, ntomo if fused else None
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

        threads = 256
        blocks = (
            (max(1, npatches), 1, 1) if fused
            else (max(1, npatches), max(1, ntomo), 1)
        )
        # The fused kernel knows NZ at compile time and takes no `ntomo`.
        tail: Tuple[Any, ...] = (np.int32(npatches),)
        if not fused:
            tail += (np.int32(ntomo),)
        raw_kernel(
            blocks,
            (threads,),
            (
                g1,
                g2,
                np.int64(g1_stride),
                np.int64(g1_elem),
                weights,
                np.int64(w_stride),
                np.int64(w_elem),
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
    kernel_cache: dict[tuple[str, str, Optional[int]], Any] = {}

    def _get_or_build_raw_kernel(
        map_c_type: str, q_c_type: str, ntomo: Optional[int]
    ) -> Optional[Any]:
        key = (map_c_type, q_c_type, ntomo)
        cached = kernel_cache.get(key, _KERNEL_CACHE_MISS)
        if cached is not _KERNEL_CACHE_MISS:
            # May be None: a previously failed compilation is cached negatively
            # so it is not retried (and re-logged) on every call.
            return cached

        if ntomo is None:
            name_expression = f"gpu_aperture_density_tomo<{map_c_type}, {q_c_type}>"
        else:
            name_expression = (
                f"gpu_aperture_density_tomo_fused<{map_c_type}, {q_c_type}, {ntomo}>"
            )
        source = _prepare_cuda_source("aperture_tomo.cu")

        try:
            kernel = _compile_raw_cuda_kernel(module, source, name_expression)
        except Exception as exc:
            logger.warning(
                "Aperture-density tomo RawKernel compilation failed; using elementwise path: %s",
                exc,
            )
            kernel_cache[key] = None
            return None

        kernel_cache[key] = kernel
        return kernel

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

        map_c_type = "float" if weights.dtype == module.float32 else "double"
        q_c_type = "float" if q_val.dtype == module.float32 else "double"
        npatches = int(q_offsets.shape[0] - 1)
        ntomo = int(values.shape[0])
        fused = _use_fused_aperture(module, ntomo, npatches)
        raw_kernel = _get_or_build_raw_kernel(
            map_c_type, q_c_type, ntomo if fused else None
        )
        if raw_kernel is None:
            return False

        values, v_stride, v_elem = _aperture_tomo_prepare_planar(module, values)
        weights, w_stride, w_elem = _aperture_tomo_prepare_planar(module, weights)

        threads = 256
        blocks = (
            (max(1, npatches), 1, 1) if fused
            else (max(1, npatches), max(1, ntomo), 1)
        )
        # The fused kernel knows NZ at compile time and takes no `ntomo`.
        tail: Tuple[Any, ...] = (np.int32(npatches),)
        if not fused:
            tail += (np.int32(ntomo),)
        raw_kernel(
            blocks,
            (threads,),
            (
                values,
                np.int64(v_stride),
                np.int64(v_elem),
                weights,
                np.int64(w_stride),
                np.int64(w_elem),
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


# ── Single-pair 2PCF kernels ─────────────────────────────────────────────

@njit(fastmath=True, parallel=True, cache=True)
def _cpu_density_density_corr_kernel(
    density_a: np.ndarray,
    density_b: np.ndarray,
    w_a: np.ndarray,
    w_b: np.ndarray,
    ind_i: np.ndarray,
    ind_j: np.ndarray,
    offsets: np.ndarray,
    out_ab: np.ndarray,
    out_ba: np.ndarray,
    out_ab_w: np.ndarray,
    out_ba_w: np.ndarray,
) -> None:
    """Galaxy clustering ξ_g numerators for a single tomo-bin pair.

    Sums w_a·w_b·δ_a·δ_b over all pixel pairs in each angular bin, for
    both pair orientations in a single pass over the pair list, and
    accumulates the weight sums (denominators) alongside.
    """
    nbins = offsets.shape[0] - 1
    for b in prange(nbins):
        sum_ab = 0.0
        sum_ba = 0.0
        sum_ab_w = 0.0
        sum_ba_w = 0.0
        start = offsets[b]
        stop = offsets[b + 1]

        for idx in range(start, stop):
            i = ind_i[idx]
            j = ind_j[idx]
            w_ab = w_a[i] * w_b[j]
            sum_ab += w_ab * density_a[i] * density_b[j]
            sum_ab_w += w_ab
            w_ba = w_a[j] * w_b[i]
            sum_ba += w_ba * density_a[j] * density_b[i]
            sum_ba_w += w_ba

        out_ab[b] = sum_ab
        out_ba[b] = sum_ba
        out_ab_w[b] = sum_ab_w
        out_ba_w[b] = sum_ba_w


@njit(fastmath=True, parallel=True, cache=True)
def _cpu_density_shear_corr_kernel(
    density_lens: np.ndarray,
    g1_source: np.ndarray,
    g2_source: np.ndarray,
    w_lens: np.ndarray,
    w_source: np.ndarray,
    ind_i: np.ndarray,
    ind_j: np.ndarray,
    exp_i: np.ndarray,
    exp_j: np.ndarray,
    offsets: np.ndarray,
    out_ab: np.ndarray,
    out_ba: np.ndarray,
    out_ab_w: np.ndarray,
    out_ba_w: np.ndarray,
) -> None:
    """Galaxy-galaxy lensing ξ_t numerators for a single tomo-bin pair.

    Computes γ_t = -γ₁·cos(2φ) + γ₂·sin(2φ) and sums
    w_lens·w_source·δ_lens·γ_t over all pairs in each angular bin, for
    both lens/source orientations in a single pass over the pair list,
    accumulating the weight sums (denominators) alongside.
    """
    nbins = offsets.shape[0] - 1
    for b in prange(nbins):
        sum_ab = 0.0
        sum_ba = 0.0
        sum_ab_w = 0.0
        sum_ba_w = 0.0
        start = offsets[b]
        stop = offsets[b + 1]

        for idx in range(start, stop):
            i = ind_i[idx]
            j = ind_j[idx]
            # A→B: pixel i = lens, pixel j = source
            rot_ab = exp_j[idx]
            gamma_t_ab = -g1_source[j] * rot_ab.real + g2_source[j] * rot_ab.imag
            w_ab = w_lens[i] * w_source[j]
            sum_ab += w_ab * density_lens[i] * gamma_t_ab
            sum_ab_w += w_ab
            # B→A: pixel j = lens, pixel i = source
            rot_ba = exp_i[idx]
            gamma_t_ba = -g1_source[i] * rot_ba.real + g2_source[i] * rot_ba.imag
            w_ba = w_lens[j] * w_source[i]
            sum_ba += w_ba * density_lens[j] * gamma_t_ba
            sum_ba_w += w_ba

        out_ab[b] = sum_ab
        out_ba[b] = sum_ba
        out_ab_w[b] = sum_ab_w
        out_ba_w[b] = sum_ba_w


# ── Vectorised tomographic CPU kernels ────────────────────────────────────
# Process all tomographic bin combinations in a single pass over the
# pair list, avoiding redundant memory traversals.

@njit(fastmath=True, parallel=True, cache=True)
def _cpu_density_density_tomo_vectorized_kernel(
    density_map: np.ndarray,
    weights: np.ndarray,
    ind_i: np.ndarray,
    ind_j: np.ndarray,
    offsets: np.ndarray,
    comb_i: np.ndarray,
    comb_j: np.ndarray,
    out_num: np.ndarray,
    out_den: np.ndarray,
) -> None:
    """Vectorised galaxy clustering ξ_g for all tomo-bin combinations.

    For cross-bin pairs (i≠j), both orientations (A→B and B→A) are summed
    into one numerator and one weight sum (ratio-of-sums estimator).  The
    pair loop is outermost within each bin so the pair indices and map
    rows are loaded once and reused for every tomographic combination; the
    weight sums (denominators) are accumulated in the same pass so no
    separate reduction is needed.
    """
    n_bins = offsets.shape[0] - 1
    ncomb = comb_i.shape[0]

    for b in prange(n_bins):
        start = offsets[b]
        stop = offsets[b + 1]
        acc_num = np.zeros(ncomb, dtype=out_num.dtype)
        acc_den = np.zeros(ncomb, dtype=out_den.dtype)

        for idx in range(start, stop):
            pix_i = int(ind_i[idx])
            pix_j = int(ind_j[idx])

            for comb_idx in range(ncomb):
                i = comb_i[comb_idx]
                j = comb_j[comb_idx]

                # A→B orientation: tomo bin i at pixel_i, tomo bin j at pixel_j
                w_ab = weights[pix_i, i] * weights[pix_j, j]
                ab = w_ab * density_map[pix_i, i] * density_map[pix_j, j]

                if i == j:
                    acc_num[comb_idx] += ab
                    acc_den[comb_idx] += w_ab
                else:
                    # B→A orientation: swap tomo bins to symmetrise
                    w_ba = weights[pix_i, j] * weights[pix_j, i]
                    ba = w_ba * density_map[pix_i, j] * density_map[pix_j, i]
                    acc_num[comb_idx] += ab + ba
                    acc_den[comb_idx] += w_ab + w_ba

        for comb_idx in range(ncomb):
            out_num[comb_idx, b] = acc_num[comb_idx]
            out_den[comb_idx, b] = acc_den[comb_idx]


@njit(fastmath=True, parallel=True, cache=True)
def _cpu_density_shear_tomo_vectorized_kernel(
    density_map: np.ndarray,
    shear_map: np.ndarray,
    lens_weights: np.ndarray,
    source_weights: np.ndarray,
    ind_i: np.ndarray,
    ind_j: np.ndarray,
    rot_i: np.ndarray,
    rot_j: np.ndarray,
    offsets: np.ndarray,
    comb_i: np.ndarray,
    comb_j: np.ndarray,
    out_num: np.ndarray,
    out_den: np.ndarray,
) -> None:
    """Vectorised galaxy-galaxy lensing ξ_t for all lens×source tomo
    combinations.

    Each pair contributes in both orientations: pixel i as lens with
    pixel j as source (A→B), and vice versa (B→A).  The pair loop is
    outermost within each bin so the pair indices and rotation factors
    are loaded once per pair and reused for every combination; the
    weight sums (denominators) are accumulated in the same pass.
    """
    n_bins = offsets.shape[0] - 1
    ncomb = comb_i.shape[0]

    for b in prange(n_bins):
        start = offsets[b]
        stop = offsets[b + 1]
        acc_num = np.zeros(ncomb, dtype=out_num.dtype)
        acc_den = np.zeros(ncomb, dtype=out_den.dtype)

        for idx in range(start, stop):
            pix_i = int(ind_i[idx])
            pix_j = int(ind_j[idx])
            exp_j = rot_j[idx]
            exp_i = rot_i[idx]
            exp_j_re = exp_j.real
            exp_j_im = exp_j.imag
            exp_i_re = exp_i.real
            exp_i_im = exp_i.imag

            for comb_idx in range(ncomb):
                lens_bin = comb_i[comb_idx]
                source_bin = comb_j[comb_idx]

                # A→B: pixel i = lens, pixel j = source
                gamma_t_ij = (
                    -shear_map[pix_j, source_bin, 0] * exp_j_re
                    + shear_map[pix_j, source_bin, 1] * exp_j_im
                )
                w_ab = (
                    lens_weights[pix_i, lens_bin]
                    * source_weights[pix_j, source_bin]
                )
                acc_num[comb_idx] += w_ab * density_map[pix_i, lens_bin] * gamma_t_ij
                acc_den[comb_idx] += w_ab

                # B→A: pixel j = lens, pixel i = source
                gamma_t_ji = (
                    -shear_map[pix_i, source_bin, 0] * exp_i_re
                    + shear_map[pix_i, source_bin, 1] * exp_i_im
                )
                w_ba = (
                    lens_weights[pix_j, lens_bin]
                    * source_weights[pix_i, source_bin]
                )
                acc_num[comb_idx] += w_ba * density_map[pix_j, lens_bin] * gamma_t_ji
                acc_den[comb_idx] += w_ba

        for comb_idx in range(ncomb):
            out_num[comb_idx, b] = acc_num[comb_idx]
            out_den[comb_idx, b] = acc_den[comb_idx]


def _build_cupy_density_density_corr_kernel(module: Any) -> Any:
    """GPU kernel: per-pair ξ_g contribution (w_a·w_b·δ_a·δ_b)."""
    return module.ElementwiseKernel(
        "raw T density_a, raw T density_b, raw T w_a, raw T w_b,"
        " raw I ind_i, raw I ind_j",
        "A out_w",
        """
        const I i_idx = ind_i[i];
        const I j_idx = ind_j[i];
        /* product at map precision T; stored at the accumulator type A */
        out_w = (A)(w_a[i_idx] * w_b[j_idx] * density_a[i_idx] * density_b[j_idx]);
        """,
        "gpu_density_density_corr_kernel",
        options=_CUPY_FASTMATH_OPTIONS,
    )


def _build_cupy_density_shear_corr_kernel(module: Any) -> Any:
    """GPU kernel: per-pair ξ_t contribution (w_lens·w_source·δ_lens·γ_t)."""
    return module.ElementwiseKernel(
        "raw T density_lens, raw T g1_source, raw T g2_source,"
        " raw T w_lens, raw T w_source, raw I ind_i, raw I ind_j, raw C exp_j",
        "A out_gt",
        """
        const I i_idx = ind_i[i];
        const I j_idx = ind_j[i];
        const C rot = exp_j[i];
        const T gamma_t = -g1_source[j_idx] * real(rot) + g2_source[j_idx] * imag(rot);
        /* product at map precision T; stored at the accumulator type A */
        out_gt = (A)(w_lens[i_idx] * w_source[j_idx] * density_lens[i_idx] * gamma_t);
        """,
        "gpu_density_shear_corr_kernel",
        options=_CUPY_FASTMATH_OPTIONS,
    )


# ── CuPy RawKernel builders (GPU fused kernels from .cu files) ────────────
# These builders compile the CUDA source templates with specific type
# parameters and cache the compiled kernels for reuse.

class _CombinationLayout:
    """How a requested list of tomographic combinations maps onto the
    canonical layout generated by the combination-tiled kernels."""

    __slots__ = ("mode", "canonical_ncomb", "rows")

    def __init__(self, mode: str, canonical_ncomb: int, rows: Optional[np.ndarray]) -> None:
        self.mode = mode                        # "triangle" | "auto" | "cartesian"
        self.canonical_ncomb = canonical_ncomb  # rows the tile produces
        self.rows = rows                        # canonical row per request, None = identity


_COMBINATION_LAYOUT_CACHE: dict[tuple[int, int, str, int, int], Optional[_CombinationLayout]] = {}


def _combination_layout(
    comb_i: Any, comb_j: Any, kind: str, n1: int, n2: int = 0
) -> Optional[_CombinationLayout]:
    """Classify ``(comb_i, comb_j)`` against the tiled kernels' layouts.

    ``kind="triangle"``: n1 bins, canonical = row-major upper triangle
    (a request (i, j) with i > j is the same symmetric combination as
    (j, i)); a request consisting of exactly the auto combinations gets
    ``mode="auto"`` (its own, smaller tile).  ``kind="cartesian"``:
    canonical = row-major n1 x n2 product.  ``rows`` is None when the
    request *is* the canonical layout, else the canonical row of every
    requested combination.  Returns None for out-of-range bins.

    The (tiny) device arrays are compared once per array object and the
    result cached by identity: the orchestrator caches and reuses them.

    By ``live_object_serial`` and not ``id()``.  The orchestrator caches the
    *canonical* combination arrays, but an explicit ``ggl_bin_combinations``
    selection builds fresh ones on every call and drops them, so a second
    selection can land on the recycled addresses of the first -- and the
    layout carries ``rows``, the canonical row of every requested
    combination, so a stale hit scatters the results into the previous
    selection's rows.  The collision needs *both* ids to recycle at once and
    CPython's LIFO free lists tend to swap rather than match them (it did not
    fire in 4000 alternating calls here), but it is allocator luck, not a
    guarantee.
    """
    key = (
        live_object_serial(comb_i),
        live_object_serial(comb_j),
        kind,
        int(n1),
        int(n2),
    )
    cached = _COMBINATION_LAYOUT_CACHE.get(key, _KERNEL_CACHE_MISS)
    if cached is not _KERNEL_CACHE_MISS:
        return cached
    to_host = lambda a: np.asarray(a.get() if hasattr(a, "get") else a).astype(np.int64)
    ci, cj = to_host(comb_i), to_host(comb_j)
    layout: Optional[_CombinationLayout]
    if kind == "triangle":
        if ci.size == 0 or ci.min() < 0 or cj.min() < 0 or ci.max() >= n1 or cj.max() >= n1:
            layout = None
        else:
            lo, hi = np.minimum(ci, cj), np.maximum(ci, cj)
            rows = lo * n1 - (lo * (lo - 1)) // 2 + (hi - lo)
            canonical = (n1 * (n1 + 1)) // 2
            identity = rows.size == canonical and np.array_equal(rows, np.arange(canonical))
            if not identity and ci.size == n1 and np.array_equal(ci, np.arange(n1)) and np.array_equal(cj, ci):
                layout = _CombinationLayout("auto", int(n1), None)
            else:
                layout = _CombinationLayout("triangle", int(canonical), None if identity else rows)
    elif kind == "cartesian":
        if ci.size == 0 or ci.min() < 0 or cj.min() < 0 or ci.max() >= n1 or cj.max() >= n2:
            layout = None
        else:
            rows = ci * n2 + cj
            canonical = int(n1 * n2)
            identity = rows.size == canonical and np.array_equal(rows, np.arange(canonical))
            layout = _CombinationLayout("cartesian", canonical, None if identity else rows)
    else:
        raise ValueError(f"unknown combination layout kind {kind!r}")
    if len(_COMBINATION_LAYOUT_CACHE) > 64:
        _COMBINATION_LAYOUT_CACHE.clear()
    _COMBINATION_LAYOUT_CACHE[key] = layout
    return layout


#: Exception names that mean "this source will never compile for these
#: template arguments".  Anything else -- an IO error reading the NVRTC disk
#: cache, a transient driver or allocation failure -- may succeed next time.
_DETERMINISTIC_COMPILE_FAILURES = frozenset(
    {"CompileException", "NVRTCError", "JitifyException", "NVRTCException"}
)


def _is_deterministic_compile_failure(exc: BaseException) -> bool:
    return type(exc).__name__ in _DETERMINISTIC_COMPILE_FAILURES


def _make_raw_kernel_builder(module: Any, filename: str, what: str) -> Any:
    """Per-builder cache of compiled RawKernels keyed by (name, template args).

    A *deterministic* compile failure is cached negatively (None) so it is
    not retried and re-logged on every call.  A transient one is **not**:
    caching it would silently pin the process to the slower fallback kernel
    for its whole lifetime, with no error and no way to tell from a
    configuration that legitimately has no tiled kernel.  The retry is
    logged once per key so a genuinely broken build still says so, without
    a line per call.
    """
    kernel_cache: dict[tuple[str, tuple[str, ...]], Any] = {}
    warned: set[tuple[str, tuple[str, ...]]] = set()

    def _get_or_build(kernel_name: str, template_args: Sequence[Any]) -> Optional[Any]:
        targs = tuple(str(a) for a in template_args)
        key = (kernel_name, targs)
        cached = kernel_cache.get(key, _KERNEL_CACHE_MISS)
        if cached is not _KERNEL_CACHE_MISS:
            return cached
        name_expression = f"{kernel_name}<{', '.join(targs)}>"
        source = _prepare_cuda_source(filename)
        try:
            kernel = _compile_raw_cuda_kernel(module, source, name_expression)
        except Exception as exc:
            if key not in warned:
                warned.add(key)
                logger.warning(
                    "%s RawKernel compilation failed (%s): %s", what, name_expression, exc
                )
            if not _is_deterministic_compile_failure(exc):
                # Retry next call rather than downgrading this process to
                # the fallback kernel for good.
                return None
            kernel = None
        kernel_cache[key] = kernel
        return kernel

    return _get_or_build


def _scatter_canonical_rows(module: Any, layout: _CombinationLayout, tmp: Any, out: Any) -> None:
    """out[r] = tmp[layout.rows[r]] (tmp holds the canonical rows)."""
    rows = module.asarray(layout.rows)
    out[...] = tmp[rows]


def _build_cupy_density_density_tomo_vectorized_kernel(module: Any) -> Any:
    """Builder for the GPU tomographic galaxy clustering ξ_g kernels.

    Contract: ``out_num`` / ``out_den`` are ``(ncomb, nbins_total)`` and hold
    both orientations of a cross combination summed.  The combination-tiled
    kernel (one block per angular bin, every pair visited once) serves the
    upper-triangle and auto-only layouts and, via a canonical temporary,
    any subset of them; the per-(bin, row) kernel is the fallback beyond the
    tile's accumulator budget.
    """
    build = _make_raw_kernel_builder(
        module, "density_density_tomo_vectorized.cu", "Vectorized density-density"
    )

    def _cupy_density_density_tomo_vectorized_kernel(
        density_map: Any,
        weights: Any,
        ind_i: Any,
        ind_j: Any,
        bin_offsets: Any,
        comb_i: Any,
        comb_j: Any,
        out_num: Any,
        out_den: Any,
    ) -> bool:
        nzbins = int(density_map.shape[1])
        if nzbins > _MAX_VECTOR_TOMO_BINS:
            return False
        if not _has_raw_cuda_compiler(module):
            return False

        map_c_type = "float" if weights.dtype == module.float32 else "double"
        index_c_type = "int" if ind_i.dtype == module.int32 else "long long"
        # Accumulator type follows the orchestrator-allocated output buffers.
        acc_c_type = "float" if out_num.dtype == module.float32 else "double"
        npairs = int(ind_i.shape[0])
        nbins_total = int(bin_offsets.shape[0] - 1)
        ncomb = int(comb_i.shape[0])
        threads = 256

        layout = _combination_layout(comb_i, comb_j, "triangle", nzbins)
        if (
            _cupy_density_density_tomo_vectorized_kernel.tiled
            and layout is not None
            and 2 * layout.canonical_ncomb <= _MAX_TILED_ACCUMULATORS
        ):
            auto_only = 1 if layout.mode == "auto" else 0
            raw_kernel = build(
                "gpu_tiled_tomo_reduce_dd",
                (map_c_type, nzbins, auto_only, index_c_type, acc_c_type),
            )
            if raw_kernel is not None:
                if layout.rows is None:
                    tgt_num, tgt_den = out_num, out_den
                else:
                    tgt_num = module.zeros((layout.canonical_ncomb, nbins_total), dtype=out_num.dtype)
                    tgt_den = module.zeros((layout.canonical_ncomb, nbins_total), dtype=out_den.dtype)
                raw_kernel(
                    (max(1, nbins_total), 1, 1),
                    (threads,),
                    (density_map, weights, ind_i, ind_j, bin_offsets,
                     tgt_num, tgt_den, np.int64(nbins_total)),
                )
                if layout.rows is not None:
                    _scatter_canonical_rows(module, layout, tgt_num, out_num)
                    _scatter_canonical_rows(module, layout, tgt_den, out_den)
                return True

        raw_kernel = build(
            "gpu_fused_tomo_reduce_dd", (map_c_type, nzbins, index_c_type, acc_c_type)
        )
        if raw_kernel is None:
            return False
        # Per-(bin, combination, orientation) rows; auto-combination B->A
        # rows are never written and must stay zero.
        tmp_num = module.zeros((2 * ncomb, nbins_total), dtype=out_num.dtype)
        tmp_den = module.zeros((2 * ncomb, nbins_total), dtype=out_den.dtype)
        raw_kernel(
            (max(1, nbins_total), max(1, 2 * ncomb), 1),
            (threads,),
            (
                density_map,
                weights,
                ind_i,
                ind_j,
                bin_offsets,
                comb_i,
                comb_j,
                tmp_num,
                tmp_den,
                np.int32(ncomb),
                np.int64(nbins_total),
                np.int64(npairs),
            ),
        )
        out_num[...] = tmp_num[0::2] + tmp_num[1::2]
        out_den[...] = tmp_den[0::2] + tmp_den[1::2]
        return True

    _cupy_density_density_tomo_vectorized_kernel.tiled = True
    return _cupy_density_density_tomo_vectorized_kernel


def _build_cupy_density_density_tomo_packed_kernel(module: Any) -> Any:
    """Builder for the tiled ξ_g kernel on packed pairs (8 B per pair)."""
    build = _make_raw_kernel_builder(
        module, "density_density_tomo_vectorized.cu", "Packed density-density"
    )

    def _cupy_density_density_tomo_packed_kernel(
        density_map: Any,
        weights: Any,
        pairs: Any,
        bin_offsets: Any,
        row_base: Any,
        comb_i: Any,
        comb_j: Any,
        out_num: Any,
        out_den: Any,
    ) -> bool:
        nzbins = int(density_map.shape[1])
        if nzbins > _MAX_VECTOR_TOMO_BINS or not _has_raw_cuda_compiler(module):
            return False
        layout = _combination_layout(comb_i, comb_j, "triangle", nzbins)
        if layout is None or 2 * layout.canonical_ncomb > _MAX_TILED_ACCUMULATORS:
            return False
        map_c_type = "float" if weights.dtype == module.float32 else "double"
        acc_c_type = "float" if out_num.dtype == module.float32 else "double"
        auto_only = 1 if layout.mode == "auto" else 0
        raw_kernel = build(
            "gpu_tiled_packed_reduce_dd", (map_c_type, nzbins, auto_only, acc_c_type)
        )
        if raw_kernel is None:
            return False
        nbins_total = int(bin_offsets.shape[0] - 1)
        if layout.rows is None:
            tgt_num, tgt_den = out_num, out_den
        else:
            tgt_num = module.zeros((layout.canonical_ncomb, nbins_total), dtype=out_num.dtype)
            tgt_den = module.zeros((layout.canonical_ncomb, nbins_total), dtype=out_den.dtype)
        raw_kernel(
            (max(1, nbins_total), 1, 1),
            (256,),
            (density_map, weights, pairs, bin_offsets, row_base,
             tgt_num, tgt_den, np.int64(nbins_total)),
        )
        if layout.rows is not None:
            _scatter_canonical_rows(module, layout, tgt_num, out_num)
            _scatter_canonical_rows(module, layout, tgt_den, out_den)
        return True

    return _cupy_density_density_tomo_packed_kernel


def _build_cupy_density_shear_tomo_vectorized_kernel(module: Any) -> Any:
    """Builder for the GPU tomographic galaxy-galaxy lensing ξ_t kernels.

    Contract: ``out_num`` / ``out_den`` are ``(ncomb, nbins_total)``; each
    pair contributes in both lens/source orientations.  The tiled kernel
    computes the full lens x source product (a requested subset is gathered
    from a canonical temporary); the per-(bin, combination) kernel is the
    fallback beyond the tile's accumulator budget.
    """
    build = _make_raw_kernel_builder(
        module, "density_shear_tomo_vectorized.cu", "Vectorized density-shear"
    )

    def _cupy_density_shear_tomo_vectorized_kernel(
        density_map: Any,
        shear_map: Any,
        lens_weights: Any,
        source_weights: Any,
        ind_i: Any,
        ind_j: Any,
        rot_i: Any,
        rot_j: Any,
        bin_offsets: Any,
        comb_i: Any,
        comb_j: Any,
        out_num: Any,
        out_den: Any,
    ) -> bool:
        nlens_bins = int(density_map.shape[1])
        nsource_bins = int(shear_map.shape[1])
        if nlens_bins > _MAX_VECTOR_TOMO_BINS or nsource_bins > _MAX_VECTOR_TOMO_BINS:
            return False
        if not _has_raw_cuda_compiler(module):
            return False

        complex_c_type = "cuFloatComplex" if rot_j.dtype == module.complex64 else "cuDoubleComplex"
        map_c_type = "float" if lens_weights.dtype == module.float32 else "double"
        index_c_type = "int" if ind_i.dtype == module.int32 else "long long"
        # Accumulator type follows the orchestrator-allocated output buffers.
        acc_c_type = "float" if out_num.dtype == module.float32 else "double"
        npairs = int(ind_i.shape[0])
        nbins_total = int(bin_offsets.shape[0] - 1)
        ncomb = int(comb_i.shape[0])
        threads = 256

        layout = _combination_layout(comb_i, comb_j, "cartesian", nlens_bins, nsource_bins)
        if (
            _cupy_density_shear_tomo_vectorized_kernel.tiled
            and layout is not None
            and 2 * layout.canonical_ncomb <= _MAX_TILED_ACCUMULATORS
        ):
            raw_kernel = build(
                "gpu_tiled_tomo_reduce_ds",
                (map_c_type, complex_c_type, nlens_bins, nsource_bins, index_c_type, acc_c_type),
            )
            if raw_kernel is not None:
                if layout.rows is None:
                    tgt_num, tgt_den = out_num, out_den
                else:
                    tgt_num = module.zeros((layout.canonical_ncomb, nbins_total), dtype=out_num.dtype)
                    tgt_den = module.zeros((layout.canonical_ncomb, nbins_total), dtype=out_den.dtype)
                raw_kernel(
                    (max(1, nbins_total), 1, 1),
                    (threads,),
                    (density_map, shear_map, lens_weights, source_weights,
                     ind_i, ind_j, rot_i, rot_j, bin_offsets,
                     tgt_num, tgt_den, np.int64(nbins_total)),
                )
                if layout.rows is not None:
                    _scatter_canonical_rows(module, layout, tgt_num, out_num)
                    _scatter_canonical_rows(module, layout, tgt_den, out_den)
                return True

        raw_kernel = build(
            "gpu_fused_tomo_reduce_ds",
            (map_c_type, complex_c_type, nlens_bins, nsource_bins, index_c_type, acc_c_type),
        )
        if raw_kernel is None:
            return False
        raw_kernel(
            (max(1, nbins_total), max(1, ncomb), 1),
            (threads,),
            (
                density_map,
                shear_map,
                lens_weights,
                source_weights,
                ind_i,
                ind_j,
                rot_i,
                rot_j,
                bin_offsets,
                comb_i,
                comb_j,
                out_num,
                out_den,
                np.int32(ncomb),
                np.int64(nbins_total),
                np.int64(npairs),
            ),
        )
        return True

    _cupy_density_shear_tomo_vectorized_kernel.tiled = True
    return _cupy_density_shear_tomo_vectorized_kernel


def _build_cupy_density_shear_tomo_packed_kernel(module: Any) -> Any:
    """Builder for the tiled ξ_t kernel on packed pairs (8 B per pair)."""
    build = _make_raw_kernel_builder(
        module, "density_shear_tomo_vectorized.cu", "Packed density-shear"
    )

    def _cupy_density_shear_tomo_packed_kernel(
        density_map: Any,
        shear_map: Any,
        lens_weights: Any,
        source_weights: Any,
        pairs: Any,
        bin_offsets: Any,
        row_base: Any,
        comb_i: Any,
        comb_j: Any,
        out_num: Any,
        out_den: Any,
    ) -> bool:
        nlens_bins = int(density_map.shape[1])
        nsource_bins = int(shear_map.shape[1])
        if nlens_bins > _MAX_VECTOR_TOMO_BINS or nsource_bins > _MAX_VECTOR_TOMO_BINS:
            return False
        if not _has_raw_cuda_compiler(module):
            return False
        layout = _combination_layout(comb_i, comb_j, "cartesian", nlens_bins, nsource_bins)
        if layout is None or 2 * layout.canonical_ncomb > _MAX_TILED_ACCUMULATORS:
            return False
        map_c_type = "float" if lens_weights.dtype == module.float32 else "double"
        acc_c_type = "float" if out_num.dtype == module.float32 else "double"
        raw_kernel = build(
            "gpu_tiled_packed_reduce_ds", (map_c_type, nlens_bins, nsource_bins, acc_c_type)
        )
        if raw_kernel is None:
            return False
        nbins_total = int(bin_offsets.shape[0] - 1)
        if layout.rows is None:
            tgt_num, tgt_den = out_num, out_den
        else:
            tgt_num = module.zeros((layout.canonical_ncomb, nbins_total), dtype=out_num.dtype)
            tgt_den = module.zeros((layout.canonical_ncomb, nbins_total), dtype=out_den.dtype)
        raw_kernel(
            (max(1, nbins_total), 1, 1),
            (256,),
            (density_map, shear_map, lens_weights, source_weights, pairs,
             bin_offsets, row_base, tgt_num, tgt_den, np.int64(nbins_total)),
        )
        if layout.rows is not None:
            _scatter_canonical_rows(module, layout, tgt_num, out_num)
            _scatter_canonical_rows(module, layout, tgt_den, out_den)
        return True

    return _cupy_density_shear_tomo_packed_kernel


# ── Shear-shear 2PCF kernels ─────────────────────────────────────────────
# Compute ξ+(θ) and ξ-(θ) from the complex shear γ = γ₁ + iγ₂.
# The shear is rotated into the pair frame via γ' = γ · e^{2iφ}, then:
#   ξ+ = Re[γ'_b · conj(γ'_a)]   (sensitive to E+B mode power)
#   ξ- = Re[γ'_b · γ'_a]          (sensitive to E-B mode power)

@njit(fastmath=True, parallel=True, cache=True)
def _cpu_xipm_cross_corr_kernel(
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
    offsets: np.ndarray,
    out_ab_p: np.ndarray,
    out_ab_m: np.ndarray,
    out_ba_p: np.ndarray,
    out_ba_m: np.ndarray,
    out_ab_w: np.ndarray,
    out_ba_w: np.ndarray,
) -> None:
    """Cross-correlation ξ+/ξ- between two different shear catalogues (a, b).

    Computes both A→B and B→A orientations for asymmetric cross-correlations.
    Only the real parts of the estimators are accumulated — every caller
    discards the imaginary parts — and the weight sums (denominators) are
    accumulated in the same pass over the pair list.
    """
    nbins = offsets.shape[0] - 1
    # float64 explicitly, not ``x[0] * 0.0``.  That expression types as
    # float64 under numba (python-float promotion) but as float32 under
    # NUMBA_DISABLE_JIT=1, which pytest-env forces -- so the suite was
    # measuring a float32 accumulation of a kernel that ships accumulating in
    # float64.  Pinning it keeps the shipped numbers and makes the tests
    # exercise them.  (It is deliberately not ``acc_dtype``: these four
    # kernels reduce over a whole aperture disc or angular bin, where a
    # float32 accumulator loses precision for nothing.)
    zero = np.float64(0)
    for b in prange(nbins):
        ab_p_re = zero
        ab_m_re = zero
        ba_p_re = zero
        ba_m_re = zero
        ab_w = zero
        ba_w = zero
        start = offsets[b]
        stop = offsets[b + 1]

        for idx in range(start, stop):
            i = ind_i[idx]
            j = ind_j[idx]

            exp_ir = exp_i[idx].real
            exp_ii = exp_i[idx].imag
            exp_jr = exp_j[idx].real
            exp_ji = exp_j[idx].imag

            ga_i_r = g1a[i]
            ga_i_i = g2a[i]
            gb_i_r = g1b[i]
            gb_i_i = g2b[i]
            ga_j_r = g1a[j]
            ga_j_i = g2a[j]
            gb_j_r = g1b[j]
            gb_j_i = g2b[j]

            ga_i_rot_r = wa[i] * (ga_i_r * exp_ir - ga_i_i * exp_ii)
            ga_i_rot_i = wa[i] * (ga_i_r * exp_ii + ga_i_i * exp_ir)
            gb_i_rot_r = wb[i] * (gb_i_r * exp_ir - gb_i_i * exp_ii)
            gb_i_rot_i = wb[i] * (gb_i_r * exp_ii + gb_i_i * exp_ir)
            ga_j_rot_r = wa[j] * (ga_j_r * exp_jr - ga_j_i * exp_ji)
            ga_j_rot_i = wa[j] * (ga_j_r * exp_ji + ga_j_i * exp_jr)
            gb_j_rot_r = wb[j] * (gb_j_r * exp_jr - gb_j_i * exp_ji)
            gb_j_rot_i = wb[j] * (gb_j_r * exp_ji + gb_j_i * exp_jr)

            ab_p_re += gb_j_rot_r * ga_i_rot_r + gb_j_rot_i * ga_i_rot_i
            ab_m_re += gb_j_rot_r * ga_i_rot_r - gb_j_rot_i * ga_i_rot_i

            ba_p_re += ga_j_rot_r * gb_i_rot_r + ga_j_rot_i * gb_i_rot_i
            ba_m_re += ga_j_rot_r * gb_i_rot_r - ga_j_rot_i * gb_i_rot_i

            ab_w += wa[i] * wb[j]
            ba_w += wb[i] * wa[j]

        out_ab_p[b] = ab_p_re
        out_ab_m[b] = ab_m_re
        out_ba_p[b] = ba_p_re
        out_ba_m[b] = ba_m_re
        out_ab_w[b] = ab_w
        out_ba_w[b] = ba_w


@njit(fastmath=True, parallel=True, cache=True)
def _cpu_xipm_auto_corr_kernel(
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
    offsets: np.ndarray,
    out_p: np.ndarray,
    out_m: np.ndarray,
    out_w: np.ndarray,
) -> None:
    """Auto-correlation ξ+/ξ- within a single tomo-bin pair.

    Only one orientation is needed (the estimator is symmetric).  Only the
    real parts are accumulated (callers discard the imaginary parts) and
    the weight sum (denominator) is accumulated in the same pass.
    """
    nbins = offsets.shape[0] - 1
    # float64 explicitly, not ``x[0] * 0.0``.  That expression types as
    # float64 under numba (python-float promotion) but as float32 under
    # NUMBA_DISABLE_JIT=1, which pytest-env forces -- so the suite was
    # measuring a float32 accumulation of a kernel that ships accumulating in
    # float64.  Pinning it keeps the shipped numbers and makes the tests
    # exercise them.  (It is deliberately not ``acc_dtype``: these four
    # kernels reduce over a whole aperture disc or angular bin, where a
    # float32 accumulator loses precision for nothing.)
    zero = np.float64(0)
    for b in prange(nbins):
        p_acc_re = zero
        m_acc_re = zero
        w_acc = zero
        start = offsets[b]
        stop = offsets[b + 1]

        for idx in range(start, stop):
            i = ind_i[idx]
            j = ind_j[idx]

            exp_ir = exp_i[idx].real
            exp_ii = exp_i[idx].imag
            exp_jr = exp_j[idx].real
            exp_ji = exp_j[idx].imag

            g2r = w1[i] * (g11[i] * exp_ir - g21[i] * exp_ii)
            g2i = w1[i] * (g11[i] * exp_ii + g21[i] * exp_ir)
            g1r = w2[j] * (g12[j] * exp_jr - g22[j] * exp_ji)
            g1i = w2[j] * (g12[j] * exp_ji + g22[j] * exp_jr)

            p_acc_re += g1r * g2r + g1i * g2i
            m_acc_re += g1r * g2r - g1i * g2i
            w_acc += w1[i] * w2[j]

        out_p[b] = p_acc_re
        out_m[b] = m_acc_re
        out_w[b] = w_acc


@njit(fastmath=True, parallel=True, cache=True)
def _cpu_vectorized_tomo_kernel(
    shear_map: np.ndarray,
    weights: np.ndarray,
    ind_i: np.ndarray,
    ind_j: np.ndarray,
    rot_i: np.ndarray,
    rot_j: np.ndarray,
    offsets: np.ndarray,
    comb_i: np.ndarray,
    comb_j: np.ndarray,
    out_p: np.ndarray,
    out_m: np.ndarray,
    out_w: np.ndarray,
) -> None:
    """Vectorised cosmic shear ξ+/ξ- for all tomo-bin combinations.

    For each pair, rotates the shear into the pair frame and computes
    both ξ+ = Re[γ'_b·conj(γ'_a)] and ξ- = Re[γ'_b·γ'_a].
    Cross-bin pairs (i≠j) contribute in both A→B and B→A orientations,
    which are summed into one numerator and one weight sum per
    combination (ratio-of-sums estimator, as for ξ_g and ξ_t).

    The pair loop is outermost within each bin so the pair indices and
    rotation factors are loaded once per pair and reused for every
    tomographic combination.  Only the real parts of the estimators are
    accumulated, and the weight sums (denominators) are accumulated in
    the same pass.  Output rows: one per combination k.
    """
    n_bins = offsets.shape[0] - 1
    ncomb = comb_i.shape[0]

    for b in prange(n_bins):
        start = offsets[b]
        stop = offsets[b + 1]
        acc_p = np.zeros(ncomb, dtype=out_p.dtype)
        acc_m = np.zeros(ncomb, dtype=out_m.dtype)
        acc_w = np.zeros(ncomb, dtype=out_w.dtype)

        for idx in range(start, stop):
            pix_i = int(ind_i[idx])
            pix_j = int(ind_j[idx])
            exp_i = rot_i[idx]
            exp_j = rot_j[idx]
            exp_i_re = exp_i.real
            exp_i_im = exp_i.imag
            exp_j_re = exp_j.real
            exp_j_im = exp_j.imag

            for comb_idx in range(ncomb):
                i = comb_i[comb_idx]
                j = comb_j[comb_idx]

                ga1 = shear_map[pix_i, i, 0]
                ga2 = shear_map[pix_i, i, 1]
                gb1 = shear_map[pix_j, j, 0]
                gb2 = shear_map[pix_j, j, 1]

                a_r = ga1 * exp_i_re - ga2 * exp_i_im
                a_i = ga1 * exp_i_im + ga2 * exp_i_re
                b_r = gb1 * exp_j_re - gb2 * exp_j_im
                b_i = gb1 * exp_j_im + gb2 * exp_j_re

                w_ij = weights[pix_i, i] * weights[pix_j, j]
                acc_p[comb_idx] += w_ij * (b_r * a_r + b_i * a_i)
                acc_m[comb_idx] += w_ij * (b_r * a_r - b_i * a_i)
                acc_w[comb_idx] += w_ij

                if i != j:
                    gc1 = shear_map[pix_j, i, 0]
                    gc2 = shear_map[pix_j, i, 1]
                    gd1 = shear_map[pix_i, j, 0]
                    gd2 = shear_map[pix_i, j, 1]

                    c_r = gc1 * exp_j_re - gc2 * exp_j_im
                    c_i = gc1 * exp_j_im + gc2 * exp_j_re
                    d_r = gd1 * exp_i_re - gd2 * exp_i_im
                    d_i = gd1 * exp_i_im + gd2 * exp_i_re

                    w_ji = weights[pix_i, j] * weights[pix_j, i]
                    acc_p[comb_idx] += w_ji * (c_r * d_r + c_i * d_i)
                    acc_m[comb_idx] += w_ji * (c_r * d_r - c_i * d_i)
                    acc_w[comb_idx] += w_ji

        for comb_idx in range(ncomb):
            out_p[comb_idx, b] = acc_p[comb_idx]
            out_m[comb_idx, b] = acc_m[comb_idx]
            out_w[comb_idx, b] = acc_w[comb_idx]


def _build_cupy_tomo_vectorized_kernel(module: Any) -> Any:
    """Builder for the GPU tomographic cosmic shear ξ+/ξ- kernels.

    Contract: ``out_num`` is ``(2, ncomb, nbins_total)`` = [ξ+ | ξ-] and
    ``out_den`` is ``(ncomb, nbins_total)``; both orientations of a cross
    combination are summed.  The combination-*tiled* kernel (default; one
    block per angular bin walks every pair once and accumulates all
    combinations) serves the upper-triangle layout the orchestrator
    produces; the per-(bin, row) kernel is the fallback (its orientation
    rows are summed here).  Set ``kernel.tiled = False`` on the returned
    callable to force the per-row kernel.
    """
    build = _make_raw_kernel_builder(module, "tomo_vectorized_xipm.cu", "Vectorized tomography")

    def _cupy_tomo_vectorized_kernel(
        shear_map: Any,
        weights: Any,
        ind_i: Any,
        ind_j: Any,
        rot_i: Any,
        rot_j: Any,
        bin_offsets: Any,
        comb_i: Any,
        comb_j: Any,
        out_num: Any,
        out_den: Any,
    ) -> bool:
        nzbins = int(shear_map.shape[1])
        if nzbins > _MAX_VECTOR_TOMO_BINS:
            return False
        if not _has_raw_cuda_compiler(module):
            return False

        complex_c_type = "cuFloatComplex" if rot_i.dtype == module.complex64 else "cuDoubleComplex"
        map_c_type = "float" if weights.dtype == module.float32 else "double"
        index_c_type = "int" if ind_i.dtype == module.int32 else "long long"
        # The accumulator type follows the (orchestrator-allocated) output
        # buffers: float64 outputs on float32 maps select double accumulation.
        acc_c_type = "float" if out_num.dtype == module.float32 else "double"
        ncomb = int(comb_i.shape[0])
        npairs = int(ind_i.shape[0])
        nbins_total = int(bin_offsets.shape[0] - 1)
        threads = 256

        layout = _combination_layout(comb_i, comb_j, "triangle", nzbins)
        if (
            _cupy_tomo_vectorized_kernel.tiled
            and layout is not None
            and layout.mode == "triangle"
            and layout.rows is None
            and 3 * layout.canonical_ncomb <= _MAX_TILED_ACCUMULATORS
        ):
            raw_kernel = build(
                "gpu_tiled_tomo_reduce_xipm",
                (map_c_type, complex_c_type, nzbins, index_c_type, acc_c_type),
            )
            if raw_kernel is not None:
                raw_kernel(
                    (max(1, nbins_total), 1, 1),
                    (threads,),
                    (shear_map, weights, ind_i, ind_j, rot_i, rot_j, bin_offsets,
                     out_num, out_den, np.int64(nbins_total)),
                )
                return True

        raw_kernel = build(
            "gpu_fused_tomo_reduce_xipm",
            (map_c_type, complex_c_type, nzbins, index_c_type, acc_c_type),
        )
        if raw_kernel is None:
            return False
        # Per-(bin, combination, orientation) rows; auto-combination B->A
        # rows are never written and must stay zero.
        tmp_num = module.zeros((2, 2 * ncomb, nbins_total), dtype=out_num.dtype)
        tmp_den = module.zeros((2 * ncomb, nbins_total), dtype=out_den.dtype)
        raw_kernel(
            (max(1, nbins_total), max(1, 2 * ncomb), 1),
            (threads,),
            (
                shear_map,
                weights,
                ind_i,
                ind_j,
                rot_i,
                rot_j,
                bin_offsets,
                comb_i,
                comb_j,
                tmp_num,
                tmp_den,
                np.int32(ncomb),
                np.int64(nbins_total),
                np.int64(npairs),
            ),
        )
        out_num[...] = tmp_num[:, 0::2] + tmp_num[:, 1::2]
        out_den[...] = tmp_den[0::2] + tmp_den[1::2]
        return True

    _cupy_tomo_vectorized_kernel.tiled = True
    return _cupy_tomo_vectorized_kernel


def _build_cupy_tomo_packed_kernel(module: Any) -> Any:
    """Builder for the combination-tiled ξ+/ξ- kernel on packed pairs
    (8 bytes per pair, see ``cuda/pair_tiles.cuh``); same output contract
    as the unpacked wrapper."""
    build = _make_raw_kernel_builder(module, "tomo_vectorized_xipm.cu", "Packed ξ±")

    def _cupy_tomo_packed_kernel(
        shear_map: Any,
        weights: Any,
        pairs: Any,
        bin_offsets: Any,
        row_base: Any,
        comb_i: Any,
        comb_j: Any,
        out_num: Any,
        out_den: Any,
    ) -> bool:
        nzbins = int(shear_map.shape[1])
        if nzbins > _MAX_VECTOR_TOMO_BINS or not _has_raw_cuda_compiler(module):
            return False
        layout = _combination_layout(comb_i, comb_j, "triangle", nzbins)
        if (
            layout is None
            or layout.mode != "triangle"
            or layout.rows is not None
            or 3 * layout.canonical_ncomb > _MAX_TILED_ACCUMULATORS
        ):
            return False
        map_c_type = "float" if weights.dtype == module.float32 else "double"
        acc_c_type = "float" if out_num.dtype == module.float32 else "double"
        raw_kernel = build("gpu_tiled_packed_reduce_xipm", (map_c_type, nzbins, acc_c_type))
        if raw_kernel is None:
            return False
        nbins_total = int(bin_offsets.shape[0] - 1)
        raw_kernel(
            (max(1, nbins_total), 1, 1),
            (256,),
            (shear_map, weights, pairs, bin_offsets, row_base,
             out_num, out_den, np.int64(nbins_total)),
        )
        return True

    return _cupy_tomo_packed_kernel


# ── Fused 3×2pt CPU kernel ────────────────────────────────────────────────
# CPU equivalent of the fused CUDA kernel in tomo_fused_3x2pt.cu.
# Computes all six 3×2pt outputs (M_ap, M_g, ξ+, ξ-, ξ_g, ξ_t) in
# separate prange loops — each loop parallelises over tomo combinations
# or patch indices.

@njit(fastmath=True, parallel=True, cache=True)
def _cpu_3x2pt_tomo_fused_kernel(
    density_map: np.ndarray,
    shear_map: np.ndarray,
    density_weights: np.ndarray,
    shear_weights: np.ndarray,
    ind_i: np.ndarray,
    ind_j: np.ndarray,
    rot_i: np.ndarray,
    rot_j: np.ndarray,
    pair_offsets: np.ndarray,
    q_inds: np.ndarray,
    q_cos: np.ndarray,
    q_sin: np.ndarray,
    q_val: np.ndarray,
    q_offsets: np.ndarray,
    q_patch_area: np.ndarray,
    ss_comb_i: np.ndarray,
    ss_comb_j: np.ndarray,
    dd_comb_i: np.ndarray,
    dd_comb_j: np.ndarray,
    ds_comb_i: np.ndarray,
    ds_comb_j: np.ndarray,
    out_ma_num: np.ndarray,
    out_ma_den: np.ndarray,
    out_mg_num: np.ndarray,
    out_mg_den: np.ndarray,
    out_xip_num: np.ndarray,
    out_xim_num: np.ndarray,
    out_xipm_den: np.ndarray,
    out_xig_num: np.ndarray,
    out_xig_den: np.ndarray,
    out_xit_num: np.ndarray,
    out_xit_den: np.ndarray,
) -> None:
    n_patches = q_offsets.shape[0] - 1
    n_shear = shear_map.shape[1]
    n_density = density_map.shape[1]
    nbins_total = pair_offsets.shape[0] - 1

    # --- Aperture mass M_ap: γ_t convolved with Q(θ) per patch ---
    # prange over the flattened (tomo, patch) product so parallelism is not
    # capped at the handful of tomographic bins.
    for flat_idx in prange(n_shear * n_patches):
        tomo_idx = flat_idx // n_patches
        patch_idx = flat_idx - tomo_idx * n_patches
        start = q_offsets[patch_idx]
        stop = q_offsets[patch_idx + 1]
        sum_w = 0.0
        sum_num = 0.0
        for q_idx in range(start, stop):
            pix_idx = q_inds[q_idx]
            weight = shear_weights[pix_idx, tomo_idx]
            gt = -shear_map[pix_idx, tomo_idx, 0] * q_cos[q_idx] - shear_map[pix_idx, tomo_idx, 1] * q_sin[q_idx]
            sum_w += weight
            sum_num += weight * gt * q_val[q_idx]

        out_ma_num[tomo_idx, patch_idx] = q_patch_area[patch_idx] * sum_num
        out_ma_den[tomo_idx, patch_idx] = sum_w

    # --- Galaxy mean density M_g: δ_g convolved with Q(θ) per patch ---
    for flat_idx in prange(n_density * n_patches):
        tomo_idx = flat_idx // n_patches
        patch_idx = flat_idx - tomo_idx * n_patches
        start = q_offsets[patch_idx]
        stop = q_offsets[patch_idx + 1]
        sum_w = 0.0
        sum_num = 0.0
        for q_idx in range(start, stop):
            pix_idx = q_inds[q_idx]
            weight = density_weights[pix_idx, tomo_idx]
            sum_w += weight
            sum_num += weight * density_map[pix_idx, tomo_idx] * q_val[q_idx]

        out_mg_num[tomo_idx, patch_idx] = q_patch_area[patch_idx] * sum_num
        out_mg_den[tomo_idx, patch_idx] = sum_w

    # --- Cosmic shear ξ+/ξ- ---
    # prange over angular bins; pair loop outermost inside each bin so the
    # pair indices/rotations are loaded once and reused for every
    # tomographic combination.  Both orientations of a cross combination
    # are summed into one row (ratio-of-sums estimator).
    n_ss_comb = ss_comb_i.shape[0]
    for bin_flat in prange(nbins_total):
        start = pair_offsets[bin_flat]
        stop = pair_offsets[bin_flat + 1]
        acc_p = np.zeros(n_ss_comb, dtype=out_xip_num.dtype)
        acc_m = np.zeros(n_ss_comb, dtype=out_xim_num.dtype)
        acc_w = np.zeros(n_ss_comb, dtype=out_xipm_den.dtype)

        for pair_idx in range(start, stop):
            pix_a = ind_i[pair_idx]
            pix_b = ind_j[pair_idx]
            exp_a = rot_i[pair_idx]
            exp_b = rot_j[pair_idx]
            exp_a_re = exp_a.real
            exp_a_im = exp_a.imag
            exp_b_re = exp_b.real
            exp_b_im = exp_b.imag

            for comb_idx in range(n_ss_comb):
                i_bin = ss_comb_i[comb_idx]
                j_bin = ss_comb_j[comb_idx]

                ga1 = shear_map[pix_a, i_bin, 0]
                ga2 = shear_map[pix_a, i_bin, 1]
                gb1 = shear_map[pix_b, j_bin, 0]
                gb2 = shear_map[pix_b, j_bin, 1]

                a_r = ga1 * exp_a_re - ga2 * exp_a_im
                a_i = ga1 * exp_a_im + ga2 * exp_a_re
                b_r = gb1 * exp_b_re - gb2 * exp_b_im
                b_i = gb1 * exp_b_im + gb2 * exp_b_re

                w_pair = shear_weights[pix_a, i_bin] * shear_weights[pix_b, j_bin]
                acc_w[comb_idx] += w_pair
                acc_p[comb_idx] += w_pair * (b_r * a_r + b_i * a_i)
                acc_m[comb_idx] += w_pair * (b_r * a_r - b_i * a_i)

                if i_bin != j_bin:
                    gc1 = shear_map[pix_a, j_bin, 0]
                    gc2 = shear_map[pix_a, j_bin, 1]
                    gd1 = shear_map[pix_b, i_bin, 0]
                    gd2 = shear_map[pix_b, i_bin, 1]

                    c_r = gc1 * exp_a_re - gc2 * exp_a_im
                    c_i = gc1 * exp_a_im + gc2 * exp_a_re
                    d_r = gd1 * exp_b_re - gd2 * exp_b_im
                    d_i = gd1 * exp_b_im + gd2 * exp_b_re

                    w_ba = shear_weights[pix_a, j_bin] * shear_weights[pix_b, i_bin]
                    acc_w[comb_idx] += w_ba
                    acc_p[comb_idx] += w_ba * (d_r * c_r + d_i * c_i)
                    acc_m[comb_idx] += w_ba * (d_r * c_r - d_i * c_i)

        for row in range(n_ss_comb):
            out_xip_num[row, bin_flat] = acc_p[row]
            out_xim_num[row, bin_flat] = acc_m[row]
            out_xipm_den[row, bin_flat] = acc_w[row]

    # --- Galaxy clustering ξ_g (both orientations summed per row) ---
    n_dd_comb = dd_comb_i.shape[0]
    for bin_flat in prange(nbins_total):
        start = pair_offsets[bin_flat]
        stop = pair_offsets[bin_flat + 1]
        acc_num = np.zeros(n_dd_comb, dtype=out_xig_num.dtype)
        acc_den = np.zeros(n_dd_comb, dtype=out_xig_den.dtype)

        for pair_idx in range(start, stop):
            pix_a = ind_i[pair_idx]
            pix_b = ind_j[pair_idx]

            for comb_idx in range(n_dd_comb):
                i_bin = dd_comb_i[comb_idx]
                j_bin = dd_comb_j[comb_idx]

                w_ab = density_weights[pix_a, i_bin] * density_weights[pix_b, j_bin]
                acc_den[comb_idx] += w_ab
                acc_num[comb_idx] += w_ab * density_map[pix_a, i_bin] * density_map[pix_b, j_bin]

                if i_bin != j_bin:
                    w_ba = density_weights[pix_a, j_bin] * density_weights[pix_b, i_bin]
                    acc_den[comb_idx] += w_ba
                    acc_num[comb_idx] += w_ba * density_map[pix_a, j_bin] * density_map[pix_b, i_bin]

        for row in range(n_dd_comb):
            out_xig_num[row, bin_flat] = acc_num[row]
            out_xig_den[row, bin_flat] = acc_den[row]

    # --- Galaxy-galaxy lensing ξ_t ---
    n_ds_comb = ds_comb_i.shape[0]
    for bin_flat in prange(nbins_total):
        start = pair_offsets[bin_flat]
        stop = pair_offsets[bin_flat + 1]
        acc_num = np.zeros(n_ds_comb, dtype=out_xit_num.dtype)
        acc_den = np.zeros(n_ds_comb, dtype=out_xit_den.dtype)

        for pair_idx in range(start, stop):
            pix_a = ind_i[pair_idx]
            pix_b = ind_j[pair_idx]
            exp_ab = rot_j[pair_idx]
            exp_ba = rot_i[pair_idx]
            exp_ab_re = exp_ab.real
            exp_ab_im = exp_ab.imag
            exp_ba_re = exp_ba.real
            exp_ba_im = exp_ba.imag

            for comb_idx in range(n_ds_comb):
                lens_bin = ds_comb_i[comb_idx]
                source_bin = ds_comb_j[comb_idx]

                gt_ab = (
                    -shear_map[pix_b, source_bin, 0] * exp_ab_re
                    + shear_map[pix_b, source_bin, 1] * exp_ab_im
                )
                w_ab = density_weights[pix_a, lens_bin] * shear_weights[pix_b, source_bin]
                acc_num[comb_idx] += w_ab * density_map[pix_a, lens_bin] * gt_ab
                acc_den[comb_idx] += w_ab

                gt_ba = (
                    -shear_map[pix_a, source_bin, 0] * exp_ba_re
                    + shear_map[pix_a, source_bin, 1] * exp_ba_im
                )
                w_ba = density_weights[pix_b, lens_bin] * shear_weights[pix_a, source_bin]
                acc_num[comb_idx] += w_ba * density_map[pix_b, lens_bin] * gt_ba
                acc_den[comb_idx] += w_ba

        for comb_idx in range(n_ds_comb):
            out_xit_num[comb_idx, bin_flat] = acc_num[comb_idx]
            out_xit_den[comb_idx, bin_flat] = acc_den[comb_idx]


def _build_cupy_3x2pt_tomo_aperture_kernel(module: Any) -> Any:
    """Builder for the aperture sections (M_ap, M_g) of the fused 3x2pt path
    (``tomo_fused_3x2pt.cu``; AoS inputs, accumulation at ACC).  The pair
    statistics of ``get_3x2pt_tomo`` run in the tiled pair kernels."""
    build = _make_raw_kernel_builder(module, "tomo_fused_3x2pt.cu", "Fused 3x2pt aperture")
    # One non-blocking stream per section, created lazily on first launch
    # (per device: each Backend instance gets its own builder closure).
    section_streams: list = []

    def _cupy_3x2pt_tomo_aperture_kernel(
        density_map: Any,
        shear_map: Any,
        density_weights: Any,
        shear_weights: Any,
        q_inds: Any,
        q_cos: Any,
        q_sin: Any,
        q_val: Any,
        q_offsets: Any,
        q_patch_area: Any,
        out_ma_num: Any,
        out_ma_den: Any,
        out_mg_num: Any,
        out_mg_den: Any,
    ) -> bool:
        if not _has_raw_cuda_compiler(module):
            return False
        n_density_bins = int(density_map.shape[1])
        n_shear_bins = int(shear_map.shape[1])
        map_c_type = "float" if density_map.dtype == module.float32 else "double"
        q_c_type = "float" if q_cos.dtype == module.float32 else "double"
        # Accumulator type follows the orchestrator-allocated output buffers.
        acc_c_type = "float" if out_ma_num.dtype == module.float32 else "double"
        npatches = int(q_offsets.shape[0] - 1)
        # One decision for both sections: they share the kernel, and the
        # widest bin set is what the accumulator budget has to cover.
        fused = _use_fused_aperture(
            module, max(n_density_bins, n_shear_bins), npatches
        )
        raw_kernel = build(
            "gpu_3x2pt_tomo_aperture_fused" if fused else "gpu_3x2pt_tomo_aperture",
            (map_c_type, q_c_type, n_density_bins, n_shear_bins, acc_c_type),
        )
        if raw_kernel is None:
            return False

        base_args = (
            density_map,
            shear_map,
            density_weights,
            shear_weights,
            np.int32(npatches),
            q_inds,
            q_cos,
            q_sin,
            q_val,
            q_offsets,
            q_patch_area,
            out_ma_num,
            out_ma_den,
            out_mg_num,
            out_mg_den,
        )
        section_grids = (
            (npatches, 1 if fused else n_shear_bins),      # z=0  M_ap
            (npatches, 1 if fused else n_density_bins),    # z=1  M_g
        )
        if not section_streams:
            section_streams.extend(
                module.cuda.Stream(non_blocking=True) for _ in range(len(section_grids))
            )
        # The sections write disjoint outputs -> run them concurrently.
        current = module.cuda.get_current_stream()
        ready = current.record()                 # inputs staged on current stream
        for z, (gx, gy) in enumerate(section_grids):
            if gx <= 0 or gy <= 0:
                continue
            stream = section_streams[z]
            stream.wait_event(ready)
            with stream:
                raw_kernel((int(gx), int(gy), 1), (256,), base_args + (np.int32(z),))
            current.wait_event(stream.record())  # downstream default-stream work waits
        return True

    return _cupy_3x2pt_tomo_aperture_kernel


def _build_cupy_3x2pt_tomo_pairs_kernel(module: Any) -> Any:
    """Builder for the multi-statistic pair tile of the fused 3x2pt path
    (``tomo_tiled_3x2pt.cu``): xi+-, xi_g and xi_t in one walk over the pairs.

    The wrapper returns the statistics it computed as ``(ss, dd, ds)`` flags;
    the orchestrator launches the standalone tiles for the rest.  The single
    pass is used while all accumulators fit the register budget
    (``max_accumulators``); ``mode = "off"`` disables it, ``"single"``
    forces it (benchmarks).  A two-pass split (xi+- with xi_t, xi_g apart)
    was measured too and dropped: slower than three tiles on packed pairs.
    """
    build = _make_raw_kernel_builder(module, "tomo_tiled_3x2pt.cu", "Tiled 3x2pt")
    none = (False, False, False)

    def _cupy_3x2pt_tomo_pairs_kernel(
        density_map: Any,
        shear_map: Any,
        density_weights: Any,
        shear_weights: Any,
        geometry: Tuple[Any, ...],
        bin_offsets: Any,
        ss_comb: Tuple[Any, Any],
        dd_comb: Tuple[Any, Any],
        ds_comb: Tuple[Any, Any],
        out_xipm_num: Any,
        out_xipm_den: Any,
        out_xig_num: Any,
        out_xig_den: Any,
        out_xit_num: Any,
        out_xit_den: Any,
    ) -> Tuple[bool, bool, bool]:
        """``geometry`` is ``(ind_i, ind_j, rot_i, rot_j)`` or, packed,
        ``(pairs, row_base)``."""
        mode = _cupy_3x2pt_tomo_pairs_kernel.mode
        if mode == "off" or not _has_raw_cuda_compiler(module):
            return none
        n_density = int(density_map.shape[1])
        n_shear = int(shear_map.shape[1])
        ss = _combination_layout(ss_comb[0], ss_comb[1], "triangle", n_shear)
        dd = _combination_layout(dd_comb[0], dd_comb[1], "triangle", n_density)
        ds = _combination_layout(ds_comb[0], ds_comb[1], "cartesian", n_density, n_shear)
        if ss is None or dd is None or ds is None or ss.mode != "triangle" or ss.rows is not None:
            return none
        n_ss, n_dd, n_ds = 3 * ss.canonical_ncomb, 2 * dd.canonical_ncomb, 2 * ds.canonical_ncomb
        budget = _cupy_3x2pt_tomo_pairs_kernel.max_accumulators
        if mode != "single" and n_ss + n_dd + n_ds > budget:
            return none
        do = (1, 1, 1)

        packed = len(geometry) == 2
        map_c_type = "float" if density_map.dtype == module.float32 else "double"
        acc_c_type = "float" if out_xipm_num.dtype == module.float32 else "double"
        auto_only = 1 if dd.mode == "auto" else 0
        if packed:
            raw_kernel = build(
                "gpu_tiled_packed_reduce_3x2pt",
                (map_c_type, n_density, n_shear, auto_only, *do, acc_c_type),
            )
        else:
            rot = geometry[2]
            complex_c_type = "cuFloatComplex" if rot.dtype == module.complex64 else "cuDoubleComplex"
            index_c_type = "int" if geometry[0].dtype == module.int32 else "long long"
            raw_kernel = build(
                "gpu_tiled_tomo_reduce_3x2pt",
                (map_c_type, complex_c_type, n_density, n_shear, auto_only, *do,
                 index_c_type, acc_c_type),
            )
        if raw_kernel is None:
            return none

        nbins_total = int(bin_offsets.shape[0] - 1)

        def target(layout: _CombinationLayout, out: Any) -> Any:
            if layout.rows is None:
                return out
            return module.zeros((layout.canonical_ncomb, nbins_total), dtype=out.dtype)

        xig_num, xig_den = target(dd, out_xig_num), target(dd, out_xig_den)
        xit_num, xit_den = target(ds, out_xit_num), target(ds, out_xit_den)
        geom_args = (
            (geometry[0], bin_offsets, geometry[1]) if packed else (*geometry, bin_offsets)
        )
        raw_kernel(
            (max(1, nbins_total), 1, 1),
            (256,),
            (density_map, shear_map, density_weights, shear_weights, *geom_args,
             out_xipm_num, out_xipm_den, xig_num, xig_den, xit_num, xit_den,
             np.int64(nbins_total)),
        )
        if dd.rows is not None:
            _scatter_canonical_rows(module, dd, xig_num, out_xig_num)
            _scatter_canonical_rows(module, dd, xig_den, out_xig_den)
        if ds.rows is not None:
            _scatter_canonical_rows(module, ds, xit_num, out_xit_num)
            _scatter_canonical_rows(module, ds, xit_den, out_xit_den)
        return (True, True, True)

    _cupy_3x2pt_tomo_pairs_kernel.mode = "auto"
    _cupy_3x2pt_tomo_pairs_kernel.max_accumulators = _MAX_TILED_3X2PT_ACCUMULATORS
    return _cupy_3x2pt_tomo_pairs_kernel


def _build_cupy_xipm_cross_corr_kernel(module: Any) -> Any:
    """GPU kernel: per-pair ξ+/ξ- for cross-correlation of two shear catalogues.

    Rotates shears into pair frame and computes g_b'·conj(g_a') (ξ+)
    and g_b'·g_a' (ξ-) for both A→B and B→A orientations.
    """
    return module.ElementwiseKernel(
        "raw T g1a, raw T g2a, raw T g1b, raw T g2b, raw T wa, raw T wb,"
        " raw I ind_i, raw I ind_j, raw C exp_i, raw C exp_j",
        "A out_ab_p, A out_ab_m, A out_ba_p, A out_ba_m",
        """
        const I idx_i = ind_i[i];
        const I idx_j = ind_j[i];

        /* Rotation components promoted to the map precision T before the
           multiply (matches the CPU reference; exact for float -> double).
           Only the real parts of the estimators are needed downstream, so
           they are emitted directly at T. */
        const T er_i = (T)exp_i[i].real();
        const T ei_i = (T)exp_i[i].imag();
        const T er_j = (T)exp_j[i].real();
        const T ei_j = (T)exp_j[i].imag();

        const T a1_i = g1a[idx_i] * wa[idx_i];
        const T a2_i = g2a[idx_i] * wa[idx_i];
        const T b1_i = g1b[idx_i] * wb[idx_i];
        const T b2_i = g2b[idx_i] * wb[idx_i];
        const T a1_j = g1a[idx_j] * wa[idx_j];
        const T a2_j = g2a[idx_j] * wa[idx_j];
        const T b1_j = g1b[idx_j] * wb[idx_j];
        const T b2_j = g2b[idx_j] * wb[idx_j];

        const T ai_r = a1_i * er_i - a2_i * ei_i;
        const T ai_i = a1_i * ei_i + a2_i * er_i;
        const T bi_r = b1_i * er_i - b2_i * ei_i;
        const T bi_i = b1_i * ei_i + b2_i * er_i;
        const T aj_r = a1_j * er_j - a2_j * ei_j;
        const T aj_i = a1_j * ei_j + a2_j * er_j;
        const T bj_r = b1_j * er_j - b2_j * ei_j;
        const T bj_i = b1_j * ei_j + b2_j * er_j;

        out_ab_p = (A)(bj_r * ai_r + bj_i * ai_i);   /* Re[gb_j' conj(ga_i')] */
        out_ab_m = (A)(bj_r * ai_r - bj_i * ai_i);   /* Re[gb_j' ga_i']       */
        out_ba_p = (A)(aj_r * bi_r + aj_i * bi_i);   /* Re[ga_j' conj(gb_i')] */
        out_ba_m = (A)(aj_r * bi_r - aj_i * bi_i);   /* Re[ga_j' gb_i']       */
        """,
        "gpu_xipm_cross_corr_kernel",
        options=_CUPY_FASTMATH_OPTIONS,
    )


def _build_cupy_xipm_auto_corr_kernel(module: Any) -> Any:
    """GPU kernel: per-pair ξ+/ξ- auto-correlation (single shear catalogue)."""
    return module.ElementwiseKernel(
        "raw T g11, raw T g21, raw T g12, raw T g22, raw T w1, raw T w2,"
        " raw I ind_i, raw I ind_j, raw C exp_i, raw C exp_j",
        "A out_p, A out_m",
        """
        const I idx_i = ind_i[i];
        const I idx_j = ind_j[i];

        /* Rotation components promoted to the map precision T before the
           multiply (matches the CPU reference; exact for float -> double).
           Only the real parts of the estimators are needed downstream, so
           they are emitted directly at T. */
        const T er_i = (T)exp_i[i].real();
        const T ei_i = (T)exp_i[i].imag();
        const T er_j = (T)exp_j[i].real();
        const T ei_j = (T)exp_j[i].imag();

        const T a1 = g11[idx_i] * w1[idx_i];
        const T a2 = g21[idx_i] * w1[idx_i];
        const T b1 = g12[idx_j] * w2[idx_j];
        const T b2 = g22[idx_j] * w2[idx_j];

        const T a_r = a1 * er_i - a2 * ei_i;
        const T a_i = a1 * ei_i + a2 * er_i;
        const T b_r = b1 * er_j - b2 * ei_j;
        const T b_i = b1 * ei_j + b2 * er_j;

        out_p = (A)(b_r * a_r + b_i * a_i);   /* Re[g_b' conj(g_a')] */
        out_m = (A)(b_r * a_r - b_i * a_i);   /* Re[g_b' g_a']       */
        """,
        "gpu_xipm_auto_corr_kernel",
        options=_CUPY_FASTMATH_OPTIONS,
    )

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
        self.kernel_density_shear_tomo_vectorized = (
            kernel_density_shear_tomo_vectorized
        )
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
        if self.name == 'numpy':
            return np.asarray(array)
        elif self.name == 'cupy':
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
        if self.name == 'numpy':
            return np.asarray(array)
        elif self.name == 'cupy':
            if stream is not None:
                with stream:
                    return self.module.asnumpy(array)
            return self.module.asnumpy(array)
        return np.asarray(array)

    def get_memory_pool(self) -> Optional[Any]:
        if self.name == 'cupy':
            return self.module.get_default_memory_pool()
        return None

    # ── CUDA stream management ──────────────────────────────────────────

    def create_stream(self, non_blocking: bool = True) -> Optional[Any]:
        """Create a CUDA stream for overlapping transfers and kernel execution.

        Returns ``None`` on CPU backends.
        """
        if self.name != 'cupy':
            return None
        with self.module.cuda.Device(self.device_id):
            return self.module.cuda.Stream(non_blocking=non_blocking)

    def synchronize_stream(self, stream: Optional[Any] = None) -> None:
        """Wait for all operations on *stream* to complete.

        If *stream* is ``None``, synchronizes the current device.
        No-op on CPU backends.
        """
        if self.name != 'cupy':
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
        if self.name == 'cupy' and stream is not None:
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
        if self.name != 'cupy':
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
        if self.name != 'numpy':
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
        self.aperture_shear_kernel(inds, q_val, q_val, q_val, q_offsets, v1, v1, ones1, q_area, outb(1))
        self.kernel_density_density(v1, v1, ones1, ones1, inds, inds, offsets, outb(1), outb(1), outb(1), outb(1))
        self.kernel_density_shear(v1, v1, v1, ones1, ones1, inds, inds, rot, rot, offsets, outb(1), outb(1), outb(1), outb(1))
        self.xipm_auto_corr_kernel(v1, v1, v1, v1, ones1, ones1, inds, inds, rot, rot, offsets, outb(1), outb(1), outb(1))
        self.xipm_cross_corr_kernel(v1, v1, v1, v1, ones1, ones1, inds, inds, rot, rot, offsets, outb(1), outb(1), outb(1), outb(1), outb(1), outb(1))
        self.kernel_density_density_tomo_vectorized(m, m, inds, inds, offsets, comb, comb, outb(1, 1), outb(1, 1))
        self.kernel_density_shear_tomo_vectorized(m, shear, m, m, inds, inds, rot, rot, offsets, comb, comb, outb(1, 1), outb(1, 1))
        self.xipm_tomo_vectorized_kernel(shear, m, inds, inds, rot, rot, offsets, comb, comb, outb(1, 1), outb(1, 1), outb(1, 1))
        self.kernel_3x2pt_tomo_fused(
            m, shear, m, m,
            inds, inds, rot, rot, offsets,
            inds.astype(np.uint32), q_val.astype(map_dtype), q_val.astype(map_dtype),
            q_val.astype(map_dtype), q_offsets, q_area.astype(map_dtype),
            comb, comb, comb, comb, comb, comb,
            outb(nz, 1), outb(nz, 1), outb(nz, 1), outb(nz, 1),
            outb(1, 1), outb(1, 1), outb(1, 1),
            outb(1, 1), outb(1, 1), outb(1, 1), outb(1, 1),
        )


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
            device_type = 'gpu'
        except ImportError:
            warnings.warn("Cupy not installed, falling back to CPU (numpy).")
            device_id = None
            device_type = 'cpu'
        except Exception as exc:
            warnings.warn(
                f"Cupy is installed but no usable CUDA device was found "
                f"({type(exc).__name__}: {exc}); falling back to CPU (numpy). "
                f"Pass device='gpu' or a device id to make this an error."
            )
            device_id = None
            device_type = 'cpu'
    else:
        raise ValueError(f"Unknown device: {device}")

    if device_type == 'cpu':
        return Backend(
            'numpy',
            np,
            xipm_cross_corr_kernel=_cpu_xipm_cross_corr_kernel,
            xipm_auto_corr_kernel=_cpu_xipm_auto_corr_kernel,
            xipm_tomo_vectorized_kernel=_cpu_vectorized_tomo_kernel,
            aperture_density_kernel=_cpu_aperture_density_kernel,
            aperture_shear_kernel=_cpu_aperture_shear_kernel,
            kernel_density_density=_cpu_density_density_corr_kernel,
            kernel_density_shear=_cpu_density_shear_corr_kernel,
            kernel_density_density_tomo_vectorized=_cpu_density_density_tomo_vectorized_kernel,
            kernel_density_shear_tomo_vectorized=_cpu_density_shear_tomo_vectorized_kernel,
            kernel_3x2pt_tomo_fused=_cpu_3x2pt_tomo_fused_kernel,
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
                xipm_cross_corr_kernel=_build_cupy_xipm_cross_corr_kernel(cupy),
                xipm_auto_corr_kernel=_build_cupy_xipm_auto_corr_kernel(cupy),
                xipm_tomo_vectorized_kernel=_build_cupy_tomo_vectorized_kernel(cupy),
                xipm_tomo_packed_kernel=_build_cupy_tomo_packed_kernel(cupy),
                aperture_density_kernel=_build_cupy_aperture_density_kernel(cupy),
                aperture_shear_kernel=_build_cupy_aperture_shear_kernel(cupy),
                aperture_tomo_shear_kernel=_build_cupy_aperture_tomo_shear_kernel(cupy),
                aperture_tomo_density_kernel=_build_cupy_aperture_tomo_density_kernel(cupy),
                kernel_density_density=_build_cupy_density_density_corr_kernel(cupy),
                kernel_density_shear=_build_cupy_density_shear_corr_kernel(cupy),
                kernel_density_density_tomo_vectorized=_build_cupy_density_density_tomo_vectorized_kernel(cupy),
                kernel_density_shear_tomo_vectorized=_build_cupy_density_shear_tomo_vectorized_kernel(cupy),
                kernel_density_density_tomo_packed=_build_cupy_density_density_tomo_packed_kernel(cupy),
                kernel_density_shear_tomo_packed=_build_cupy_density_shear_tomo_packed_kernel(cupy),
                kernel_3x2pt_tomo_aperture=_build_cupy_3x2pt_tomo_aperture_kernel(cupy),
                kernel_3x2pt_tomo_pairs=_build_cupy_3x2pt_tomo_pairs_kernel(cupy),
                degrade_rows_kernel=_build_cupy_degrade_rows_kernel(cupy),
            )
        except ImportError:
            if device == 'auto':
                warnings.warn("Cupy not installed, falling back to CPU (numpy).")
                return Backend('numpy', np)
            else:
                raise ImportError("Cupy not installed but GPU requested.")
