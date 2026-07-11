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
from typing import Any, Optional, Tuple, Union

import numpy as np
from numba import njit, prange

logger = logging.getLogger(__name__)

# --std is mandatory: NVRTC refuses to instantiate name_expressions
# (C++ templates) without an explicit standard. Keep in sync with
# scripts/compile_check_nvrtc.py.
_CUPY_FASTMATH_OPTIONS = ("--use_fast_math", "--std=c++14")
_MAX_VECTOR_TOMO_BINS = 64
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


def _prepare_cuda_source(filename: str) -> str:
    source = _load_cuda_source_file(filename)
    return source.replace("__COMMON_CUDA_SOURCE__", _COMMON_CUDA_SOURCE)


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
    zero = map_values[0] * 0.0

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
    zero = g1[0] * 0.0

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

def _aperture_tomo_prepare_planar(module: Any, arr: Any) -> Tuple[Any, int]:
    """Return (array, row stride in elements) for a 2D device view.

    The kernel requires a contiguous innermost dimension; strided views
    such as ``shear[:, 0]`` of a ``(nz, 2, npix)`` array satisfy this and
    are passed without a copy — only their row stride differs.
    """
    if arr.strides[-1] != arr.itemsize:
        arr = module.ascontiguousarray(arr)
    return arr, int(arr.strides[0] // arr.itemsize)


def _build_cupy_aperture_tomo_shear_kernel(module: Any) -> Any:
    """Builder for the GPU block-reduced tomographic aperture-mass kernel."""
    kernel_cache: dict[tuple[str, str], Any] = {}

    def _get_or_build_raw_kernel(map_c_type: str, q_c_type: str) -> Optional[Any]:
        key = (map_c_type, q_c_type)
        cached = kernel_cache.get(key, _KERNEL_CACHE_MISS)
        if cached is not _KERNEL_CACHE_MISS:
            # May be None: a previously failed compilation is cached negatively
            # so it is not retried (and re-logged) on every call.
            return cached

        name_expression = f"gpu_aperture_shear_tomo<{map_c_type}, {q_c_type}>"
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
        raw_kernel = _get_or_build_raw_kernel(map_c_type, q_c_type)
        if raw_kernel is None:
            return False

        g1, g1_stride = _aperture_tomo_prepare_planar(module, g1)
        g2, g2_stride = _aperture_tomo_prepare_planar(module, g2)
        if g1_stride != g2_stride:
            g1 = module.ascontiguousarray(g1)
            g2 = module.ascontiguousarray(g2)
            g1_stride = g2_stride = int(g1.shape[1])
        weights, w_stride = _aperture_tomo_prepare_planar(module, weights)

        npatches = int(q_offsets.shape[0] - 1)
        ntomo = int(g1.shape[0])
        threads = 256
        blocks = (max(1, npatches), max(1, ntomo), 1)
        raw_kernel(
            blocks,
            (threads,),
            (
                g1,
                g2,
                np.int64(g1_stride),
                weights,
                np.int64(w_stride),
                q_inds,
                q_cos,
                q_sin,
                q_val,
                q_offsets,
                q_patch_area,
                out_num,
                out_den,
                np.int32(npatches),
                np.int32(ntomo),
            ),
        )
        return True

    return _cupy_aperture_tomo_shear_kernel


def _build_cupy_aperture_tomo_density_kernel(module: Any) -> Any:
    """Builder for the GPU block-reduced tomographic aperture-density kernel."""
    kernel_cache: dict[tuple[str, str], Any] = {}

    def _get_or_build_raw_kernel(map_c_type: str, q_c_type: str) -> Optional[Any]:
        key = (map_c_type, q_c_type)
        cached = kernel_cache.get(key, _KERNEL_CACHE_MISS)
        if cached is not _KERNEL_CACHE_MISS:
            # May be None: a previously failed compilation is cached negatively
            # so it is not retried (and re-logged) on every call.
            return cached

        name_expression = f"gpu_aperture_density_tomo<{map_c_type}, {q_c_type}>"
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
        raw_kernel = _get_or_build_raw_kernel(map_c_type, q_c_type)
        if raw_kernel is None:
            return False

        values, v_stride = _aperture_tomo_prepare_planar(module, values)
        weights, w_stride = _aperture_tomo_prepare_planar(module, weights)

        npatches = int(q_offsets.shape[0] - 1)
        ntomo = int(values.shape[0])
        threads = 256
        blocks = (max(1, npatches), max(1, ntomo), 1)
        raw_kernel(
            blocks,
            (threads,),
            (
                values,
                np.int64(v_stride),
                weights,
                np.int64(w_stride),
                q_inds,
                q_val,
                q_offsets,
                q_patch_area,
                out_num,
                out_den,
                np.int32(npatches),
                np.int32(ntomo),
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

    For cross-bin pairs (i≠j), averages the A→B and B→A orientations
    to symmetrise the estimator.  The pair loop is outermost within each
    bin so the pair indices and map rows are loaded once and reused for
    every tomographic combination; the weight sums (denominators) are
    accumulated in the same pass so no separate reduction is needed.
    """
    n_bins = offsets.shape[0] - 1
    ncomb = comb_i.shape[0]
    half = 0.5

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
                    acc_num[comb_idx] += half * (ab + ba)
                    acc_den[comb_idx] += half * (w_ab + w_ba)

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
        "T out_w",
        """
        const I i_idx = ind_i[i];
        const I j_idx = ind_j[i];
        out_w = w_a[i_idx] * w_b[j_idx] * density_a[i_idx] * density_b[j_idx];
        """,
        "gpu_density_density_corr_kernel",
        options=_CUPY_FASTMATH_OPTIONS,
    )


def _build_cupy_density_shear_corr_kernel(module: Any) -> Any:
    """GPU kernel: per-pair ξ_t contribution (w_lens·w_source·δ_lens·γ_t)."""
    return module.ElementwiseKernel(
        "raw T density_lens, raw T g1_source, raw T g2_source,"
        " raw T w_lens, raw T w_source, raw I ind_i, raw I ind_j, raw C exp_j",
        "T out_gt",
        """
        const I i_idx = ind_i[i];
        const I j_idx = ind_j[i];
        const C rot = exp_j[i];
        const T gamma_t = -g1_source[j_idx] * real(rot) + g2_source[j_idx] * imag(rot);
        out_gt = w_lens[i_idx] * w_source[j_idx] * density_lens[i_idx] * gamma_t;
        """,
        "gpu_density_shear_corr_kernel",
        options=_CUPY_FASTMATH_OPTIONS,
    )


# ── CuPy RawKernel builders (GPU fused kernels from .cu files) ────────────
# These builders compile the CUDA source templates with specific type
# parameters and cache the compiled kernels for reuse.

def _build_cupy_density_density_tomo_vectorized_kernel(module: Any) -> Any:
    """Builder for GPU tomographic galaxy clustering ξ_g kernel."""
    kernel_cache: dict[tuple[str, int, str], Any] = {}

    def _get_or_build_raw_kernel(map_c_type: str, nzbins: int, index_c_type: str) -> Optional[Any]:
        key = (map_c_type, nzbins, index_c_type)
        cached = kernel_cache.get(key, _KERNEL_CACHE_MISS)
        if cached is not _KERNEL_CACHE_MISS:
            # May be None: a previously failed compilation is cached negatively
            # so it is not retried (and re-logged) on every call.
            return cached

        name_expression = f"gpu_fused_tomo_reduce_dd<{map_c_type}, {nzbins}, {index_c_type}>"
        source = _prepare_cuda_source("density_density_tomo_vectorized.cu")

        try:
            kernel = _compile_raw_cuda_kernel(module, source, name_expression)
        except Exception as exc:
            logger.warning(
                "Vectorized density-density RawKernel compilation failed for %d bins; using legacy path: %s",
                nzbins,
                exc,
            )
            kernel_cache[key] = None
            return None

        kernel_cache[key] = kernel
        return kernel

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
        raw_kernel = _get_or_build_raw_kernel(map_c_type, nzbins, index_c_type)
        if raw_kernel is None:
            return False

        npairs = int(ind_i.shape[0])
        nbins_total = int(bin_offsets.shape[0] - 1)
        ncomb = int(comb_i.shape[0])
        threads = 256
        blocks = (max(1, nbins_total), max(1, 2 * ncomb), 1)
        raw_kernel(
            blocks,
            (threads,),
            (
                density_map,
                weights,
                ind_i,
                ind_j,
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

    return _cupy_density_density_tomo_vectorized_kernel


def _build_cupy_density_shear_tomo_vectorized_kernel(module: Any) -> Any:
    """Builder for GPU tomographic galaxy-galaxy lensing ξ_t kernel."""
    kernel_cache: dict[tuple[str, str, int, int, str], Any] = {}

    def _get_or_build_raw_kernel(
        map_c_type: str,
        complex_c_type: str,
        suffix: str,
        nlens_bins: int,
        nsource_bins: int,
        index_c_type: str,
    ) -> Optional[Any]:
        key = (map_c_type, suffix, nlens_bins, nsource_bins, index_c_type)
        cached = kernel_cache.get(key, _KERNEL_CACHE_MISS)
        if cached is not _KERNEL_CACHE_MISS:
            # May be None: a previously failed compilation is cached negatively
            # so it is not retried (and re-logged) on every call.
            return cached

        name_expression = (
            f"gpu_fused_tomo_reduce_ds<{map_c_type}, {complex_c_type}, "
            f"{nlens_bins}, {nsource_bins}, {index_c_type}>"
        )
        source = _prepare_cuda_source("density_shear_tomo_vectorized.cu")

        try:
            kernel = _compile_raw_cuda_kernel(module, source, name_expression)
        except Exception as exc:
            logger.warning(
                "Vectorized density-shear RawKernel compilation failed for lens/source bins (%d, %d); using legacy path: %s",
                nlens_bins,
                nsource_bins,
                exc,
            )
            kernel_cache[key] = None
            return None

        kernel_cache[key] = kernel
        return kernel

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

        if rot_j.dtype == module.complex64:
            complex_c_type = "cuFloatComplex"
            suffix = "c64"
        else:
            complex_c_type = "cuDoubleComplex"
            suffix = "c128"

        map_c_type = "float" if lens_weights.dtype == module.float32 else "double"
        index_c_type = "int" if ind_i.dtype == module.int32 else "long long"
        raw_kernel = _get_or_build_raw_kernel(
            map_c_type,
            complex_c_type,
            suffix,
            nlens_bins,
            nsource_bins,
            index_c_type,
        )
        if raw_kernel is None:
            return False

        npairs = int(ind_i.shape[0])
        nbins_total = int(bin_offsets.shape[0] - 1)
        ncomb = int(comb_i.shape[0])
        threads = 256
        blocks = (max(1, nbins_total), max(1, ncomb), 1)
        raw_kernel(
            blocks,
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

    return _cupy_density_shear_tomo_vectorized_kernel


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
    zero = wa[0] * 0.0
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
    zero = w1[0] * 0.0
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
    Cross-bin pairs (i≠j) contribute in both A→B and B→A orientations.

    The pair loop is outermost within each bin so the pair indices and
    rotation factors are loaded once per pair and reused for every
    tomographic combination.  Only the real parts of the estimators are
    accumulated, and the weight sums (denominators) are accumulated in
    the same pass.  Output rows: 2k = A→B, 2k+1 = B→A (zero for
    auto-combinations, which have no distinct B→A orientation).
    """
    n_bins = offsets.shape[0] - 1
    ncomb = comb_i.shape[0]

    for b in prange(n_bins):
        start = offsets[b]
        stop = offsets[b + 1]
        acc_ab_p = np.zeros(ncomb, dtype=out_p.dtype)
        acc_ab_m = np.zeros(ncomb, dtype=out_m.dtype)
        acc_ab_w = np.zeros(ncomb, dtype=out_w.dtype)
        acc_ba_p = np.zeros(ncomb, dtype=out_p.dtype)
        acc_ba_m = np.zeros(ncomb, dtype=out_m.dtype)
        acc_ba_w = np.zeros(ncomb, dtype=out_w.dtype)

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
                acc_ab_p[comb_idx] += w_ij * (b_r * a_r + b_i * a_i)
                acc_ab_m[comb_idx] += w_ij * (b_r * a_r - b_i * a_i)
                acc_ab_w[comb_idx] += w_ij

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
                    acc_ba_p[comb_idx] += w_ji * (c_r * d_r + c_i * d_i)
                    acc_ba_m[comb_idx] += w_ji * (c_r * d_r - c_i * d_i)
                    acc_ba_w[comb_idx] += w_ji

        for comb_idx in range(ncomb):
            out_row_ab = 2 * comb_idx
            out_row_ba = out_row_ab + 1
            out_p[out_row_ab, b] = acc_ab_p[comb_idx]
            out_m[out_row_ab, b] = acc_ab_m[comb_idx]
            out_w[out_row_ab, b] = acc_ab_w[comb_idx]
            out_p[out_row_ba, b] = acc_ba_p[comb_idx]
            out_m[out_row_ba, b] = acc_ba_m[comb_idx]
            out_w[out_row_ba, b] = acc_ba_w[comb_idx]


def _build_cupy_tomo_vectorized_kernel(module: Any) -> Any:
    """Builder for GPU tomographic cosmic shear ξ+/ξ- kernel."""
    kernel_cache: dict[tuple[str, str, int, str], Any] = {}

    def _get_or_build_raw_kernel(
        map_c_type: str,
        complex_c_type: str,
        suffix: str,
        nzbins: int,
        index_c_type: str,
    ) -> Optional[Any]:
        key = (map_c_type, suffix, nzbins, index_c_type)
        cached = kernel_cache.get(key, _KERNEL_CACHE_MISS)
        if cached is not _KERNEL_CACHE_MISS:
            # May be None: a previously failed compilation is cached negatively
            # so it is not retried (and re-logged) on every call.
            return cached

        name_expression = (
            f"gpu_fused_tomo_reduce_xipm<{map_c_type}, {complex_c_type}, {nzbins}, {index_c_type}>"
        )
        source = _prepare_cuda_source("tomo_vectorized_xipm.cu")

        try:
            kernel = _compile_raw_cuda_kernel(module, source, name_expression)
        except Exception as exc:
            logger.warning(
                "Vectorized tomography RawKernel compilation failed for %d bins; using legacy path: %s",
                nzbins,
                exc,
            )
            kernel_cache[key] = None
            return None
        kernel_cache[key] = kernel
        return kernel

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

        if rot_i.dtype == module.complex64:
            complex_c_type = "cuFloatComplex"
            suffix = "c64"
        else:
            complex_c_type = "cuDoubleComplex"
            suffix = "c128"

        map_c_type = "float" if weights.dtype == module.float32 else "double"
        index_c_type = "int" if ind_i.dtype == module.int32 else "long long"
        raw_kernel = _get_or_build_raw_kernel(
            map_c_type,
            complex_c_type,
            suffix,
            nzbins,
            index_c_type,
        )
        if raw_kernel is None:
            return False

        npairs = int(ind_i.shape[0])
        nbins_total = int(bin_offsets.shape[0] - 1)
        ncomb = int(comb_i.shape[0])
        threads = 256
        blocks = (max(1, nbins_total), max(1, 2 * ncomb), 1)
        raw_kernel(
            blocks,
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
                out_num,
                out_den,
                np.int32(ncomb),
                np.int64(nbins_total),
                np.int64(npairs),
            ),
        )
        return True

    return _cupy_tomo_vectorized_kernel


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
    # tomographic combination and orientation.
    n_ss_comb = ss_comb_i.shape[0]
    for bin_flat in prange(nbins_total):
        start = pair_offsets[bin_flat]
        stop = pair_offsets[bin_flat + 1]
        acc_p = np.zeros(2 * n_ss_comb, dtype=out_xip_num.dtype)
        acc_m = np.zeros(2 * n_ss_comb, dtype=out_xim_num.dtype)
        acc_w = np.zeros(2 * n_ss_comb, dtype=out_xipm_den.dtype)

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
                row_ab = 2 * comb_idx
                acc_w[row_ab] += w_pair
                acc_p[row_ab] += w_pair * (b_r * a_r + b_i * a_i)
                acc_m[row_ab] += w_pair * (b_r * a_r - b_i * a_i)

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
                    row_ba = row_ab + 1
                    acc_w[row_ba] += w_ba
                    acc_p[row_ba] += w_ba * (d_r * c_r + d_i * c_i)
                    acc_m[row_ba] += w_ba * (d_r * c_r - d_i * c_i)

        for row in range(2 * n_ss_comb):
            out_xip_num[row, bin_flat] = acc_p[row]
            out_xim_num[row, bin_flat] = acc_m[row]
            out_xipm_den[row, bin_flat] = acc_w[row]

    # --- Galaxy clustering ξ_g ---
    n_dd_comb = dd_comb_i.shape[0]
    for bin_flat in prange(nbins_total):
        start = pair_offsets[bin_flat]
        stop = pair_offsets[bin_flat + 1]
        acc_num = np.zeros(2 * n_dd_comb, dtype=out_xig_num.dtype)
        acc_den = np.zeros(2 * n_dd_comb, dtype=out_xig_den.dtype)

        for pair_idx in range(start, stop):
            pix_a = ind_i[pair_idx]
            pix_b = ind_j[pair_idx]

            for comb_idx in range(n_dd_comb):
                i_bin = dd_comb_i[comb_idx]
                j_bin = dd_comb_j[comb_idx]

                w_ab = density_weights[pix_a, i_bin] * density_weights[pix_b, j_bin]
                row_ab = 2 * comb_idx
                acc_den[row_ab] += w_ab
                acc_num[row_ab] += w_ab * density_map[pix_a, i_bin] * density_map[pix_b, j_bin]

                if i_bin != j_bin:
                    w_ba = density_weights[pix_a, j_bin] * density_weights[pix_b, i_bin]
                    row_ba = row_ab + 1
                    acc_den[row_ba] += w_ba
                    acc_num[row_ba] += w_ba * density_map[pix_a, j_bin] * density_map[pix_b, i_bin]

        for row in range(2 * n_dd_comb):
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


def _build_cupy_3x2pt_tomo_fused_kernel(module: Any) -> Any:
    kernel_cache: dict[tuple[str, str, str, str, int, int], Any] = {}
    # One non-blocking stream per section, created lazily on first launch
    # (per device: each Backend instance gets its own builder closure).
    section_streams: list = []

    def _get_or_build_raw_kernel(
        map_c_type: str,
        complex_c_type: str,
        suffix: str,
        index_c_type: str,
        q_c_type: str,
        n_density_bins: int,
        n_shear_bins: int,
    ) -> Optional[Any]:
        key = (map_c_type, suffix, index_c_type, q_c_type, n_density_bins, n_shear_bins)
        cached = kernel_cache.get(key, _KERNEL_CACHE_MISS)
        if cached is not _KERNEL_CACHE_MISS:
            # May be None: a previously failed compilation is cached negatively
            # so it is not retried (and re-logged) on every call.
            return cached

        name_expression = (
            f"gpu_3x2pt_tomo_fused<{map_c_type}, {complex_c_type}, "
            f"{index_c_type}, {q_c_type}, {n_density_bins}, {n_shear_bins}>"
        )
        source = _prepare_cuda_source("tomo_fused_3x2pt.cu")

        try:
            kernel = _compile_raw_cuda_kernel(module, source, name_expression)
        except Exception as exc:
            logger.warning(
                "Fused 3x2pt RawKernel compilation failed; using unfused path: %s",
                exc,
            )
            kernel_cache[key] = None
            return None

        kernel_cache[key] = kernel
        return kernel

    def _cupy_3x2pt_tomo_fused_kernel(
        density_map: Any,
        shear_map: Any,
        density_weights: Any,
        shear_weights: Any,
        ind_i: Any,
        ind_j: Any,
        rot_i: Any,
        rot_j: Any,
        pair_offsets: Any,
        q_inds: Any,
        q_cos: Any,
        q_sin: Any,
        q_val: Any,
        q_offsets: Any,
        q_patch_area: Any,
        ss_comb_i: Any,
        ss_comb_j: Any,
        dd_comb_i: Any,
        dd_comb_j: Any,
        ds_comb_i: Any,
        ds_comb_j: Any,
        out_ma_num: Any,
        out_ma_den: Any,
        out_mg_num: Any,
        out_mg_den: Any,
        out_xip_num: Any,
        out_xim_num: Any,
        out_xipm_den: Any,
        out_xig_num: Any,
        out_xig_den: Any,
        out_xit_num: Any,
        out_xit_den: Any,
    ) -> bool:
        if not _has_raw_cuda_compiler(module):
            return False

        if rot_i.dtype == module.complex64:
            complex_c_type = "cuFloatComplex"
            suffix = "c64"
        else:
            complex_c_type = "cuDoubleComplex"
            suffix = "c128"

        n_density_bins = int(density_map.shape[1])
        n_shear_bins = int(shear_map.shape[1])
        map_c_type = "float" if density_map.dtype == module.float32 else "double"
        index_c_type = "int" if ind_i.dtype == module.int32 else "long long"
        q_c_type = "float" if q_cos.dtype == module.float32 else "double"
        raw_kernel = _get_or_build_raw_kernel(
            map_c_type,
            complex_c_type,
            suffix,
            index_c_type,
            q_c_type,
            n_density_bins,
            n_shear_bins,
        )
        if raw_kernel is None:
            return False

        nbins_total = int(pair_offsets.shape[0] - 1)
        npatches = int(q_offsets.shape[0] - 1)
        npix = int(density_map.shape[0])
        n_ss_comb = int(ss_comb_i.shape[0])
        n_dd_comb = int(dd_comb_i.shape[0])
        n_ds_comb = int(ds_comb_i.shape[0])

        # One launch per correlation section with an exactly-sized grid,
        # instead of one dense (max_x, max_y, 5) grid that is mostly no-op
        # blocks at realistic sizes.
        section_grids = (
            (npatches, n_shear_bins),        # z=0  M_ap
            (npatches, n_density_bins),      # z=1  M_g
            (nbins_total, 2 * n_ss_comb),    # z=2  xi+/-
            (nbins_total, 2 * n_dd_comb),    # z=3  xi_g
            (nbins_total, n_ds_comb),        # z=4  xi_t
        )
        threads = 256
        base_args = (
            density_map,
            shear_map,
            density_weights,
            shear_weights,
            ind_i,
            ind_j,
            rot_i,
            rot_j,
            pair_offsets,
            np.int64(nbins_total),
            np.int32(npatches),
            np.int32(npix),
            q_inds,
            q_cos,
            q_sin,
            q_val,
            q_offsets,
            q_patch_area,
            ss_comb_i,
            ss_comb_j,
            np.int32(n_ss_comb),
            dd_comb_i,
            dd_comb_j,
            np.int32(n_dd_comb),
            ds_comb_i,
            ds_comb_j,
            np.int32(n_ds_comb),
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

        if not section_streams:
            section_streams.extend(
                module.cuda.Stream(non_blocking=True) for _ in range(5)
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
                raw_kernel(
                    (int(gx), int(gy), 1),
                    (threads,),
                    base_args + (np.int32(z),),
                )
            current.wait_event(stream.record())  # downstream default-stream work waits
        return True

    return _cupy_3x2pt_tomo_fused_kernel


def _build_cupy_xipm_cross_corr_kernel(module: Any) -> Any:
    """GPU kernel: per-pair ξ+/ξ- for cross-correlation of two shear catalogues.

    Rotates shears into pair frame and computes g_b'·conj(g_a') (ξ+)
    and g_b'·g_a' (ξ-) for both A→B and B→A orientations.
    """
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
        "gpu_xipm_cross_corr_kernel",
        options=_CUPY_FASTMATH_OPTIONS,
    )


def _build_cupy_xipm_auto_corr_kernel(module: Any) -> Any:
    """GPU kernel: per-pair ξ+/ξ- auto-correlation (single shear catalogue)."""
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
        "gpu_xipm_auto_corr_kernel",
        options=_CUPY_FASTMATH_OPTIONS,
    )

class Backend:
    def __init__(
        self,
        name: str,
        module: Any,
        device_id: Optional[int] = None,
        xipm_cross_corr_kernel: Optional[Any] = None,
        xipm_auto_corr_kernel: Optional[Any] = None,
        xipm_tomo_vectorized_kernel: Optional[Any] = None,
        aperture_density_kernel: Optional[Any] = None,
        aperture_shear_kernel: Optional[Any] = None,
        aperture_tomo_shear_kernel: Optional[Any] = None,
        aperture_tomo_density_kernel: Optional[Any] = None,
        kernel_density_density: Optional[Any] = None,
        kernel_density_shear: Optional[Any] = None,
        kernel_density_density_tomo_vectorized: Optional[Any] = None,
        kernel_density_shear_tomo_vectorized: Optional[Any] = None,
        kernel_3x2pt_tomo_fused: Optional[Any] = None,
    ) -> None:
        self.name = name
        self.module = module
        self.device_id = device_id
        self.xipm_cross_corr_kernel = xipm_cross_corr_kernel
        self.xipm_auto_corr_kernel = xipm_auto_corr_kernel
        self.xipm_tomo_vectorized_kernel = xipm_tomo_vectorized_kernel
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
        self.kernel_3x2pt_tomo_fused = kernel_3x2pt_tomo_fused

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
        self.xipm_tomo_vectorized_kernel(shear, m, inds, inds, rot, rot, offsets, comb, comb, outb(2, 1), outb(2, 1), outb(2, 1))
        self.kernel_3x2pt_tomo_fused(
            m, shear, m, m,
            inds, inds, rot, rot, offsets,
            inds.astype(np.uint32), q_val.astype(map_dtype), q_val.astype(map_dtype),
            q_val.astype(map_dtype), q_offsets, q_area.astype(map_dtype),
            comb, comb, comb, comb, comb, comb,
            outb(nz, 1), outb(nz, 1), outb(nz, 1), outb(nz, 1),
            outb(2, 1), outb(2, 1), outb(2, 1),
            outb(2, 1), outb(2, 1), outb(1, 1), outb(1, 1),
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
                aperture_density_kernel=_build_cupy_aperture_density_kernel(cupy),
                aperture_shear_kernel=_build_cupy_aperture_shear_kernel(cupy),
                aperture_tomo_shear_kernel=_build_cupy_aperture_tomo_shear_kernel(cupy),
                aperture_tomo_density_kernel=_build_cupy_aperture_tomo_density_kernel(cupy),
                kernel_density_density=_build_cupy_density_density_corr_kernel(cupy),
                kernel_density_shear=_build_cupy_density_shear_corr_kernel(cupy),
                kernel_density_density_tomo_vectorized=_build_cupy_density_density_tomo_vectorized_kernel(cupy),
                kernel_density_shear_tomo_vectorized=_build_cupy_density_shear_tomo_vectorized_kernel(cupy),
                kernel_3x2pt_tomo_fused=_build_cupy_3x2pt_tomo_fused_kernel(cupy),
            )
        except ImportError:
            if device == 'auto':
                warnings.warn("Cupy not installed, falling back to CPU (numpy).")
                return Backend('numpy', np)
            else:
                raise ImportError("Cupy not installed but GPU requested.")
