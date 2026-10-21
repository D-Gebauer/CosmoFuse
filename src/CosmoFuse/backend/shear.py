"""Cosmic shear ξ± pair kernels.

CPU: Numba kernels (single tomographic pair and all combinations at once).
GPU: per-pair ``ElementwiseKernel`` contributions and the combination-tiled
/ packed / per-(bin, row) kernels of ``cuda/tomo_vectorized_xipm.cu``.
"""

from typing import Any

import numpy as np

from .._numba import njit, prange
from .combinations import (
    _MAX_TILED_ACCUMULATORS,
    _MAX_VECTOR_TOMO_BINS,
    TiledWrapper,
    _combination_layout,
)
from .raw import (
    _CUPY_FASTMATH_OPTIONS,
    KernelFile,
    _has_raw_cuda_compiler,
    c_complex,
    c_index,
    c_real,
)

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
    build = KernelFile(module, "tomo_vectorized_xipm.cu", "Vectorized tomography")

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

        complex_c_type = c_complex(rot_i.dtype)
        map_c_type = c_real(weights.dtype)
        index_c_type = c_index(ind_i.dtype)
        # The accumulator type follows the (orchestrator-allocated) output
        # buffers: float64 outputs on float32 maps select double accumulation.
        acc_c_type = c_real(out_num.dtype)
        ncomb = int(comb_i.shape[0])
        npairs = int(ind_i.shape[0])
        nbins_total = int(bin_offsets.shape[0] - 1)
        threads = 256

        layout = _combination_layout(comb_i, comb_j, "triangle", nzbins)
        if (
            wrapper.tiled
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
                    (
                        shear_map,
                        weights,
                        ind_i,
                        ind_j,
                        rot_i,
                        rot_j,
                        bin_offsets,
                        out_num,
                        out_den,
                        nbins_total,
                    ),
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
                ncomb,
                nbins_total,
                npairs,
            ),
        )
        out_num[...] = tmp_num[:, 0::2] + tmp_num[:, 1::2]
        out_den[...] = tmp_den[0::2] + tmp_den[1::2]
        return True

    wrapper = TiledWrapper(_cupy_tomo_vectorized_kernel)
    return wrapper


def _build_cupy_tomo_packed_kernel(module: Any) -> Any:
    """Builder for the combination-tiled ξ+/ξ- kernel on packed pairs
    (8 bytes per pair, see ``cuda/pair_tiles.cuh``); same output contract
    as the unpacked wrapper."""
    build = KernelFile(module, "tomo_vectorized_xipm.cu", "Packed ξ±")

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
        map_c_type = c_real(weights.dtype)
        acc_c_type = c_real(out_num.dtype)
        raw_kernel = build(
            "gpu_tiled_packed_reduce_xipm", (map_c_type, nzbins, acc_c_type)
        )
        if raw_kernel is None:
            return False
        nbins_total = int(bin_offsets.shape[0] - 1)
        raw_kernel(
            (max(1, nbins_total), 1, 1),
            (256,),
            (
                shear_map,
                weights,
                pairs,
                bin_offsets,
                row_base,
                out_num,
                out_den,
                nbins_total,
            ),
        )
        return True

    return _cupy_tomo_packed_kernel


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
