"""Galaxy clustering ξ_g and galaxy-galaxy lensing ξ_t pair kernels.

CPU: Numba kernels (single tomographic pair and all combinations at once).
GPU: per-pair ``ElementwiseKernel`` contributions and the combination-tiled
/ packed / per-(bin, row) kernels of ``cuda/density_density_tomo_vectorized.cu``
and ``cuda/density_shear_tomo_vectorized.cu``.
"""

from typing import Any

import numpy as np
from numba import njit, prange

from .combinations import (
    _MAX_TILED_ACCUMULATORS,
    _MAX_VECTOR_TOMO_BINS,
    _combination_layout,
    _scatter_canonical_rows,
)
from .raw import (
    _CUPY_FASTMATH_OPTIONS,
    KernelFile,
    _has_raw_cuda_compiler,
    c_complex,
    c_index,
    c_real,
)

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
                w_ab = lens_weights[pix_i, lens_bin] * source_weights[pix_j, source_bin]
                acc_num[comb_idx] += w_ab * density_map[pix_i, lens_bin] * gamma_t_ij
                acc_den[comb_idx] += w_ab

                # B→A: pixel j = lens, pixel i = source
                gamma_t_ji = (
                    -shear_map[pix_i, source_bin, 0] * exp_i_re
                    + shear_map[pix_i, source_bin, 1] * exp_i_im
                )
                w_ba = lens_weights[pix_j, lens_bin] * source_weights[pix_i, source_bin]
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


def _build_cupy_density_density_tomo_vectorized_kernel(module: Any) -> Any:
    """Builder for the GPU tomographic galaxy clustering ξ_g kernels.

    Contract: ``out_num`` / ``out_den`` are ``(ncomb, nbins_total)`` and hold
    both orientations of a cross combination summed.  The combination-tiled
    kernel (one block per angular bin, every pair visited once) serves the
    upper-triangle and auto-only layouts and, via a canonical temporary,
    any subset of them; the per-(bin, row) kernel is the fallback beyond the
    tile's accumulator budget.
    """
    build = KernelFile(
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

        map_c_type = c_real(weights.dtype)
        index_c_type = c_index(ind_i.dtype)
        # Accumulator type follows the orchestrator-allocated output buffers.
        acc_c_type = c_real(out_num.dtype)
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
                    tgt_num = module.zeros(
                        (layout.canonical_ncomb, nbins_total), dtype=out_num.dtype
                    )
                    tgt_den = module.zeros(
                        (layout.canonical_ncomb, nbins_total), dtype=out_den.dtype
                    )
                raw_kernel(
                    (max(1, nbins_total), 1, 1),
                    (threads,),
                    (
                        density_map,
                        weights,
                        ind_i,
                        ind_j,
                        bin_offsets,
                        tgt_num,
                        tgt_den,
                        nbins_total,
                    ),
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
                ncomb,
                nbins_total,
                npairs,
            ),
        )
        out_num[...] = tmp_num[0::2] + tmp_num[1::2]
        out_den[...] = tmp_den[0::2] + tmp_den[1::2]
        return True

    _cupy_density_density_tomo_vectorized_kernel.tiled = True
    return _cupy_density_density_tomo_vectorized_kernel


def _build_cupy_density_density_tomo_packed_kernel(module: Any) -> Any:
    """Builder for the tiled ξ_g kernel on packed pairs (8 B per pair)."""
    build = KernelFile(
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
        map_c_type = c_real(weights.dtype)
        acc_c_type = c_real(out_num.dtype)
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
            tgt_num = module.zeros(
                (layout.canonical_ncomb, nbins_total), dtype=out_num.dtype
            )
            tgt_den = module.zeros(
                (layout.canonical_ncomb, nbins_total), dtype=out_den.dtype
            )
        raw_kernel(
            (max(1, nbins_total), 1, 1),
            (256,),
            (
                density_map,
                weights,
                pairs,
                bin_offsets,
                row_base,
                tgt_num,
                tgt_den,
                nbins_total,
            ),
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
    build = KernelFile(
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

        complex_c_type = c_complex(rot_j.dtype)
        map_c_type = c_real(lens_weights.dtype)
        index_c_type = c_index(ind_i.dtype)
        # Accumulator type follows the orchestrator-allocated output buffers.
        acc_c_type = c_real(out_num.dtype)
        npairs = int(ind_i.shape[0])
        nbins_total = int(bin_offsets.shape[0] - 1)
        ncomb = int(comb_i.shape[0])
        threads = 256

        layout = _combination_layout(
            comb_i, comb_j, "cartesian", nlens_bins, nsource_bins
        )
        if (
            _cupy_density_shear_tomo_vectorized_kernel.tiled
            and layout is not None
            and 2 * layout.canonical_ncomb <= _MAX_TILED_ACCUMULATORS
        ):
            raw_kernel = build(
                "gpu_tiled_tomo_reduce_ds",
                (
                    map_c_type,
                    complex_c_type,
                    nlens_bins,
                    nsource_bins,
                    index_c_type,
                    acc_c_type,
                ),
            )
            if raw_kernel is not None:
                if layout.rows is None:
                    tgt_num, tgt_den = out_num, out_den
                else:
                    tgt_num = module.zeros(
                        (layout.canonical_ncomb, nbins_total), dtype=out_num.dtype
                    )
                    tgt_den = module.zeros(
                        (layout.canonical_ncomb, nbins_total), dtype=out_den.dtype
                    )
                raw_kernel(
                    (max(1, nbins_total), 1, 1),
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
                        tgt_num,
                        tgt_den,
                        nbins_total,
                    ),
                )
                if layout.rows is not None:
                    _scatter_canonical_rows(module, layout, tgt_num, out_num)
                    _scatter_canonical_rows(module, layout, tgt_den, out_den)
                return True

        raw_kernel = build(
            "gpu_fused_tomo_reduce_ds",
            (
                map_c_type,
                complex_c_type,
                nlens_bins,
                nsource_bins,
                index_c_type,
                acc_c_type,
            ),
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
                ncomb,
                nbins_total,
                npairs,
            ),
        )
        return True

    _cupy_density_shear_tomo_vectorized_kernel.tiled = True
    return _cupy_density_shear_tomo_vectorized_kernel


def _build_cupy_density_shear_tomo_packed_kernel(module: Any) -> Any:
    """Builder for the tiled ξ_t kernel on packed pairs (8 B per pair)."""
    build = KernelFile(
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
        layout = _combination_layout(
            comb_i, comb_j, "cartesian", nlens_bins, nsource_bins
        )
        if layout is None or 2 * layout.canonical_ncomb > _MAX_TILED_ACCUMULATORS:
            return False
        map_c_type = c_real(lens_weights.dtype)
        acc_c_type = c_real(out_num.dtype)
        raw_kernel = build(
            "gpu_tiled_packed_reduce_ds",
            (map_c_type, nlens_bins, nsource_bins, acc_c_type),
        )
        if raw_kernel is None:
            return False
        nbins_total = int(bin_offsets.shape[0] - 1)
        if layout.rows is None:
            tgt_num, tgt_den = out_num, out_den
        else:
            tgt_num = module.zeros(
                (layout.canonical_ncomb, nbins_total), dtype=out_num.dtype
            )
            tgt_den = module.zeros(
                (layout.canonical_ncomb, nbins_total), dtype=out_den.dtype
            )
        raw_kernel(
            (max(1, nbins_total), 1, 1),
            (256,),
            (
                density_map,
                shear_map,
                lens_weights,
                source_weights,
                pairs,
                bin_offsets,
                row_base,
                tgt_num,
                tgt_den,
                nbins_total,
            ),
        )
        if layout.rows is not None:
            _scatter_canonical_rows(module, layout, tgt_num, out_num)
            _scatter_canonical_rows(module, layout, tgt_den, out_den)
        return True

    return _cupy_density_shear_tomo_packed_kernel
