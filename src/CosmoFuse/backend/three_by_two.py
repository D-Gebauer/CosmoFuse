"""The fused 3x2pt path: M_ap, M_g, ξ±, ξ_g and ξ_t of one map set.

CPU: one Numba kernel computing all six outputs.  GPU: the aperture
sections of ``cuda/tomo_fused_3x2pt.cu`` and the single-pass pair tile of
``cuda/tomo_tiled_3x2pt.cu`` (the standalone tiles cover what it cannot).
"""

from typing import Any, List, Tuple

import numpy as np

from .._numba import njit, prange
from .aperture import _use_fused_aperture
from .combinations import (
    _MAX_TILED_3X2PT_ACCUMULATORS,
    SinglePassWrapper,
    _combination_layout,
    _CombinationLayout,
    _scatter_canonical_rows,
)
from .raw import KernelFile, _has_raw_cuda_compiler, c_complex, c_index, c_real

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
    k_val: np.ndarray,
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
            gt = (
                -shear_map[pix_idx, tomo_idx, 0] * q_cos[q_idx]
                - shear_map[pix_idx, tomo_idx, 1] * q_sin[q_idx]
            )
            sum_w += weight
            sum_num += weight * gt * q_val[q_idx]

        out_ma_num[tomo_idx, patch_idx] = q_patch_area[patch_idx] * sum_num
        out_ma_den[tomo_idx, patch_idx] = sum_w

    # --- Scalar aperture M_g: δ_g convolved with its own filter U(θ) ---
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
            sum_num += weight * density_map[pix_idx, tomo_idx] * k_val[q_idx]

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
                acc_num[comb_idx] += (
                    w_ab * density_map[pix_a, i_bin] * density_map[pix_b, j_bin]
                )

                if i_bin != j_bin:
                    w_ba = density_weights[pix_a, j_bin] * density_weights[pix_b, i_bin]
                    acc_den[comb_idx] += w_ba
                    acc_num[comb_idx] += (
                        w_ba * density_map[pix_a, j_bin] * density_map[pix_b, i_bin]
                    )

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
                w_ab = (
                    density_weights[pix_a, lens_bin] * shear_weights[pix_b, source_bin]
                )
                acc_num[comb_idx] += w_ab * density_map[pix_a, lens_bin] * gt_ab
                acc_den[comb_idx] += w_ab

                gt_ba = (
                    -shear_map[pix_a, source_bin, 0] * exp_ba_re
                    + shear_map[pix_a, source_bin, 1] * exp_ba_im
                )
                w_ba = (
                    density_weights[pix_b, lens_bin] * shear_weights[pix_a, source_bin]
                )
                acc_num[comb_idx] += w_ba * density_map[pix_b, lens_bin] * gt_ba
                acc_den[comb_idx] += w_ba

        for comb_idx in range(n_ds_comb):
            out_xit_num[comb_idx, bin_flat] = acc_num[comb_idx]
            out_xit_den[comb_idx, bin_flat] = acc_den[comb_idx]


def _build_cupy_3x2pt_tomo_aperture_kernel(module: Any) -> Any:
    """Builder for the aperture sections (M_ap, M_g) of the fused 3x2pt path
    (``tomo_fused_3x2pt.cu``; AoS inputs, accumulation at ACC).  The pair
    statistics of ``three_by_two_pt`` run in the tiled pair kernels."""
    build = KernelFile(module, "tomo_fused_3x2pt.cu", "Fused 3x2pt aperture")
    # One non-blocking stream per section, created lazily on first launch
    # (per device: each Backend instance gets its own builder closure).
    section_streams: List[Any] = []

    def _cupy_3x2pt_tomo_aperture_kernel(
        density_map: Any,
        shear_map: Any,
        density_weights: Any,
        shear_weights: Any,
        q_inds: Any,
        q_cos: Any,
        q_sin: Any,
        q_val: Any,
        k_val: Any,
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
        map_c_type = c_real(density_map.dtype)
        q_c_type = c_real(q_cos.dtype)
        # Accumulator type follows the orchestrator-allocated output buffers.
        acc_c_type = c_real(out_ma_num.dtype)
        npatches = int(q_offsets.shape[0] - 1)
        # One decision for both sections: they share the kernel, and the
        # widest bin set is what the accumulator budget has to cover.
        fused = _use_fused_aperture(module, max(n_density_bins, n_shear_bins), npatches)
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
            npatches,
            q_inds,
            q_cos,
            q_sin,
            q_val,
            k_val,
            q_offsets,
            q_patch_area,
            out_ma_num,
            out_ma_den,
            out_mg_num,
            out_mg_den,
        )
        section_grids = (
            (npatches, 1 if fused else n_shear_bins),  # z=0  M_ap
            (npatches, 1 if fused else n_density_bins),  # z=1  M_g
        )
        if not section_streams:
            section_streams.extend(
                module.cuda.Stream(non_blocking=True) for _ in range(len(section_grids))
            )
        # The sections write disjoint outputs -> run them concurrently.
        current = module.cuda.get_current_stream()
        ready = current.record()  # inputs staged on current stream
        for z, (gx, gy) in enumerate(section_grids):
            if gx <= 0 or gy <= 0:
                continue
            stream = section_streams[z]
            stream.wait_event(ready)
            with stream:
                raw_kernel((int(gx), int(gy), 1), (256,), base_args + (z,))
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
    build = KernelFile(module, "tomo_tiled_3x2pt.cu", "Tiled 3x2pt")
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
        mode = wrapper.mode
        if mode == "off" or not _has_raw_cuda_compiler(module):
            return none
        n_density = int(density_map.shape[1])
        n_shear = int(shear_map.shape[1])
        ss = _combination_layout(ss_comb[0], ss_comb[1], "triangle", n_shear)
        dd = _combination_layout(dd_comb[0], dd_comb[1], "triangle", n_density)
        ds = _combination_layout(
            ds_comb[0], ds_comb[1], "cartesian", n_density, n_shear
        )
        if (
            ss is None
            or dd is None
            or ds is None
            or ss.mode != "triangle"
            or ss.rows is not None
        ):
            return none
        n_ss, n_dd, n_ds = (
            3 * ss.canonical_ncomb,
            2 * dd.canonical_ncomb,
            2 * ds.canonical_ncomb,
        )
        budget = wrapper.max_accumulators
        if mode != "single" and n_ss + n_dd + n_ds > budget:
            return none
        do = (1, 1, 1)

        packed = len(geometry) == 2
        map_c_type = c_real(density_map.dtype)
        acc_c_type = c_real(out_xipm_num.dtype)
        auto_only = 1 if dd.mode == "auto" else 0
        if packed:
            raw_kernel = build(
                "gpu_tiled_packed_reduce_3x2pt",
                (map_c_type, n_density, n_shear, auto_only, *do, acc_c_type),
            )
        else:
            rot = geometry[2]
            complex_c_type = c_complex(rot.dtype)
            index_c_type = c_index(geometry[0].dtype)
            raw_kernel = build(
                "gpu_tiled_tomo_reduce_3x2pt",
                (
                    map_c_type,
                    complex_c_type,
                    n_density,
                    n_shear,
                    auto_only,
                    *do,
                    index_c_type,
                    acc_c_type,
                ),
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
            (geometry[0], bin_offsets, geometry[1])
            if packed
            else (*geometry, bin_offsets)
        )
        raw_kernel(
            (max(1, nbins_total), 1, 1),
            (256,),
            (
                density_map,
                shear_map,
                density_weights,
                shear_weights,
                *geom_args,
                out_xipm_num,
                out_xipm_den,
                xig_num,
                xig_den,
                xit_num,
                xit_den,
                nbins_total,
            ),
        )
        if dd.rows is not None:
            _scatter_canonical_rows(module, dd, xig_num, out_xig_num)
            _scatter_canonical_rows(module, dd, xig_den, out_xig_den)
        if ds.rows is not None:
            _scatter_canonical_rows(module, ds, xit_num, out_xit_num)
            _scatter_canonical_rows(module, ds, xit_den, out_xit_den)
        return (True, True, True)

    wrapper = SinglePassWrapper(
        _cupy_3x2pt_tomo_pairs_kernel, _MAX_TILED_3X2PT_ACCUMULATORS
    )
    return wrapper
