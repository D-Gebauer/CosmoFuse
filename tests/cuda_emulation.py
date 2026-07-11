"""NumPy emulation of the CUDA RawKernel launch contracts.

Each emulator mirrors one ``__global__`` kernel in ``src/CosmoFuse/cuda/``:
same template parameters (parsed from the name expression), same launch
signature ``(grid, block, args)``, same flat index arithmetic, and the same
output layout.  ``EmulatedCupyModule`` duck-types the parts of cupy that the
``_build_cupy_*`` wrapper builders in ``CosmoFuse.backend`` touch, so the
*real* wrappers and the *real* GPU orchestrator branches in
``correlations.py`` can run end-to-end on a CPU-only machine and be compared
against the independently implemented (treecorr-validated) CPU Numba
kernels.

Faithfulness notes:
  - Block reductions are replaced by ``np.sum`` over the pairs of a bin;
    only the summation order differs (roundoff-level for float64).
  - ``gpu_fused_tomo_reduce_xipm`` performs its complex rotation at the
    rotation precision ``C`` (float32 math for cuFloatComplex); the
    emulator reproduces that by computing in ``complex64``.
  - All other kernels promote float32 rotation/filter values into the map
    type at use; NumPy's dtype promotion reproduces this exactly.
"""

import re

import numpy as np

_SCALAR_TYPES = {
    "float": np.float32,
    "double": np.float64,
    "cuFloatComplex": np.complex64,
    "cuDoubleComplex": np.complex128,
    "int": np.int32,
    "long long": np.int64,
}


def _parse_name_expression(name_expression):
    match = re.fullmatch(r"(\w+)<(.+)>", name_expression.strip())
    if match is None:
        raise ValueError(f"Unrecognised name expression: {name_expression!r}")
    name = match.group(1)
    params = [p.strip() for p in match.group(2).split(",")]
    return name, params


class _EmulatedRawKernel:
    """Callable with the RawKernel launch signature (grid, block, args)."""

    def __init__(self, name_expression):
        self.name, self.params = _parse_name_expression(name_expression)
        self._fn = _KERNEL_EMULATORS[self.name]

    def __call__(self, grid, block, args):
        self._fn(self.params, grid, args)


class _EmulatedStream:
    def __init__(self, non_blocking=False):
        self.non_blocking = non_blocking

    def record(self):
        return object()

    def wait_event(self, _event):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _EmulatedCuda:
    Stream = _EmulatedStream

    @staticmethod
    def get_current_stream():
        return _EmulatedStream()


class EmulatedCupyModule:
    """Duck-typed cupy module whose RawKernel returns numpy emulators."""

    float32 = np.float32
    float64 = np.float64
    int32 = np.int32
    int64 = np.int64
    complex64 = np.complex64
    complex128 = np.complex128
    cuda = _EmulatedCuda
    ascontiguousarray = staticmethod(np.ascontiguousarray)
    asarray = staticmethod(np.asarray)

    @staticmethod
    def RawKernel(_source, name_expression, options=None):
        return _EmulatedRawKernel(name_expression)


# ── Kernel emulators ──────────────────────────────────────────────────────


def _emulate_xipm(params, grid, args):
    """tomo_vectorized_xipm.cu :: gpu_fused_tomo_reduce_xipm<T, C, TOMO, I>."""
    map_dtype = _SCALAR_TYPES[params[0]]
    tomo_bins = int(params[2])

    (shear, weights, ind_i, ind_j, rot_i, rot_j, bin_offsets,
     comb_i, comb_j, out_num, out_den, ncomb, nbins_total, _npairs) = args
    ncomb = int(ncomb)
    nbins_total = int(nbins_total)
    shear_flat = np.asarray(shear).reshape(-1)
    weights_flat = np.asarray(weights).reshape(-1)
    num_flat = out_num.reshape(-1)
    den_flat = out_den.reshape(-1)

    gx, gy = int(grid[0]), int(grid[1])
    for bin_flat in range(gx):
        if bin_flat >= nbins_total:
            continue
        start = int(bin_offsets[bin_flat])
        stop = int(bin_offsets[bin_flat + 1])
        idx_a = ind_i[start:stop].astype(np.int64)
        idx_b = ind_j[start:stop].astype(np.int64)
        exp_a = rot_i[start:stop]
        exp_b = rot_j[start:stop]
        for comb_ori in range(gy):
            if comb_ori >= 2 * ncomb:
                continue
            comb_idx = comb_ori >> 1
            i = int(comb_i[comb_idx])
            j = int(comb_j[comb_idx])
            use_ba = (comb_ori & 1) == 1
            if use_ba and i == j:
                continue  # auto-combination rows stay zero (never written)
            ai, bj = (j, i) if use_ba else (i, j)

            idx_a_bin = idx_a * tomo_bins + ai
            idx_b_bin = idx_b * tomo_bins + bj
            # The rotation components are promoted to the map precision
            # before the multiply (matches the CPU reference and the fused
            # 3x2pt kernel; exact for float -> double).
            ga1 = shear_flat[idx_a_bin * 2]
            ga2 = shear_flat[idx_a_bin * 2 + 1]
            gb1 = shear_flat[idx_b_bin * 2]
            gb2 = shear_flat[idx_b_bin * 2 + 1]
            ea_r = exp_a.real.astype(map_dtype)
            ea_i = exp_a.imag.astype(map_dtype)
            eb_r = exp_b.real.astype(map_dtype)
            eb_i = exp_b.imag.astype(map_dtype)
            a_r = ga1 * ea_r - ga2 * ea_i
            a_i = ga1 * ea_i + ga2 * ea_r
            b_r = gb1 * eb_r - gb2 * eb_i
            b_i = gb1 * eb_i + gb2 * eb_r

            w_pair = weights_flat[idx_a_bin] * weights_flat[idx_b_bin]
            out_p_idx = comb_ori * nbins_total + bin_flat
            out_m_idx = (2 * ncomb + comb_ori) * nbins_total + bin_flat
            num_flat[out_p_idx] = np.sum(w_pair * (b_r * a_r + b_i * a_i))
            num_flat[out_m_idx] = np.sum(w_pair * (b_r * a_r - b_i * a_i))
            den_flat[out_p_idx] = np.sum(w_pair)


def _emulate_dd(params, grid, args):
    """density_density_tomo_vectorized.cu :: gpu_fused_tomo_reduce_dd<T, TOMO, I>."""
    tomo_bins = int(params[1])

    (density, weights, ind_i, ind_j, bin_offsets, comb_i, comb_j,
     out_num, out_den, ncomb, nbins_total, _npairs) = args
    ncomb = int(ncomb)
    nbins_total = int(nbins_total)
    density_flat = np.asarray(density).reshape(-1)
    weights_flat = np.asarray(weights).reshape(-1)
    num_flat = out_num.reshape(-1)
    den_flat = out_den.reshape(-1)

    gx, gy = int(grid[0]), int(grid[1])
    for bin_flat in range(gx):
        if bin_flat >= nbins_total:
            continue
        start = int(bin_offsets[bin_flat])
        stop = int(bin_offsets[bin_flat + 1])
        idx_a = ind_i[start:stop].astype(np.int64)
        idx_b = ind_j[start:stop].astype(np.int64)
        for comb_ori in range(gy):
            if comb_ori >= 2 * ncomb:
                continue
            comb_idx = comb_ori >> 1
            i = int(comb_i[comb_idx])
            j = int(comb_j[comb_idx])
            use_ba = (comb_ori & 1) == 1
            if use_ba and i == j:
                continue
            ai, bj = (j, i) if use_ba else (i, j)

            base_a = idx_a * tomo_bins + ai
            base_b = idx_b * tomo_bins + bj
            w_pair = weights_flat[base_a] * weights_flat[base_b]
            out_idx = comb_ori * nbins_total + bin_flat
            num_flat[out_idx] = np.sum(w_pair * density_flat[base_a] * density_flat[base_b])
            den_flat[out_idx] = np.sum(w_pair)


def _emulate_ds(params, grid, args):
    """density_shear_tomo_vectorized.cu :: gpu_fused_tomo_reduce_ds<T, C, L, S, I>."""
    lens_bins = int(params[2])
    source_bins = int(params[3])

    (density, shear, lens_w, source_w, ind_i, ind_j, rot_i, rot_j,
     bin_offsets, comb_i, comb_j, out_num, out_den,
     ncomb, nbins_total, _npairs) = args
    ncomb = int(ncomb)
    nbins_total = int(nbins_total)
    density_flat = np.asarray(density).reshape(-1)
    shear_flat = np.asarray(shear).reshape(-1)
    lens_w_flat = np.asarray(lens_w).reshape(-1)
    source_w_flat = np.asarray(source_w).reshape(-1)
    num_flat = out_num.reshape(-1)
    den_flat = out_den.reshape(-1)

    gx, gy = int(grid[0]), int(grid[1])
    for bin_flat in range(gx):
        if bin_flat >= nbins_total:
            continue
        start = int(bin_offsets[bin_flat])
        stop = int(bin_offsets[bin_flat + 1])
        idx_a = ind_i[start:stop].astype(np.int64)
        idx_b = ind_j[start:stop].astype(np.int64)
        rot_ab = rot_j[start:stop]
        rot_ba = rot_i[start:stop]
        for comb_idx in range(gy):
            if comb_idx >= ncomb:
                continue
            lens_bin = int(comb_i[comb_idx])
            source_bin = int(comb_j[comb_idx])

            # A->B: pixel a lens, pixel b source
            lens_ab = idx_a * lens_bins + lens_bin
            src_ab = idx_b * source_bins + source_bin
            gt_ab = (
                -shear_flat[src_ab * 2] * rot_ab.real
                + shear_flat[src_ab * 2 + 1] * rot_ab.imag
            )
            w_ab = lens_w_flat[lens_ab] * source_w_flat[src_ab]

            # B->A: pixel b lens, pixel a source
            lens_ba = idx_b * lens_bins + lens_bin
            src_ba = idx_a * source_bins + source_bin
            gt_ba = (
                -shear_flat[src_ba * 2] * rot_ba.real
                + shear_flat[src_ba * 2 + 1] * rot_ba.imag
            )
            w_ba = lens_w_flat[lens_ba] * source_w_flat[src_ba]

            out_idx = comb_idx * nbins_total + bin_flat
            num_flat[out_idx] = np.sum(
                w_ab * density_flat[lens_ab] * gt_ab
            ) + np.sum(w_ba * density_flat[lens_ba] * gt_ba)
            den_flat[out_idx] = np.sum(w_ab) + np.sum(w_ba)


def _check_planar_stride(arr, stride):
    actual = arr.strides[0] // arr.itemsize
    if arr.shape[0] > 1 and int(stride) != actual:
        raise AssertionError(
            f"stride contract violated: passed {int(stride)}, view has {actual}"
        )


def _emulate_aperture_shear_tomo(params, grid, args):
    """aperture_tomo.cu :: gpu_aperture_shear_tomo<T, QT>."""
    (g1, g2, g_stride, weights, w_stride, q_inds, q_cos, q_sin, q_val,
     q_offsets, q_patch_area, out_num, out_den, npatches, ntomo) = args
    _check_planar_stride(g1, g_stride)
    _check_planar_stride(weights, w_stride)
    npatches = int(npatches)
    ntomo = int(ntomo)

    gx, gy = int(grid[0]), int(grid[1])
    for patch in range(gx):
        if patch >= npatches:
            continue
        start = int(q_offsets[patch])
        stop = int(q_offsets[patch + 1])
        pix = q_inds[start:stop].astype(np.int64)
        qc = q_cos[start:stop]
        qs = q_sin[start:stop]
        qv = q_val[start:stop]
        for bin_idx in range(gy):
            if bin_idx >= ntomo:
                continue
            wv = weights[bin_idx, :][pix]
            gt = -g1[bin_idx, :][pix] * qc - g2[bin_idx, :][pix] * qs
            out_num[bin_idx, patch] = q_patch_area[patch] * np.sum(wv * gt * qv)
            out_den[bin_idx, patch] = np.sum(wv)


def _emulate_aperture_density_tomo(params, grid, args):
    """aperture_tomo.cu :: gpu_aperture_density_tomo<T, QT>."""
    (values, v_stride, weights, w_stride, q_inds, q_val, q_offsets,
     q_patch_area, out_num, out_den, npatches, ntomo) = args
    _check_planar_stride(values, v_stride)
    _check_planar_stride(weights, w_stride)
    npatches = int(npatches)
    ntomo = int(ntomo)

    gx, gy = int(grid[0]), int(grid[1])
    for patch in range(gx):
        if patch >= npatches:
            continue
        start = int(q_offsets[patch])
        stop = int(q_offsets[patch + 1])
        pix = q_inds[start:stop].astype(np.int64)
        qv = q_val[start:stop]
        for bin_idx in range(gy):
            if bin_idx >= ntomo:
                continue
            wv = weights[bin_idx, :][pix]
            out_num[bin_idx, patch] = q_patch_area[patch] * np.sum(
                wv * values[bin_idx, :][pix] * qv
            )
            out_den[bin_idx, patch] = np.sum(wv)


def _emulate_fused_3x2pt(params, grid, args):
    """tomo_fused_3x2pt.cu :: gpu_3x2pt_tomo_fused<T, C, I, QT, ND, NS>.

    One call per section (the trailing launch argument selects it), like
    the per-section launches of the cupy wrapper.
    """
    n_density = int(params[4])
    n_shear = int(params[5])

    (density, shear, density_w, shear_w, ind_i, ind_j, rot_i, rot_j,
     pair_offsets, nbins_total, npatches, _npix,
     q_inds, q_cos, q_sin, q_val, q_offsets, q_patch_area,
     ss_comb_i, ss_comb_j, n_ss_comb, dd_comb_i, dd_comb_j, n_dd_comb,
     ds_comb_i, ds_comb_j, n_ds_comb,
     out_ma_num, out_ma_den, out_mg_num, out_mg_den,
     out_xip_num, out_xim_num, out_xipm_den,
     out_xig_num, out_xig_den, out_xit_num, out_xit_den, section) = args

    nbins_total = int(nbins_total)
    npatches = int(npatches)
    n_ss_comb = int(n_ss_comb)
    n_dd_comb = int(n_dd_comb)
    n_ds_comb = int(n_ds_comb)
    section = int(section)
    gx, gy = int(grid[0]), int(grid[1])

    density_flat = np.asarray(density).reshape(-1)
    shear_flat = np.asarray(shear).reshape(-1)
    density_w_flat = np.asarray(density_w).reshape(-1)
    shear_w_flat = np.asarray(shear_w).reshape(-1)

    if section == 0:  # aperture mass M_ap
        ma_num = out_ma_num.reshape(-1)
        ma_den = out_ma_den.reshape(-1)
        for x in range(gx):
            if x >= npatches:
                continue
            start, stop = int(q_offsets[x]), int(q_offsets[x + 1])
            pix = q_inds[start:stop].astype(np.int64)
            qc, qs, qv = q_cos[start:stop], q_sin[start:stop], q_val[start:stop]
            for y in range(gy):
                if y >= n_shear:
                    continue
                shear_idx = (pix * n_shear + y) * 2
                wv = shear_w_flat[pix * n_shear + y]
                gt = -shear_flat[shear_idx] * qc - shear_flat[shear_idx + 1] * qs
                ma_num[y * npatches + x] = q_patch_area[x] * np.sum(wv * gt * qv)
                ma_den[y * npatches + x] = np.sum(wv)
        return

    if section == 1:  # galaxy mean density M_g
        mg_num = out_mg_num.reshape(-1)
        mg_den = out_mg_den.reshape(-1)
        for x in range(gx):
            if x >= npatches:
                continue
            start, stop = int(q_offsets[x]), int(q_offsets[x + 1])
            pix = q_inds[start:stop].astype(np.int64)
            qv = q_val[start:stop]
            for y in range(gy):
                if y >= n_density:
                    continue
                d_idx = pix * n_density + y
                wv = density_w_flat[d_idx]
                mg_num[y * npatches + x] = q_patch_area[x] * np.sum(
                    wv * density_flat[d_idx] * qv
                )
                mg_den[y * npatches + x] = np.sum(wv)
        return

    if section == 2:  # cosmic shear xi+/xi-
        xip_num = out_xip_num.reshape(-1)
        xim_num = out_xim_num.reshape(-1)
        xipm_den = out_xipm_den.reshape(-1)
        for x in range(gx):
            if x >= nbins_total:
                continue
            start, stop = int(pair_offsets[x]), int(pair_offsets[x + 1])
            pix_a = ind_i[start:stop].astype(np.int64)
            pix_b = ind_j[start:stop].astype(np.int64)
            ex_a = rot_i[start:stop]
            ex_b = rot_j[start:stop]
            for y in range(gy):
                if y >= 2 * n_ss_comb:
                    continue
                comb_idx = y >> 1
                ori = y & 1
                i = int(ss_comb_i[comb_idx])
                j = int(ss_comb_j[comb_idx])
                if ori == 1 and i == j:
                    continue
                ai, bj = (j, i) if ori == 1 else (i, j)

                a_base = (pix_a * n_shear + ai) * 2
                b_base = (pix_b * n_shear + bj) * 2
                ga1, ga2 = shear_flat[a_base], shear_flat[a_base + 1]
                gb1, gb2 = shear_flat[b_base], shear_flat[b_base + 1]
                # Rotation expanded at map precision (real/imag parts of C
                # promote exactly).
                a_r = ga1 * ex_a.real - ga2 * ex_a.imag
                a_i = ga1 * ex_a.imag + ga2 * ex_a.real
                b_r = gb1 * ex_b.real - gb2 * ex_b.imag
                b_i = gb1 * ex_b.imag + gb2 * ex_b.real

                wv = shear_w_flat[pix_a * n_shear + ai] * shear_w_flat[pix_b * n_shear + bj]
                out_idx = y * nbins_total + x
                xip_num[out_idx] = np.sum(wv * (b_r * a_r + b_i * a_i))
                xim_num[out_idx] = np.sum(wv * (b_r * a_r - b_i * a_i))
                xipm_den[out_idx] = np.sum(wv)
        return

    if section == 3:  # galaxy clustering xi_g
        xig_num = out_xig_num.reshape(-1)
        xig_den = out_xig_den.reshape(-1)
        for x in range(gx):
            if x >= nbins_total:
                continue
            start, stop = int(pair_offsets[x]), int(pair_offsets[x + 1])
            pix_a = ind_i[start:stop].astype(np.int64)
            pix_b = ind_j[start:stop].astype(np.int64)
            for y in range(gy):
                if y >= 2 * n_dd_comb:
                    continue
                comb_idx = y >> 1
                ori = y & 1
                i = int(dd_comb_i[comb_idx])
                j = int(dd_comb_j[comb_idx])
                if ori == 1 and i == j:
                    continue
                ai, bj = (j, i) if ori == 1 else (i, j)

                ia = pix_a * n_density + ai
                jb = pix_b * n_density + bj
                wv = density_w_flat[ia] * density_w_flat[jb]
                out_idx = y * nbins_total + x
                xig_num[out_idx] = np.sum(wv * density_flat[ia] * density_flat[jb])
                xig_den[out_idx] = np.sum(wv)
        return

    if section == 4:  # galaxy-galaxy lensing xi_t
        xit_num = out_xit_num.reshape(-1)
        xit_den = out_xit_den.reshape(-1)
        for x in range(gx):
            if x >= nbins_total:
                continue
            start, stop = int(pair_offsets[x]), int(pair_offsets[x + 1])
            pix_a = ind_i[start:stop].astype(np.int64)
            pix_b = ind_j[start:stop].astype(np.int64)
            ex_ab = rot_j[start:stop]
            ex_ba = rot_i[start:stop]
            for y in range(gy):
                if y >= n_ds_comb:
                    continue
                lens_bin = int(ds_comb_i[y])
                source_bin = int(ds_comb_j[y])

                lens_ab = pix_a * n_density + lens_bin
                src_ab = pix_b * n_shear + source_bin
                gt_ab = (
                    -shear_flat[src_ab * 2] * ex_ab.real
                    + shear_flat[src_ab * 2 + 1] * ex_ab.imag
                )
                w_ab = density_w_flat[lens_ab] * shear_w_flat[src_ab]

                lens_ba = pix_b * n_density + lens_bin
                src_ba = pix_a * n_shear + source_bin
                gt_ba = (
                    -shear_flat[src_ba * 2] * ex_ba.real
                    + shear_flat[src_ba * 2 + 1] * ex_ba.imag
                )
                w_ba = density_w_flat[lens_ba] * shear_w_flat[src_ba]

                out_idx = y * nbins_total + x
                xit_num[out_idx] = np.sum(
                    w_ab * density_flat[lens_ab] * gt_ab
                ) + np.sum(w_ba * density_flat[lens_ba] * gt_ba)
                xit_den[out_idx] = np.sum(w_ab) + np.sum(w_ba)
        return

    raise ValueError(f"Unknown fused section: {section}")


_KERNEL_EMULATORS = {
    "gpu_fused_tomo_reduce_xipm": _emulate_xipm,
    "gpu_fused_tomo_reduce_dd": _emulate_dd,
    "gpu_fused_tomo_reduce_ds": _emulate_ds,
    "gpu_aperture_shear_tomo": _emulate_aperture_shear_tomo,
    "gpu_aperture_density_tomo": _emulate_aperture_density_tomo,
    "gpu_3x2pt_tomo_fused": _emulate_fused_3x2pt,
}
