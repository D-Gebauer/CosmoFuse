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
  - All kernels promote float32 rotation/filter values into the map type
    at use; NumPy's dtype promotion reproduces this exactly.
  - The packed loaders evaluate the rotation angles with ``sincospi``;
    the emulator uses ``np.cos(np.pi * x)`` (1e-16-level difference).
"""

import functools
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


# Names of the emulated kernels in launch order (tests clear and inspect it).
LAUNCH_LOG = []


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
        LAUNCH_LOG.append(self.name)
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
    zeros = staticmethod(np.zeros)
    empty = staticmethod(np.empty)

    @staticmethod
    def RawKernel(_source, name_expression, options=None):
        return _EmulatedRawKernel(name_expression)


# ── Kernel emulators ──────────────────────────────────────────────────────


def _emulate_xipm(params, grid, args):
    """tomo_vectorized_xipm.cu :: gpu_fused_tomo_reduce_xipm<T, C, TOMO, I, ACC>."""
    map_dtype = _SCALAR_TYPES[params[0]]
    tomo_bins = int(params[2])
    acc = _SCALAR_TYPES[params[4]]

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
            # Per-pair products at map precision, accumulated at ACC.
            num_flat[out_p_idx] = np.sum(w_pair * (b_r * a_r + b_i * a_i), dtype=acc)
            num_flat[out_m_idx] = np.sum(w_pair * (b_r * a_r - b_i * a_i), dtype=acc)
            den_flat[out_p_idx] = np.sum(w_pair, dtype=acc)


# ── Tile helpers (pair_tiles.cuh) ─────────────────────────────────────────
#
# One block per angular bin; every (combination) row holds BOTH orientations
# of a cross combination summed (ratio-of-sums estimator).


def _unpacked_geometry(map_dtype, ind_i, ind_j, rot_i, rot_j, start, stop):
    idx_a = ind_i[start:stop].astype(np.int64)
    idx_b = ind_j[start:stop].astype(np.int64)
    ea_r = rot_i[start:stop].real.astype(map_dtype)
    ea_i = rot_i[start:stop].imag.astype(map_dtype)
    eb_r = rot_j[start:stop].real.astype(map_dtype)
    eb_i = rot_j[start:stop].imag.astype(map_dtype)
    return idx_a, idx_b, ea_r, ea_i, eb_r, eb_i


def _packed_geometry(map_dtype, pairs, base, start, stop):
    """ushort4 {local_a, local_b, angle_a, angle_b}; alpha = pi * code / 32768
    (sincospi in the kernel; np.cos(np.pi * x) here, 1e-16-level difference)."""
    p = np.asarray(pairs[start:stop])
    assert p.dtype == np.uint16 and p.shape[1] == 4
    idx_a = base + p[:, 0].astype(np.int64)
    idx_b = base + p[:, 1].astype(np.int64)
    ha = p[:, 2].astype(map_dtype) * map_dtype(1.0 / 32768.0)
    hb = p[:, 3].astype(map_dtype) * map_dtype(1.0 / 32768.0)
    pi = map_dtype(np.pi)
    return (idx_a, idx_b,
            np.cos(pi * ha).astype(map_dtype), np.sin(pi * ha).astype(map_dtype),
            np.cos(pi * hb).astype(map_dtype), np.sin(pi * hb).astype(map_dtype))


def _tile_xipm(shear, weights, tomo_bins, acc, geom, out_num, out_den, nbins_total, bin_flat):
    idx_a, idx_b, ea_r, ea_i, eb_r, eb_i = geom
    ncomb = (tomo_bins * (tomo_bins + 1)) // 2
    shear3 = np.asarray(shear).reshape(-1, tomo_bins, 2)
    weights2 = np.asarray(weights).reshape(-1, tomo_bins)
    num_flat = out_num.reshape(-1)
    den_flat = out_den.reshape(-1)
    ga, gb = shear3[idx_a], shear3[idx_b]          # (npairs, TOMO, 2)
    a_r = ga[..., 0] * ea_r[:, None] - ga[..., 1] * ea_i[:, None]
    a_i = ga[..., 0] * ea_i[:, None] + ga[..., 1] * ea_r[:, None]
    b_r = gb[..., 0] * eb_r[:, None] - gb[..., 1] * eb_i[:, None]
    b_i = gb[..., 0] * eb_i[:, None] + gb[..., 1] * eb_r[:, None]
    w_a, w_b = weights2[idx_a], weights2[idx_b]
    k = 0
    for i in range(tomo_bins):
        for j in range(i, tomo_bins):
            sum_p = sum_m = sum_w = acc(0)
            for ori, (ta, tb) in enumerate(((i, j), (j, i))):
                if ori == 1 and i == j:
                    continue
                w_pair = w_a[:, ta] * w_b[:, tb]
                rr = b_r[:, tb] * a_r[:, ta]
                ii = b_i[:, tb] * a_i[:, ta]
                sum_p += np.sum(w_pair * (rr + ii), dtype=acc)
                sum_m += np.sum(w_pair * (rr - ii), dtype=acc)
                sum_w += np.sum(w_pair, dtype=acc)
            num_flat[k * nbins_total + bin_flat] = sum_p
            num_flat[(ncomb + k) * nbins_total + bin_flat] = sum_m
            den_flat[k * nbins_total + bin_flat] = sum_w
            k += 1


def _tile_dd(density, weights, tomo_bins, auto_only, acc, geom, out_num, out_den, nbins_total, bin_flat):
    idx_a, idx_b = geom[0], geom[1]
    density2 = np.asarray(density).reshape(-1, tomo_bins)
    weights2 = np.asarray(weights).reshape(-1, tomo_bins)
    num_flat = out_num.reshape(-1)
    den_flat = out_den.reshape(-1)
    d_a, d_b = density2[idx_a], density2[idx_b]
    w_a, w_b = weights2[idx_a], weights2[idx_b]
    k = 0
    for i in range(tomo_bins):
        for j in range(i, (i + 1) if auto_only else tomo_bins):
            w_ab = w_a[:, i] * w_b[:, j]
            sum_n = np.sum(w_ab * d_a[:, i] * d_b[:, j], dtype=acc)
            sum_w = np.sum(w_ab, dtype=acc)
            if i != j:
                w_ba = w_a[:, j] * w_b[:, i]
                sum_n += np.sum(w_ba * d_a[:, j] * d_b[:, i], dtype=acc)
                sum_w += np.sum(w_ba, dtype=acc)
            num_flat[k * nbins_total + bin_flat] = sum_n
            den_flat[k * nbins_total + bin_flat] = sum_w
            k += 1


def _tile_ds(density, shear, lens_w, source_w, lens_bins, source_bins, acc, geom,
             out_num, out_den, nbins_total, bin_flat):
    idx_a, idx_b, ea_r, ea_i, eb_r, eb_i = geom
    density2 = np.asarray(density).reshape(-1, lens_bins)
    lens_w2 = np.asarray(lens_w).reshape(-1, lens_bins)
    shear3 = np.asarray(shear).reshape(-1, source_bins, 2)
    source_w2 = np.asarray(source_w).reshape(-1, source_bins)
    num_flat = out_num.reshape(-1)
    den_flat = out_den.reshape(-1)
    d_a, d_b = density2[idx_a], density2[idx_b]
    dw_a, dw_b = lens_w2[idx_a], lens_w2[idx_b]
    ga, gb = shear3[idx_a], shear3[idx_b]
    gt_a = -ga[..., 0] * ea_r[:, None] + ga[..., 1] * ea_i[:, None]
    gt_b = -gb[..., 0] * eb_r[:, None] + gb[..., 1] * eb_i[:, None]
    sw_a, sw_b = source_w2[idx_a], source_w2[idx_b]
    for l in range(lens_bins):
        for s_ in range(source_bins):
            k = l * source_bins + s_
            w_ab = dw_a[:, l] * sw_b[:, s_]
            w_ba = dw_b[:, l] * sw_a[:, s_]
            num_flat[k * nbins_total + bin_flat] = (
                np.sum(w_ab * d_a[:, l] * gt_b[:, s_], dtype=acc)
                + np.sum(w_ba * d_b[:, l] * gt_a[:, s_], dtype=acc)
            )
            den_flat[k * nbins_total + bin_flat] = (
                np.sum(w_ab, dtype=acc) + np.sum(w_ba, dtype=acc)
            )


def _emulate_xipm_tiled(params, grid, args):
    """tomo_vectorized_xipm.cu :: gpu_tiled_tomo_reduce_xipm<T, C, TOMO, I, ACC>."""
    map_dtype = _SCALAR_TYPES[params[0]]
    tomo_bins = int(params[2])
    acc = _SCALAR_TYPES[params[4]]
    (shear, weights, ind_i, ind_j, rot_i, rot_j, bin_offsets, out_num, out_den, nbins_total) = args
    nbins_total = int(nbins_total)
    assert int(grid[1]) == 1, "tiled kernel launches one block per angular bin"
    for bin_flat in range(int(grid[0])):
        if bin_flat >= nbins_total:
            continue
        start, stop = int(bin_offsets[bin_flat]), int(bin_offsets[bin_flat + 1])
        geom = _unpacked_geometry(map_dtype, ind_i, ind_j, rot_i, rot_j, start, stop)
        _tile_xipm(shear, weights, tomo_bins, acc, geom, out_num, out_den, nbins_total, bin_flat)


def _emulate_xipm_packed(params, grid, args):
    """tomo_vectorized_xipm.cu :: gpu_tiled_packed_reduce_xipm<T, TOMO, ACC>."""
    map_dtype = _SCALAR_TYPES[params[0]]
    tomo_bins = int(params[1])
    acc = _SCALAR_TYPES[params[2]]
    (shear, weights, pairs, bin_offsets, row_base, out_num, out_den, nbins_total) = args
    nbins_total = int(nbins_total)
    assert int(grid[1]) == 1
    for bin_flat in range(int(grid[0])):
        if bin_flat >= nbins_total:
            continue
        start, stop = int(bin_offsets[bin_flat]), int(bin_offsets[bin_flat + 1])
        geom = _packed_geometry(map_dtype, pairs, int(row_base[bin_flat]), start, stop)
        _tile_xipm(shear, weights, tomo_bins, acc, geom, out_num, out_den, nbins_total, bin_flat)


def _emulate_dd_tiled(params, grid, args):
    """density_density_tomo_vectorized.cu :: gpu_tiled_tomo_reduce_dd<T, TOMO, AUTO, I, ACC>."""
    map_dtype = _SCALAR_TYPES[params[0]]
    tomo_bins, auto_only = int(params[1]), bool(int(params[2]))
    acc = _SCALAR_TYPES[params[4]]
    (density, weights, ind_i, ind_j, bin_offsets, out_num, out_den, nbins_total) = args
    nbins_total = int(nbins_total)
    assert int(grid[1]) == 1
    for bin_flat in range(int(grid[0])):
        if bin_flat >= nbins_total:
            continue
        start, stop = int(bin_offsets[bin_flat]), int(bin_offsets[bin_flat + 1])
        geom = (ind_i[start:stop].astype(np.int64), ind_j[start:stop].astype(np.int64))
        _tile_dd(density, weights, tomo_bins, auto_only, acc, geom, out_num, out_den, nbins_total, bin_flat)


def _emulate_dd_packed(params, grid, args):
    """density_density_tomo_vectorized.cu :: gpu_tiled_packed_reduce_dd<T, TOMO, AUTO, ACC>."""
    map_dtype = _SCALAR_TYPES[params[0]]
    tomo_bins, auto_only = int(params[1]), bool(int(params[2]))
    acc = _SCALAR_TYPES[params[3]]
    (density, weights, pairs, bin_offsets, row_base, out_num, out_den, nbins_total) = args
    nbins_total = int(nbins_total)
    assert int(grid[1]) == 1
    for bin_flat in range(int(grid[0])):
        if bin_flat >= nbins_total:
            continue
        start, stop = int(bin_offsets[bin_flat]), int(bin_offsets[bin_flat + 1])
        geom = _packed_geometry(map_dtype, pairs, int(row_base[bin_flat]), start, stop)
        _tile_dd(density, weights, tomo_bins, auto_only, acc, geom, out_num, out_den, nbins_total, bin_flat)


def _emulate_ds_tiled(params, grid, args):
    """density_shear_tomo_vectorized.cu :: gpu_tiled_tomo_reduce_ds<T, C, L, S, I, ACC>."""
    map_dtype = _SCALAR_TYPES[params[0]]
    lens_bins, source_bins = int(params[2]), int(params[3])
    acc = _SCALAR_TYPES[params[5]]
    (density, shear, lens_w, source_w, ind_i, ind_j, rot_i, rot_j, bin_offsets,
     out_num, out_den, nbins_total) = args
    nbins_total = int(nbins_total)
    assert int(grid[1]) == 1
    for bin_flat in range(int(grid[0])):
        if bin_flat >= nbins_total:
            continue
        start, stop = int(bin_offsets[bin_flat]), int(bin_offsets[bin_flat + 1])
        geom = _unpacked_geometry(map_dtype, ind_i, ind_j, rot_i, rot_j, start, stop)
        _tile_ds(density, shear, lens_w, source_w, lens_bins, source_bins, acc, geom,
                 out_num, out_den, nbins_total, bin_flat)


def _emulate_ds_packed(params, grid, args):
    """density_shear_tomo_vectorized.cu :: gpu_tiled_packed_reduce_ds<T, L, S, ACC>."""
    map_dtype = _SCALAR_TYPES[params[0]]
    lens_bins, source_bins = int(params[1]), int(params[2])
    acc = _SCALAR_TYPES[params[3]]
    (density, shear, lens_w, source_w, pairs, bin_offsets, row_base,
     out_num, out_den, nbins_total) = args
    nbins_total = int(nbins_total)
    assert int(grid[1]) == 1
    for bin_flat in range(int(grid[0])):
        if bin_flat >= nbins_total:
            continue
        start, stop = int(bin_offsets[bin_flat]), int(bin_offsets[bin_flat + 1])
        geom = _packed_geometry(map_dtype, pairs, int(row_base[bin_flat]), start, stop)
        _tile_ds(density, shear, lens_w, source_w, lens_bins, source_bins, acc, geom,
                 out_num, out_den, nbins_total, bin_flat)


def _emulate_3x2pt_pairs(packed):
    """tomo_tiled_3x2pt.cu :: gpu_tiled_{tomo,packed}_reduce_3x2pt (tile_multi):
    the statistics selected by DO_SS / DO_DD / DO_DS in one walk; per
    statistic identical to the standalone tiles."""

    def _emulate(params, grid, args):
        map_dtype = _SCALAR_TYPES[params[0]]
        off = 1 if packed else 2
        n_density, n_shear, auto_only = (int(p) for p in params[off:off + 3])
        do_ss, do_dd, do_ds = (int(p) for p in params[off + 3:off + 6])
        acc = _SCALAR_TYPES[params[-1]]
        if packed:
            (density, shear, density_w, shear_w, pairs, bin_offsets, row_base,
             xipm_num, xipm_den, xig_num, xig_den, xit_num, xit_den, nbins_total) = args
        else:
            (density, shear, density_w, shear_w, ind_i, ind_j, rot_i, rot_j, bin_offsets,
             xipm_num, xipm_den, xig_num, xig_den, xit_num, xit_den, nbins_total) = args
        nbins_total = int(nbins_total)
        assert int(grid[1]) == 1
        for bin_flat in range(int(grid[0])):
            if bin_flat >= nbins_total:
                continue
            start, stop = int(bin_offsets[bin_flat]), int(bin_offsets[bin_flat + 1])
            if packed:
                geom = _packed_geometry(map_dtype, pairs, int(row_base[bin_flat]), start, stop)
            else:
                geom = _unpacked_geometry(map_dtype, ind_i, ind_j, rot_i, rot_j, start, stop)
            if do_ss:
                _tile_xipm(shear, shear_w, n_shear, acc, geom, xipm_num, xipm_den,
                           nbins_total, bin_flat)
            if do_dd:
                _tile_dd(density, density_w, n_density, bool(auto_only), acc, geom,
                         xig_num, xig_den, nbins_total, bin_flat)
            if do_ds:
                _tile_ds(density, shear, density_w, shear_w, n_density, n_shear, acc, geom,
                         xit_num, xit_den, nbins_total, bin_flat)

    return _emulate


def _emulate_dd(params, grid, args):
    """density_density_tomo_vectorized.cu :: gpu_fused_tomo_reduce_dd<T, TOMO, I, ACC>."""
    tomo_bins = int(params[1])
    acc = _SCALAR_TYPES[params[3]]

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
            # Per-pair products at map precision, accumulated at ACC.
            num_flat[out_idx] = np.sum(
                w_pair * density_flat[base_a] * density_flat[base_b], dtype=acc
            )
            den_flat[out_idx] = np.sum(w_pair, dtype=acc)


def _emulate_ds(params, grid, args):
    """density_shear_tomo_vectorized.cu :: gpu_fused_tomo_reduce_ds<T, C, L, S, I, ACC>."""
    lens_bins = int(params[2])
    source_bins = int(params[3])
    acc = _SCALAR_TYPES[params[5]]

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
            # Per-pair products at map precision, accumulated at ACC.
            num_flat[out_idx] = np.sum(
                w_ab * density_flat[lens_ab] * gt_ab, dtype=acc
            ) + np.sum(w_ba * density_flat[lens_ba] * gt_ba, dtype=acc)
            den_flat[out_idx] = np.sum(w_ab, dtype=acc) + np.sum(w_ba, dtype=acc)


def _check_planar_stride(arr, stride, elem_stride):
    """Both element strides the aperture kernels are given must be the
    ones the view actually has -- the kernel does its own pointer
    arithmetic, so a wrong stride would silently read the wrong pixels."""
    bin_actual = arr.strides[0] // arr.itemsize
    row_actual = arr.strides[1] // arr.itemsize
    if arr.shape[0] > 1 and int(stride) != bin_actual:
        raise AssertionError(
            f"bin stride contract violated: passed {int(stride)}, "
            f"view has {bin_actual}"
        )
    if arr.shape[1] > 1 and int(elem_stride) != row_actual:
        raise AssertionError(
            f"row stride contract violated: passed {int(elem_stride)}, "
            f"view has {row_actual}"
        )


def _emulate_aperture_shear_tomo(params, grid, args):
    """aperture_tomo.cu :: gpu_aperture_shear_tomo<T, QT>."""
    (g1, g2, g_stride, g_elem, weights, w_stride, w_elem, q_inds, q_cos,
     q_sin, q_val, q_offsets, q_patch_area, out_num, out_den, npatches,
     ntomo) = args
    _check_planar_stride(g1, g_stride, g_elem)
    _check_planar_stride(g2, g_stride, g_elem)
    _check_planar_stride(weights, w_stride, w_elem)
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
    (values, v_stride, v_elem, weights, w_stride, w_elem, q_inds, q_val,
     q_offsets, q_patch_area, out_num, out_den, npatches, ntomo) = args
    _check_planar_stride(values, v_stride, v_elem)
    _check_planar_stride(weights, w_stride, w_elem)
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


def _emulate_aperture_shear_tomo_fused(params, grid, args):
    """aperture_tomo.cu :: gpu_aperture_shear_tomo_fused<T, QT, NZ>.

    One block per patch, NZ a template parameter, no trailing `ntomo`
    launch argument -- so the bin loop is driven by NZ, not by grid.y.
    """
    ntomo = int(params[2])
    (g1, g2, g_stride, g_elem, weights, w_stride, w_elem, q_inds, q_cos,
     q_sin, q_val, q_offsets, q_patch_area, out_num, out_den, npatches) = args
    _check_planar_stride(g1, g_stride, g_elem)
    _check_planar_stride(g2, g_stride, g_elem)
    _check_planar_stride(weights, w_stride, w_elem)
    npatches = int(npatches)

    for patch in range(int(grid[0])):
        if patch >= npatches:
            continue
        start = int(q_offsets[patch])
        stop = int(q_offsets[patch + 1])
        pix = q_inds[start:stop].astype(np.int64)
        qc = q_cos[start:stop]
        qs = q_sin[start:stop]
        qv = q_val[start:stop]
        for bin_idx in range(ntomo):
            wv = weights[bin_idx, :][pix]
            gt = -g1[bin_idx, :][pix] * qc - g2[bin_idx, :][pix] * qs
            out_num[bin_idx, patch] = q_patch_area[patch] * np.sum(wv * gt * qv)
            out_den[bin_idx, patch] = np.sum(wv)


def _emulate_aperture_density_tomo_fused(params, grid, args):
    """aperture_tomo.cu :: gpu_aperture_density_tomo_fused<T, QT, NZ>."""
    ntomo = int(params[2])
    (values, v_stride, v_elem, weights, w_stride, w_elem, q_inds, q_val,
     q_offsets, q_patch_area, out_num, out_den, npatches) = args
    _check_planar_stride(values, v_stride, v_elem)
    _check_planar_stride(weights, w_stride, w_elem)
    npatches = int(npatches)

    for patch in range(int(grid[0])):
        if patch >= npatches:
            continue
        start = int(q_offsets[patch])
        stop = int(q_offsets[patch + 1])
        pix = q_inds[start:stop].astype(np.int64)
        qv = q_val[start:stop]
        for bin_idx in range(ntomo):
            wv = weights[bin_idx, :][pix]
            out_num[bin_idx, patch] = q_patch_area[patch] * np.sum(
                wv * values[bin_idx, :][pix] * qv
            )
            out_den[bin_idx, patch] = np.sum(wv)


def _emulate_fused_aperture(params, grid, args, fused=False):
    """tomo_fused_3x2pt.cu :: gpu_3x2pt_tomo_aperture[_fused]<T, QT, ND, NS, ACC>.

    One call per section (the trailing launch argument selects it), like
    the per-section launches of the cupy wrapper.  ``fused`` selects the
    one-block-per-patch kernel, whose bin loop is driven by the compile-time
    N_SHEAR / N_DENSITY rather than by grid.y.
    """
    n_density = int(params[2])
    n_shear = int(params[3])
    acc = _SCALAR_TYPES[params[4]]

    (density, shear, density_w, shear_w, npatches,
     q_inds, q_cos, q_sin, q_val, q_offsets, q_patch_area,
     out_ma_num, out_ma_den, out_mg_num, out_mg_den, section) = args

    npatches = int(npatches)
    section = int(section)
    gx, gy = int(grid[0]), int(grid[1])
    if fused:
        gy = n_shear if section == 0 else n_density

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
                ma_num[y * npatches + x] = acc(q_patch_area[x]) * np.sum(
                    wv * gt * qv, dtype=acc
                )
                ma_den[y * npatches + x] = np.sum(wv, dtype=acc)
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
                mg_num[y * npatches + x] = acc(q_patch_area[x]) * np.sum(
                    wv * density_flat[d_idx] * qv, dtype=acc
                )
                mg_den[y * npatches + x] = np.sum(wv, dtype=acc)
        return

    raise ValueError(f"Unknown aperture section: {section}")


def _emulate_degrade_level(params, grid, args):
    """degrade_rows.cu :: gpu_degrade_level<TSRC, ACC, NVAL, WEIGHTED, SEG>.

    SEG lanes per (cell, lead); the emulator does the same flat gid ->
    (cell, lead) split and the same per-cell gather, summing with np.sum
    instead of the shuffle tree (roundoff-level difference only).  Every
    buffer is addressed by the kernel's four element strides,

        offset(lead, k, row) = base + lead*lead_s + k*comp_s + row*row_s

    so the emulator also checks the SoA / interleaved / AoS layouts.
    """
    n_val = int(params[2])
    weighted = str(params[3]).lower() == "true"
    seg = int(params[4])
    (indptr, indices,
     w_src, w_src_base, w_src_lead, w_src_row,
     v_src, v_src_base, v_src_lead, v_src_comp, v_src_row,
     w_dst, w_dst_base, w_dst_lead, w_dst_row,
     v_dst, v_dst_base, v_dst_lead, v_dst_comp, v_dst_row,
     n_cells, n_lead) = args
    n_cells, n_lead = int(n_cells), int(n_lead)
    w_src_base, w_src_lead, w_src_row = (
        int(w_src_base), int(w_src_lead), int(w_src_row))
    v_src_base, v_src_lead, v_src_comp, v_src_row = (
        int(v_src_base), int(v_src_lead), int(v_src_comp), int(v_src_row))
    w_dst_base, w_dst_lead, w_dst_row = (
        int(w_dst_base), int(w_dst_lead), int(w_dst_row))
    v_dst_base, v_dst_lead, v_dst_comp, v_dst_row = (
        int(v_dst_base), int(v_dst_lead), int(v_dst_comp), int(v_dst_row))

    acc = w_dst.dtype
    wf = w_src.reshape(-1)
    vf = v_src.reshape(-1)
    wo = w_dst.reshape(-1)
    vo = v_dst.reshape(-1)

    # the launch must cover every (cell, lead)
    if seg not in (1, 2, 4, 8, 16, 32):
        raise AssertionError(f"SEG must be a power of two in 1..32, got {seg}")
    covered = int(grid[0]) * (256 // seg)
    if covered < n_cells * n_lead:
        raise AssertionError("degrade launch does not cover all (cell, lead)")

    for cell in range(n_cells):
        child = indices[int(indptr[cell]):int(indptr[cell + 1])].astype(np.int64)
        for lead in range(n_lead):
            wj = wf[w_src_base + lead * w_src_lead + child * w_src_row].astype(acc)
            wo[w_dst_base + lead * w_dst_lead + cell * w_dst_row] = np.sum(wj)
            for k in range(n_val):
                row = v_src_base + lead * v_src_lead + k * v_src_comp
                vj = vf[row + child * v_src_row].astype(acc)
                total = np.sum(wj * vj) if weighted else np.sum(vj)
                vo[v_dst_base + lead * v_dst_lead + k * v_dst_comp
                   + cell * v_dst_row] = total


def _emulate_degrade_finalize(params, grid, args):
    """degrade_rows.cu :: gpu_degrade_finalize<T, ACC, NVAL>."""
    n_val = int(params[2])
    (w_app, w_app_lead, v_app, v_app_lead, v_app_comp,
     w_rows, w_rows_base, w_rows_lead, w_rows_row,
     v_rows, v_rows_base, v_rows_lead, v_rows_comp, v_rows_row,
     n_lead, lo, hi, row_inner) = args
    w_app_lead = int(w_app_lead)
    v_app_lead, v_app_comp = int(v_app_lead), int(v_app_comp)
    w_rows_base, w_rows_lead, w_rows_row = (
        int(w_rows_base), int(w_rows_lead), int(w_rows_row))
    v_rows_base, v_rows_lead, v_rows_comp, v_rows_row = (
        int(v_rows_base), int(v_rows_lead), int(v_rows_comp), int(v_rows_row))
    n_lead, lo, hi = int(n_lead), int(lo), int(hi)
    del row_inner  # only decides which index runs fastest on the device

    wa = w_app.reshape(-1)
    va = v_app.reshape(-1)
    wr = w_rows.reshape(-1)
    vr = v_rows.reshape(-1)
    out_dtype = w_rows.dtype

    for lead in range(n_lead):
        for r in range(lo, hi):
            w = wa[lead * w_app_lead + r]
            wr[w_rows_base + lead * w_rows_lead + r * w_rows_row] = out_dtype.type(w)
            inv = (1.0 / w) if w != 0 else 0.0
            for k in range(n_val):
                sv = va[lead * v_app_lead + k * v_app_comp + r]
                vr[v_rows_base + lead * v_rows_lead + k * v_rows_comp
                   + r * v_rows_row] = out_dtype.type(sv * inv)


_KERNEL_EMULATORS = {
    "gpu_fused_tomo_reduce_xipm": _emulate_xipm,
    "gpu_tiled_tomo_reduce_xipm": _emulate_xipm_tiled,
    "gpu_tiled_packed_reduce_xipm": _emulate_xipm_packed,
    "gpu_fused_tomo_reduce_dd": _emulate_dd,
    "gpu_tiled_tomo_reduce_dd": _emulate_dd_tiled,
    "gpu_tiled_packed_reduce_dd": _emulate_dd_packed,
    "gpu_fused_tomo_reduce_ds": _emulate_ds,
    "gpu_tiled_tomo_reduce_ds": _emulate_ds_tiled,
    "gpu_tiled_packed_reduce_ds": _emulate_ds_packed,
    "gpu_tiled_tomo_reduce_3x2pt": _emulate_3x2pt_pairs(packed=False),
    "gpu_tiled_packed_reduce_3x2pt": _emulate_3x2pt_pairs(packed=True),
    "gpu_aperture_shear_tomo": _emulate_aperture_shear_tomo,
    "gpu_aperture_shear_tomo_fused": _emulate_aperture_shear_tomo_fused,
    "gpu_aperture_density_tomo": _emulate_aperture_density_tomo,
    "gpu_aperture_density_tomo_fused": _emulate_aperture_density_tomo_fused,
    "gpu_3x2pt_tomo_aperture": _emulate_fused_aperture,
    "gpu_3x2pt_tomo_aperture_fused": functools.partial(
        _emulate_fused_aperture, fused=True
    ),
    "gpu_degrade_level": _emulate_degrade_level,
    "gpu_degrade_finalize": _emulate_degrade_finalize,
}
