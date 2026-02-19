import logging
import warnings
from typing import Any, Optional, Union

import numpy as np
from numba import njit, prange

logger = logging.getLogger(__name__)

_CUPY_FASTMATH_OPTIONS = ("--use_fast_math",)
_MAX_VECTOR_TOMO_BINS = 64

_COMMON_CUDA_SOURCE = """
#define BLOCK_SIZE 256

template<typename T>
__device__ inline T block_reduce_sum(T val) {
    __shared__ T shared[BLOCK_SIZE];
    int lane = threadIdx.x;
    shared[lane] = val;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            shared[lane] += shared[lane + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

template<typename T>
__device__ inline void block_reduce_sum_pair(T val1, T val2, T* out1, T* out2) {
    __shared__ T s1[BLOCK_SIZE];
    __shared__ T s2[BLOCK_SIZE];
    int lane = threadIdx.x;
    s1[lane] = val1;
    s2[lane] = val2;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            s1[lane] += s1[lane + stride];
            s2[lane] += s2[lane + stride];
        }
        __syncthreads();
    }
    *out1 = s1[0];
    *out2 = s2[0];
}
"""


@njit(fastmath=True, parallel=True)
def _cpu_aperture_density_kernel(
    Q_inds: np.ndarray,
    Q_val: np.ndarray,
    Q_offsets: np.ndarray,
    map_values: np.ndarray,
    weights: np.ndarray,
    Q_patch_area: np.ndarray,
    out_aperture: np.ndarray,
) -> None:
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


def _build_cupy_aperture_density_kernel(module: Any) -> Any:
    return module.ElementwiseKernel(
        "raw I Q_inds, raw T Q_val, raw T map_values, raw T weights",
        "T out_num, T out_den",
        """
        const I idx = Q_inds[i];
        const T w = weights[idx];
        out_num = w * map_values[idx] * Q_val[i];
        out_den = w;
        """,
        "gpu_aperture_density_kernel",
        options=_CUPY_FASTMATH_OPTIONS,
    )


@njit(fastmath=True, parallel=True)
def _cpu_density_density_corr_kernel(
    density_a: np.ndarray,
    density_b: np.ndarray,
    w_a: np.ndarray,
    w_b: np.ndarray,
    ind_i: np.ndarray,
    ind_j: np.ndarray,
    offsets: np.ndarray,
    out_w: np.ndarray,
) -> None:
    nbins = offsets.shape[0] - 1
    for b in prange(nbins):
        sum_w = 0.0
        start = offsets[b]
        stop = offsets[b + 1]

        for idx in range(start, stop):
            i = ind_i[idx]
            j = ind_j[idx]
            sum_w += w_a[i] * w_b[j] * density_a[i] * density_b[j]

        out_w[b] = sum_w


@njit(fastmath=True, parallel=True)
def _cpu_density_shear_corr_kernel(
    density_lens: np.ndarray,
    g1_source: np.ndarray,
    g2_source: np.ndarray,
    w_lens: np.ndarray,
    w_source: np.ndarray,
    ind_i: np.ndarray,
    ind_j: np.ndarray,
    exp_j: np.ndarray,
    offsets: np.ndarray,
    out_gt: np.ndarray,
) -> None:
    nbins = offsets.shape[0] - 1
    for b in prange(nbins):
        sum_gt = 0.0
        start = offsets[b]
        stop = offsets[b + 1]

        for idx in range(start, stop):
            i = ind_i[idx]
            j = ind_j[idx]
            rot = exp_j[idx]
            gamma_t = g1_source[j] * rot.real + g2_source[j] * rot.imag
            sum_gt += w_lens[i] * w_source[j] * density_lens[i] * gamma_t

        out_gt[b] = sum_gt


@njit(fastmath=True, parallel=True)
def _cpu_density_density_tomo_vectorized_kernel(
    density_map: np.ndarray,
    weights: np.ndarray,
    ind_i: np.ndarray,
    ind_j: np.ndarray,
    offsets: np.ndarray,
    comb_i: np.ndarray,
    comb_j: np.ndarray,
    out_num: np.ndarray,
) -> None:
    n_bins = offsets.shape[0] - 1
    ncomb = comb_i.shape[0]
    half = 0.5

    for b in prange(n_bins):
        start = offsets[b]
        stop = offsets[b + 1]

        for comb_idx in range(ncomb):
            i = comb_i[comb_idx]
            j = comb_j[comb_idx]
            sum_w = 0.0

            for idx in range(start, stop):
                pix_i = int(ind_i[idx])
                pix_j = int(ind_j[idx])

                ab = (
                    weights[pix_i, i]
                    * weights[pix_j, j]
                    * density_map[pix_i, i]
                    * density_map[pix_j, j]
                )

                if i == j:
                    sum_w += ab
                else:
                    ba = (
                        weights[pix_i, j]
                        * weights[pix_j, i]
                        * density_map[pix_i, j]
                        * density_map[pix_j, i]
                    )
                    sum_w += half * (ab + ba)

            out_num[comb_idx, b] = sum_w


@njit(fastmath=True, parallel=True)
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
) -> None:
    n_bins = offsets.shape[0] - 1
    ncomb = comb_i.shape[0]

    for b in prange(n_bins):
        start = offsets[b]
        stop = offsets[b + 1]

        for comb_idx in range(ncomb):
            lens_bin = comb_i[comb_idx]
            source_bin = comb_j[comb_idx]
            sum_gt = 0.0

            for idx in range(start, stop):
                pix_i = int(ind_i[idx])
                pix_j = int(ind_j[idx])
                exp_j = rot_j[idx]
                gamma_t_ij = (
                    shear_map[pix_j, source_bin, 0] * exp_j.real
                    + shear_map[pix_j, source_bin, 1] * exp_j.imag
                )
                sum_gt += (
                    lens_weights[pix_i, lens_bin]
                    * source_weights[pix_j, source_bin]
                    * density_map[pix_i, lens_bin]
                    * gamma_t_ij
                )

                exp_i = rot_i[idx]
                gamma_t_ji = (
                    shear_map[pix_i, source_bin, 0] * exp_i.real
                    + shear_map[pix_i, source_bin, 1] * exp_i.imag
                )
                sum_gt += (
                    lens_weights[pix_j, lens_bin]
                    * source_weights[pix_i, source_bin]
                    * density_map[pix_j, lens_bin]
                    * gamma_t_ji
                )

            out_num[comb_idx, b] = sum_gt


def _build_cupy_density_density_corr_kernel(module: Any) -> Any:
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
    return module.ElementwiseKernel(
        "raw T density_lens, raw T g1_source, raw T g2_source,"
        " raw T w_lens, raw T w_source, raw I ind_i, raw I ind_j, raw C exp_j",
        "T out_gt",
        """
        const I i_idx = ind_i[i];
        const I j_idx = ind_j[i];
        const C rot = exp_j[i];
        const T gamma_t = g1_source[j_idx] * real(rot) + g2_source[j_idx] * imag(rot);
        out_gt = w_lens[i_idx] * w_source[j_idx] * density_lens[i_idx] * gamma_t;
        """,
        "gpu_density_shear_corr_kernel",
        options=_CUPY_FASTMATH_OPTIONS,
    )


def _build_cupy_density_density_tomo_vectorized_kernel(module: Any) -> Any:
    kernel_cache: dict[tuple[str, int], Any] = {}

    def _get_or_build_raw_kernel(map_c_type: str, nzbins: int) -> Optional[Any]:
        key = (map_c_type, nzbins)
        cached = kernel_cache.get(key)
        if cached is not None:
            return cached

        kernel_name = f"gpu_fused_tomo_reduce_dd_{map_c_type}_{nzbins}"
        source = (
            _COMMON_CUDA_SOURCE
            + f"""
        #define TOMO_BINS {nzbins}

        extern "C" __global__
        void {kernel_name}(
            const {map_c_type}* density,
            const {map_c_type}* weights,
            const long long* ind_i,
            const long long* ind_j,
            const long long* bin_offsets,
            const int* comb_i,
            const int* comb_j,
            {map_c_type}* out_num,
            const int ncomb,
            const long long nbins_total,
            const long long npairs)
        {{
            const int lane = (int)threadIdx.x;
            const int comb_ori = (int)blockIdx.y;
            const long long bin_flat = (long long)blockIdx.x;
            if (bin_flat >= nbins_total || comb_ori >= (2 * ncomb)) {{
                return;
            }}

            const int comb_idx = comb_ori >> 1;
            const int i = comb_i[comb_idx];
            const int j = comb_j[comb_idx];
            const bool use_ba = (comb_ori & 1) == 1;
            if (use_ba && i == j) {{
                return;
            }}

            const long long start = bin_offsets[bin_flat];
            const long long stop = bin_offsets[bin_flat + 1];

            {map_c_type} sum_val = ({map_c_type})0.0;

            for (long long tid = start + lane; tid < stop; tid += BLOCK_SIZE) {{
                const long long idx_a = ind_i[tid];
                const long long idx_b = ind_j[tid];

                int ai = i;
                int bj = j;
                if (use_ba && i != j) {{
                    ai = j;
                    bj = i;
                }}

                const long long base_a = idx_a * (long long)TOMO_BINS + ai;
                const long long base_b = idx_b * (long long)TOMO_BINS + bj;

                sum_val += (
                    weights[base_a]
                    * weights[base_b]
                    * density[base_a]
                    * density[base_b]
                );
            }}

            sum_val = block_reduce_sum(sum_val);

            if (lane == 0) {{
                const long long out_idx =
                    ((long long)comb_ori) * nbins_total + bin_flat;
                out_num[out_idx] = sum_val;
            }}
        }}
        """
        )

        try:
            kernel = module.RawKernel(
                source,
                kernel_name,
                options=_CUPY_FASTMATH_OPTIONS,
            )
        except Exception as exc:
            logger.warning(
                "Vectorized density-density RawKernel compilation failed for %d bins; using legacy path: %s",
                nzbins,
                exc,
            )
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
    ) -> bool:
        nzbins = int(density_map.shape[1])
        if nzbins > _MAX_VECTOR_TOMO_BINS:
            return False
        if getattr(module, "RawKernel", None) is None:
            return False

        map_c_type = "float" if weights.dtype == module.float32 else "double"
        raw_kernel = _get_or_build_raw_kernel(map_c_type, nzbins)
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
                np.int32(ncomb),
                np.int64(nbins_total),
                np.int64(npairs),
            ),
        )
        return True

    return _cupy_density_density_tomo_vectorized_kernel


def _build_cupy_density_shear_tomo_vectorized_kernel(module: Any) -> Any:
    kernel_cache: dict[tuple[str, str, int], Any] = {}

    def _get_or_build_raw_kernel(
        map_c_type: str,
        complex_c_type: str,
        complex_real_type: str,
        suffix: str,
        nzbins: int,
    ) -> Optional[Any]:
        key = (map_c_type, suffix, nzbins)
        cached = kernel_cache.get(key)
        if cached is not None:
            return cached

        kernel_name = f"gpu_fused_tomo_reduce_ds_{map_c_type}_{suffix}_{nzbins}"
        source = (
            _COMMON_CUDA_SOURCE
            + f"""
        #include <cuComplex.h>
        #define TOMO_BINS {nzbins}

        extern "C" __global__
        void {kernel_name}(
            const {map_c_type}* density,
            const {map_c_type}* shear,
            const {map_c_type}* lens_weights,
            const {map_c_type}* source_weights,
            const long long* ind_i,
            const long long* ind_j,
            const {complex_c_type}* rot_i,
            const {complex_c_type}* rot_j,
            const long long* bin_offsets,
            const int* comb_i,
            const int* comb_j,
            {map_c_type}* out_num,
            const int ncomb,
            const long long nbins_total,
            const long long npairs)
        {{
            const int lane = (int)threadIdx.x;
            const int comb_idx = (int)blockIdx.y;
            const long long bin_flat = (long long)blockIdx.x;
            if (bin_flat >= nbins_total || comb_idx >= ncomb) {{
                return;
            }}

            const int lens_bin = comb_i[comb_idx];
            const int source_bin = comb_j[comb_idx];

            const long long start = bin_offsets[bin_flat];
            const long long stop = bin_offsets[bin_flat + 1];

            {map_c_type} sum_val = ({map_c_type})0.0;

            for (long long tid = start + lane; tid < stop; tid += BLOCK_SIZE) {{
                const long long idx_a = ind_i[tid];
                const long long idx_b = ind_j[tid];
                const {complex_c_type} rot_ab = rot_j[tid];
                const {complex_c_type} rot_ba = rot_i[tid];

                const long long lens_idx_ab = idx_a * (long long)TOMO_BINS + lens_bin;
                const long long source_idx_ab = idx_b * (long long)TOMO_BINS + source_bin;
                const long long shear_base_ab = source_idx_ab * 2;

                const {complex_real_type} gamma_t_ab = (
                    ({complex_real_type})shear[shear_base_ab] * rot_ab.x
                    + ({complex_real_type})shear[shear_base_ab + 1] * rot_ab.y
                );

                sum_val += (
                    lens_weights[lens_idx_ab]
                    * source_weights[source_idx_ab]
                    * density[lens_idx_ab]
                    * ({map_c_type})gamma_t_ab
                );

                const long long lens_idx_ba = idx_b * (long long)TOMO_BINS + lens_bin;
                const long long source_idx_ba = idx_a * (long long)TOMO_BINS + source_bin;
                const long long shear_base_ba = source_idx_ba * 2;

                const {complex_real_type} gamma_t_ba = (
                    ({complex_real_type})shear[shear_base_ba] * rot_ba.x
                    + ({complex_real_type})shear[shear_base_ba + 1] * rot_ba.y
                );

                sum_val += (
                    lens_weights[lens_idx_ba]
                    * source_weights[source_idx_ba]
                    * density[lens_idx_ba]
                    * ({map_c_type})gamma_t_ba
                );
            }}

            sum_val = block_reduce_sum(sum_val);

            if (lane == 0) {{
                const long long out_idx =
                    ((long long)comb_idx) * nbins_total + bin_flat;
                out_num[out_idx] = sum_val;
            }}
        }}
        """
        )

        try:
            kernel = module.RawKernel(
                source,
                kernel_name,
                options=_CUPY_FASTMATH_OPTIONS,
            )
        except Exception as exc:
            logger.warning(
                "Vectorized density-shear RawKernel compilation failed for %d bins; using legacy path: %s",
                nzbins,
                exc,
            )
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
    ) -> bool:
        nzbins = int(density_map.shape[1])
        if nzbins > _MAX_VECTOR_TOMO_BINS:
            return False
        if getattr(module, "RawKernel", None) is None:
            return False

        if rot_j.dtype == module.complex64:
            complex_c_type = "cuFloatComplex"
            complex_real_type = "float"
            suffix = "c64"
        else:
            complex_c_type = "cuDoubleComplex"
            complex_real_type = "double"
            suffix = "c128"

        map_c_type = "float" if lens_weights.dtype == module.float32 else "double"
        raw_kernel = _get_or_build_raw_kernel(
            map_c_type,
            complex_c_type,
            complex_real_type,
            suffix,
            nzbins,
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
                np.int32(ncomb),
                np.int64(nbins_total),
                np.int64(npairs),
            ),
        )
        return True

    return _cupy_density_shear_tomo_vectorized_kernel


@njit(fastmath=True, parallel=True)
def _cpu_xipm_cross_corr_kernel_c64(
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
) -> None:
    nbins = offsets.shape[0] - 1
    for b in prange(nbins):
        ab_p_acc = np.complex64(0.0 + 0.0j)
        ab_m_acc = np.complex64(0.0 + 0.0j)
        ba_p_acc = np.complex64(0.0 + 0.0j)
        ba_m_acc = np.complex64(0.0 + 0.0j)
        start = offsets[b]
        stop = offsets[b + 1]

        for idx in range(start, stop):
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

            ab_p_acc += gb_j_rot * np.conjugate(ga_i_rot)
            ab_m_acc += gb_j_rot * ga_i_rot
            ba_p_acc += ga_j_rot * np.conjugate(gb_i_rot)
            ba_m_acc += ga_j_rot * gb_i_rot

        out_ab_p[b] = ab_p_acc
        out_ab_m[b] = ab_m_acc
        out_ba_p[b] = ba_p_acc
        out_ba_m[b] = ba_m_acc


@njit(fastmath=True, parallel=True)
def _cpu_xipm_cross_corr_kernel_c128(
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
) -> None:
    nbins = offsets.shape[0] - 1
    for b in prange(nbins):
        ab_p_acc = np.complex128(0.0 + 0.0j)
        ab_m_acc = np.complex128(0.0 + 0.0j)
        ba_p_acc = np.complex128(0.0 + 0.0j)
        ba_m_acc = np.complex128(0.0 + 0.0j)
        start = offsets[b]
        stop = offsets[b + 1]

        for idx in range(start, stop):
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

            ab_p_acc += gb_j_rot * np.conjugate(ga_i_rot)
            ab_m_acc += gb_j_rot * ga_i_rot
            ba_p_acc += ga_j_rot * np.conjugate(gb_i_rot)
            ba_m_acc += ga_j_rot * gb_i_rot

        out_ab_p[b] = ab_p_acc
        out_ab_m[b] = ab_m_acc
        out_ba_p[b] = ba_p_acc
        out_ba_m[b] = ba_m_acc


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
) -> None:
    if exp_i.dtype == np.complex64:
        _cpu_xipm_cross_corr_kernel_c64(
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
            offsets,
            out_ab_p,
            out_ab_m,
            out_ba_p,
            out_ba_m,
        )
    else:
        _cpu_xipm_cross_corr_kernel_c128(
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
            offsets,
            out_ab_p,
            out_ab_m,
            out_ba_p,
            out_ba_m,
        )


@njit(fastmath=True, parallel=True)
def _cpu_xipm_auto_corr_kernel_c64(
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
) -> None:
    nbins = offsets.shape[0] - 1
    for b in prange(nbins):
        p_acc = np.complex64(0.0 + 0.0j)
        m_acc = np.complex64(0.0 + 0.0j)
        start = offsets[b]
        stop = offsets[b + 1]

        for idx in range(start, stop):
            i = ind_i[idx]
            j = ind_j[idx]

            g2 = w1[i] * np.complex64(g11[i] + 1j * g21[i]) * exp_i[idx]
            g1 = w2[j] * np.complex64(g12[j] + 1j * g22[j]) * exp_j[idx]

            p_acc += g1 * np.conjugate(g2)
            m_acc += g1 * g2

        out_p[b] = p_acc
        out_m[b] = m_acc


@njit(fastmath=True, parallel=True)
def _cpu_xipm_auto_corr_kernel_c128(
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
) -> None:
    nbins = offsets.shape[0] - 1
    for b in prange(nbins):
        p_acc = np.complex128(0.0 + 0.0j)
        m_acc = np.complex128(0.0 + 0.0j)
        start = offsets[b]
        stop = offsets[b + 1]

        for idx in range(start, stop):
            i = ind_i[idx]
            j = ind_j[idx]

            g2 = w1[i] * np.complex128(g11[i] + 1j * g21[i]) * exp_i[idx]
            g1 = w2[j] * np.complex128(g12[j] + 1j * g22[j]) * exp_j[idx]

            p_acc += g1 * np.conjugate(g2)
            m_acc += g1 * g2

        out_p[b] = p_acc
        out_m[b] = m_acc


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
) -> None:
    if exp_i.dtype == np.complex64:
        _cpu_xipm_auto_corr_kernel_c64(
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
            offsets,
            out_p,
            out_m,
        )
    else:
        _cpu_xipm_auto_corr_kernel_c128(
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
            offsets,
            out_p,
            out_m,
        )


@njit(fastmath=True, parallel=True)
def _cpu_vectorized_tomo_kernel(
    shear_map: np.ndarray,
    weights: np.ndarray,
    ind_i: np.ndarray,
    ind_j: np.ndarray,
    rot_i: np.ndarray,
    rot_j: np.ndarray,
    offsets: np.ndarray,
    out_p: np.ndarray,
    out_m: np.ndarray,
) -> None:
    n_bins = offsets.shape[0] - 1
    nz = shear_map.shape[1]
    half = 0.5

    for b in prange(n_bins):
        start = offsets[b]
        stop = offsets[b + 1]

        comb_idx = 0
        for i in range(nz):
            for j in range(i, nz):
                sum_p = 0.0 + 0.0j
                sum_m = 0.0 + 0.0j

                for idx in range(start, stop):
                    pix_i = int(ind_i[idx])
                    pix_j = int(ind_j[idx])
                    exp_i = rot_i[idx]
                    exp_j = rot_j[idx]

                    ga_i = (
                        shear_map[pix_i, i, 0] + 1j * shear_map[pix_i, i, 1]
                    ) * exp_i
                    gb_j = (
                        shear_map[pix_j, j, 0] + 1j * shear_map[pix_j, j, 1]
                    ) * exp_j
                    w_ij = weights[pix_i, i] * weights[pix_j, j]

                    ab_p = w_ij * gb_j * np.conjugate(ga_i)
                    ab_m = w_ij * gb_j * ga_i

                    if i == j:
                        sum_p += ab_p
                        sum_m += ab_m
                    else:
                        ga_q = (
                            shear_map[pix_j, i, 0] + 1j * shear_map[pix_j, i, 1]
                        ) * exp_j
                        gb_p = (
                            shear_map[pix_i, j, 0] + 1j * shear_map[pix_i, j, 1]
                        ) * exp_i
                        w_ji = weights[pix_i, j] * weights[pix_j, i]

                        ba_p = w_ji * ga_q * np.conjugate(gb_p)
                        ba_m = w_ji * ga_q * gb_p

                        sum_p += half * (ab_p + ba_p)
                        sum_m += half * (ab_m + ba_m)

                out_p[comb_idx, b] = sum_p
                out_m[comb_idx, b] = sum_m
                comb_idx += 1


def _build_cupy_tomo_vectorized_kernel(module: Any) -> Any:
    kernel_cache: dict[tuple[str, str, int], Any] = {}

    def _get_or_build_raw_kernel(
        map_c_type: str,
        complex_c_type: str,
        complex_real_type: str,
        make_complex: str,
        cmul_fn: str,
        conj_fn: str,
        cadd_fn: str,
        real_fn: str,
        suffix: str,
        nzbins: int,
    ) -> Optional[Any]:
        key = (map_c_type, suffix, nzbins)
        cached = kernel_cache.get(key)
        if cached is not None:
            return cached

        kernel_name = f"gpu_fused_tomo_reduce_xipm_{map_c_type}_{suffix}_{nzbins}"
        source = (
            _COMMON_CUDA_SOURCE
            + f"""
        #include <cuComplex.h>
        #define TOMO_BINS {nzbins}

        extern "C" __global__
        void {kernel_name}(
            const {map_c_type}* shear,
            const {map_c_type}* weights,
            const long long* ind_i,
            const long long* ind_j,
            const {complex_c_type}* rot_i,
            const {complex_c_type}* rot_j,
            const long long* bin_offsets,
            const int* comb_i,
            const int* comb_j,
            {map_c_type}* out_num,
            const int ncomb,
            const long long nbins_total,
            const long long npairs)
        {{
            const int lane = (int)threadIdx.x;
            const int comb_ori = (int)blockIdx.y;
            const long long bin_flat = (long long)blockIdx.x;
            if (bin_flat >= nbins_total || comb_ori >= (2 * ncomb)) {{
                return;
            }}

            const int comb_idx = comb_ori >> 1;
            const int i = comb_i[comb_idx];
            const int j = comb_j[comb_idx];
            const bool use_ba = (comb_ori & 1) == 1;
            if (use_ba && i == j) {{
                return;
            }}

            const long long start = bin_offsets[bin_flat];
            const long long stop = bin_offsets[bin_flat + 1];

            {map_c_type} sum_p = ({map_c_type})0.0;
            {map_c_type} sum_m = ({map_c_type})0.0;

            for (long long tid = start + lane; tid < stop; tid += BLOCK_SIZE) {{
                const long long idx_a = ind_i[tid];
                const long long idx_b = ind_j[tid];
                const {complex_c_type} exp_a = rot_i[tid];
                const {complex_c_type} exp_b = rot_j[tid];

                int ai = i;
                int bj = j;
                if (use_ba && i != j) {{
                    ai = j;
                    bj = i;
                }}

                const long long idx_a_bin = idx_a * (long long)TOMO_BINS + ai;
                const long long idx_b_bin = idx_b * (long long)TOMO_BINS + bj;
                const long long base_a = idx_a_bin * 2;
                const long long base_b = idx_b_bin * 2;

                const {complex_c_type} g_a = {make_complex}(
                    ({complex_real_type})shear[base_a],
                    ({complex_real_type})shear[base_a + 1]
                );
                const {complex_c_type} g_b = {make_complex}(
                    ({complex_real_type})shear[base_b],
                    ({complex_real_type})shear[base_b + 1]
                );

                {complex_c_type} term_a = {cmul_fn}(g_a, exp_a);
                {complex_c_type} term_b = {cmul_fn}(g_b, exp_b);

                {map_c_type} w_pair = 
                    weights[idx_a_bin] * weights[idx_b_bin];

                const {complex_real_type} a_R = term_a.x;
                const {complex_real_type} a_I = term_a.y;
                const {complex_real_type} b_R = term_b.x;
                const {complex_real_type} b_I = term_b.y;

                sum_p += ({complex_real_type})w_pair * (b_R * a_R + b_I * a_I);
                sum_m += ({complex_real_type})w_pair * (b_R * a_R - b_I * a_I);
            }}

            block_reduce_sum_pair(sum_p, sum_m, &sum_p, &sum_m);

            if (lane == 0) {{
                const long long out_p_idx =
                    ((long long)comb_ori) * nbins_total + bin_flat;
                const long long out_m_idx =
                    ((long long)(2 * ncomb + comb_ori)) * nbins_total + bin_flat;

                out_num[out_p_idx] = ({map_c_type})sum_p;
                out_num[out_m_idx] = ({map_c_type})sum_m;
            }}
        }}
        """
        )

        try:
            kernel = module.RawKernel(
                source,
                kernel_name,
                options=_CUPY_FASTMATH_OPTIONS,
            )
        except Exception as exc:
            logger.warning(
                "Vectorized tomography RawKernel compilation failed for %d bins; using legacy path: %s",
                nzbins,
                exc,
            )
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
    ) -> bool:
        nzbins = int(shear_map.shape[1])
        if nzbins > _MAX_VECTOR_TOMO_BINS:
            return False
        if getattr(module, "RawKernel", None) is None:
            return False

        if rot_i.dtype == module.complex64:
            complex_c_type = "cuFloatComplex"
            complex_real_type = "float"
            make_complex = "make_cuFloatComplex"
            cmul_fn = "cuCmulf"
            conj_fn = "cuConjf"
            cadd_fn = "cuCaddf"
            real_fn = "cuCrealf"
            suffix = "c64"
        else:
            complex_c_type = "cuDoubleComplex"
            complex_real_type = "double"
            make_complex = "make_cuDoubleComplex"
            cmul_fn = "cuCmul"
            conj_fn = "cuConj"
            cadd_fn = "cuCadd"
            real_fn = "cuCreal"
            suffix = "c128"

        map_c_type = "float" if weights.dtype == module.float32 else "double"
        raw_kernel = _get_or_build_raw_kernel(
            map_c_type,
            complex_c_type,
            complex_real_type,
            make_complex,
            cmul_fn,
            conj_fn,
            cadd_fn,
            real_fn,
            suffix,
            nzbins,
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
                np.int32(ncomb),
                np.int64(nbins_total),
                np.int64(npairs),
            ),
        )
        return True

    return _cupy_tomo_vectorized_kernel


def _build_cupy_xipm_cross_corr_kernel(module: Any) -> Any:
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
        kernel_density_density: Optional[Any] = None,
        kernel_density_shear: Optional[Any] = None,
        kernel_density_density_tomo_vectorized: Optional[Any] = None,
        kernel_density_shear_tomo_vectorized: Optional[Any] = None,
    ) -> None:
        self.name = name
        self.module = module
        self.device_id = device_id
        self.xipm_cross_corr_kernel = xipm_cross_corr_kernel
        self.xipm_auto_corr_kernel = xipm_auto_corr_kernel
        self.xipm_tomo_vectorized_kernel = xipm_tomo_vectorized_kernel
        self.aperture_density_kernel = aperture_density_kernel
        self.kernel_density_density = kernel_density_density
        self.kernel_density_shear = kernel_density_shear
        self.kernel_density_density_tomo_vectorized = (
            kernel_density_density_tomo_vectorized
        )
        self.kernel_density_shear_tomo_vectorized = (
            kernel_density_shear_tomo_vectorized
        )

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
            xipm_cross_corr_kernel=_cpu_xipm_cross_corr_kernel,
            xipm_auto_corr_kernel=_cpu_xipm_auto_corr_kernel,
            xipm_tomo_vectorized_kernel=_cpu_vectorized_tomo_kernel,
            aperture_density_kernel=_cpu_aperture_density_kernel,
            kernel_density_density=_cpu_density_density_corr_kernel,
            kernel_density_shear=_cpu_density_shear_corr_kernel,
            kernel_density_density_tomo_vectorized=_cpu_density_density_tomo_vectorized_kernel,
            kernel_density_shear_tomo_vectorized=_cpu_density_shear_tomo_vectorized_kernel,
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
                kernel_density_density=_build_cupy_density_density_corr_kernel(cupy),
                kernel_density_shear=_build_cupy_density_shear_corr_kernel(cupy),
                kernel_density_density_tomo_vectorized=_build_cupy_density_density_tomo_vectorized_kernel(cupy),
                kernel_density_shear_tomo_vectorized=_build_cupy_density_shear_tomo_vectorized_kernel(cupy),
            )
        except ImportError:
            if device == 'auto':
                warnings.warn("Cupy not installed, falling back to CPU (numpy).")
                return Backend('numpy', np)
            else:
                raise ImportError("Cupy not installed but GPU requested.")
