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


@njit(fastmath=True, parallel=True)
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
            gt = -g1[pix_idx] * Q_cos[i] - g2[pix_idx] * Q_sin[i]
            sum_w += weight
            sum_wgt_q += weight * gt * Q_val[i]
        out_aperture[patch_idx] = Q_patch_area[patch_idx] * sum_wgt_q / sum_w


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


def _build_cupy_aperture_shear_kernel(module: Any) -> Any:
    return module.ElementwiseKernel(
        "raw I Q_inds, raw T Q_cos, raw T Q_sin, raw T Q_val,"
        " raw T g1, raw T g2, raw T weights",
        "T out_num, T out_den",
        """
        const I idx = Q_inds[i];
        const T w = weights[idx];
        const T gt = -g1[idx] * Q_cos[i] - g2[idx] * Q_sin[i];
        out_num = w * gt * Q_val[i];
        out_den = w;
        """,
        "gpu_aperture_shear_kernel",
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
            gamma_t = -g1_source[j] * rot.real + g2_source[j] * rot.imag
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
                    -shear_map[pix_j, source_bin, 0] * exp_j.real
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
                    -shear_map[pix_i, source_bin, 0] * exp_i.real
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
        const T gamma_t = -g1_source[j_idx] * real(rot) + g2_source[j_idx] * imag(rot);
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
    kernel_cache: dict[tuple[str, str, int, int], Any] = {}

    def _get_or_build_raw_kernel(
        map_c_type: str,
        complex_c_type: str,
        complex_real_type: str,
        suffix: str,
        nlens_bins: int,
        nsource_bins: int,
    ) -> Optional[Any]:
        key = (map_c_type, suffix, nlens_bins, nsource_bins)
        cached = kernel_cache.get(key)
        if cached is not None:
            return cached

        kernel_name = (
            f"gpu_fused_tomo_reduce_ds_{map_c_type}_{suffix}_"
            f"{nlens_bins}_{nsource_bins}"
        )
        source = (
            _COMMON_CUDA_SOURCE
            + f"""
        #include <cuComplex.h>
        #define LENS_TOMO_BINS {nlens_bins}
        #define SOURCE_TOMO_BINS {nsource_bins}

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

                const long long lens_idx_ab = idx_a * (long long)LENS_TOMO_BINS + lens_bin;
                const long long source_idx_ab = idx_b * (long long)SOURCE_TOMO_BINS + source_bin;
                const long long shear_base_ab = source_idx_ab * 2;

                const {complex_real_type} gamma_t_ab = (
                    -({complex_real_type})shear[shear_base_ab] * rot_ab.x
                    + ({complex_real_type})shear[shear_base_ab + 1] * rot_ab.y
                );

                sum_val += (
                    lens_weights[lens_idx_ab]
                    * source_weights[source_idx_ab]
                    * density[lens_idx_ab]
                    * ({map_c_type})gamma_t_ab
                );

                const long long lens_idx_ba = idx_b * (long long)LENS_TOMO_BINS + lens_bin;
                const long long source_idx_ba = idx_a * (long long)SOURCE_TOMO_BINS + source_bin;
                const long long shear_base_ba = source_idx_ba * 2;

                const {complex_real_type} gamma_t_ba = (
                    -({complex_real_type})shear[shear_base_ba] * rot_ba.x
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
                "Vectorized density-shear RawKernel compilation failed for lens/source bins (%d, %d); using legacy path: %s",
                nlens_bins,
                nsource_bins,
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
        nlens_bins = int(density_map.shape[1])
        nsource_bins = int(shear_map.shape[1])
        if nlens_bins > _MAX_VECTOR_TOMO_BINS or nsource_bins > _MAX_VECTOR_TOMO_BINS:
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
            nlens_bins,
            nsource_bins,
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
    comb_i: np.ndarray,
    comb_j: np.ndarray,
    out_p: np.ndarray,
    out_m: np.ndarray,
) -> None:
    n_bins = offsets.shape[0] - 1
    ncomb = comb_i.shape[0]

    for b in prange(n_bins):
        start = offsets[b]
        stop = offsets[b + 1]

        for comb_idx in range(ncomb):
            i = comb_i[comb_idx]
            j = comb_j[comb_idx]

            sum_ab_p = 0.0 + 0.0j
            sum_ab_m = 0.0 + 0.0j
            sum_ba_p = 0.0 + 0.0j
            sum_ba_m = 0.0 + 0.0j

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

                sum_ab_p += w_ij * gb_j * np.conjugate(ga_i)
                sum_ab_m += w_ij * gb_j * ga_i

                if i != j:
                    ga_j = (
                        shear_map[pix_j, i, 0] + 1j * shear_map[pix_j, i, 1]
                    ) * exp_j
                    gb_i = (
                        shear_map[pix_i, j, 0] + 1j * shear_map[pix_i, j, 1]
                    ) * exp_i
                    w_ji = weights[pix_i, j] * weights[pix_j, i]

                    sum_ba_p += w_ji * ga_j * np.conjugate(gb_i)
                    sum_ba_m += w_ji * ga_j * gb_i

            out_row_ab = 2 * comb_idx
            out_p[out_row_ab, b] = sum_ab_p
            out_m[out_row_ab, b] = sum_ab_m

            if i != j:
                out_row_ba = out_row_ab + 1
                out_p[out_row_ba, b] = sum_ba_p
                out_m[out_row_ba, b] = sum_ba_m


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


@njit(fastmath=True, parallel=True)
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

    for tomo_idx in prange(n_shear):
        for patch_idx in range(n_patches):
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

    for tomo_idx in prange(n_density):
        for patch_idx in range(n_patches):
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

    n_ss_comb = ss_comb_i.shape[0]
    for comb_ori in prange(2 * n_ss_comb):
        comb_idx = comb_ori >> 1
        ori = comb_ori & 1
        i_bin = ss_comb_i[comb_idx]
        j_bin = ss_comb_j[comb_idx]

        if ori == 1 and i_bin == j_bin:
            continue

        ai = i_bin
        bj = j_bin
        if ori == 1 and i_bin != j_bin:
            ai = j_bin
            bj = i_bin

        for bin_flat in range(nbins_total):
            start = pair_offsets[bin_flat]
            stop = pair_offsets[bin_flat + 1]
            sum_p = 0.0
            sum_m = 0.0
            sum_w = 0.0
            for pair_idx in range(start, stop):
                pix_a = ind_i[pair_idx]
                pix_b = ind_j[pair_idx]

                exp_a = rot_i[pair_idx]
                exp_b = rot_j[pair_idx]

                ga1 = shear_map[pix_a, ai, 0]
                ga2 = shear_map[pix_a, ai, 1]
                gb1 = shear_map[pix_b, bj, 0]
                gb2 = shear_map[pix_b, bj, 1]

                a_r = ga1 * exp_a.real - ga2 * exp_a.imag
                a_i = ga1 * exp_a.imag + ga2 * exp_a.real
                b_r = gb1 * exp_b.real - gb2 * exp_b.imag
                b_i = gb1 * exp_b.imag + gb2 * exp_b.real

                w_pair = shear_weights[pix_a, ai] * shear_weights[pix_b, bj]
                sum_w += w_pair
                sum_p += w_pair * (b_r * a_r + b_i * a_i)
                sum_m += w_pair * (b_r * a_r - b_i * a_i)

            out_xip_num[comb_ori, bin_flat] = sum_p
            out_xim_num[comb_ori, bin_flat] = sum_m
            out_xipm_den[comb_ori, bin_flat] = sum_w

    n_dd_comb = dd_comb_i.shape[0]
    for comb_ori in prange(2 * n_dd_comb):
        comb_idx = comb_ori >> 1
        ori = comb_ori & 1
        i_bin = dd_comb_i[comb_idx]
        j_bin = dd_comb_j[comb_idx]

        if ori == 1 and i_bin == j_bin:
            continue

        ai = i_bin
        bj = j_bin
        if ori == 1 and i_bin != j_bin:
            ai = j_bin
            bj = i_bin

        for bin_flat in range(nbins_total):
            start = pair_offsets[bin_flat]
            stop = pair_offsets[bin_flat + 1]
            sum_num = 0.0
            sum_den = 0.0
            for pair_idx in range(start, stop):
                pix_a = ind_i[pair_idx]
                pix_b = ind_j[pair_idx]
                weight = density_weights[pix_a, ai] * density_weights[pix_b, bj]
                sum_den += weight
                sum_num += weight * density_map[pix_a, ai] * density_map[pix_b, bj]

            out_xig_num[comb_ori, bin_flat] = sum_num
            out_xig_den[comb_ori, bin_flat] = sum_den

    n_ds_comb = ds_comb_i.shape[0]
    for comb_idx in prange(n_ds_comb):
        lens_bin = ds_comb_i[comb_idx]
        source_bin = ds_comb_j[comb_idx]

        for bin_flat in range(nbins_total):
            start = pair_offsets[bin_flat]
            stop = pair_offsets[bin_flat + 1]
            sum_num = 0.0
            sum_den = 0.0
            for pair_idx in range(start, stop):
                pix_a = ind_i[pair_idx]
                pix_b = ind_j[pair_idx]

                exp_ab = rot_j[pair_idx]
                gt_ab = (
                    -shear_map[pix_b, source_bin, 0] * exp_ab.real
                    + shear_map[pix_b, source_bin, 1] * exp_ab.imag
                )
                w_ab = density_weights[pix_a, lens_bin] * shear_weights[pix_b, source_bin]
                sum_num += w_ab * density_map[pix_a, lens_bin] * gt_ab
                sum_den += w_ab

                exp_ba = rot_i[pair_idx]
                gt_ba = (
                    -shear_map[pix_a, source_bin, 0] * exp_ba.real
                    + shear_map[pix_a, source_bin, 1] * exp_ba.imag
                )
                w_ba = density_weights[pix_b, lens_bin] * shear_weights[pix_a, source_bin]
                sum_num += w_ba * density_map[pix_b, lens_bin] * gt_ba
                sum_den += w_ba

            out_xit_num[comb_idx, bin_flat] = sum_num
            out_xit_den[comb_idx, bin_flat] = sum_den


def _build_cupy_3x2pt_tomo_fused_kernel(module: Any) -> Any:
    kernel_cache: dict[tuple[str, str], Any] = {}

    def _get_or_build_raw_kernel(
        map_c_type: str,
        complex_c_type: str,
        suffix: str,
    ) -> Optional[Any]:
        key = (map_c_type, suffix)
        cached = kernel_cache.get(key)
        if cached is not None:
            return cached

        kernel_name = f"gpu_3x2pt_tomo_fused_{map_c_type}_{suffix}"
        source = (
            _COMMON_CUDA_SOURCE
            + f"""
        #include <cuComplex.h>

        extern "C" __global__
        void {kernel_name}(
            const {map_c_type}* density,
            const {map_c_type}* shear,
            const {map_c_type}* density_w,
            const {map_c_type}* shear_w,
            const long long* ind_i,
            const long long* ind_j,
            const {complex_c_type}* rot_i,
            const {complex_c_type}* rot_j,
            const long long* pair_offsets,
            const long long nbins_total,
            const int n_density_bins,
            const int n_shear_bins,
            const int npatches,
            const int npix,
            const unsigned int* q_inds,
            const {map_c_type}* q_cos,
            const {map_c_type}* q_sin,
            const {map_c_type}* q_val,
            const long long* q_offsets,
            const {map_c_type}* q_patch_area,
            const int* ss_comb_i,
            const int* ss_comb_j,
            const int n_ss_comb,
            const int* dd_comb_i,
            const int* dd_comb_j,
            const int n_dd_comb,
            const int* ds_comb_i,
            const int* ds_comb_j,
            const int n_ds_comb,
            {map_c_type}* out_ma_num,
            {map_c_type}* out_ma_den,
            {map_c_type}* out_mg_num,
            {map_c_type}* out_mg_den,
            {map_c_type}* out_xip_num,
            {map_c_type}* out_xim_num,
            {map_c_type}* out_xipm_den,
            {map_c_type}* out_xig_num,
            {map_c_type}* out_xig_den,
            {map_c_type}* out_xit_num,
            {map_c_type}* out_xit_den)
        {{
            const int lane = (int)threadIdx.x;
            const long long x = (long long)blockIdx.x;
            const int y = (int)blockIdx.y;
            const int z = (int)blockIdx.z;

            if (z == 0) {{
                if (x >= npatches || y >= n_shear_bins) return;
                const long long start = q_offsets[x];
                const long long stop = q_offsets[x + 1];
                {map_c_type} sum_num = ({map_c_type})0.0;
                {map_c_type} sum_den = ({map_c_type})0.0;
                for (long long idx = start + lane; idx < stop; idx += BLOCK_SIZE) {{
                    const unsigned int pix = q_inds[idx];
                    const long long shear_idx = ((long long)pix * (long long)n_shear_bins + (long long)y) * 2LL;
                    const long long w_idx = (long long)pix * (long long)n_shear_bins + (long long)y;
                    const {map_c_type} g1 = shear[shear_idx];
                    const {map_c_type} g2 = shear[shear_idx + 1LL];
                    const {map_c_type} wv = shear_w[w_idx];
                    const {map_c_type} gt = -g1 * q_cos[idx] - g2 * q_sin[idx];
                    sum_num += wv * gt * q_val[idx];
                    sum_den += wv;
                }}
                block_reduce_sum_pair(sum_num, sum_den, &sum_num, &sum_den);
                if (lane == 0) {{
                    const long long out_idx = (long long)y * (long long)npatches + x;
                    out_ma_num[out_idx] = q_patch_area[x] * sum_num;
                    out_ma_den[out_idx] = sum_den;
                }}
                return;
            }}

            if (z == 1) {{
                if (x >= npatches || y >= n_density_bins) return;
                const long long start = q_offsets[x];
                const long long stop = q_offsets[x + 1];
                {map_c_type} sum_num = ({map_c_type})0.0;
                {map_c_type} sum_den = ({map_c_type})0.0;
                for (long long idx = start + lane; idx < stop; idx += BLOCK_SIZE) {{
                    const unsigned int pix = q_inds[idx];
                    const long long d_idx = (long long)pix * (long long)n_density_bins + (long long)y;
                    const {map_c_type} wv = density_w[d_idx];
                    sum_num += wv * density[d_idx] * q_val[idx];
                    sum_den += wv;
                }}
                block_reduce_sum_pair(sum_num, sum_den, &sum_num, &sum_den);
                if (lane == 0) {{
                    const long long out_idx = (long long)y * (long long)npatches + x;
                    out_mg_num[out_idx] = q_patch_area[x] * sum_num;
                    out_mg_den[out_idx] = sum_den;
                }}
                return;
            }}

            if (z == 2) {{
                if (x >= nbins_total || y >= (2 * n_ss_comb)) return;
                const int comb_idx = y >> 1;
                const int ori = y & 1;
                const int i = ss_comb_i[comb_idx];
                const int j = ss_comb_j[comb_idx];
                if (ori == 1 && i == j) return;

                int ai = i;
                int bj = j;
                if (ori == 1 && i != j) {{
                    ai = j;
                    bj = i;
                }}

                const long long start = pair_offsets[x];
                const long long stop = pair_offsets[x + 1];
                {map_c_type} sum_p = ({map_c_type})0.0;
                {map_c_type} sum_m = ({map_c_type})0.0;
                {map_c_type} sum_w = ({map_c_type})0.0;
                for (long long idx = start + lane; idx < stop; idx += BLOCK_SIZE) {{
                    const long long pix_a = ind_i[idx];
                    const long long pix_b = ind_j[idx];
                    const {complex_c_type} ex_a = rot_i[idx];
                    const {complex_c_type} ex_b = rot_j[idx];

                    const long long a_base = ((pix_a * (long long)n_shear_bins + (long long)ai) * 2LL);
                    const long long b_base = ((pix_b * (long long)n_shear_bins + (long long)bj) * 2LL);
                    const {map_c_type} ga1 = shear[a_base];
                    const {map_c_type} ga2 = shear[a_base + 1LL];
                    const {map_c_type} gb1 = shear[b_base];
                    const {map_c_type} gb2 = shear[b_base + 1LL];

                    const {map_c_type} a_r = ga1 * ex_a.x - ga2 * ex_a.y;
                    const {map_c_type} a_i = ga1 * ex_a.y + ga2 * ex_a.x;
                    const {map_c_type} b_r = gb1 * ex_b.x - gb2 * ex_b.y;
                    const {map_c_type} b_i = gb1 * ex_b.y + gb2 * ex_b.x;

                    const long long wa_idx = pix_a * (long long)n_shear_bins + (long long)ai;
                    const long long wb_idx = pix_b * (long long)n_shear_bins + (long long)bj;
                    const {map_c_type} wv = shear_w[wa_idx] * shear_w[wb_idx];
                    sum_w += wv;
                    sum_p += wv * (b_r * a_r + b_i * a_i);
                    sum_m += wv * (b_r * a_r - b_i * a_i);
                }}
                block_reduce_sum_pair(sum_p, sum_m, &sum_p, &sum_m);
                sum_w = block_reduce_sum(sum_w);
                if (lane == 0) {{
                    const long long out_idx = (long long)y * nbins_total + x;
                    out_xip_num[out_idx] = sum_p;
                    out_xim_num[out_idx] = sum_m;
                    out_xipm_den[out_idx] = sum_w;
                }}
                return;
            }}

            if (z == 3) {{
                if (x >= nbins_total || y >= (2 * n_dd_comb)) return;
                const int comb_idx = y >> 1;
                const int ori = y & 1;
                const int i = dd_comb_i[comb_idx];
                const int j = dd_comb_j[comb_idx];
                if (ori == 1 && i == j) return;

                int ai = i;
                int bj = j;
                if (ori == 1 && i != j) {{
                    ai = j;
                    bj = i;
                }}

                const long long start = pair_offsets[x];
                const long long stop = pair_offsets[x + 1];
                {map_c_type} sum_num = ({map_c_type})0.0;
                {map_c_type} sum_den = ({map_c_type})0.0;
                for (long long idx = start + lane; idx < stop; idx += BLOCK_SIZE) {{
                    const long long pix_a = ind_i[idx];
                    const long long pix_b = ind_j[idx];
                    const long long ia = pix_a * (long long)n_density_bins + (long long)ai;
                    const long long jb = pix_b * (long long)n_density_bins + (long long)bj;
                    const {map_c_type} wv = density_w[ia] * density_w[jb];
                    sum_den += wv;
                    sum_num += wv * density[ia] * density[jb];
                }}
                block_reduce_sum_pair(sum_num, sum_den, &sum_num, &sum_den);
                if (lane == 0) {{
                    const long long out_idx = (long long)y * nbins_total + x;
                    out_xig_num[out_idx] = sum_num;
                    out_xig_den[out_idx] = sum_den;
                }}
                return;
            }}

            if (z == 4) {{
                if (x >= nbins_total || y >= n_ds_comb) return;
                const int lens_bin = ds_comb_i[y];
                const int source_bin = ds_comb_j[y];
                const long long start = pair_offsets[x];
                const long long stop = pair_offsets[x + 1];
                {map_c_type} sum_num = ({map_c_type})0.0;
                {map_c_type} sum_den = ({map_c_type})0.0;
                for (long long idx = start + lane; idx < stop; idx += BLOCK_SIZE) {{
                    const long long pix_a = ind_i[idx];
                    const long long pix_b = ind_j[idx];

                    const long long lens_ab = pix_a * (long long)n_density_bins + (long long)lens_bin;
                    const long long src_ab = pix_b * (long long)n_shear_bins + (long long)source_bin;
                    const long long src_ab_base = src_ab * 2LL;
                    const {complex_c_type} ex_ab = rot_j[idx];
                    const {map_c_type} gt_ab = -shear[src_ab_base] * ex_ab.x + shear[src_ab_base + 1LL] * ex_ab.y;
                    const {map_c_type} w_ab = density_w[lens_ab] * shear_w[src_ab];
                    sum_num += w_ab * density[lens_ab] * gt_ab;
                    sum_den += w_ab;

                    const long long lens_ba = pix_b * (long long)n_density_bins + (long long)lens_bin;
                    const long long src_ba = pix_a * (long long)n_shear_bins + (long long)source_bin;
                    const long long src_ba_base = src_ba * 2LL;
                    const {complex_c_type} ex_ba = rot_i[idx];
                    const {map_c_type} gt_ba = -shear[src_ba_base] * ex_ba.x + shear[src_ba_base + 1LL] * ex_ba.y;
                    const {map_c_type} w_ba = density_w[lens_ba] * shear_w[src_ba];
                    sum_num += w_ba * density[lens_ba] * gt_ba;
                    sum_den += w_ba;
                }}
                block_reduce_sum_pair(sum_num, sum_den, &sum_num, &sum_den);
                if (lane == 0) {{
                    const long long out_idx = (long long)y * nbins_total + x;
                    out_xit_num[out_idx] = sum_num;
                    out_xit_den[out_idx] = sum_den;
                }}
                return;
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
                "Fused 3x2pt RawKernel compilation failed; using unfused path: %s",
                exc,
            )
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
        if getattr(module, "RawKernel", None) is None:
            return False

        if rot_i.dtype == module.complex64:
            complex_c_type = "cuFloatComplex"
            suffix = "c64"
        else:
            complex_c_type = "cuDoubleComplex"
            suffix = "c128"

        map_c_type = "float" if density_map.dtype == module.float32 else "double"
        raw_kernel = _get_or_build_raw_kernel(
            map_c_type,
            complex_c_type,
            suffix,
        )
        if raw_kernel is None:
            return False

        nbins_total = int(pair_offsets.shape[0] - 1)
        npatches = int(q_offsets.shape[0] - 1)
        n_density_bins = int(density_map.shape[1])
        n_shear_bins = int(shear_map.shape[1])
        npix = int(density_map.shape[0])
        n_ss_comb = int(ss_comb_i.shape[0])
        n_dd_comb = int(dd_comb_i.shape[0])
        n_ds_comb = int(ds_comb_i.shape[0])

        max_x = max(1, nbins_total, npatches)
        max_y = max(1, n_shear_bins, n_density_bins, 2 * n_ss_comb, 2 * n_dd_comb, n_ds_comb)
        threads = 256
        blocks = (max_x, max_y, 5)

        raw_kernel(
            blocks,
            (threads,),
            (
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
                np.int32(n_density_bins),
                np.int32(n_shear_bins),
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
            ),
        )
        return True

    return _cupy_3x2pt_tomo_fused_kernel


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
        aperture_shear_kernel: Optional[Any] = None,
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
