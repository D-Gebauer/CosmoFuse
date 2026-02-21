__COMMON_CUDA_SOURCE__

#include <cuComplex.h>
#define TOMO_BINS __TOMO_BINS__

extern "C" __global__
void __KERNEL_NAME__(
    const __MAP_C_TYPE__* shear,
    const __MAP_C_TYPE__* weights,
    const long long* ind_i,
    const long long* ind_j,
    const __COMPLEX_C_TYPE__* rot_i,
    const __COMPLEX_C_TYPE__* rot_j,
    const long long* bin_offsets,
    const int* comb_i,
    const int* comb_j,
    __MAP_C_TYPE__* out_num,
    const int ncomb,
    const long long nbins_total,
    const long long npairs)
{
    const int lane = (int)threadIdx.x;
    const int comb_ori = (int)blockIdx.y;
    const long long bin_flat = (long long)blockIdx.x;
    if (bin_flat >= nbins_total || comb_ori >= (2 * ncomb)) {
        return;
    }

    const int comb_idx = comb_ori >> 1;
    const int i = comb_i[comb_idx];
    const int j = comb_j[comb_idx];
    const bool use_ba = (comb_ori & 1) == 1;
    if (use_ba && i == j) {
        return;
    }

    const long long start = bin_offsets[bin_flat];
    const long long stop = bin_offsets[bin_flat + 1];

    __MAP_C_TYPE__ sum_p = (__MAP_C_TYPE__)0.0;
    __MAP_C_TYPE__ sum_m = (__MAP_C_TYPE__)0.0;

    for (long long tid = start + lane; tid < stop; tid += BLOCK_SIZE) {
        const long long idx_a = ind_i[tid];
        const long long idx_b = ind_j[tid];
        const __COMPLEX_C_TYPE__ exp_a = rot_i[tid];
        const __COMPLEX_C_TYPE__ exp_b = rot_j[tid];

        int ai = i;
        int bj = j;
        if (use_ba && i != j) {
            ai = j;
            bj = i;
        }

        const long long idx_a_bin = idx_a * (long long)TOMO_BINS + ai;
        const long long idx_b_bin = idx_b * (long long)TOMO_BINS + bj;
        const long long base_a = idx_a_bin * 2;
        const long long base_b = idx_b_bin * 2;

        const __COMPLEX_C_TYPE__ g_a = __MAKE_COMPLEX__(
            (__COMPLEX_REAL_TYPE__)shear[base_a],
            (__COMPLEX_REAL_TYPE__)shear[base_a + 1]
        );
        const __COMPLEX_C_TYPE__ g_b = __MAKE_COMPLEX__(
            (__COMPLEX_REAL_TYPE__)shear[base_b],
            (__COMPLEX_REAL_TYPE__)shear[base_b + 1]
        );

        __COMPLEX_C_TYPE__ term_a = __CMUL_FN__(g_a, exp_a);
        __COMPLEX_C_TYPE__ term_b = __CMUL_FN__(g_b, exp_b);

        __MAP_C_TYPE__ w_pair =
            weights[idx_a_bin] * weights[idx_b_bin];

        const __COMPLEX_REAL_TYPE__ a_R = term_a.x;
        const __COMPLEX_REAL_TYPE__ a_I = term_a.y;
        const __COMPLEX_REAL_TYPE__ b_R = term_b.x;
        const __COMPLEX_REAL_TYPE__ b_I = term_b.y;

        sum_p += (__COMPLEX_REAL_TYPE__)w_pair * (b_R * a_R + b_I * a_I);
        sum_m += (__COMPLEX_REAL_TYPE__)w_pair * (b_R * a_R - b_I * a_I);
    }

    block_reduce_sum_pair(sum_p, sum_m, &sum_p, &sum_m);

    if (lane == 0) {
        const long long out_p_idx =
            ((long long)comb_ori) * nbins_total + bin_flat;
        const long long out_m_idx =
            ((long long)(2 * ncomb + comb_ori)) * nbins_total + bin_flat;

        out_num[out_p_idx] = (__MAP_C_TYPE__)sum_p;
        out_num[out_m_idx] = (__MAP_C_TYPE__)sum_m;
    }
}
