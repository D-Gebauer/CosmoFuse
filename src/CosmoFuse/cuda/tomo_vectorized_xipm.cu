__COMMON_CUDA_SOURCE__

#include <cuComplex.h>

template<typename C>
__device__ inline C complex_make(double re, double im);

template<>
__device__ inline cuFloatComplex complex_make<cuFloatComplex>(double re, double im) {
    return make_cuFloatComplex((float)re, (float)im);
}

template<>
__device__ inline cuDoubleComplex complex_make<cuDoubleComplex>(double re, double im) {
    return make_cuDoubleComplex(re, im);
}

template<typename C>
__device__ inline C complex_mul(C a, C b);

template<>
__device__ inline cuFloatComplex complex_mul<cuFloatComplex>(cuFloatComplex a, cuFloatComplex b) {
    return cuCmulf(a, b);
}

template<>
__device__ inline cuDoubleComplex complex_mul<cuDoubleComplex>(cuDoubleComplex a, cuDoubleComplex b) {
    return cuCmul(a, b);
}

template<typename T, typename C, int TOMO_BINS_T>
__global__ void gpu_fused_tomo_reduce_xipm(
    const T* shear,
    const T* weights,
    const long long* ind_i,
    const long long* ind_j,
    const C* rot_i,
    const C* rot_j,
    const long long* bin_offsets,
    const int* comb_i,
    const int* comb_j,
    T* out_num,
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

    T sum_p = (T)0.0;
    T sum_m = (T)0.0;

    for (long long tid = start + lane; tid < stop; tid += BLOCK_SIZE) {
        const long long idx_a = ind_i[tid];
        const long long idx_b = ind_j[tid];
        const C exp_a = rot_i[tid];
        const C exp_b = rot_j[tid];

        int ai = i;
        int bj = j;
        if (use_ba && i != j) {
            ai = j;
            bj = i;
        }

        const long long idx_a_bin = idx_a * (long long)TOMO_BINS_T + ai;
        const long long idx_b_bin = idx_b * (long long)TOMO_BINS_T + bj;
        const long long base_a = idx_a_bin * 2;
        const long long base_b = idx_b_bin * 2;

        const C g_a = complex_make<C>((double)shear[base_a], (double)shear[base_a + 1]);
        const C g_b = complex_make<C>((double)shear[base_b], (double)shear[base_b + 1]);

        C term_a = complex_mul<C>(g_a, exp_a);
        C term_b = complex_mul<C>(g_b, exp_b);

        T w_pair =
            weights[idx_a_bin] * weights[idx_b_bin];

        const T a_R = (T)term_a.x;
        const T a_I = (T)term_a.y;
        const T b_R = (T)term_b.x;
        const T b_I = (T)term_b.y;

        sum_p += w_pair * (b_R * a_R + b_I * a_I);
        sum_m += w_pair * (b_R * a_R - b_I * a_I);
    }

    block_reduce_sum_pair<T>(sum_p, sum_m, &sum_p, &sum_m);

    if (lane == 0) {
        const long long out_p_idx =
            ((long long)comb_ori) * nbins_total + bin_flat;
        const long long out_m_idx =
            ((long long)(2 * ncomb + comb_ori)) * nbins_total + bin_flat;

        out_num[out_p_idx] = sum_p;
        out_num[out_m_idx] = sum_m;
    }
}
