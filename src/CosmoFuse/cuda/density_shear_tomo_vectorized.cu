__COMMON_CUDA_SOURCE__

#include <cuComplex.h>
template<typename T, typename C, int LENS_TOMO_BINS, int SOURCE_TOMO_BINS>
__global__ void gpu_fused_tomo_reduce_ds(
    const T* density,
    const T* shear,
    const T* lens_weights,
    const T* source_weights,
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
    const int comb_idx = (int)blockIdx.y;
    const long long bin_flat = (long long)blockIdx.x;
    if (bin_flat >= nbins_total || comb_idx >= ncomb) {
        return;
    }

    const int lens_bin = comb_i[comb_idx];
    const int source_bin = comb_j[comb_idx];

    const long long start = bin_offsets[bin_flat];
    const long long stop = bin_offsets[bin_flat + 1];

    T sum_val = (T)0.0;

    for (long long tid = start + lane; tid < stop; tid += BLOCK_SIZE) {
        const long long idx_a = ind_i[tid];
        const long long idx_b = ind_j[tid];
        const C rot_ab = rot_j[tid];
        const C rot_ba = rot_i[tid];

        const long long lens_idx_ab = idx_a * (long long)LENS_TOMO_BINS + lens_bin;
        const long long source_idx_ab = idx_b * (long long)SOURCE_TOMO_BINS + source_bin;
        const long long shear_base_ab = source_idx_ab * 2;

        const T gamma_t_ab = (
            -shear[shear_base_ab] * rot_ab.x
            + shear[shear_base_ab + 1] * rot_ab.y
        );

        sum_val += (
            lens_weights[lens_idx_ab]
            * source_weights[source_idx_ab]
            * density[lens_idx_ab]
            * gamma_t_ab
        );

        const long long lens_idx_ba = idx_b * (long long)LENS_TOMO_BINS + lens_bin;
        const long long source_idx_ba = idx_a * (long long)SOURCE_TOMO_BINS + source_bin;
        const long long shear_base_ba = source_idx_ba * 2;

        const T gamma_t_ba = (
            -shear[shear_base_ba] * rot_ba.x
            + shear[shear_base_ba + 1] * rot_ba.y
        );

        sum_val += (
            lens_weights[lens_idx_ba]
            * source_weights[source_idx_ba]
            * density[lens_idx_ba]
            * gamma_t_ba
        );
    }

    sum_val = block_reduce_sum<T>(sum_val);

    if (lane == 0) {
        const long long out_idx =
            ((long long)comb_idx) * nbins_total + bin_flat;
        out_num[out_idx] = sum_val;
    }
}
