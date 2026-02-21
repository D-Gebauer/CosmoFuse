__COMMON_CUDA_SOURCE__

#define TOMO_BINS __TOMO_BINS__

extern "C" __global__
void __KERNEL_NAME__(
    const __MAP_C_TYPE__* density,
    const __MAP_C_TYPE__* weights,
    const long long* ind_i,
    const long long* ind_j,
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

    __MAP_C_TYPE__ sum_val = (__MAP_C_TYPE__)0.0;

    for (long long tid = start + lane; tid < stop; tid += BLOCK_SIZE) {
        const long long idx_a = ind_i[tid];
        const long long idx_b = ind_j[tid];

        int ai = i;
        int bj = j;
        if (use_ba && i != j) {
            ai = j;
            bj = i;
        }

        const long long base_a = idx_a * (long long)TOMO_BINS + ai;
        const long long base_b = idx_b * (long long)TOMO_BINS + bj;

        sum_val += (
            weights[base_a]
            * weights[base_b]
            * density[base_a]
            * density[base_b]
        );
    }

    sum_val = block_reduce_sum(sum_val);

    if (lane == 0) {
        const long long out_idx =
            ((long long)comb_ori) * nbins_total + bin_flat;
        out_num[out_idx] = sum_val;
    }
}
