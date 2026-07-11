/*
 * density_density_tomo_vectorized.cu -- Tomographic galaxy clustering
 *                                       correlation xi_g(theta).
 *
 * Computes the galaxy auto-correlation function, which measures the
 * excess probability of finding two galaxies at angular separation theta
 * relative to a random distribution:
 *
 *   xi_g(theta) = <delta_g(x_a) * delta_g(x_b)>
 *
 * where delta_g is the galaxy overdensity contrast (delta = n/n_bar - 1).
 * The weighted numerator accumulated here is:
 *
 *   Sum_pairs  w_a * w_b * delta_a * delta_b
 *
 * Grid layout:
 *   blockIdx.x  = angular separation bin  (0 .. nbins_total-1)
 *   blockIdx.y  = tomographic combination x orientation (A->B / B->A)
 *   threadIdx.x = pair index within the bin (strided loop)
 */

__COMMON_CUDA_SOURCE__

/*
 * T          -- scalar type (float / double)
 * TOMO_BINS  -- number of tomographic redshift bins (compile-time constant)
 * ACC        -- accumulator/output type (double for float32 maps with
 *              accumulation_precision="float64"; otherwise same as T)
 */
template<typename T, int TOMO_BINS, typename I, typename ACC>
__global__ void gpu_fused_tomo_reduce_dd(
    const T* density,        /* galaxy overdensity delta_g per pixel per tomo bin */
    const T* weights,        /* per-pixel, per-tomo-bin weights               */
    const I* ind_i,          /* pixel index of first member of each pair      */
    const I* ind_j,          /* pixel index of second member                  */
    const long long* bin_offsets,  /* CSR offsets per angular bin              */
    const int* comb_i,       /* tomo bin index for the "i" side               */
    const int* comb_j,       /* tomo bin index for the "j" side               */
    ACC* out_num,            /* output: weighted deltadelta numerators                */
    ACC* out_den,            /* output: weight sums, one row per comb_ori     */
    const int ncomb,
    const long long nbins_total,
    const long long npairs)
{
    const int lane = (int)threadIdx.x;
    /* comb_ori encodes tomo combination + pair orientation (A->B / B->A) */
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
        return;   /* auto-bin is symmetric -- skip duplicate */
    }

    const long long start = bin_offsets[bin_flat];
    const long long stop = bin_offsets[bin_flat + 1];

    ACC sum_val = (ACC)0.0;
    ACC sum_w = (ACC)0.0;

    /* Sum w_a * w_b * delta_a * delta_b over all pairs in this angular bin,
       accumulating the weight sum (denominator) in the same pass */
    for (long long tid = start + lane; tid < stop; tid += BLOCK_SIZE) {
        const long long idx_a = (long long)ind_i[tid];
        const long long idx_b = (long long)ind_j[tid];

        /* Swap tomo bins for B->A orientation */
        int ai = i;
        int bj = j;
        if (use_ba && i != j) {
            ai = j;
            bj = i;
        }

        const long long base_a = idx_a * (long long)TOMO_BINS + ai;
        const long long base_b = idx_b * (long long)TOMO_BINS + bj;

        const T w_pair = weights[base_a] * weights[base_b];
        sum_w += (ACC)w_pair;
        sum_val += (ACC)(w_pair * density[base_a] * density[base_b]);
    }

    block_reduce_sum_pair<ACC>(sum_val, sum_w, &sum_val, &sum_w);

    if (lane == 0) {
        const long long out_idx =
            ((long long)comb_ori) * nbins_total + bin_flat;
        out_num[out_idx] = sum_val;
        out_den[out_idx] = sum_w;
    }
}
