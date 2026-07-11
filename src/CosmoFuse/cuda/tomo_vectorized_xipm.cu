/*
 * tomo_vectorized_xipm.cu -- Tomographic cosmic shear 2-point correlation
 *                            functions xi+(theta) and xi-(theta).
 *
 * Computes the shear-shear correlation numerators for all tomographic bin
 * combinations in a single kernel launch.  The two correlation functions
 * probe the projected matter power spectrum:
 *
 *   xi+(theta) = Re<gamma'_b * conj(gamma'_a)>  (parity-even; sensitive to E+B modes)
 *   xi-(theta) = Re<gamma'_b * gamma'_a>        (parity-odd;  sensitive to E-B modes)
 *
 * where gamma' = gamma * e^{2iphi} is the complex shear rotated into the local
 * frame defined by the line connecting the pixel pair.
 *
 * Grid layout:
 *   blockIdx.x  = angular separation bin index  (0 .. nbins_total-1)
 *   blockIdx.y  = tomographic combination x orientation
 *                 (even = A->B, odd = B->A; skipped when i==j)
 *   threadIdx.x = pair index within the bin (strided loop)
 */

__COMMON_CUDA_SOURCE__


/*
 * T            -- scalar type (float / double) for map values and weights
 * C            -- complex type (cuFloatComplex / cuDoubleComplex) for rotations
 * TOMO_BINS_T  -- number of tomographic redshift bins (compile-time constant
 *                for efficient index arithmetic)
 * ACC          -- accumulator/output type (double for float32 maps with
 *                accumulation_precision="float64"; otherwise same as T)
 */
template<typename T, typename C, int TOMO_BINS_T, typename I, typename ACC>
__global__ void gpu_fused_tomo_reduce_xipm(
    const T* shear,          /* complex shear gamma = (gamma_1, gamma_2) interleaved,
                                layout: [npix x TOMO_BINS x 2]           */
    const T* weights,        /* per-pixel, per-tomo-bin weights            */
    const I* ind_i,          /* pixel index of first member of each pair   */
    const I* ind_j,          /* pixel index of second member               */
    const C* rot_i,          /* e^{2iphi_i}: rotation factor for pixel i     */
    const C* rot_j,          /* e^{2iphi_j}: rotation factor for pixel j     */
    const long long* bin_offsets,  /* CSR offsets into the pair arrays
                                      per angular separation bin           */
    const int* comb_i,       /* tomographic bin index for the "i" side     */
    const int* comb_j,       /* tomographic bin index for the "j" side     */
    ACC* out_num,            /* output numerators [xi+ block | xi- block]    */
    ACC* out_den,            /* weight sums, one row per comb_ori            */
    const int ncomb,         /* number of unique tomo-bin combinations     */
    const long long nbins_total,
    const long long npairs)
{
    const int lane = (int)threadIdx.x;
    /* comb_ori encodes both the tomo combination and the pair orientation:
       even index = (A->B) ordering, odd = (B->A) to symmetrise the estimator */
    const int comb_ori = (int)blockIdx.y;
    const long long bin_flat = (long long)blockIdx.x;
    if (bin_flat >= nbins_total || comb_ori >= (2 * ncomb)) {
        return;
    }

    const int comb_idx = comb_ori >> 1;
    const int i = comb_i[comb_idx];   /* tomo bin for pixel a */
    const int j = comb_j[comb_idx];   /* tomo bin for pixel b */
    const bool use_ba = (comb_ori & 1) == 1;
    if (use_ba && i == j) {
        return;   /* auto-correlation is symmetric -- skip duplicate */
    }

    const long long start = bin_offsets[bin_flat];
    const long long stop = bin_offsets[bin_flat + 1];

    ACC sum_p = (ACC)0.0;   /* accumulator for xi+ numerator */
    ACC sum_m = (ACC)0.0;   /* accumulator for xi- numerator */
    ACC sum_w = (ACC)0.0;   /* accumulator for the weight sum (denominator) */

    /* Loop over all pixel pairs in this angular bin */
    for (long long tid = start + lane; tid < stop; tid += BLOCK_SIZE) {
        const long long idx_a = (long long)ind_i[tid];
        const long long idx_b = (long long)ind_j[tid];
        const C exp_a = rot_i[tid];   /* e^{2iphi_a} */
        const C exp_b = rot_j[tid];   /* e^{2iphi_b} */

        /* For the B->A orientation, swap the tomo bin assignment */
        int ai = i;
        int bj = j;
        if (use_ba && i != j) {
            ai = j;
            bj = i;
        }

        /* Fetch the complex shear gamma = gamma_1 + igamma_2 for each pixel and tomo bin */
        const long long idx_a_bin = idx_a * (long long)TOMO_BINS_T + ai;
        const long long idx_b_bin = idx_b * (long long)TOMO_BINS_T + bj;
        const long long base_a = idx_a_bin * 2;
        const long long base_b = idx_b_bin * 2;

        const T ga1 = shear[base_a];
        const T ga2 = shear[base_a + 1];
        const T gb1 = shear[base_b];
        const T gb2 = shear[base_b + 1];

        /* Rotate shears into the pair frame: gamma' = gamma * e^{2iphi}.
           The multiply runs at map precision T (the rotation components are
           promoted, exact for float -> double) to match the CPU reference
           kernel and the fused 3x2pt kernel; doing it at C precision would
           truncate float64 maps to float32 mid-computation. */
        const T ea_R = (T)exp_a.x;
        const T ea_I = (T)exp_a.y;
        const T eb_R = (T)exp_b.x;
        const T eb_I = (T)exp_b.y;

        T w_pair =
            weights[idx_a_bin] * weights[idx_b_bin];
        sum_w += (ACC)w_pair;

        const T a_R = ga1 * ea_R - ga2 * ea_I;
        const T a_I = ga1 * ea_I + ga2 * ea_R;
        const T b_R = gb1 * eb_R - gb2 * eb_I;
        const T b_I = gb1 * eb_I + gb2 * eb_R;

        /* xi+ numerator: Re[gamma'_b * conj(gamma'_a)] = b_R*a_R + b_I*a_I
           xi- numerator: Re[gamma'_b * gamma'_a]        = b_R*a_R - b_I*a_I */
        sum_p += (ACC)(w_pair * (b_R * a_R + b_I * a_I));
        sum_m += (ACC)(w_pair * (b_R * a_R - b_I * a_I));
    }

    /* Parallel reduction: sum contributions from all threads in this block */
    block_reduce_sum_triple<ACC>(sum_p, sum_m, sum_w, &sum_p, &sum_m, &sum_w);

    /* Thread 0 writes the final bin result to global memory.
       Output layout: [xi+ for all comb_ori | xi- for all comb_ori];
       out_den has one row per comb_ori (shared by xi+ and xi-) */
    if (lane == 0) {
        const long long out_p_idx =
            ((long long)comb_ori) * nbins_total + bin_flat;
        const long long out_m_idx =
            ((long long)(2 * ncomb + comb_ori)) * nbins_total + bin_flat;

        out_num[out_p_idx] = sum_p;
        out_num[out_m_idx] = sum_m;
        out_den[out_p_idx] = sum_w;
    }
}
