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

#include <cuComplex.h>

/* --- Complex number helpers for precision-generic code --- */

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

/*
 * T            -- scalar type (float / double) for map values and weights
 * C            -- complex type (cuFloatComplex / cuDoubleComplex) for rotations
 * TOMO_BINS_T  -- number of tomographic redshift bins (compile-time constant
 *                for efficient index arithmetic)
 */
template<typename T, typename C, int TOMO_BINS_T, typename I>
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
    T* out_num,              /* output numerators [xi+ block | xi- block]    */
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

    T sum_p = (T)0.0;   /* accumulator for xi+ numerator */
    T sum_m = (T)0.0;   /* accumulator for xi- numerator */

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

        const C g_a = complex_make<C>((double)shear[base_a], (double)shear[base_a + 1]);
        const C g_b = complex_make<C>((double)shear[base_b], (double)shear[base_b + 1]);

        /* Rotate shears into the pair frame: gamma' = gamma * e^{2iphi} */
        C term_a = complex_mul<C>(g_a, exp_a);
        C term_b = complex_mul<C>(g_b, exp_b);

        T w_pair =
            weights[idx_a_bin] * weights[idx_b_bin];

        const T a_R = (T)term_a.x;
        const T a_I = (T)term_a.y;
        const T b_R = (T)term_b.x;
        const T b_I = (T)term_b.y;

        /* xi+ numerator: Re[gamma'_b * conj(gamma'_a)] = b_R*a_R + b_I*a_I
           xi- numerator: Re[gamma'_b * gamma'_a]        = b_R*a_R - b_I*a_I */
        sum_p += w_pair * (b_R * a_R + b_I * a_I);
        sum_m += w_pair * (b_R * a_R - b_I * a_I);
    }

    /* Parallel reduction: sum contributions from all threads in this block */
    block_reduce_sum_pair<T>(sum_p, sum_m, &sum_p, &sum_m);

    /* Thread 0 writes the final bin result to global memory.
       Output layout: [xi+ for all comb_ori | xi- for all comb_ori] */
    if (lane == 0) {
        const long long out_p_idx =
            ((long long)comb_ori) * nbins_total + bin_flat;
        const long long out_m_idx =
            ((long long)(2 * ncomb + comb_ori)) * nbins_total + bin_flat;

        out_num[out_p_idx] = sum_p;
        out_num[out_m_idx] = sum_m;
    }
}
