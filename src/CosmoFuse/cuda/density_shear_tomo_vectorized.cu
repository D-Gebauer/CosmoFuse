/*
 * density_shear_tomo_vectorized.cu — Tomographic galaxy-galaxy lensing
 *                                     correlation ξ_t(θ).
 *
 * Measures the mean tangential shear γ_t of background (source) galaxies
 * around foreground (lens) galaxy positions, weighted by the lens
 * overdensity δ_g:
 *
 *   ξ_t(θ) = ⟨δ_g,lens · γ_t,source⟩(θ)
 *
 * The tangential shear γ_t is the projection of the source shear onto
 * the direction perpendicular to the line connecting the lens-source pair:
 *
 *   γ_t = -Re[γ · e^{-2iφ}] = -γ₁·cos(2φ) - γ₂·sin(2φ)
 *
 * where φ is the position angle of the pair.  This probes the
 * galaxy-matter cross-power spectrum.
 *
 * Each pair contributes in both orientations (A as lens, B as source
 * AND B as lens, A as source) to fully use all pair information.
 *
 * Grid layout:
 *   blockIdx.x  = angular separation bin  (0 .. nbins_total-1)
 *   blockIdx.y  = lens×source tomographic bin combination
 *   threadIdx.x = pair index within the bin (strided loop)
 */

__COMMON_CUDA_SOURCE__

#include <cuComplex.h>

/*
 * T               — scalar type (float / double)
 * C               — complex type for rotation factors
 * LENS_TOMO_BINS  — number of lens (foreground) tomographic bins
 * SOURCE_TOMO_BINS — number of source (background) tomographic bins
 */
template<typename T, typename C, int LENS_TOMO_BINS, int SOURCE_TOMO_BINS, typename I>
__global__ void gpu_fused_tomo_reduce_ds(
    const T* density,         /* lens galaxy overdensity δ_g             */
    const T* shear,           /* source shear (γ₁, γ₂) interleaved      */
    const T* lens_weights,    /* per-pixel lens weights                  */
    const T* source_weights,  /* per-pixel source weights                */
    const I* ind_i,           /* pixel index of first pair member        */
    const I* ind_j,           /* pixel index of second pair member       */
    const C* rot_i,           /* rotation factor e^{2iφ} at pixel i      */
    const C* rot_j,           /* rotation factor e^{2iφ} at pixel j      */
    const long long* bin_offsets,
    const int* comb_i,        /* lens tomo bin for each combination      */
    const int* comb_j,        /* source tomo bin for each combination    */
    T* out_num,               /* output: weighted δ_g·γ_t numerators     */
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
        const long long idx_a = (long long)ind_i[tid];
        const long long idx_b = (long long)ind_j[tid];
        const C rot_ab = rot_j[tid];   /* rotation for A→B direction */
        const C rot_ba = rot_i[tid];   /* rotation for B→A direction */

        /* --- A→B: pixel a is lens, pixel b is source --- */
        const long long lens_idx_ab = idx_a * (long long)LENS_TOMO_BINS + lens_bin;
        const long long source_idx_ab = idx_b * (long long)SOURCE_TOMO_BINS + source_bin;
        const long long shear_base_ab = source_idx_ab * 2;

        /* Tangential shear: γ_t = -γ₁·cos(2φ) + γ₂·sin(2φ)
           using rot.x = cos(2φ), rot.y = sin(2φ) */
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

        /* --- B→A: pixel b is lens, pixel a is source --- */
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
