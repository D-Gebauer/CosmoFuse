/*
 * density_shear_tomo_vectorized.cu -- Tomographic galaxy-galaxy lensing
 *                                     correlation xi_t(theta).
 *
 * Measures the mean tangential shear gamma_t of background (source) galaxies
 * around foreground (lens) galaxy positions, weighted by the lens
 * overdensity delta_g:
 *
 *   xi_t(theta) = <delta_g,lens * gamma_t,source>(theta)
 *
 * The tangential shear gamma_t is the projection of the source shear onto
 * the direction perpendicular to the line connecting the lens-source pair:
 *
 *   gamma_t = -Re[gamma * e^{-2iphi}] = -gamma_1*cos(2phi) - gamma_2*sin(2phi)
 *
 * where phi is the position angle of the pair.  This probes the
 * galaxy-matter cross-power spectrum.
 *
 * Each pair contributes in both orientations (A as lens, B as source
 * AND B as lens, A as source) to fully use all pair information.
 *
 * Grid layout:
 *   blockIdx.x  = angular separation bin  (0 .. nbins_total-1)
 *   blockIdx.y  = lensxsource tomographic bin combination
 *   threadIdx.x = pair index within the bin (strided loop)
 */

__COMMON_CUDA_SOURCE__


/*
 * T               -- scalar type (float / double)
 * C               -- complex type for rotation factors
 * LENS_TOMO_BINS  -- number of lens (foreground) tomographic bins
 * SOURCE_TOMO_BINS -- number of source (background) tomographic bins
 * ACC             -- accumulator/output type (double for float32 maps with
 *                   accumulation_precision="float64"; otherwise same as T)
 */
template<typename T, typename C, int LENS_TOMO_BINS, int SOURCE_TOMO_BINS, typename I, typename ACC>
__global__ void gpu_fused_tomo_reduce_ds(
    const T* density,         /* lens galaxy overdensity delta_g             */
    const T* shear,           /* source shear (gamma_1, gamma_2) interleaved      */
    const T* lens_weights,    /* per-pixel lens weights                  */
    const T* source_weights,  /* per-pixel source weights                */
    const I* ind_i,           /* pixel index of first pair member        */
    const I* ind_j,           /* pixel index of second pair member       */
    const C* rot_i,           /* rotation factor e^{2iphi} at pixel i      */
    const C* rot_j,           /* rotation factor e^{2iphi} at pixel j      */
    const long long* bin_offsets,
    const int* comb_i,        /* lens tomo bin for each combination      */
    const int* comb_j,        /* source tomo bin for each combination    */
    ACC* out_num,             /* output: weighted delta_g*gamma_t numerators     */
    ACC* out_den,             /* output: A->B plus B->A weight sums          */
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

    ACC sum_val = (ACC)0.0;
    ACC sum_w = (ACC)0.0;

    for (long long tid = start + lane; tid < stop; tid += BLOCK_SIZE) {
        const long long idx_a = (long long)ind_i[tid];
        const long long idx_b = (long long)ind_j[tid];
        const C rot_ab = rot_j[tid];   /* rotation for A->B direction */
        const C rot_ba = rot_i[tid];   /* rotation for B->A direction */

        /* --- A->B: pixel a is lens, pixel b is source --- */
        const long long lens_idx_ab = idx_a * (long long)LENS_TOMO_BINS + lens_bin;
        const long long source_idx_ab = idx_b * (long long)SOURCE_TOMO_BINS + source_bin;
        const long long shear_base_ab = source_idx_ab * 2;

        /* Tangential shear: gamma_t = -gamma_1*cos(2phi) + gamma_2*sin(2phi)
           using rot.x = cos(2phi), rot.y = sin(2phi) */
        const T gamma_t_ab = (
            -shear[shear_base_ab] * rot_ab.x
            + shear[shear_base_ab + 1] * rot_ab.y
        );

        const T w_ab = lens_weights[lens_idx_ab] * source_weights[source_idx_ab];
        sum_w += (ACC)w_ab;
        sum_val += (ACC)(w_ab * density[lens_idx_ab] * gamma_t_ab);

        /* --- B->A: pixel b is lens, pixel a is source --- */
        const long long lens_idx_ba = idx_b * (long long)LENS_TOMO_BINS + lens_bin;
        const long long source_idx_ba = idx_a * (long long)SOURCE_TOMO_BINS + source_bin;
        const long long shear_base_ba = source_idx_ba * 2;

        const T gamma_t_ba = (
            -shear[shear_base_ba] * rot_ba.x
            + shear[shear_base_ba + 1] * rot_ba.y
        );

        const T w_ba = lens_weights[lens_idx_ba] * source_weights[source_idx_ba];
        sum_w += (ACC)w_ba;
        sum_val += (ACC)(w_ba * density[lens_idx_ba] * gamma_t_ba);
    }

    block_reduce_sum_pair<ACC>(sum_val, sum_w, &sum_val, &sum_w);

    if (lane == 0) {
        const long long out_idx =
            ((long long)comb_idx) * nbins_total + bin_flat;
        out_num[out_idx] = sum_val;
        out_den[out_idx] = sum_w;
    }
}
