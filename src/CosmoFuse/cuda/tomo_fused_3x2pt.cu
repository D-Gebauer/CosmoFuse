/*
 * tomo_fused_3x2pt.cu -- Fused 3x2pt tomographic correlation kernel.
 *
 * Computes ALL six correlation outputs of a 3x2pt weak lensing analysis
 * in a single kernel launch, maximising GPU occupancy and minimising
 * kernel launch overhead.  The six outputs are:
 *
 *   z=0  Aperture mass  M_ap  -- tangential shear convolved with a
 *        compensated filter Q(theta) around each sky patch centre.
 *        Probes the projected mass within the aperture.
 *
 *   z=1  Galaxy mean density  M_g  -- galaxy overdensity delta_g convolved
 *        with the same compensated filter Q(theta).
 *        Probes the smoothed galaxy number density contrast.
 *
 *   z=2  Cosmic shear  xi+/xi-  -- shear-shear 2-point correlations.
 *        xi+ is sensitive to the sum of E- and B-mode power,
 *        xi- to their difference.
 *
 *   z=3  Galaxy clustering  xi_g  -- density-density 2-point correlation,
 *        measuring galaxy clustering strength vs angular separation.
 *
 *   z=4  Galaxy-galaxy lensing  xi_t  -- density-shear cross-correlation,
 *        measuring the tangential shear of source galaxies around
 *        foreground lens positions.
 *
 * Grid layout (2D, one launch per section):
 *   blockIdx.x = work item (patch index for z=0,1; angular bin for z=2,3,4)
 *   blockIdx.y = tomographic combination index
 *   threadIdx.x = parallel worker within the pair/pixel loop
 *
 * The correlation type selector z (0-4, see above) is passed as the
 * `section` launch argument: the caller launches the kernel once per
 * section with a grid sized exactly for that section, instead of one
 * dense max-sized 3D grid full of no-op blocks.  The five sections write
 * disjoint outputs, so the launches can run on concurrent streams.
 */

__COMMON_CUDA_SOURCE__

#include <cuComplex.h>

/*
 * QT -- scalar type of the aperture filter geometry (may be narrower than
 *       the map type T; promotion to T at use is exact for float -> double)
 */
template<typename T, typename C, typename I, typename QT, int N_DENSITY, int N_SHEAR>
__global__ void gpu_3x2pt_tomo_fused(
    /* --- Input maps (SoA layout, all tomo bins concatenated) --- */
    const T* density,        /* galaxy overdensity delta_g  [npix x N_DENSITY]      */
    const T* shear,          /* complex shear (gamma_1,gamma_2)  [npix x N_SHEAR x 2]    */
    const T* density_w,      /* density weights         [npix x N_DENSITY]      */
    const T* shear_w,        /* shear weights           [npix x N_SHEAR]        */
    /* --- Precomputed pair geometry --- */
    const I* ind_i,          /* pixel index of first pair member                     */
    const I* ind_j,          /* pixel index of second pair member                    */
    const C* rot_i,          /* e^{2iphi_i} rotation factor for pixel i                */
    const C* rot_j,          /* e^{2iphi_j} rotation factor for pixel j                */
    const long long* pair_offsets,  /* CSR offsets per angular bin                    */
    const long long nbins_total,
    const int npatches,
    const int npix,
    /* --- Aperture filter geometry (for M_ap and M_g) --- */
    const unsigned int* q_inds,   /* pixel indices within each patch's aperture  */
    const QT* q_cos,              /* cos(2phi) of pixel w.r.t. patch centre        */
    const QT* q_sin,              /* sin(2phi) of pixel w.r.t. patch centre        */
    const QT* q_val,              /* Q(theta): compensated filter value              */
    const long long* q_offsets,   /* CSR offsets per patch                        */
    const QT* q_patch_area,       /* solid angle of each patch (steradians)       */
    /* --- Tomographic bin combinations for each correlation type --- */
    const int* ss_comb_i,    /* shear-shear: tomo bin for side i            */
    const int* ss_comb_j,    /* shear-shear: tomo bin for side j            */
    const int n_ss_comb,
    const int* dd_comb_i,    /* density-density: tomo bin for side i         */
    const int* dd_comb_j,    /* density-density: tomo bin for side j         */
    const int n_dd_comb,
    const int* ds_comb_i,    /* density-shear: lens tomo bin                */
    const int* ds_comb_j,    /* density-shear: source tomo bin              */
    const int n_ds_comb,
    /* --- Output buffers (numerators and denominators) --- */
    T* out_ma_num,           /* aperture mass numerator   [n_shear x npatches]   */
    T* out_ma_den,           /* aperture mass denominator (sum of weights)        */
    T* out_mg_num,           /* galaxy mean density numerator [n_density x npatches] */
    T* out_mg_den,           /* galaxy mean density denominator                   */
    T* out_xip_num,          /* xi+ numerator                                     */
    T* out_xim_num,          /* xi- numerator                                     */
    T* out_xipm_den,         /* xi+/xi- shared denominator (sum of weight products) */
    T* out_xig_num,          /* xi_g numerator (galaxy clustering)                 */
    T* out_xig_den,          /* xi_g denominator                                  */
    T* out_xit_num,          /* xi_t numerator (galaxy-galaxy lensing)             */
    T* out_xit_den,          /* xi_t denominator                                  */
    const int section)       /* correlation type selector (0-4), one per launch */
{
    const int lane = (int)threadIdx.x;
    const long long x = (long long)blockIdx.x;  /* patch or angular bin index */
    const int y = (int)blockIdx.y;               /* tomo combination index     */
    const int z = section;                       /* correlation type selector  */

    /* ================================================================
     * z=0 : Aperture mass M_ap(patch, tomo_bin)
     *
     * M_ap = A_patch * Sum_pix [ w * gamma_t * Q(theta) ] / Sum_pix [ w ]
     * where gamma_t = -gamma_1*cos(2phi) - gamma_2*sin(2phi) is the tangential shear
     * around the patch centre, and Q(theta) is the compensated aperture
     * filter.  The numerator and denominator are stored separately so
     * the caller can normalise after the kernel.
     * ================================================================ */
    if (z == 0) {
        if (x >= npatches || y >= N_SHEAR) return;
        const long long start = q_offsets[x];
        const long long stop = q_offsets[x + 1];
        T sum_num = (T)0.0;
        T sum_den = (T)0.0;
        for (long long idx = start + lane; idx < stop; idx += BLOCK_SIZE) {
            const unsigned int pix = q_inds[idx];
            const long long shear_idx = ((long long)pix * (long long)N_SHEAR + (long long)y) * 2LL;
            const long long w_idx = (long long)pix * (long long)N_SHEAR + (long long)y;
            const T g1 = shear[shear_idx];
            const T g2 = shear[shear_idx + 1LL];
            const T wv = shear_w[w_idx];
            /* Tangential shear w.r.t. patch centre */
            const T gt = -g1 * (T)q_cos[idx] - g2 * (T)q_sin[idx];
            sum_num += wv * gt * (T)q_val[idx];
            sum_den += wv;
        }
        block_reduce_sum_pair(sum_num, sum_den, &sum_num, &sum_den);
        if (lane == 0) {
            const long long out_idx = (long long)y * (long long)npatches + x;
            out_ma_num[out_idx] = (T)q_patch_area[x] * sum_num;
            out_ma_den[out_idx] = sum_den;
        }
        return;
    }

    /* ================================================================
     * z=1 : Galaxy mean density M_g(patch, tomo_bin)
     *
     * M_g = A_patch * Sum_pix [ w * delta_g * Q(theta) ] / Sum_pix [ w ]
     * Smoothed galaxy overdensity within the aperture, used as the
     * "central" field in i3PCF measurements.
     * ================================================================ */
    if (z == 1) {
        if (x >= npatches || y >= N_DENSITY) return;
        const long long start = q_offsets[x];
        const long long stop = q_offsets[x + 1];
        T sum_num = (T)0.0;
        T sum_den = (T)0.0;
        for (long long idx = start + lane; idx < stop; idx += BLOCK_SIZE) {
            const unsigned int pix = q_inds[idx];
            const long long d_idx = (long long)pix * (long long)N_DENSITY + (long long)y;
            const T wv = density_w[d_idx];
            sum_num += wv * density[d_idx] * (T)q_val[idx];
            sum_den += wv;
        }
        block_reduce_sum_pair(sum_num, sum_den, &sum_num, &sum_den);
        if (lane == 0) {
            const long long out_idx = (long long)y * (long long)npatches + x;
            out_mg_num[out_idx] = (T)q_patch_area[x] * sum_num;
            out_mg_den[out_idx] = sum_den;
        }
        return;
    }

    /* ================================================================
     * z=2 : Cosmic shear xi+(theta) and xi-(theta)
     *
     * Rotates shears into the pair frame gamma' = gamma * e^{2iphi}, then:
     *   xi+ = Re[gamma'_b * conj(gamma'_a)]  (E+B mode power)
     *   xi- = Re[gamma'_b * gamma'_a]         (E-B mode power)
     *
     * Both orientations (A->B, B->A) are computed for cross-bin pairs.
     * ================================================================ */
    if (z == 2) {
        if (x >= nbins_total || y >= (2 * n_ss_comb)) return;
        const int comb_idx = y >> 1;
        const int ori = y & 1;   /* 0 = A->B, 1 = B->A */
        const int i = ss_comb_i[comb_idx];
        const int j = ss_comb_j[comb_idx];
        if (ori == 1 && i == j) return;

        int ai = i;
        int bj = j;
        if (ori == 1 && i != j) {
            ai = j;
            bj = i;
        }

        const long long start = pair_offsets[x];
        const long long stop = pair_offsets[x + 1];
        T sum_p = (T)0.0;
        T sum_m = (T)0.0;
        T sum_w = (T)0.0;
        for (long long idx = start + lane; idx < stop; idx += BLOCK_SIZE) {
            const long long pix_a = (long long)ind_i[idx];
            const long long pix_b = (long long)ind_j[idx];
            const C ex_a = rot_i[idx];
            const C ex_b = rot_j[idx];

            /* Fetch gamma = (gamma_1, gamma_2) for each pixel in their respective tomo bins */
            const long long a_base = ((pix_a * (long long)N_SHEAR + (long long)ai) * 2LL);
            const long long b_base = ((pix_b * (long long)N_SHEAR + (long long)bj) * 2LL);
            const T ga1 = shear[a_base];
            const T ga2 = shear[a_base + 1LL];
            const T gb1 = shear[b_base];
            const T gb2 = shear[b_base + 1LL];

            /* Rotate into pair frame: gamma' = (gamma_1 + igamma_2) * e^{2iphi}
               Expanded as real/imag parts to avoid complex arithmetic */
            const T a_r = ga1 * ex_a.x - ga2 * ex_a.y;
            const T a_i = ga1 * ex_a.y + ga2 * ex_a.x;
            const T b_r = gb1 * ex_b.x - gb2 * ex_b.y;
            const T b_i = gb1 * ex_b.y + gb2 * ex_b.x;

            const long long wa_idx = pix_a * (long long)N_SHEAR + (long long)ai;
            const long long wb_idx = pix_b * (long long)N_SHEAR + (long long)bj;
            const T wv = shear_w[wa_idx] * shear_w[wb_idx];
            sum_w += wv;
            sum_p += wv * (b_r * a_r + b_i * a_i);  /* xi+ */
            sum_m += wv * (b_r * a_r - b_i * a_i);  /* xi- */
        }
        block_reduce_sum_triple(sum_p, sum_m, sum_w, &sum_p, &sum_m, &sum_w);
        if (lane == 0) {
            const long long out_idx = (long long)y * nbins_total + x;
            out_xip_num[out_idx] = sum_p;
            out_xim_num[out_idx] = sum_m;
            out_xipm_den[out_idx] = sum_w;
        }
        return;
    }

    /* ================================================================
     * z=3 : Galaxy clustering xi_g(theta)
     *
     * xi_g = Sum w_a * w_b * delta_a * delta_b  /  Sum w_a * w_b
     * Measures the angular galaxy auto-correlation.
     * ================================================================ */
    if (z == 3) {
        if (x >= nbins_total || y >= (2 * n_dd_comb)) return;
        const int comb_idx = y >> 1;
        const int ori = y & 1;
        const int i = dd_comb_i[comb_idx];
        const int j = dd_comb_j[comb_idx];
        if (ori == 1 && i == j) return;

        int ai = i;
        int bj = j;
        if (ori == 1 && i != j) {
            ai = j;
            bj = i;
        }

        const long long start = pair_offsets[x];
        const long long stop = pair_offsets[x + 1];
        T sum_num = (T)0.0;
        T sum_den = (T)0.0;
        for (long long idx = start + lane; idx < stop; idx += BLOCK_SIZE) {
            const long long pix_a = (long long)ind_i[idx];
            const long long pix_b = (long long)ind_j[idx];
            const long long ia = pix_a * (long long)N_DENSITY + (long long)ai;
            const long long jb = pix_b * (long long)N_DENSITY + (long long)bj;
            const T wv = density_w[ia] * density_w[jb];
            sum_den += wv;
            sum_num += wv * density[ia] * density[jb];
        }
        block_reduce_sum_pair(sum_num, sum_den, &sum_num, &sum_den);
        if (lane == 0) {
            const long long out_idx = (long long)y * nbins_total + x;
            out_xig_num[out_idx] = sum_num;
            out_xig_den[out_idx] = sum_den;
        }
        return;
    }

    /* ================================================================
     * z=4 : Galaxy-galaxy lensing xi_t(theta)
     *
     * xi_t = Sum w_lens * w_source * delta_lens * gamma_t  /  Sum w_lens * w_source
     * where gamma_t = -gamma_1*cos(2phi) + gamma_2*sin(2phi) is the tangential shear
     * of the source around the lens position.
     *
     * Both pair orientations (A->B and B->A) are accumulated to use
     * all lens-source information from each pair.
     * ================================================================ */
    if (z == 4) {
        if (x >= nbins_total || y >= n_ds_comb) return;
        const int lens_bin = ds_comb_i[y];
        const int source_bin = ds_comb_j[y];
        const long long start = pair_offsets[x];
        const long long stop = pair_offsets[x + 1];
        T sum_num = (T)0.0;
        T sum_den = (T)0.0;
        for (long long idx = start + lane; idx < stop; idx += BLOCK_SIZE) {
            const long long pix_a = (long long)ind_i[idx];
            const long long pix_b = (long long)ind_j[idx];

            /* A->B: pixel a = lens, pixel b = source */
            const long long lens_ab = pix_a * (long long)N_DENSITY + (long long)lens_bin;
            const long long src_ab = pix_b * (long long)N_SHEAR + (long long)source_bin;
            const long long src_ab_base = src_ab * 2LL;
            const C ex_ab = rot_j[idx];
            const T gt_ab = -shear[src_ab_base] * ex_ab.x + shear[src_ab_base + 1LL] * ex_ab.y;
            const T w_ab = density_w[lens_ab] * shear_w[src_ab];
            sum_num += w_ab * density[lens_ab] * gt_ab;
            sum_den += w_ab;

            /* B->A: pixel b = lens, pixel a = source */
            const long long lens_ba = pix_b * (long long)N_DENSITY + (long long)lens_bin;
            const long long src_ba = pix_a * (long long)N_SHEAR + (long long)source_bin;
            const long long src_ba_base = src_ba * 2LL;
            const C ex_ba = rot_i[idx];
            const T gt_ba = -shear[src_ba_base] * ex_ba.x + shear[src_ba_base + 1LL] * ex_ba.y;
            const T w_ba = density_w[lens_ba] * shear_w[src_ba];
            sum_num += w_ba * density[lens_ba] * gt_ba;
            sum_den += w_ba;
        }
        block_reduce_sum_pair(sum_num, sum_den, &sum_num, &sum_den);
        if (lane == 0) {
            const long long out_idx = (long long)y * nbins_total + x;
            out_xit_num[out_idx] = sum_num;
            out_xit_den[out_idx] = sum_den;
        }
        return;
    }
}
