/*
 * tomo_fused_3x2pt.cu -- Aperture sections of the fused 3x2pt tomographic
 *                        measurement.
 *
 * `Correlation.get_3x2pt_tomo` computes all six outputs of a 3x2pt analysis
 * from one set of device inputs (AoS layout, all tomo bins of a pixel
 * contiguous):
 *
 *   M_ap   aperture mass: tangential shear convolved with the compensated
 *          filter Q(theta) around each patch centre        (this file, z=0)
 *   M_g    aperture galaxy density: delta_g convolved with Q  (this file, z=1)
 *   xi+/-  cosmic shear        (tomo_vectorized_xipm.cu, tiled kernels)
 *   xi_g   galaxy clustering   (density_density_tomo_vectorized.cu)
 *   xi_t   galaxy-galaxy lensing (density_shear_tomo_vectorized.cu)
 *
 * The pair statistics used to be sections 2-4 of this kernel, one thread
 * block per (angular bin, combination, orientation); they now run in the
 * combination-tiled pair kernels shared with the standalone tomographic
 * methods (each pair visited once, packed geometry supported).  The
 * aperture sections stay here because they accumulate at ACC precision
 * (the standalone aperture kernels in aperture_tomo.cu accumulate at the
 * map precision, which the default path keeps for compatibility).
 *
 * Grid layout (one launch per section, exactly-sized grid):
 *   blockIdx.x  = patch index
 *   blockIdx.y  = tomographic bin        (gpu_3x2pt_tomo_aperture)
 *   threadIdx.x = aperture pixel (strided loop)
 *
 * `gpu_3x2pt_tomo_aperture_fused` is the default: one block per patch with
 * the tomographic bins looped inside it, so the disc geometry (q_inds +
 * q_cos + q_sin + q_val, 16 B per disc pixel) is read once per pixel
 * rather than once per bin.  Same restructure, same reasoning and the same
 * bitwise guarantee as the standalone kernels in aperture_tomo.cu -- see
 * that file's header.  The per-(patch, bin) kernel stays as the fallback
 * for bin counts beyond the accumulator budget.
 */

__COMMON_CUDA_SOURCE__


/*
 * QT  -- scalar type of the aperture filter geometry (may be narrower than
 *        the map type T; promotion to T at use is exact for float -> double)
 * ACC -- accumulator/output type (double for float32 maps with
 *        accumulation_precision="float64"; otherwise same as T)
 */
template<typename T, typename QT, int N_DENSITY, int N_SHEAR, typename ACC>
__global__ void gpu_3x2pt_tomo_aperture(
    const T* density,        /* galaxy overdensity delta_g  [npix x N_DENSITY]   */
    const T* shear,          /* complex shear (gamma_1,gamma_2) [npix x N_SHEAR x 2] */
    const T* density_w,      /* density weights [npix x N_DENSITY]            */
    const T* shear_w,        /* shear weights   [npix x N_SHEAR]              */
    const int npatches,
    const unsigned int* q_inds,   /* pixel indices within each patch's aperture */
    const QT* q_cos,              /* cos(2phi) of pixel w.r.t. patch centre     */
    const QT* q_sin,              /* sin(2phi) of pixel w.r.t. patch centre     */
    const QT* q_val,              /* Q(theta): compensated filter value          */
    const long long* q_offsets,   /* CSR offsets per patch                       */
    const QT* q_patch_area,       /* solid angle of each patch (steradians)      */
    ACC* out_ma_num,           /* aperture mass numerator   [n_shear x npatches]   */
    ACC* out_ma_den,           /* aperture mass denominator (sum of weights)        */
    ACC* out_mg_num,           /* galaxy density numerator [n_density x npatches]   */
    ACC* out_mg_den,           /* galaxy density denominator                        */
    const int section)         /* 0 = M_ap, 1 = M_g                                */
{
    const int lane = (int)threadIdx.x;
    const long long x = (long long)blockIdx.x;  /* patch index     */
    const int y = (int)blockIdx.y;               /* tomographic bin */

    /* z=0 : M_ap = A_patch * Sum_pix [ w * gamma_t * Q ] / Sum_pix [ w ],
       gamma_t = -gamma_1 cos(2phi) - gamma_2 sin(2phi) around the centre */
    if (section == 0) {
        if (x >= npatches || y >= N_SHEAR) return;
        const long long start = q_offsets[x];
        const long long stop = q_offsets[x + 1];
        ACC sum_num = (ACC)0.0;
        ACC sum_den = (ACC)0.0;
        for (long long idx = start + lane; idx < stop; idx += BLOCK_SIZE) {
            const unsigned int pix = q_inds[idx];
            const long long shear_idx = ((long long)pix * (long long)N_SHEAR + (long long)y) * 2LL;
            const long long w_idx = (long long)pix * (long long)N_SHEAR + (long long)y;
            const T g1 = shear[shear_idx];
            const T g2 = shear[shear_idx + 1LL];
            const T wv = shear_w[w_idx];
            const T gt = -g1 * (T)q_cos[idx] - g2 * (T)q_sin[idx];
            sum_num += (ACC)(wv * gt * (T)q_val[idx]);
            sum_den += (ACC)wv;
        }
        block_reduce_sum_pair(sum_num, sum_den, &sum_num, &sum_den);
        if (lane == 0) {
            const long long out_idx = (long long)y * (long long)npatches + x;
            out_ma_num[out_idx] = (ACC)q_patch_area[x] * sum_num;
            out_ma_den[out_idx] = sum_den;
        }
        return;
    }

    /* z=1 : M_g = A_patch * Sum_pix [ w * delta_g * Q ] / Sum_pix [ w ] */
    if (section == 1) {
        if (x >= npatches || y >= N_DENSITY) return;
        const long long start = q_offsets[x];
        const long long stop = q_offsets[x + 1];
        ACC sum_num = (ACC)0.0;
        ACC sum_den = (ACC)0.0;
        for (long long idx = start + lane; idx < stop; idx += BLOCK_SIZE) {
            const unsigned int pix = q_inds[idx];
            const long long d_idx = (long long)pix * (long long)N_DENSITY + (long long)y;
            const T wv = density_w[d_idx];
            sum_num += (ACC)(wv * density[d_idx] * (T)q_val[idx]);
            sum_den += (ACC)wv;
        }
        block_reduce_sum_pair(sum_num, sum_den, &sum_num, &sum_den);
        if (lane == 0) {
            const long long out_idx = (long long)y * (long long)npatches + x;
            out_mg_num[out_idx] = (ACC)q_patch_area[x] * sum_num;
            out_mg_den[out_idx] = sum_den;
        }
        return;
    }
}


/*
 * One block per patch, the tomographic bins looped inside it.  Bitwise
 * identical to gpu_3x2pt_tomo_aperture: only the bin loop moves inside
 * the block, the thread->pixel mapping and the reduction tree are
 * unchanged.  N_SHEAR / N_DENSITY are already compile-time constants
 * here, so the per-thread accumulator arrays stay in registers.
 */
template<typename T, typename QT, int N_DENSITY, int N_SHEAR, typename ACC>
__global__ void gpu_3x2pt_tomo_aperture_fused(
    const T* density,
    const T* shear,
    const T* density_w,
    const T* shear_w,
    const int npatches,
    const unsigned int* q_inds,
    const QT* q_cos,
    const QT* q_sin,
    const QT* q_val,
    const long long* q_offsets,
    const QT* q_patch_area,
    ACC* out_ma_num,
    ACC* out_ma_den,
    ACC* out_mg_num,
    ACC* out_mg_den,
    const int section)
{
    const int lane = (int)threadIdx.x;
    const long long x = (long long)blockIdx.x;  /* patch index */
    if (x >= (long long)npatches) return;

    const long long start = q_offsets[x];
    const long long stop = q_offsets[x + 1];
    /* One buffer pair for every bin of either section (ACC is the wider
       of the two types in play, so the allocation covers both). */
    __shared__ ACC s1[BLOCK_SIZE];
    __shared__ ACC s2[BLOCK_SIZE];
    const ACC area = (ACC)q_patch_area[x];

    /* z=0 : M_ap = A_patch * Sum_pix [ w * gamma_t * Q ] / Sum_pix [ w ],
       gamma_t = -gamma_1 cos(2phi) - gamma_2 sin(2phi) around the centre */
    if (section == 0) {
        ACC sn[N_SHEAR];
        ACC sd[N_SHEAR];
#pragma unroll
        for (int y = 0; y < N_SHEAR; ++y) { sn[y] = (ACC)0.0; sd[y] = (ACC)0.0; }

        for (long long idx = start + lane; idx < stop; idx += BLOCK_SIZE) {
            const unsigned int pix = q_inds[idx];
            const T qc = (T)q_cos[idx];
            const T qs = (T)q_sin[idx];
            const T qv = (T)q_val[idx];
#pragma unroll
            for (int y = 0; y < N_SHEAR; ++y) {
                const long long shear_idx = ((long long)pix * (long long)N_SHEAR + (long long)y) * 2LL;
                const long long w_idx = (long long)pix * (long long)N_SHEAR + (long long)y;
                const T g1 = shear[shear_idx];
                const T g2 = shear[shear_idx + 1LL];
                const T wv = shear_w[w_idx];
                const T gt = -g1 * qc - g2 * qs;
                sn[y] += (ACC)(wv * gt * qv);
                sd[y] += (ACC)wv;
            }
        }
#pragma unroll
        for (int y = 0; y < N_SHEAR; ++y) {
            ACC n, d;
            __syncthreads();            /* the buffers are being reused */
            block_reduce_sum_pair_into(sn[y], sd[y], s1, s2, &n, &d);
            if (lane == 0) {
                const long long out_idx = (long long)y * (long long)npatches + x;
                out_ma_num[out_idx] = area * n;
                out_ma_den[out_idx] = d;
            }
        }
        return;
    }

    /* z=1 : M_g = A_patch * Sum_pix [ w * delta_g * Q ] / Sum_pix [ w ] */
    if (section == 1) {
        ACC sn[N_DENSITY];
        ACC sd[N_DENSITY];
#pragma unroll
        for (int y = 0; y < N_DENSITY; ++y) { sn[y] = (ACC)0.0; sd[y] = (ACC)0.0; }

        for (long long idx = start + lane; idx < stop; idx += BLOCK_SIZE) {
            const unsigned int pix = q_inds[idx];
            const T qv = (T)q_val[idx];
#pragma unroll
            for (int y = 0; y < N_DENSITY; ++y) {
                const long long d_idx = (long long)pix * (long long)N_DENSITY + (long long)y;
                const T wv = density_w[d_idx];
                sn[y] += (ACC)(wv * density[d_idx] * qv);
                sd[y] += (ACC)wv;
            }
        }
#pragma unroll
        for (int y = 0; y < N_DENSITY; ++y) {
            ACC n, d;
            __syncthreads();
            block_reduce_sum_pair_into(sn[y], sd[y], s1, s2, &n, &d);
            if (lane == 0) {
                const long long out_idx = (long long)y * (long long)npatches + x;
                out_mg_num[out_idx] = area * n;
                out_mg_den[out_idx] = d;
            }
        }
        return;
    }
}
