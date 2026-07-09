/*
 * aperture_tomo.cu -- Block-reduced aperture statistics for all
 * tomographic bins in one launch.  Grid: blockIdx.x = patch,
 * blockIdx.y = tomo bin; threadIdx.x strides the aperture pixels.
 * Replaces the per-pixel ElementwiseKernel + add.reduceat path
 * (two npixels_in_apertures-sized temporaries per call).
 *
 * Stride contract: g1/g2/values/weights are base pointers of 2D views
 * whose innermost dimension is contiguous; the caller passes the row
 * stride (in elements) explicitly so strided views such as
 * shear[:, 0] of a (nz, 2, npix) array can be used without a copy.
 *
 * T  -- scalar type of the maps/weights (float / double)
 * QT -- scalar type of the aperture filter geometry (may be narrower
 *       than T; promotion to T at use is exact for float -> double)
 */

__COMMON_CUDA_SOURCE__

template<typename T, typename QT>
__global__ void gpu_aperture_shear_tomo(
    const T* g1,                 /* base ptr, row b at g1 + b*g_stride  */
    const T* g2,
    const long long g_stride,    /* elements between tomo rows (2*npix for
                                    (nz,2,npix) views, npix for planar)  */
    const T* weights,            /* base ptr, row b at weights + b*w_stride */
    const long long w_stride,
    const unsigned int* q_inds,
    const QT* q_cos,
    const QT* q_sin,
    const QT* q_val,
    const long long* q_offsets,
    const QT* q_patch_area,
    T* out_num,                  /* [ntomo x npatches] */
    T* out_den,
    const int npatches,
    const int ntomo)
{
    const int lane = (int)threadIdx.x;
    const int patch = (int)blockIdx.x;
    const int bin = (int)blockIdx.y;
    if (patch >= npatches || bin >= ntomo) return;

    const T* g1b = g1 + (long long)bin * g_stride;
    const T* g2b = g2 + (long long)bin * g_stride;
    const T* wb  = weights + (long long)bin * w_stride;

    const long long start = q_offsets[patch];
    const long long stop  = q_offsets[patch + 1];
    T sum_num = (T)0.0;
    T sum_den = (T)0.0;
    for (long long idx = start + lane; idx < stop; idx += BLOCK_SIZE) {
        const unsigned int pix = q_inds[idx];
        const T wv = wb[pix];
        /* Tangential shear w.r.t. the patch centre */
        const T gt = -g1b[pix] * (T)q_cos[idx] - g2b[pix] * (T)q_sin[idx];
        sum_num += wv * gt * (T)q_val[idx];
        sum_den += wv;
    }
    block_reduce_sum_pair(sum_num, sum_den, &sum_num, &sum_den);
    if (lane == 0) {
        const long long o = (long long)bin * npatches + patch;
        out_num[o] = (T)q_patch_area[patch] * sum_num;
        out_den[o] = sum_den;
    }
}

template<typename T, typename QT>
__global__ void gpu_aperture_density_tomo(
    const T* values,
    const long long v_stride,
    const T* weights,
    const long long w_stride,
    const unsigned int* q_inds,
    const QT* q_val,
    const long long* q_offsets,
    const QT* q_patch_area,
    T* out_num,
    T* out_den,
    const int npatches,
    const int ntomo)
{
    const int lane = (int)threadIdx.x;
    const int patch = (int)blockIdx.x;
    const int bin = (int)blockIdx.y;
    if (patch >= npatches || bin >= ntomo) return;

    const T* vb = values + (long long)bin * v_stride;
    const T* wb = weights + (long long)bin * w_stride;

    const long long start = q_offsets[patch];
    const long long stop  = q_offsets[patch + 1];
    T sum_num = (T)0.0;
    T sum_den = (T)0.0;
    for (long long idx = start + lane; idx < stop; idx += BLOCK_SIZE) {
        const unsigned int pix = q_inds[idx];
        const T wv = wb[pix];
        sum_num += wv * vb[pix] * (T)q_val[idx];
        sum_den += wv;
    }
    block_reduce_sum_pair(sum_num, sum_den, &sum_num, &sum_den);
    if (lane == 0) {
        const long long o = (long long)bin * npatches + patch;
        out_num[o] = (T)q_patch_area[patch] * sum_num;
        out_den[o] = sum_den;
    }
}
