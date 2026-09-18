/*
 * aperture_tomo.cu -- Block-reduced aperture statistics for all
 * tomographic bins in one launch.  Replaces the per-pixel
 * ElementwiseKernel + add.reduceat path (two npixels_in_apertures-sized
 * temporaries per call).
 *
 * Two kernels per statistic, one launch contract:
 *
 *   gpu_aperture_*_tomo         one block per (patch, tomo bin);
 *                               grid (npatches, ntomo).  The fallback.
 *   gpu_aperture_*_tomo_fused   one block per patch, the NZ bins looped
 *                               inside it; grid (npatches,).  The default.
 *
 * The per-(patch, bin) form re-reads the disc geometry -- q_inds + q_cos +
 * q_sin + q_val, 16 B per disc pixel -- once per tomographic bin while
 * using 12 B of map data (g1, g2, w), so it moves 28*nz B/pixel where one
 * block per patch needs 16 + 12*nz: 112 vs 64 at nz = 4.  L2 does not
 * catch the reuse (measured: benchmarks/static_treecode/APERTURE_RESULTS.md),
 * and the fused form duly runs 1.75x faster at nside 512.
 *
 * The fused form is *bitwise* identical, which it has to be --
 * resolution_factor=None must stay bit-for-bit identical to 4.20.0 and
 * this kernel is on that path.  Only the bin loop moves inside the block:
 * the thread->pixel mapping, BLOCK_SIZE and the reduction tree of
 * block_reduce_sum_pair(_into) are unchanged, so each bin sums exactly the
 * same partials in exactly the same order.  Changing the stride or
 * BLOCK_SIZE would leave that regime.
 *
 * NZ is a template parameter, not a run-time argument: run-time indexed
 * per-thread accumulators spill to local memory, which is what made the
 * first combination-tiled pair kernel 3x slower than the per-row one.
 *
 * Stride contract: g1/g2/values/weights are base pointers of 2D
 * (tomo bin, row) views; the caller passes BOTH element strides
 * explicitly, so any view can be used without a copy -- strided rows
 * such as shear[:, 0] of an (nz, 2, npix) array, and equally the AoS
 * buffers (npix, nz, 2) / (npix, nz) that the pair kernels load and the
 * fused degrade writes directly.
 *
 * T  -- scalar type of the maps/weights (float / double)
 * QT -- scalar type of the aperture filter geometry (may be narrower
 *       than T; promotion to T at use is exact for float -> double)
 */

__COMMON_CUDA_SOURCE__

template<typename T, typename QT>
__global__ void gpu_aperture_shear_tomo(
    const T* g1,                 /* base ptr, bin b at g1 + b*g_stride  */
    const T* g2,
    const long long g_stride,    /* elements between tomo bins (2*npix for
                                    (nz,2,npix) views, npix for planar,
                                    2 for an (npix,nz,2) AoS buffer)     */
    const long long g_elem,      /* elements between rows (1 when the row
                                    axis is contiguous, nz*2 for AoS)    */
    const T* weights,            /* base ptr, bin b at weights + b*w_stride */
    const long long w_stride,
    const long long w_elem,
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
        const long long pix = (long long)q_inds[idx];
        const T wv = wb[pix * w_elem];
        /* Tangential shear w.r.t. the patch centre */
        const T gt = -g1b[pix * g_elem] * (T)q_cos[idx]
                   - g2b[pix * g_elem] * (T)q_sin[idx];
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
    const long long v_elem,
    const T* weights,
    const long long w_stride,
    const long long w_elem,
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
        const long long pix = (long long)q_inds[idx];
        const T wv = wb[pix * w_elem];
        sum_num += wv * vb[pix * v_elem] * (T)q_val[idx];
        sum_den += wv;
    }
    block_reduce_sum_pair(sum_num, sum_den, &sum_num, &sum_den);
    if (lane == 0) {
        const long long o = (long long)bin * npatches + patch;
        out_num[o] = (T)q_patch_area[patch] * sum_num;
        out_den[o] = sum_den;
    }
}


/*
 * One block per patch, all NZ tomographic bins inside it.  The disc
 * geometry is read once per pixel instead of NZ times; bitwise identical
 * to gpu_aperture_shear_tomo (see the file header).
 */
template<typename T, typename QT, int NZ>
__global__ void gpu_aperture_shear_tomo_fused(
    const T* g1,                 /* base ptr, bin b at g1 + b*g_stride  */
    const T* g2,
    const long long g_stride,
    const long long g_elem,
    const T* weights,
    const long long w_stride,
    const long long w_elem,
    const unsigned int* q_inds,
    const QT* q_cos,
    const QT* q_sin,
    const QT* q_val,
    const long long* q_offsets,
    const QT* q_patch_area,
    T* out_num,                  /* [NZ x npatches] */
    T* out_den,
    const int npatches)
{
    const int lane = (int)threadIdx.x;
    const int patch = (int)blockIdx.x;
    if (patch >= npatches) return;

    const long long start = q_offsets[patch];
    const long long stop  = q_offsets[patch + 1];

    T sn[NZ];
    T sd[NZ];
#pragma unroll
    for (int b = 0; b < NZ; ++b) { sn[b] = (T)0.0; sd[b] = (T)0.0; }

    for (long long idx = start + lane; idx < stop; idx += BLOCK_SIZE) {
        const long long pix = (long long)q_inds[idx];
        const T qc = (T)q_cos[idx];     /* the 16 B read once, not NZ times */
        const T qs = (T)q_sin[idx];
        const T qv = (T)q_val[idx];
#pragma unroll
        for (int b = 0; b < NZ; ++b) {
            const T wv = weights[(long long)b * w_stride + pix * w_elem];
            /* Tangential shear w.r.t. the patch centre */
            const T gt = -g1[(long long)b * g_stride + pix * g_elem] * qc
                       - g2[(long long)b * g_stride + pix * g_elem] * qs;
            sn[b] += wv * gt * qv;
            sd[b] += wv;
        }
    }

    __shared__ T s1[BLOCK_SIZE];        /* one buffer pair for all NZ bins */
    __shared__ T s2[BLOCK_SIZE];
    const T area = (T)q_patch_area[patch];
#pragma unroll
    for (int b = 0; b < NZ; ++b) {
        T n, d;
        __syncthreads();                /* the buffers are being reused */
        block_reduce_sum_pair_into(sn[b], sd[b], s1, s2, &n, &d);
        if (lane == 0) {
            const long long o = (long long)b * npatches + patch;
            out_num[o] = area * n;
            out_den[o] = d;
        }
    }
}

/* Density counterpart of gpu_aperture_shear_tomo_fused.  One value array
   instead of two, so 8 B of map data per visit: 12*nz B/pixel becomes
   8 + 4*nz, a slightly larger relative win than the shear kernel's. */
template<typename T, typename QT, int NZ>
__global__ void gpu_aperture_density_tomo_fused(
    const T* values,
    const long long v_stride,
    const long long v_elem,
    const T* weights,
    const long long w_stride,
    const long long w_elem,
    const unsigned int* q_inds,
    const QT* q_val,
    const long long* q_offsets,
    const QT* q_patch_area,
    T* out_num,
    T* out_den,
    const int npatches)
{
    const int lane = (int)threadIdx.x;
    const int patch = (int)blockIdx.x;
    if (patch >= npatches) return;

    const long long start = q_offsets[patch];
    const long long stop  = q_offsets[patch + 1];

    T sn[NZ];
    T sd[NZ];
#pragma unroll
    for (int b = 0; b < NZ; ++b) { sn[b] = (T)0.0; sd[b] = (T)0.0; }

    for (long long idx = start + lane; idx < stop; idx += BLOCK_SIZE) {
        const long long pix = (long long)q_inds[idx];
        const T qv = (T)q_val[idx];
#pragma unroll
        for (int b = 0; b < NZ; ++b) {
            const T wv = weights[(long long)b * w_stride + pix * w_elem];
            sn[b] += wv * values[(long long)b * v_stride + pix * v_elem] * qv;
            sd[b] += wv;
        }
    }

    __shared__ T s1[BLOCK_SIZE];
    __shared__ T s2[BLOCK_SIZE];
    const T area = (T)q_patch_area[patch];
#pragma unroll
    for (int b = 0; b < NZ; ++b) {
        T n, d;
        __syncthreads();
        block_reduce_sum_pair_into(sn[b], sd[b], s1, s2, &n, &d);
        if (lane == 0) {
            const long long o = (long long)b * npatches + patch;
            out_num[o] = area * n;
            out_den[o] = d;
        }
    }
}
