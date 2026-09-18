/*
 * degrade_rows.cu -- Fused construction of the static-treecode virtual rows.
 *
 * A cell row holds the weight sum of its members and their weighted mean,
 * so the unchanged pair kernels reproduce the fine pair sums exactly.
 * Building those rows used to be a chain of sparse matrix products
 * (cupyx.scipy.sparse) over an (n_active, K * n_lead) temporary, plus two
 * transposes -- ~0.94 ms of a 4.37 ms device-resident call at nside 512,
 * k = 2.9, and ~10 ms at nside 2048.
 *
 * Here one block-per-cell-group kernel walks each level's CSR children
 * directly and accumulates into an (n_lead, n_appended) scratch at the
 * accumulation dtype; a second kernel normalises and scatters into the
 * final row buffers.  No pair-sized temporary, no transpose.
 *
 * Exactly the same recursion as the sparse path, so the results differ
 * only by the order of summation *within* a cell:
 *
 *   level 0   (WEIGHTED):  w_I = sum_j w_j          v_I = sum_j w_j v_j
 *   level > 0 (!WEIGHTED): w_I = sum_J w_J          v_I = sum_J v_J
 *
 * i.e. deeper levels sum the *unnormalised* partial sums of their
 * children, and the single division by w_I happens once, at the end, in
 * gpu_degrade_finalize.  Normalising per level instead would add a
 * divide-then-multiply round-trip at every level.
 *
 * Determinism: each (cell, lead) is reduced by one fixed group of SEG
 * lanes -- lane l takes children l, l+SEG, ... and the shuffle tree is
 * fixed -- so repeated runs give bit-identical output.  No atomics.
 *
 * SEG is sized from the mean number of children: a treecode level halves
 * nside, so a cell has exactly 4 children and a full 32-lane warp per cell
 * would leave 87 % of its lanes idle (measured: 2.1x instead of 4.3x).
 * A coarse aperture level can have many more, hence the template.
 *
 * Addressing.  Every buffer is described by four element strides,
 *
 *     offset(lead, k, row) = base + lead*lead_s + k*comp_s + row*row_s
 *
 * (the weights have no component axis), so the same kernel writes any of
 * the layouts the callers want without a transpose afterwards:
 *
 *     SoA, per component  (K*n_lead, n_rows)  lead_s=n_rows,   comp_s=n_lead*n_rows, row_s=1
 *     SoA, interleaved    (n_lead, K, n_rows) lead_s=K*n_rows, comp_s=n_rows,        row_s=1
 *     AoS, kernel layout  (n_rows, n_lead, K) lead_s=K,        comp_s=1,             row_s=n_lead*K
 *
 * The interleaved form is what the shear leaves want -- (nz, 2, n_rows)
 * without the stack that used to glue g1 and g2 back together -- and the
 * AoS form is what the pair kernels load, so the SoA->AoS transpose in
 * front of them disappears too.
 *
 * TSRC -- scalar type of the source rows (map dtype at level 0, ACC deeper)
 * ACC  -- accumulation dtype
 * NVAL -- number of value arrays sharing the weights (0, 1 or 2)
 * SEG  -- lanes cooperating on one cell (power of two, 1..32)
 */

__COMMON_CUDA_SOURCE__

#define DEGRADE_BLOCK 256

template<typename TSRC, typename ACC, int NVAL, bool WEIGHTED, int SEG>
__global__ void gpu_degrade_level(
    const long long* __restrict__ indptr,   /* (n_cells + 1) */
    const int* __restrict__ indices,        /* children, local to the source block */
    const TSRC* __restrict__ w_src,
    const long long w_src_base,             /* elements to (lead 0, row 0) */
    const long long w_src_lead,             /* elements between lead rows */
    const long long w_src_row,              /* elements between rows */
    const TSRC* __restrict__ v_src,
    const long long v_src_base,
    const long long v_src_lead,
    const long long v_src_comp,             /* elements between components */
    const long long v_src_row,
    ACC* __restrict__ w_dst,
    const long long w_dst_base,
    const long long w_dst_lead,
    const long long w_dst_row,
    ACC* __restrict__ v_dst,
    const long long v_dst_base,
    const long long v_dst_lead,
    const long long v_dst_comp,
    const long long v_dst_row,
    const int n_cells,
    const int n_lead)
{
    const int lane = (int)threadIdx.x % SEG;          /* lane within the group */
    const int group = (int)threadIdx.x / SEG;         /* group within the block */
    const long long gid =
        (long long)blockIdx.x * (DEGRADE_BLOCK / SEG) + (long long)group;
    const long long total = (long long)n_cells * (long long)n_lead;
    if (gid >= total) return;

    const int cell = (int)(gid / n_lead);
    const int lead = (int)(gid - (long long)cell * n_lead);

    const long long begin = indptr[cell];
    const long long end = indptr[cell + 1];

    const TSRC* wrow = w_src + w_src_base + (long long)lead * w_src_lead;

    ACC sw = (ACC)0;
    ACC sv[NVAL > 0 ? NVAL : 1];
#pragma unroll
    for (int k = 0; k < NVAL; ++k) sv[k] = (ACC)0;

    for (long long j = begin + lane; j < end; j += SEG) {
        const long long child = (long long)indices[j];
        const ACC wj = (ACC)wrow[child * w_src_row];
        sw += wj;
#pragma unroll
        for (int k = 0; k < NVAL; ++k) {
            const TSRC* vrow = v_src + v_src_base
                + (long long)lead * v_src_lead + (long long)k * v_src_comp;
            const ACC vj = (ACC)vrow[child * v_src_row];
            sv[k] += WEIGHTED ? wj * vj : vj;
        }
    }

#pragma unroll
    for (int off = SEG / 2; off > 0; off >>= 1) {
        sw += __shfl_down_sync(0xffffffffu, sw, off, SEG);
#pragma unroll
        for (int k = 0; k < NVAL; ++k) {
            sv[k] += __shfl_down_sync(0xffffffffu, sv[k], off, SEG);
        }
    }

    if (lane == 0) {
        w_dst[w_dst_base + (long long)lead * w_dst_lead
              + (long long)cell * w_dst_row] = sw;
#pragma unroll
        for (int k = 0; k < NVAL; ++k) {
            v_dst[v_dst_base + (long long)lead * v_dst_lead
                  + (long long)k * v_dst_comp + (long long)cell * v_dst_row] = sv[k];
        }
    }
}

/*
 * Divide the accumulated value sums by their weight sum and scatter both
 * into the final row buffers (map dtype), for appended rows [lo, hi).
 * Cells with zero weight get exactly zero, as the sparse path's
 * `nonzero / where(nonzero, W, 1)` did.
 *
 * The scratch is always SoA (row stride 1); the destination uses the same
 * four-stride description as above, with its base already pointing at the
 * first appended row.  `row_inner` picks which of (row, lead) the flat
 * thread index runs over fastest, so the destination writes coalesce for
 * both the SoA and the AoS layout.
 */
template<typename T, typename ACC, int NVAL>
__global__ void gpu_degrade_finalize(
    const ACC* __restrict__ w_app,
    const long long w_app_lead,
    const ACC* __restrict__ v_app,
    const long long v_app_lead,
    const long long v_app_comp,
    T* __restrict__ w_rows,
    const long long w_rows_base,            /* includes the n_active offset */
    const long long w_rows_lead,
    const long long w_rows_row,
    T* __restrict__ v_rows,
    const long long v_rows_base,
    const long long v_rows_lead,
    const long long v_rows_comp,
    const long long v_rows_row,
    const int n_lead,
    const long long lo,
    const long long hi,
    const int row_inner)
{
    const long long span = hi - lo;
    const long long total = span * (long long)n_lead;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        long long r;
        int lead;
        if (row_inner) {
            lead = (int)(idx / span);
            r = lo + (idx - (long long)lead * span);
        } else {
            r = lo + idx / (long long)n_lead;
            lead = (int)(idx - (r - lo) * (long long)n_lead);
        }

        const ACC w = w_app[(long long)lead * w_app_lead + r];
        w_rows[w_rows_base + (long long)lead * w_rows_lead + r * w_rows_row] = (T)w;
        const ACC inv = (w != (ACC)0) ? ((ACC)1) / w : (ACC)0;
#pragma unroll
        for (int k = 0; k < NVAL; ++k) {
            const ACC s = v_app[(long long)lead * v_app_lead
                                + (long long)k * v_app_comp + r];
            v_rows[v_rows_base + (long long)lead * v_rows_lead
                   + (long long)k * v_rows_comp + r * v_rows_row] = (T)(s * inv);
        }
    }
}
