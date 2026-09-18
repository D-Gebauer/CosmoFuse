/*
 * tomo_packed_xipm.cu -- combination-tiled xi+/xi- kernel on *packed* pairs.
 *
 * Same estimator, arithmetic, lane striding and output layout as
 * gpu_tiled_tomo_reduce_xipm (tomo_vectorized_xipm.cu), but every pair is
 * stored in 8 bytes instead of 24:
 *
 *   pairs[4*p + 0], pairs[4*p + 1]   uint16 row index of pixel a / b, local to
 *                                    the row block of this (patch, level):
 *                                    row = row_base[bin_flat] + local
 *   pairs[4*p + 2], pairs[4*p + 3]   uint16 rotation angle: the pair-frame
 *                                    rotation e^{2i phi} has unit modulus, so
 *                                    only its angle alpha = a * 2 pi / 65536
 *                                    is kept and cos/sin are recomputed here
 *                                    (the kernel is memory-bound).
 *
 * The map rows are gathered per patch into contiguous blocks by the
 * orchestrator, so the random gathers of one thread block stay inside a
 * sub-MB window.  Angle quantisation (<= pi/65536 = 4.8e-5 rad) changes the
 * real part of a pair product by <= 1e-9 (relative) and mixes in at most
 * 1e-4 of its parity-odd imaginary part, which averages to zero over pairs.
 *
 * Grid layout:
 *   blockIdx.x  = angular separation bin index (0 .. nbins_total-1)
 *   threadIdx.x = pair index within the bin (strided loop)
 */

__COMMON_CUDA_SOURCE__


template<typename T, int TOMO_BINS_T, typename ACC>
__global__ void gpu_tiled_packed_reduce_xipm(
    const T* shear,                /* [n_packed_rows x TOMO_BINS x 2]      */
    const T* weights,              /* [n_packed_rows x TOMO_BINS]          */
    const unsigned short* pairs,   /* [npairs x 4] (see above)             */
    const long long* bin_offsets,  /* CSR offsets per angular bin block    */
    const long long* row_base,     /* first packed row of each bin's block */
    ACC* out_num,
    ACC* out_den,
    const int ncomb,
    const long long nbins_total)
{
    enum { NCOMB = (TOMO_BINS_T * (TOMO_BINS_T + 1)) / 2 };

    const int lane = (int)threadIdx.x;
    const long long bin_flat = (long long)blockIdx.x;
    /* Uniform per block, so the collective reductions stay consistent. */
    if (bin_flat >= nbins_total || ncomb != NCOMB) {
        return;
    }

    ACC acc_p[2 * NCOMB];
    ACC acc_m[2 * NCOMB];
    ACC acc_w[2 * NCOMB];
    for (int r = 0; r < 2 * NCOMB; ++r) {
        acc_p[r] = (ACC)0.0;
        acc_m[r] = (ACC)0.0;
        acc_w[r] = (ACC)0.0;
    }

    const long long start = bin_offsets[bin_flat];
    const long long stop = bin_offsets[bin_flat + 1];
    const long long base = row_base[bin_flat];
    const T angle_unit = (T)9.587379924285257e-05;   /* 2 pi / 65536 */

    for (long long tid = start + lane; tid < stop; tid += BLOCK_SIZE) {
        const unsigned short* p = pairs + 4 * tid;
        const long long row_a = (base + (long long)p[0]) * (long long)TOMO_BINS_T;
        const long long row_b = (base + (long long)p[1]) * (long long)TOMO_BINS_T;
        const T ang_a = (T)p[2] * angle_unit;
        const T ang_b = (T)p[3] * angle_unit;
        const T ea_R = cos(ang_a);
        const T ea_I = sin(ang_a);
        const T eb_R = cos(ang_b);
        const T eb_I = sin(ang_b);

        T a_R[TOMO_BINS_T], a_I[TOMO_BINS_T], w_a[TOMO_BINS_T];
        T b_R[TOMO_BINS_T], b_I[TOMO_BINS_T], w_b[TOMO_BINS_T];
        for (int t = 0; t < TOMO_BINS_T; ++t) {
            const T ga1 = shear[(row_a + t) * 2];
            const T ga2 = shear[(row_a + t) * 2 + 1];
            const T gb1 = shear[(row_b + t) * 2];
            const T gb2 = shear[(row_b + t) * 2 + 1];
            a_R[t] = ga1 * ea_R - ga2 * ea_I;
            a_I[t] = ga1 * ea_I + ga2 * ea_R;
            b_R[t] = gb1 * eb_R - gb2 * eb_I;
            b_I[t] = gb1 * eb_I + gb2 * eb_R;
            w_a[t] = weights[row_a + t];
            w_b[t] = weights[row_b + t];
        }

        int k = 0;
        for (int i = 0; i < TOMO_BINS_T; ++i) {
            for (int j = i; j < TOMO_BINS_T; ++j, ++k) {
                const T w_ab = w_a[i] * w_b[j];
                acc_w[2 * k] += (ACC)w_ab;
                acc_p[2 * k] += (ACC)(w_ab * (b_R[j] * a_R[i] + b_I[j] * a_I[i]));
                acc_m[2 * k] += (ACC)(w_ab * (b_R[j] * a_R[i] - b_I[j] * a_I[i]));
                if (i != j) {
                    const T w_ba = w_a[j] * w_b[i];
                    acc_w[2 * k + 1] += (ACC)w_ba;
                    acc_p[2 * k + 1] += (ACC)(w_ba * (b_R[i] * a_R[j] + b_I[i] * a_I[j]));
                    acc_m[2 * k + 1] += (ACC)(w_ba * (b_R[i] * a_R[j] - b_I[i] * a_I[j]));
                }
            }
        }
    }

    int k = 0;
    for (int i = 0; i < TOMO_BINS_T; ++i) {
        for (int j = i; j < TOMO_BINS_T; ++j, ++k) {
            for (int ori = 0; ori < 2; ++ori) {
                const int r = 2 * k + ori;
                ACC sum_p = acc_p[r];
                ACC sum_m = acc_m[r];
                ACC sum_w = acc_w[r];
                block_reduce_sum_triple<ACC>(sum_p, sum_m, sum_w, &sum_p, &sum_m, &sum_w);
                if (lane == 0 && !(ori == 1 && i == j)) {
                    const long long out_p_idx = ((long long)r) * nbins_total + bin_flat;
                    const long long out_m_idx =
                        ((long long)(2 * NCOMB + r)) * nbins_total + bin_flat;
                    out_num[out_p_idx] = sum_p;
                    out_num[out_m_idx] = sum_m;
                    out_den[out_p_idx] = sum_w;
                }
            }
        }
    }
}
