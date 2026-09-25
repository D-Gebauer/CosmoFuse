/*
 * threepoint_multipoles.cu -- step 1 of the multipole three-point estimator.
 *
 * Contract (identical to the Numba kernel in threepoint_kernels.py):
 *
 *   out[c, b, o] = sum_{j in bin b of centre c0 + c}
 *                  A(j, fa[o])^{ca[o]} * B(j, o) * exp(-i e[o] phi_cj)
 *   B(j, o)      = A(j, fb[o])^{cb[o]} if fb[o] >= 0 else 1
 *   A(j, f)      = w_f(j) v_f(j)                                  (spin 0)
 *                = -w_f(j) (v1_f(j) + i v2_f(j)) exp(-2i psi_j)    (spin 2)
 *
 * One block per centre; the neighbours of one (centre, bin) are staged in
 * shared memory NB at a time (their projected field values and the powers
 * exp(-i e phi), e = -E..E, by recurrence), then every thread accumulates the
 * outputs o = tid + k * blockDim.x, k < K, in registers.  K, F, E, O and NB
 * are template parameters so the per-thread accumulators are indexed with
 * compile-time constants (run-time indices spill them to local memory).
 * Pairs are CSR by (centre, bin): offsets[(c0 + c) * nbins + b].
 * out is complex (interleaved re, im) of type TA.
 */

static __device__ __forceinline__ void cf_sincos(float x, float* s, float* c) {
    sincosf(x, s, c);
}
static __device__ __forceinline__ void cf_sincos(double x, double* s, double* c) {
    sincos(x, s, c);
}

template <typename TV, typename TR, typename TA, int F, int E, int O, int NB, int K>
__global__ void gpu_3pcf_multipoles(
    const long long* __restrict__ offsets,
    const long long c0,
    const int nbins,
    const int* __restrict__ nbr,
    const TR* __restrict__ phi_c,
    const TR* __restrict__ phi_n,
    const TV* __restrict__ vre,
    const TV* __restrict__ vim,
    const TV* __restrict__ wts,
    const long long n_rows,
    const int* __restrict__ spin,
    const int* __restrict__ d_fa,
    const int* __restrict__ d_fb,
    const int* __restrict__ d_ca,
    const int* __restrict__ d_cb,
    const int* __restrict__ d_e,
    TA* __restrict__ out)
{
    constexpr int P = 2 * E + 1;
    __shared__ TA sa_re[NB][F];
    __shared__ TA sa_im[NB][F];
    __shared__ TA sp_re[NB][P];
    __shared__ TA sp_im[NB][P];

    const long long c = blockIdx.x;
    const int tid = threadIdx.x;

    int fa[K], fb[K], ca[K], cb[K], ex[K];
    bool act[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int o = tid + k * blockDim.x;
        act[k] = o < O;
        fa[k] = act[k] ? d_fa[o] : 0;
        fb[k] = act[k] ? d_fb[o] : -1;
        ca[k] = act[k] ? d_ca[o] : 0;
        cb[k] = act[k] ? d_cb[o] : 0;
        ex[k] = act[k] ? d_e[o] + E : E;
    }

    for (int b = 0; b < nbins; ++b) {
        const long long lo = offsets[(c0 + c) * nbins + b];
        const long long hi = offsets[(c0 + c) * nbins + b + 1];
        TA acc_re[K], acc_im[K];
#pragma unroll
        for (int k = 0; k < K; ++k) {
            acc_re[k] = TA(0);
            acc_im[k] = TA(0);
        }

        for (long long base = lo; base < hi; base += NB) {
            const int cnt = (int)((hi - base) < NB ? (hi - base) : NB);
            for (int idx = tid; idx < cnt; idx += blockDim.x) {
                const long long p = base + idx;
                const long long j = nbr[p];
                TA s2, c2;
                cf_sincos(TA(2) * TA(phi_n[p]), &s2, &c2);
                for (int f = 0; f < F; ++f) {
                    const TA w = TA(wts[f * n_rows + j]);
                    const TA gr = TA(vre[f * n_rows + j]);
                    if (spin[f] == 2) {
                        const TA gi = TA(vim[f * n_rows + j]);
                        // -w (gr + i gi)(c2 - i s2)
                        sa_re[idx][f] = -w * (gr * c2 + gi * s2);
                        sa_im[idx][f] = -w * (gi * c2 - gr * s2);
                    } else {
                        sa_re[idx][f] = w * gr;
                        sa_im[idx][f] = TA(0);
                    }
                }
                TA s1, c1;
                cf_sincos(TA(phi_c[p]), &s1, &c1);
                // powers of exp(-i phi) = c1 - i s1
                TA pr = TA(1), pi = TA(0);
                sp_re[idx][E] = pr;
                sp_im[idx][E] = pi;
                for (int e = 1; e <= E; ++e) {
                    const TA nr = pr * c1 + pi * s1;
                    const TA ni = pi * c1 - pr * s1;
                    pr = nr;
                    pi = ni;
                    sp_re[idx][E + e] = pr;
                    sp_im[idx][E + e] = pi;
                    sp_re[idx][E - e] = pr;
                    sp_im[idx][E - e] = -pi;
                }
            }
            __syncthreads();
#pragma unroll
            for (int k = 0; k < K; ++k) {
                if (!act[k]) continue;
                for (int idx = 0; idx < cnt; ++idx) {
                    TA xr = sa_re[idx][fa[k]];
                    TA xi = ca[k] ? -sa_im[idx][fa[k]] : sa_im[idx][fa[k]];
                    if (fb[k] >= 0) {
                        const TA yr = sa_re[idx][fb[k]];
                        const TA yi = cb[k] ? -sa_im[idx][fb[k]] : sa_im[idx][fb[k]];
                        const TA tr = xr * yr - xi * yi;
                        const TA ti = xr * yi + xi * yr;
                        xr = tr;
                        xi = ti;
                    }
                    const TA qr = sp_re[idx][ex[k]];
                    const TA qi = sp_im[idx][ex[k]];
                    acc_re[k] += xr * qr - xi * qi;
                    acc_im[k] += xr * qi + xi * qr;
                }
            }
            __syncthreads();
        }

#pragma unroll
        for (int k = 0; k < K; ++k) {
            if (!act[k]) continue;
            const long long o = tid + k * blockDim.x;
            const long long flat = (c * nbins + b) * O + o;
            out[2 * flat] = acc_re[k];
            out[2 * flat + 1] = acc_im[k];
        }
    }
}
