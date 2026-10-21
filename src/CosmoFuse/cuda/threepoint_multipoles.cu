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
 * exp(-i e phi), e = -E..E, by recurrence, each as one 2-vector), then every
 * thread accumulates the outputs order[tid + k * blockDim.x], k < K, in
 * registers.  K, F, E, O and NB are template parameters so the per-thread
 * accumulators are indexed with compile-time constants (run-time indices
 * spill them to local memory).  Pairs are CSR by (centre, bin):
 * offsets[(c0 + c) * nbins + b].  out is complex (interleaved re, im) of
 * type TA.
 *
 * Measured on an A100 (benchmarks/global/step1_probe.py, 7.3.0): ~90 % of
 * the time is the accumulation loop, which is FP64-instruction bound, so it
 * is written as four FMAs per complex multiply-add (instead of mul + fma +
 * add per component).  `order` puts the G_m of one m for all fields on
 * neighbouring threads, so a warp reads few distinct powers.  Together
 * 1.12x; register tiling, split staging, a half power table and other block
 * shapes were measured and are not faster.
 */

static __device__ __forceinline__ void cf_sincos(float x, float* s, float* c) {
    sincosf(x, s, c);
}
static __device__ __forceinline__ void cf_sincos(double x, double* s, double* c) {
    sincos(x, s, c);
}

template <typename T> struct vec2;
template <> struct vec2<double> { typedef double2 type; };
template <> struct vec2<float> { typedef float2 type; };

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
    const int* __restrict__ d_order,
    TA* __restrict__ out)
{
    typedef typename vec2<TA>::type T2;
    constexpr int P = 2 * E + 1;
    __shared__ T2 sa[NB][F];
    __shared__ T2 sp[NB][P];

    const long long c = blockIdx.x;
    const int tid = threadIdx.x;

    int fa[K], fb[K], ca[K], cb[K], ex[K], oo[K];
    bool act[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int slot = tid + k * blockDim.x;
        const int o = slot < O ? d_order[slot] : -1;
        act[k] = o >= 0;
        oo[k] = o;
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
                        sa[idx][f].x = -w * (gr * c2 + gi * s2);
                        sa[idx][f].y = -w * (gi * c2 - gr * s2);
                    } else {
                        sa[idx][f].x = w * gr;
                        sa[idx][f].y = TA(0);
                    }
                }
                TA s1, c1;
                cf_sincos(TA(phi_c[p]), &s1, &c1);
                // powers of exp(-i phi) = c1 - i s1
                TA pr = TA(1), pi = TA(0);
                sp[idx][E].x = pr;
                sp[idx][E].y = pi;
                for (int e = 1; e <= E; ++e) {
                    const TA nr = pr * c1 + pi * s1;
                    const TA ni = pi * c1 - pr * s1;
                    pr = nr;
                    pi = ni;
                    sp[idx][E + e].x = pr;
                    sp[idx][E + e].y = pi;
                    sp[idx][E - e].x = pr;
                    sp[idx][E - e].y = -pi;
                }
            }
            __syncthreads();
#pragma unroll
            for (int k = 0; k < K; ++k) {
                if (!act[k]) continue;
                for (int idx = 0; idx < cnt; ++idx) {
                    const T2 a = sa[idx][fa[k]];
                    TA xr = a.x;
                    TA xi = ca[k] ? -a.y : a.y;
                    if (fb[k] >= 0) {
                        const T2 bb = sa[idx][fb[k]];
                        const TA yr = bb.x;
                        const TA yi = cb[k] ? -bb.y : bb.y;
                        const TA tr = fma(xr, yr, -xi * yi);
                        const TA ti = fma(xr, yi, xi * yr);
                        xr = tr;
                        xi = ti;
                    }
                    const T2 q = sp[idx][ex[k]];
                    acc_re[k] = fma(xr, q.x, acc_re[k]);
                    acc_re[k] = fma(-xi, q.y, acc_re[k]);
                    acc_im[k] = fma(xr, q.y, acc_im[k]);
                    acc_im[k] = fma(xi, q.x, acc_im[k]);
                }
            }
            __syncthreads();
        }

#pragma unroll
        for (int k = 0; k < K; ++k) {
            if (!act[k]) continue;
            const long long flat = (c * nbins + b) * O + oo[k];
            out[2 * flat] = acc_re[k];
            out[2 * flat + 1] = acc_im[k];
        }
    }
}
