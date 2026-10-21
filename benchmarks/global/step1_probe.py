"""Probe kernels for step 1 of Global3PCF (``gpu_3pcf_multipoles``).

On the real geometry (DES Y3), for the ``ggg`` fields, times the step-1 kernel
over all centres and checks every variant bitwise against the shipped kernel:

* P0   the installed kernel with the <= 7.2 table (every sigma x both pair
       types); against CosmoFuse 7.2 this is the kernel the numbers in
       RESULTS.md compare to;
* P1   the installed kernel, only the (sigma, type) pair products ggg reads;
* v3   the kernel shipped in 7.3.0 and its ablations: 4-FMA accumulation,
       2-vector shared values, thread map (m-major), no-accumulation /
       no-staging timings;
* --all also the variants measured and rejected: register tiling (T
       multipoles per thread) and staging changes (half power table, two
       threads per staged neighbour, block / chunk shapes).

Results are in ``RESULTS.md`` (2026-10-21).

    python step1_probe.py --nside 512 --mask DESY3_Mask.fits
"""

import argparse
import time
import warnings

import cupy as cp
import healpy as hp
import numpy as np

from CosmoFuse import Global3PCF

TILED_SRC = r"""
static __device__ __forceinline__ void cf_sincos(float x, float* s, float* c) { sincosf(x, s, c); }
static __device__ __forceinline__ void cf_sincos(double x, double* s, double* c) { sincos(x, s, c); }

template <typename TV, typename TR, typename TA, int F, int E, int NB, int T, int TP, int NGT>
__global__ void tiled(
    const long long* __restrict__ offsets, const long long c0, const int nbins,
    const int* __restrict__ nbr, const TR* __restrict__ phi_c, const TR* __restrict__ phi_n,
    const TV* __restrict__ vre, const TV* __restrict__ vim, const TV* __restrict__ wts,
    const long long n_rows, const int* __restrict__ spin,
    const int n_gt,                      // G tiles in use (< NGT)
    const int n_pt,                      // pair tiles
    const int* __restrict__ pt_fa, const int* __restrict__ pt_ca,
    const int* __restrict__ pt_fb, const int* __restrict__ pt_cb,
    const int* __restrict__ pt_e, const int* __restrict__ pt_o, const int* __restrict__ pt_n,
    const int O, TA* __restrict__ out)
{
    constexpr int P = 2 * E + 1;
    constexpr int MT = (P + T - 1) / T;
    __shared__ TA sa_re[NB][F];
    __shared__ TA sa_im[NB][F];
    __shared__ TA sp_re[NB][P];
    __shared__ TA sp_im[NB][P];
    const long long c = blockIdx.x;
    const int tid = threadIdx.x;
    const bool is_g = tid < n_gt;
    const bool is_p = tid >= NGT && tid - NGT < n_pt;
    // G tile
    const int gf = is_g ? tid / MT : 0;
    const int ge0 = is_g ? (tid % MT) * T : 0;
    // pair tile
    const int pi_ = is_p ? tid - NGT : 0;
    const int pfa = is_p ? pt_fa[pi_] : 0, pca = is_p ? pt_ca[pi_] : 0;
    const int pfb = is_p ? pt_fb[pi_] : 0, pcb = is_p ? pt_cb[pi_] : 0;
    const int pe = is_p ? pt_e[pi_] + E : E, po = is_p ? pt_o[pi_] : 0;
    const int pn = is_p ? pt_n[pi_] : 0;

    for (int b = 0; b < nbins; ++b) {
        const long long lo = offsets[(c0 + c) * nbins + b];
        const long long hi = offsets[(c0 + c) * nbins + b + 1];
        TA acc_re[T > TP ? T : TP], acc_im[T > TP ? T : TP];
#pragma unroll
        for (int t = 0; t < (T > TP ? T : TP); ++t) { acc_re[t] = TA(0); acc_im[t] = TA(0); }
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
                        sa_re[idx][f] = -w * (gr * c2 + gi * s2);
                        sa_im[idx][f] = -w * (gi * c2 - gr * s2);
                    } else {
                        sa_re[idx][f] = w * gr;
                        sa_im[idx][f] = TA(0);
                    }
                }
                TA s1, c1;
                cf_sincos(TA(phi_c[p]), &s1, &c1);
                TA pr = TA(1), pim = TA(0);
                sp_re[idx][E] = pr;
                sp_im[idx][E] = pim;
                for (int e = 1; e <= E; ++e) {
                    const TA nr = pr * c1 + pim * s1;
                    const TA ni = pim * c1 - pr * s1;
                    pr = nr;
                    pim = ni;
                    sp_re[idx][E + e] = pr;
                    sp_im[idx][E + e] = pim;
                    sp_re[idx][E - e] = pr;
                    sp_im[idx][E - e] = -pim;
                }
            }
            __syncthreads();
            if (is_g) {
                for (int idx = 0; idx < cnt; ++idx) {
                    const TA xr = sa_re[idx][gf];
                    const TA xi = sa_im[idx][gf];
#pragma unroll
                    for (int t = 0; t < T; ++t) {
                        if (ge0 + t < P) {
                            const TA qr = sp_re[idx][ge0 + t];
                            const TA qi = sp_im[idx][ge0 + t];
                            acc_re[t] += xr * qr - xi * qi;
                            acc_im[t] += xr * qi + xi * qr;
                        }
                    }
                }
            } else if (is_p) {
                for (int idx = 0; idx < cnt; ++idx) {
                    const TA xr = sa_re[idx][pfa];
                    const TA xi = pca ? -sa_im[idx][pfa] : sa_im[idx][pfa];
                    const TA qr = sp_re[idx][pe];
                    const TA qi = sp_im[idx][pe];
#pragma unroll
                    for (int t = 0; t < TP; ++t) {
                        if (t < pn) {
                            const TA yr = sa_re[idx][pfb + t];
                            const TA yi = pcb ? -sa_im[idx][pfb + t] : sa_im[idx][pfb + t];
                            const TA tr = xr * yr - xi * yi;
                            const TA ti = xr * yi + xi * yr;
                            acc_re[t] += tr * qr - ti * qi;
                            acc_im[t] += tr * qi + ti * qr;
                        }
                    }
                }
            }
            __syncthreads();
        }
        if (is_g) {
#pragma unroll
            for (int t = 0; t < T; ++t) {
                if (ge0 + t < P) {
                    const long long flat = (c * nbins + b) * (long long)O + gf * P + ge0 + t;
                    out[2 * flat] = acc_re[t];
                    out[2 * flat + 1] = acc_im[t];
                }
            }
        } else if (is_p) {
#pragma unroll
            for (int t = 0; t < TP; ++t) {
                if (t < pn) {
                    const long long flat = (c * nbins + b) * (long long)O + po + t;
                    out[2 * flat] = acc_re[t];
                    out[2 * flat + 1] = acc_im[t];
                }
            }
        }
    }
}
"""

V2_SRC = r"""
static __device__ __forceinline__ void cf_sincos(float x, float* s, float* c) { sincosf(x, s, c); }
static __device__ __forceinline__ void cf_sincos(double x, double* s, double* c) { sincos(x, s, c); }

// The shipped kernel with two staging options (the accumulation loop is
// unchanged): HALF stores only e >= 0 of the power table and reads
// e < 0 as its conjugate (exact); SPLIT stages every neighbour with two
// threads (fields + rotation | the power chain).
template <typename TV, typename TR, typename TA, int F, int E, int O, int NB, int K,
          bool HALF, bool SPLIT>
__global__ void v2(
    const long long* __restrict__ offsets, const long long c0, const int nbins,
    const int* __restrict__ nbr, const TR* __restrict__ phi_c, const TR* __restrict__ phi_n,
    const TV* __restrict__ vre, const TV* __restrict__ vim, const TV* __restrict__ wts,
    const long long n_rows, const int* __restrict__ spin,
    const int* __restrict__ d_fa, const int* __restrict__ d_fb, const int* __restrict__ d_ca,
    const int* __restrict__ d_cb, const int* __restrict__ d_e, TA* __restrict__ out)
{
    constexpr int P = 2 * E + 1;
    constexpr int PS = HALF ? E + 1 : P;
    __shared__ TA sa_re[NB][F];
    __shared__ TA sa_im[NB][F];
    __shared__ TA sp_re[NB][PS];
    __shared__ TA sp_im[NB][PS];
    const long long c = blockIdx.x;
    const int tid = threadIdx.x;
    int fa[K], fb[K], ca[K], cb[K], ex[K];
    bool act[K], neg[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int o = tid + k * blockDim.x;
        act[k] = o < O;
        fa[k] = act[k] ? d_fa[o] : 0;
        fb[k] = act[k] ? d_fb[o] : -1;
        ca[k] = act[k] ? d_ca[o] : 0;
        cb[k] = act[k] ? d_cb[o] : 0;
        const int e = act[k] ? d_e[o] : 0;
        neg[k] = HALF && e < 0;
        ex[k] = HALF ? (e < 0 ? -e : e) : e + E;
    }
    for (int b = 0; b < nbins; ++b) {
        const long long lo = offsets[(c0 + c) * nbins + b];
        const long long hi = offsets[(c0 + c) * nbins + b + 1];
        TA acc_re[K], acc_im[K];
#pragma unroll
        for (int k = 0; k < K; ++k) { acc_re[k] = TA(0); acc_im[k] = TA(0); }
        for (long long base = lo; base < hi; base += NB) {
            const int cnt = (int)((hi - base) < NB ? (hi - base) : NB);
            for (int s = tid; s < (SPLIT ? 2 * cnt : cnt); s += blockDim.x) {
                const int idx = SPLIT ? (s >> 1) : s;
                const bool do_fields = !SPLIT || (s & 1) == 0;
                const bool do_powers = !SPLIT || (s & 1) == 1;
                const long long p = base + idx;
                if (do_fields) {
                    const long long j = nbr[p];
                    TA s2, c2;
                    cf_sincos(TA(2) * TA(phi_n[p]), &s2, &c2);
                    for (int f = 0; f < F; ++f) {
                        const TA w = TA(wts[f * n_rows + j]);
                        const TA gr = TA(vre[f * n_rows + j]);
                        if (spin[f] == 2) {
                            const TA gi = TA(vim[f * n_rows + j]);
                            sa_re[idx][f] = -w * (gr * c2 + gi * s2);
                            sa_im[idx][f] = -w * (gi * c2 - gr * s2);
                        } else {
                            sa_re[idx][f] = w * gr;
                            sa_im[idx][f] = TA(0);
                        }
                    }
                }
                if (do_powers) {
                    TA s1, c1;
                    cf_sincos(TA(phi_c[p]), &s1, &c1);
                    TA pr = TA(1), pi = TA(0);
                    sp_re[idx][HALF ? 0 : E] = pr;
                    sp_im[idx][HALF ? 0 : E] = pi;
                    for (int e = 1; e <= E; ++e) {
                        const TA nr = pr * c1 + pi * s1;
                        const TA ni = pi * c1 - pr * s1;
                        pr = nr;
                        pi = ni;
                        if (HALF) {
                            sp_re[idx][e] = pr;
                            sp_im[idx][e] = pi;
                        } else {
                            sp_re[idx][E + e] = pr;
                            sp_im[idx][E + e] = pi;
                            sp_re[idx][E - e] = pr;
                            sp_im[idx][E - e] = -pi;
                        }
                    }
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
                    const TA qi = neg[k] ? -sp_im[idx][ex[k]] : sp_im[idx][ex[k]];
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
"""

V3_SRC = r"""
static __device__ __forceinline__ void cf_sincos(float x, float* s, float* c) { sincosf(x, s, c); }
static __device__ __forceinline__ void cf_sincos(double x, double* s, double* c) { sincos(x, s, c); }
template <typename T> struct vec2;
template <> struct vec2<double> { typedef double2 type; };
template <> struct vec2<float> { typedef float2 type; };

// The shipped kernel with (a) a thread -> output map (perm) and (b) complex
// shared values stored as one 2-vector (V2=true) instead of two arrays.  The
// arithmetic of every output is the shipped kernel's.
template <typename TV, typename TR, typename TA, int F, int E, int O, int NB, int K, bool V2,
          bool FMA, int ABL>
__global__ void v3(
    const long long* __restrict__ offsets, const long long c0, const int nbins,
    const int* __restrict__ nbr, const TR* __restrict__ phi_c, const TR* __restrict__ phi_n,
    const TV* __restrict__ vre, const TV* __restrict__ vim, const TV* __restrict__ wts,
    const long long n_rows, const int* __restrict__ spin,
    const int* __restrict__ d_fa, const int* __restrict__ d_fb, const int* __restrict__ d_ca,
    const int* __restrict__ d_cb, const int* __restrict__ d_e, const int* __restrict__ perm,
    TA* __restrict__ out)
{
    typedef typename vec2<TA>::type T2;
    constexpr int P = 2 * E + 1;
    __shared__ T2 sa[V2 ? NB : 1][F];
    __shared__ T2 sp[V2 ? NB : 1][P];
    __shared__ TA sa_re[V2 ? 1 : NB][F];
    __shared__ TA sa_im[V2 ? 1 : NB][F];
    __shared__ TA sp_re[V2 ? 1 : NB][P];
    __shared__ TA sp_im[V2 ? 1 : NB][P];
    const long long c = blockIdx.x;
    const int tid = threadIdx.x;
    int fa[K], fb[K], ca[K], cb[K], ex[K], oo[K];
    bool act[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int slot = tid + k * blockDim.x;
        const int o = slot < O ? perm[slot] : -1;
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
        for (int k = 0; k < K; ++k) { acc_re[k] = TA(0); acc_im[k] = TA(0); }
        for (long long base = lo; base < hi; base += NB) {
            const int cnt = (int)((hi - base) < NB ? (hi - base) : NB);
            for (int idx = tid; idx < cnt && ABL != 2; idx += blockDim.x) {
                const long long p = base + idx;
                const long long j = nbr[p];
                TA s2, c2;
                cf_sincos(TA(2) * TA(phi_n[p]), &s2, &c2);
                for (int f = 0; f < F; ++f) {
                    const TA w = TA(wts[f * n_rows + j]);
                    const TA gr = TA(vre[f * n_rows + j]);
                    TA ar, ai;
                    if (spin[f] == 2) {
                        const TA gi = TA(vim[f * n_rows + j]);
                        ar = -w * (gr * c2 + gi * s2);
                        ai = -w * (gi * c2 - gr * s2);
                    } else {
                        ar = w * gr;
                        ai = TA(0);
                    }
                    if (V2) { sa[idx][f].x = ar; sa[idx][f].y = ai; }
                    else { sa_re[idx][f] = ar; sa_im[idx][f] = ai; }
                }
                TA s1, c1;
                cf_sincos(TA(phi_c[p]), &s1, &c1);
                TA pr = TA(1), pi = TA(0);
                if (V2) { sp[idx][E].x = pr; sp[idx][E].y = pi; }
                else { sp_re[idx][E] = pr; sp_im[idx][E] = pi; }
                for (int e = 1; e <= E; ++e) {
                    const TA nr = pr * c1 + pi * s1;
                    const TA ni = pi * c1 - pr * s1;
                    pr = nr;
                    pi = ni;
                    if (V2) {
                        sp[idx][E + e].x = pr; sp[idx][E + e].y = pi;
                        sp[idx][E - e].x = pr; sp[idx][E - e].y = -pi;
                    } else {
                        sp_re[idx][E + e] = pr; sp_im[idx][E + e] = pi;
                        sp_re[idx][E - e] = pr; sp_im[idx][E - e] = -pi;
                    }
                }
            }
            __syncthreads();
#pragma unroll
            for (int k = 0; k < K; ++k) {
                if (!act[k] || ABL == 1) continue;
                for (int idx = 0; idx < cnt; ++idx) {
                    TA ar, ai, br = TA(0), bi = TA(0), qr, qi;
                    if (V2) {
                        const T2 a = sa[idx][fa[k]];
                        ar = a.x; ai = a.y;
                        if (fb[k] >= 0) { const T2 bb = sa[idx][fb[k]]; br = bb.x; bi = bb.y; }
                        const T2 q = sp[idx][ex[k]];
                        qr = q.x; qi = q.y;
                    } else {
                        ar = sa_re[idx][fa[k]]; ai = sa_im[idx][fa[k]];
                        if (fb[k] >= 0) { br = sa_re[idx][fb[k]]; bi = sa_im[idx][fb[k]]; }
                        qr = sp_re[idx][ex[k]]; qi = sp_im[idx][ex[k]];
                    }
                    TA xr = ar;
                    TA xi = ca[k] ? -ai : ai;
                    if (fb[k] >= 0) {
                        const TA yr = br;
                        const TA yi = cb[k] ? -bi : bi;
                        if (FMA) {
                            const TA tr = fma(xr, yr, -xi * yi);
                            const TA ti = fma(xr, yi, xi * yr);
                            xr = tr;
                            xi = ti;
                        } else {
                            const TA tr = xr * yr - xi * yi;
                            const TA ti = xr * yi + xi * yr;
                            xr = tr;
                            xi = ti;
                        }
                    }
                    if (FMA) {
                        acc_re[k] = fma(xr, qr, acc_re[k]);
                        acc_re[k] = fma(-xi, qi, acc_re[k]);
                        acc_im[k] = fma(xr, qi, acc_im[k]);
                        acc_im[k] = fma(xi, qr, acc_im[k]);
                    } else {
                        acc_re[k] += xr * qr - xi * qi;
                        acc_im[k] += xr * qi + xi * qr;
                    }
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
"""


def pair_tiles(desc, n_g, tp):
    """Group consecutive outputs with the same (fa, ca, cb, e) and consecutive fb."""
    fa, fb, ca, cb, e = desc
    tiles = []
    o, n = n_g, fa.size
    while o < n:
        t = 1
        while (
            t < tp
            and o + t < n
            and fa[o + t] == fa[o]
            and ca[o + t] == ca[o]
            and cb[o + t] == cb[o]
            and e[o + t] == e[o]
            and fb[o + t] == fb[o] + t
        ):
            t += 1
        tiles.append((fa[o], ca[o], fb[o], cb[o], e[o], o, t))
        o += t
    return np.asarray(tiles, dtype=np.int32).T


def sweep_tiled(env):
    """Round 1: register tiling (T multipoles / TP pair products per thread)."""
    (
        n_f,
        E,
        P,
        O,
        n_g,
        nb,
        red_desc,
        dev,
        vre,
        vim,
        wts,
        spin,
        run,
        compare,
        t0,
        t1,
    ) = (
        env[k]
        for k in (
            "n_f",
            "E",
            "P",
            "O",
            "n_g",
            "nb",
            "red_desc",
            "dev",
            "vre",
            "vim",
            "wts",
            "spin",
            "run",
            "compare",
            "t0",
            "t1",
        )
    )
    for T, TP, NB in (
        (3, 2, 32),
        (5, 4, 32),
        (5, 4, 16),
        (9, 4, 32),
        (9, 8, 16),
        (15, 4, 16),
    ):
        n_gt = n_f * -(-P // T)
        NGT = -(-n_gt // 32) * 32
        tiles = pair_tiles(red_desc, n_g, TP)
        n_pt = tiles.shape[1]
        block = -(-(NGT + n_pt) // 32) * 32
        name = f"tiled<float,float,double,{n_f},{E},{NB},{T},{TP},{NGT}>"
        kern = cp.RawModule(
            code=TILED_SRC, options=("-std=c++14",), name_expressions=[name]
        ).get_function(name)
        tl = [cp.asarray(np.ascontiguousarray(r)) for r in tiles]

        def launch(c0, cc, out, kern=kern, block=block, n_gt=n_gt, n_pt=n_pt, tl=tl):
            kern(
                (cc,),
                (block,),
                (
                    dev["offsets"],
                    np.int64(c0),
                    np.int32(nb),
                    dev["nbr"],
                    dev["phi_c"],
                    dev["phi_n"],
                    vre,
                    vim,
                    wts,
                    np.int64(vre.shape[1]),
                    spin,
                    np.int32(n_gt),
                    np.int32(n_pt),
                    *tl,
                    np.int32(O),
                    out,
                ),
            )

        dt, a, b = run(launch, O)
        print(
            f"tiled T={T:2d} TP={TP} NB={NB:2d} block={block:3d}: {dt:.3f} s "
            f"({t0 / dt:.2f}x P0, {t1 / dt:.2f}x P1), {compare(a, b)}",
            flush=True,
        )


def sweep_staging(env):
    """Round 2: half power table, two threads per staged neighbour, shapes."""
    n_f, E, O, nb, d, dev, vre, vim, wts, spin, run, compare, t0, t1 = (
        env[k]
        for k in (
            "n_f",
            "E",
            "O",
            "nb",
            "d",
            "dev",
            "vre",
            "vim",
            "wts",
            "spin",
            "run",
            "compare",
            "t0",
            "t1",
        )
    )
    for B, NB, half, split in [
        (128, 32, 1, 0),
        (128, 32, 0, 1),
        (128, 32, 1, 1),
        (128, 16, 1, 1),
        (128, 64, 1, 1),
        (256, 32, 1, 1),
        (256, 64, 1, 1),
        (64, 32, 1, 1),
        (64, 16, 1, 1),
        (256, 64, 1, 0),
        (128, 64, 1, 0),
    ]:
        K = -(-O // B)
        name = (
            f"v2<float,float,double,{n_f},{E},{O},{NB},{K},"
            f"{'true' if half else 'false'},{'true' if split else 'false'}>"
        )
        kern = cp.RawModule(
            code=V2_SRC, options=("-std=c++14",), name_expressions=[name]
        ).get_function(name)

        def launch(c0, cc, out, kern=kern, B=B):
            kern(
                (cc,),
                (B,),
                (
                    dev["offsets"],
                    np.int64(c0),
                    np.int32(nb),
                    dev["nbr"],
                    dev["phi_c"],
                    dev["phi_n"],
                    vre,
                    vim,
                    wts,
                    np.int64(vre.shape[1]),
                    spin,
                    *d[:5],
                    out,
                ),
            )

        dt, a, b = run(launch, O)
        print(
            f"staging block={B:3d} NB={NB:2d} half={half} split={split}: "
            f"{dt:.3f} s ({t0 / dt:.2f}x P0, {t1 / dt:.2f}x P1), {compare(a, b)}",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nside", type=int, default=512)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--n-max", type=int, default=20)
    ap.add_argument(
        "--all",
        action="store_true",
        help="also the register-tiled and staging variants (rounds 1-2)",
    )
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    mask = hp.ud_grade(hp.read_map(args.mask, dtype=np.float64), args.nside) >= 0.5
    est = Global3PCF(
        args.nside,
        mask=mask,
        nbins=12,
        theta_min=15,
        theta_max=250,
        n_max=args.n_max,
        device="gpu",
        map_precision="float32",
        rotation_precision="float32",
        accumulation_precision="float64",
        centre_chunk=args.chunk,
    )
    t = time.perf_counter()
    est.find_pairs()
    est.prepare()
    print(
        f"nside {args.nside}: {est.ntotpairs:.3e} ordered pairs, {est.centre_rows.size} centres, "
        f"find+prepare {time.perf_counter() - t:.0f} s",
        flush=True,
    )
    rng = np.random.default_rng(0)
    n = est.n_active
    g = rng.normal(size=(4, 2, n)).astype(np.float32) * 0.3
    w = rng.uniform(0.5, 1.5, size=(4, n)).astype(np.float32)
    f = est._field("g", g, w, 2, signs=(-1.0, 1.0))
    dev = est._device()
    be = est.backend
    n_f, E, nb = f.nz, est.E, est.nbins
    vre, vim, wts = (cp.ascontiguousarray(a) for a in (f.vre, f.vim, f.w))
    spin = cp.asarray(np.full(n_f, 2, np.int32))
    n_c = int(est.centre_rows.size)
    C = args.chunk

    def run(launch, n_out, keep=None):
        """All centres; returns (seconds, outputs of the first chunk[:, :, keep])."""
        first = None
        launch(
            0, min(C, n_c), cp.zeros((min(C, n_c), nb, n_out), cp.complex128)
        )  # compile
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        for c0 in range(0, n_c, C):
            cc = min(C, n_c - c0)
            out = cp.zeros((cc, nb, n_out), cp.complex128)
            launch(c0, cc, out)
            if first is None:
                first = out
        cp.cuda.Device().synchronize()
        dt = time.perf_counter() - t0
        sample = first if keep is None else first[:, :, keep]
        # bitwise over every chunk too: a checksum of the raw bits of the last one
        return dt, sample, out if keep is None else out[:, :, keep]

    def shipped(desc):
        d = tuple(cp.asarray(a) for a in desc)

        def launch(c0, cc, out):
            be.kernel_3pcf_multipoles(
                dev["offsets"],
                c0,
                cc,
                nb,
                dev["nbr"],
                dev["phi_c"],
                dev["phi_n"],
                vre,
                vim,
                wts,
                spin,
                d,
                E,
                out,
            )

        return launch

    every = [(s, t) for s in (-2, 0, 2) for t in (0, 1)]  # the <= 7.2 table
    full_desc, n_g = est._descriptor(n_f, every)
    combos = sorted([(2, 0), (-2, 0), (2, 1), (-2, 1)])  # what ggg's jobs read
    red_desc, _ = est._descriptor(n_f, combos)
    # positions of the reduced outputs inside the full table
    sig_i = {-2: 0, 0: 1, 2: 2}
    keep = list(range(n_g)) + [
        n_g + ((sig_i[s] * 2 + typ) * n_f + x) * n_f + y
        for s, typ in combos
        for x in range(n_f)
        for y in range(n_f)
    ]
    t0, ref_first, ref_last = run(shipped(full_desc), int(full_desc[0].size), keep)
    flops = est.ntotpairs * 8  # per complex MAC: 4 mul + 4 add
    print(
        f"P0 shipped, O={full_desc[0].size}: {t0:.3f} s "
        f"({flops * full_desc[0].size / t0 / 1e12:.2f} TFLOP/s)",
        flush=True,
    )
    t1, a, b = run(shipped(red_desc), int(red_desc[0].size))
    same = bool(cp.array_equal(a, ref_first)) and bool(cp.array_equal(b, ref_last))
    print(
        f"P1 shipped kernel, O={red_desc[0].size}: {t1:.3f} s, bitwise == P0: {same}",
        flush=True,
    )

    def compare(a, b):
        if bool(cp.array_equal(a, ref_first)) and bool(cp.array_equal(b, ref_last)):
            return "bitwise == P0"
        rel = max(
            float(cp.abs(x - y).max() / cp.abs(y).max())
            for x, y in ((a, ref_first), (b, ref_last))
        )
        return f"NOT bitwise, max rel diff {rel:.1e}"

    O = int(red_desc[0].size)
    P = 2 * E + 1
    d = tuple(cp.asarray(x) for x in red_desc)
    if args.all:
        sweep_tiled(locals())
        sweep_staging(locals())
    # thread -> output maps: "shipped" = identity; "m-major" = the G outputs of
    # one m for all fields on neighbouring threads (power loads become
    # broadcasts), pair outputs grouped by (sigma, type) as in the table.
    fa_h = red_desc[0]
    e_h = red_desc[4]
    g_order = sorted(range(n_g), key=lambda o: (e_h[o], fa_h[o]))
    maps = {
        "shipped": np.arange(O, dtype=np.int32),
        "m-major": np.asarray(g_order + list(range(n_g, O)), dtype=np.int32),
    }
    for B, NB, v2, mp, fma, abl in [
        (128, 32, 0, "shipped", 0, 0),
        (128, 32, 0, "shipped", 1, 0),
        (128, 32, 1, "m-major", 1, 0),
        (256, 32, 1, "m-major", 1, 0),
        (128, 32, 0, "shipped", 0, 1),
        (128, 32, 0, "shipped", 0, 2),
        (128, 32, 0, "shipped", 1, 2),
    ]:
        K = -(-O // B)
        name = (
            f"v3<float,float,double,{n_f},{E},{O},{NB},{K},{'true' if v2 else 'false'},"
            f"{'true' if fma else 'false'},{abl}>"
        )
        kern = cp.RawModule(
            code=V3_SRC, options=("-std=c++14",), name_expressions=[name]
        ).get_function(name)
        perm = cp.asarray(maps[mp])

        def launch(c0, cc, out, kern=kern, B=B, perm=perm):
            kern(
                (cc,),
                (B,),
                (
                    dev["offsets"],
                    np.int64(c0),
                    np.int32(nb),
                    dev["nbr"],
                    dev["phi_c"],
                    dev["phi_n"],
                    vre,
                    vim,
                    wts,
                    np.int64(vre.shape[1]),
                    spin,
                    *d[:5],
                    perm,
                    out,
                ),
            )

        dt, a, b = run(launch, O)
        tag = {
            0: "",
            1: "  [ablation: no accumulation]",
            2: "  [ablation: no staging]",
        }[abl]
        print(
            f"v3 block={B:3d} NB={NB:2d} vec2={v2} map={mp:8s} fma={fma}: {dt:.3f} s  "
            f"({t0 / dt:.2f}x P0, {t1 / dt:.2f}x P1), "
            f"{compare(a, b) if abl == 0 else '-'}{tag}",
            flush=True,
        )


if __name__ == "__main__":
    main()
