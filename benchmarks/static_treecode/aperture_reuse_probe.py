#!/usr/bin/env python
"""Does `gpu_aperture_shear_tomo` pay for re-reading the disc geometry per bin?

The shipped kernel runs one block per (patch, tomo bin), so it re-reads the
aperture disc geometry -- q_inds + q_cos + q_sin + q_val, 16 B per disc pixel
-- once for every tomographic bin, while using 12 B of map data per visit.
At nz = 4 that is 112 B/pixel where one block per patch would need 64.

`ncu` cannot answer whether those re-reads actually reach DRAM or are caught
by L2: profiling counters need root on seitz1 (ERR_NVGPUCTRPERM).  So this
measures the end state directly instead, which is the decisive number anyway.
Two probe kernels are compiled here and never touch the library:

  probe_current  one block per (patch, bin)  -- a replica of the shipped kernel
  probe_fused    one block per patch, bins looped inside, NZ a template arg

Both take the same four-element-stride map addressing as `aperture_tomo.cu`,
so `probe_fused` is run twice: once on SoA maps (nz, 2, n_rows) and once on
AoS (n_rows, nz, 2).  AoS matters here because the restructure changes its
verdict -- AoS cost the *current* kernel 0.450 -> 0.600 ms (consecutive
pixels land nz*2 apart), but with one block per patch a thread holds `pix`
and wants that pixel's whole (nz, 2) column, which AoS makes contiguous.

The per-thread accumulators are a compile-time-sized array (template<int NZ>):
run-time indexing spills them to local memory, which is what made the first
combination-tiled pair kernel 3x slower.

Everything is checked bitwise against the shipped kernel first.  That is not
just a correctness check -- `resolution_factor=None` must stay bit-for-bit
identical to 4.20.0, so if the restructure cannot be bitwise it cannot ship,
and this script is where that is established.

Usage (on seitz1):
    PYTHONPATH=src python benchmarks/static_treecode/aperture_reuse_probe.py \
        --nside 512 --nz 4
"""

from __future__ import annotations

import argparse
import json
import time

import healpy as hp
import numpy as np

from CosmoFuse import Correlation

PROBE_SOURCE = r"""
__COMMON_CUDA_SOURCE__

/* Replica of gpu_aperture_shear_tomo: one block per (patch, bin). */
template<typename T, typename QT>
__global__ void probe_current(
    const T* g1, const T* g2,
    const long long g_stride, const long long g_elem,
    const T* weights, const long long w_stride, const long long w_elem,
    const unsigned int* q_inds, const QT* q_cos, const QT* q_sin,
    const QT* q_val, const long long* q_offsets, const QT* q_patch_area,
    T* out_num, T* out_den, const int npatches, const int ntomo)
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

/*
 * One block per patch, all NZ bins inside.  The disc geometry is read once
 * per pixel instead of NZ times; the thread->pixel mapping and the reduction
 * tree are unchanged, so every bin sums exactly the same partials in exactly
 * the same order as probe_current -- the result is bitwise identical.
 */
template<typename T, typename QT, int NZ>
__global__ void probe_fused(
    const T* g1, const T* g2,
    const long long g_stride, const long long g_elem,
    const T* weights, const long long w_stride, const long long w_elem,
    const unsigned int* q_inds, const QT* q_cos, const QT* q_sin,
    const QT* q_val, const long long* q_offsets, const QT* q_patch_area,
    T* out_num, T* out_den, const int npatches, const int ntomo)
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
        const T qc = (T)q_cos[idx];
        const T qs = (T)q_sin[idx];
        const T qv = (T)q_val[idx];
#pragma unroll
        for (int b = 0; b < NZ; ++b) {
            const T wv = weights[(long long)b * w_stride + pix * w_elem];
            const T gt = -g1[(long long)b * g_stride + pix * g_elem] * qc
                       - g2[(long long)b * g_stride + pix * g_elem] * qs;
            sn[b] += wv * gt * qv;
            sd[b] += wv;
        }
    }

    const T area = (T)q_patch_area[patch];
#pragma unroll
    for (int b = 0; b < NZ; ++b) {
        T n, d;
        __syncthreads();          /* the reduction buffers are reused */
        block_reduce_sum_pair(sn[b], sd[b], &n, &d);
        if (lane == 0) {
            const long long o = (long long)b * npatches + patch;
            out_num[o] = area * n;
            out_den[o] = d;
        }
    }
}
"""


def build(args):
    """A Correlation with the aperture geometry only -- no pair search.

    The pair search is what makes nside 2048 expensive, and this kernel
    never touches a pair: it walks Q_inds.  So `_ensure_aperture_pairs` +
    `_prepare_aperture_device_buffers` is all that is needed, which keeps
    the 2048 point affordable.
    """
    mask = np.zeros(hp.nside2npix(args.nside), dtype=bool)
    theta, phi = hp.pix2ang(args.nside, np.arange(mask.size))
    mask[(np.degrees(phi) < 90.0) & (np.abs(90 - np.degrees(theta)) < 40)] = True
    rng = np.random.default_rng(0)
    mask[rng.choice(mask.size, mask.size // 50, replace=False)] = False
    corr = Correlation.from_mask(
        args.nside, mask, args.nside_centers,
        patch_size=args.patch_size, theta_Q=args.theta_Q, f_mask=0.3,
        nbins=args.nbins, theta_min=args.theta_min, theta_max=args.theta_max,
        device=0, map_precision=args.map_precision,
        rotation_precision="float32", accumulation_precision="float64",
        aperture_nside=args.aperture_nside,
    )
    t0 = time.time()
    corr.calculate_pairs_M_a()      # `_ensure_aperture_pairs` needs a
    corr._prepare_aperture_flat()   # preprocess() to have run first
    corr._prepare_aperture_device_buffers()
    print(f"aperture geometry: {time.time() - t0:.1f} s", flush=True)
    return corr


def timeit(fn, cupy, n=30, warmup=5):
    for _ in range(warmup):
        fn()
    cupy.cuda.runtime.deviceSynchronize()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        cupy.cuda.runtime.deviceSynchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nside", type=int, default=512)
    ap.add_argument("--nz", type=int, default=4)
    ap.add_argument("--nz-scan", type=int, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--aperture-nside", type=int, default=None)
    ap.add_argument("--nbins", type=int, default=7)
    ap.add_argument("--theta-min", type=float, default=18.0)
    ap.add_argument("--theta-max", type=float, default=148.0)
    ap.add_argument("--patch-size", type=float, default=110.0)
    ap.add_argument("--theta-Q", type=float, default=110.0)
    ap.add_argument("--nside-centers", type=int, default=16)
    ap.add_argument("--map-precision", default="float32")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import cupy

    corr = build(args)
    ctx = corr.compute_context
    q_inds = ctx.Q_inds_dev
    geom_tail = (ctx.Q_cos_dev, ctx.Q_sin_dev, ctx.Q_val_dev,
                 ctx.Q_offsets_dev, ctx.Q_patch_area_dev)
    npatches = int(ctx.Q_offsets_dev.shape[0] - 1)
    n_visits = int(q_inds.shape[0])
    n_rows = int(q_inds.max()) + 1
    map_dt = getattr(cupy, corr.map_dtype.name)
    map_c = "float" if corr.map_dtype == np.float32 else "double"
    q_c = "float" if ctx.Q_cos_dev.dtype == cupy.float32 else "double"

    print(f"nside={args.nside} patches={npatches} disc-pixel visits={n_visits} "
          f"({n_visits / npatches:.0f}/patch) rows={n_rows} "
          f"aperture_nside={corr.aperture_nside}", flush=True)

    from CosmoFuse.backend import _COMMON_CUDA_SOURCE
    source = PROBE_SOURCE.replace("__COMMON_CUDA_SOURCE__", _COMMON_CUDA_SOURCE)
    mod = cupy.RawModule(code=source, options=("-std=c++14",),
                         name_expressions=[f"probe_current<{map_c}, {q_c}>"]
                         + [f"probe_fused<{map_c}, {q_c}, {nz}>"
                            for nz in sorted(set(args.nz_scan) | {args.nz})])
    k_current = mod.get_function(f"probe_current<{map_c}, {q_c}>")

    rng = np.random.default_rng(3)
    report = {"config": vars(args), "npatches": npatches,
              "disc_pixel_visits": n_visits, "n_rows": n_rows,
              "gpu": cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()}

    def buffers(nz):
        g = cupy.asarray(rng.normal(size=(nz, 2, n_rows)).astype(corr.map_dtype) * 0.3)
        w = cupy.asarray(
            rng.uniform(0.2, 2.0, size=(nz, n_rows)).astype(corr.map_dtype))
        g_aos = cupy.ascontiguousarray(cupy.transpose(g, (2, 0, 1)))   # (rows, nz, 2)
        w_aos = cupy.ascontiguousarray(w.T)                            # (rows, nz)
        return g, w, g_aos, w_aos

    def launch(kern, grid, g1, g2, g_stride, g_elem, w, w_stride, w_elem,
               out_num, out_den, nz):
        kern(grid, (256,),
             (g1, g2, np.int64(g_stride), np.int64(g_elem),
              w, np.int64(w_stride), np.int64(w_elem),
              q_inds, *geom_tail, out_num, out_den,
              np.int32(npatches), np.int32(nz)))

    # ---------------- correctness + the headline timing at --nz ----------
    nz = args.nz
    g, w, g_aos, w_aos = buffers(nz)
    outs = {k: (cupy.zeros((nz, npatches), dtype=map_dt),
                cupy.zeros((nz, npatches), dtype=map_dt))
            for k in ("lib", "current", "fused_soa", "fused_aos")}

    ok = corr.backend.aperture_tomo_shear_kernel(
        g[:, 0], g[:, 1], w, q_inds, *geom_tail, *outs["lib"])
    assert ok, "the shipped aperture kernel declined the launch"

    k_fused = mod.get_function(f"probe_fused<{map_c}, {q_c}, {nz}>")
    launch(k_current, (npatches, nz, 1), g[:, 0], g[:, 1], 2 * n_rows, 1,
           w, n_rows, 1, *outs["current"], nz)
    launch(k_fused, (npatches, 1, 1), g[:, 0], g[:, 1], 2 * n_rows, 1,
           w, n_rows, 1, *outs["fused_soa"], nz)
    launch(k_fused, (npatches, 1, 1), g_aos[:, :, 0], g_aos[:, :, 1], 2, nz * 2,
           w_aos, 1, nz, *outs["fused_aos"], nz)
    cupy.cuda.runtime.deviceSynchronize()

    bitwise = {}
    for name in ("current", "fused_soa", "fused_aos"):
        bitwise[name] = {
            "num": bool(cupy.array_equal(outs[name][0], outs["lib"][0])),
            "den": bool(cupy.array_equal(outs[name][1], outs["lib"][1])),
            "max_rel": float(cupy.abs(outs[name][0] - outs["lib"][0]).max()
                             / cupy.abs(outs["lib"][0]).max()),
        }
    report["bitwise_vs_shipped_kernel"] = bitwise
    print(json.dumps(bitwise, indent=1), flush=True)

    rep = args.repeats
    on, od = outs["current"]
    ms = {
        "shipped_lib_kernel": timeit(
            lambda: corr.backend.aperture_tomo_shear_kernel(
                g[:, 0], g[:, 1], w, q_inds, *geom_tail, on, od), cupy, n=rep),
        "probe_current": timeit(
            lambda: launch(k_current, (npatches, nz, 1), g[:, 0], g[:, 1],
                           2 * n_rows, 1, w, n_rows, 1, on, od, nz), cupy, n=rep),
        "probe_fused_soa": timeit(
            lambda: launch(k_fused, (npatches, 1, 1), g[:, 0], g[:, 1],
                           2 * n_rows, 1, w, n_rows, 1, on, od, nz), cupy, n=rep),
        "probe_fused_aos": timeit(
            lambda: launch(k_fused, (npatches, 1, 1), g_aos[:, :, 0],
                           g_aos[:, :, 1], 2, nz * 2, w_aos, 1, nz, on, od, nz),
            cupy, n=rep),
    }
    base = ms["probe_current"]
    report["ms"] = ms
    report["speedup_vs_current"] = {k: base / v for k, v in ms.items()}
    print(json.dumps({"ms": ms, "speedup": report["speedup_vs_current"]},
                     indent=1), flush=True)

    # ---------------- nz scaling: does the geometry re-read reach DRAM? ---
    # current:  t = L + c*(16 + 12)*nz      (geometry refetched every bin)
    # fused:    t = L + c*16 + c*12*nz      (geometry read once)
    # so the intercept/slope ratio of the *current* kernel separates the two.
    def scan_point(nzs):
        """One nz point.  A function so its buffers are freed on return --
        at nside 2048 each set is 250 MB."""
        gs, ws, gsa, wsa = buffers(nzs)
        o1 = cupy.zeros((nzs, npatches), dtype=map_dt)
        o2 = cupy.zeros((nzs, npatches), dtype=map_dt)
        kf = mod.get_function(f"probe_fused<{map_c}, {q_c}, {nzs}>")
        return {
            "current": timeit(
                lambda: launch(k_current, (npatches, nzs, 1), gs[:, 0],
                               gs[:, 1], 2 * n_rows, 1, ws, n_rows, 1,
                               o1, o2, nzs), cupy, n=rep),
            "fused_soa": timeit(
                lambda: launch(kf, (npatches, 1, 1), gs[:, 0], gs[:, 1],
                               2 * n_rows, 1, ws, n_rows, 1, o1, o2, nzs),
                cupy, n=rep),
            "fused_aos": timeit(
                lambda: launch(kf, (npatches, 1, 1), gsa[:, :, 0],
                               gsa[:, :, 1], 2, nzs * 2, wsa, 1, nzs,
                               o1, o2, nzs), cupy, n=rep),
        }

    scan = {nzs: scan_point(nzs) for nzs in sorted(set(args.nz_scan))}
    report["nz_scan_ms"] = scan
    xs = np.array(sorted(scan), dtype=float)
    for variant in ("current", "fused_soa", "fused_aos"):
        ys = np.array([scan[int(x)][variant] for x in xs])
        slope, intercept = np.polyfit(xs, ys, 1)
        report[f"fit_{variant}"] = {
            "slope_ms_per_bin": float(slope), "intercept_ms": float(intercept),
            "intercept_over_slope": float(intercept / slope) if slope else None,
        }
    print(json.dumps({k: v for k, v in report.items()
                      if k.startswith("fit_") or k == "nz_scan_ms"}, indent=1),
          flush=True)

    out = args.out or (f"benchmarks/static_treecode/aperture_reuse_probe_"
                       f"{args.nside}.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
