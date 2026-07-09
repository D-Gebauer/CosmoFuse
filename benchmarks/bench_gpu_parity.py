"""GPU↔CPU parity and timing harness for the CosmoFuse measurement paths.

Run this BEFORE and AFTER every GPU kernel change:

    python benchmarks/bench_gpu_parity.py --device 0 --out after.json
    python benchmarks/bench_gpu_parity.py --compare before.json after.json

The CPU backend is the validated reference (its outputs are covered by the
treecorr end-to-end tests). For every public measurement path this script
computes the CPU result and the GPU result on identical inputs and identical
precomputed pairs, reports scale-relative differences, and records steady-state
per-map timings.

Tolerances (float64 maps, complex64 rotations — the defaults):
  parity:   max|gpu - cpu| / rms(cpu)  <=  1e-8   (reduction-order roundoff)
  With map_precision="float32" use 2e-4 instead.

Runs on a CPU-only machine as a self-check (GPU section is skipped).
"""
import argparse
import json
import os
import sys
import time

os.environ.pop("NUMBA_DISABLE_JIT", None)

import numpy as np


def build_correlation(device):
    import healpy as hp
    from CosmoFuse.correlations import Correlation

    nside = 512
    n_patches = 56
    grid = 8
    lon = np.linspace(10, 10 + 4 * grid, grid, endpoint=False)
    lat = np.linspace(-14, -14 + 4 * (n_patches // grid), n_patches // grid, endpoint=False)
    LON, LAT = np.meshgrid(lon, lat)
    phi = np.radians(LON.ravel()[:n_patches])
    theta = np.pi / 2 - np.radians(LAT.ravel()[:n_patches])
    corr = Correlation(
        nside, phi, theta, nbins=10, theta_min=10, theta_max=170,
        patch_size=90, theta_Q=90, device=device,
    )
    return corr, hp.nside2npix(nside)


def make_maps(npix, nzbins=5, seed=12345):
    rng = np.random.default_rng(seed)
    return {
        "shear": rng.standard_normal((nzbins, 2, npix)) * 0.02,
        "density": rng.standard_normal((nzbins, npix)) * 0.1,
        "w_shear": 0.5 + rng.random((nzbins, npix)),
        "w_density": 0.5 + rng.random((nzbins, npix)),
    }


def run_paths(corr, maps, reps=4):
    """Run every measurement path; return ({name: ndarray}, {name: seconds})."""
    out, t = {}, {}

    def bench(name, fn, *args, **kwargs):
        res = fn(*args, **kwargs)          # warm (JIT / NVRTC / caches)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            res = fn(*args, **kwargs)
            times.append(time.perf_counter() - t0)
        t[name] = float(np.median(times))
        return res

    S, D, WS, WD = maps["shear"], maps["density"], maps["w_shear"], maps["w_density"]
    to_np = lambda a: np.asarray(corr.backend.to_numpy(a))

    xip, xim = bench("vectorized_shear_shear", corr.vectorized_shear_shear, S, WS)
    out["xip"], out["xim"] = to_np(xip), to_np(xim)

    wt = bench("vectorized_density_density", corr.vectorized_density_density, D, WD)
    out["wtheta"] = to_np(wt)

    gt = bench("vectorized_density_shear", corr.vectorized_density_shear, D, S, WD, WS)
    out["gammat"] = to_np(gt)

    Ma, xp, xm = bench("get_full_tomo_shear", corr.get_full_tomo_shear, S, WS)
    out["Ma"], out["xip_full"], out["xim_full"] = to_np(Ma), to_np(xp), to_np(xm)

    r = bench("get_3x2pt_tomo", corr.get_3x2pt_tomo, shear_maps=S, density_maps=D,
              weights={"shear": WS, "density": WD})
    for k, v in zip(("f_Ma", "f_Mg", "f_xip", "f_xim", "f_xig", "f_xit"), r):
        out[k] = to_np(v)

    g1, g2, w1 = S[0, 0], S[0, 1], WS[0]
    xp1, xm1 = bench("compute_shear_shear", corr.compute_shear_shear, g1, g2, g1, g2, w1, w1)
    out["xp1"], out["xm1"] = to_np(xp1), to_np(xm1)
    (wt1,) = bench("compute_density_density", corr.compute_density_density,
                   D[0], D[1], WD[0], WD[1])
    out["wt1"] = to_np(wt1)
    (gt1,) = bench("compute_density_shear", corr.compute_density_shear,
                   D[0], g1, g2, WD[0], w1)
    out["gt1"] = to_np(gt1)

    Ma1 = bench("get_aperture_shear", corr.get_aperture_shear, g1, g2, w1,
                return_device=False)
    out["Ma1"] = np.asarray(Ma1)
    Mg1 = bench("get_aperture_density", corr.get_aperture_density, D[0], WD[0],
                return_device=False)
    out["Mg1"] = np.asarray(Mg1)

    return out, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0", help="GPU id (int) or 'cpu'")
    ap.add_argument("--pairs-file", default=None,
                    help="optional .h5 pair file (otherwise computed once and shared)")
    ap.add_argument("--out", default=None, help="write timings JSON here")
    ap.add_argument("--rtol", type=float, default=1e-8,
                    help="scale-relative parity tolerance (2e-4 for float32 maps)")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                    help="compare two timing JSON files and exit")
    args = ap.parse_args()

    if args.compare:
        before = json.load(open(args.compare[0]))
        after = json.load(open(args.compare[1]))
        print(f"{'path':32s} {'before':>10s} {'after':>10s} {'speedup':>8s}")
        for k in before["timings"]:
            b, a = before["timings"][k], after["timings"].get(k)
            if a:
                print(f"{k:32s} {b*1000:9.1f}ms {a*1000:9.1f}ms {b/a:7.2f}x")
        return 0

    # CPU reference
    corr_cpu, npix = build_correlation("cpu")
    if args.pairs_file:
        corr_cpu.load_pairs(args.pairs_file)
    else:
        corr_cpu.calculate_pairs_M_a()
        corr_cpu.calculate_pairs_2PCF()
        corr_cpu.prepare()
        args.pairs_file = "/tmp/cosmofuse_parity_pairs.h5"
        corr_cpu.save_pairs(args.pairs_file)
    maps = make_maps(npix)
    print("== CPU reference ==")
    ref, t_cpu = run_paths(corr_cpu, maps)
    for k, v in t_cpu.items():
        print(f"  {k:32s} {v*1000:9.1f} ms")

    result = {"device": "cpu", "timings": t_cpu}

    # GPU under test
    have_gpu = False
    if str(args.device).lower() != "cpu":
        try:
            import cupy
            cupy.cuda.runtime.getDeviceCount()
            have_gpu = True
        except Exception as exc:
            print(f"\nNo usable GPU ({exc}); parity section skipped.")

    if have_gpu:
        corr_gpu, _ = build_correlation(int(args.device))
        corr_gpu.load_pairs(args.pairs_file)
        print("== GPU under test ==")
        got, t_gpu = run_paths(corr_gpu, maps)
        result = {"device": f"gpu:{args.device}", "timings": t_gpu,
                  "cpu_timings": t_cpu}

        print("\n== parity (scale-relative, vs CPU reference) ==")
        failures = []
        for k, vref in ref.items():
            v = got[k]
            scale = max(float(np.sqrt(np.mean(np.abs(vref) ** 2))), 1e-30)
            diff = float(np.max(np.abs(v - vref))) / scale
            ok = diff <= args.rtol
            if not ok:
                failures.append(k)
            print(f"  {'OK ' if ok else 'FAIL'} {k:12s} max|Δ|/rms = {diff:.3e}")
        if failures:
            print(f"\nPARITY FAILURES: {failures}")
            return 1
        print("\nAll parity checks passed.")

    if args.out:
        json.dump(result, open(args.out, "w"), indent=1)
        print(f"timings written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
