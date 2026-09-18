"""A100 gate for the fused treecode degrade (idea #3).

Runs the real CUDA kernels against the sparse chain they replace:

  * accuracy -- every public method, fused vs sparse, on the same maps;
  * the virtual rows themselves, which is where any error would start;
  * speed -- `_expand_rows` alone and the whole device-resident call.

Usage (on seitz1):
    PYTHONPATH=src python benchmarks/static_treecode/degrade_kernel_gate.py \
        --nside 512 --k 2.9 --nz 4
"""

import argparse
import json
import time

import healpy as hp
import numpy as np

from CosmoFuse import Correlation


def build(args):
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
        resolution_factor=args.k,
    )
    corr.preprocess()
    return corr


def timeit(fn, cupy, n=20, warmup=3):
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
    ap.add_argument("--k", type=float, default=2.9)
    ap.add_argument("--nz", type=int, default=4)
    ap.add_argument("--nbins", type=int, default=7)
    ap.add_argument("--theta-min", type=float, default=18.0)
    ap.add_argument("--theta-max", type=float, default=148.0)
    ap.add_argument("--patch-size", type=float, default=110.0)
    ap.add_argument("--theta-Q", type=float, default=110.0)
    ap.add_argument("--nside-centers", type=int, default=16)
    ap.add_argument("--map-precision", default="float32")
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()

    import cupy

    corr = build(args)
    kern = corr.backend.degrade_rows_kernel
    print(f"patches={corr.n_patches} n_active={corr.n_active} "
          f"n_appended={corr.n_appended} pairs={corr.ntotpairs} "
          f"levels={np.asarray(corr.level_nside).tolist()}")
    print(f"degrade kernel present: {kern is not None}")

    rng = np.random.default_rng(3)
    shear = cupy.asarray(
        rng.normal(size=(args.nz, 2, corr.n_active)).astype(corr.map_dtype) * 0.3)
    dens = cupy.asarray(
        rng.normal(size=(args.nz, corr.n_active)).astype(corr.map_dtype))
    w = cupy.asarray(
        rng.uniform(0.2, 2.0, size=(args.nz, corr.n_active)).astype(corr.map_dtype))

    def with_fused(flag):
        corr._use_fused_degrade = (
            (lambda weights: flag and corr.backend.degrade_rows_kernel is not None
             and not isinstance(weights, np.ndarray)))
        corr.compute_context.degrade_ops = None
        corr.compute_context.degrade_csr = None

    report = {"config": vars(args), "n_patches": int(corr.n_patches),
              "n_appended": int(corr.n_appended),
              "ntotpairs": int(corr.ntotpairs)}

    # ---- accuracy: the virtual rows themselves -----------------------
    with_fused(True)
    (g1_f, g2_f), w_f = corr._expand_rows((shear[:, 0], shear[:, 1]), w, "all")
    with_fused(False)
    (g1_s, g2_s), w_s = corr._expand_rows((shear[:, 0], shear[:, 1]), w, "all")
    rows = {}
    for name, a, b in (("g1", g1_f, g1_s), ("g2", g2_f, g2_s), ("w", w_f, w_s)):
        a, b = cupy.asnumpy(a), cupy.asnumpy(b)
        scale = float(np.max(np.abs(b)))
        rows[name] = {
            "max_abs": float(np.max(np.abs(a - b))),
            "max_rel_to_scale": float(np.max(np.abs(a - b)) / scale),
            "bitwise": bool(np.array_equal(a, b)),
        }
    report["rows"] = rows
    print("virtual rows fused vs sparse:", json.dumps(rows, indent=2))

    # ---- accuracy: the public methods --------------------------------
    cases = {
        "get_full_tomo_shear": lambda: corr.get_full_tomo_shear(
            shear, w, return_device=False),
        "vectorized_shear_shear": lambda: corr.vectorized_shear_shear(
            shear, w, return_device=False),
        "get_full_tomo_density": lambda: corr.get_full_tomo_density(
            dens, w, return_device=False),
        "get_full_tomo_ggl": lambda: corr.get_full_tomo_ggl(
            dens, shear, w, w, return_device=False),
        "get_3x2pt_tomo": lambda: corr.get_3x2pt_tomo(
            shear_maps=shear, density_maps=dens,
            weights={"shear": w, "density": w}, return_device=False),
    }
    methods = {}
    for name, call in cases.items():
        with_fused(True)
        got = call()
        with_fused(False)
        want = call()
        got = got if isinstance(got, tuple) else (got,)
        want = want if isinstance(want, tuple) else (want,)
        worst = 0.0
        for a, b in zip(got, want):
            a, b = np.asarray(a), np.asarray(b)
            scale = float(np.max(np.abs(b))) or 1.0
            worst = max(worst, float(np.max(np.abs(a - b))) / scale)
        methods[name] = worst
        print(f"  {name:26s} max |fused-sparse| / scale = {worst:.3e}")
    report["methods"] = methods

    # ---- layouts and the folded sign flip ------------------------------
    # Both are pure data movement, so the bar is bitwise equality with the
    # transpose-and-flip-afterwards code they replace.
    with_fused(True)
    layouts = {}
    soa_g, soa_w = corr._expand_shear_rows(shear, w, blocks="pairs")
    aos_g, aos_w = corr._expand_shear_rows(shear, w, blocks="pairs",
                                           layout="aos")
    layouts["aos_values_bitwise"] = bool(np.array_equal(
        cupy.asnumpy(aos_g), np.transpose(cupy.asnumpy(soa_g), (2, 0, 1))))
    layouts["aos_weights_bitwise"] = bool(np.array_equal(
        cupy.asnumpy(aos_w), np.transpose(cupy.asnumpy(soa_w), (1, 0))))
    for signs in ((-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)):
        folded, _ = corr._expand_shear_rows(
            shear, w, blocks="pairs", signs=signs, layout="aos")
        pre, _ = corr._expand_shear_rows(
            shear * cupy.asarray(np.asarray(signs).reshape(1, 2, 1),
                                 dtype=corr.map_dtype),
            w, blocks="pairs", layout="aos")
        layouts[f"signs_{signs[0]:+.0f}{signs[1]:+.0f}_bitwise"] = bool(
            np.array_equal(cupy.asnumpy(folded), cupy.asnumpy(pre)))
    # ... and end to end, where the aperture kernel's new element stride
    # and the packed gather are also exercised.
    for name, call in (
        ("get_full_tomo_shear", lambda sh, f: corr.get_full_tomo_shear(
            sh, w, flip_g1=f, return_device=False)),
        ("get_3x2pt_tomo", lambda sh, f: corr.get_3x2pt_tomo(
            shear_maps=sh, density_maps=dens,
            weights={"shear": w, "density": w}, flip_g1=f,
            return_device=False)),
    ):
        flagged = call(shear, True)
        pre_flipped = call(
            shear * cupy.asarray(np.asarray([-1.0, 1.0]).reshape(1, 2, 1),
                                 dtype=corr.map_dtype), False)
        flagged = flagged if isinstance(flagged, tuple) else (flagged,)
        pre_flipped = pre_flipped if isinstance(pre_flipped, tuple) else (pre_flipped,)
        worst = 0.0
        for a, b in zip(flagged, pre_flipped):
            a, b = np.asarray(a), np.asarray(b)
            scale = float(np.max(np.abs(b))) or 1.0
            worst = max(worst, float(np.max(np.abs(a - b))) / scale)
        layouts[f"flip_{name}"] = worst
    report["layouts"] = layouts
    print("layouts and signs:", json.dumps(layouts, indent=2))

    # ---- speed --------------------------------------------------------
    speed = {}
    for label, flag in (("sparse", False), ("fused", True)):
        with_fused(flag)
        speed[f"expand_rows_{label}_ms"] = timeit(
            lambda: corr._expand_rows((shear[:, 0], shear[:, 1]), w, "all"),
            cupy, n=args.repeats)
        speed[f"full_tomo_shear_{label}_ms"] = timeit(
            lambda: corr.get_full_tomo_shear(shear, w, return_device=True),
            cupy, n=args.repeats)
        speed[f"3x2pt_{label}_ms"] = timeit(
            lambda: corr.get_3x2pt_tomo(
                shear_maps=shear, density_maps=dens,
                weights={"shear": w, "density": w}, return_device=True),
            cupy, n=args.repeats)
    for key in ("expand_rows", "full_tomo_shear", "3x2pt"):
        a, b = speed[f"{key}_sparse_ms"], speed[f"{key}_fused_ms"]
        speed[f"{key}_speedup"] = a / b
        print(f"  {key:20s} sparse {a:8.3f} ms -> fused {b:8.3f} ms  "
              f"({a / b:.2f}x)")
    report["speed"] = speed

    with open("degrade_kernel_gate.json", "w") as fp:
        json.dump(report, fp, indent=2)
    print("wrote degrade_kernel_gate.json")


if __name__ == "__main__":
    main()
