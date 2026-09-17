"""Stage 3/4 GPU gates: static treecode on the A100.

Part A (nside 512, production Q-patch centres, real baryonified maps):
  * T7 on a real GPU: GPU float64 vs CPU float64 (<= 1e-13) and GPU
    float32 + float64 accumulators vs CPU float64 (<= 1e-6), treecode on;
  * per-map-set wall time and device memory for full resolution and k.

Part B (nside 2048, DES Y3 mask, same patch centres, 5'...): geometry fits on
the device? preprocess time, memory, per-map-set time (random maps).
"""

import argparse
import json
import time

import h5py
import healpy as hp
import numpy as np

from CosmoFuse.correlations import Correlation

SBI = "/e/ocean1/users/dgebauer/sbi"
MASK = "/home/moon/dgebauer/research/lfi/local/DESY3_Mask.fits"


def sync(corr):
    if corr.backend.name == "cupy":
        corr.backend.module.cuda.runtime.deviceSynchronize()


def pool_gb(corr):
    if corr.backend.name != "cupy":
        return float("nan")
    return corr.backend.module.get_default_memory_pool().total_bytes() / 1e9


def free_pool(corr):
    if corr.backend.name == "cupy":
        corr.backend.module.get_default_memory_pool().free_all_blocks()


def scale_rel(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - b)) / np.max(np.abs(b)))


def timed_calls(corr, maps, w, n):
    corr.get_full_tomo_shear(maps[0], w, flip_g1=True, return_device=False)
    sync(corr)
    times, results = [], []
    for k in range(n):
        sync(corr)
        t0 = time.perf_counter()
        out = corr.get_full_tomo_shear(maps[k % len(maps)], w, flip_g1=True, return_device=False)
        sync(corr)
        times.append(time.perf_counter() - t0)
        if k < len(maps):
            results.append([np.asarray(o, dtype=np.float64) for o in out])
    return float(np.median(times) * 1e3), results


def part_a(args, out):
    nside = 512
    des_map = hp.ud_grade(hp.read_map(MASK), nside)
    Q = args.Q
    pair_file = f"{SBI}/CosmoFuse/Q{Q}/2PCF_pairs_512_15_250_8.h5"
    with h5py.File(pair_file, "r") as fp:
        attrs = dict(fp.attrs)
    phi = np.loadtxt(f"{SBI}/CosmoFuse/Q{Q}/patch_center_original_phi.dat")
    theta = np.loadtxt(f"{SBI}/CosmoFuse/Q{Q}/patch_center_original_theta.dat")
    geo = dict(
        nbins=int(attrs["nbins"]),
        theta_min=float(np.degrees(attrs["theta_min"]) * 60),
        theta_max=float(np.degrees(attrs["theta_max"]) * 60),
        patch_size=float(attrs["patch_size"]),
        theta_Q=float(attrs["theta_Q"]),
        mask=des_map,
    )
    cut = np.load(f"{SBI}/shear_maps/grid_baryonified/4BINS/gamma_0000.npy")
    cut = cut.reshape((-1,) + cut.shape[2:])[: args.nsets]
    w_full = np.load(f"{SBI}/shear_maps/SumOfWeights_512.npy")

    # ---- T7: real GPU vs CPU, treecode on, first n patches ---------------
    n_par = args.parity_patches
    res = {}
    for name, dev, mp, acc in (
        ("cpu_f64", "cpu", "float64", "same"),
        ("gpu_f64", "gpu", "float64", "same"),
        ("gpu_f32_acc64", "gpu", "float32", "float64"),
    ):
        corr = Correlation(nside, phi[:n_par], theta[:n_par], device=dev, map_precision=mp,
                           accumulation_precision=acc, rotation_precision="float64",
                           resolution_factor=args.k_parity, aperture_nside=256, **geo)
        corr.preprocess()
        rows = np.ascontiguousarray(cut[0], dtype=mp)
        w = np.ascontiguousarray(w_full[:, corr.row_pix], dtype=mp)
        res[name] = [np.asarray(o, dtype=np.float64) for o in
                     corr.get_full_tomo_shear(rows, w, flip_g1=True, return_device=False)]
        levels = corr.level_table["nside"].tolist()
        del corr
    out["A_parity"] = {
        "k": args.k_parity, "levels": levels, "patches": n_par,
        "gpu_f64_vs_cpu_f64": [scale_rel(a, b) for a, b in zip(res["gpu_f64"], res["cpu_f64"])],
        "gpu_f32acc64_vs_cpu_f64": [scale_rel(a, b) for a, b in zip(res["gpu_f32_acc64"], res["cpu_f64"])],
    }
    print(json.dumps(out["A_parity"], indent=1), flush=True)

    # ---- speed / memory on the full production patch set ------------------
    out["A_speed"] = {}
    ref = None
    for k in [None] + list(args.k):
        corr = Correlation(nside, phi, theta, device="gpu", map_precision="float32",
                           accumulation_precision="float64", resolution_factor=k, **geo)
        t0 = time.perf_counter()
        if k is None:
            corr.load_pairs(pair_file)
        else:
            corr.preprocess()
        t_geo = time.perf_counter() - t0
        n_pairs = int(corr.ntotpairs)
        bins = np.asarray(corr.bins, dtype=np.int64).sum(axis=0).tolist()
        corr.pair_inds = corr.pair_exp2phi = corr.bins = None
        maps = [np.ascontiguousarray(c, dtype=np.float32) for c in cut]
        w = np.ascontiguousarray(w_full[:, corr.row_pix], dtype=np.float32)
        w.flags.writeable = False
        ms, results = timed_calls(corr, maps, w, args.ncalls)
        entry = {"n_pairs": n_pairs, "pairs_per_bin": bins, "ms_per_mapset": ms,
                 "geometry_s": t_geo, "device_pool_GB": pool_gb(corr),
                 "levels": corr.level_table["nside"].tolist(), "n_rows": int(corr.n_rows)}
        if k is None:
            ref = results
        else:
            xip = np.array([r[1] for r in results]); xip0 = np.array([r[1] for r in ref])
            xim = np.array([r[2] for r in results]); xim0 = np.array([r[2] for r in ref])
            entry["xip_ratio_of_means"] = (xip.mean(axis=(0, 1, 2)) / xip0.mean(axis=(0, 1, 2))).tolist()
            entry["xim_ratio_of_means"] = (xim.mean(axis=(0, 1, 2)) / xim0.mean(axis=(0, 1, 2))).tolist()
            entry["M_a_identical"] = bool(all(np.array_equal(a[0], b[0]) for a, b in zip(results, ref)))
        out["A_speed"][str(k)] = entry
        print(k, json.dumps(entry), flush=True)
        del corr, results
        import gc; gc.collect()
    return out


def part_b(args, out):
    nside = 2048
    mask = hp.read_map(MASK)
    if hp.npix2nside(mask.size) != nside:
        mask = hp.ud_grade(mask.astype(np.float64), nside)
    mask = mask != 0
    Q = args.Q
    phi = np.loadtxt(f"{SBI}/CosmoFuse/Q{Q}/patch_center_original_phi.dat")
    theta = np.loadtxt(f"{SBI}/CosmoFuse/Q{Q}/patch_center_original_theta.dat")
    edges = np.geomspace(5.0, 250.0, 12)
    edges = edges[edges < 2 * Q - 5]
    out["B"] = {}
    for k in args.k2048:
        corr = Correlation(nside, phi, theta, nbins=len(edges) - 1, theta_min=edges[0],
                           theta_max=edges[-1], patch_size=Q, theta_Q=Q, mask=mask,
                           device="gpu", map_precision="float32", accumulation_precision="float64",
                           resolution_factor=k, aperture_nside=512)
        t0 = time.perf_counter()
        corr.calculate_pairs_M_a()
        t_ap = time.perf_counter() - t0
        t0 = time.perf_counter()
        corr.calculate_pairs_2PCF()
        t_pairs = time.perf_counter() - t0
        bins = np.asarray(corr.bins, dtype=np.int64).sum(axis=0).tolist()
        t0 = time.perf_counter()
        corr.prepare(release_host_pairs=True)
        sync(corr)
        t_prep = time.perf_counter() - t0
        rng = np.random.default_rng(0)
        maps = [rng.normal(size=(4, 2, corr.n_active)).astype(np.float32) * 0.1 for _ in range(2)]
        w = rng.uniform(0.5, 2.0, size=(4, corr.n_active)).astype(np.float32)
        w.flags.writeable = False
        ms, results = timed_calls(corr, maps, w, args.ncalls)
        entry = {
            "n_patches": int(corr.n_patches), "levels": corr.level_table["nside"].tolist(),
            "theta_lo_arcmin": np.round(corr.level_table["theta_lo_arcmin"], 2).tolist(),
            "n_pairs": int(corr.ntotpairs), "pairs_per_bin": bins,
            "n_active": int(corr.n_active), "n_aperture_cells": int(corr.n_aperture_cells),
            "n_treecode_cells": int(corr._treecode.n_cells), "n_rows": int(corr.n_rows),
            "aperture_geometry_s": t_ap, "pair_finding_s": t_pairs, "prepare_s": t_prep,
            "device_pool_GB": pool_gb(corr), "ms_per_mapset": ms,
            "finite": bool(all(np.isfinite(o).all() for r in results for o in r)),
        }
        out["B"][str(k)] = entry
        print("2048", k, json.dumps(entry), flush=True)
        del corr, maps, results
        import gc; gc.collect()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Q", type=int, default=110)
    ap.add_argument("--parts", default="AB")
    ap.add_argument("--k", type=float, nargs="*", default=[2.9, 4.0])
    ap.add_argument("--k2048", type=float, nargs="*", default=[2.9])
    ap.add_argument("--k-parity", type=float, default=2.9)
    ap.add_argument("--parity-patches", type=int, default=24)
    ap.add_argument("--nsets", type=int, default=6)
    ap.add_argument("--ncalls", type=int, default=12)
    ap.add_argument("--out", default="benchmarks/static_treecode/results/stage4_gpu.json")
    args = ap.parse_args()
    out = {}
    if "A" in args.parts:
        part_a(args, out)
    if "B" in args.parts:
        part_b(args, out)
    with open(args.out, "w") as fp:
        json.dump(out, fp, indent=1)


if __name__ == "__main__":
    main()
