"""Stage 7 GPU gates: combination-tiled + packed kernels for ALL pair
statistics (xi+-, xi_g, xi_t, fused 3x2pt) on the A100.

Part A (nside 512, production Q110 pair file, real shear maps, random
density maps, 4 + 4 tomographic bins):
  * parity: tiled vs per-row kernels (unpacked), packed vs unpacked, and
    GPU float64 vs CPU float64 on a 24-patch subset (packed and unpacked);
  * per-map-set wall time of get_full_tomo_shear / density / ggl and
    get_3x2pt_tomo for per-row (5.0 before this stage), tiled, tiled+packed.

Part B (nside 2048, k = 2.9 pair file from stage 4, packed): the same
timings, plus device memory.
"""

import argparse
import gc
import json
import time

import h5py
import healpy as hp
import numpy as np

from CosmoFuse.correlations import Correlation

SBI = "/e/ocean1/users/dgebauer/sbi"
MASK = "/home/moon/dgebauer/research/lfi/local/DESY3_Mask.fits"
PAIRS_2048 = "benchmarks/static_treecode/results/pairs_2048_k2.9_Q110.h5"


def sync(corr):
    if corr.backend.name == "cupy":
        corr.backend.module.cuda.runtime.deviceSynchronize()


def pool_gb(corr):
    if corr.backend.name != "cupy":
        return float("nan")
    return corr.backend.module.get_default_memory_pool().total_bytes() / 1e9


def rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scale = np.max(np.abs(b))
    return float(np.max(np.abs(a - b)) / scale) if scale > 0 else float(np.max(np.abs(a - b)))


def set_tiled(corr, flag):
    for name in ("xipm_tomo_vectorized_kernel", "kernel_density_density_tomo_vectorized",
                 "kernel_density_shear_tomo_vectorized"):
        getattr(corr.backend, name).tiled = flag


def methods(corr, shear, w, dens, wd):
    return {
        "shear": lambda: corr.get_full_tomo_shear(shear, w, flip_g1=True, return_device=False),
        "density": lambda: corr.get_full_tomo_density(dens, wd, return_device=False),
        "ggl": lambda: corr.get_full_tomo_ggl(dens, shear, wd, w, flip_g1=True, return_device=False),
        "3x2pt": lambda: corr.get_3x2pt_tomo(
            shear_maps=shear, density_maps=dens, weights={"shear": w, "density": wd},
            flip_g1=True, return_device=False),
    }


def as_list(r):
    return [np.asarray(x, dtype=np.float64) for x in (r if isinstance(r, tuple) else (r,))]


def timed(fn, corr, n):
    fn(); sync(corr)
    t = []
    for _ in range(n):
        sync(corr); t0 = time.perf_counter(); fn(); sync(corr); t.append(time.perf_counter() - t0)
    return float(np.median(t) * 1e3)


def geometry(Q):
    des_map = hp.ud_grade(hp.read_map(MASK), 512)
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
    return phi, theta, geo, pair_file


def make_maps(corr, rng, nz=4):
    cut = np.load(f"{SBI}/shear_maps/grid_baryonified/4BINS/gamma_0000.npy")
    cut = cut.reshape((-1,) + cut.shape[2:])[0]
    w_full = np.load(f"{SBI}/shear_maps/SumOfWeights_512.npy")
    shear = np.ascontiguousarray(cut, dtype=np.float32)
    w = np.ascontiguousarray(w_full[:, corr.row_pix], dtype=np.float32)
    dens = (rng.normal(size=(nz, corr.n_active)) * 0.3).astype(np.float32)
    wd = rng.uniform(0.5, 2.0, size=(nz, corr.n_active)).astype(np.float32)
    return shear, w, dens, wd


def part_a(args, out):
    phi, theta, geo, pair_file = geometry(args.Q)
    rng = np.random.default_rng(0)

    # ---- parity on a subset: CPU f64 vs GPU f64, packed and unpacked ---------
    n_par = args.parity_patches
    res = {}
    for name, dev, pack in (("cpu", "cpu", False), ("cpu_packed", "cpu", True),
                            ("gpu", "gpu", False), ("gpu_packed", "gpu", True),
                            ("gpu_perrow", "gpu", False)):
        corr = Correlation(512, phi[:n_par], theta[:n_par], device=dev, map_precision="float64",
                           rotation_precision="float64", pack_pairs=pack, **geo)
        corr.preprocess()
        if name == "gpu_perrow":
            set_tiled(corr, False)
        shear, w, dens, wd = make_maps(corr, np.random.default_rng(1))
        shear, w, dens, wd = (x.astype(np.float64) for x in (shear, w, dens, wd))
        res[name] = {k: as_list(f()) for k, f in methods(corr, shear, w, dens, wd).items()}
        if name == "gpu_perrow":
            set_tiled(corr, True)
        del corr; gc.collect()
    par = {}
    for a, b in (("gpu", "cpu"), ("gpu_packed", "cpu_packed"), ("gpu_perrow", "gpu"),
                 ("gpu_packed", "gpu"), ("cpu_packed", "cpu")):
        par[f"{a}_vs_{b}"] = {k: [rel(x, y) for x, y in zip(res[a][k], res[b][k])] for k in res[a]}
    out["A_parity"] = {"patches": n_par, **par}
    print(json.dumps(out["A_parity"], indent=1), flush=True)

    # ---- timings on the full production patch set ---------------------------
    out["A_speed"] = {}
    for variant, pack in (("per_row", False), ("tiled", False), ("tiled_packed", True)):
        corr = Correlation(512, phi, theta, device="gpu", map_precision="float32",
                           accumulation_precision="float64", pack_pairs=pack, **geo)
        corr.load_pairs(pair_file)
        corr.prepare(release_host_pairs=True)
        set_tiled(corr, variant != "per_row")
        shear, w, dens, wd = make_maps(corr, rng)
        for arr in (w, wd):
            arr.flags.writeable = False
        entry = {"n_pairs": int(corr.ntotpairs), "device_pool_GB": pool_gb(corr)}
        for k, f in methods(corr, shear, w, dens, wd).items():
            entry[f"ms_{k}"] = timed(f, corr, args.ncalls)
        set_tiled(corr, True)
        out["A_speed"][variant] = entry
        print(variant, json.dumps(entry), flush=True)
        del corr; gc.collect()
    return out


def part_b(args, out):
    phi, theta, geo, _ = geometry(args.Q)
    mask = hp.read_map(MASK)
    mask = hp.ud_grade(mask.astype(np.float64), 2048) != 0
    out["B"] = {}
    for variant, pack in (("tiled", False), ("tiled_packed", True)):
        edges = np.geomspace(5.0, 250.0, 12)
        edges = edges[edges < 2 * args.Q - 5]
        corr = Correlation(2048, phi, theta, nbins=len(edges) - 1, theta_min=edges[0],
                           theta_max=edges[-1], patch_size=args.Q, theta_Q=args.Q, mask=mask,
                           device="gpu", map_precision="float32", accumulation_precision="float64",
                           resolution_factor=2.9, aperture_nside=512, pack_pairs=pack)
        t0 = time.perf_counter()
        corr.load_pairs(PAIRS_2048, release_host_pairs=True); sync(corr)
        t_prep = time.perf_counter() - t0
        rng = np.random.default_rng(0)
        nz = 4
        shear = (rng.normal(size=(nz, 2, corr.n_active)) * 0.1).astype(np.float32)
        w = rng.uniform(0.5, 2.0, size=(nz, corr.n_active)).astype(np.float32)
        dens = (rng.normal(size=(nz, corr.n_active)) * 0.3).astype(np.float32)
        wd = rng.uniform(0.5, 2.0, size=(nz, corr.n_active)).astype(np.float32)
        for arr in (w, wd):
            arr.flags.writeable = False
        entry = {"n_pairs": int(corr.ntotpairs), "n_patches": int(corr.n_patches),
                 "prepare_s": t_prep, "device_pool_GB": pool_gb(corr)}
        for k, f in methods(corr, shear, w, dens, wd).items():
            entry[f"ms_{k}"] = timed(f, corr, args.ncalls)
        out["B"][variant] = entry
        print("2048", variant, json.dumps(entry), flush=True)
        del corr; gc.collect()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Q", type=int, default=110)
    ap.add_argument("--parts", default="AB")
    ap.add_argument("--parity-patches", type=int, default=24)
    ap.add_argument("--ncalls", type=int, default=8)
    ap.add_argument("--out", default="benchmarks/static_treecode/results/stage7_gpu_tiles.json")
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
