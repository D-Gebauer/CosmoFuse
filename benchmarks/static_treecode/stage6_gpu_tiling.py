"""Stage 6.1 keep-if: combination-tiled xi+- kernel vs the per-row kernel.

Interleaved A/B rounds on (a) the production nside-512 full-resolution
geometry with real maps and (b) the nside-2048 treecode geometry."""
import argparse, json, time
import h5py, healpy as hp, numpy as np
from CosmoFuse.correlations import Correlation

SBI = "/e/ocean1/users/dgebauer/sbi"
MASK = "/home/moon/dgebauer/research/lfi/local/DESY3_Mask.fits"


def sync(corr):
    corr.backend.module.cuda.runtime.deviceSynchronize()


def ab(corr, shear_dev, w_dev, rounds, calls):
    kern = corr.backend.xipm_tomo_vectorized_kernel
    res, times = {}, {True: [], False: []}
    for tiled in (True, False):  # warm-up / compile
        kern.tiled = tiled
        res[tiled] = [np.asarray(o.get(), dtype=np.float64) for o in corr.vectorized_shear_shear(shear_dev, w_dev, flip_g1=True)]
    for _ in range(rounds):
        for tiled in (True, False):
            kern.tiled = tiled
            ts = []
            for _ in range(calls):
                sync(corr); t0 = time.perf_counter()
                corr.vectorized_shear_shear(shear_dev, w_dev, flip_g1=True)
                sync(corr); ts.append(time.perf_counter() - t0)
            times[tiled].append(np.median(ts))
    kern.tiled = True
    scale = [np.max(np.abs(b)) for b in res[False]]
    return {
        "tiled_ms": float(min(times[True]) * 1e3),
        "per_row_ms": float(min(times[False]) * 1e3),
        "speedup": float(min(times[False]) / min(times[True])),
        "parity_scale_rel": [float(np.max(np.abs(a - b)) / s) for a, b, s in zip(res[True], res[False], scale)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--calls", type=int, default=6)
    ap.add_argument("--pairs2048", default="benchmarks/static_treecode/results/pairs_2048_k2.9_Q110.h5")
    ap.add_argument("--parts", default="AB")
    args = ap.parse_args()
    out = {}
    if "A" in args.parts:
        nside = 512
        des_map = hp.ud_grade(hp.read_map(MASK), nside)
        pair_file = f"{SBI}/CosmoFuse/Q110/2PCF_pairs_512_15_250_8.h5"
        with h5py.File(pair_file, "r") as fp:
            attrs = dict(fp.attrs)
        phi = np.loadtxt(f"{SBI}/CosmoFuse/Q110/patch_center_original_phi.dat")
        theta = np.loadtxt(f"{SBI}/CosmoFuse/Q110/patch_center_original_theta.dat")
        cut = np.load(f"{SBI}/shear_maps/grid_baryonified/4BINS/gamma_0000.npy").reshape(-1, 4, 2, 369190)[0]
        w = np.load(f"{SBI}/shear_maps/SumOfWeights_512.npy")
        for mp, acc in (("float32", "float64"), ("float64", "same")):
            corr = Correlation(nside, phi, theta, nbins=int(attrs["nbins"]),
                               theta_min=float(np.degrees(attrs["theta_min"]) * 60),
                               theta_max=float(np.degrees(attrs["theta_max"]) * 60),
                               patch_size=110., theta_Q=110., mask=des_map, device="gpu",
                               map_precision=mp, accumulation_precision=acc)
            corr.load_pairs(pair_file, release_host_pairs=True)
            xp = corr.backend.module
            g = xp.asarray(cut.astype(mp)); ww = xp.asarray(w[:, corr.row_pix].astype(mp))
            out[f"A_512_fullres_{mp}"] = ab(corr, g, ww, args.rounds, args.calls)
            print(f"A {mp}", json.dumps(out[f"A_512_fullres_{mp}"]), flush=True)
            del corr, g, ww
            xp.get_default_memory_pool().free_all_blocks()
    if "B" in args.parts:
        nside = 2048
        mask = hp.read_map(MASK)
        if hp.npix2nside(mask.size) != nside:
            mask = hp.ud_grade(mask.astype(np.float64), nside)
        mask = mask != 0
        phi = np.loadtxt(f"{SBI}/CosmoFuse/Q110/patch_center_original_phi.dat")
        theta = np.loadtxt(f"{SBI}/CosmoFuse/Q110/patch_center_original_theta.dat")
        edges = np.geomspace(5.0, 250.0, 12); edges = edges[edges < 215]
        corr = Correlation(nside, phi, theta, nbins=len(edges) - 1, theta_min=edges[0], theta_max=edges[-1],
                           patch_size=110, theta_Q=110, mask=mask, device="gpu", map_precision="float32",
                           accumulation_precision="float64", resolution_factor=2.9, aperture_nside=512)
        corr.load_pairs(args.pairs2048, release_host_pairs=True)
        xp = corr.backend.module
        rng = np.random.default_rng(0)
        g = xp.asarray(rng.normal(size=(4, 2, corr.n_active)).astype(np.float32) * 0.1)
        ww = xp.asarray(rng.uniform(0.5, 2.0, size=(4, corr.n_active)).astype(np.float32))
        out["B_2048_k2.9_float32"] = ab(corr, g, ww, args.rounds, args.calls)
        print("B", json.dumps(out["B_2048_k2.9_float32"]), flush=True)
    with open("benchmarks/static_treecode/results/stage6_tiling.json", "w") as fp:
        json.dump(out, fp, indent=1)


if __name__ == "__main__":
    main()
