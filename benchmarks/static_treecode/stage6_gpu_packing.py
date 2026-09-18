"""Stage 6.2 keep-if: payload packing (8 B/pair) on the A100.

(P) parity: GPU packed vs CPU packed (same quantised estimator), float64.
(A) nside 512 production geometry, 28 real noisy map-sets: packed vs exact --
    per-estimate shift in units of the patch scatter, bias, patch-mean shift,
    timing, device memory.
(B) nside 2048 treecode geometry: timing and device memory.
"""
import argparse, gc, json, time
import h5py, healpy as hp, numpy as np
from CosmoFuse import Correlation

SBI = "/e/ocean1/users/dgebauer/sbi"
MASK = "/home/moon/dgebauer/research/lfi/local/DESY3_Mask.fits"


def sync(corr):
    corr.backend.module.cuda.runtime.deviceSynchronize()


def free(corr):
    xp = corr.backend.module
    del corr
    gc.collect()
    xp.get_default_memory_pool().free_all_blocks()


def timed(corr, g, w, n=8):
    corr.vectorized_shear_shear(g, w, flip_g1=True); sync(corr)
    ts = []
    for _ in range(n):
        sync(corr); t0 = time.perf_counter()
        corr.vectorized_shear_shear(g, w, flip_g1=True)
        sync(corr); ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3)


def geo512():
    nside = 512
    des_map = hp.ud_grade(hp.read_map(MASK), nside)
    pair_file = f"{SBI}/CosmoFuse/Q110/2PCF_pairs_512_15_250_8.h5"
    with h5py.File(pair_file, "r") as fp:
        attrs = dict(fp.attrs)
    phi = np.loadtxt(f"{SBI}/CosmoFuse/Q110/patch_center_original_phi.dat")
    theta = np.loadtxt(f"{SBI}/CosmoFuse/Q110/patch_center_original_theta.dat")
    geo = dict(nbins=int(attrs["nbins"]), theta_min=float(np.degrees(attrs["theta_min"]) * 60),
               theta_max=float(np.degrees(attrs["theta_max"]) * 60), patch_size=110., theta_Q=110., mask=des_map)
    return nside, phi, theta, geo, pair_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="PAB")
    ap.add_argument("--pairs2048", default="benchmarks/static_treecode/results/pairs_2048_k2.9_Q110.h5")
    args = ap.parse_args()
    out = {}
    nside, phi, theta, geo, pair_file = geo512()
    cut = np.load(f"{SBI}/shear_maps/grid_baryonified/4BINS/gamma_0000.npy").reshape(-1, 4, 2, 369190)
    w_full = np.load(f"{SBI}/shear_maps/SumOfWeights_512.npy")

    if "P" in args.parts:
        res = {}
        for dev in ("cpu", "gpu"):
            corr = Correlation(nside, phi[:24], theta[:24], device=dev, map_precision="float64",
                               rotation_precision="float64", resolution_factor=4.0, pack_pairs=True, **geo)
            corr.preprocess()
            w = np.ascontiguousarray(w_full[:, corr.row_pix])
            res[dev] = [np.asarray(o, dtype=np.float64) for o in
                        corr.get_full_tomo_shear(cut[0], w, flip_g1=True, return_device=False)]
            if dev == "gpu":
                assert corr.inds_dev is None and corr.compute_context.packed_pairs_dev is not None
        out["P_gpu_packed_vs_cpu_packed"] = [
            float(np.max(np.abs(a - b)) / np.max(np.abs(b))) for a, b in zip(res["gpu"], res["cpu"])]
        print("P", out["P_gpu_packed_vs_cpu_packed"], flush=True)

    if "A" in args.parts:
        results = {}
        for pack in (False, True):
            corr = Correlation(nside, phi, theta, device="gpu", map_precision="float32",
                               accumulation_precision="float64", pack_pairs=pack, **geo)
            corr.load_pairs(pair_file, release_host_pairs=True)
            sync(corr)
            pool = corr.backend.module.get_default_memory_pool().used_bytes() / 1e9
            xp = corr.backend.module
            w = xp.asarray(w_full[:, corr.row_pix].astype(np.float32))
            res = []
            for k in range(cut.shape[0]):
                g = xp.asarray(cut[k].astype(np.float32))
                res.append([o.get().astype(np.float64) for o in corr.vectorized_shear_shear(g, w, flip_g1=True)])
            ms = timed(corr, g, w)
            results[pack] = res
            out[f"A_512_pack{pack}"] = {"ms": ms, "device_used_GB_after_prepare": pool, "n_pairs": int(corr.ntotpairs)}
            print("A", pack, out[f"A_512_pack{pack}"], flush=True)
            del g, w; free(corr)
        stats = {}
        for idx, name in ((0, "xip"), (1, "xim")):
            e = np.array([r[idx] for r in results[False]])      # (maps, comb, patch, bin)
            p = np.array([r[idx] for r in results[True]])
            sigma = e.std(axis=(0, 2), keepdims=True)
            z = (p - e) / sigma
            shift_mean = (p.mean(axis=2) - e.mean(axis=2)) / (sigma[:, :, 0] / np.sqrt(e.shape[2]))
            stats[name] = {
                "max_abs_shift_in_sigma_patch": float(np.max(np.abs(z))),
                "rms_shift_in_sigma_patch": float(z.std()),
                "mean_shift_in_sigma_patch": float(z.mean()),
                "mean_shift_significance": float(z.mean() / (z.std() / np.sqrt(z.size))),
                "max_abs_shift_of_patch_mean_in_its_sigma": float(np.max(np.abs(shift_mean))),
                "max_rel_diff_patch_mean": float(np.max(np.abs(p.mean(axis=(0, 2)) / e.mean(axis=(0, 2)) - 1))),
            }
        out["A_error_statistics"] = stats
        print(json.dumps(stats, indent=1), flush=True)

    if "B" in args.parts:
        nside = 2048
        mask = hp.read_map(MASK)
        if hp.npix2nside(mask.size) != nside:
            mask = hp.ud_grade(mask.astype(np.float64), nside)
        mask = mask != 0
        edges = np.geomspace(5.0, 250.0, 12); edges = edges[edges < 215]
        for pack in (False, True):
            corr = Correlation(nside, phi, theta, nbins=len(edges) - 1, theta_min=edges[0], theta_max=edges[-1],
                               patch_size=110, theta_Q=110, mask=mask, device="gpu", map_precision="float32",
                               accumulation_precision="float64", resolution_factor=2.9, aperture_nside=512,
                               pack_pairs=pack)
            t0 = time.perf_counter()
            corr.load_pairs(args.pairs2048, release_host_pairs=True); sync(corr)
            t_load = time.perf_counter() - t0
            xp = corr.backend.module
            pool = xp.get_default_memory_pool().used_bytes() / 1e9
            rng = np.random.default_rng(0)
            g = xp.asarray(rng.normal(size=(4, 2, corr.n_active)).astype(np.float32) * 0.1)
            w = xp.asarray(rng.uniform(0.5, 2.0, size=(4, corr.n_active)).astype(np.float32))
            ms = timed(corr, g, w)
            n_packed = 0 if not pack else int(corr.compute_context.packed_perm_dev.shape[0])
            out[f"B_2048_pack{pack}"] = {"ms": ms, "device_used_GB_after_prepare": pool, "load_prepare_s": t_load,
                                          "n_pairs": int(corr.ntotpairs), "n_rows": int(corr.n_rows), "n_packed_rows": n_packed}
            print("B", pack, out[f"B_2048_pack{pack}"], flush=True)
            del g, w; free(corr)
    with open("benchmarks/static_treecode/results/stage6_packing.json", "w") as fp:
        json.dump(out, fp, indent=1)


if __name__ == "__main__":
    main()
