"""Reference workload at nside 2048 (k from --k): build or load the geometry,
then time get_full_tomo_shear with host row-space inputs, device inputs, and
the individual phases.  Run under nsys for the kernel split."""
import argparse, json, os, time
import healpy as hp, numpy as np
from CosmoFuse.correlations import Correlation

SBI = "/e/ocean1/users/dgebauer/sbi"
MASK = "/home/moon/dgebauer/research/lfi/local/DESY3_Mask.fits"


def sync(corr):
    corr.backend.module.cuda.runtime.deviceSynchronize()


def build(args):
    nside = 2048
    mask = hp.read_map(MASK)
    if hp.npix2nside(mask.size) != nside:
        mask = hp.ud_grade(mask.astype(np.float64), nside)
    mask = mask != 0
    phi = np.loadtxt(f"{SBI}/CosmoFuse/Q{args.Q}/patch_center_original_phi.dat")
    theta = np.loadtxt(f"{SBI}/CosmoFuse/Q{args.Q}/patch_center_original_theta.dat")
    edges = np.geomspace(5.0, 250.0, 12)
    edges = edges[edges < 2 * args.Q - 5]
    corr = Correlation(nside, phi, theta, nbins=len(edges) - 1, theta_min=edges[0], theta_max=edges[-1],
                       patch_size=args.Q, theta_Q=args.Q, mask=mask, device="gpu", map_precision="float32",
                       accumulation_precision="float64", resolution_factor=args.k, aperture_nside=512)
    if os.path.exists(args.pairs):
        t0 = time.perf_counter(); corr.load_pairs(args.pairs, release_host_pairs=True)
        print(f"loaded geometry in {time.perf_counter()-t0:.1f}s", flush=True)
    else:
        t0 = time.perf_counter(); corr.calculate_pairs_M_a(); corr.calculate_pairs_2PCF()
        print(f"geometry in {time.perf_counter()-t0:.1f}s", flush=True)
        t0 = time.perf_counter(); corr.save_pairs(args.pairs)
        print(f"saved in {time.perf_counter()-t0:.1f}s, {os.path.getsize(args.pairs)/1e9:.1f} GB", flush=True)
        corr.prepare(release_host_pairs=True)
    return corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Q", type=int, default=110)
    ap.add_argument("--k", type=float, default=2.9)
    ap.add_argument("--pairs", default="benchmarks/static_treecode/results/pairs_2048_k2.9_Q110.h5")
    ap.add_argument("--ncalls", type=int, default=8)
    args = ap.parse_args()
    corr = build(args)
    xp = corr.backend.module
    rng = np.random.default_rng(0)
    maps = [rng.normal(size=(4, 2, corr.n_active)).astype(np.float32) * 0.1 for _ in range(2)]
    w = rng.uniform(0.5, 2.0, size=(4, corr.n_active)).astype(np.float32)
    w.flags.writeable = False
    corr.get_full_tomo_shear(maps[0], w, flip_g1=True, return_device=False); sync(corr)

    def med(fn, n=args.ncalls):
        ts = []
        for i in range(n):
            sync(corr); t0 = time.perf_counter(); fn(i); sync(corr); ts.append(time.perf_counter() - t0)
        return float(np.median(ts) * 1e3)

    out = {"n_pairs": int(corr.ntotpairs), "n_rows": int(corr.n_rows), "n_active": int(corr.n_active)}
    out["call_host_rows_ms"] = med(lambda i: corr.get_full_tomo_shear(maps[i % 2], w, flip_g1=True, return_device=False))
    g_dev = xp.asarray(maps[0]); w_dev = corr._coerce_map_input_array(w)
    out["call_device_rows_ms"] = med(lambda i: corr.get_full_tomo_shear(g_dev, w_dev, flip_g1=True, return_device=False))
    out["h2d_ms"] = med(lambda i: xp.asarray(maps[i % 2]))
    out["expand_pairs_ms"] = med(lambda i: corr._expand_shear_rows(g_dev, w_dev, "pairs"))
    out["expand_aperture_ms"] = med(lambda i: corr._expand_shear_rows(g_dev, w_dev, "aperture"))
    out["aperture_total_ms"] = med(lambda i: corr._compute_tomo_aperture_shear(g_dev, w_dev, flip_g1=True))
    out["xipm_total_ms"] = med(lambda i: corr.vectorized_shear_shear(g_dev, w_dev, flip_g1=True))
    out["device_pool_GB"] = xp.get_default_memory_pool().total_bytes() / 1e9
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
