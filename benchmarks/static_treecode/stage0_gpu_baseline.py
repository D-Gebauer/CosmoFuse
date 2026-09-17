"""Stage 0 (GPU part): reference workload timing on the production geometry.

Reference workload = production Q-patch pair file + real baryonified shear
maps (footprint cut-outs scattered into full-sky arrays, exactly like the
production script), ``get_full_tomo_shear(..., flip_g1=True)``.

Reports, per map-set, synchronised wall times for
  * host scatter (footprint -> full sky),
  * the full call with host inputs (what production pays today),
  * the full call with device-resident inputs (= device prep + kernels + D2H),
and derives pair-evaluations per second for the xi+/- kernel.

Run under ``nsys profile --stats=true`` for the per-kernel / memcpy split.
"""

import argparse
import json
import time

import h5py
import healpy as hp
import numpy as np

from CosmoFuse.correlations import Correlation

SBI = "/e/ocean1/users/dgebauer/sbi"


def sync(corr):
    if corr.backend.name == "cupy":
        corr.backend.module.cuda.Stream.null.synchronize()
        corr.backend.module.cuda.runtime.deviceSynchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--Q", type=int, default=110)
    ap.add_argument("--nsets", type=int, default=28)
    ap.add_argument("--stop-ind", type=int, default=None)
    ap.add_argument("--map-precision", default="float32")
    ap.add_argument("--acc", default="float64")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    nside = 512
    npix = hp.nside2npix(nside)
    pair_file = f"{SBI}/CosmoFuse/Q{args.Q}/2PCF_pairs_512_15_250_8.h5"
    des_map = hp.ud_grade(
        hp.read_map("/home/moon/dgebauer/research/lfi/local/DESY3_Mask.fits"), nside
    )
    map_inds = np.where(des_map != 0)[0]
    w = np.load(f"{SBI}/shear_maps/SumOfWeights_512.npy")

    with h5py.File(pair_file, "r") as fp:
        attrs = dict(fp.attrs)
    print("pair file attrs:", attrs)

    phi = np.loadtxt(f"{SBI}/CosmoFuse/Q{args.Q}/patch_center_original_phi.dat")
    theta = np.loadtxt(f"{SBI}/CosmoFuse/Q{args.Q}/patch_center_original_theta.dat")
    corr = Correlation(
        nside,
        phi,
        theta,
        nbins=int(attrs["nbins"]),
        theta_min=float(np.degrees(attrs["theta_min"]) * 60),
        theta_max=float(np.degrees(attrs["theta_max"]) * 60),
        patch_size=float(attrs["patch_size"]),
        theta_Q=float(attrs["theta_Q"]),
        mask=des_map,
        device=args.device,
        map_precision=args.map_precision,
        accumulation_precision=args.acc,
    )
    t0 = time.perf_counter()
    corr.load_pairs(pair_file, stop_ind=args.stop_ind)
    sync(corr)
    t_load = time.perf_counter() - t0
    bins_all = np.asarray(corr.bins, dtype=np.int64)
    corr.pair_inds = corr.pair_exp2phi = corr.bins = None  # free host RAM
    n_pairs = int(corr.ntotpairs)
    print(f"loaded {corr.n_patches} patches, {n_pairs/1e6:.1f} M pairs in {t_load:.1f}s")

    # --- maps: one file = 7 x 4 map-sets of (4, 2, n_footprint) ----------
    t0 = time.perf_counter()
    cut = np.load(f"{SBI}/shear_maps/grid_baryonified/4BINS/gamma_0000.npy")
    t_read = time.perf_counter() - t0
    cut = cut.reshape((-1,) + cut.shape[2:])  # (28, 4, 2, nfoot)
    nsets = min(args.nsets, cut.shape[0])
    nz = cut.shape[1]
    ncomb = nz * (nz + 1) // 2
    print(f"read {cut.shape} in {t_read:.2f}s ({t_read/cut.shape[0]*1e3:.1f} ms/set)")

    dt = np.dtype(args.map_precision)
    w_host = np.ascontiguousarray(w.astype(dt))
    w_host.flags.writeable = False

    full = np.zeros((nz, 2, npix), dtype=dt)
    t_scatter, t_call_host, t_call_dev = [], [], []
    results = []

    # warm-up (kernel compilation, buffer allocation)
    full[:, :, map_inds] = cut[0]
    corr.get_full_tomo_shear(full, w_host, flip_g1=True, return_device=False)
    sync(corr)

    for k in range(nsets):
        t0 = time.perf_counter()
        full[:, :, map_inds] = cut[k]
        t_scatter.append(time.perf_counter() - t0)

        sync(corr)
        t0 = time.perf_counter()
        out = corr.get_full_tomo_shear(full, w_host, flip_g1=True, return_device=False)
        sync(corr)
        t_call_host.append(time.perf_counter() - t0)
        results.append([np.asarray(o, dtype=np.float64) for o in out])

        if corr.backend.name == "cupy":
            g_dev = corr.backend.to_device(full)
            w_dev = corr.backend.to_device(w_host)
            sync(corr)
            t0 = time.perf_counter()
            corr.get_full_tomo_shear(g_dev, w_dev, flip_g1=True, return_device=False)
            sync(corr)
            t_call_dev.append(time.perf_counter() - t0)

    def med(x):
        return float(np.median(x) * 1e3) if len(x) else float("nan")

    # pair evaluations: auto combs 1 orientation, cross combs 2
    evals = n_pairs * (nz + 2 * (ncomb - nz))
    summary = {
        "Q": args.Q,
        "n_patches": int(corr.n_patches),
        "n_pairs": n_pairs,
        "pairs_per_bin": bins_all.sum(axis=0).tolist(),
        "map_precision": args.map_precision,
        "acc": args.acc,
        "read_ms_per_set": t_read / cut.shape[0] * 1e3,
        "scatter_ms": med(t_scatter),
        "call_host_inputs_ms": med(t_call_host),
        "call_device_inputs_ms": med(t_call_dev),
        "h2d_and_staging_ms": med(t_call_host) - med(t_call_dev),
        "xipm_pair_evals": int(evals),
    }
    if corr.backend.name == "cupy":
        pool = corr.backend.module.get_default_memory_pool()
        summary["device_pool_GB"] = pool.total_bytes() / 1e9
    print(json.dumps(summary, indent=1))
    if args.out:
        with open(args.out, "w") as fp:
            json.dump(summary, fp, indent=1)
        np.savez(
            args.out.replace(".json", "_results.npz"),
            M_a=np.array([r[0] for r in results]),
            xip=np.array([r[1] for r in results]),
            xim=np.array([r[2] for r in results]),
        )


if __name__ == "__main__":
    main()
