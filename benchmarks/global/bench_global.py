"""Cost of the footprint-wide estimators on the DES Y3 footprint.

    python bench_global.py --nside 256 --mask /path/DESY3_Mask.fits [--g3] [--g2]

Random maps (4 source bins); times pair finding, preparation and the median
per-map call (device-resident inputs, synchronised) and reports pair counts
and device memory.
"""

import argparse
import json
import time

import healpy as hp
import numpy as np


def sync():
    try:
        import cupy

        cupy.cuda.Device().synchronize()
    except Exception:
        pass


def timed(fn, reps):
    out, ts = None, []
    for _ in range(reps):
        sync()
        t = time.perf_counter()
        out = fn()
        sync()
        ts.append(time.perf_counter() - t)
    return out, float(np.median(ts)), ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nside", type=int, default=256)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--nz", type=int, default=4)
    ap.add_argument("--theta-min", type=float, default=15.0)
    ap.add_argument("--theta-max", type=float, default=250.0)
    ap.add_argument("--nbins2", type=int, default=20)
    ap.add_argument("--nbins3", type=int, default=12)
    ap.add_argument("--n-max", type=int, default=20)
    ap.add_argument("--resolution-factor", type=float, default=None)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--g2", action="store_true")
    ap.add_argument("--g3", action="store_true")
    ap.add_argument("--device", default="gpu")
    args = ap.parse_args()

    import cupy

    from CosmoFuse import Global2PCF, Global3PCF

    m = hp.read_map(args.mask, dtype=np.float64)
    mask = hp.ud_grade(m, args.nside) >= 0.5
    rng = np.random.default_rng(0)
    report = {"nside": args.nside, "n_active": int(mask.sum())}
    npix = hp.nside2npix(args.nside)
    rows = np.flatnonzero(mask)
    shear = (rng.normal(size=(args.nz, 2, rows.size)) * 0.3).astype(np.float32)
    w = rng.uniform(0.5, 1.5, size=(args.nz, rows.size)).astype(np.float32)

    if args.g2:
        t = time.perf_counter()
        g2 = Global2PCF(args.nside, mask=mask, nbins=args.nbins2, theta_min=args.theta_min,
                        theta_max=args.theta_max, device=args.device, map_precision="float32",
                        accumulation_precision="float64", resolution_factor=args.resolution_factor,
                        pack_pairs=True)
        g2.find_pairs()
        t_find = time.perf_counter() - t
        t = time.perf_counter()
        g2.prepare(release_host_pairs=True)
        sync()
        t_prep = time.perf_counter() - t
        s_dev, w_dev = cupy.asarray(shear), cupy.asarray(w)
        _, t_call, _ = timed(lambda: g2.shear(s_dev, w_dev, flip_g1=True), args.reps + 1)
        report["global2pcf"] = {
            "pairs": g2.ntotpairs, "tiles": g2.n_tiles, "tile_nside": g2.tile_nside,
            "levels": g2.level_table["nside"].tolist(),
            "find_s": t_find, "prepare_s": t_prep, "shear_call_ms": 1e3 * t_call,
        }
        del g2
        cupy.get_default_memory_pool().free_all_blocks()
        print(json.dumps(report, indent=1), flush=True)

    if args.g3:
        t = time.perf_counter()
        g3 = Global3PCF(args.nside, mask=mask, nbins=args.nbins3, theta_min=args.theta_min,
                        theta_max=args.theta_max, n_max=args.n_max, device=args.device,
                        map_precision="float32", rotation_precision="float32",
                        accumulation_precision="float64", centre_chunk=args.chunk)
        g3.find_pairs()
        t_find = time.perf_counter() - t
        t = time.perf_counter()
        g3.prepare()
        sync()
        t_prep = time.perf_counter() - t
        s_dev, w_dev = cupy.asarray(shear), cupy.asarray(w)
        pool = cupy.get_default_memory_pool()
        _, t_first, _ = timed(lambda: g3.shear(s_dev, w_dev, flip_g1=True), 1)  # + norm
        _, t_call, ts = timed(lambda: g3.shear(s_dev, w_dev, flip_g1=True), args.reps)
        report["global3pcf"] = {
            "ordered_pairs": g3.ntotpairs, "tiles": g3.n_tiles, "tile_nside": g3.tile_nside,
            "n_max": args.n_max, "nbins": args.nbins3, "nz": args.nz, "chunk": args.chunk,
            "find_s": t_find, "prepare_s": t_prep,
            "first_call_incl_norm_s": t_first, "shear_call_s": t_call,
            "device_pool_gb": pool.total_bytes() / 1e9,
        }
        print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
