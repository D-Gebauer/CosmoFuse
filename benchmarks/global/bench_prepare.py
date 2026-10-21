"""Global2PCF precompute: find_pairs / prepare / load_pairs, and what they build.

Run once per CosmoFuse tree (``PYTHONPATH=<tree>/src``) on the same machine;
the JSON records the timings *and* a hash of the whole device state and of
the ``gg`` output, so two trees can be compared for speed and for bitwise
identity in one go::

    python bench_prepare.py --nside 512 --mask DESY3_Mask.fits --out new.json
    python bench_prepare.py --nside 2048 --mask DESY3_Mask.fits \\
        --resolution-factor 2.9 --pack-host-pairs --io-dir /scratch --out x.json

``--pack-host-pairs`` (7.2.0+) also times the find-time packing and the
packed pair file.  The device state must hash the same either way.
"""

import argparse
import hashlib
import json
import os
import time
import warnings

import healpy as hp
import numpy as np

import CosmoFuse
from CosmoFuse import Global2PCF


def _sync():
    try:
        import cupy

        cupy.cuda.runtime.deviceSynchronize()
    except Exception:
        pass


def _host(a):
    return np.ascontiguousarray(a.get() if hasattr(a, "get") else np.asarray(a))


def _digest(arrays):
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        if a is None:
            h.update(b"None")
            continue
        a = _host(a)
        h.update(str((a.dtype.str, a.shape)).encode())
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def _state(g):
    e = g._engine
    ctx = e.compute_context
    return _digest(
        [
            ctx.packed_pairs_dev,
            ctx.packed_row_base_dev,
            ctx.packed_perm_dev,
            e.inds_dev,
            e.exp2phi_dev,
            e.bins_dev,
            e.tot_bins_reduceat_dev,
        ]
    )


def _timed(fn):
    _sync()
    t = time.perf_counter()
    out = fn()
    _sync()
    return time.perf_counter() - t, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nside", type=int, required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--resolution-factor", type=float, default=None)
    ap.add_argument("--tile-nside", type=int, default=None)
    ap.add_argument("--nbins", type=int, default=20)
    ap.add_argument("--theta-min", type=float, default=15.0)
    ap.add_argument("--theta-max", type=float, default=250.0)
    ap.add_argument("--nz", type=int, default=4)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--pack-host-pairs", action="store_true")
    ap.add_argument("--io-dir", default=None, help="also time save/load_pairs here")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    mask = hp.ud_grade(hp.read_map(args.mask, dtype=np.float64), args.nside) >= 0.5
    kw = dict(
        nbins=args.nbins,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        tile_nside=args.tile_nside,
        device=args.device,
        map_precision="float32",
        rotation_precision="float32",
        accumulation_precision="float64",
        resolution_factor=args.resolution_factor,
        pack_pairs=True,
    )
    rep = {
        "cosmofuse": CosmoFuse.__version__,
        "tree": os.path.dirname(os.path.dirname(CosmoFuse.__file__)),
        "nside": args.nside,
        "resolution_factor": args.resolution_factor,
        "n_active": int(mask.sum()),
        "host_threads": int(__import__("numba").get_num_threads()),
    }

    def measure(g, tag):
        rng = np.random.default_rng(0)
        n = g.n_active
        shear = rng.normal(size=(args.nz, 2, n)).astype(np.float32)
        w = rng.uniform(0.5, 1.5, size=(args.nz, n)).astype(np.float32)
        g.gg(shear, w, flip_g1=True)  # compile
        times = []
        for _ in range(5):
            dt, res = _timed(lambda: g.gg(shear, w, flip_g1=True))
            times.append(dt)
        res = res.to_numpy()
        rep[f"{tag}_state"] = _state(g)
        rep[f"{tag}_gg"] = _digest(
            [res.tile_num["xi_p"], res.tile_num["xi_m"], res.tile_weight]
        )
        rep[f"{tag}_gg_ms"] = 1e3 * float(np.median(times))

    g = Global2PCF(args.nside, mask=mask, **kw)
    rep["tiles"], rep["tile_nside"] = g.n_tiles, g.tile_nside
    rep["level_table"] = {k: np.asarray(v).tolist() for k, v in g.level_table.items()}
    rep["find_s"], _ = _timed(g.find_pairs)
    rep["pairs"] = int(sum(int(np.sum(b, dtype=np.int64)) for b in g._engine.bins))
    if args.io_dir:
        path = os.path.join(args.io_dir, f"g2_{args.nside}_{os.getpid()}_exact.h5")
        rep["save_exact_s"], _ = _timed(lambda: g.save_pairs(path))
        rep["file_exact_gb"] = os.path.getsize(path) / 1e9
    rep["prepare_s"], _ = _timed(g.prepare)
    measure(g, "exact")
    if args.io_dir:
        g2 = Global2PCF(args.nside, mask=mask, **kw)
        rep["load_exact_s"], _ = _timed(lambda: g2.load_pairs(path))
        rep["load_exact_state_equal"] = _state(g2) == rep["exact_state"]
        del g2
        os.remove(path)
    del g
    print(json.dumps(rep, indent=1), flush=True)

    if args.pack_host_pairs:
        g = Global2PCF(args.nside, mask=mask, pack_host_pairs=True, **kw)
        rep["find_packed_s"], _ = _timed(g.find_pairs)
        if args.io_dir:
            path = os.path.join(args.io_dir, f"g2_{args.nside}_{os.getpid()}_packed.h5")
            rep["save_packed_s"], _ = _timed(lambda: g.save_pairs(path))
            rep["file_packed_gb"] = os.path.getsize(path) / 1e9
        rep["prepare_packed_s"], _ = _timed(g.prepare)
        measure(g, "packed")
        if args.io_dir:
            g2 = Global2PCF(args.nside, mask=mask, **kw)
            rep["load_packed_s"], _ = _timed(lambda: g2.load_pairs(path))
            rep["load_packed_state_equal"] = _state(g2) == rep["exact_state"]
            os.remove(path)
    with open(args.out, "w") as fp:
        json.dump(rep, fp, indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
