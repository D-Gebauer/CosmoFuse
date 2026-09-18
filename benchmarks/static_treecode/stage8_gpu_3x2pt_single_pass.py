"""Stage 8 keep-if: multi-statistic 3x2pt pair tile (one walk over the pairs
for xi+-, xi_g, xi_t) vs the three standalone tiles.  A100, production Q110
geometry (nside 512) and the nside-2048 k = 2.9 pair file.

mode "off" = three tiles (stage 7), "single" = everything in one pass.  (A
two-pass split, xi+- with xi_t and xi_g apart, was measured in the same runs
-- results/stage8_3x2pt_single_pass.json -- and dropped: slower than three
tiles on packed pairs.)
"""

import argparse
import gc
import json

import numpy as np

from CosmoFuse.correlations import Correlation
from stage7_gpu_tiles import PAIRS_2048, geometry, make_maps, sync, timed, MASK

import healpy as hp


def run(corr, shear, w, dens, wd, ncalls, **kw):
    fn = lambda: corr.get_3x2pt_tomo(shear_maps=shear, density_maps=dens,
                                     weights={"shear": w, "density": wd},
                                     flip_g1=True, return_device=False, **kw)
    out = {}
    ref = None
    for mode in ("off", "single"):
        corr.backend.kernel_3x2pt_tomo_pairs.mode = mode
        res = [np.asarray(x, dtype=np.float64) for x in fn()]
        ms = timed(fn, corr, ncalls)
        entry = {"ms": ms}
        if ref is None:
            ref = res
        else:
            entry["bitwise"] = [bool(np.array_equal(a, b)) for a, b in zip(res, ref)]
            entry["max_rel"] = [float(np.max(np.abs(a - b)) / np.max(np.abs(b))) for a, b in zip(res, ref)]
        out[mode] = entry
        print("   ", mode, json.dumps(entry), flush=True)
    corr.backend.kernel_3x2pt_tomo_pairs.mode = "auto"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncalls", type=int, default=8)
    ap.add_argument("--parts", default="AB")
    ap.add_argument("--out", default="benchmarks/static_treecode/results/stage8_3x2pt_single_pass.json")
    args = ap.parse_args()
    phi, theta, geo, pair_file = geometry(110)
    out = {}
    rng = np.random.default_rng(0)
    if "A" in args.parts:
        for pack in (False, True):
            corr = Correlation(512, phi, theta, device="gpu", map_precision="float32",
                               accumulation_precision="float64", pack_pairs=pack, **geo)
            corr.load_pairs(pair_file)
            corr.prepare(release_host_pairs=True)
            for n_lens in (4, 6):
                shear, w, dens, wd = make_maps(corr, rng, nz=4)
                dens = (rng.normal(size=(n_lens, corr.n_active)) * 0.3).astype(np.float32)
                wd = rng.uniform(0.5, 2.0, size=(n_lens, corr.n_active)).astype(np.float32)
                for label, kw in (("full", {}), ("gc_auto", {"gc_auto_correlations_only": True})):
                    key = f"512_{'packed' if pack else 'unpacked'}_{n_lens}lens_{label}"
                    print(key, flush=True)
                    out[key] = run(corr, shear, w, dens, wd, args.ncalls, **kw)
            del corr; gc.collect()
    if "B" in args.parts:
        mask = hp.ud_grade(hp.read_map(MASK).astype(np.float64), 2048) != 0
        edges = np.geomspace(5.0, 250.0, 12)
        edges = edges[edges < 2 * 110 - 5]
        corr = Correlation(2048, phi, theta, nbins=len(edges) - 1, theta_min=edges[0],
                           theta_max=edges[-1], patch_size=110, theta_Q=110, mask=mask,
                           device="gpu", map_precision="float32", accumulation_precision="float64",
                           resolution_factor=2.9, aperture_nside=512, pack_pairs=True)
        corr.load_pairs(PAIRS_2048, release_host_pairs=True); sync(corr)
        shear = (rng.normal(size=(4, 2, corr.n_active)) * 0.1).astype(np.float32)
        w = rng.uniform(0.5, 2.0, size=(4, corr.n_active)).astype(np.float32)
        dens = (rng.normal(size=(4, corr.n_active)) * 0.3).astype(np.float32)
        wd = rng.uniform(0.5, 2.0, size=(4, corr.n_active)).astype(np.float32)
        print("2048_packed_4lens_full", flush=True)
        out["2048_packed_4lens_full"] = run(corr, shear, w, dens, wd, args.ncalls)
    with open(args.out, "w") as fp:
        json.dump(out, fp, indent=1)


if __name__ == "__main__":
    main()
