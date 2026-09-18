"""Stage 0 (CPU part): real geometry numbers for the production configuration.

Replaces the [estimate] pair counts of the (retired) implementation guide
with counts
from the actual pair finder on the DES Y3 mask:

* production geometry (nside 512, 15-250', 8 bins, Q110, nside_centers 32,
  f_mask 0.5): total pairs, pairs per bin, pairs per patch, active pixels,
  footprint pixels, aperture entries, host/device bytes;
* one representative patch at nside 1024 / 2048 full resolution (5-250',
  11 bins) to anchor the nside^4 scaling.

The GPU part of Stage 0 (nsys profile, pair-evaluations per second) needs the
A100 and is not covered here.
"""

import argparse
import json
import time

import healpy as hp
import numpy as np

from CosmoFuse.correlations import Correlation
from CosmoFuse.utils import select_patch_centers

MASK = "/home/david/Documents/research/PDFast/data/DESY3_Mask.fits"


def degrade_mask(mask: np.ndarray, nside: int) -> np.ndarray:
    if hp.npix2nside(mask.size) == nside:
        return mask.astype(bool)
    return hp.ud_grade(mask.astype(np.float64), nside) >= 0.5


def geometry_stats(corr: Correlation) -> dict:
    bins = np.asarray(corr.bins, dtype=np.int64)
    per_patch = bins.sum(axis=1)
    pair_pix = np.unique(np.concatenate([p.ravel() for p in corr.pair_inds]))
    q_pix = np.unique(np.concatenate(corr.Q_inds))
    active = np.union1d(pair_pix, q_pix)
    n_q = int(sum(q.size for q in corr.Q_inds))
    pair_bytes = per_patch.sum() * (
        2 * corr.index_dtype.itemsize + 2 * corr.rotation_complex_dtype.itemsize
    )
    q_bytes = n_q * (4 + 3 * corr.rotation_dtype.itemsize)
    return {
        "nside": int(corr.nside),
        "n_patches": int(corr.n_patches),
        "total_pairs": int(per_patch.sum()),
        "pairs_per_bin": bins.sum(axis=0).tolist(),
        "pairs_per_patch_min_med_max": [
            int(per_patch.min()),
            int(np.median(per_patch)),
            int(per_patch.max()),
        ],
        "footprint_pixels": int(corr.map_inds.size),
        "npix": int(hp.nside2npix(corr.nside)),
        "active_pixels_pairs": int(pair_pix.size),
        "active_pixels_aperture": int(q_pix.size),
        "active_pixels_union": int(active.size),
        "aperture_entries": n_q,
        "pair_bytes_GB": float(pair_bytes / 1e9),
        "aperture_bytes_GB": float(q_bytes / 1e9),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta-q", type=float, default=110.0)
    ap.add_argument("--nside-centers", type=int, default=32)
    ap.add_argument("--f-mask", type=float, default=0.5)
    ap.add_argument("--out", default="stage0_geometry.json")
    args = ap.parse_args()

    mask2048 = hp.read_map(MASK).astype(bool)
    out = {}

    # --- production geometry at nside 512 --------------------------------
    mask512 = degrade_mask(mask2048, 512)
    t0 = time.perf_counter()
    corr = Correlation.from_mask(
        512,
        mask512,
        args.nside_centers,
        patch_size=args.theta_q,
        theta_Q=args.theta_q,
        f_mask=args.f_mask,
        nbins=8,
        theta_min=15,
        theta_max=250,
        device="cpu",
    )
    corr.calculate_pairs_M_a()
    corr.calculate_pairs_2PCF()
    out["production_512"] = geometry_stats(corr)
    out["production_512"]["preprocess_s"] = time.perf_counter() - t0
    phi_c, theta_c = corr.phi_center, corr.theta_center

    # --- one central, fully unmasked-ish patch at higher nside -----------
    # pick the patch with the largest nside-512 pair count (cleanest patch)
    per_patch = np.asarray(corr.bins, dtype=np.int64).sum(axis=1)
    best = int(np.argmax(per_patch))
    for nside in (512, 1024, 2048):
        m = degrade_mask(mask2048, nside)
        c = Correlation(
            nside,
            phi_c[best : best + 1],
            theta_c[best : best + 1],
            nbins=11,
            theta_min=5,
            theta_max=250,
            patch_size=args.theta_q,
            theta_Q=args.theta_q,
            mask=m,
            device="cpu",
        )
        t0 = time.perf_counter()
        c.calculate_pairs_2PCF()
        dt = time.perf_counter() - t0
        b = np.asarray(c.bins[0], dtype=np.int64)
        vec = hp.ang2vec(theta_c[best], phi_c[best])
        disc = hp.query_disc(nside, vec, np.radians(args.theta_q / 60))
        out[f"single_patch_fullres_{nside}"] = {
            "patch_pixels": int(m[disc].sum()),
            "pairs_per_bin": b.tolist(),
            "pairs": int(b.sum()),
            "bytes_per_pair": 24,
            "GB_per_1000_patches": float(b.sum() * 24 * 1000 / 1e9),
            "pairfind_s": dt,
        }

    with open(args.out, "w") as fp:
        json.dump(out, fp, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
