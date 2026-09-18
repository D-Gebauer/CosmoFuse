#!/usr/bin/env python
"""T10 -- zeta level: i3PCF data vectors, static treecode vs full resolution.

Production geometry: DES Y3 mask at nside 512, the Q110 patch set (917
patches), 15-250' in 8 log bins (edges beyond 2R-5 dropped), 4 source bins,
real baryonified N-body shear maps (non-Gaussian) from the SBI grid --
Stage 1 could only use Gaussian fields, so the zeta level was deferred to
the real path.

Each map-set is measured twice, at full resolution and with
``resolution_factor=k``, and turned into

    zeta_a,pm = <M_a xi_pm>_patches - <M_a><xi_pm>

with the per-patch contributions kept, so the patch-to-patch error of a
single map-set is available alongside the map-to-map scatter.

Stage A (``--measure``) writes M_a / xi_p / xi_m per resolution; stage B
(``--analyse``) compares them.  Run A twice (once per resolution) so only
one pair set is on the GPU at a time.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import healpy as hp
import numpy as np

MAP_DIR = "/e/ocean1/users/dgebauer/sbi/shear_maps/grid_baryonified/4BINS/"
W_FILE = "/e/ocean1/users/dgebauer/sbi/shear_maps/SumOfWeights_512.npy"
CENTER_DIR = "/e/ocean1/users/dgebauer/sbi/CosmoFuse/Q{r}/"
MASK_FILE = "~/research/lfi/local/DESY3_Mask.fits"
NSIDE = 512


def build_correlation(args, resolution_factor, aperture_nside=None):
    from CosmoFuse import Correlation

    mask = hp.ud_grade(hp.read_map(os.path.expanduser(MASK_FILE)), NSIDE)
    map_inds = np.where(mask != 0)[0]
    cdir = CENTER_DIR.format(r=args.radius)
    phi = np.loadtxt(cdir + "patch_center_original_phi.dat")
    theta = np.loadtxt(cdir + "patch_center_original_theta.dat")

    edges = np.geomspace(args.theta_min, args.theta_max, args.theta_bins + 1)
    edges = edges[edges < 2 * args.radius - 5]

    corr = Correlation(
        NSIDE,
        phi,
        theta,
        nbins=len(edges) - 1,
        theta_min=edges[0],
        theta_max=edges[-1],
        patch_size=args.radius,
        theta_Q=args.radius,
        mask=mask,
        device="gpu",
        map_precision="float32",
        accumulation_precision="float64",
        resolution_factor=resolution_factor,
        aperture_nside=aperture_nside,
    )
    return corr, map_inds, edges


def measure(args):
    tag = "fine" if args.k is None else f"k{args.k:g}"
    if args.aperture_nside is not None:
        tag += f"_ap{args.aperture_nside}"
    if args.f16:
        tag += "_f16"
    corr, map_inds, edges = build_correlation(args, args.k, args.aperture_nside)
    print(f"[{tag}] {corr.n_patches} patches, {corr.nbins} bins "
          f"{edges[0]:.1f}-{edges[-1]:.1f}'", flush=True)
    if getattr(corr, "level_table", None) is not None:
        print(f"[{tag}] level table: {corr.level_table}", flush=True)

    t0 = time.time()
    corr.preprocess()
    print(f"[{tag}] preprocess {time.time() - t0:.1f} s, "
          f"{corr.ntotpairs / 1e6:.1f} M pairs", flush=True)
    corr.prepare()

    w = np.load(W_FILE)[: args.nz]
    if w.shape[1] != map_inds.size:          # full-sky file -> row space
        w = w[:, map_inds]
    w = np.ascontiguousarray(w, dtype=np.float32)
    assert w.shape[1] == map_inds.size, (w.shape, map_inds.size)

    M_a, xi_p, xi_m = [], [], []
    wall = 0.0
    for ifile in range(args.nfiles):
        maps = np.load(MAP_DIR + f"gamma_{ifile:04d}.npy")  # (7,4,nz,2,nrows)
        for ireal in range(maps.shape[0]):
            for iftp in range(maps.shape[1]):
                shear = np.ascontiguousarray(
                    maps[ireal, iftp, : args.nz], dtype=np.float32
                )
                if args.f16:   # T11: what a float16 map archive would deliver
                    shear = shear.astype(np.float16).astype(np.float32)
                t1 = time.time()
                out = corr.get_full_tomo_shear(
                    shear, w, flip_g1=True, return_device=False
                )
                wall += time.time() - t1
                M_a.append(np.asarray(out[0]))
                xi_p.append(np.asarray(out[1]))
                xi_m.append(np.asarray(out[2]))
        del maps
        print(f"[{tag}] file {ifile}: {len(M_a)} map-sets, "
              f"{1e3 * wall / len(M_a):.1f} ms/map-set", flush=True)

    out_file = os.path.join(args.outdir, f"t10_{tag}.npz")
    np.savez_compressed(
        out_file,
        M_a=np.stack(M_a),
        xi_p=np.stack(xi_p),
        xi_m=np.stack(xi_m),
        edges=edges,
        ms_per_mapset=1e3 * wall / len(M_a),
        level_table=json.dumps(
            {k: np.asarray(v).tolist()
             for k, v in (getattr(corr, "level_table", None) or {}).items()}
        ),
    )
    print(f"[{tag}] wrote {out_file}", flush=True)


def _pair_index(nz, i, j):
    """Index of (i, j) in combinations_with_replacement(range(nz), 2)."""
    combs = list(itertools.combinations_with_replacement(range(nz), 2))
    return combs.index((min(i, j), max(i, j)))


def zeta_per_patch(M, xi):
    """Per-patch contributions q with mean(q) == zeta, for every triplet.

    M:  (nmaps, nz, npatch)      xi: (nmaps, ncomb, npatch, nbins)
    ->  (nmaps, ntriplet, npatch, nbins)
    """
    nmaps, nz, npatch = M.shape
    nbins = xi.shape[-1]
    triplets = list(itertools.combinations_with_replacement(range(nz), 3))
    q = np.empty((nmaps, len(triplets), npatch, nbins))
    for t, (zc, z2, z3) in enumerate(triplets):
        c = M[:, zc, :]
        a = xi[:, _pair_index(nz, z2, z3), :, :]
        dc = c - c.mean(axis=1, keepdims=True)
        da = a - a.mean(axis=1, keepdims=True)
        q[:, t] = dc[:, :, None] * da
    return q, triplets


def analyse(args):
    tag_a = args.a or "fine"
    tag_b = args.b
    if tag_b is None:
        tag_b = "fine" if args.k is None else f"k{args.k:g}"
        if args.aperture_nside is not None:
            tag_b += f"_ap{args.aperture_nside}"
    fine = np.load(os.path.join(args.outdir, f"t10_{tag_a}.npz"))
    coarse = np.load(os.path.join(args.outdir, f"t10_{tag_b}.npz"))
    edges = fine["edges"]
    centres = np.sqrt(edges[:-1] * edges[1:])
    lines = []
    W = lines.append

    W(f"# T10 -- zeta level: `{tag_b}` vs `{tag_a}`")
    W("")
    W(f"Q{args.radius}, nside {NSIDE}, {fine['M_a'].shape[2]} patches, "
      f"{fine['M_a'].shape[0]} map-sets, {args.nz} source bins, "
      f"baryonified N-body shear maps.")
    W(f"Timing: {float(fine['ms_per_mapset']):.1f} ms/map-set for `{tag_a}`, "
      f"{float(coarse['ms_per_mapset']):.1f} ms for `{tag_b}` (whole host "
      f"loop: slicing + cast + upload + M_ap + xi+-).")
    for name, npz in ((tag_a, fine), (tag_b, coarse)):
        ap_n = json.loads(str(npz["level_table"])).get("aperture_nside")
        W(f"`{name}`: aperture_nside = "
          f"{ap_n if ap_n else 'full resolution'}.")
    lt = json.loads(str(coarse["level_table"]))
    if lt:
        W(f"Level table: nside per bin {lt.get('nside')}, "
          f"effective k per bin "
          f"{[round(x, 2) for x in lt.get('effective_resolution_factor', [])]}")
    W("")

    for stat in ("p", "m"):
        qf, triplets = zeta_per_patch(fine["M_a"], fine[f"xi_{stat}"])
        qc, _ = zeta_per_patch(coarse["M_a"], coarse[f"xi_{stat}"])
        zf = qf.mean(axis=2)                       # (nmaps, ntrip, nbins)
        zc = qc.mean(axis=2)
        nmaps, _, npatch, _ = qf.shape
        # patch-to-patch error of a single map-set, averaged over map-sets
        sig_patch = (qf.std(axis=2, ddof=1) / np.sqrt(npatch)).mean(axis=0)
        # map-to-map scatter of the data vector, and of the paired difference
        sig_map = zf.std(axis=0, ddof=1)
        d = (zc - zf).mean(axis=0)
        err_d = (zc - zf).std(axis=0, ddof=1) / np.sqrt(nmaps)
        mf = zf.mean(axis=0)
        ratio = zc.mean(axis=0) / np.where(mf == 0, np.nan, mf)
        # the ratio is only meaningful where zeta is resolved by the map sample
        resolved = np.abs(mf) > 3 * sig_map / np.sqrt(nmaps)

        W(f"## zeta_a,{stat}")
        W("")
        W("Ratio coarse/fine of the mean data vector; `.` where the mean "
          "zeta itself is below 3x its own error, so the ratio carries no "
          "information (rows: triplets, cols: angular bins)")
        W("")
        W("| triplet | " + " | ".join(f"{c:.0f}'" for c in centres) + " |")
        W("|---" * (len(centres) + 1) + "|")
        for t, trip in enumerate(triplets):
            W(f"| {trip} | " + " | ".join(
                f"{r:.3f}" if ok else "." for r, ok in zip(ratio[t], resolved[t])
            ) + " |")
        W("")
        W("Mean difference in units of the patch-to-patch error of ONE "
          "map-set (|dzeta| / sigma_patch) -- what a single DES-like "
          "measurement would see")
        W("")
        W("| triplet | " + " | ".join(f"{c:.0f}'" for c in centres) + " |")
        W("|---" * (len(centres) + 1) + "|")
        for t, trip in enumerate(triplets):
            W(f"| {trip} | "
              + " | ".join(f"{v:.2f}" for v in np.abs(d[t]) / sig_patch[t]) + " |")
        W("")
        nsig = np.abs(d) / np.where(err_d == 0, np.inf, err_d)
        W(f"Summary: max |ratio-1| (resolved entries) = "
          f"{np.nanmax(np.abs(ratio[resolved] - 1)):.3f}; "
          f"max |dzeta|/sigma_patch = {(np.abs(d) / sig_patch).max():.2f}; "
          f"max |dzeta|/sigma_map = {(np.abs(d) / sig_map).max():.2f}; "
          f"quadrature sum over the vector = "
          f"{np.sqrt(((d / sig_patch) ** 2).sum()):.1f} sigma_patch "
          f"({d.size} entries).")
        W(f"Against the paired scatter of the {nmaps} map-sets the shift "
          f"reaches {nsig.max():.1f} sigma at most, with "
          f"{int((nsig > 3).sum())} of {d.size} entries above 3 sigma -- "
          f"so at this k the level change is "
          + ("not even detectable with this map sample"
             if (nsig > 3).sum() == 0 else
             "a resolved systematic, but still small compared to the error "
             "of one measurement") + ".")
        W("")

    text = "\n".join(lines)
    out = os.path.join(args.outdir, f"T10_{tag_b}_vs_{tag_a}.md")
    with open(out, "w") as fh:
        fh.write(text + "\n")
    print(text)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--k", type=float, default=None,
                    help="resolution_factor (omit for full resolution)")
    ap.add_argument("--aperture-nside", type=int, default=None,
                    help="evaluate M_ap on the map degraded to this nside "
                         "(T12; omit for the full-resolution aperture)")
    ap.add_argument("--radius", type=int, default=110)
    ap.add_argument("--theta-min", type=float, default=15.0)
    ap.add_argument("--theta-max", type=float, default=250.0)
    ap.add_argument("--theta-bins", type=int, default=8)
    ap.add_argument("--nz", type=int, default=4)
    ap.add_argument("--nfiles", type=int, default=2)
    ap.add_argument("--threads", type=int, default=40)
    ap.add_argument("--outdir", default="benchmarks/static_treecode/results")
    ap.add_argument("--f16", action="store_true",
                    help="T11: round the maps to float16 before measuring")
    ap.add_argument("--a", help="reference tag for --analyse (default: fine)")
    ap.add_argument("--b", help="candidate tag for --analyse (default: k<K>)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if args.measure:
        measure(args)
    if args.analyse:
        analyse(args)


if __name__ == "__main__":
    main()
