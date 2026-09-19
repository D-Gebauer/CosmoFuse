"""Realistic benchmark against TreeCorr (one map-set per scenario).

Scenario "512":  nside 512, DES Y3 mask, the ~1000 production Q90 patches
                 (R = theta_Q = 90'), a real baryonified noisy production map
                 with the real DES weights, 4 tomographic bins.
Scenario "2048": nside 2048, same patches, CosmoGrid N-body kappa (bins 1-4)
                 -> Kaiser-Squires shear, DES mask at 2048, uniform weights,
                 DES-like shape noise (optional), theta_min = 5'.

Reference: TreeCorr, bin_slop = 0 AND angle_slop = 0, metric Arc, on the pixel
catalogue of every patch (exact pixel-pair estimator).  bin_slop = 0 alone is
not exact: TreeCorr >= 5 then still approximates the shear projection angles
(xi- off by ~1e-2 of the patch scatter); with angle_slop = 0 it agrees with an
independent brute-force calculation to 1e-10, as CosmoFuse does to 1e-16.  Also TreeCorr with its default bin_slop
(TreeCorr's own tree approximation) for context.  CosmoFuse runs on the GPU:
full resolution (where it fits), resolution_factor = 4, and 4 + pack_pairs.

Reported per method vs the reference, separately for xi+ / xi- and for
auto / cross tomographic combinations:
  * absolute differences of the per-patch estimates (max, rms) and the same
    in units of the patch-to-patch scatter of that bin,
  * relative differences of the patch-averaged correlation function,
  * timings.
Cross-combinations: CosmoFuse averages the two orientation *ratios*,
TreeCorr takes the ratio of the summed orientations; with different weights
per bin these differ slightly by construction.
"""
import argparse, gc, json, os, time
from multiprocessing import Pool

import h5py, healpy as hp, numpy as np

SBI = "/e/ocean1/users/dgebauer/sbi"
MASK = "/home/moon/dgebauer/research/lfi/local/DESY3_Mask.fits"
DATA = os.path.expanduser("~/research/CosmoFuse-treecode-data")
G = {}  # globals shared with the TreeCorr worker processes (fork)


def kappa_to_shear(kappa, nthreads_note=None):
    nside = hp.npix2nside(kappa.size)
    lmax = 3 * nside - 1
    ell = np.arange(lmax + 1)
    f = np.sqrt(np.clip((ell + 2) * (ell - 1), 0, None) / np.maximum(ell * (ell + 1), 1))
    elm = hp.almxfl(hp.map2alm(kappa.astype(np.float64), lmax=lmax), f)
    return hp.alm2map_spin([elm, 0 * elm], nside, 2, lmax)


def treecorr_patch(i):
    import treecorr

    nside, mask = G["nside"], G["mask"]
    vec = hp.ang2vec(G["theta_c"][i], G["phi_c"][i])
    disc = hp.query_disc(nside, vec, np.radians(G["R"] / 60))
    pix = disc[mask[disc]]
    rows = G["lut"][pix]
    th, ph = hp.pix2ang(nside, pix)
    ra, dec = ph, np.pi / 2 - th
    nz = G["shear"].shape[0]
    cats = [treecorr.Catalog(ra=ra, dec=dec, ra_units="rad", dec_units="rad",
                             g1=G["shear"][z, 0, rows], g2=G["shear"][z, 1, rows],
                             w=G["w"][z, rows], flip_g1=True) for z in range(nz)]
    edges = G["edges"]
    out_p, out_m = [], []
    for z1 in range(nz):
        for z2 in range(z1, nz):
            gg = treecorr.GGCorrelation(min_sep=edges[0], max_sep=edges[-1], nbins=len(edges) - 1,
                                        sep_units="rad", metric="Arc", num_threads=1,
                                        **G["tc_kwargs"])
            if z1 == z2:
                gg.process(cats[z1])
            else:
                gg.process(cats[z1], cats[z2])
            out_p.append(gg.xip.copy()); out_m.append(gg.xim.copy())
    return np.array(out_p), np.array(out_m)


TC_MODES = {
    "treecorr_brute": {"brute": True},                        # reference
    "treecorr_bin_slop0_angle_slop0": {"bin_slop": 0.0, "angle_slop": 0.0},
    "treecorr_bin_slop0": {"bin_slop": 0.0},
    "treecorr_default": {},
}


def run_treecorr(mode, nproc, npatch):
    G["tc_kwargs"] = TC_MODES[mode]
    t0 = time.perf_counter()
    with Pool(nproc) as pool:
        res = pool.map(treecorr_patch, range(npatch), chunksize=1)
    wall = time.perf_counter() - t0
    xip = np.array([r[0] for r in res]).transpose(1, 0, 2)
    xim = np.array([r[1] for r in res]).transpose(1, 0, 2)
    return xip, xim, wall


def run_cosmofuse(label, k, pack, chunk=None, search="float64", **kw):
    from CosmoFuse import Correlation

    npatch = G["npatch"]
    chunks = [(0, npatch)] if chunk is None else [(a, min(a + chunk, npatch)) for a in range(0, npatch, chunk)]
    Ma, xip, xim = [], [], []
    t_pre = t_meas = 0.0
    n_pairs = 0
    pool_gb = 0.0
    for a, b in chunks:
        corr = Correlation(G["nside"], G["phi_c"][a:b], G["theta_c"][a:b], nbins=len(G["edges"]) - 1,
                           theta_min=np.degrees(G["edges"][0]) * 60, theta_max=np.degrees(G["edges"][-1]) * 60,
                           patch_size=G["R"], theta_Q=G["R"], mask=G["mask"], device="gpu",
                           map_precision="float32", accumulation_precision="float64",
                           resolution_factor=k, pack_pairs=pack, pair_search_precision=search,
                           memory_budget_gb=float("inf"), **kw)
        xp = corr.backend.module
        t0 = time.perf_counter()
        corr.preprocess(release_host_pairs=True)
        xp.cuda.runtime.deviceSynchronize()
        t_pre += time.perf_counter() - t0
        n_pairs += int(corr.ntotpairs)
        assert np.array_equal(corr.row_pix, G["row_pix"])
        g = np.ascontiguousarray(G["shear"], dtype=np.float32)
        w = np.ascontiguousarray(G["w"], dtype=np.float32); w.flags.writeable = False
        out = corr.get_full_tomo_shear(g, w, flip_g1=True, return_device=False)   # warm-up + result
        ts = []
        for _ in range(3 if chunk is None else 1):
            xp.cuda.runtime.deviceSynchronize(); t0 = time.perf_counter()
            corr.get_full_tomo_shear(g, w, flip_g1=True, return_device=False)
            xp.cuda.runtime.deviceSynchronize(); ts.append(time.perf_counter() - t0)
        t_meas += float(np.median(ts))
        pool_gb = max(pool_gb, xp.get_default_memory_pool().total_bytes() / 1e9)
        Ma.append(np.asarray(out[0], dtype=np.float64)); xip.append(np.asarray(out[1], dtype=np.float64))
        xim.append(np.asarray(out[2], dtype=np.float64))
        levels = corr.level_table["nside"].tolist()
        del corr; gc.collect(); xp.get_default_memory_pool().free_all_blocks()
    info = {"label": label, "preprocess_s_one_off": t_pre, "measure_s_per_map": t_meas, "n_pairs": n_pairs,
            "device_pool_GB": pool_gb, "levels": levels, "n_chunks": len(chunks)}
    return np.concatenate(Ma, axis=1), np.concatenate(xip, axis=1), np.concatenate(xim, axis=1), info


def compare(x, ref, combs_auto):
    """x, ref: (ncomb, npatch, nbins)."""
    out = {}
    for name, sel in (("auto", combs_auto), ("cross", ~combs_auto)):
        a, r = x[sel], ref[sel]
        d = a - r
        sigma = r.std(axis=1, keepdims=True)
        mean_r, mean_a = r.mean(axis=1), a.mean(axis=1)
        out[name] = {
            "abs_diff_per_patch_max": float(np.max(np.abs(d))),
            "abs_diff_per_patch_rms": float(np.sqrt(np.mean(d * d))),
            "typical_abs_value_rms": float(np.sqrt(np.mean(r * r))),
            "diff_per_patch_in_sigma_patch_rms": float(np.sqrt(np.mean((d / sigma) ** 2))),
            "diff_per_patch_in_sigma_patch_max": float(np.max(np.abs(d / sigma))),
            "rel_diff_patch_mean_per_bin": (mean_a / mean_r - 1).mean(axis=0).tolist(),
            "rel_diff_patch_mean_max_abs": float(np.max(np.abs(mean_a / mean_r - 1))),
            "abs_diff_patch_mean_per_bin": (mean_a - mean_r).mean(axis=0).tolist(),
            "patch_mean_per_bin_ref": mean_r.mean(axis=0).tolist(),
            "diff_patch_mean_in_sigma_mean_rms": float(np.sqrt(np.mean(
                ((mean_a - mean_r) / (sigma[:, 0] / np.sqrt(r.shape[1]))) ** 2))),
        }
    return out


def zeta(Ma, xi, combs):
    """zeta(z, z, z) for the auto combinations: <M xi> - <M><xi> over patches."""
    out, err = [], []
    autos = [c for c, (i, j) in enumerate(combs) if i == j]
    for z, c in enumerate(autos):
        prod = Ma[z][:, None] * xi[c]
        out.append(prod.mean(0) - Ma[z].mean() * xi[c].mean(0))
        err.append(((Ma[z] - Ma[z].mean())[:, None] * (xi[c] - xi[c].mean(0))).std(0) / np.sqrt(Ma.shape[1]))
    return np.array(out), np.array(err)


def make_shear(path, nside, run):
    """Child process: multi-threaded Kaiser-Squires, row-space shear to disk."""
    import os.path

    row_pix = np.load(os.path.join(os.path.dirname(path), "row_pix.npy"))
    with h5py.File(f"{DATA}/kappa_run_{run:04d}.h5", "r") as fp:
        kappa = fp["kappa"][1:5].astype(np.float64)
    if nside != 2048:
        kappa = np.array([hp.ud_grade(k_, nside) for k_ in kappa])
    shear = np.empty((4, 2, row_pix.size))
    for z in range(4):
        g1, g2 = kappa_to_shear(kappa[z])
        shear[z, 0], shear[z, 1] = g1[row_pix], g2[row_pix]
    np.save(path, shear)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-shear", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--scenario", choices=["512", "2048"], required=True)
    ap.add_argument("--npatch", type=int, default=None)
    ap.add_argument("--nproc", type=int, default=48)
    ap.add_argument("--map", choices=["real", "nbody"], default=None,
                    help="real = production map + DES weights (nside 512 only); nbody = CosmoGrid")
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0],
                    help="nbody: shape noise per component and pixel (DES-like: 0.032 at 512, 0.128 at 2048)")
    ap.add_argument("--fullres-patches", type=int, default=48, help="2048: full-resolution CosmoFuse on this many patches")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.make_shear:
        make_shear(args.make_shear, int(args.scenario), args.run)
        return

    Q = 90
    phi_c = np.loadtxt(f"{SBI}/CosmoFuse/Q{Q}/patch_center_original_phi.dat")
    theta_c = np.loadtxt(f"{SBI}/CosmoFuse/Q{Q}/patch_center_original_theta.dat")
    npatch = min(args.npatch or len(phi_c), len(phi_c))
    nside = int(args.scenario)
    m = hp.read_map(MASK)
    if nside == 512:
        mask = hp.ud_grade(m, nside) != 0                       # as in production
        edges = np.geomspace(15.0, 250.0, 9)
    else:
        if hp.npix2nside(m.size) != nside:
            m = hp.ud_grade(m.astype(np.float64), nside)
        mask = m != 0
        edges = np.geomspace(5.0, 250.0, 12)
    edges = np.radians(edges[edges < 2 * Q - 5] / 60)
    row_pix = np.flatnonzero(mask)
    lut = np.full(mask.size, -1, dtype=np.int64); lut[row_pix] = np.arange(row_pix.size)

    use_real = (args.map or ("real" if nside == 512 else "nbody")) == "real"
    t0 = time.perf_counter()
    if use_real:
        assert nside == 512
        shear = np.load(f"{SBI}/shear_maps/grid_baryonified/4BINS/gamma_0000.npy", mmap_mode="r")[0, 0]
        shear = np.ascontiguousarray(shear, dtype=np.float64)   # (4, 2, n_footprint) = row space
        w = np.ascontiguousarray(np.load(f"{SBI}/shear_maps/SumOfWeights_512.npy")[:, row_pix], dtype=np.float64)
        map_note = "real baryonified production map gamma_0000[0,0], real DES weights"
    else:
        import subprocess, sys, tempfile

        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            np.save(f"{tmp}/row_pix.npy", row_pix)
            subprocess.run([sys.executable, __file__, "--make-shear", f"{tmp}/shear.npy",
                            "--scenario", str(nside), "--run", str(args.run)], check=True)
            shear = np.load(f"{tmp}/shear.npy")
        w = np.ones((4, row_pix.size))
        map_note = f"CosmoGrid N-body kappa run {args.run} bins 1-4 -> KS shear, uniform weights"
    t_maps = time.perf_counter() - t0
    signal = shear
    cases = []
    for noise in (args.noise if not use_real else [None]):
        shear = signal
        note = map_note
        if noise:
            shear = signal + np.random.default_rng(1234).normal(0.0, noise, signal.shape)
            note += f", shape noise {noise:g} per component and pixel"
        elif noise is not None:
            note += ", noise-free"
        cases.append([shear, note, "_real" if use_real else f"_nbody_noise{noise:g}", None])
    # Phase 1: all TreeCorr runs (forked worker pools) before numba / CUDA are
    # initialised in this process.
    modes = list(TC_MODES) if nside == 512 else ["treecorr_brute", "treecorr_default"]
    for case in cases:
        G.update(nside=nside, mask=mask, lut=lut, row_pix=row_pix, shear=case[0], w=w, edges=edges,
                 phi_c=phi_c[:npatch], theta_c=theta_c[:npatch], R=float(Q), npatch=npatch)
        tc = {}
        for mode in modes:
            xp_, xm_, wall = run_treecorr(mode, args.nproc, npatch)
            tc[mode] = (xp_, xm_, wall)
            print(case[2], mode, wall, flush=True)
        case[3] = tc
    # Phase 2: CosmoFuse
    for shear, note, tag, tc in cases:
        one_run(args, nside, mask, lut, row_pix, shear, w, edges, phi_c, theta_c, npatch, Q, note, t_maps, tag, tc)


def one_run(args, nside, mask, lut, row_pix, shear, w, edges, phi_c, theta_c, npatch, Q, map_note, t_maps, tag, tc):
    G.update(nside=nside, mask=mask, lut=lut, row_pix=row_pix, shear=shear, w=w, edges=edges,
             phi_c=phi_c[:npatch], theta_c=theta_c[:npatch], R=float(Q), npatch=npatch)
    nz = 4
    combs = [(i, j) for i in range(nz) for j in range(i, nz)]
    autos = np.array([i == j for i, j in combs])
    report = {"scenario": args.scenario, "map": map_note, "n_patches": npatch, "patch_radius_arcmin": Q,
              "theta_edges_arcmin": np.round(np.degrees(edges) * 60, 2).tolist(), "n_active": int(row_pix.size),
              "map_preparation_s": t_maps, "treecorr_processes": args.nproc, "methods": {}}
    print(json.dumps({k: v for k, v in report.items() if k != "methods"}), flush=True)

    # ---- TreeCorr -------------------------------------------------------------
    ref_p, ref_m, wall = tc["treecorr_brute"]
    report["methods"]["treecorr_brute"] = {"wall_s": wall, "core_s": wall * args.nproc, "role": "reference"}
    for mode, (tc_p, tc_m, wall) in tc.items():
        if mode == "treecorr_brute":
            continue
        report["methods"][mode] = {
            "wall_s": wall, "core_s": wall * args.nproc,
            "xip": compare(tc_p, ref_p, autos), "xim": compare(tc_m, ref_m, autos)}

    # ---- CosmoFuse ---------------------------------------------------------------
    runs = [("cosmofuse_k4", 4.0, False, "float64"), ("cosmofuse_k4_packed", 4.0, True, "float64")]
    if nside == 512:
        runs = [("cosmofuse_full_resolution", None, False, "float64"),
                # the 4.20.0 default: pair search at float32 (rotation precision)
                ("cosmofuse_full_resolution_float32_search", None, False, "rotation")] + runs
    results = {}
    for label, k, pack, search in runs:
        Ma, xp_, xm_, info = run_cosmofuse(label, k, pack, None, search=search)
        info["xip"] = compare(xp_, ref_p, autos); info["xim"] = compare(xm_, ref_m, autos)
        report["methods"][label] = info
        results[label] = (Ma, xp_, xm_)
        print(label, json.dumps({k_: v for k_, v in info.items() if k_ not in ("xip", "xim")}), flush=True)
    if nside == 2048 and args.fullres_patches > 0:
        n = min(args.fullres_patches, npatch)
        G["npatch"] = n
        Ma, xp_, xm_, info = run_cosmofuse("cosmofuse_full_resolution_subset", None, False, chunk=12)
        G["npatch"] = npatch
        info["n_patches"] = n
        info["xip"] = compare(xp_, ref_p[:, :n], autos); info["xim"] = compare(xm_, ref_m[:, :n], autos)
        report["methods"]["cosmofuse_full_resolution_subset"] = info
        print("full-res subset", json.dumps({k_: v for k_, v in info.items() if k_ not in ("xip", "xim")}), flush=True)

    # ---- zeta level (auto combinations): exact xi from TreeCorr, M_ap from CosmoFuse ----
    Ma = results["cosmofuse_k4"][0]
    zl = {}
    for name, ref, idx in (("zeta_plus", ref_p, 1), ("zeta_minus", ref_m, 2)):
        z_ref, z_err = zeta(Ma, ref, combs)
        entry = {"zeta_ref": z_ref.tolist(), "zeta_err_patch_scatter": z_err.tolist()}
        for label in results:
            z_x, _ = zeta(Ma, results[label][idx], combs)
            entry[label] = {"delta_over_sigma_max": float(np.max(np.abs((z_x - z_ref) / z_err))),
                            "delta_over_sigma_rms": float(np.sqrt(np.mean(((z_x - z_ref) / z_err) ** 2))),
                            "ratio_to_ref_mean_per_bin": np.mean(z_x / z_ref, axis=0).tolist()}
        zl[name] = entry
    report["zeta_level"] = zl

    # ---- cross-combination estimator definition (CPU, small subset) ---------------
    # cross_definition_check() removed: it recombined CosmoFuse's raw
    # numerators/denominators "TreeCorr-style" to compare against the
    # pre-5.0 average-of-ratios cross estimator.  Since 5.0 CosmoFuse *is*
    # the ratio of summed orientations, so the check compared the estimator
    # against itself and could not fail.

    tag = args.tag or tag
    out = f"benchmarks/static_treecode/results/benchmark_treecorr_{args.scenario}{tag}.json"
    with open(out, "w") as fp:
        json.dump(report, fp, indent=1)
    print("written", out)


if __name__ == "__main__":
    main()
