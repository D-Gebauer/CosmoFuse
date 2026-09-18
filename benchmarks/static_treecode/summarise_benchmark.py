"""Print the TreeCorr benchmark report(s) as tables."""
import json, sys

for path in sys.argv[1:]:
    r = json.load(open(path))
    print(f"\n=== {path}\n{r['map']}\n{r['n_patches']} patches, R = {r['patch_radius_arcmin']}', "
          f"theta edges {r['theta_edges_arcmin']}")
    print("\n-- timings")
    for m, v in r["methods"].items():
        if "wall_s" in v:
            print(f"  {m:34s} wall {v['wall_s']:8.1f} s on {r['treecorr_processes']} processes ({v['core_s']:.0f} core-s)")
        else:
            print(f"  {m:34s} {v['measure_s_per_map']*1e3:8.1f} ms per map-set | one-off preprocess {v['preprocess_s_one_off']:.0f} s | "
                  f"{v['n_pairs']/1e6:.0f} M pairs | device {v['device_pool_GB']:.1f} GB | levels {v['levels']}"
                  + (f" | {v['n_patches']} patches, {v['n_chunks']} chunks" if "n_patches" in v else ""))
    print("\n-- differences to TreeCorr brute=True (per-patch estimates; patch mean = mean over patches)")
    hdr = f"  {'method':34s} {'stat':4s} {'combos':6s} {'|d| rms':>9s} {'|d| max':>9s} {'typ |xi|':>9s} {'d/sig rms':>9s} {'d/sig max':>9s} {'rel mean max':>12s} {'dmean/sig_mean':>14s}"
    print(hdr)
    for m, v in r["methods"].items():
        if "xip" not in v:
            continue
        for s in ("xip", "xim"):
            for c in ("auto", "cross"):
                d = v[s][c]
                print(f"  {m:34s} {s:4s} {c:6s} {d['abs_diff_per_patch_rms']:9.2e} {d['abs_diff_per_patch_max']:9.2e} "
                      f"{d['typical_abs_value_rms']:9.2e} {d['diff_per_patch_in_sigma_patch_rms']:9.2e} "
                      f"{d['diff_per_patch_in_sigma_patch_max']:9.2e} {d['rel_diff_patch_mean_max_abs']:12.2e} "
                      f"{d['diff_patch_mean_in_sigma_mean_rms']:14.2e}")
    print("\n-- patch-mean xi (auto combos averaged): reference, and relative difference per bin")
    for s in ("xip", "xim"):
        ref = r["methods"]["cosmofuse_k4"][s]["auto"]["patch_mean_per_bin_ref"]
        print(f"  {s} reference      ", " ".join(f"{x:10.3e}" for x in ref))
        for m, v in r["methods"].items():
            if s in v:
                print(f"  {m[:22]:22s}", " ".join(f"{x:+10.2e}" for x in v[s]["auto"]["rel_diff_patch_mean_per_bin"]))
    if "cross_definition_check" in r:
        c = r["cross_definition_check"]
        print(f"\n-- cross-bin estimator definition ({c['n_patches']} patches, CPU float64, max rel. to max|xi+|): "
              f"CosmoFuse mean-of-ratios vs TreeCorr {c['xip_cross_cosmofuse_mean_of_ratios_vs_treecorr_max_rel']:.1e}; "
              f"CosmoFuse recombined as ratio-of-sums vs TreeCorr {c['xip_cross_cosmofuse_recombined_as_ratio_of_sums_vs_treecorr_max_rel']:.1e}")
    if "zeta_level" in r:
        print("\n-- zeta level (auto combos), difference to exact in units of the patch-scatter error of zeta")
        for name, e in r["zeta_level"].items():
            for m, v in e.items():
                if isinstance(v, dict):
                    print(f"  {name:10s} {m:28s} max {v['delta_over_sigma_max']:.2e}  rms {v['delta_over_sigma_rms']:.2e}")
