"""Analyse stage1_prototype.py output: ratio matrices, T9, information loss,
and per-k level tables with real pair counts."""

import argparse

import healpy as hp
import numpy as np

STATS = {"xip": (0, 2), "xim": (1, 2), "xit": (3, 4), "xig": (5, 6)}
BASE = 2048


def level_table(edges_arcmin, k, base=BASE, min_nside=64):
    out = []
    for lo in edges_arcmin[:-1]:
        nside = min_nside
        while hp.nside2resol(nside, arcmin=True) > lo / k and nside < base:
            nside *= 2
        out.append(nside)
    return np.array(out)


def ratio_of_sums(c_num, c_den, f_num, f_den):
    """sum-over-samples estimator ratio, with delete-one-patch jackknife.
    inputs: (npatch, nmaps_sel, nbins)."""
    def est(sel):
        c = c_num[sel].sum(axis=(0, 1)) / c_den[sel].sum(axis=(0, 1))
        f = f_num[sel].sum(axis=(0, 1)) / f_den[sel].sum(axis=(0, 1))
        return c / f

    n = c_num.shape[0]
    full = est(np.arange(n))
    jk = np.array([est(np.delete(np.arange(n), i)) for i in range(n)])
    err = np.sqrt((n - 1) / n * ((jk - jk.mean(0)) ** 2).sum(0))
    return full, err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="results/stage1.npz")
    ap.add_argument("--geo", default="centroid")
    args = ap.parse_args()
    np.set_printoptions(linewidth=220, precision=4, suppress=True)

    r = np.load(args.inp)
    fine, coarse = r["fine"], r["coarse"]
    edges = r["edges_arcmin"]
    levels = r["levels"]
    geo = list(r["geo_variants"])
    nvar = len(r["variants"])
    nreal = int(r["nreal"])
    nb = edges.size - 1
    vidx = {v: np.arange(nreal) * nvar + i for i, v in enumerate(r["variants"])}
    ig = geo.index(args.geo)

    print(f"{fine.shape[0]} patches x {nreal} realisations; bins (arcmin):")
    print("  lo:", np.round(edges[:-1], 1))

    print("\n=== 0. full-resolution reference / theory (variant A) ===")
    for s, (inum, iden) in STATS.items():
        m = vidx["A"]
        v = fine[:, inum][:, m].sum((0, 1)) / fine[:, iden][:, m].sum((0, 1))
        print(f"  {s}: ", v / r["theory_" + s])

    print(f"\n=== 1. coarse/fine ratio, variant A, geometry '{args.geo}' (rows: level) ===")
    R = {}
    for s, (inum, iden) in STATS.items():
        print(f"  -- {s}")
        R[s] = np.zeros((len(levels), nb))
        for il, ns in enumerate(levels):
            m = vidx["A"]
            ratio, err = ratio_of_sums(
                coarse[ig, il][:, inum][:, m], coarse[ig, il][:, iden][:, m],
                fine[:, inum][:, m], fine[:, iden][:, m],
            )
            R[s][il] = ratio
            print(f"   {ns:5d}: ", ratio, " +-", np.round(err.max(), 4))

    print("\n=== 2. geometry variants: (ratio - 1) relative to 'centroid' ===")
    for s, (inum, iden) in STATS.items():
        for gv in geo:
            if gv == args.geo:
                continue
            jg = geo.index(gv)
            print(f"  -- {s}, {gv} minus {args.geo}")
            for il, ns in enumerate(levels[:4]):
                m = vidx["A"]
                ratio, _ = ratio_of_sums(
                    coarse[jg, il][:, inum][:, m], coarse[jg, il][:, iden][:, m],
                    fine[:, inum][:, m], fine[:, iden][:, m],
                )
                print(f"   {ns:5d}: ", ratio - R[s][il])

    print("\n=== 3. T9 data/sim consistency: ratio(B: n_gal weights) / ratio(A: uniform) - 1 ===")
    for s, (inum, iden) in STATS.items():
        print(f"  -- {s}")
        for il, ns in enumerate(levels[:4]):
            m = vidx["B"]
            ratio, err = ratio_of_sums(
                coarse[ig, il][:, inum][:, m], coarse[ig, il][:, iden][:, m],
                fine[:, inum][:, m], fine[:, iden][:, m],
            )
            print(f"   {ns:5d}: ", ratio / R[s][il] - 1, " +-", np.round(err.max(), 4))

    print("\n=== 4. information loss (variant C, noisy): corr(coarse, fine) and var ratio over patches x reals ===")
    RHO = {}
    for s, (inum, iden) in STATS.items():
        print(f"  -- {s}")
        m = vidx["C"]
        f = (fine[:, inum][:, m] / fine[:, iden][:, m]).reshape(-1, nb)
        RHO[s] = np.zeros((len(levels), nb))
        for il, ns in enumerate(levels):
            with np.errstate(invalid="ignore", divide="ignore"):
                c = (coarse[ig, il][:, inum][:, m] / coarse[ig, il][:, iden][:, m]).reshape(-1, nb)
            rho = np.array([np.corrcoef(c[:, b], f[:, b])[0, 1] for b in range(nb)])
            vr = c.var(0) / f.var(0)
            RHO[s][il] = rho
            print(f"   {ns:5d} rho: ", rho)
            print(f"         var: ", vr)

    print("\n=== 5. special patches: xim and xit ratio (variant A) ===")
    for name, ip in zip(r["special"], range(len(r["special"]))):
        print(f"  -- {name} (dec {r['dec'][ip]:.1f}, f_mask {r['fmask'][ip]:.2f})")
        for s in ("xim", "xit"):
            inum, iden = STATS[s]
            for gv in ("centroid", "rot"):
                jg = geo.index(gv)
                for il, ns in enumerate(levels[:3]):
                    m = vidx["A"]
                    c = coarse[jg, il, ip, inum][m].sum(0) / coarse[jg, il, ip, iden][m].sum(0)
                    f = fine[ip, inum][m].sum(0) / fine[ip, iden][m].sum(0)
                    print(f"   {s} {gv:8s} {ns:5d}: ", c / f)

    print("\n=== 6. per-k level tables (median patch pair counts, real) ===")
    npair_fine = np.median(fine[:, 7, 0, :], axis=0)
    npair_coarse = np.median(coarse[ig, :, :, 7, 0, :], axis=1)  # (level, bin)
    print("  full resolution: %.1f M pairs/patch" % (npair_fine.sum() / 1e6))
    lev_list = [BASE] + list(levels)
    for k in (2.0, 2.9, 4.0, 5.8, 8.0):
        tab = level_table(edges, k)
        keff = edges[:-1] / np.array([hp.nside2resol(int(n), arcmin=True) for n in tab])
        npairs = np.array(
            [npair_fine[b] if tab[b] == BASE else npair_coarse[list(levels).index(tab[b]), b] for b in range(nb)]
        )
        print(f"\n  k = {k}:  nside/bin {tab.tolist()}")
        print(f"     k_eff   ", np.round(keff, 2))
        print(f"     pairs/patch {npairs.sum()/1e6:.2f} M  -> {npairs.sum()*24*1000/1e9:.1f} GB / 1000 patches (24 B), "
              f"{npairs.sum()*8*1000/1e9:.1f} GB (8 B packed);  per bin (M): ", np.round(npairs / 1e6, 3))
        for s in STATS:
            row = np.array([1.0 if tab[b] == BASE else R[s][list(levels).index(tab[b]), b] for b in range(nb)])
            rho = np.array([1.0 if tab[b] == BASE else RHO[s][list(levels).index(tab[b]), b] for b in range(nb)])
            print(f"     {s} ratio", row, "| min rho %.4f" % rho.min())


if __name__ == "__main__":
    main()
