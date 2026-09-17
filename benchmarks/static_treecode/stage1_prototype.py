"""Stage 1 science gate: what does per-bin coarsening do to the statistics?

Standalone prototype (numpy/numba/healpy/pyccl; does not use the CosmoFuse
measurement code, only re-implements its pair geometry) that measures, on
nside-2048 maps with the real DES Y3 mask, the patch-level xi+, xi-, gamma_t
and xi_g

* at full resolution (brute force over all fine pixel pairs), and
* at every coarser HEALPix level (1024 ... 64), for *all* angular bins,

so that the result is a complete (statistic, level, bin) matrix.  Any
resolution factor k is then just a rule that picks one level per bin.

Coarse cells are per patch (members = unmasked fine pixels inside the patch
disc), values are weighted means, weights are summed.  Three geometry
variants are compared for the coarse cells:

  centre    HEALPix parent pixel centre
  centroid  unit-vector centroid of the member pixels (binary mask)
  rot       centroid + each member shear parallel-transported to the centroid

Map variants per realisation:

  A  noise-free, uniform weights
  B  noise-free, Poisson n_gal weights          (data/sim consistency, T9)
  C  shape noise + lens shot noise, n_gal weights (information loss)
"""

import argparse
import time

import healpy as hp
import numpy as np
from numba import get_num_threads, njit, prange

MASK = "/home/david/Documents/research/PDFast/data/DESY3_Mask.fits"
NSIDE = 2048
LEVELS = (1024, 512, 256, 128, 64)
NSTAT = 8  # xip, xim, ss_den, xit, ds_den, xig, dd_den, npairs


# --------------------------------------------------------------------------
# geometry (same formulas as CosmoFuse.correlations._compute_pairs_impl)
# --------------------------------------------------------------------------
@njit(cache=True, inline="always")
def pair_rot(x1, y1, z1, x2, y2, z2):
    sinC1 = x1 * y2 - x2 * y1
    dsq_AC1 = x1 * x1 + y1 * y1 + (z1 - 1.0) * (z1 - 1.0)
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    dsq_BC1 = dx * dx + dy * dy + dz * dz
    dsq_AB1 = x2 * x2 + y2 * y2 + (z2 - 1.0) * (z2 - 1.0)
    cosC1 = 0.5 * (dsq_AC1 + dsq_BC1 - dsq_AB1 - 0.5 * dsq_AC1 * dsq_BC1)
    R2 = sinC1 * sinC1 + cosC1 * cosC1
    if R2 > 0:
        c1 = (sinC1 * sinC1 - cosC1 * cosC1) / R2
        s1 = (2.0 * sinC1 * cosC1) / R2
    else:
        c1 = -1.0
        s1 = 0.0
    sinC2 = -sinC1
    cosC2 = 0.5 * (dsq_AB1 + dsq_BC1 - dsq_AC1 - 0.5 * dsq_AB1 * dsq_BC1)
    R2 = sinC2 * sinC2 + cosC2 * cosC2
    if R2 > 0:
        c2 = (sinC2 * sinC2 - cosC2 * cosC2) / R2
        s2 = (2.0 * sinC2 * cosC2) / R2
    else:
        c2 = -1.0
        s2 = 0.0
    return c1, s1, c2, s2


@njit(parallel=True, cache=True)
def accumulate(vec, g1, g2, ws, d, wd, cos_edges):
    """Brute-force all pairs.  Map arrays have layout (npts, nmaps).

    Returns acc[NSTAT, nmaps, nbins].
    """
    npts = vec.shape[0]
    nmaps = g1.shape[1]
    nedges = cos_edges.shape[0]
    nbins = nedges - 1
    cos_max = cos_edges[0]
    cos_min = cos_edges[nedges - 1]
    nchunks = 8 * get_num_threads()
    acc = np.zeros((nchunks, NSTAT, nmaps, nbins))
    nrows = npts - 1 if npts > 1 else 0
    for ch in prange(nchunks):
        for m_ in range(ch, nrows, nchunks):
            # light/heavy interleave to balance the triangular loop
            half = m_ >> 1
            odd = m_ & 1
            i = half + odd * (nrows - 1 - 2 * half)
            x1 = vec[i, 0]
            y1 = vec[i, 1]
            z1 = vec[i, 2]
            for j in range(i + 1, npts):
                x2 = vec[j, 0]
                y2 = vec[j, 1]
                z2 = vec[j, 2]
                ct = x1 * x2 + y1 * y2 + z1 * z2
                if ct >= cos_max or ct <= cos_min:
                    continue
                lo = 0
                hi = nedges
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if cos_edges[mid] > ct:
                        lo = mid + 1
                    else:
                        hi = mid
                b = lo - 1
                if b < 0 or b >= nbins or not (ct > cos_edges[b + 1]):
                    continue
                c1, s1, c2, s2 = pair_rot(x1, y1, z1, x2, y2, z2)
                acc[ch, 7, 0, b] += 1.0
                for m in range(nmaps):
                    a_r = g1[i, m] * c1 - g2[i, m] * s1
                    a_i = g1[i, m] * s1 + g2[i, m] * c1
                    b_r = g1[j, m] * c2 - g2[j, m] * s2
                    b_i = g1[j, m] * s2 + g2[j, m] * c2
                    wss = ws[i, m] * ws[j, m]
                    acc[ch, 0, m, b] += wss * (b_r * a_r + b_i * a_i)
                    acc[ch, 1, m, b] += wss * (b_r * a_r - b_i * a_i)
                    acc[ch, 2, m, b] += wss
                    # gamma_t: lens i / source j and lens j / source i
                    w_ab = wd[i, m] * ws[j, m]
                    w_ba = wd[j, m] * ws[i, m]
                    acc[ch, 3, m, b] += w_ab * d[i, m] * (-b_r) + w_ba * d[j, m] * (-a_r)
                    acc[ch, 4, m, b] += w_ab + w_ba
                    wdd = wd[i, m] * wd[j, m]
                    acc[ch, 5, m, b] += wdd * d[i, m] * d[j, m]
                    acc[ch, 6, m, b] += wdd
    return acc.sum(axis=0)


@njit(cache=True)
def transport_factors(vec_members, vec_cells, cell_of):
    """e^{2i delta} that parallel-transports a spin-2 value from each member
    pixel to its cell position (frame of the member -> frame at the cell)."""
    n = vec_members.shape[0]
    fr = np.empty(n)
    fi = np.empty(n)
    for i in range(n):
        c = cell_of[i]
        c1, s1, c2, s2 = pair_rot(
            vec_members[i, 0], vec_members[i, 1], vec_members[i, 2],
            vec_cells[c, 0], vec_cells[c, 1], vec_cells[c, 2],
        )
        # gamma * e1 * conj(e2)
        fr[i] = c1 * c2 + s1 * s2
        fi[i] = s1 * c2 - c1 * s2
    return fr, fi


# --------------------------------------------------------------------------
# degrade
# --------------------------------------------------------------------------
def degrade(pix_nest, vec, fields, nside_c, variant):
    """Per-patch degrade of fine pixels to coarse cells.

    fields = dict(g1, g2, ws, d, wd) with arrays (npts, nmaps).
    """
    shift = 2 * (int(np.log2(NSIDE)) - int(np.log2(nside_c)))
    cells, cell_of = np.unique(pix_nest >> shift, return_inverse=True)
    nc = cells.size
    if variant == "centre":
        cvec = np.array(hp.pix2vec(nside_c, cells, nest=True)).T
    else:
        cvec = np.zeros((nc, 3))
        np.add.at(cvec, cell_of, vec)
        cvec /= np.linalg.norm(cvec, axis=1)[:, None]

    g1, g2, ws, d, wd = (fields[k] for k in ("g1", "g2", "ws", "d", "wd"))
    if variant == "rot":
        fr, fi = transport_factors(vec, cvec, cell_of)
        g1r = g1 * fr[:, None] - g2 * fi[:, None]
        g2r = g1 * fi[:, None] + g2 * fr[:, None]
        g1, g2 = g1r, g2r

    def wsum(w, val=None):
        out = np.zeros((nc, w.shape[1]))
        np.add.at(out, cell_of, w if val is None else w * val)
        return out

    Ws = wsum(ws)
    Wd = wsum(wd)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {
            "g1": np.where(Ws > 0, wsum(ws, g1) / Ws, 0.0),
            "g2": np.where(Ws > 0, wsum(ws, g2) / Ws, 0.0),
            "d": np.where(Wd > 0, wsum(wd, d) / Wd, 0.0),
            "ws": Ws,
            "wd": Wd,
        }
    return cvec, out


# --------------------------------------------------------------------------
# fields
# --------------------------------------------------------------------------
def theory_cls(lmax):
    import pyccl as ccl

    cosmo = ccl.Cosmology(
        Omega_c=0.26, Omega_b=0.049, h=0.68, sigma8=0.81, n_s=0.965,
        matter_power_spectrum="halofit",
    )
    z = np.linspace(0.0, 3.0, 600)
    nz_s = z**2 * np.exp(-((z / 0.5) ** 1.5))  # <z> ~ 0.75
    nz_l = np.exp(-0.5 * ((z - 0.4) / 0.06) ** 2)
    src = ccl.WeakLensingTracer(cosmo, dndz=(z, nz_s))
    lens = ccl.NumberCountsTracer(
        cosmo, has_rsd=False, dndz=(z, nz_l), bias=(z, 1.6 * np.ones_like(z))
    )
    ell = np.unique(np.geomspace(2, lmax, 400).astype(int))
    full = np.arange(lmax + 1)

    def interp(t1, t2):
        cl = ccl.angular_cl(cosmo, t1, t2, ell)
        out = np.zeros(lmax + 1)
        out[2:] = np.exp(np.interp(np.log(full[2:]), np.log(ell), np.log(np.abs(cl))))
        return out

    return cosmo, interp(lens, lens), interp(src, src), interp(lens, src)


def theory_xi(cosmo, cl_dd, cl_kk, cl_dk, binedges_arcmin):
    """Bin-averaged theory (area-weighted) for the pixel-windowed spectra."""
    import pyccl as ccl

    lmax = cl_kk.size - 1
    ell = np.arange(lmax + 1)
    pw = hp.pixwin(NSIDE, lmax=lmax) ** 2
    out = {}
    for name, cl, typ in (
        ("xip", cl_kk, "GG+"), ("xim", cl_kk, "GG-"), ("xit", cl_dk, "NG"), ("xig", cl_dd, "NN"),
    ):
        vals = []
        for lo, hi in zip(binedges_arcmin[:-1], binedges_arcmin[1:]):
            th = np.linspace(lo, hi, 9)
            xi = ccl.correlation(
                cosmo, ell=ell[2:], C_ell=(cl * pw)[2:], theta=th / 60.0, type=typ,
                method="fftlog",
            )
            vals.append(np.trapezoid(xi * th, th) / np.trapezoid(th, th))
        out[name] = np.array(vals)
    return out


def make_realisation(cl_dd, cl_kk, cl_dk, lmax, rng):
    np.random.seed(int(rng.integers(2**31)))
    alm_d, alm_k = hp.synalm([cl_dd, cl_kk, cl_dk], lmax=lmax, new=True)
    pw = hp.pixwin(NSIDE, lmax=lmax)
    ell = np.arange(lmax + 1, dtype=float)
    fl = np.zeros(lmax + 1)
    fl[2:] = -np.sqrt((ell[2:] + 2) * (ell[2:] - 1) / (ell[2:] * (ell[2:] + 1)))
    E = hp.almxfl(alm_k, fl * pw)
    g1, g2 = hp.alm2map_spin([E, np.zeros_like(E)], NSIDE, 2, lmax)
    delta = hp.alm2map(hp.almxfl(alm_d, pw), NSIDE, lmax=lmax)
    return delta, g1, g2


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nreal", type=int, default=6)
    ap.add_argument("--npatch", type=int, default=48)
    ap.add_argument("--theta-q", type=float, default=110.0)
    ap.add_argument("--theta-min", type=float, default=5.0)
    ap.add_argument("--theta-max", type=float, default=250.0)
    ap.add_argument("--nbins", type=int, default=11)
    ap.add_argument("--flip-g1", type=int, default=0)
    ap.add_argument("--flip-g2", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--out", default="results/stage1.npz")
    args = ap.parse_args()

    from CosmoFuse.utils import select_patch_centers

    rng = np.random.default_rng(args.seed)
    lmax = 3 * NSIDE - 1
    mask = hp.read_map(MASK).astype(bool)
    edges_arcmin = np.geomspace(args.theta_min, args.theta_max, args.nbins + 1)
    cos_edges = np.cos(np.radians(edges_arcmin / 60.0))

    # ---- patches ---------------------------------------------------------
    t0 = time.perf_counter()
    phi_c, theta_c = select_patch_centers(
        mask, 32, patch_size=args.theta_q, theta_Q=args.theta_q, f_mask=0.5
    )
    print(f"{phi_c.size} candidate patches ({time.perf_counter()-t0:.0f}s)")
    R = np.radians(args.theta_q / 60.0)
    fmask = np.empty(phi_c.size)
    for p in range(phi_c.size):
        disc = hp.query_disc(NSIDE, hp.ang2vec(theta_c[p], phi_c[p]), R)
        fmask[p] = 1.0 - mask[disc].mean()
    dec_c = 90.0 - np.degrees(theta_c)
    special = {
        "lowdec_clean": int(np.argmin(np.abs(dec_c) + 100 * (fmask > 0.02))),
        "highdec_clean": int(np.argmax(np.abs(dec_c) - 100 * (fmask > 0.02))),
        "most_masked": int(np.argmax(fmask)),
        "masked_25pct": int(np.argmin(np.abs(fmask - 0.25))),
    }
    chosen = list(special.values())
    rest = [p for p in rng.permutation(phi_c.size) if p not in chosen]
    chosen += rest[: max(0, args.npatch - len(chosen))]
    chosen = np.array(chosen)
    print("special patches:", {k: (v, round(dec_c[v], 1), round(fmask[v], 3)) for k, v in special.items()})

    patch_pix = []
    for p in chosen:
        disc = hp.query_disc(NSIDE, hp.ang2vec(theta_c[p], phi_c[p]), R)
        patch_pix.append(disc[mask[disc]])
    union = np.unique(np.concatenate(patch_pix))
    lookup = {p: np.searchsorted(union, pp) for p, pp in zip(chosen, patch_pix)}

    # ---- maps on the union of patch pixels --------------------------------
    cosmo, cl_dd, cl_kk, cl_dk = theory_cls(lmax)
    theory = theory_xi(cosmo, cl_dd, cl_kk, cl_dk, edges_arcmin)
    A_pix = hp.nside2pixarea(NSIDE, degrees=True) * 3600.0
    nbar_s, nbar_l, sigma_e = 1.5, 0.15, 0.26
    variants = ("A", "B", "C")
    nmaps = args.nreal * len(variants)
    F = {k: np.zeros((union.size, nmaps)) for k in ("g1", "g2", "ws", "d", "wd")}
    for r in range(args.nreal):
        t0 = time.perf_counter()
        delta, g1, g2 = make_realisation(cl_dd, cl_kk, cl_dk, lmax, rng)
        delta, g1, g2 = delta[union], g1[union], g2[union]
        if args.flip_g1:
            g1 = -g1
        if args.flip_g2:
            g2 = -g2
        n_s = rng.poisson(nbar_s * A_pix, union.size).astype(float)
        n_l = rng.poisson(nbar_l * A_pix * np.clip(1.0 + delta, 0.0, None)).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            sig = np.where(n_s > 0, sigma_e / np.sqrt(n_s), 0.0)
        for v, name in enumerate(variants):
            m = r * len(variants) + v
            if name == "A":
                F["g1"][:, m], F["g2"][:, m], F["ws"][:, m] = g1, g2, 1.0
                F["d"][:, m], F["wd"][:, m] = delta, 1.0
            elif name == "B":
                F["g1"][:, m], F["g2"][:, m], F["ws"][:, m] = g1, g2, n_s
                F["d"][:, m], F["wd"][:, m] = delta, 1.0
            else:
                F["g1"][:, m] = g1 + sig * rng.standard_normal(union.size)
                F["g2"][:, m] = g2 + sig * rng.standard_normal(union.size)
                F["ws"][:, m] = n_s
                F["d"][:, m] = n_l / (nbar_l * A_pix) - 1.0
                F["wd"][:, m] = 1.0
        print(f"realisation {r}: {time.perf_counter()-t0:.0f}s")
        del delta, g1, g2

    # ---- measure -----------------------------------------------------------
    geo_variants = ("centre", "centroid", "rot")
    nlev = len(LEVELS)
    fine = np.zeros((chosen.size, NSTAT, nmaps, args.nbins))
    coarse = np.zeros((len(geo_variants), nlev, chosen.size, NSTAT, nmaps, args.nbins))
    ncells = np.zeros((nlev, chosen.size), dtype=np.int64)
    for ip, p in enumerate(chosen):
        t0 = time.perf_counter()
        pix = patch_pix[ip]
        vec = np.array(hp.pix2vec(NSIDE, pix)).T
        nest = hp.ring2nest(NSIDE, pix)
        f = {k: np.ascontiguousarray(a[lookup[p]]) for k, a in F.items()}
        fine[ip] = accumulate(vec, f["g1"], f["g2"], f["ws"], f["d"], f["wd"], cos_edges)
        for il, nside_c in enumerate(LEVELS):
            for ig, gv in enumerate(geo_variants):
                cvec, cf = degrade(nest, vec, f, nside_c, gv)
                coarse[ig, il, ip] = accumulate(
                    np.ascontiguousarray(cvec), cf["g1"], cf["g2"], cf["ws"], cf["d"], cf["wd"], cos_edges
                )
            ncells[il, ip] = cvec.shape[0]
        print(f"patch {ip+1}/{chosen.size} (N={pix.size}) {time.perf_counter()-t0:.1f}s", flush=True)

    np.savez(
        args.out,
        fine=fine, coarse=coarse, ncells=ncells, chosen=chosen, dec=dec_c[chosen],
        fmask=fmask[chosen], edges_arcmin=edges_arcmin, levels=np.array(LEVELS),
        geo_variants=np.array(geo_variants), variants=np.array(variants),
        nreal=args.nreal, special=np.array(list(special.keys())),
        **{f"theory_{k}": v for k, v in theory.items()},
    )
    print("saved", args.out)


if __name__ == "__main__":
    main()
