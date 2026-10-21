"""Probe variants of step 2 of Global3PCF (the contraction over centres).

Times ``ggg()`` on the real geometry (DES Y3) with ``Global3PCF._accumulate``
replaced by variants, and compares every result with the shipped one:

* M  one batched matmul per group of jobs that share their leg-3 factor
     (gamma0..2 all use G_n of leg 3): the A operands are written side by
     side into one buffer, ``L3^T @ [A_0 | A_1 | A_2]``;
* D  gamma3 from gamma2 by the leg swap, Gamma3_n(z2,b2; z3,b3) =
     Gamma2_{sigma1-n}(z3,b3; z2,b2), with gamma2 computed for the multipoles
     that needs (n up to n_max + 2);
* MD both.

    python step2_probe.py --nside 512 --mask DESY3_Mask.fits
"""

import argparse
import time
import types
import warnings

import cupy as cp
import healpy as hp
import numpy as np

from CosmoFuse import Global3PCF


def accumulate(self, fields, centre_values, jobs, merge=False, derive=False):
    """Copy of Global3PCF._accumulate with the M and D options."""
    be = self.backend
    xp = be.module
    dev = self._device()
    kernel = be.kernel_3pcf_multipoles
    offsets_f, off = {}, 0
    for f in fields:
        offsets_f[f.name] = (off, off + f.nz)
        off += f.nz
    n_f = off
    vre = xp.ascontiguousarray(xp.concatenate([f.vre for f in fields]))
    vim = xp.ascontiguousarray(xp.concatenate([f.vim for f in fields]))
    wts = xp.ascontiguousarray(xp.concatenate([f.w for f in fields]))
    spin = be.to_device(
        np.concatenate([np.full(f.nz, f.spin, np.int32) for f in fields])
    )
    products = self._pair_products(jobs)
    desc_h, n_g = self._descriptor(n_f, products)
    desc = tuple(be.to_device(d) for d in desc_h)
    n_out = int(desc_h[0].size)
    E, P, nb = self.E, 2 * self.E + 1, self.nbins
    prod_index = {p: i for i, p in enumerate(products)}
    cdt = getattr(xp, self.acc_complex.name)
    N = self.n_max

    # D: jobs whose leg-swapped twin is also requested are derived from it
    derived = {}  # label -> (source label, sigma1)
    if derive:
        by_key = {(j[1], j[2], j[3], j[4], j[5]): j for j in jobs}
        for j in jobs:
            label, ck, l2, l3, c2, c3, s1 = j
            twin = by_key.get((ck, l3, l2, c3, c2))
            if (
                twin is not None
                and twin[0] != label
                and twin[0] not in derived
                and label < twin[0]
            ):
                derived[twin[0]] = (label, s1)
    # multipole range per computed job
    n_range = {}
    for j in jobs:
        if j[0] in derived:
            continue
        lo, hi = -N, N
        for d, (src, s1) in derived.items():
            if src == j[0]:
                lo, hi = min(lo, s1 - N), max(hi, s1 + N)
        n_range[j[0]] = np.arange(lo, hi + 1)
    compute = [j for j in jobs if j[0] not in derived]

    acc, deg = {}, {}
    n_centres = int(self.centre_rows.size)
    idx_cache = {}

    def _idx(a):
        key = tuple(int(v) for v in a)
        if key not in idx_cache:
            idx_cache[key] = be.to_device(np.asarray(a, dtype=np.int64))
        return idx_cache[key]

    for c0 in range(0, n_centres, self.centre_chunk):
        C = min(self.centre_chunk, n_centres - c0)
        out = xp.zeros((C, nb, n_out), dtype=cdt)
        kernel(
            dev["offsets"],
            c0,
            C,
            nb,
            dev["nbr"],
            dev["phi_c"],
            dev["phi_n"],
            vre,
            vim,
            wts,
            spin,
            desc,
            E,
            out,
        )
        G = xp.ascontiguousarray(
            out[:, :, :n_g].reshape(C, nb, n_f, P).transpose(3, 0, 1, 2)
        )
        S = out[:, :, n_g:].reshape(C, nb, len(products), n_f, n_f)
        rows = dev["centre_rows"][c0 : c0 + C]
        legs = {}

        def _leg(m, conj, lo, hi):
            key = (tuple(int(v) for v in m), conj, lo, hi)
            if key not in legs:
                L = G[_idx(m + E)][:, :, :, lo:hi]
                if conj:
                    L = xp.conj(L)
                legs[key] = xp.ascontiguousarray(L).reshape(m.size, C, nb * (hi - lo))
            return legs[key]

        def operands(job):
            label, ckey, leg2, leg3, c2, c3, sigma1 = job
            ns = n_range[label]
            vals = centre_values[ckey][0][:, rows].T.astype(cdt)
            b0, b1 = offsets_f[leg2]
            d0, d1 = offsets_f[leg3]
            L2 = (
                _leg(ns - sigma1, True, b0, b1)
                if c2
                else _leg(sigma1 - ns, False, b0, b1)
            )
            L3 = _leg(-ns, True, d0, d1) if c3 else _leg(ns, False, d0, d1)
            if not c2 and not c3:
                Sv = S[:, :, prod_index[(sigma1, 0)]]
            elif c2 and not c3:
                Sv = S[:, :, prod_index[(sigma1, 1)]]
            elif not c2 and c3:
                Sv = xp.conj(S[:, :, prod_index[(-sigma1, 1)]])
            else:
                Sv = xp.conj(S[:, :, prod_index[(-sigma1, 0)]])
            corr = xp.tensordot(vals, Sv[:, :, b0:b1, d0:d1], axes=([0], [0]))
            return vals, L2, L3, corr

        def add(label, R, corr):
            if label in acc:
                acc[label] += R
                deg[label] += corr
            else:
                acc[label] = R
                deg[label] = corr

        if not merge:
            for job in compute:
                vals, L2, L3, corr = operands(job)
                nA = int(vals.shape[1])
                A = (L2[:, :, None, :] * vals[None, :, :, None]).reshape(
                    L2.shape[0], C, -1
                )
                add(job[0], xp.matmul(L3.transpose(0, 2, 1), A), corr)
        else:
            groups = {}
            for job in compute:
                key = (job[3], job[5], tuple(n_range[job[0]]))
                groups.setdefault(key, []).append(job)
            for gjobs in groups.values():
                ops = [operands(j) for j in gjobs]
                L3 = ops[0][2]
                nN = L3.shape[0]
                widths = [o[1].shape[2] * o[0].shape[1] for o in ops]
                Abuf = xp.empty((nN, C, sum(widths)), dtype=cdt)
                o0 = 0
                for (vals, L2, _L3, _c), w in zip(ops, widths):
                    xp.multiply(
                        L2[:, :, None, :],
                        vals[None, :, :, None],
                        out=Abuf[:, :, o0 : o0 + w].reshape(
                            nN, C, vals.shape[1], L2.shape[2]
                        ),
                    )
                    o0 += w
                R = xp.matmul(L3.transpose(0, 2, 1), Abuf)
                o0 = 0
                for job, (vals, L2, _L3, corr), w in zip(gjobs, ops, widths):
                    add(job[0], R[:, :, o0 : o0 + w], corr)
                    o0 += w

    results = {}
    for label, ckey, leg2, leg3, _c2, _c3, _s in compute:
        ns = n_range[label]
        nA = int(centre_values[ckey][0].shape[0])
        nB = offsets_f[leg2][1] - offsets_f[leg2][0]
        nC = offsets_f[leg3][1] - offsets_f[leg3][0]
        R6 = acc[label].reshape(ns.size, nb, nC, nA, nb, nB)
        corr = deg[label]
        for b in range(nb):
            R6[:, b, :, :, b, :] -= corr[:, b].transpose(2, 0, 1)[None]
        out6 = xp.ascontiguousarray(
            R6.transpose(0, 3, 5, 2, 4, 1)
        )  # n, a, z2, z3, b2, b3
        results[label] = (ns, out6)
    final = {}
    for label, (ns, out6) in results.items():
        sel = np.searchsorted(ns, np.arange(-N, N + 1))
        final[label] = np.asarray(be.to_numpy(out6[_idx(sel)])).astype(np.complex128)
    for label, (src, s1) in derived.items():
        ns, out6 = results[src]
        sel = np.searchsorted(ns, s1 - np.arange(-N, N + 1))
        # Gamma'_n(a, z2, z3, b2, b3) = Gamma_{s1-n}(a, z3, z2, b3, b2)
        swapped = out6[_idx(sel)].transpose(0, 1, 3, 2, 5, 4)
        final[label] = np.asarray(be.to_numpy(xp.ascontiguousarray(swapped))).astype(
            np.complex128
        )
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nside", type=int, default=512)
    ap.add_argument("--mask", required=True)
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    mask = hp.ud_grade(hp.read_map(args.mask, dtype=np.float64), args.nside) >= 0.5
    est = Global3PCF(
        args.nside,
        mask=mask,
        nbins=12,
        theta_min=15,
        theta_max=250,
        n_max=20,
        device="gpu",
        map_precision="float32",
        rotation_precision="float32",
        accumulation_precision="float64",
    )
    est.find_pairs()
    est.prepare()
    rng = np.random.default_rng(0)
    n = est.n_active
    g = cp.asarray((rng.normal(size=(4, 2, n)) * 0.3).astype(np.float32))
    w = cp.asarray(rng.uniform(0.5, 1.5, size=(4, n)).astype(np.float32))
    shipped = Global3PCF._accumulate

    def timed(label):
        est.ggg(g, w, flip_g1=True)  # warm-up (norms cached)
        ts = []
        for _ in range(3):
            cp.cuda.Device().synchronize()
            t = time.perf_counter()
            res = est.ggg(g, w, flip_g1=True)
            cp.cuda.Device().synchronize()
            ts.append(time.perf_counter() - t)
        return float(np.median(ts)), res

    t_ref, ref = timed("shipped")
    print(f"shipped: {t_ref:.3f} s per ggg()", flush=True)
    for name, merge, derive in (
        ("M", True, False),
        ("D", False, True),
        ("MD", True, True),
    ):
        est._accumulate = types.MethodType(
            lambda self, f, cv, j, merge=merge, derive=derive: accumulate(
                self, f, cv, j, merge, derive
            ),
            est,
        )
        dt, res = timed(name)
        est._accumulate = types.MethodType(shipped, est)
        rel = max(
            float(np.abs(res.num[k] - ref.num[k]).max() / np.abs(ref.num[k]).max())
            for k in ref.num
        )
        same = all(np.array_equal(res.num[k], ref.num[k]) for k in ref.num)
        print(
            f"{name:2s}: {dt:.3f} s per ggg() ({t_ref / dt:.2f}x), "
            f"{'bitwise' if same else f'max rel diff {rel:.1e}'} vs shipped",
            flush=True,
        )


if __name__ == "__main__":
    main()
