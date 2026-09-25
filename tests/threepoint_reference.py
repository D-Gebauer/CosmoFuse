"""Brute-force three-point reference for the Global3PCF gates.

An explicit loop over every pixel triplet with *independent* geometry: the
position angles come from the local east/north tangent vectors (not from the
pair finder's formulas), so a convention error in either shows up.
"""

import healpy as hp
import numpy as np


def _frame(nside, pix):
    th, ph = hp.pix2ang(nside, pix)
    ra, dec = ph, np.pi / 2 - th
    p = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)
    e = np.stack([-np.sin(ra), np.cos(ra), np.zeros_like(ra)], -1)
    n = np.stack([-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)], -1)
    return p, e, n


def position_angle(p, e, n, q):
    """Angle of ``q`` seen from ``p``, from local east towards north."""
    t = q - (p @ q) * p
    return np.arctan2(t @ n, t @ e)


def brute_force(nside, pix, fields, components, binedges, n_max):
    """Multipoles of every component by direct summation over triplets.

    ``fields``: name -> (values, weights, spin) indexed by position in
    ``pix``; ``components``: (label, (centre, leg2, leg3), (c1, c2, c3)).
    Returns ``(num, norm)`` dicts of ``(2n_max+1, nA, nB, nC, nb, nb)``.
    """
    P, Ee, Nn = _frame(nside, pix)
    cos_edges = np.cos(binedges)
    nb = len(binedges) - 1
    ns = np.arange(-n_max, n_max + 1)

    def value(name, idx):
        v, w, s = fields[name]
        if s == 2:
            return v[:, 0, idx] + 1j * v[:, 1, idx], w[:, idx], 2
        return v[:, idx].astype(complex), w[:, idx], 0

    def proj(name, idx, beta, conj):
        v, w, s = value(name, idx)
        if s == 2:
            v = -v * np.exp(-2j * beta)
        if conj:
            v = np.conj(v)
        return v * w, w

    num, norm = {}, {}
    for label, (fc, f2, f3), (c1, c2, c3) in components:
        nA, nB, nC = (fields[f][1].shape[0] for f in (fc, f2, f3))
        acc = np.zeros((ns.size, nA, nB, nC, nb, nb), complex)
        den = np.zeros_like(acc)
        for i in range(len(pix)):
            cos = P @ P[i]
            b = np.full(len(pix), -1)
            for k in range(nb):
                b[(cos < cos_edges[k]) & (cos > cos_edges[k + 1])] = k
            b[i] = -1
            nbr = np.flatnonzero(b >= 0)
            phi = np.array([position_angle(P[i], Ee[i], Nn[i], P[j]) for j in nbr])
            psi = np.array([position_angle(P[j], Ee[j], Nn[j], P[i]) for j in nbr])
            for a2, j in enumerate(nbr):
                vc, wc = proj(fc, i, phi[a2], c1)
                vj, wj = proj(f2, j, psi[a2], c2)
                for a3, k in enumerate(nbr):
                    if j == k:
                        continue
                    vk, wk = proj(f3, k, psi[a3], c3)
                    ph = np.exp(-1j * ns * (phi[a3] - phi[a2]))[:, None, None, None]
                    acc[:, :, :, :, b[j], b[k]] += ph * np.einsum("a,b,c->abc", vc, vj, vk)[None]
                    den[:, :, :, :, b[j], b[k]] += ph * np.einsum("a,b,c->abc", wc, wj, wk)[None]
        num[label], norm[label] = acc, den
    return num, norm
