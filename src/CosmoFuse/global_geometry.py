"""
Tiles and the footprint-wide pair search of the global estimators.

The patch estimator (:class:`~CosmoFuse.correlations.PatchCorrelation`) finds
the pairs *inside* every patch disc; a pair enters once per patch that holds
both of its ends.  The global estimators instead need every pair of the
footprint exactly once (:class:`~CosmoFuse.global_2pcf.Global2PCF`) or every
ordered pair exactly once (:class:`~CosmoFuse.global_3pcf.Global3PCF`).  They
are found **per tile**:

* a tile is a HEALPix pixel at ``tile_nside`` (NESTED id); its *core* is the
  set of footprint elements (pixels, or treecode cells) inside it;
* the search runs from every core element to every element within
  ``theta_max`` of it -- the core plus a halo gathered from the tiles within
  reach (:meth:`TileSet.reach`);
* ownership: an **unordered** pair ``{a, b}`` belongs to the tile of its
  lower global id (``a < b``), an **ordered** pair ``(i -> j)`` to the tile of
  its centre ``i``.  Either rule assigns every pair to exactly one tile, so
  tile sums add up to the footprint sum and a tile is also a natural
  jackknife region.

Tiles are HEALPix-aligned, so as long as ``tile_nside`` does not exceed the
coarsest treecode level every coarse cell lies inside one tile, and the
per-tile cells built by :func:`~CosmoFuse.treecode.build_patch_cells` are the
global cells.

Angles follow the patch pair finder exactly: ``phi`` is the position angle of
the other point measured at a point from local east towards local north (the
frame of the shear components).  The unordered search returns the rotation
factors ``e^{2i phi}`` of both ends with the same arithmetic as
:func:`CosmoFuse.correlations._compute_pairs_impl`; the ordered search
returns the *full* angles (needed for odd multipoles) at both ends.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import healpy as hp
import numpy as np
from numba import njit, prange

from .treecode import is_power_of_two


# ── tiles ─────────────────────────────────────────────────────────────────


def default_tile_nside(nside: int, theta_max: float) -> int:
    """Smallest tiles whose side is still >= ``theta_max`` (radians).

    The halo of a tile then spans about one tile in every direction, which
    keeps the candidate set small without making the tiles tiny.  Clipped to
    ``[1, nside]``.
    """
    tile = 1
    while tile * 2 <= nside and hp.nside2resol(tile * 2) >= theta_max:
        tile *= 2
    return int(tile)


@dataclass
class TileSet:
    """The tiles covering a row space.

    ``tile_ids``: NESTED ids (at ``tile_nside``) of the non-empty tiles,
    ascending.  ``row_indptr`` / ``rows``: CSR over tiles of the row indices
    (ascending inside a tile).
    """

    nside: int
    tile_nside: int
    tile_ids: np.ndarray
    row_indptr: np.ndarray
    rows: np.ndarray
    centre_vec: np.ndarray

    @classmethod
    def from_rows(cls, nside: int, row_pix: np.ndarray, tile_nside: int) -> "TileSet":
        nside = int(nside)
        tile_nside = int(tile_nside)
        if not (is_power_of_two(nside) and is_power_of_two(tile_nside)):
            raise ValueError(
                "tiles need power-of-two nside and tile_nside; got "
                f"nside={nside}, tile_nside={tile_nside}"
            )
        if tile_nside > nside:
            raise ValueError(f"tile_nside={tile_nside} exceeds nside={nside}")
        shift = 2 * (int(np.log2(nside)) - int(np.log2(tile_nside)))
        nest = hp.ring2nest(nside, np.asarray(row_pix, dtype=np.int64))
        tile_of_row = nest >> shift
        tile_ids, inv = np.unique(tile_of_row, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        indptr = np.zeros(tile_ids.size + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(np.bincount(inv, minlength=tile_ids.size))
        vec = np.asarray(hp.pix2vec(tile_nside, tile_ids, nest=True), dtype=np.float64).T
        return cls(nside, tile_nside, tile_ids.astype(np.int64), indptr, order.astype(np.int64), vec)

    @property
    def n_tiles(self) -> int:
        return int(self.tile_ids.size)

    @property
    def centre_ang(self) -> Tuple[np.ndarray, np.ndarray]:
        """(phi, theta) of the tile centres (HEALPix convention)."""
        theta, phi = hp.vec2ang(self.centre_vec)
        return np.asarray(phi), np.asarray(theta)

    def tile_of_nest(self, nest_ids: np.ndarray, nside_ids: int) -> np.ndarray:
        """Tile index of NESTED ids given at ``nside_ids`` (>= tile_nside);
        -1 for a tile that holds no row."""
        shift = 2 * (int(np.log2(nside_ids)) - int(np.log2(self.tile_nside)))
        parent = np.asarray(nest_ids, dtype=np.int64) >> shift
        idx = np.searchsorted(self.tile_ids, parent)
        idx = np.minimum(idx, self.n_tiles - 1)
        return np.where(self.tile_ids[idx] == parent, idx, -1)

    def reach(self, t: int, radius: float) -> np.ndarray:
        """Indices of the tiles that may hold a point within ``radius``
        (radians) of any point of tile ``t`` (tile ``t`` included)."""
        r = hp.max_pixrad(self.tile_nside) + float(radius)
        if r >= np.pi:
            return np.arange(self.n_tiles)
        # max_pixrad bounds the centre-to-corner distance of every tile, so a
        # disc of max_pixrad + radius around tile t's centre, queried
        # inclusively, contains every tile with a point within ``radius`` of
        # tile t.  The 1e-9 guards the boundary of the inclusive test.
        cand = hp.query_disc(
            self.tile_nside, self.centre_vec[t], r + 1e-9, inclusive=True, nest=True
        )
        idx = np.searchsorted(self.tile_ids, cand)
        idx = np.minimum(idx, self.n_tiles - 1)
        keep = self.tile_ids[idx] == cand
        return np.unique(idx[keep])


@dataclass
class LevelElements:
    """The elements (pixels or cells) of one resolution level, sorted by tile.

    ``ids``: global ids (RING pixel id, or ``npix + virtual row`` for a cell).
    ``xyz``: (3, n) unit vectors at the pair-search precision.
    ``indptr``: CSR over tiles.
    """

    ids: np.ndarray
    xyz: np.ndarray
    indptr: np.ndarray

    def of_tiles(self, tiles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        parts = [np.arange(self.indptr[t], self.indptr[t + 1]) for t in tiles]
        sel = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
        return self.ids[sel], np.ascontiguousarray(self.xyz[:, sel])

    def core(self, t: int) -> Tuple[np.ndarray, np.ndarray]:
        a, b = int(self.indptr[t]), int(self.indptr[t + 1])
        return self.ids[a:b], np.ascontiguousarray(self.xyz[:, a:b])


def radec_to_xyz(ra: np.ndarray, dec: np.ndarray, dtype: Any) -> np.ndarray:
    """Unit vectors with the arithmetic of the patch pair finder."""
    ra = np.asarray(ra, dtype=dtype)
    dec = np.asarray(dec, dtype=dtype)
    cos_dec = np.cos(dec)
    return np.stack((cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)))


# ── pair kernels ──────────────────────────────────────────────────────────


def _bin_of(cos_theta: float, cos_edges: np.ndarray) -> int:
    """Angular bin of a separation given as its cosine; -1 outside.

    Same acceptance rule as the patch pair finder: bin ``b`` holds
    ``cos_edges[b] > cos_theta > cos_edges[b + 1]``.
    """
    nedges = cos_edges.size
    if cos_theta >= cos_edges[0] or cos_theta <= cos_edges[nedges - 1]:
        return -1
    lo = 0
    hi = nedges
    while lo < hi:
        mid = (lo + hi) >> 1
        if cos_edges[mid] > cos_theta:
            lo = mid + 1
        else:
            hi = mid
    b = lo - 1
    if b >= 0 and b < nedges - 1 and cos_theta > cos_edges[b + 1]:
        return b
    return -1


def _angle_terms(
    x1: float, y1: float, z1: float, x2: float, y2: float, z2: float
) -> Tuple[float, float, float, float]:
    """Unnormalised (sin C, cos C) at point 1 and at point 2, where C is the
    bearing (north through east) of the other point; the position angle from
    east towards north is ``pi/2 - C``.  Arithmetic of the patch finder."""
    sinC1 = x1 * y2 - x2 * y1
    dsq_AC1 = x1 * x1 + y1 * y1 + (z1 - 1.0) * (z1 - 1.0)
    dx12 = x1 - x2
    dy12 = y1 - y2
    dz12 = z1 - z2
    dsq_BC1 = dx12 * dx12 + dy12 * dy12 + dz12 * dz12
    dsq_AB1 = x2 * x2 + y2 * y2 + (z2 - 1.0) * (z2 - 1.0)
    cosC1 = 0.5 * (dsq_AC1 + dsq_BC1 - dsq_AB1 - 0.5 * dsq_AC1 * dsq_BC1)
    sinC2 = x2 * y1 - x1 * y2
    cosC2 = 0.5 * (dsq_AB1 + dsq_BC1 - dsq_AC1 - 0.5 * dsq_AB1 * dsq_BC1)
    return sinC1, cosC1, sinC2, cosC2


def _exp2(sinC: float, cosC: float) -> Tuple[float, float]:
    """e^{2i phi} from the unnormalised bearing terms (patch-finder formula)."""
    R2 = sinC * sinC + cosC * cosC
    if R2 > 0:
        inv = 1.0 / R2
        return (sinC * sinC - cosC * cosC) * inv, (2.0 * sinC * cosC) * inv
    return -1.0, 0.0


_bin_of_jit = njit(cache=True, inline="always")(_bin_of)
_angle_terms_jit = njit(cache=True, inline="always")(_angle_terms)
_exp2_jit = njit(cache=True, inline="always")(_exp2)


def _unique_pairs_impl(
    core_ids: np.ndarray,
    core_xyz: np.ndarray,
    cand_ids: np.ndarray,
    cand_xyz: np.ndarray,
    cos_edges: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    """Pairs ``(core i, candidate j)`` with ``core_ids[i] < cand_ids[j]``,
    grouped by angular bin (rows ascending inside a bin).

    Returns ``(ids_a, ids_b, c1, s1, c2, s2, bin_counts)``: global ids and the
    rotation factors ``e^{2i phi}`` at both ends.
    """
    ncore = core_ids.size
    ncand = cand_ids.size
    nbins = cos_edges.size - 1
    counts = np.zeros((ncore, nbins), dtype=np.int64)
    for i in prange(ncore):
        x1 = core_xyz[0, i]
        y1 = core_xyz[1, i]
        z1 = core_xyz[2, i]
        id1 = core_ids[i]
        for j in range(ncand):
            if cand_ids[j] <= id1:
                continue
            cos_theta = x1 * cand_xyz[0, j] + y1 * cand_xyz[1, j] + z1 * cand_xyz[2, j]
            b = _bin_of_jit(cos_theta, cos_edges)
            if b >= 0:
                counts[i, b] += 1

    bin_counts = np.zeros(nbins, dtype=np.int64)
    cursor = np.zeros((ncore, nbins), dtype=np.int64)
    running = 0
    for b in range(nbins):
        start = running
        for i in range(ncore):
            cursor[i, b] = running
            running += counts[i, b]
        bin_counts[b] = running - start
    total = running

    ids_a = np.empty(total, dtype=core_ids.dtype)
    ids_b = np.empty(total, dtype=core_ids.dtype)
    c1 = np.empty(total, dtype=core_xyz.dtype)
    s1 = np.empty(total, dtype=core_xyz.dtype)
    c2 = np.empty(total, dtype=core_xyz.dtype)
    s2 = np.empty(total, dtype=core_xyz.dtype)
    for i in prange(ncore):
        x1 = core_xyz[0, i]
        y1 = core_xyz[1, i]
        z1 = core_xyz[2, i]
        id1 = core_ids[i]
        for j in range(ncand):
            if cand_ids[j] <= id1:
                continue
            x2 = cand_xyz[0, j]
            y2 = cand_xyz[1, j]
            z2 = cand_xyz[2, j]
            cos_theta = x1 * x2 + y1 * y2 + z1 * z2
            b = _bin_of_jit(cos_theta, cos_edges)
            if b < 0:
                continue
            out = cursor[i, b]
            cursor[i, b] = out + 1
            sinC1, cosC1, sinC2, cosC2 = _angle_terms_jit(x1, y1, z1, x2, y2, z2)
            a_re, a_im = _exp2_jit(sinC1, cosC1)
            b_re, b_im = _exp2_jit(sinC2, cosC2)
            ids_a[out] = id1
            ids_b[out] = cand_ids[j]
            c1[out] = a_re
            s1[out] = a_im
            c2[out] = b_re
            s2[out] = b_im
    return ids_a, ids_b, c1, s1, c2, s2, bin_counts


def _ordered_pairs_impl(
    core_ids: np.ndarray,
    core_xyz: np.ndarray,
    cand_ids: np.ndarray,
    cand_xyz: np.ndarray,
    cos_edges: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    """Ordered pairs ``(centre i -> neighbour j)``, ``j != i``, sorted by
    ``(centre, bin)`` (candidates ascending inside).

    Returns ``(nbr, phi_c, phi_n, counts)``: candidate *positions* (into
    ``cand_ids``), the full position angle of the neighbour seen from the
    centre, the full position angle of the centre seen from the neighbour,
    and ``counts[i, b]``.
    """
    ncore = core_ids.size
    ncand = cand_ids.size
    nbins = cos_edges.size - 1
    counts = np.zeros((ncore, nbins), dtype=np.int64)
    for i in prange(ncore):
        x1 = core_xyz[0, i]
        y1 = core_xyz[1, i]
        z1 = core_xyz[2, i]
        id1 = core_ids[i]
        for j in range(ncand):
            if cand_ids[j] == id1:
                continue
            cos_theta = x1 * cand_xyz[0, j] + y1 * cand_xyz[1, j] + z1 * cand_xyz[2, j]
            b = _bin_of_jit(cos_theta, cos_edges)
            if b >= 0:
                counts[i, b] += 1

    cursor = np.zeros((ncore, nbins), dtype=np.int64)
    running = 0
    for i in range(ncore):
        for b in range(nbins):
            cursor[i, b] = running
            running += counts[i, b]
    total = running

    nbr = np.empty(total, dtype=np.int64)
    phi_c = np.empty(total, dtype=core_xyz.dtype)
    phi_n = np.empty(total, dtype=core_xyz.dtype)
    for i in prange(ncore):
        x1 = core_xyz[0, i]
        y1 = core_xyz[1, i]
        z1 = core_xyz[2, i]
        id1 = core_ids[i]
        for j in range(ncand):
            if cand_ids[j] == id1:
                continue
            x2 = cand_xyz[0, j]
            y2 = cand_xyz[1, j]
            z2 = cand_xyz[2, j]
            cos_theta = x1 * x2 + y1 * y2 + z1 * z2
            b = _bin_of_jit(cos_theta, cos_edges)
            if b < 0:
                continue
            out = cursor[i, b]
            cursor[i, b] = out + 1
            sinC1, cosC1, sinC2, cosC2 = _angle_terms_jit(x1, y1, z1, x2, y2, z2)
            # phi = pi/2 - C  =>  e^{i phi} = (sin C + i cos C) / |.|
            nbr[out] = j
            phi_c[out] = np.arctan2(cosC1, sinC1)
            phi_n[out] = np.arctan2(cosC2, sinC2)
    return nbr, phi_c, phi_n, counts


_kernel_cache: Dict[Tuple[str, bool], Any] = {}


def _kernel(name: str, fastmath: bool) -> Any:
    key = (name, bool(fastmath))
    k = _kernel_cache.get(key)
    if k is None:
        impl = {"unique": _unique_pairs_impl, "ordered": _ordered_pairs_impl}[name]
        k = njit(fastmath=bool(fastmath), parallel=True, cache=True)(impl)
        _kernel_cache[key] = k
    return k


def unique_pairs(
    core_ids: np.ndarray,
    core_xyz: np.ndarray,
    cand_ids: np.ndarray,
    cand_xyz: np.ndarray,
    binedges: np.ndarray,
    fastmath: bool = True,
) -> Tuple[np.ndarray, ...]:
    cos_edges = np.cos(np.asarray(binedges, dtype=core_xyz.dtype))
    return _kernel("unique", fastmath)(core_ids, core_xyz, cand_ids, cand_xyz, cos_edges)


def ordered_pairs(
    core_ids: np.ndarray,
    core_xyz: np.ndarray,
    cand_ids: np.ndarray,
    cand_xyz: np.ndarray,
    binedges: np.ndarray,
    fastmath: bool = True,
) -> Tuple[np.ndarray, ...]:
    cos_edges = np.cos(np.asarray(binedges, dtype=core_xyz.dtype))
    return _kernel("ordered", fastmath)(core_ids, core_xyz, cand_ids, cand_xyz, cos_edges)
