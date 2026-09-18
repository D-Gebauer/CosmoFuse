"""
Static treecode: per-angular-bin HEALPix resolution levels.

For every angular bin the pair geometry is built on the coarsest HEALPix
level whose pixel size ``p`` still satisfies ``p <= theta_lo / k`` (``k`` =
``resolution_factor``).  Coarse "cells" are formed *per patch* from the
unmasked base-resolution pixels inside the patch disc, so the patch window
stays an exact top-hat at base resolution.  A cell carries the weighted mean
of its members and the sum of their weights; because

    W_I W_J g_I g_J = sum_{i in I} sum_{j in J} w_i w_j g_i g_j ,

one coarse pair is exactly the sum over its fine child pairs -- the only
change is that the child pairs are binned and rotated with the geometry of
their parent cells (their binary-mask centroids).

Levels are assigned once, at preprocess time ("static"): the measurement
kernels stay branch-free and the estimator is a fixed linear operator on the
maps.

Conventions
-----------
* Host pair / aperture indices use *global ids*: ``id < npix`` is a HEALPix
  RING pixel, ``id >= npix`` is the appended virtual row ``id - npix``.
* Appended rows are numbered ``[aperture block | level 1 | level 2 | ...]``
  with the coarse levels ordered by descending nside; within a level the
  cells of patch ``p`` occupy ``cell_offsets[level, p] : cell_offsets[level,
  p + 1]``, which makes patch-range slicing a pure offset operation.
* NESTED ordering is used internally for the parent relation; everything
  user-facing stays RING.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import healpy as hp
import numpy as np


def is_power_of_two(n: int) -> bool:
    n = int(n)
    return n > 0 and (n & (n - 1)) == 0


# Resolution factor used when the static treecode is switched on without an
# explicit value (``resolution_factor=True`` / ``"default"``): xi- is
# suppressed by <= 5 %, gamma_t by <= 2.5 %, xi+/xi_g by < 1 %.
DEFAULT_RESOLUTION_FACTOR = 4.0


def validate_resolution_factor(resolution_factor: object) -> Optional[float]:
    """Return the resolution factor as a float (``None`` = full resolution).

    ``None`` / ``False`` -> full resolution (the default); ``True`` /
    ``"default"`` -> :data:`DEFAULT_RESOLUTION_FACTOR`; a positive number ->
    that value.
    """
    if resolution_factor is None or resolution_factor is False:
        return None
    if resolution_factor is True or (
        isinstance(resolution_factor, str) and resolution_factor.lower() == "default"
    ):
        return DEFAULT_RESOLUTION_FACTOR
    if not isinstance(resolution_factor, (int, float, np.integer, np.floating)):
        raise ValueError(
            "resolution_factor must be None/False (full resolution), "
            "True/'default' (static treecode with the default factor "
            f"{DEFAULT_RESOLUTION_FACTOR:g}) or a positive number; got "
            f"{resolution_factor!r}"
        )
    k = float(resolution_factor)
    if not (k > 0.0) or np.isnan(k):
        raise ValueError(
            f"resolution_factor must be positive; got {resolution_factor!r}"
        )
    return k


def assign_levels(
    binedges: np.ndarray, nside: int, resolution_factor: Optional[float]
) -> np.ndarray:
    """nside used by every angular bin.

    ``binedges`` in radians.  For bin ``b`` the result is the smallest
    power-of-two nside whose pixel size (``sqrt`` of the pixel area) is
    ``<= binedges[b] / resolution_factor``, clipped to ``[1, nside]``.
    """
    nbins = len(binedges) - 1
    levels = np.full(nbins, int(nside), dtype=np.int64)
    if resolution_factor is None:
        return levels
    if not is_power_of_two(nside):
        raise ValueError(
            "resolution_factor requires a power-of-two nside (NESTED parent "
            f"relation); got nside={nside}"
        )
    for b in range(nbins):
        limit = float(binedges[b]) / float(resolution_factor)
        ns = int(nside)
        while ns > 1 and hp.nside2resol(ns // 2) <= limit:
            ns //= 2
        levels[b] = ns
    return levels


def level_groups(level_nside: np.ndarray) -> List[Tuple[int, int, int]]:
    """Contiguous bin ranges ``(nside, first_bin, stop_bin)`` in bin order."""
    groups: List[Tuple[int, int, int]] = []
    start = 0
    for b in range(1, len(level_nside) + 1):
        if b == len(level_nside) or level_nside[b] != level_nside[start]:
            groups.append((int(level_nside[start]), start, b))
            start = b
    return groups


def effective_resolution_factor(binedges: np.ndarray, level_nside: np.ndarray) -> np.ndarray:
    """theta_lo / pixel size for every bin."""
    resol = np.array([hp.nside2resol(int(ns)) for ns in level_nside])
    return np.asarray(binedges[:-1], dtype=np.float64) / resol


# ── analytic pair-count model (preflight) ─────────────────────────────────


def disc_pair_fraction(theta_lo: float, theta_hi: float, radius: float) -> float:
    """Fraction of point pairs of a uniform disc (flat sky, radius ``radius``)
    whose separation lies in ``[theta_lo, theta_hi]``."""

    def cdf(d: float) -> float:
        t = min(max(d / (2.0 * radius), 0.0), 1.0)
        if t >= 1.0:
            return 1.0
        # P(sep <= d) for two uniform points in a disc
        a = np.arccos(t)
        s = np.sqrt(1.0 - t * t)
        return float(
            1.0
            + (2.0 / np.pi) * ((4.0 * t * t - 1.0) * a)
            - (2.0 / np.pi) * (t * s * (1.0 + 2.0 * t * t))
        )

    return max(cdf(theta_hi) - cdf(theta_lo), 0.0)


def estimate_pairs_per_bin(
    n_pix_patch: np.ndarray,
    binedges: np.ndarray,
    level_nside: np.ndarray,
    base_nside: int,
    radius: float,
) -> np.ndarray:
    """Projected number of pairs per bin, summed over patches.

    ``n_pix_patch``: unmasked base-resolution pixels per patch.
    """
    n_pix_patch = np.asarray(n_pix_patch, dtype=np.float64)
    out = np.zeros(len(level_nside), dtype=np.float64)
    for b, ns in enumerate(level_nside):
        n_cells = n_pix_patch * (float(ns) / float(base_nside)) ** 2
        frac = disc_pair_fraction(float(binedges[b]), float(binedges[b + 1]), radius)
        out[b] = float(np.sum(0.5 * n_cells * n_cells)) * frac
    return out


# ── per-patch cell construction ───────────────────────────────────────────


@dataclass
class PatchCells:
    """Coarse cells of one patch, one entry per coarse level."""

    n_cells: List[int] = field(default_factory=list)
    ra: List[np.ndarray] = field(default_factory=list)
    dec: List[np.ndarray] = field(default_factory=list)
    # CSR children of every cell: level 0 -> HEALPix RING ids of the member
    # pixels; deeper levels -> patch-local cell index in the previous level.
    child_indptr: List[np.ndarray] = field(default_factory=list)
    child_indices: List[np.ndarray] = field(default_factory=list)


def build_patch_cells(
    nside: int, pix_ring: np.ndarray, coarse_nsides: Sequence[int]
) -> PatchCells:
    """Group the unmasked base pixels of one patch into coarse cells.

    ``coarse_nsides`` must be sorted by descending nside.  Cell positions are
    the unit-vector centroids of the member *pixel centres* (binary mask, no
    weights), so the geometry is independent of the maps.
    """
    cells = PatchCells()
    pix_ring = np.asarray(pix_ring, dtype=np.int64)
    if len(coarse_nsides) == 0:
        return cells
    base_order = int(np.log2(nside))
    nest = hp.ring2nest(nside, pix_ring) if pix_ring.size else pix_ring
    vec = (
        np.asarray(hp.pix2vec(nside, pix_ring), dtype=np.float64)
        if pix_ring.size
        else np.zeros((3, 0))
    )

    prev_ids = nest  # NESTED ids of the previous level's cells (level 0: pixels)
    prev_shift = 0
    for level, nside_c in enumerate(coarse_nsides):
        shift = 2 * (base_order - int(np.log2(nside_c)))
        parent_of_fine = nest >> shift
        cell_ids, inv_fine = np.unique(parent_of_fine, return_inverse=True)
        n_cells = int(cell_ids.size)

        cvec = np.stack(
            [np.bincount(inv_fine, weights=vec[k], minlength=n_cells) for k in range(3)]
        )
        norm = np.sqrt(np.sum(cvec * cvec, axis=0))
        cvec = cvec / np.where(norm > 0, norm, 1.0)
        theta, phi = hp.vec2ang(cvec.T) if n_cells else (np.zeros(0), np.zeros(0))
        cells.n_cells.append(n_cells)
        cells.ra.append(np.asarray(phi, dtype=np.float64))
        cells.dec.append(np.pi / 2.0 - np.asarray(theta, dtype=np.float64))

        # children = elements of the previous level
        parent_of_prev = prev_ids >> (shift - prev_shift)
        inv_prev = np.searchsorted(cell_ids, parent_of_prev)
        order = np.argsort(inv_prev, kind="stable")
        indptr = np.zeros(n_cells + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(np.bincount(inv_prev, minlength=n_cells))
        cells.child_indptr.append(indptr)
        cells.child_indices.append(pix_ring[order] if level == 0 else order.astype(np.int64))

        prev_ids = cell_ids
        prev_shift = shift
    return cells


# ── global (all patches) treecode geometry ────────────────────────────────


@dataclass
class TreecodeGeometry:
    """Per-patch coarse cells of all patches (host side)."""

    base_nside: int
    coarse_nsides: Tuple[int, ...]
    # (n_levels, n_patches + 1): cumulative cell counts per level
    cell_offsets: np.ndarray
    # per level: CSR over that level's cells (all patches concatenated).
    # level 0 children are HEALPix RING ids, deeper levels index the previous
    # level's cells.
    child_indptr: List[np.ndarray]
    child_indices: List[np.ndarray]
    cell_ra: List[np.ndarray]
    cell_dec: List[np.ndarray]

    @property
    def n_levels(self) -> int:
        return len(self.coarse_nsides)

    @property
    def n_patches(self) -> int:
        return int(self.cell_offsets.shape[1] - 1)

    @property
    def cells_per_level(self) -> np.ndarray:
        return self.cell_offsets[:, -1].astype(np.int64)

    @property
    def n_cells(self) -> int:
        return int(self.cells_per_level.sum())

    def level_starts(self, first: int = 0) -> np.ndarray:
        """Appended-row index of the first cell of every level (+ total)."""
        starts = np.zeros(self.n_levels + 1, dtype=np.int64)
        starts[1:] = np.cumsum(self.cells_per_level)
        return starts + int(first)

    def level_index(self, nside: int) -> int:
        return self.coarse_nsides.index(int(nside))

    @classmethod
    def from_patches(
        cls, base_nside: int, coarse_nsides: Sequence[int], patches: Sequence[PatchCells]
    ) -> "TreecodeGeometry":
        n_levels = len(coarse_nsides)
        n_patches = len(patches)
        cell_offsets = np.zeros((n_levels, n_patches + 1), dtype=np.int64)
        for level in range(n_levels):
            counts = [p.n_cells[level] for p in patches]
            cell_offsets[level, 1:] = np.cumsum(counts)

        child_indptr, child_indices, cell_ra, cell_dec = [], [], [], []
        for level in range(n_levels):
            n_cells = int(cell_offsets[level, -1])
            indptr = np.zeros(n_cells + 1, dtype=np.int64)
            indices = []
            nnz = 0
            for p, patch in enumerate(patches):
                a, b = cell_offsets[level, p], cell_offsets[level, p + 1]
                indptr[a + 1 : b + 1] = patch.child_indptr[level][1:] + nnz
                idx = patch.child_indices[level]
                if level > 0:
                    idx = idx + cell_offsets[level - 1, p]
                indices.append(idx)
                nnz += int(idx.size)
            child_indptr.append(indptr)
            child_indices.append(
                np.concatenate(indices) if indices else np.zeros(0, dtype=np.int64)
            )
            cell_ra.append(
                np.concatenate([p.ra[level] for p in patches]) if patches else np.zeros(0)
            )
            cell_dec.append(
                np.concatenate([p.dec[level] for p in patches]) if patches else np.zeros(0)
            )
        return cls(
            base_nside=int(base_nside),
            coarse_nsides=tuple(int(n) for n in coarse_nsides),
            cell_offsets=cell_offsets,
            child_indptr=child_indptr,
            child_indices=child_indices,
            cell_ra=cell_ra,
            cell_dec=cell_dec,
        )

    def slice_patches(self, start: int, stop: int) -> "TreecodeGeometry":
        """Geometry restricted to patches ``start:stop`` (cells renumbered)."""
        n_levels = self.n_levels
        offsets = self.cell_offsets[:, start : stop + 1] - self.cell_offsets[:, start : start + 1]
        child_indptr, child_indices, cell_ra, cell_dec = [], [], [], []
        for level in range(n_levels):
            a, b = int(self.cell_offsets[level, start]), int(self.cell_offsets[level, stop])
            indptr = self.child_indptr[level][a : b + 1]
            i0, i1 = int(indptr[0]), int(indptr[-1])
            idx = self.child_indices[level][i0:i1]
            if level > 0:
                idx = idx - self.cell_offsets[level - 1, start]
            child_indptr.append(indptr - i0)
            child_indices.append(idx)
            cell_ra.append(self.cell_ra[level][a:b])
            cell_dec.append(self.cell_dec[level][a:b])
        return TreecodeGeometry(
            base_nside=self.base_nside,
            coarse_nsides=self.coarse_nsides,
            cell_offsets=offsets,
            child_indptr=child_indptr,
            child_indices=child_indices,
            cell_ra=cell_ra,
            cell_dec=cell_dec,
        )

    def members_per_cell(self, level: int) -> np.ndarray:
        """Number of base pixels in every cell of ``level`` (for tests)."""
        counts = np.diff(self.child_indptr[0]).astype(np.int64)
        for lv in range(1, level + 1):
            indptr = self.child_indptr[lv]
            csum = np.concatenate(([0], np.cumsum(counts[self.child_indices[lv]])))
            counts = csum[indptr[1:]] - csum[indptr[:-1]]
        return counts
