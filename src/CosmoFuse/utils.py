import itertools
import weakref
from typing import Any, Callable, Dict, Optional, Tuple, Union

import healpy as hp
import numpy as np

from .correlation_helpers import Q_crittenden


def pixel2RaDec(
    pixel_indices: Union[int, np.ndarray], NSIDE: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert pixel indices to right ascension and declination.

    Args:
        pixel_indices: Pixel indices in the HEALPix map
        NSIDE: HEALPix resolution parameter

    Returns:
        Tuple of (ra, dec) in radians
    """
    if np.isscalar(pixel_indices):
        pix_for_healpy = int(pixel_indices)
    else:
        pix_arr = np.asarray(pixel_indices)
        if np.issubdtype(pix_arr.dtype, np.unsignedinteger):
            max_int64 = np.iinfo(np.int64).max
            # int(): under legacy (numpy<2) promotion, uint64 vs python-int
            # comparison goes through float64 and loses the last bits.
            if pix_arr.size > 0 and int(np.max(pix_arr)) > max_int64:
                raise ValueError("pixel indices exceed int64 range")
        pix_for_healpy = pix_arr.astype(np.int64, copy=False)

    theta, phi = hp.pixelfunc.pix2ang(NSIDE, pix_for_healpy, nest=False)
    ra = phi
    dec = np.pi / 2.0 - theta
    return ra, dec


def _aperture_filter_weights(
    aperture_filter: Optional[Callable[..., np.ndarray]],
    theta: np.ndarray,
    theta_Q: float,
) -> np.ndarray:
    """Signed filter(θ) values for the filter-weighted masking check."""
    if aperture_filter is None:
        values = Q_crittenden(theta, theta_Q)
    else:
        try:
            values = aperture_filter(theta, theta_Q)
        except TypeError:
            values = aperture_filter(theta)
    return np.asarray(values, dtype=np.float64)


_FILTER_WEIGHTINGS = ("abs", "signed")


def select_patch_centers(
    mask: np.ndarray,
    nside_centers: int,
    patch_size: float = 90.0,
    theta_Q: Optional[float] = None,
    f_mask: float = 0.2,
    f_mask_filter: Optional[float] = None,
    aperture_filter: Optional[Callable[..., np.ndarray]] = None,
    filter_weighting: str = "abs",
) -> Tuple[np.ndarray, np.ndarray]:
    """Select patch centres on a coarse grid whose surroundings are
    sufficiently unmasked.

    Candidate centres are the pixel centres of an ``nside_centers``
    HEALPix grid that fall inside the (downgraded) footprint.  A
    candidate is accepted when the masked fraction of the
    full-resolution ``mask`` is at most ``f_mask`` within the 2PCF patch
    disc (radius ``patch_size``) and at most ``f_mask_filter`` within
    the compensated-filter support disc (radius ``5 * theta_Q`` — the
    same region ``Correlation.calculate_pairs_M_a`` uses).

    ``nside_centers`` controls the patch (over)sampling density: a finer
    grid yields more, more strongly overlapping patches.

    Args:
        mask: Full-resolution HEALPix mask/footprint (nonzero = observed).
        nside_centers: Resolution of the candidate-centre grid (coarser
            than the mask resolution).
        patch_size: Patch radius in arcminutes (same meaning as
            ``Correlation(patch_size=...)``).
        theta_Q: Compensated filter scale in arcminutes; defaults to
            ``patch_size``.
        f_mask: Maximum tolerated masked fraction inside the patch disc.
        f_mask_filter: Maximum tolerated masked fraction inside the
            filter support disc; defaults to ``f_mask``.
        filter_weighting: How the masking of the filter support disc is
            measured.  The mask is always used as a *binary* mask at its
            own resolution: a pixel contributes its whole filter value
            (evaluated at the pixel centre) if it is unmasked and nothing
            otherwise -- never a fraction, and never the survey weights.

            ``"abs"`` (default) -- fraction of *importance* lost,
            ``Σ_masked |Q| / Σ_all |Q| <= f_mask_filter``.  A hole near the
            disc edge, where the filter is negligible, no longer vetoes a
            patch, while a hole at the filter peak counts more.  Filter
            regions of either sign count as lost support.

            ``"signed"`` -- *deviation of the filter
            integral*, ``|Σ_masked Q| / Σ_all |Q| <= f_mask_filter``.  For a
            compensated filter (``Σ_all U = 0``, e.g. ``U_crittenden``) this
            is how far the mask pushes the aperture away from being
            compensated: masked regions of opposite sign cancel.  For a
            non-negative filter (``Q_crittenden``, ``Q_schneider``) it is
            identical to ``"abs"``.

            The patch-disc (2PCF) check always uses the pixel fraction.
        aperture_filter: Filter used for the weighting; defaults to the
            built-in ``Q_crittenden`` (same convention as
            ``Correlation.preprocess``: called as
            ``aperture_filter(theta, theta_Q)`` with theta in radians
            and theta_Q in arcminutes, falling back to
            ``aperture_filter(theta)``).  Pass the same filter here and
            to ``preprocess`` for a consistent selection.

    Returns:
        ``(phi_center, theta_center)`` in radians, ordered to match the
        ``Correlation`` constructor, so
        ``Correlation(nside, *select_patch_centers(...), ...)`` works.
    """
    mask = np.asarray(mask)
    if mask.ndim != 1:
        raise ValueError("mask must be a 1-D HEALPix map")
    nside_mask = hp.npix2nside(mask.size)
    if theta_Q is None:
        theta_Q = float(patch_size)
    if f_mask_filter is None:
        f_mask_filter = f_mask
    if patch_size <= 0 or theta_Q <= 0:
        raise ValueError("patch_size and theta_Q must be positive")
    if not (0 <= f_mask <= 1) or not (0 <= f_mask_filter <= 1):
        raise ValueError("f_mask and f_mask_filter must lie in [0, 1]")
    if filter_weighting not in _FILTER_WEIGHTINGS:
        raise ValueError(
            f"filter_weighting must be 'abs' or 'signed'; got {filter_weighting!r}"
        )
    weighting = filter_weighting

    patch_radius = np.radians(patch_size / 60.0)
    filter_radius = 5.0 * np.radians(theta_Q / 60.0)

    # Candidate centres: coarse-grid pixels inside the footprint
    mask_lr = hp.ud_grade(mask.astype(np.float64), nside_centers)
    candidate_pix = np.flatnonzero(mask_lr != 0)
    if candidate_pix.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty

    theta_c, phi_c = hp.pix2ang(nside_centers, candidate_pix)
    vecs = hp.ang2vec(theta_c, phi_c)

    unmasked = mask != 0
    accepted = np.zeros(candidate_pix.size, dtype=bool)
    for i in range(candidate_pix.size):
        # Filter support disc first: it is the larger of the two, so it
        # rejects earlier and the patch disc is only queried on survivors.
        disc = hp.query_disc(nside_mask, vecs[i], filter_radius)
        if disc.size == 0:
            continue
        pix_vec = np.asarray(hp.pix2vec(nside_mask, disc))
        cos_theta = np.clip(vecs[i] @ pix_vec, -1.0, 1.0)
        weights = _aperture_filter_weights(
            aperture_filter, np.arccos(cos_theta), theta_Q
        )
        total = np.abs(weights).sum()
        if total <= 0:
            # degenerate filter over this disc: cannot assess -> reject
            continue
        masked = weights[~unmasked[disc]]
        if weighting == "abs":
            masked_fraction = np.abs(masked).sum() / total
        else:
            masked_fraction = abs(masked.sum()) / total
        if masked_fraction > f_mask_filter:
            continue
        disc = hp.query_disc(nside_mask, vecs[i], patch_radius)
        if disc.size == 0:
            continue
        masked_fraction = 1.0 - np.count_nonzero(unmasked[disc]) / disc.size
        accepted[i] = masked_fraction <= f_mask

    return phi_c[accepted], theta_c[accepted]


_OBJECT_SERIALS: Dict[int, Tuple[Any, int]] = {}
_SERIAL_COUNTER = itertools.count(1)


def live_object_serial(obj: Any) -> int:
    """A number identifying *this* object for as long as it is alive.

    ``id()`` and a CUDA pool pointer are unique only among live objects:
    CPython reuses addresses as soon as an object is collected, and cupy's
    memory pool hands a freed pointer straight to the next allocation.  A
    cache keyed on either can therefore be handed a *different* object that
    compares equal to the one it stored — which is how a transient aperture
    filter, and a weight map rebuilt once per realisation, both end up
    silently reusing the previous one's cached result.

    Pairing the address with a serial closes that.  The entry is reused only
    while the weak reference still resolves to the same object; a new object
    at a recycled address gets a fresh serial.  The reference keeps nothing
    alive and its callback drops the entry when the object dies, so the table
    is bounded by the number of live objects that were ever fingerprinted.

    Objects that cannot be weak-referenced fall back to their address, which
    is no worse than what this replaces.
    """
    key = id(obj)
    entry = _OBJECT_SERIALS.get(key)
    if entry is not None and entry[0]() is obj:
        return entry[1]

    serial = next(_SERIAL_COUNTER)

    def _drop(dead_ref: Any, _key: int = key) -> None:
        # Only if the address has not already been re-registered: the
        # callback runs after the object is gone, by which time a new one
        # may already occupy it.
        current = _OBJECT_SERIALS.get(_key)
        if current is not None and current[0] is dead_ref:
            del _OBJECT_SERIALS[_key]

    try:
        ref = weakref.ref(obj, _drop)
    except TypeError:
        return key
    _OBJECT_SERIALS[key] = (ref, serial)
    return serial
