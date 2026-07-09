from typing import Callable, Optional, Tuple, Union

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
            if pix_arr.size > 0 and np.max(pix_arr) > max_int64:
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
    """|filter(θ)| for the filter-weighted masking check.

    The absolute value is deliberate: compensated aperture filters can be
    negative at large radii, and signed weights would let those regions
    cancel masked area elsewhere (and push the "fraction" outside
    [0, 1]).  A masked pixel costs the aperture |weight| of filter
    support regardless of the weight's sign.
    """
    if aperture_filter is None:
        values = Q_crittenden(theta, theta_Q)
    else:
        try:
            values = aperture_filter(theta, theta_Q)
        except TypeError:
            values = aperture_filter(theta)
    return np.abs(np.asarray(values, dtype=np.float64))


def select_patch_centers(
    mask: np.ndarray,
    nside_centers: int,
    patch_size: float = 90.0,
    theta_Q: Optional[float] = None,
    f_mask: float = 0.2,
    f_mask_filter: Optional[float] = None,
    filter_weighted: bool = False,
    aperture_filter: Optional[Callable[..., np.ndarray]] = None,
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
        filter_weighted: If ``True``, the filter-disc check uses the
            aperture-filter-weighted masked fraction
            ``Σ_masked |Q(θ)| / Σ_all |Q(θ)|`` instead of the raw pixel
            fraction, so masked pixels only matter in proportion to the
            filter support they remove from the aperture mass (a hole
            near the disc edge, where Q is negligible, no longer vetoes
            the patch, while a hole at the filter peak counts more).
            The magnitude ``|Q|`` is used because compensated filters
            can be negative at large radii.  The patch-disc (2PCF)
            check always uses the raw fraction.
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
        if filter_weighted:
            pix_vec = np.asarray(hp.pix2vec(nside_mask, disc))
            cos_theta = np.clip(vecs[i] @ pix_vec, -1.0, 1.0)
            weights = _aperture_filter_weights(
                aperture_filter, np.arccos(cos_theta), theta_Q
            )
            total = weights.sum()
            if total <= 0:
                # degenerate filter over this disc: cannot assess -> reject
                continue
            masked_fraction = weights[~unmasked[disc]].sum() / total
        else:
            masked_fraction = 1.0 - np.count_nonzero(unmasked[disc]) / disc.size
        if masked_fraction > f_mask_filter:
            continue
        disc = hp.query_disc(nside_mask, vecs[i], patch_radius)
        if disc.size == 0:
            continue
        masked_fraction = 1.0 - np.count_nonzero(unmasked[disc]) / disc.size
        accepted[i] = masked_fraction <= f_mask

    return phi_c[accepted], theta_c[accepted]
