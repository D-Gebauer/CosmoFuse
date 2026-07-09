"""
Integrated three-point correlation function (i3PCF) helpers.

Implements the eight i3PCF estimators (zeta functions) following
Halder et al. notation.  Each zeta is the covariance between a
"central" aperture quantity (M_ap or M_g) measured at the centre of
a sky patch and an "annular" two-point correlation (ξ+, ξ-, ξ_g, or
γ_t) measured on annuli around that centre:

    ζ(θ) = ⟨ central · annular_2PCF(θ) ⟩ - ⟨central⟩·⟨annular_2PCF(θ)⟩

The eight estimators are:
    ζ_g+, ζ_g-  — galaxy density centre × cosmic shear annulus
    ζ_a+, ζ_a-  — aperture mass centre × cosmic shear annulus
    ζ_gg        — galaxy density centre × galaxy clustering annulus
    ζ_ag        — aperture mass centre × galaxy clustering annulus
    ζ_gt        — galaxy density centre × galaxy-galaxy lensing annulus
    ζ_at        — aperture mass centre × galaxy-galaxy lensing annulus
"""

import itertools
from typing import Tuple, Dict, Optional

import numpy as np


def Q_T(theta: float, theta_Q: float = 90) -> float:
    """Compensated filter for aperture mass M_ap (Schneider et al. 1998).

    Q_T(θ) = θ² / (4π θ_Q⁴) · exp(-θ² / (2θ_Q²))

    This filter has zero integral (compensated), so M_ap is insensitive
    to the mass-sheet degeneracy.  θ_Q sets the characteristic smoothing
    scale of the aperture.

    Args:
        theta: Angular distance to the aperture centre (radians).
        theta_Q: Aperture filter scale (arcminutes, default 90').

    Returns:
        Filter value Q_T(θ).
    """

    theta_Q = np.radians(theta_Q / 60)
    return theta**2 / (4 * np.pi * theta_Q**4) * np.exp(-(theta**2) / (2 * theta_Q**2))


def _get_pair_index(nbins: int, i: int, j: int) -> int:
    """Get the index of pair (i, j) in the flattened correlation vector.

    Assumes standard upper-triangular ordering (including diagonal):
    (0,0), (0,1), ..., (0, n-1), (1,1), ..., (n-1, n-1).
    """
    if i > j:
        i, j = j, i
    
    # number of elements before row i
    # row 0 has n elements
    # row 1 has n-1 elements
    # ...
    # row k has n-k elements
    # sum_{k=0}^{i-1} (n - k) = i*n - i*(i-1)/2
    
    idx = int(i * nbins - i * (i - 1) / 2)
    # index within row i is j - i
    idx += j - i
    return idx


def _validate_and_cast_fields(
    central_field: np.ndarray,
    annulus_field: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    central = np.asarray(central_field)
    annulus = np.asarray(annulus_field)

    if central.ndim != 3:
        raise ValueError(
            "central_field must have shape (nmaps, nzbins, n_patches); "
            f"got {central.shape}"
        )
    if annulus.ndim != 4:
        raise ValueError(
            "annulus_field must have shape (nmaps, n_correlations, n_patches, nbins); "
            f"got {annulus.shape}"
        )

    if central.shape[0] != annulus.shape[0]:
        raise ValueError(
            "central_field and annulus_field must have the same number of maps; "
            f"got {central.shape[0]} and {annulus.shape[0]}"
        )
    if central.shape[2] != annulus.shape[2]:
        raise ValueError(
            "central_field and annulus_field must share n_patches; "
            f"got {central.shape[2]} and {annulus.shape[2]}"
        )

    nzbins = central.shape[1]
    expected_pairs = nzbins * (nzbins + 1) // 2
    if annulus.shape[1] != expected_pairs:
        raise ValueError(
            "annulus_field has incompatible number of tomographic pairs; "
            f"expected {expected_pairs} for {nzbins} bins, got {annulus.shape[1]}"
        )

    return central, annulus


def _zeta_from_fields(
    central_field: np.ndarray,
    annulus_field: np.ndarray,
) -> np.ndarray:
    """Compute i3PCF: covariance between a central aperture field and an
    annular 2PCF field.

    The i3PCF for a triplet of tomographic bins (z_center, z2, z3) is:

        ζ(θ) = ⟨ C_{z_center} · A_{z2,z3}(θ) ⟩_patches
             - ⟨ C_{z_center} ⟩ · ⟨ A_{z2,z3}(θ) ⟩

    where C is the central field (M_ap or M_g) and A is the annular
    2PCF (ξ+, ξ-, ξ_g) evaluated in angular bins.
    """
    central, annulus = _validate_and_cast_fields(central_field, annulus_field)
    nmaps, nzbins, _ = central.shape
    nbins = annulus.shape[3]
    # All unique triplets of tomo bins (z_center, z2, z3) with z2 ≤ z3
    zeta_combs = list(itertools.combinations_with_replacement(range(nzbins), 3))

    out = np.zeros(
        (nmaps, len(zeta_combs), nbins),
        dtype=np.result_type(central.dtype, annulus.dtype),
    )

    # Hoist the per-center and per-annulus means out of the triplet loop:
    # there are only nzbins distinct centers and ncomb distinct annuli.
    all_center_means = np.mean(central, axis=2)
    all_annulus_means = np.mean(annulus, axis=2)

    for k, (z_center, z2, z3) in enumerate(zeta_combs):
        pair_idx = _get_pair_index(nzbins, z2, z3)
        center_vals = central[:, z_center, :]
        annulus_vals = annulus[:, pair_idx, :, :]

        mean_center = all_center_means[:, z_center]
        mean_annulus = all_annulus_means[:, pair_idx]
        mean_product = np.mean(center_vals[:, :, None] * annulus_vals, axis=1)

        out[:, k, :] = mean_product - mean_center[:, None] * mean_annulus

    return out


def _zeta_from_cross_fields(
    central_field: np.ndarray,
    annulus_field: np.ndarray,
) -> np.ndarray:
    """Compute i3PCF covariance for cross-correlation annulus fields.

    Used for ζ_gt and ζ_at where the annular 2PCF (galaxy-galaxy
    lensing γ_t) has distinct lens and source tomographic bins,
    so the number of annulus combinations may differ from the
    standard upper-triangular count.

    Supports any number of annulus tomographic combinations:
    - If annulus combinations match upper-triangular size for ``nzbins``, preserves
      legacy ordering ``(z_center, z2, z3)`` with ``z2 <= z3``.
    - Otherwise, treats annulus combinations as generic entries and returns all
      ``(z_center, annulus_combination)`` covariances.
    """
    central = np.asarray(central_field)
    annulus = np.asarray(annulus_field)

    if central.ndim != 3:
        raise ValueError(
            "central_field must have shape (nmaps, nzbins, n_patches); "
            f"got {central.shape}"
        )
    if annulus.ndim != 4:
        raise ValueError(
            "annulus_field must have shape (nmaps, n_correlations, n_patches, nbins); "
            f"got {annulus.shape}"
        )
    if central.shape[0] != annulus.shape[0]:
        raise ValueError(
            "central_field and annulus_field must have the same number of maps; "
            f"got {central.shape[0]} and {annulus.shape[0]}"
        )
    if central.shape[2] != annulus.shape[2]:
        raise ValueError(
            "central_field and annulus_field must share n_patches; "
            f"got {central.shape[2]} and {annulus.shape[2]}"
        )

    nmaps, nzbins, _ = central.shape
    n_correlations = annulus.shape[1]
    nbins = annulus.shape[3]

    triangular_pairs = nzbins * (nzbins + 1) // 2
    dtype = np.result_type(central.dtype, annulus.dtype)

    if n_correlations == triangular_pairs:
        zeta_combs = list(itertools.combinations_with_replacement(range(nzbins), 3))
        out = np.zeros((nmaps, len(zeta_combs), nbins), dtype=dtype)
        all_center_means = np.mean(central, axis=2)
        all_annulus_means = np.mean(annulus, axis=2)
        for k, (z_center, z2, z3) in enumerate(zeta_combs):
            pair_idx = _get_pair_index(nzbins, z2, z3)
            center_vals = central[:, z_center, :]
            annulus_vals = annulus[:, pair_idx, :, :]

            mean_center = all_center_means[:, z_center]
            mean_annulus = all_annulus_means[:, pair_idx]
            mean_product = np.mean(center_vals[:, :, None] * annulus_vals, axis=1)
            out[:, k, :] = mean_product - mean_center[:, None] * mean_annulus
        return out

    out = np.zeros((nmaps, nzbins * n_correlations, nbins), dtype=dtype)
    all_annulus_means = np.mean(annulus, axis=2)
    k = 0
    for z_center in range(nzbins):
        center_vals = central[:, z_center, :]
        mean_center = np.mean(center_vals, axis=1)
        for pair_idx in range(n_correlations):
            annulus_vals = annulus[:, pair_idx, :, :]
            mean_annulus = all_annulus_means[:, pair_idx]
            mean_product = np.mean(center_vals[:, :, None] * annulus_vals, axis=1)
            out[:, k, :] = mean_product - mean_center[:, None] * mean_annulus
            k += 1
    return out


def zeta_g_plus(M_g: np.ndarray, xi_p: np.ndarray) -> np.ndarray:
    """i3PCF: galaxy density M_g at centre × cosmic shear ξ+ on annulus.

    Correlates the smoothed galaxy overdensity with the parity-even
    shear-shear correlation, probing the galaxy-matter-matter bispectrum.
    """
    return _zeta_from_fields(M_g, xi_p)


def zeta_g_minus(M_g: np.ndarray, xi_m: np.ndarray) -> np.ndarray:
    """i3PCF: galaxy density M_g at centre × cosmic shear ξ- on annulus.

    Like ζ_g+ but using the parity-odd shear correlation ξ-; sensitive
    to B-mode contamination.
    """
    return _zeta_from_fields(M_g, xi_m)


def zeta_a_plus(M_a: np.ndarray, xi_p: np.ndarray) -> np.ndarray:
    """i3PCF: aperture mass M_ap at centre × cosmic shear ξ+ on annulus.

    Correlates the aperture mass (a pure E-mode measure of projected
    mass) with the shear-shear correlation ξ+.
    """
    return _zeta_from_fields(M_a, xi_p)


def zeta_a_minus(M_a: np.ndarray, xi_m: np.ndarray) -> np.ndarray:
    """i3PCF: aperture mass M_ap at centre × cosmic shear ξ- on annulus."""
    return _zeta_from_fields(M_a, xi_m)


def zeta_g_g(M_g: np.ndarray, xi_g: np.ndarray) -> np.ndarray:
    """i3PCF: galaxy density M_g at centre × galaxy clustering ξ_g on annulus.

    Probes the galaxy-galaxy-galaxy three-point function — the excess
    probability of finding three galaxies in a specific triangular
    configuration.
    """
    return _zeta_from_fields(M_g, xi_g)


def zeta_a_g(M_a: np.ndarray, xi_g: np.ndarray) -> np.ndarray:
    """i3PCF: aperture mass M_ap at centre × galaxy clustering ξ_g on annulus.

    Cross-correlates the projected mass (via lensing) with galaxy
    clustering, probing the matter-galaxy-galaxy bispectrum.
    """
    return _zeta_from_fields(M_a, xi_g)


def zeta_g_t(M_g: np.ndarray, xi_t: np.ndarray) -> np.ndarray:
    """i3PCF: galaxy density M_g at centre × tangential shear γ_t on annulus.

    Uses the galaxy-galaxy lensing signal as the annular field; probes
    the galaxy-galaxy-matter bispectrum.
    """
    return _zeta_from_cross_fields(M_g, xi_t)


def zeta_a_t(M_a: np.ndarray, xi_t: np.ndarray) -> np.ndarray:
    """i3PCF: aperture mass M_ap at centre × tangential shear γ_t on annulus.

    Correlates lensing mass with galaxy-galaxy lensing; probes the
    matter-galaxy-matter bispectrum.
    """
    return _zeta_from_cross_fields(M_a, xi_t)


def calculate_all_zetas(
    M_g: Optional[np.ndarray] = None,
    M_a: Optional[np.ndarray] = None,
    xi_p: Optional[np.ndarray] = None,
    xi_m: Optional[np.ndarray] = None,
    xi_g: Optional[np.ndarray] = None,
    xi_t: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Calculate all supported i3PCFs in Halder et al. notation.

    Keys in the returned dictionary are exactly the implemented helper names.
    """
    results: Dict[str, np.ndarray] = {}

    if M_g is not None and xi_p is not None:
        results["zeta_g_plus"] = zeta_g_plus(M_g, xi_p)
    if M_g is not None and xi_m is not None:
        results["zeta_g_minus"] = zeta_g_minus(M_g, xi_m)
    if M_a is not None and xi_p is not None:
        results["zeta_a_plus"] = zeta_a_plus(M_a, xi_p)
    if M_a is not None and xi_m is not None:
        results["zeta_a_minus"] = zeta_a_minus(M_a, xi_m)
    if M_g is not None and xi_g is not None:
        results["zeta_g_g"] = zeta_g_g(M_g, xi_g)
    if M_a is not None and xi_g is not None:
        results["zeta_a_g"] = zeta_a_g(M_a, xi_g)
    if M_g is not None and xi_t is not None:
        results["zeta_g_t"] = zeta_g_t(M_g, xi_t)
    if M_a is not None and xi_t is not None:
        results["zeta_a_t"] = zeta_a_t(M_a, xi_t)

    return results

