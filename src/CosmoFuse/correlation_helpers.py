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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _xp(*arrays: object) -> object:
    """The array module owning ``arrays`` -- numpy, or cupy on a GPU.

    The zeta reduction is a handful of means over the patch axis, so it can
    run wherever the measurement left its output.  Reducing on the device
    first shrinks a ~1 MB per-patch result to a ~9 kB data vector before it
    ever crosses PCIe (see :class:`CosmoFuse.zeta_writer.ZetaWriter`).
    """
    for a in arrays:
        if type(a).__module__.split(".")[0] == "cupy":
            import cupy

            return cupy
    return np


def Q_crittenden(theta: float, theta_Q: float = 90) -> float:
    """Exponential compensated aperture filter of Crittenden et al. (2002).

    Tangential-shear filter of the Gaussian compensated aperture:

        Q(θ) = θ² / (4π θ_Q⁴) · exp(-θ² / (2 θ_Q²))

    corresponding to the convergence-space filter
    U(θ) = 1/(2π θ_Q²) · (1 - θ²/(2θ_Q²)) · exp(-θ²/(2θ_Q²)), which is
    compensated (∫ dθ θ U(θ) = 0), so M_ap is insensitive to the
    mass-sheet degeneracy.  Q is non-negative, peaks at θ = √2 θ_Q, is
    normalised to ∫ Q(θ) dΩ = 1, and has formally unbounded support;
    CosmoFuse truncates the aperture geometry at 5 θ_Q, where Q has
    decayed to below 10⁻³ of its peak value.

    This is CosmoFuse's default aperture filter, and the filter used for
    the integrated 3-point correlation functions in Halder et al. (2021).

    References:
        Crittenden, Natarajan, Pen & Theuns 2002, ApJ 568, 20
            (arXiv:astro-ph/0012336)
        Halder, Friedrich, Seitz & Wang 2021, MNRAS 506, 2780
            (arXiv:2102.10177)

    Args:
        theta: Angular distance to the aperture centre (radians).
        theta_Q: Aperture filter scale (arcminutes, default 90').

    Returns:
        Filter value Q(θ).
    """

    theta_Q = np.radians(theta_Q / 60)
    return theta**2 / (4 * np.pi * theta_Q**4) * np.exp(-(theta**2) / (2 * theta_Q**2))


def U_crittenden(theta: float, theta_Q: float = 90) -> np.ndarray:
    """Convergence-space partner of :func:`Q_crittenden`:

        U(θ) = 1/(2π θ_Q²) · (1 - θ²/(2θ_Q²)) · exp(-θ²/(2θ_Q²)).

    Compensated (∫ dθ θ U(θ) = 0) and negative beyond θ = √2 θ_Q.  Not used
    for measuring (CosmoFuse measures M_ap from the tangential shear with
    Q); useful for mask-based patch selection, e.g.
    ``select_patch_centers(..., aperture_filter=U_crittenden,
    filter_weighting="signed")`` limits how far the mask may push the
    aperture away from being compensated.
    """
    theta_q = np.radians(theta_Q / 60)
    x2 = np.asarray(theta) ** 2 / (2 * theta_q**2)
    return 1.0 / (2 * np.pi * theta_q**2) * (1.0 - x2) * np.exp(-x2)


def U_schneider(theta: float, theta_Q: float = 90) -> np.ndarray:
    """Convergence-space partner of :func:`Q_schneider`:

        U(θ) = 9/(π θ_Q²) · (1 - x²)(1/3 - x²)   for x = θ/θ_Q ≤ 1, else 0.
    """
    theta_ap = np.radians(theta_Q / 60)
    x2 = (np.asarray(theta) / theta_ap) ** 2
    values = 9.0 / (np.pi * theta_ap**2) * (1.0 - x2) * (1.0 / 3.0 - x2)
    return np.where(x2 < 1.0, values, 0.0)


def Q_schneider(theta: float, theta_Q: float = 90) -> np.ndarray:
    """Polynomial compensated aperture filter of Schneider et al. (1998).

    The widely used ℓ = 1 member of the polynomial filter family:

        Q(θ) = 6/(π θ_Q²) · x² (1 - x²)    for x = θ/θ_Q ≤ 1,
        Q(θ) = 0                           for x > 1,

    corresponding to the convergence-space filter
    U(θ) = 9/(π θ_Q²) · (1 - x²)(1/3 - x²) for x ≤ 1, which is
    compensated and negative for 1/√3 < x < 1.  Like the Crittenden
    et al. (2002) filter it is normalised to ∫ Q(θ) dΩ = 1, but it has
    compact support: it vanishes identically beyond θ_Q.  (CosmoFuse
    builds aperture geometry out to 5 θ_Q; with this filter the pixels
    beyond θ_Q simply receive zero weight.)

    Use it via the modular filter hooks, e.g.
    ``preprocess(aperture_filter=Q_schneider)`` or
    ``select_patch_centers(..., aperture_filter=Q_schneider)``.

    References:
        Schneider, van Waerbeke, Jain & Kruse 1998, MNRAS 296, 873
            (arXiv:astro-ph/9708143)

    Args:
        theta: Angular distance to the aperture centre (radians).
        theta_Q: Aperture radius (arcminutes, default 90').  The filter's
            support ends exactly at this radius.

    Returns:
        Filter value Q(θ).
    """

    theta_ap = np.radians(theta_Q / 60)
    x = np.asarray(theta) / theta_ap
    values = 6.0 / (np.pi * theta_ap**2) * x**2 * (1.0 - x**2)
    return np.where(x < 1.0, values, 0.0)


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
    require_triangle: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shape checks shared by every estimator.

    ``require_triangle`` asserts that the annulus holds the upper triangle of
    the centre's tomographic bins.  That is right when the centre and the
    annulus are built from the *same* sample (M_ap with xi_pm, M_g with xi_g)
    and wrong when they are not (M_g with xi_pm, M_a with xi_g, anything with
    xi_t), where the two carry independent binnings.
    """
    xp = _xp(central_field, annulus_field)
    central = xp.asarray(central_field)
    annulus = xp.asarray(annulus_field)

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

    if require_triangle:
        nzbins = central.shape[1]
        expected_pairs = nzbins * (nzbins + 1) // 2
        if annulus.shape[1] != expected_pairs:
            raise ValueError(
                "annulus_field has incompatible number of tomographic pairs; "
                f"expected {expected_pairs} for {nzbins} bins, "
                f"got {annulus.shape[1]}"
            )

    return central, annulus


class _ZetaIndices:
    """Which centre bin and which annulus combination each output row needs.

    The arrays are tiny (one entry per output row) and fixed by the binning,
    but on a GPU a fresh host array would be uploaded on every use, which
    costs more than the reduction it indexes.  So the device copy is made
    once per device and kept; ``getDevice()`` keys it so a second GPU gets
    its own rather than silently reading the first one's memory.
    """

    __slots__ = ("centers", "pairs", "_device_copies")

    def __init__(self, centers: np.ndarray, pairs: np.ndarray) -> None:
        centers.flags.writeable = False
        pairs.flags.writeable = False
        self.centers = centers
        self.pairs = pairs
        self._device_copies: Dict[int, Tuple[object, object]] = {}

    def for_module(self, xp: object) -> Tuple[object, object]:
        runtime = getattr(getattr(xp, "cuda", None), "runtime", None)
        if runtime is None:
            # numpy, or a numpy-backed stand-in: the host arrays index it.
            return self.centers, self.pairs
        device = runtime.getDevice()
        copies = self._device_copies.get(device)
        if copies is None:
            copies = (xp.asarray(self.centers), xp.asarray(self.pairs))
            self._device_copies[device] = copies
        return copies


_TRIPLET_INDEX_MEMO: Dict[int, _ZetaIndices] = {}
_CROSS_INDEX_MEMO: Dict[Tuple[int, int], _ZetaIndices] = {}


def _triplet_indices(nzbins: int) -> _ZetaIndices:
    """Row indices of the upper-triangular zeta triplets.

    Triplet ``k`` of ``combinations_with_replacement(range(nzbins), 3)`` is
    ``(z_center, z2, z3)`` with ``z2 <= z3``; the annulus it needs sits at
    ``_get_pair_index(nzbins, z2, z3)`` of the pair vector.
    """
    cached = _TRIPLET_INDEX_MEMO.get(nzbins)
    if cached is None:
        combs = list(itertools.combinations_with_replacement(range(nzbins), 3))
        cached = _ZetaIndices(
            np.fromiter((c[0] for c in combs), dtype=np.intp, count=len(combs)),
            np.fromiter(
                (_get_pair_index(nzbins, c[1], c[2]) for c in combs),
                dtype=np.intp,
                count=len(combs),
            ),
        )
        _TRIPLET_INDEX_MEMO[nzbins] = cached
    return cached


def _cross_indices(nzbins: int, n_correlations: int) -> _ZetaIndices:
    """Row indices of the generic γ_t layout.

    Every centre bin against every annulus combination, centre-major — the
    order the nested ``(z_center, pair_idx)`` loop produced.
    """
    key = (nzbins, n_correlations)
    cached = _CROSS_INDEX_MEMO.get(key)
    if cached is None:
        cached = _ZetaIndices(
            np.repeat(np.arange(nzbins, dtype=np.intp), n_correlations),
            np.tile(np.arange(n_correlations, dtype=np.intp), nzbins),
        )
        _CROSS_INDEX_MEMO[key] = cached
    return cached


def _zeta_covariance(
    central: np.ndarray,
    annulus: np.ndarray,
    indices: _ZetaIndices,
) -> np.ndarray:
    """Patch covariance of every (centre, annulus) combination at once.

        out[:, k, :] = ⟨C_{c_k} · A_{p_k}⟩_patches - ⟨C_{c_k}⟩·⟨A_{p_k}⟩

    ``indices`` names the centre bin and the annulus combination of each
    output row, so the whole triplet list is one gather, one product and one
    mean.  The per-triplet loop this replaces issued about five array
    operations per triplet; at the production geometry the reduction is bound
    by that launch count and not by its flops, the arrays being ~1 MB against
    a ~4 ms measurement.

    The means of the centres and of the annuli are taken over the
    *ungathered* arrays and then indexed — cheaper (``nzbins`` and ``ncomb``
    rows instead of one per triplet) and bit-for-bit what the loop did.
    """
    xp = _xp(central, annulus)
    center_inds, pair_inds = indices.for_module(xp)
    center_vals = central[:, center_inds, :]
    annulus_vals = annulus[:, pair_inds, :, :]

    mean_product = xp.mean(center_vals[:, :, :, None] * annulus_vals, axis=2)
    mean_center = xp.mean(central, axis=2)[:, center_inds]
    mean_annulus = xp.mean(annulus, axis=2)[:, pair_inds]

    return mean_product - mean_center[:, :, None] * mean_annulus


def _layout_indices(
    nzbins: int, n_correlations: int, symmetric: Optional[bool]
) -> _ZetaIndices:
    """Pick the row layout of a (centre, annulus) combination.

    ``symmetric=None`` infers it from the counts, which is only safe when the
    two fields cannot be confused -- see :func:`_zeta_from_fields`.
    """
    if symmetric is None:
        symmetric = n_correlations == nzbins * (nzbins + 1) // 2
    if symmetric:
        return _triplet_indices(nzbins)
    return _cross_indices(nzbins, n_correlations)


def _zeta_from_fields(
    central_field: np.ndarray,
    annulus_field: np.ndarray,
    symmetric: Optional[bool] = None,
) -> np.ndarray:
    """Compute i3PCF: covariance between a central aperture field and an
    annular 2PCF field.

    Two row layouts, selected by ``symmetric``:

    - **symmetric** -- centre and annulus share one tomographic binning, so
      the rows are ``combinations_with_replacement(range(nzbins), 3)``,
      ``(z_center, z2, z3)`` with ``z2 <= z3``::

          ζ(θ) = ⟨ C_{z_center} · A_{z2,z3}(θ) ⟩_patches
               - ⟨ C_{z_center} ⟩ · ⟨ A_{z2,z3}(θ) ⟩

    - **generic** -- they do not, so every centre bin meets every annulus
      combination, centre-major: ``nzbins * n_correlations`` rows.  This is
      what a weak-lensing centre against a clustering annulus needs (and the
      reverse), and what γ_t has always needed.

    ``symmetric=None`` *infers* the layout from the counts.  That is right
    whenever the two counts cannot coincide by accident, and wrong when they
    can: ``n_lens == n_source`` makes a cross annulus hold exactly
    ``nz(nz+1)/2`` entries and it is then read as the triangle, silently
    giving the wrong pairings and too few rows.  Pass ``symmetric`` explicitly
    for any estimator whose centre and annulus come from different samples.
    """
    central, annulus = _validate_and_cast_fields(
        central_field, annulus_field, require_triangle=symmetric is True
    )
    indices = _layout_indices(
        int(central.shape[1]), int(annulus.shape[1]), symmetric
    )
    return _zeta_covariance(central, annulus, indices)


def _zeta_from_cross_fields(
    central_field: np.ndarray,
    annulus_field: np.ndarray,
    symmetric: Optional[bool] = None,
) -> np.ndarray:
    """Deprecated alias of :func:`_zeta_from_fields`.

    The two differed only in whether the upper-triangle rule was enforced
    before the layout was chosen; `_zeta_from_fields` now takes ``symmetric``
    and decides both.  Kept because it names the generic case at the call
    sites that need it.
    """
    return _zeta_from_fields(central_field, annulus_field, symmetric=symmetric)


def zeta_g_plus(
    M_g: np.ndarray, xi_p: np.ndarray, symmetric: Optional[bool] = None
) -> np.ndarray:
    """i3PCF: galaxy density M_g at centre × cosmic shear ξ+ on annulus.

    Correlates the smoothed galaxy overdensity with the parity-even
    shear-shear correlation, probing the galaxy-matter-matter bispectrum.

    The centre and the annulus are built from **different samples**, so their
    tomographic binnings are independent and the output is one row per
    ``(z_center, annulus_combination)``, centre-major --
    :meth:`Correlation.zeta_cross_triplets` gives the order.  ``symmetric``
    is inferred from the counts and is only ambiguous when the two samples
    happen to have the same number of bins; pass it explicitly there.
    """
    return _zeta_from_fields(M_g, xi_p, symmetric=symmetric)


def zeta_g_minus(
    M_g: np.ndarray, xi_m: np.ndarray, symmetric: Optional[bool] = None
) -> np.ndarray:
    """i3PCF: galaxy density M_g at centre × cosmic shear ξ- on annulus.

    Like ζ_g+ but using the parity-odd shear correlation ξ-; sensitive
    to B-mode contamination.

    The centre and the annulus are built from **different samples**, so their
    tomographic binnings are independent and the output is one row per
    ``(z_center, annulus_combination)``, centre-major --
    :meth:`Correlation.zeta_cross_triplets` gives the order.  ``symmetric``
    is inferred from the counts and is only ambiguous when the two samples
    happen to have the same number of bins; pass it explicitly there.
    """
    return _zeta_from_fields(M_g, xi_m, symmetric=symmetric)


def zeta_a_plus(M_a: np.ndarray, xi_p: np.ndarray) -> np.ndarray:
    """i3PCF: aperture mass M_ap at centre × cosmic shear ξ+ on annulus.

    Correlates the aperture mass (a pure E-mode measure of projected
    mass) with the shear-shear correlation ξ+.
    """
    return _zeta_from_fields(M_a, xi_p, symmetric=True)


def zeta_a_minus(M_a: np.ndarray, xi_m: np.ndarray) -> np.ndarray:
    """i3PCF: aperture mass M_ap at centre × cosmic shear ξ- on annulus."""
    return _zeta_from_fields(M_a, xi_m, symmetric=True)


def zeta_g_g(M_g: np.ndarray, xi_g: np.ndarray) -> np.ndarray:
    """i3PCF: galaxy density M_g at centre × galaxy clustering ξ_g on annulus.

    Probes the galaxy-galaxy-galaxy three-point function — the excess
    probability of finding three galaxies in a specific triangular
    configuration.
    """
    return _zeta_from_fields(M_g, xi_g, symmetric=True)


def zeta_a_g(
    M_a: np.ndarray, xi_g: np.ndarray, symmetric: Optional[bool] = None
) -> np.ndarray:
    """i3PCF: aperture mass M_ap at centre × galaxy clustering ξ_g on annulus.

    Cross-correlates the projected mass (via lensing) with galaxy
    clustering, probing the matter-galaxy-galaxy bispectrum.

    The centre and the annulus are built from **different samples**, so their
    tomographic binnings are independent and the output is one row per
    ``(z_center, annulus_combination)``, centre-major --
    :meth:`Correlation.zeta_cross_triplets` gives the order.  ``symmetric``
    is inferred from the counts and is only ambiguous when the two samples
    happen to have the same number of bins; pass it explicitly there.
    """
    return _zeta_from_fields(M_a, xi_g, symmetric=symmetric)


def zeta_g_t(
    M_g: np.ndarray, xi_t: np.ndarray, symmetric: Optional[bool] = None
) -> np.ndarray:
    """i3PCF: galaxy density M_g at centre × tangential shear γ_t on annulus.

    Uses the galaxy-galaxy lensing signal as the annular field; probes
    the galaxy-galaxy-matter bispectrum.
    """
    return _zeta_from_cross_fields(M_g, xi_t, symmetric=symmetric)


def zeta_a_t(
    M_a: np.ndarray, xi_t: np.ndarray, symmetric: Optional[bool] = None
) -> np.ndarray:
    """i3PCF: aperture mass M_ap at centre × tangential shear γ_t on annulus.

    Correlates lensing mass with galaxy-galaxy lensing; probes the
    matter-galaxy-matter bispectrum.
    """
    return _zeta_from_cross_fields(M_a, xi_t, symmetric=symmetric)


#: (result name, central field, annulus field, symmetric-triplet layout).
#: ``True`` only where the centre and the annulus are built from the *same*
#: sample, so the upper triangle is the right layout and the count is a real
#: check.  ``None`` -- decided per call -- for every mixed-sample estimator:
#: the two gamma_t entries, and the three that cross weak lensing with galaxy
#: clustering.
_ZETA_PLAN: Tuple[Tuple[str, str, str, Optional[bool]], ...] = (
    ("zeta_g_plus", "M_g", "xi_p", None),
    ("zeta_g_minus", "M_g", "xi_m", None),
    ("zeta_a_plus", "M_a", "xi_p", True),
    ("zeta_a_minus", "M_a", "xi_m", True),
    ("zeta_g_g", "M_g", "xi_g", True),
    ("zeta_a_g", "M_a", "xi_g", None),
    ("zeta_g_t", "M_g", "xi_t", None),
    ("zeta_a_t", "M_a", "xi_t", None),
)


def _batched_zetas(
    centrals: Dict[str, Any],
    annuli: Dict[str, Any],
    requested: Sequence[Tuple[str, str, str, Optional[bool]]],
) -> Optional[Dict[str, np.ndarray]]:
    """All requested estimators in one gather, one product and one mean.

    Each estimator is already a single gather after the per-triplet loop was
    vectorised, but eight of them is still eight times the fixed array-op
    cost -- and at these sizes that cost *is* the runtime (~0.03 ms per cupy
    operation against ~1 MB of data).  Concatenating the centres once and the
    annuli once turns eight sets of operations into one: measured 3.20 ->
    0.42 ms at the production shape, bit-for-bit identical.

    Returns ``None`` when the batch cannot be formed, and the caller falls
    back to one call per estimator:

    - a single estimator, where there is nothing to batch;
    - mixed dtypes, because concatenation would promote them and silently
      change an output's precision;
    - annuli that disagree on the patch or angular-bin axes, or centrals that
      disagree on the patch axis, which concatenation cannot express.

    The centrals may hold *different* numbers of tomographic bins -- M_ap over
    source bins beside M_g over lens bins -- so every offset and layout below
    is taken per field rather than from one shared ``nzbins``.
    """
    if len(requested) < 2:
        return None

    central_names = sorted({c for _, c, _, _ in requested})
    annulus_names = sorted({a for _, _, a, _ in requested})
    used_centrals = [centrals[name] for name in central_names]
    used_annuli = [annuli[name] for name in annulus_names]

    dtypes = {np.dtype(a.dtype) for a in used_centrals + used_annuli}
    if len(dtypes) != 1:
        return None
    if len({a.shape[2:] for a in used_annuli}) != 1:
        return None
    if len({a.shape[2] for a in used_centrals}) != 1:
        return None
    if len({a.shape[0] for a in used_centrals + used_annuli}) != 1:
        return None

    xp = _xp(*used_centrals, *used_annuli)

    centre_offset, offset = {}, 0
    for name in central_names:
        centre_offset[name] = offset
        offset += int(centrals[name].shape[1])
    annulus_offset, offset = {}, 0
    for name in annulus_names:
        annulus_offset[name] = offset
        offset += int(annuli[name].shape[1])

    centre_parts, pair_parts, plan = [], [], []
    for result_name, central_name, annulus_name, symmetric in requested:
        nzbins = int(centrals[central_name].shape[1])
        n_correlations = int(annuli[annulus_name].shape[1])
        if symmetric and n_correlations != nzbins * (nzbins + 1) // 2:
            return None  # let the per-estimator path raise the real error
        indices = _layout_indices(nzbins, n_correlations, symmetric)
        centre_parts.append(indices.centers + centre_offset[central_name])
        pair_parts.append(indices.pairs + annulus_offset[annulus_name])
        plan.append((result_name, len(indices.centers)))

    central = (
        used_centrals[0]
        if len(central_names) == 1
        else xp.concatenate(used_centrals, axis=1)
    )
    annulus = (
        used_annuli[0]
        if len(annulus_names) == 1
        else xp.concatenate(used_annuli, axis=1)
    )
    indices = _ZetaIndices(
        np.concatenate(centre_parts), np.concatenate(pair_parts)
    )
    stacked = _zeta_covariance(central, annulus, indices)

    out: Dict[str, np.ndarray] = {}
    row = 0
    for result_name, n_rows in plan:
        out[result_name] = stacked[:, row : row + n_rows, :]
        row += n_rows
    return out


def calculate_all_zetas(
    M_g: Optional[np.ndarray] = None,
    M_a: Optional[np.ndarray] = None,
    xi_p: Optional[np.ndarray] = None,
    xi_m: Optional[np.ndarray] = None,
    xi_g: Optional[np.ndarray] = None,
    xi_t: Optional[np.ndarray] = None,
    xi_t_symmetric: Optional[bool] = None,
    symmetric: Optional[Mapping[str, bool]] = None,
) -> Dict[str, np.ndarray]:
    """Calculate all supported i3PCFs in Halder et al. notation.

    Keys in the returned dictionary are exactly the implemented helper names.

    The centrals may carry different tomographies -- ``M_a`` over source bins
    and ``M_g`` over lens bins is the 3x2pt case -- and each estimator is then
    laid out accordingly: the upper triangle where the centre and the annulus
    share a sample, one row per ``(z_center, annulus_combination)`` where they
    do not.  :meth:`Correlation.zeta_triplets` and
    :meth:`Correlation.zeta_cross_triplets` give the two row orders.

    ``symmetric`` forces the layout of named estimators, e.g.
    ``{"zeta_a_g": False}``.  Needed only when the counts are ambiguous --
    equal numbers of source and lens bins make a cross annulus look like the
    triangle -- and it is then the only way to say which was meant.
    ``xi_t_symmetric`` is the older spelling of that override for the two γ_t
    estimators; ``symmetric`` wins where both name one.
    """
    centrals = {"M_g": M_g, "M_a": M_a}
    annuli = {"xi_p": xi_p, "xi_m": xi_m, "xi_g": xi_g, "xi_t": xi_t}
    overrides = dict(symmetric or {})
    unknown = set(overrides) - {entry[0] for entry in _ZETA_PLAN}
    if unknown:
        raise ValueError(f"unknown estimator name(s) in symmetric: {sorted(unknown)}")

    requested: List[Tuple[str, str, str, Optional[bool]]] = []
    for name, central_name, annulus_name, plan_symmetric in _ZETA_PLAN:
        if centrals[central_name] is None or annuli[annulus_name] is None:
            continue
        if name in overrides:
            resolved: Optional[bool] = bool(overrides[name])
        elif plan_symmetric is None and annulus_name == "xi_t":
            resolved = xi_t_symmetric
        else:
            resolved = plan_symmetric
        requested.append((name, central_name, annulus_name, resolved))

    if requested:
        # Validate every pair through the individual helpers' checks first,
        # so a bad shape raises the same error whichever path runs.
        for _name, central_name, annulus_name, resolved in requested:
            if resolved:
                _validate_and_cast_fields(
                    centrals[central_name], annuli[annulus_name]
                )
        batched = _batched_zetas(centrals, annuli, requested)
        if batched is not None:
            return batched

    return {
        name: _zeta_from_fields(
            centrals[central_name], annuli[annulus_name], symmetric=resolved
        )
        for name, central_name, annulus_name, resolved in requested
    }

