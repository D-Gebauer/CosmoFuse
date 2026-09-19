import itertools

import numpy as np
import pytest
from CosmoFuse.correlation_helpers import (
    Q_crittenden,
    Q_schneider,
    _get_pair_index,
    calculate_all_zetas,
    zeta_a_g,
    zeta_a_minus,
    zeta_a_plus,
    zeta_a_t,
    zeta_g_g,
    zeta_g_minus,
    zeta_g_plus,
    zeta_g_t,
)

def test_get_pair_index():
    # nbins = 3
    # Pairs: (0,0)->0, (0,1)->1, (0,2)->2, (1,1)->3, (1,2)->4, (2,2)->5
    assert _get_pair_index(3, 0, 0) == 0
    assert _get_pair_index(3, 0, 1) == 1
    assert _get_pair_index(3, 1, 0) == 1
    assert _get_pair_index(3, 0, 2) == 2
    assert _get_pair_index(3, 1, 1) == 3
    assert _get_pair_index(3, 1, 2) == 4
    assert _get_pair_index(3, 2, 2) == 5


def _reference_zeta(center, annulus):
    nmaps, nzbins, _ = center.shape
    nbins = annulus.shape[3]
    out = np.zeros((nmaps, int((nzbins + 2) * (nzbins + 1) * nzbins / 6), nbins))
    k = 0
    for z1 in range(nzbins):
        for z2 in range(z1, nzbins):
            for z3 in range(z2, nzbins):
                pair_idx = _get_pair_index(nzbins, z2, z3)
                c = center[:, z1, :]
                a = annulus[:, pair_idx, :, :]
                out[:, k, :] = np.mean(c[:, :, None] * a, axis=1) - np.mean(
                    c, axis=1
                )[:, None] * np.mean(a, axis=1)
                k += 1
    return out


def test_all_8_zeta_variants_match_reference():
    nmaps = 2
    nzbins = 2
    npatches = 9
    nbins = 5
    npairs = nzbins * (nzbins + 1) // 2

    rng = np.random.default_rng(0)
    M_g = rng.normal(size=(nmaps, nzbins, npatches))
    M_a = rng.normal(size=(nmaps, nzbins, npatches))
    xi_p = rng.normal(size=(nmaps, npairs, npatches, nbins))
    xi_m = rng.normal(size=(nmaps, npairs, npatches, nbins))
    xi_g = rng.normal(size=(nmaps, npairs, npatches, nbins))
    xi_t = rng.normal(size=(nmaps, npairs, npatches, nbins))

    expectations = {
        "zeta_g_plus": (zeta_g_plus(M_g, xi_p), _reference_zeta(M_g, xi_p)),
        "zeta_g_minus": (zeta_g_minus(M_g, xi_m), _reference_zeta(M_g, xi_m)),
        "zeta_a_plus": (zeta_a_plus(M_a, xi_p), _reference_zeta(M_a, xi_p)),
        "zeta_a_minus": (zeta_a_minus(M_a, xi_m), _reference_zeta(M_a, xi_m)),
        "zeta_g_g": (zeta_g_g(M_g, xi_g), _reference_zeta(M_g, xi_g)),
        "zeta_a_g": (zeta_a_g(M_a, xi_g), _reference_zeta(M_a, xi_g)),
        "zeta_g_t": (zeta_g_t(M_g, xi_t), _reference_zeta(M_g, xi_t)),
        "zeta_a_t": (zeta_a_t(M_a, xi_t), _reference_zeta(M_a, xi_t)),
    }

    for measured, expected in expectations.values():
        assert measured.shape == (nmaps, 4, nbins)
        np.testing.assert_allclose(measured, expected)

def test_calculate_all_zetas():
    nmaps = 2
    nzbins = 2
    npatches = 10
    nbins = 5
    npairs = nzbins * (nzbins + 1) // 2

    rng = np.random.default_rng(1)
    g = rng.normal(size=(nmaps, nzbins, npatches))
    a = rng.normal(size=(nmaps, nzbins, npatches))
    xi_p = rng.normal(size=(nmaps, npairs, npatches, nbins))
    xi_m = rng.normal(size=(nmaps, npairs, npatches, nbins))
    xi_g = rng.normal(size=(nmaps, npairs, npatches, nbins))
    xi_t = rng.normal(size=(nmaps, npairs, npatches, nbins))

    partial = calculate_all_zetas(M_g=g, xi_p=xi_p, xi_m=xi_m)
    assert set(partial.keys()) == {"zeta_g_plus", "zeta_g_minus"}

    full = calculate_all_zetas(
        M_g=g,
        M_a=a,
        xi_p=xi_p,
        xi_m=xi_m,
        xi_g=xi_g,
        xi_t=xi_t,
    )
    assert set(full.keys()) == {
        "zeta_g_plus",
        "zeta_g_minus",
        "zeta_a_plus",
        "zeta_a_minus",
        "zeta_g_g",
        "zeta_a_g",
        "zeta_g_t",
        "zeta_a_t",
    }


def test_shape_validation_raises_value_error():
    M_g = np.zeros((2, 2, 4))
    bad_annulus = np.zeros((2, 2, 5, 3))
    with pytest.raises(ValueError):
        zeta_g_plus(M_g, bad_annulus)


def test_shape_validation_raises_on_central_ndim():
    M_g = np.zeros((2, 4))
    annulus = np.zeros((2, 3, 4, 2))
    with pytest.raises(ValueError):
        zeta_g_plus(M_g, annulus)


def test_shape_validation_raises_on_annulus_ndim():
    M_g = np.zeros((2, 2, 4))
    annulus = np.zeros((2, 3, 4))
    with pytest.raises(ValueError):
        zeta_g_plus(M_g, annulus)


def test_shape_validation_raises_on_map_count_mismatch():
    M_g = np.zeros((2, 2, 4))
    annulus = np.zeros((3, 3, 4, 2))
    with pytest.raises(ValueError):
        zeta_g_plus(M_g, annulus)


def test_shape_validation_raises_on_patch_count_mismatch():
    M_g = np.zeros((2, 2, 4))
    annulus = np.zeros((2, 3, 5, 2))
    with pytest.raises(ValueError):
        zeta_g_plus(M_g, annulus)


def test_shape_validation_raises_on_pair_count_mismatch():
    M_g = np.zeros((2, 2, 4))
    annulus = np.zeros((2, 4, 4, 2))
    with pytest.raises(ValueError):
        zeta_g_plus(M_g, annulus)


def test_zeta_t_accepts_arbitrary_cross_combination_count():
    nmaps = 2
    nzbins = 3
    npatches = 5
    ncorrelations = 7
    nbins = 4

    rng = np.random.default_rng(42)
    center = rng.normal(size=(nmaps, nzbins, npatches))
    xi_t = rng.normal(size=(nmaps, ncorrelations, npatches, nbins))

    z_gt = zeta_g_t(center, xi_t)
    z_at = zeta_a_t(center, xi_t)

    assert z_gt.shape == (nmaps, nzbins * ncorrelations, nbins)
    assert z_at.shape == (nmaps, nzbins * ncorrelations, nbins)

    expected = np.zeros_like(z_gt)
    out_idx = 0
    for z_center in range(nzbins):
        center_vals = center[:, z_center, :]
        mean_center = np.mean(center_vals, axis=1)
        for pair_idx in range(ncorrelations):
            annulus_vals = xi_t[:, pair_idx, :, :]
            mean_annulus = np.mean(annulus_vals, axis=1)
            mean_product = np.mean(center_vals[:, :, None] * annulus_vals, axis=1)
            expected[:, out_idx, :] = mean_product - mean_center[:, None] * mean_annulus
            out_idx += 1

    np.testing.assert_allclose(z_gt, expected)
    np.testing.assert_allclose(z_at, expected)


def test_zeta_t_validation_raises_on_central_ndim():
    center = np.zeros((2, 4))
    xi_t = np.zeros((2, 3, 4, 2))
    with pytest.raises(ValueError):
        zeta_g_t(center, xi_t)


def test_zeta_t_validation_raises_on_annulus_ndim():
    center = np.zeros((2, 2, 4))
    xi_t = np.zeros((2, 3, 4))
    with pytest.raises(ValueError):
        zeta_g_t(center, xi_t)


def test_zeta_t_validation_raises_on_map_count_mismatch():
    center = np.zeros((2, 2, 4))
    xi_t = np.zeros((3, 5, 4, 2))
    with pytest.raises(ValueError):
        zeta_g_t(center, xi_t)


def test_zeta_t_validation_raises_on_patch_count_mismatch():
    center = np.zeros((2, 2, 4))
    xi_t = np.zeros((2, 5, 6, 2))
    with pytest.raises(ValueError):
        zeta_g_t(center, xi_t)



# ---------------------------------------------------------------------------
# Aperture filter functions
# ---------------------------------------------------------------------------

def test_Q_crittenden_formula():
    theta_Q_arcmin = 90.0
    theta_ap = np.radians(theta_Q_arcmin / 60)
    theta = np.linspace(0.0, 5 * theta_ap, 64)
    expected = theta**2 / (4 * np.pi * theta_ap**4) * np.exp(
        -(theta**2) / (2 * theta_ap**2)
    )
    np.testing.assert_allclose(Q_crittenden(theta, theta_Q_arcmin), expected)
    # peak at theta = sqrt(2) * theta_ap
    fine = np.linspace(0.5 * theta_ap, 3 * theta_ap, 20001)
    peak = fine[np.argmax(Q_crittenden(fine, theta_Q_arcmin))]
    np.testing.assert_allclose(peak, np.sqrt(2) * theta_ap, rtol=1e-3)


def test_Q_schneider_formula_and_support():
    theta_Q_arcmin = 90.0
    theta_ap = np.radians(theta_Q_arcmin / 60)
    x = np.array([0.0, 0.25, 1 / np.sqrt(2), 0.9, 1.0, 1.5, 4.0])
    values = Q_schneider(x * theta_ap, theta_Q_arcmin)

    inside = x < 1.0
    expected = 6.0 / (np.pi * theta_ap**2) * x[inside] ** 2 * (1 - x[inside] ** 2)
    np.testing.assert_allclose(values[inside], expected)
    # compact support: identically zero at and beyond theta_Q
    np.testing.assert_array_equal(values[~inside], 0.0)
    assert values[0] == 0.0
    # peak at x = 1/sqrt(2) with value 1.5/(pi theta_ap^2)
    np.testing.assert_allclose(
        Q_schneider(theta_ap / np.sqrt(2), theta_Q_arcmin),
        1.5 / (np.pi * theta_ap**2),
    )


@pytest.mark.parametrize("filter_fn", [Q_crittenden, Q_schneider])
def test_filters_share_unit_normalisation(filter_fn):
    """Both filters obey the same convention: ∫ Q(θ) dΩ = 2π ∫ Q θ dθ = 1,
    so aperture masses measured with either are directly comparable."""
    theta_Q_arcmin = 90.0
    theta_ap = np.radians(theta_Q_arcmin / 60)
    theta = np.linspace(0.0, 8 * theta_ap, 400001)
    q = np.asarray(filter_fn(theta, theta_Q_arcmin), dtype=np.float64)
    # np.trapezoid is the numpy>=2 name; fall back for numpy 1.x
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    integral = 2 * np.pi * trapezoid(q * theta, theta)
    np.testing.assert_allclose(integral, 1.0, rtol=1e-5)


def _loop_zeta_cross_generic(center, annulus):
    """The per-(z_center, combination) loop the generic γ_t branch replaced."""
    nmaps, nzbins, _ = center.shape
    n_correlations, nbins = annulus.shape[1], annulus.shape[3]
    out = np.zeros((nmaps, nzbins * n_correlations, nbins))
    k = 0
    for z_center in range(nzbins):
        c = center[:, z_center, :]
        mean_c = np.mean(c, axis=1)
        for pair_idx in range(n_correlations):
            a = annulus[:, pair_idx, :, :]
            out[:, k, :] = np.mean(c[:, :, None] * a, axis=1) - mean_c[
                :, None
            ] * np.mean(a, axis=1)
            k += 1
    return out


@pytest.mark.parametrize(
    "nmaps, nzbins, npatches, nbins", [(1, 4, 450, 4), (3, 5, 97, 8), (2, 1, 13, 3)]
)
def test_vectorised_reduction_is_bitwise_the_per_triplet_loop(
    nmaps, nzbins, npatches, nbins
):
    """The triplet loop is the definition; vectorising it must not move a bit.

    ζ is a difference of two nearly equal means, so a reordered summation
    shows up in the low bits of a cancelling quantity.  On numpy the gathered
    form reduces the same axis in the same order, so the gate is exact rather
    than a tolerance.
    """
    rng = np.random.default_rng(11)
    npairs = nzbins * (nzbins + 1) // 2
    center = rng.normal(size=(nmaps, nzbins, npatches))
    annulus = rng.normal(size=(nmaps, npairs, npatches, nbins))

    np.testing.assert_array_equal(
        zeta_g_plus(center, annulus), _reference_zeta(center, annulus)
    )
    # the upper-triangular branch of the γ_t path must agree with it exactly
    np.testing.assert_array_equal(
        zeta_g_t(center, annulus), _reference_zeta(center, annulus)
    )

    generic = rng.normal(size=(nmaps, npairs + 3, npatches, nbins))
    np.testing.assert_array_equal(
        zeta_g_t(center, generic), _loop_zeta_cross_generic(center, generic)
    )


def test_reduction_does_not_depend_on_how_many_map_sets_are_stacked():
    """``ZetaWriter`` reduces one map-set at a time on the device route and a
    stacked batch on the host route; both must land on the same numbers."""
    nmaps, nzbins, npatches, nbins = 5, 4, 200, 4
    npairs = nzbins * (nzbins + 1) // 2
    rng = np.random.default_rng(12)
    center = rng.normal(size=(nmaps, nzbins, npatches))
    annulus = rng.normal(size=(nmaps, npairs, npatches, nbins))

    batched = calculate_all_zetas(M_a=center, xi_p=annulus)
    one_at_a_time = {
        key: np.concatenate(
            [
                calculate_all_zetas(
                    M_a=center[i : i + 1], xi_p=annulus[i : i + 1]
                )[key]
                for i in range(nmaps)
            ]
        )
        for key in batched
    }
    for key, expected in batched.items():
        np.testing.assert_array_equal(one_at_a_time[key], expected, err_msg=key)


def test_triplet_indices_follow_combinations_with_replacement():
    """The output row order is API: it is what ``ZetaWriter`` stores and what
    a stored data vector is indexed by."""
    from CosmoFuse.correlation_helpers import _triplet_indices

    for nzbins in (1, 2, 4, 5):
        indices = _triplet_indices(nzbins)
        combs = list(itertools.combinations_with_replacement(range(nzbins), 3))
        assert len(indices.centers) == len(combs)
        for k, (z_center, z2, z3) in enumerate(combs):
            assert indices.centers[k] == z_center
            assert indices.pairs[k] == _get_pair_index(nzbins, z2, z3)


def _per_estimator(**fields):
    """What ``calculate_all_zetas`` does one estimator at a time."""
    symmetric = fields.pop("xi_t_symmetric", None)
    M_g, M_a = fields.get("M_g"), fields.get("M_a")
    xi_p, xi_m = fields.get("xi_p"), fields.get("xi_m")
    xi_g, xi_t = fields.get("xi_g"), fields.get("xi_t")
    out = {}
    if M_g is not None and xi_p is not None:
        out["zeta_g_plus"] = zeta_g_plus(M_g, xi_p)
    if M_g is not None and xi_m is not None:
        out["zeta_g_minus"] = zeta_g_minus(M_g, xi_m)
    if M_a is not None and xi_p is not None:
        out["zeta_a_plus"] = zeta_a_plus(M_a, xi_p)
    if M_a is not None and xi_m is not None:
        out["zeta_a_minus"] = zeta_a_minus(M_a, xi_m)
    if M_g is not None and xi_g is not None:
        out["zeta_g_g"] = zeta_g_g(M_g, xi_g)
    if M_a is not None and xi_g is not None:
        out["zeta_a_g"] = zeta_a_g(M_a, xi_g)
    if M_g is not None and xi_t is not None:
        out["zeta_g_t"] = zeta_g_t(M_g, xi_t, symmetric=symmetric)
    if M_a is not None and xi_t is not None:
        out["zeta_a_t"] = zeta_a_t(M_a, xi_t, symmetric=symmetric)
    return out


@pytest.mark.parametrize(
    "nzbins, npatches, nbins, n_xi_t",
    [(4, 40, 10, 16), (4, 40, 10, 10), (3, 17, 5, 9), (2, 11, 4, 6), (1, 7, 3, 1)],
)
def test_batched_reduction_equals_one_call_per_estimator(
    nzbins, npatches, nbins, n_xi_t
):
    """All eight estimators share one gather; that must not move a bit.

    ``n_xi_t == nz(nz+1)/2`` is the case where the γ_t layout is inferred as
    the symmetric triangle, so it exercises both index plans in one batch.
    """
    rng = np.random.default_rng(23)
    npairs = nzbins * (nzbins + 1) // 2
    fields = dict(
        M_g=rng.normal(size=(1, nzbins, npatches)),
        M_a=rng.normal(size=(1, nzbins, npatches)),
        xi_p=rng.normal(size=(1, npairs, npatches, nbins)),
        xi_m=rng.normal(size=(1, npairs, npatches, nbins)),
        xi_g=rng.normal(size=(1, npairs, npatches, nbins)),
        xi_t=rng.normal(size=(1, n_xi_t, npatches, nbins)),
    )
    batched = calculate_all_zetas(**fields)
    expected = _per_estimator(**fields)
    assert set(batched) == set(expected)
    for key, want in expected.items():
        assert batched[key].shape == want.shape, key
        np.testing.assert_array_equal(batched[key], want, err_msg=key)


def test_batching_falls_back_rather_than_promoting_a_dtype():
    """Concatenating mixed dtypes would silently raise an output's precision.

    ``zeta_g_plus`` is float32 x float32; batching it with a float64 ``xi_m``
    would make it float64.  The batch is skipped instead.
    """
    fields = dict(
        M_g=np.zeros((1, 2, 7), dtype=np.float32),
        M_a=np.zeros((1, 2, 7), dtype=np.float32),
        xi_p=np.zeros((1, 3, 7, 2), dtype=np.float32),
        xi_m=np.zeros((1, 3, 7, 2), dtype=np.float64),
    )
    mixed = calculate_all_zetas(**fields)
    assert mixed["zeta_g_plus"].dtype == zeta_g_plus(
        fields["M_g"], fields["xi_p"]
    ).dtype

    fields["xi_m"] = fields["xi_m"].astype(np.float32)
    uniform = calculate_all_zetas(**fields)
    assert uniform["zeta_g_plus"].dtype == np.float32


def test_a_single_estimator_still_works():
    rng = np.random.default_rng(24)
    M_g = rng.normal(size=(1, 2, 9))
    xi_p = rng.normal(size=(1, 3, 9, 4))
    only = calculate_all_zetas(M_g=M_g, xi_p=xi_p)
    assert set(only) == {"zeta_g_plus"}
    np.testing.assert_array_equal(only["zeta_g_plus"], zeta_g_plus(M_g, xi_p))
