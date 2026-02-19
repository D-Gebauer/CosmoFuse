import numpy as np
import pytest
from CosmoFuse.correlation_helpers import (
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

