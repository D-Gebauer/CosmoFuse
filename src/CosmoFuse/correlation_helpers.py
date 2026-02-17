import itertools
from typing import Tuple

import numpy as np


def M_a_patch(
    Q_inds: np.ndarray,
    Q_cos: np.ndarray,
    Q_sin: np.ndarray,
    Q_val: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    Q_w: np.ndarray,
    Q_patch_area: float,
) -> float:
    """Calculate aperture mass for a patch.

    Args:
        Q_inds: Indices of pixels in the filter
        Q_cos: Cosine values for filter
        Q_sin: Sine values for filter
        Q_val: Filter values
        g1: First component of shear
        g2: Second component of shear
        Q_w: Weights
        Q_patch_area: Area of the patch

    Returns:
        Aperture mass value
    """
    gt = -g1[Q_inds] * Q_cos - g2[Q_inds] * Q_sin
    M_a_Re = Q_patch_area * np.sum(Q_w[Q_inds] * gt * Q_val) / np.sum(Q_w[Q_inds])

    return M_a_Re


def Q_T(theta: float, theta_Q: float = 90) -> float:
    """The compensated filter used for aperture mass.

    Args:
        theta (float): Great Circle distance to center of filter.

    Returns:
        (float): Value of compensated filter.
    """

    theta_Q = np.radians(theta_Q / 60)
    return theta**2 / (4 * np.pi * theta_Q**4) * np.exp(-(theta**2) / (2 * theta_Q**2))


def zeta(
    M_ap: np.ndarray, xip: np.ndarray, xim: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the i3PCF from aperture mass and local 2PCF.

    Args:
        M_ap (float): Aperture mass, shape:(nmaps, n_zbins, n_patches)
        xip (float): Shear 2PCF, shape:(nmaps, n_correlations, n_patches, nbins)
        xim (float): Shear 2PCF, shape:(nmaps, n_correlations, n_patches, nbins)

    Returns:
        (float): The i3PCF zetap & zetam.
    """

    nmaps = M_ap.shape[0]
    zbins = M_ap.shape[1]
    nbins = xip.shape[3]
    zbin_combs = np.array(
        list(itertools.combinations_with_replacement(range(zbins), 2))
    )
    zeta_combs = np.array(
        list(itertools.combinations_with_replacement(range(zbins), 3))
    )

    zeta_2combs = list(range(len(zbin_combs)))
    min_idx = 0
    for i in range(zbins - 1):
        min_idx += zbins - i
        for j in range(min_idx, len(zbin_combs)):
            zeta_2combs.append(j)

    zetap = np.zeros((nmaps, len(zeta_combs), nbins))
    zetam = np.zeros((nmaps, len(zeta_combs), nbins))

    for i, (z1, z2) in enumerate(zip(zeta_combs[:, 0], zeta_2combs)):
        zetap[:, i, :] = np.mean(M_ap[:, z1, :, None] * xip[:, z2], axis=1) - np.mean(
            M_ap[:, z1, :, None], axis=1
        ) * np.mean(xip[:, z2], axis=1)
        zetam[:, i, :] = np.mean(M_ap[:, z1, :, None] * xim[:, z2], axis=1) - np.mean(
            M_ap[:, z1, :, None], axis=1
        ) * np.mean(xim[:, z2], axis=1)

    return zetap, zetam
