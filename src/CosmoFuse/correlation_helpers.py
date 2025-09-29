import itertools
import math
from typing import Tuple

import numpy as np
from numba import njit


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


def xipm_patch(
    inds: np.ndarray,
    exp2theta: np.ndarray,
    bin_inds: np.ndarray,
    g11: np.ndarray,
    g21: np.ndarray,
    g12: np.ndarray,
    g22: np.ndarray,
    nbins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate xi+ and xi- correlation functions for a patch.

    Args:
        inds: Pair indices
        exp2theta: Exponential phase factors
        bin_inds: Number of pairs per bin
        g11: First shear component of first map
        g21: Second shear component of first map
        g12: First shear component of second map
        g22: Second shear component of second map
        nbins: Number of angular bins

    Returns:
        Tuple of (xi+, xi-) correlation functions
    """
    xip, xim = np.zeros(nbins, dtype="c8"), np.zeros(nbins, dtype="c8")

    g1 = ((g11[inds[0]]) + 1j * g21[inds[0]]) * exp2theta[0]
    g2 = ((g12[inds[1]]) + 1j * g22[inds[1]]) * exp2theta[1]

    bin_edges = np.append([0], np.cumsum(bin_inds))

    for bin_idx in range(nbins):
        xip[bin_idx] = np.sum(
            g1[bin_edges[bin_idx] : bin_edges[bin_idx + 1]]
            * np.conjugate(g2[bin_edges[bin_idx] : bin_edges[bin_idx + 1]])
        ) / (bin_inds[bin_idx])
        xim[bin_idx] = np.sum(
            g1[bin_edges[bin_idx] : bin_edges[bin_idx + 1]]
            * g2[bin_edges[bin_idx] : bin_edges[bin_idx + 1]]
        ) / (bin_inds[bin_idx])

    return np.real(xip), np.real(xim)


@njit(fastmath=False)
def radec_to_xyz(ra: float, dec: float) -> Tuple[float, float, float]:
    """Convert ra, dec (in radians) to 3D x,y,z coordinates on the unit sphere.

    :param ra:      The right ascension(s) in radians. May be a numpy array.
    :param dec:     The declination(s) in radians. May be a numpy array.

    :returns: x, y, z as a tuple.
    """
    cosdec = np.cos(dec)
    x = cosdec * np.cos(ra)
    y = cosdec * np.sin(ra)
    z = np.sin(dec)
    return x, y, z


@njit(fastmath=False)
def getAngle(
    ra1: float,
    dec1: float,
    ra2: float,
    dec2: float,
    ra3: float = 0,
    dec3: float = np.pi / 2,
) -> float:
    """Find the open angle at location 1  between (ra2,dec2) and (ra3, dec3) (north pole by default).

    The current coordinate along with the two other coordinates form a spherical triangle
    on the sky.  This function calculates the angle between the two sides at the location of
    the current coordinate.

    Note that this returns a signed angle.  The angle is positive if the sweep direction from
    ``coord2`` to ``coord3`` is counter-clockwise (as observed from Earth).  It is negative if
    the direction is clockwise.

    :param ra1:       A second CelestialCoord
    :param dec1:       A second CelestialCoord
    :param ra2:       A third CelestialCoord
    :param dec2:       A second CelestialCoord

    :returns: the angle between the great circles joining the other two coordinates to the
                current coordinate.
    """
    # Call A = coord2, B = coord3, C = self
    # Then we are looking for the angle ACB.
    # If we treat each coord as a (x,y,z) vector, then we can use the following spherical
    # trig identities:
    #
    # (A x C) . B = sina sinb sinC
    # (A x C) . (B x C) = sina sinb cosC
    #
    # Then we can just use atan2 to find C, and atan2 automatically gets the sign right.
    # And we only need 1 trig call, assuming that x,y,z are already set up, which is often
    # the case.

    c1 = radec_to_xyz(ra1, dec1)
    c2 = radec_to_xyz(ra2, dec2)
    c3 = radec_to_xyz(ra3, dec3)

    a = np.array([[c3[0], c3[1], c3[2]], [c1[0], c1[1], c1[2]], [c2[0], c2[1], c2[2]]])

    sinC = (
        a[0][0] * (a[1][1] * a[2][2] - a[2][1] * a[1][2])
        - a[1][0] * (a[0][1] * a[2][2] - a[2][1] * a[0][2])
        + a[2][0] * (a[0][1] * a[1][2] - a[1][1] * a[0][2])
    )

    dsq_AC = (c1[0] - c3[0]) ** 2 + (c1[1] - c3[1]) ** 2 + (c1[2] - c3[2]) ** 2
    dsq_BC = (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2
    dsq_AB = (c2[0] - c3[0]) ** 2 + (c2[1] - c3[1]) ** 2 + (c2[2] - c3[2]) ** 2
    cosC = 0.5 * (dsq_AC + dsq_BC - dsq_AB - 0.5 * dsq_AC * dsq_BC)

    C = math.atan2(sinC, cosC)
    return C


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
