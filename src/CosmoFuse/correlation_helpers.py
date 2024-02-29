import numpy as np
from numba import njit
import math
import itertools

def M_a_patch(Q_inds, Q_cos, Q_sin, Q_val, g1, g2, Q_w, Q_patch_area):
    
    gt = - g1[Q_inds]*Q_cos - g2[Q_inds]*Q_sin
    M_a_Re = Q_patch_area * np.sum(Q_w[Q_inds]*gt*Q_val) / np.sum(Q_w[Q_inds])

    return M_a_Re


def xipm_patch(inds, exp2theta, bin_inds, g11, g21, g12, g22, nbins):

    xip, xim = np.zeros(nbins, dtype='c8'), np.zeros(nbins, dtype='c8')
    
    g1 = ((g11[inds[0]]) + 1j* g21[inds[0]]) * exp2theta[0]
    g2 = ((g12[inds[1]]) + 1j* g22[inds[1]]) * exp2theta[1]

    bin_edges = np.append([0], np.cumsum(bin_inds))
    
    for bin in range(nbins):
        xip[bin] = np.sum(g1[bin_edges[bin]:bin_edges[bin+1]] * np.conjugate(g2[bin_edges[bin]:bin_edges[bin+1]]))/(bin_inds[bin])
        xim[bin] = np.sum(g1[bin_edges[bin]:bin_edges[bin+1]] * g2[bin_edges[bin]:bin_edges[bin+1]])/(bin_inds[bin])

    return np.real(xip), np.real(xim)

@njit(fastmath=False)
def radec_to_xyz(ra, dec):
    """Convert ra, dec (in radians) to 3D x,y,z coordinates on the unit sphere.

    :param ra:      The right ascension(s) in radians. May be a numpy array.
    :param dec:     The declination(s) in radians. May be a numpy array.

    :returns: x, y, z as a tuple.
    """
    cosdec = np.cos(dec)
    x = cosdec * np.cos(ra)
    y = cosdec * np.sin(ra)
    z = np.sin(dec)
    return x,y,z


@njit(fastmath=False)
def getAngle(ra1, dec1, ra2, dec2, ra3=0, dec3=np.pi/2):
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

    a = np.array( [ [ c3[0], c3[1], c3[2] ],
                    [ c1[0], c1[1], c1[2] ],
                    [ c2[0], c2[1], c2[2] ] ])

    sinC =  (a[0][0] * (a[1][1] * a[2][2] - a[2][1] * a[1][2])
            -a[1][0] * (a[0][1] * a[2][2] - a[2][1] * a[0][2])
            +a[2][0] * (a[0][1] * a[1][2] - a[1][1] * a[0][2]))
    
    dsq_AC = (c1[0]-c3[0])**2 + (c1[1]-c3[1])**2 + (c1[2]-c3[2])**2
    dsq_BC = (c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2
    dsq_AB = (c2[0]-c3[0])**2 + (c2[1]-c3[1])**2 + (c2[2]-c3[2])**2
    cosC = 0.5 * (dsq_AC + dsq_BC - dsq_AB - 0.5 * dsq_AC * dsq_BC)

    C = math.atan2(sinC, cosC)
    return C


def Q_T(theta, theta_Q):
    """The compensated filter used for aperture mass.

    Args:
        theta (float): Great Circle distance to center of filter.

    Returns:
        (float): Value of compensated filter.
    """
    
    theta_Q = np.radians(theta_Q/60)
    return theta**2/(4*np.pi*theta_Q**4)*np.exp(-theta**2/(2*theta_Q**2))


def zeta(M_ap, xip, xim):
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
    zbin_combs = np.array(list(itertools.combinations_with_replacement(range(zbins),2)))
    zeta_combs = np.array(list(itertools.combinations_with_replacement(range(zbins),3)))
    
    zeta_2combs = list(range(len(zbin_combs)))
    min = 0
    for i in range(zbins-1):
        min += zbins-i
        for j in range(min, len(zbin_combs)):
            zeta_2combs.append(j)
    
    zetap = np.zeros((nmaps, len(zeta_combs), nbins))
    zetam = np.zeros((nmaps, len(zeta_combs), nbins))

    for i, (z1, z2) in enumerate(zip(zeta_combs[:,0], zeta_2combs)):
        zetap[:,i,:] = np.mean(M_ap[:,z1,:,None] * xip[:,z2], axis=1) - np.mean(M_ap[:,z1,:,None], axis=1) * np.mean(xip[:,z2], axis=1)
        zetam[:,i,:] = np.mean(M_ap[:,z1,:,None] * xim[:,z2], axis=1) - np.mean(M_ap[:,z1,:,None], axis=1) * np.mean(xim[:,z2], axis=1)

    return zetap, zetam