import numpy as np
from numba import njit

@njit(fastmath=True)
def rotate_shear(ra1, dec1, ra2, dec2):
    cos_vartheta = np.cos(ra1 - ra2)*np.cos(dec2)*np.cos(dec1) + np.sin(dec2)*np.sin(dec1)
    sin_vartheta = np.sqrt(1-cos_vartheta**2)
    cos_phi1 = np.sin(ra1 - ra2)*np.cos(dec1) / sin_vartheta
    sin_phi1 = (np.cos(dec1)*np.sin(dec2) - np.sin(dec1)*np.cos(dec2)*np.cos(ra1 - ra2)) / sin_vartheta
    cos_2phi = cos_phi1*cos_phi1 - sin_phi1*sin_phi1
    sin_2phi = 2*sin_phi1*cos_phi1
    
    return cos_2phi, sin_2phi

def xipm_patch_auto(inds, cos_sin_2_phi_1, cos_sin_2_phi_2, bin_inds, g11, g21, g12, g22, w1, w2, nbins):
    
    xip, xim = np.zeros(nbins, dtype=np.float32), np.zeros(nbins, dtype=np.float32)

    gt1 =  g11[inds[0]]*cos_sin_2_phi_1[0] - g21[inds[0]]*cos_sin_2_phi_1[1]
    gx1 =  g11[inds[0]]*cos_sin_2_phi_1[1] + g21[inds[0]]*cos_sin_2_phi_1[0]
    gt2 =  g12[inds[1]]*cos_sin_2_phi_2[0] - g22[inds[1]]*cos_sin_2_phi_2[1]
    gx2 =  g12[inds[1]]*cos_sin_2_phi_2[1] + g22[inds[1]]*cos_sin_2_phi_2[0]

    gt = gt1*gt2
    gx = gx1*gx2
    
    bin_edges = np.append([0], np.cumsum(bin_inds)).astype('int')
    
    for bin in range(nbins):
        xip[bin] = np.sum(gt[bin_edges[bin]:bin_edges[bin+1]]+gx[bin_edges[bin]:bin_edges[bin+1]])/(bin_inds[bin])
        xim[bin] = np.sum(gt[bin_edges[bin]:bin_edges[bin+1]]-gx[bin_edges[bin]:bin_edges[bin+1]])/(bin_inds[bin])

    return xip, xim

def xipm_patch_cross(inds, cos_sin_2_phi_1, cos_sin_2_phi_2, bin_inds, g11, g21, g12, g22, w1, w2, nbins):

    xip1, xim1 = xipm_patch_auto(inds, cos_sin_2_phi_1, cos_sin_2_phi_2, bin_inds, g11, g21, g12, g22, w1, w2, nbins)
    xip2, xim2 = xipm_patch_auto(inds, cos_sin_2_phi_2, cos_sin_2_phi_1, bin_inds, g12, g22, g11, g21, w2, w1, nbins)

    return xip1+xip2/2, xim1+xim2/2


def M_a_patch(Q_inds, Q_cos, Q_sin, Q_val, g1, g2, Q_w, Q_patch_area):
    
    gt = g1[Q_inds]*Q_cos - g2[Q_inds]*Q_sin
    gx = g1[Q_inds]*Q_sin + g2[Q_inds]*Q_cos

    M_a_Re = Q_patch_area * np.sum(Q_w*gt*Q_val) / np.sum(Q_w)

    return M_a_Re


@njit(fastmath=False)
def xipm_coord(inds, exp2theta, bin_inds, g11, g21, g12, g22, nbins):

    xip, xim = np.zeros(nbins, dtype='c8'), np.zeros(nbins, dtype='c8')
    
    g1 = ((g11[inds[0]]) + 1j* g21[inds[0]]) * exp2theta[0] # minus in front of all g1?
    g2 = ((g12[inds[1]]) + 1j* g22[inds[1]]) * exp2theta[1]

    bin_edges = np.append([0], np.cumsum(bin_inds))
    
    for bin in range(nbins):
        xip[bin] = np.sum(g1[bin_edges[bin]:bin_edges[bin+1]] * np.conjugate(g2[bin_edges[bin]:bin_edges[bin+1]]))/(bin_inds[bin])
        xim[bin] = np.sum(g1[bin_edges[bin]:bin_edges[bin+1]] * g2[bin_edges[bin]:bin_edges[bin+1]])/(bin_inds[bin])
        

    return np.real(xip), np.real(xim)

