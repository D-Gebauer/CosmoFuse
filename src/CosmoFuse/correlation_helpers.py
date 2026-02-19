import itertools
from typing import Tuple, Dict, Optional

import numpy as np


def Q_T(theta: float, theta_Q: float = 90) -> float:
    """The compensated filter used for aperture mass.

    Args:
        theta (float): Great Circle distance to center of filter.

    Returns:
        (float): Value of compensated filter.
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


def zeta_shear(
    M_ap: np.ndarray, xip: np.ndarray, xim: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the i3PCF from aperture mass and local 2PCF.

    Calculates the integrated 3-point correlation function (i3PCF) which is the
    covariance between the aperture mass M_ap in one bin and the 2-point correlation
    function xi+/- in two other bins.

    Args:
        M_ap (np.ndarray): Aperture mass, shape:(nmaps, n_zbins, n_patches)
        xip (np.ndarray): Shear 2PCF, shape:(nmaps, n_correlations, n_patches, nbins)
        xim (np.ndarray): Shear 2PCF, shape:(nmaps, n_correlations, n_patches, nbins)

    Returns:
        Tuple[np.ndarray, np.ndarray]: The i3PCF zetap & zetam.
                                       Shape: (nmaps, n_zbin_combinations, nbins)
    """
    nmaps = M_ap.shape[0]
    nzbins = M_ap.shape[1]
    nbins = xip.shape[3]
    
    # Generate all combinations of (z1, z2, z3)
    # z1 is for M_ap, (z2, z3) is for xi
    zeta_combs = list(itertools.combinations_with_replacement(range(nzbins), 3))
    n_zeta_combs = len(zeta_combs)

    zetap = np.zeros((nmaps, n_zeta_combs, nbins))
    zetam = np.zeros((nmaps, n_zeta_combs, nbins))

    for k, (z1, z2, z3) in enumerate(zeta_combs):
        # Find index for the pair (z2, z3) in the correlation vector
        pair_idx = _get_pair_index(nzbins, z2, z3)
        
        # Calculate covariance: <A * B> - <A><B>
        # A = M_ap[:, z1, :]
        # B = xi[:, pair_idx, :]
        
        mean_Map = np.mean(M_ap[:, z1, :], axis=1) # (nmaps,)
        mean_xip = np.mean(xip[:, pair_idx], axis=1) # (nmaps, nbins)
        mean_xim = np.mean(xim[:, pair_idx], axis=1) # (nmaps, nbins)
        
        term1_p = np.mean(M_ap[:, z1, :, None] * xip[:, pair_idx], axis=1)
        term1_m = np.mean(M_ap[:, z1, :, None] * xim[:, pair_idx], axis=1)
        
        zetap[:, k, :] = term1_p - mean_Map[:, None] * mean_xip
        zetam[:, k, :] = term1_m - mean_Map[:, None] * mean_xim

    return zetap, zetam


def zeta_clust(
    N_ap: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """Calculate the Clustering i3PCF.
    
    Covariance between Aperture Number Count (N_ap) and Angular Clustering (w).

    Args:
        N_ap (np.ndarray): Aperture number count, shape:(nmaps, n_zbins, n_patches)
        w (np.ndarray): Angular clustering, shape:(nmaps, n_correlations, n_patches, nbins)

    Returns:
        np.ndarray: The clustering i3PCF. Shape: (nmaps, n_zbin_combinations, nbins)
    """
    nmaps = N_ap.shape[0]
    nzbins = N_ap.shape[1]
    nbins = w.shape[3]
    
    # Generate all combinations of (z1, z2, z3)
    zeta_combs = list(itertools.combinations_with_replacement(range(nzbins), 3))
    n_zeta_combs = len(zeta_combs)

    zeta_c = np.zeros((nmaps, n_zeta_combs, nbins))

    for k, (z1, z2, z3) in enumerate(zeta_combs):
        pair_idx = _get_pair_index(nzbins, z2, z3)
        
        mean_Nap = np.mean(N_ap[:, z1, :], axis=1)
        mean_w = np.mean(w[:, pair_idx], axis=1)
        
        term1 = np.mean(N_ap[:, z1, :, None] * w[:, pair_idx], axis=1)
        
        zeta_c[:, k, :] = term1 - mean_Nap[:, None] * mean_w

    return zeta_c


def zeta_ggl(
    N_ap: np.ndarray, 
    gammat: np.ndarray, 
    lens_bins: int,
    source_bins: int
) -> np.ndarray:
    """Calculate the Galaxy-Galaxy Lensing i3PCF.
    
    Covariance between Aperture Number Count (N_ap) and GGL Shear (gamma_t).

    Args:
        N_ap (np.ndarray): Aperture number count (lenses), shape:(nmaps, n_lens_bins, n_patches)
        gammat (np.ndarray): GGL Shear, shape:(nmaps, n_ggl_pairs, n_patches, nbins)
        lens_bins (int): Number of lens bins
        source_bins (int): Number of source bins

    Returns:
        np.ndarray: The GGL i3PCF. Shape: (nmaps, n_combinations, nbins)
                    Output combinations are all (z_lens_Map, z_lens_GGL, z_source_GGL)
                    where z_lens_GGL <= z_source_GGL typically, but we iterate available pairs.
    """
    nmaps = N_ap.shape[0]
    nbins = gammat.shape[3]
    
    # GGL pairs are typically (lens, source) combinations.
    # Assuming gammat is ordered as:
    # (L0, S0), (L0, S1), ..., (L0, Sn), (L1, S1), ..., (Ln, Sn)
    # i.e., Loop Lens i: Loop Source j >= i
    # We need to reconstruct this mapping or assume inputs match a specific convention.
    
    # Let's iterate over ALL valid triples (z1, z2, z3) where:
    # z1 is index in N_ap (lens bin)
    # (z2, z3) corresponds to a GGL pair index.
    # z2 is lens bin for gammat, z3 is source bin for gammat.
    
    # First, let's map pair (z2, z3) to flat index.
    # We assume standard Upper Triangular ordering for pairs if lens_bins == source_bins
    # OR if they are different, we need to know the exact storage convention.
    # Assuming standard CosmoFuse/CosmoLike convention: lens < source usually, or lens <= source.
    # Here we assume the input `gammat` contains pairs generated by:
    # for i in range(lens_bins): for j in range(source_bins): (maybe with j>=i condition?)
    
    # To keep it generic to the likely usage in CosmoFuse (based on correlations.py):
    # It likely computes all requested pairs.
    # Let's assume (Lens, Source) pairs are stored in the same order as `_get_pair_index` 
    # if lens_bins == source_bins, or a full rectangle if not.
    # Given the previous context, let's assume `_get_pair_index` logic applies 
    # (upper triangular) for lens <= source.
    
    # We will generate (z1, z2, z3) where:
    # z1 in range(lens_bins)
    # z2 in range(lens_bins)
    # z3 in range(source_bins)
    # AND z2 <= z3 (to match typical GGL storage)
    
    zeta_combinations = []
    pair_indices = []
    
    current_pair_idx = 0
    # Assuming stored as: for i in 0..lens: for j in 0..source: if j >= i: ...
    # This matches `_get_pair_index` if lens_bins == source_bins.
    # If lens_bins != source_bins, `_get_pair_index` might not be correct if it assumes square symmetric matrix.
    # But typically GGL uses the same bins for lenses and sources in 3x2pt, or distinct.
    # If distinct, the ordering is usually full rectangle? 
    # Let's proceed with the assumption of upper triangular (z2 <= z3) 
    # matching the correlation_helpers.py context.
    
    idx_map = {}
    k = 0
    for i in range(lens_bins):
        start_j = i # assume Source >= Lens
        for j in range(start_j, source_bins):
            idx_map[(i, j)] = k
            k += 1
            
    # Triples: (Map_Bin, GGL_Lens_Bin, GGL_Source_Bin)
    results = []
    
    zetag = np.zeros((nmaps, lens_bins * k, nbins)) # approximate size, refined below
    
    out_idx = 0
    for z1 in range(lens_bins):
        for z2 in range(lens_bins):
            for z3 in range(source_bins):
                if (z2, z3) in idx_map:
                    pair_idx = idx_map[(z2, z3)]
                    
                    mean_Nap = np.mean(N_ap[:, z1, :], axis=1)
                    mean_gammat = np.mean(gammat[:, pair_idx], axis=1)
                    
                    term1 = np.mean(N_ap[:, z1, :, None] * gammat[:, pair_idx], axis=1)
                    
                    zetag[:, out_idx, :] = term1 - mean_Nap[:, None] * mean_gammat
                    out_idx += 1
                    
    return zetag[: , :out_idx, :]


def calculate_all_zetas(
    M_ap: Optional[np.ndarray] = None,
    xip: Optional[np.ndarray] = None,
    xim: Optional[np.ndarray] = None,
    N_ap: Optional[np.ndarray] = None,
    w: Optional[np.ndarray] = None,
    gammat: Optional[np.ndarray] = None,
    lens_bins: Optional[int] = None,
    source_bins: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Calculate all integrated 3-point correlation functions (i3PCFs).

    Args:
        M_ap: Aperture mass (for zeta_shear).
        xip: Shear 2PCF + (for zeta_shear).
        xim: Shear 2PCF - (for zeta_shear).
        N_ap: Aperture number count (for zeta_clust and zeta_ggl).
        w: Angular clustering (for zeta_clust).
        gammat: GGL Shear (for zeta_ggl).
        lens_bins: Number of lens bins (for zeta_ggl).
        source_bins: Number of source bins (for zeta_ggl).

    Returns:
        Dict[str, np.ndarray]: Dictionary containing computed zetas keys:
            - 'zetap'
            - 'zetam'
            - 'zeta_clust'
            - 'zeta_ggl'
            If inputs for a specific zeta are missing, that key will not be present.
    """
    results = {}

    if M_ap is not None and xip is not None and xim is not None:
        zp, zm = zeta_shear(M_ap, xip, xim)
        results['zetap'] = zp
        results['zetam'] = zm

    if N_ap is not None and w is not None:
        zc = zeta_clust(N_ap, w)
        results['zeta_clust'] = zc

    if N_ap is not None and gammat is not None and lens_bins is not None and source_bins is not None:
        zg = zeta_ggl(N_ap, gammat, lens_bins, source_bins)
        results['zeta_ggl'] = zg

    return results
