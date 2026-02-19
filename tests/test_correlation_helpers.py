import numpy as np
import pytest
from CosmoFuse.correlation_helpers import zeta_shear, zeta_clust, zeta_ggl, _get_pair_index, calculate_all_zetas

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

def test_zeta_shear_shape_and_logic():
    nmaps = 2
    nzbins = 2
    npatches = 10
    nbins = 5
    
    # 2 bins: pairs are (0,0), (0,1), (1,1) -> 3 pairs
    npairs = 3
    
    M_ap = np.random.randn(nmaps, nzbins, npatches)
    xip = np.random.randn(nmaps, npairs, npatches, nbins)
    xim = np.random.randn(nmaps, npairs, npatches, nbins)
    
    zetap, zetam = zeta_shear(M_ap, xip, xim)
    
    # Combinations of 3 from 2 bins: (0,0,0), (0,0,1), (0,1,1), (1,1,1) -> 4 combs 
    assert zetap.shape == (nmaps, 4, nbins)
    assert zetam.shape == (nmaps, 4, nbins)

def test_zeta_clust_shape():
    nmaps = 2
    nzbins = 2
    npatches = 10
    nbins = 5
    npairs = 3 # (0,0), (0,1), (1,1)
    
    N_ap = np.random.randn(nmaps, nzbins, npatches)
    w = np.random.randn(nmaps, npairs, npatches, nbins)
    
    zeta_c = zeta_clust(N_ap, w)
    
    # 4 combs for 2 bins
    assert zeta_c.shape == (nmaps, 4, nbins)

def test_zeta_ggl_shape():
    nmaps = 2
    lens_bins = 2
    source_bins = 2
    npatches = 10
    nbins = 5
    
    # GGL pairs:
    # L0S0, L0S1, L1S1 -> 3 pairs (if L<=S)
    nggl_pairs = 3 
    
    N_ap = np.random.randn(nmaps, lens_bins, npatches)
    gammat = np.random.randn(nmaps, nggl_pairs, npatches, nbins)
    
    zetag = zeta_ggl(N_ap, gammat, lens_bins, source_bins)
    
    # Output combs:
    # L0, (L0,S0) -> 0,0
    # L0, (L0,S1) -> 0,1
    # L0, (L1,S1) -> 0,2
    # L1, (L0,S0) -> 1,0
    # L1, (L0,S1) -> 1,1
    # L1, (L1,S1) -> 1,2
    # Total 2 * 3 = 6 combinations
    
    assert zetag.shape == (nmaps, 6, nbins)

def test_calculate_all_zetas():
    nmaps = 2
    nzbins = 2
    npatches = 10
    nbins = 5
    npairs = 3
    nggl_pairs = 3
    
    M_ap = np.random.randn(nmaps, nzbins, npatches)
    xip = np.random.randn(nmaps, npairs, npatches, nbins)
    xim = np.random.randn(nmaps, npairs, npatches, nbins)
    N_ap = np.random.randn(nmaps, nzbins, npatches)
    w = np.random.randn(nmaps, npairs, npatches, nbins)
    gammat = np.random.randn(nmaps, nggl_pairs, npatches, nbins)
    
    # Test partial inputs (only shear)
    res = calculate_all_zetas(M_ap=M_ap, xip=xip, xim=xim)
    assert 'zetap' in res
    assert 'zetam' in res
    assert 'zeta_clust' not in res
    
    # Test all inputs
    res_all = calculate_all_zetas(
        M_ap=M_ap, xip=xip, xim=xim,
        N_ap=N_ap, w=w,
        gammat=gammat, lens_bins=nzbins, source_bins=nzbins
    )
    assert 'zetap' in res_all
    assert 'zeta_clust' in res_all
    assert 'zeta_ggl' in res_all

