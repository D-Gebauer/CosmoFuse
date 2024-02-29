# CosmoFuse

A package for efficiently measuring integrated 3-point correlation functions on GPU/CPU written in Python 3.

The integrated 3-point correlation function probes squeezed configurations of the bispectrum without the computational expense of the full 3-point correlation function.

The shear i3PCF is calculated with 

$$ \zeta_{\pm} = \langle M_{ap} \xi_{\pm} \rangle $$

where $M_{ap}$ is the aperture mass and $\xi_{\pm}$ are the 2PCFs. These are measured in patches. On a pixelised map the aperture mass is calculated as

$$ M_{ap} = \frac{A \sum_{\text{p}}{w_p g_t Q_p}}{\sum_{\text{p}}{w_p}} $$

where $A$ is the patch area, $w_p$ is the pixel's weight, $g_t$ is the tangential shear (relative to the patch center), and $Q_p$ is the value of the compensated filter evaluated at the pixel position.

The 2PCFs are calculated as:

$$ \xi_+ = \frac{\sum_{\text{pairs}}{w_1 w_2 g_1 g_2^*}}{\sum_{\text{pairs}}{w_1 w_2}} $$

$$ \xi_- = \frac{\sum_{\text{pairs}}{w_1 w_2 g_1 g_2}}{\sum_{\text{pairs}}{w_1 w_2}} $$

where $g_1$ and $g_2$ are the complex shear values rotated relative to the 2 positions ($g_i = g_x + i g_t$).

## Features

- Calculate pairs for given mask & resolution once
- Save/Load pairs using hdf5 files
- Reuse pairs to measure i3PCF across maps

## Installation
Install using:

    pip install git+https://github.com/D-Gebauer/CosmoFuse/tree/main

Note: to use full functionality including correlation_GPU, Cupy has to be installed additionally.

## USAGE

First create a Correlation object:

    from CosmoFuse.correlations import Correlation_GPU
    correlation = Correlation_GPU(nside,                        # resolution of healpy maps
                                 phi_center, theta_center,      # center of patches (in radians)
                                 patch_size=90,                 # size of each patch (in arcminutes)
                                 theta_Q=90                     # Size of compensated filter
                                 nbins=10,                      # Number of angular bins
                                 theta_min=10, theta_max=170,   # Minimum and maximum angular seperation (in arcminutes)
                                 mask=mask,                     # Mask
                                 fastmath=False)                # Whether to use fastmath in jit compiled functions

Then Calculate pairs:

    correlation.preprocess(threads=100) # Calculation of 2PCF pairs can be multithreaded

These can be saved & loaded using:

    correlation.save_pairs("/path/to/pairs.h5")
    correlation.load("/path/to/pairs.h5")

To then measure the i3PCF using GPU:

    correlation.prepare_gpu()
    correlation.load_maps(g11, g21, g12, g22, w1, w2)
    M_ap = correlation.get_M_a(g1, g2)
    xip, xim = correlation.get_all_xipm()

Or directly for all bin combinations:

    correlation.prepare_gpu()
    M_ap, xip, xim = correlation.get_full_tomo(shear_maps, w, sumofweights)

These (in the tomographic case) can be converted to $\zeta_+$ & $\zeta_-$:

    from CosmoFuse.correlation_helpers import zeta
    zetap, zetam = zeta(M_ap, xip, xim)

