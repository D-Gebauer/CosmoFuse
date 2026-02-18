[![CI](https://github.com/D-Gebauer/CosmoFuse/actions/workflows/ci.yml/badge.svg)](https://github.com/D-Gebauer/CosmoFuse/actions/workflows/ci.yml)

[![codecov](https://codecov.io/github/D-Gebauer/CosmoFuse/graph/badge.svg?token=F4JC08UEJP)](https://codecov.io/github/D-Gebauer/CosmoFuse)



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
- Optimized scalar backend kernels for Spin-0×Spin-0 (`w(θ)`) and Spin-0×Spin-2 (`γ_t`) workloads
- Fused-reduction tomography kernels for scalar correlation paths on CPU and GPU

## Installation
Install using:

    pip install git+https://github.com/D-Gebauer/CosmoFuse.git

Note: for GPU execution, install CuPy in your environment.

## USAGE

First create a Correlation object:

    from CosmoFuse import Correlation
    correlation = Correlation(
        nside,                              # resolution of healpy maps
        phi_center, theta_center,           # patch centers (radians)
        patch_size=90,                      # patch size (arcminutes)
        theta_Q=90,                         # compensated filter scale (arcminutes)
        nbins=10,                           # number of angular bins
        theta_min=10, theta_max=170,        # angular range (arcminutes)
        mask=mask,                          # mask
        fastmath=False,                     # numba/cupy fastmath toggle
        device="auto",                      # "cpu", "gpu", "auto", or GPU id
        map_precision="float32",            # float32 / float64
        rotation_precision="float32",       # float32 / float64
        index_precision="uint32",           # uint32 / uint64
    )

Then Calculate pairs:

    correlation.preprocess()

If host RAM is limited for large runs, you can optionally release host-side pair arrays after preparing backend buffers:

    correlation.prepare(release_host_pairs=True)

This keeps only arrays needed for later computations in memory. Save pairs before releasing host arrays (or reload/recompute before saving again).

These can be saved & loaded using:

    correlation.save_pairs("/path/to/pairs.h5")
    correlation.load_pairs("/path/to/pairs.h5")

To measure patch-level 2PCFs directly for one map pair:
    
    M_ap = correlation.get_aperture_shear(g1, g2, w)
    xip, xim = correlation.compute_shear_shear(g11, g21, g12, g22, w1, w2)

For direct scalar/spin-mixed 2PCFs:

    wtheta, = correlation.compute_density_density(delta1, delta2, w1, w2)
    gamma_t, = correlation.compute_density_shear(delta_lens, g1_source, g2_source, w_lens, w_source)

For scalar (spin-0) fields (e.g. galaxy density or convergence), use:

    M_delta = correlation.get_aperture_density(delta_map, w)

Or directly for all tomographic bin combinations:

    # shear_maps shape: [nzbins, 2, npix]
    # w shape:          [nzbins, npix]
    M_ap, xip, xim = correlation.get_full_tomo(shear_maps, w)

For a direct 3x2pt tomographic bundle (with map-type dependent `None` outputs):

    # shear_maps:   [nzbins_s, 2, npix] or None
    # density_maps: [nzbins_d, npix] or None
    # weights can be:
    #   - None
    #   - {"shear": shear_weights, "density": density_weights}
    #   - (shear_weights, density_weights)
    M_ap, N_ap, xipm, wtheta, gammat = correlation.get_3x2pt_tomo(
        shear_maps=shear_maps,
        density_maps=density_maps,
        weights={"shear": shear_w, "density": density_w},
    )

These (in the tomographic case) can be converted to $\zeta_+$ & $\zeta_-$:

    from CosmoFuse.correlation_helpers import zeta
    zetap, zetam = zeta(M_ap, xip, xim)
