[![CI](https://github.com/D-Gebauer/CosmoFuse/actions/workflows/ci.yml/badge.svg)](https://github.com/D-Gebauer/CosmoFuse/actions/workflows/ci.yml)

[![codecov](https://codecov.io/github/D-Gebauer/CosmoFuse/graph/badge.svg?token=F4JC08UEJP)](https://codecov.io/github/D-Gebauer/CosmoFuse)



# CosmoFuse

A package for efficiently measuring integrated 3-point correlation functions on GPU/CPU written in Python 3.

The integrated 3-point correlation function probes squeezed configurations of the bispectrum without the computational expense of the full 3-point correlation function.

## i3PCF Notation (Halder et al. 2023)

- $g$: galaxy density / aperture number-count field at the center
- $a$: aperture mass field at the center
- $+$ / $-$: shear 2PCFs, $\xi_+$ and $\xi_-$, on the annulus
- $t$: tangential shear $\gamma_t$ on the annulus

Supported i3PCFs (counting $\pm$ separately):

- $\zeta_{g,+}$
- $\zeta_{g,-}$
- $\zeta_{a,+}$
- $\zeta_{a,-}$
- $\zeta_{g,g}$
- $\zeta_{a,g}$
- $\zeta_{g,t}$
- $\zeta_{a,t}$

### 1. Cosmic Shear ($\zeta_\pm$)

The shear i3PCF is calculated as the correlation between the aperture mass $M_{ap}$ and the shear 2PCFs $\xi_\pm$:

$$ \zeta_{\pm} = \langle M_{ap} \xi_{\pm} \rangle $$

On a pixelated map, the aperture mass is calculated as:

$$ M_{ap} = \frac{A \sum_{\text{p}}{w_p g_t Q_p}}{\sum_{\text{p}}{w_p}} $$

where $g_t$ is the tangential shear. The shear 2PCFs are calculated as:

$$ \xi_+ = \frac{\sum_{\text{pairs}}{w_1 w_2 g_1 g_2^*}}{\sum_{\text{pairs}}{w_1 w_2}}, \quad \xi_- = \frac{\sum_{\text{pairs}}{w_1 w_2 g_1 g_2}}{\sum_{\text{pairs}}{w_1 w_2}} $$

### 2. Galaxy Clustering ($\zeta_{clust}$)

The clustering i3PCF is the correlation between aperture number counts $N_{ap}$ and angular clustering $w(\theta)$:

$$ \zeta_{clust} = \langle N_{ap} w(\theta) \rangle $$

The aperture number count is given by:

$$ N_{ap} = \frac{A \sum_{\text{p}}{w_p \delta_g Q_p}}{\sum_{\text{p}}{w_p}} $$

where $\delta_g$ is the galaxy overdensity (or counts). The angular clustering is:

$$ w(\theta) = \frac{\sum_{\text{pairs}}{w_1 w_2 \delta_1 \delta_2}}{\sum_{\text{pairs}}{w_1 w_2}} $$

### 3. Galaxy-Galaxy Lensing ($\zeta_{ggl}$)

The GGL i3PCF couples aperture number counts (lenses) with tangential shear (sources):

$$ \zeta_{ggl} = \langle N_{ap} \gamma_t(\theta) \rangle $$

where $N_{ap}$ is calculated on the lens map as above. The tangential shear is:

$$ \gamma_t(\theta) = \frac{\sum_{\text{pairs}}{w_l w_s \delta_l g_{s,t}}}{\sum_{\text{pairs}}{w_l w_s}} $$

where $\delta_l$ is the lens overdensity and $g_{s,t}$ is the source tangential shear relative to the lens.

## Features

- Calculate pairs for given mask & resolution once
- Save/Load pairs using hdf5 files
- Reuse pairs to measure i3PCFs across maps
- Optimized backend kernels for Spin-0×Spin-0 ($w(\theta)$), Spin-0×Spin-2 ($\gamma_t$), and Spin-2×Spin-2 ($\xi_\pm$) workloads
- Optimized tomographic kernels for all probes on CPU and GPU

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

### Measuring Correlations

The package supports 3 main probes: Cosmic Shear, Galaxy Clustering, and Galaxy-Galaxy Lensing (GGL).

#### Probes & Inputs

| Probe | Aperture Quantity | 2-Point Correlation | Inputs |
| :--- | :--- | :--- | :--- |
| **Shear** | Aperture Mass ($M_a$) | Shear 2PCF ($\xi_\pm$) | Shear maps ($g_1, g_2$), Weights ($w$) |
| **Clustering** | Aperture Count ($M_g$) | Angular Clustering ($\xi_g$) | Density maps ($\delta$ or counts), Weights ($w$) |
| **GGL** | Aperture Count ($M_g$) | Tangential Shear ($\xi_t$) | Lens density + Source shear |

#### 1. Single Map Pair (Patch-Level)

Calculate quantities for a single pair of maps (e.g. one tomographic bin pair).

**Cosmic Shear**:
```python
# Aperture Mass
M_a = correlation.get_aperture_shear(g1, g2, w)
# Shear 2PCF
xi_p, xi_m = correlation.compute_shear_shear(g1_a, g2_a, g1_b, g2_b, w_a, w_b)
```

**Galaxy Clustering**:
```python
# Aperture Number Count
M_g = correlation.get_aperture_density(delta, w)
# Angular Clustering
xi_g, = correlation.compute_density_density(delta_a, delta_b, w_a, w_b)
```

**Galaxy-Galaxy Lensing**:
```python
# Aperture Number Count (Lenses)
M_g = correlation.get_aperture_density(delta_lens, w_lens)
# Tangential Shear
xi_t, = correlation.compute_density_shear(delta_lens, g1_source, g2_source, w_lens, w_source)
```

#### 2. Full Tomography (3x2pt)

Calculate all correlations for all requested tomographic bin combinations at once.

**Specific Probes**:

*Cosmic Shear*:
```python
# Returns: xi_p, xi_m
xi_p, xi_m = correlation.vectorized_shear_shear(shear_maps, weights)

# Full shear tomography (includes aperture mass)
M_a, xi_p, xi_m = correlation.get_full_tomo_shear(shear_maps, weights)
```

*Galaxy Clustering*:
```python
# Returns: xi_g
xi_g = correlation.vectorized_density_density(density_maps, weights)

# Full clustering tomography (includes aperture counts)
M_g, xi_g = correlation.get_full_tomo_density(density_maps, weights)
```

*Galaxy-Galaxy Lensing*:
```python
# Returns: xi_t (Lens->Source combinations)
xi_t = correlation.vectorized_density_shear(
    density_maps, shear_maps, density_weights, shear_weights
)

# Full GGL helper; by default returns only xi_t
xi_t = correlation.get_full_tomo_ggl(
    density_maps, shear_maps, density_weights, shear_weights
)

# Optional extras: also return M_g and/or M_a
xi_t, M_g = correlation.get_full_tomo_ggl(
    density_maps, shear_maps, density_weights, shear_weights,
    return_N_ap=True,
)
xi_t, M_g, M_a = correlation.get_full_tomo_ggl(
    density_maps, shear_maps, density_weights, shear_weights,
    return_N_ap=True,
    return_M_ap=True,
)
```

**Combined 3x2pt Bundle**:

```python
# shear_maps:   [nzbins_s, 2, npix] or None
# density_maps: [nzbins_d, npix] or None
# weights:      dict of weights or None

M_a, M_g, xi_p, xi_m, xi_g, xi_t = correlation.get_3x2pt_tomo(
    shear_maps=shear_maps,
    density_maps=density_maps,
    weights={"shear": shear_w, "density": density_w},
)
```

## Calculating i3PCFs

The 8 i3PCFs can be computed with `CosmoFuse.correlation_helpers`:

```python
from CosmoFuse.correlation_helpers import (
    zeta_g_plus, zeta_g_minus, zeta_a_plus, zeta_a_minus,
    zeta_g_g, zeta_a_g, zeta_g_t, zeta_a_t,
)

# Central fields (nmaps, nzbins, npatches)
# M_g: galaxy-density-like center field
# M_a: aperture-mass center field

# Annulus fields (nmaps, n_correlations, npatches, nbins)
# xi_p: xi_plus, xi_m: xi_minus, xi_g: galaxy auto-correlation, xi_t: tangential shear

zg_plus = zeta_g_plus(M_g, xi_p)
zg_minus = zeta_g_minus(M_g, xi_m)
za_plus = zeta_a_plus(M_a, xi_p)
za_minus = zeta_a_minus(M_a, xi_m)
zg_g = zeta_g_g(M_g, xi_g)
za_g = zeta_a_g(M_a, xi_g)
zg_t = zeta_g_t(M_g, xi_t)
za_t = zeta_a_t(M_a, xi_t)
```

Unified helper:

```python
from CosmoFuse.correlation_helpers import calculate_all_zetas

results = calculate_all_zetas(
    M_g=M_g,
    M_a=M_a,
    xi_p=xi_p,
    xi_m=xi_m,
    xi_g=xi_g,
    xi_t=xi_t,
)
```
