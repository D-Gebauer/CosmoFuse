[![CI](https://github.com/D-Gebauer/CosmoFuse/actions/workflows/ci.yml/badge.svg)](https://github.com/D-Gebauer/CosmoFuse/actions/workflows/ci.yml)

[![codecov](https://codecov.io/github/D-Gebauer/CosmoFuse/graph/badge.svg?token=F4JC08UEJP)](https://codecov.io/github/D-Gebauer/CosmoFuse)



# CosmoFuse

A package for efficiently measuring integrated 3-point correlation functions on GPU/CPU written in Python 3.

The integrated 3-point correlation function probes squeezed configurations of the bispectrum without the computational expense of the full 3-point correlation function.

### 1. Cosmic Shear ($\zeta_\pm$)

The shear i3PCF is calculated as the covariance between the aperture mass $M_{ap}$ and the shear 2PCFs $\xi_\pm$:

$$ \zeta_{\pm} = \langle M_{ap} \xi_{\pm} \rangle $$

On a pixelated map, the aperture mass is calculated as:

$$ M_{ap} = \frac{A \sum_{\text{p}}{w_p g_t Q_p}}{\sum_{\text{p}}{w_p}} $$

where $g_t$ is the tangential shear. The shear 2PCFs are calculated as:

$$ \xi_+ = \frac{\sum_{\text{pairs}}{w_1 w_2 g_1 g_2^*}}{\sum_{\text{pairs}}{w_1 w_2}}, \quad \xi_- = \frac{\sum_{\text{pairs}}{w_1 w_2 g_1 g_2}}{\sum_{\text{pairs}}{w_1 w_2}} $$

### 2. Galaxy Clustering ($\zeta_{clust}$)

The clustering i3PCF is the covariance between aperture number counts $N_{ap}$ and angular clustering $w(\theta)$:

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
| **Shear** | Aperture Mass ($M_{ap}$) | Shear 2PCF ($\xi_\pm$) | Shear maps ($g_1, g_2$), Weights ($w$) |
| **Clustering** | Aperture Count ($N_{ap}$) | Angular Clustering ($w(\theta)$) | Density maps ($\delta$ or counts), Weights ($w$) |
| **GGL** | Aperture Count ($N_{ap}$) | Tangential Shear ($\gamma_t$) | Lens density + Source shear |

#### 1. Single Map Pair (Patch-Level)

Calculate quantities for a single pair of maps (e.g. one tomographic bin pair).

**Cosmic Shear**:
```python
# Aperture Mass
M_ap = correlation.get_aperture_shear(g1, g2, w)
# Shear 2PCF
xip, xim = correlation.compute_shear_shear(g1_a, g2_a, g1_b, g2_b, w_a, w_b)
```

**Galaxy Clustering**:
```python
# Aperture Number Count
N_ap = correlation.get_aperture_density(delta, w)
# Angular Clustering
wtheta, = correlation.compute_density_density(delta_a, delta_b, w_a, w_b)
```

**Galaxy-Galaxy Lensing**:
```python
# Aperture Number Count (Lenses)
N_ap = correlation.get_aperture_density(delta_lens, w_lens)
# Tangential Shear
gammat, = correlation.compute_density_shear(delta_lens, g1_source, g2_source, w_lens, w_source)
```

#### 2. Full Tomography (3x2pt)

Calculate all correlations for all requested tomographic bin combinations at once.

**Specific Probes**:

*Cosmic Shear*:
```python
# Returns: M_ap, xip, xim
M_ap, xip, xim = correlation.get_full_tomo(shear_maps, weights)
```

*Galaxy Clustering*:
```python
# Returns: wtheta
wtheta = correlation.vectorized_density_density(density_maps, weights)
```

*Galaxy-Galaxy Lensing*:
```python
# Returns: gammat (Lens->Source combinations)
gammat = correlation.vectorized_density_shear(
    density_maps, shear_maps, density_weights, shear_weights
)
```

**Combined 3x2pt Bundle**:

```python
# shear_maps:   [nzbins_s, 2, npix] or None
# density_maps: [nzbins_d, npix] or None
# weights:      dict of weights or None

M_ap, N_ap, xipm, wtheta, gammat = correlation.get_3x2pt_tomo(
    shear_maps=shear_maps,
    density_maps=density_maps,
    weights={"shear": shear_w, "density": density_w},
)
```

## Calculating i3PCFs

The integrated 3-point correlation functions can be calculated from the aperture masses/counts and 2PCFs using `CosmoFuse.correlation_helpers`.

### Shear i3PCF ($\zeta_\pm$)

Covariance between Aperture Mass ($M_{ap}$) and Shear 2PCF ($\xi_\pm$).

```python
from CosmoFuse.correlation_helpers import zeta_shear

# M_ap shape: (nmaps, nzbins, npatches)
# xip/xim shape: (nmaps, n_correlations, npatches, nbins)
zetap, zetam = zeta_shear(M_ap, xip, xim)
```

### Clustering i3PCF ($\zeta_{clust}$)

Covariance between Aperture Number Count ($N_{ap}$) and Angular Clustering ($w(\theta)$).

```python
from CosmoFuse.correlation_helpers import zeta_clust

# N_ap shape: (nmaps, nzbins, npatches)
# w shape: (nmaps, n_correlations, npatches, nbins)
zeta_c = zeta_clust(N_ap, w)
```

### GGL i3PCF ($\zeta_{ggl}$)

Covariance between Aperture Number Count ($N_{ap}$) and GGL Shear ($\gamma_t$).

```python
from CosmoFuse.correlation_helpers import zeta_ggl

# N_ap shape: (nmaps, n_lens_bins, npatches)
# gammat shape: (nmaps, n_ggl_pairs, npatches, nbins)
zeta_g = zeta_ggl(N_ap, gammat, lens_bins=nzbins_l, source_bins=nzbins_s)
```

### Unified Calculation

You can also calculate all available i3PCFs at once:

```python
from CosmoFuse.correlation_helpers import calculate_all_zetas

results = calculate_all_zetas(
    M_ap=M_ap, xip=xip, xim=xim,
    N_ap=N_ap, w=w,
    gammat=gammat, lens_bins=nzbins_l, source_bins=nzbins_s
)

# Access results
zetap = results.get('zetap')
zetam = results.get('zetam')
zeta_c = results.get('zeta_clust')
zeta_g = results.get('zeta_ggl')
```
