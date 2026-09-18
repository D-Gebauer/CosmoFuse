[![CI](https://github.com/D-Gebauer/CosmoFuse/actions/workflows/ci.yml/badge.svg)](https://github.com/D-Gebauer/CosmoFuse/actions/workflows/ci.yml)

[![codecov](https://codecov.io/github/D-Gebauer/CosmoFuse/graph/badge.svg?token=F4JC08UEJP)](https://codecov.io/github/D-Gebauer/CosmoFuse)



# CosmoFuse

A package for efficiently measuring integrated 3-point correlation functions on GPU/CPU written in Python 3.

The integrated 3-point correlation function probes squeezed configurations of the bispectrum without the computational expense of the full 3-point correlation function.

### 1. Cosmic Shear ($\zeta_{a,+}$ and $\zeta_{a,-}$)

The shear i3PCFs are calculated as correlations between central aperture mass $M_a$ and annular shear 2PCFs $\xi_\pm$:

$$ \zeta_{a,+} = \langle M_a \, \xi_+ \rangle, \quad \zeta_{a,-} = \langle M_a \, \xi_- \rangle $$

On a pixelated map, the aperture mass is calculated as:

$$ M_a = \frac{A \sum_{\text{p}}{w_p g_t Q_p}}{\sum_{\text{p}}{w_p}} $$

where $g_t$ is the tangential shear. The shear 2PCFs are calculated as:

$$ \xi_+ = \frac{\sum_{\text{pairs}}{w_1 w_2 g_1 g_2^*}}{\sum_{\text{pairs}}{w_1 w_2}}, \quad \xi_- = \frac{\sum_{\text{pairs}}{w_1 w_2 g_1 g_2}}{\sum_{\text{pairs}}{w_1 w_2}} $$

### 2. Galaxy Clustering ($\zeta_{g,g}$)

The clustering i3PCF is the correlation between central aperture count $M_g$ and annular galaxy auto-correlation $\xi_g$:

$$ \zeta_{g,g} = \langle M_g \, \xi_g \rangle $$

The aperture number count is given by:

$$ M_g = \frac{A \sum_{\text{p}}{w_p \delta_g Q_p}}{\sum_{\text{p}}{w_p}} $$

where $\delta_g$ is the galaxy overdensity (or counts). The annular galaxy auto-correlation is:

$$ \xi_g = \frac{\sum_{\text{pairs}}{w_1 w_2 \delta_1 \delta_2}}{\sum_{\text{pairs}}{w_1 w_2}} $$

### 3. Galaxy-Galaxy Lensing ($\zeta_{g,t}$)

The GGL i3PCF couples central aperture count (lenses) with annular tangential shear (sources):

$$ \zeta_{g,t} = \langle M_g \, \xi_t \rangle $$

where $M_g$ is calculated on the lens map as above. The tangential shear estimator is:

$$ \xi_t = \frac{\sum_{\text{pairs}}{w_l w_s \delta_l g_{s,t}}}{\sum_{\text{pairs}}{w_l w_s}} $$

where $\delta_l$ is the lens overdensity and $g_{s,t}$ is the source tangential shear relative to the lens.

## Features

- Calculate pairs for given mask & resolution once
- Save/Load pairs using hdf5 files
- Reuse pairs to measure i3PCFs across maps
- Optimized backend kernels for Spin-0×Spin-0 ($`w(\theta)`$), Spin-0×Spin-2 ($\gamma_t$), and Spin-2×Spin-2 ($\xi_\pm$) workloads
- Optimized tomographic kernels for all probes on CPU and GPU
- Automatic patch-center selection from a survey mask (optionally weighted by the compensated filter)
- Modular compensated aperture filters (Crittenden et al. 2002 by default; Schneider et al. 1998 included)
- **Compact row space**: device buffers and indices only cover the unmasked pixels; maps can be passed full-sky or already cut to the footprint (`Correlation.row_pix` order) — no scatter, `npix / n_active` times less host→device traffic
- **Static treecode (opt-in, `resolution_factor`)**: every angular bin is measured on the coarsest HEALPix level that still resolves it, cutting the pair geometry by orders of magnitude (nside 2048, $\theta_{\min}=5'$, ~900 patches fit on one GPU)
- Combination-tiled GPU $\xi_\pm$ kernel (all tomographic combinations in one pass over the pairs) and a ring-buffer map loader (`RowSpaceMapLoader`) that keeps the GPU busy while measuring $10^5$ maps

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
        accumulation_precision="float64",   # "same" / "float64"
        resolution_factor=None,             # None = full resolution (default); True = static treecode with k=4; or a number k
        aperture_nside=None,                # None = map resolution; e.g. 512 = aperture statistics on the degraded map
        pair_search_precision="auto",       # "float64" recommended for theta_min < ~10'
        pack_pairs=False,                   # True: 8 instead of 24 bytes per pair on the GPU
    )

For GPU runs the recommended precision configuration is
`map_precision="float32", accumulation_precision="float64"`: the maps,
uploads, and per-pair gathers run at float32 (roughly halving the
per-map wall time on bandwidth-bound hardware), while every pair sum is
accumulated and reduced at float64, so the estimators avoid float32
cancellation error. On an A100 this measures within a few times 1e-7
(scale-relative) of the float64 CPU reference — compared to a few times
1e-3 when also accumulating at float32 (`accumulation_precision="same"`).
Keep `map_precision="float64"` when bitwise-tight agreement with the
float64 reference (~1e-14) is required.

### Selecting patch centers from a mask

Instead of providing patch centers manually, they can be selected automatically from a survey footprint. Candidate centers are the pixel centers of a coarse `nside_centers` HEALPix grid (which controls the patch oversampling density); a candidate is accepted if the masked fraction of the full-resolution mask stays below `f_mask` within the 2PCF patch disc (radius `patch_size`) and below `f_mask_filter` (default: `f_mask`) within the aperture-mass filter disc (radius $5\,\theta_Q$):

    from CosmoFuse import select_patch_centers

    phi_center, theta_center = select_patch_centers(
        mask,                    # HEALPix footprint (nonzero = observed)
        nside_centers=32,        # candidate-center grid resolution
        patch_size=90,           # patch radius (arcminutes)
        theta_Q=90,              # compensated filter scale (arcminutes)
        f_mask=0.2,              # max masked fraction in the patch disc
    )
    correlation = Correlation(nside, phi_center, theta_center, mask=mask, ...)

or, in one step:

    correlation = Correlation.from_mask(
        nside, mask, nside_centers=32,
        patch_size=90, theta_Q=90, f_mask=0.2,
        nbins=10, theta_min=10, theta_max=170,
    )

The masking of the filter support disc (radius $5\,\theta_Q$) is judged with `filter_weighting`, always from the **binary** mask at its own resolution — a pixel contributes its whole filter value if it is unmasked and nothing otherwise; survey weights are never used:

- `"abs"` (default): fraction of *importance* lost, $\sum_{\rm masked} |Q| \,/\, \sum_{\rm all} |Q| \le$ `f_mask_filter`. Holes near the edge of the disc (negligible filter weight) no longer veto a patch, holes at the filter peak count more.
- `"signed"` (alias `"raw"`): deviation of the filter integral, $|\sum_{\rm masked} Q| \,/\, \sum_{\rm all} |Q|$. With a compensated filter such as `U_crittenden` / `U_schneider` (the convergence-space partners of the shipped $Q$ filters) this is how far the mask pushes the aperture away from being compensated; for the non-negative $Q$ filters it equals `"abs"`.
- `"pixels"`: unweighted masked pixel fraction (the default before 4.21; `filter_weighted=True/False` still works as an alias for `"abs"`/`"pixels"`).

A custom `aperture_filter` can be supplied for the weighting (same calling convention as `preprocess`, see [Aperture filters](#aperture-filters)); the 2PCF patch-disc check always uses the pixel fraction.

Then Calculate pairs:

    correlation.preprocess()

`preprocess()` accepts an `aperture_filter`: the compensated filter used for the aperture mass (see [Aperture filters](#aperture-filters); default `Q_crittenden`).

Optionally, the one-off Numba JIT compilation of the measurement kernels can be moved out of the first measurement call:

    correlation.warmup()

You can also release host-side pair arrays immediately after preprocessing:

    correlation.preprocess(release_host_pairs=True)

If host RAM is limited for large runs, you can optionally release host-side pair arrays after preparing backend buffers:

    correlation.prepare(release_host_pairs=True)

This keeps only arrays needed for later computations in memory. Save pairs before releasing host arrays (or reload/recompute before saving again).

These can be saved & loaded using:

    correlation.save_pairs("/path/to/pairs.h5")
    correlation.load_pairs("/path/to/pairs.h5")

To load pairs and immediately release host-side pair arrays after backend preparation:

    correlation.load_pairs("/path/to/pairs.h5", release_host_pairs=True)

Pair files are written in a consolidated layout (format version 2) that loads with a handful of bulk reads; files written by older CosmoFuse versions remain fully readable. Geometries with virtual rows (`resolution_factor` / `aperture_nside`) are written as format version 3 with different dataset names, so that CosmoFuse ≤ 4.20 refuses them instead of mis-reading them; full-resolution files stay version 2.

### Choosing the resolution (`resolution_factor`)

**The default is full resolution** (`resolution_factor=None`): every bin is measured on the map's own pixels, exactly as in previous versions (bit-for-bit). Explicit pair geometry grows as nside⁴, though: a 110′ patch holds 0.33 M pairs at nside 512 but 81 M at nside 2048 — ~2 TB for 1000 patches.

`resolution_factor=True` switches the static treecode on with the default factor $k=4$ (`CosmoFuse.DEFAULT_RESOLUTION_FACTOR`); a number sets $k$ explicitly. With `resolution_factor=k` bin $b$ is measured on the coarsest HEALPix level whose pixel size $p$ satisfies $p \le \theta_{\rm lo}(b)/k$. Coarse cells are built **per patch** from the unmasked map pixels inside the patch disc (the patch window stays an exact top-hat), carry the weighted mean of their members and the sum of their weights, and sit at the centroid of their members. Because $W_I W_J \gamma_I \gamma_J = \sum_{i\in I}\sum_{j\in J} w_i w_j \gamma_i \gamma_j$, no pair is dropped and no noise is added — but every fine pair is binned and rotated with the geometry of its parent cells. **This is a different (windowed) estimator; use the same `resolution_factor` for data, simulations and covariances, and never mix it with full-resolution measurements** (the two agree in the mean to the numbers below, but their per-patch noise realisations differ, since pairs move between neighbouring bins).

Measured on nside-2048 maps with the DES Y3 mask (5′–250′, 11 bins, 110′ patches; signal ratio to full resolution as a function of the effective $k_{\rm eff}=\theta_{\rm lo}/p \in [k, 2k)$):

| $k_{\rm eff}$ | $\xi_-$ | $\gamma_t$ | $\xi_+$, $\xi_g$ | noise variance |
| :--- | :--- | :--- | :--- | :--- |
| ≈ 2.1 | −19 % | −2…−6 % | ≲ 1 % | unchanged |
| ≈ 3.0 | −9 % | −2 % | ≲ 1 % | unchanged |
| ≈ 4.3 | −5 % | −1…−2 % | < 0.7 % | unchanged |
| ≈ 6.1 | −2.5 % | −0.4 % | < 0.3 % | unchanged |
| ≈ 8.7 | −1 % | ≤ 0.3 % | < 0.3 % | unchanged |

| `resolution_factor` | pairs / patch (nside 2048) | pair memory, 1000 patches |
| :--- | :--- | :--- |
| `None` (full) | 78 M | ~1.9 TB |
| 2 | 0.25 M | 6 GB |
| 2.9 | 0.64 M | 16 GB |
| 4 | 1.4 M | 35 GB |
| 5.8 | 3.0 M | 72 GB |

With $\sqrt 2$-spaced bin edges (e.g. `theta_min=5`, `theta_max=5*2**5.5`, `nbins=11`) the levels change every second bin and $k_{\rm eff}$ alternates cleanly between $k$ and $\sqrt2\,k$.

    corr = Correlation(2048, phi, theta, nbins=10, theta_min=5, theta_max=175,
                       patch_size=110, theta_Q=110, mask=mask,
                       resolution_factor=2.9, aperture_nside=512,
                       map_precision="float32", accumulation_precision="float64")
    corr.preprocess()
    corr.level_table        # nside and k_eff per bin -- store it with every data vector

Before pair finding a **preflight check** projects the pair memory from the mask and raises a `MemoryError` that names a `resolution_factor` that fits (budget: free device memory, or `memory_budget_gb=`), instead of failing with an out-of-memory error an hour into preprocessing. `load_pairs` adopts the resolution stored in the pair file and raises if the constructor explicitly asked for a different one.

**Pair packing (`pack_pairs=True`).** On the device a pair normally costs 24 bytes (two int32 rows + two complex64 rotation factors). Packed it costs 8: the rotation factors have unit modulus, so only their angle is kept (uint16, resolution $2\pi/65536$), and the row indices become uint16 indices local to the row block of their patch and resolution level (the map rows of every patch are gathered into contiguous blocks once per map). Three times more pairs fit on the GPU — a higher `resolution_factor` or a denser patch grid — at unchanged speed (A100: 14.5 → 5.2 GB for nside 2048, 917 patches, $k=2.9$). It is not bit-identical: on real DES Y3 maps every per-patch estimate moves by $3\times10^{-5}$ (rms; max $2\times10^{-4}$) of its own patch-to-patch scatter, with no bias (mean shift $<10^{-7}\sigma$), and the same holds for patch averages and the i3PCFs — the error is a fixed tiny fraction of the statistical error at every level of averaging. Pair files and host arrays stay exact; the CPU backend uses the same quantised rotations, so both backends measure the same estimator. On GPU backends the packed geometry currently serves the shear path (`vectorized_shear_shear`, `get_full_tomo_shear`) and the aperture statistics; the other methods raise.

At small scales also set `pair_search_precision="float64"` (automatic with `resolution_factor`): a float32 pair search resolves separations only to $\delta\theta/\theta \approx 6\times10^{-8}/\theta^2$ — 0.3 % at 15′ but 3 % at 5′, where it puts ~4 % of the pairs into the wrong bin. The stored rotation factors stay at `rotation_precision`, so this costs no memory.

### Aperture filters

The aperture statistics $M_a$ and $M_g$ convolve the maps with a compensated filter $Q(\theta)$, evaluated for all pixels within $5\,\theta_Q$ of each patch center. The filter is modular: any callable `Q(theta, theta_Q)` (with `theta` in radians and `theta_Q` in arcminutes; a single-argument `Q(theta)` also works) can be passed as `aperture_filter` to `preprocess()`, `calculate_pairs_M_a()`, `select_patch_centers()`, and `Correlation.from_mask()`. Two filters ship with the package:

**`Q_crittenden` (default)** — the exponential compensated filter of [Crittenden et al. (2002)](https://arxiv.org/abs/astro-ph/0012336), as used for the i3PCF in [Halder et al. (2021)](https://arxiv.org/abs/2102.10177):

$$ Q(\theta) = \frac{\theta^2}{4\pi\theta_Q^4} \exp\left(-\frac{\theta^2}{2\theta_Q^2}\right) $$

It peaks at $\theta = \sqrt{2}\,\theta_Q$ and has decayed to below $10^{-3}$ of its peak value at the $5\,\theta_Q$ truncation radius.

**`Q_schneider`** — the polynomial ($\ell = 1$) compensated filter of [Schneider et al. (1998)](https://arxiv.org/abs/astro-ph/9708143), which has compact support:

$$ Q(\theta) = \frac{6}{\pi\theta_Q^2}\, x^2 \left(1 - x^2\right) \;\; \text{for } x = \theta/\theta_Q \le 1, \qquad Q(\theta) = 0 \;\; \text{for } \theta > \theta_Q $$

Pixels between $\theta_Q$ and the $5\,\theta_Q$ aperture disc simply receive zero weight.

Both filters are normalised to $\int Q(\theta)\, \mathrm{d}\Omega = 1$, so aperture masses measured with either are directly comparable:

    from CosmoFuse import Q_crittenden, Q_schneider

    correlation.preprocess(aperture_filter=Q_schneider)

Since $\theta_Q$ is much larger than a pixel, the aperture statistics do not need the map resolution: `aperture_nside=512` evaluates $M_a$/$M_g$ on the (weighted-mean, globally) degraded map — numerically the aperture statistic of an nside-512 analysis — and keeps the aperture geometry at that size (×16 smaller than at nside 2048). Default: the map resolution.

### Measuring Correlations

The package supports 3 main probes: Cosmic Shear, Galaxy Clustering, and Galaxy-Galaxy Lensing (GGL).

**Map inputs.** Every measurement method accepts maps whose last axis is either the full sky (`npix`) or the **row space** (`correlation.n_active` unmasked pixels in `correlation.row_pix` order, ascending RING). Full-sky maps are cut down on the fly; row-space maps are used as they are and are the fast path — masked pixels are never touched (they may be NaN). Any other length raises. Weight maps that are read-only (`w.flags.writeable = False`) are gathered, uploaded and degraded once and then reused.

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

**Measuring many maps (GPU)**:

Store maps in row space (`correlation.to_row_space(full_sky_maps)`, together with `correlation.row_pix_hash`) and let `RowSpaceMapLoader` feed the GPU: reader threads fill a ring of pinned buffers directly from your files, uploads run on their own CUDA stream, and the device arrays arrive strictly in order. On an A100 this keeps the GPU > 95 % busy (DES Y3, 917 patches, 4 source bins: 26 ms per map-set at nside 512 full resolution; 78 ms at nside 2048, $\theta_{\min}=5'$, `resolution_factor=2.9`).

```python
from CosmoFuse import RowSpaceMapLoader

w = np.ascontiguousarray(weights_full_sky[:, correlation.row_pix], dtype=np.float32)
w.flags.writeable = False                      # fixed weights: uploaded/degraded once

def read(path, out):                           # runs in a reader thread
    out["shear"][...] = np.load(path, mmap_mode="r")   # (nz, 2, n_active), any dtype

loader = RowSpaceMapLoader(correlation, {"shear": (nz, 2, correlation.n_active)},
                           sources=files, read_fn=read, n_slots=4, n_readers=2,
                           row_pix_hash=stored_hash)
for k, dev in loader:                          # dev arrays are valid until the next iteration
    M_a, xi_p, xi_m = correlation.get_full_tomo_shear(dev["shear"], w, flip_g1=True)
    results.append([o.copy() for o in (M_a, xi_p, xi_m)])
```

**Overlapping uploads with compute, two slots (GPU)**:

When measuring many maps in a loop, `PinnedMapPipeline` double-buffers the
host→device transfers through pinned memory on a dedicated CUDA stream, so
map k+1 uploads while map k computes (a no-op passthrough on CPU):

```python
from CosmoFuse import PinnedMapPipeline

pipe = PinnedMapPipeline(correlation, {"shear": (nz, 2, npix), "w": (nz, npix)})
dev = pipe.wait(pipe.stage({"shear": shear_np[0], "w": w_np[0]}))
for k in range(nmaps):
    nxt = pipe.stage({"shear": shear_np[k + 1], "w": w_np[k + 1]}) if k + 1 < nmaps else None
    results.append(correlation.get_full_tomo_shear(dev["shear"], dev["w"]))
    dev = pipe.wait(nxt)
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
