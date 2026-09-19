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

$$ \xi_+ = \frac{\sum_{\text{pairs}}{w_1 w_2 \, g_1' g_2'^*}}{\sum_{\text{pairs}}{w_1 w_2}}, \quad \xi_- = \frac{\sum_{\text{pairs}}{w_1 w_2 \, g_1' g_2'}}{\sum_{\text{pairs}}{w_1 w_2}} $$

where $g' = g\,e^{-2i\varphi}$ is the shear rotated into the frame of the pair
($\varphi$ the position angle of the separation vector at that pixel, one per
pair member). The rotation is not optional bookkeeping: without it $\xi_-$ is
not invariant under a rotation of the coordinate frame.

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
        device="auto",                      # "cpu", "gpu", "auto", a GPU id, or a list of GPU ids
        map_precision="float32",            # float32 / float64
        rotation_precision="float32",       # float32 / float64
        accumulation_precision="float64",   # "same" / "float64"
        resolution_factor=None,             # None = full resolution; True (k=4) or a number k: static treecode
        aperture_nside=None,                # coarser nside for the aperture statistics (None = nside)
        memory_budget_gb=None,              # pair-memory budget of the preflight check (None = free device memory)
        pair_search_precision="float64",    # "float64" / "float32"
        pack_pairs=False,                   # 8-byte pairs on the device (tomographic methods only)
        pack_host_pairs=False,              # 8-byte pairs in host RAM and in the pair file too
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
        filter_weighting="abs",  # "abs" / "signed"
    )
    correlation = Correlation(nside, phi_center, theta_center, mask=mask, ...)

or, in one step:

    correlation = Correlation.from_mask(
        nside, mask, nside_centers=32,
        patch_size=90, theta_Q=90, f_mask=0.2,
        nbins=10, theta_min=10, theta_max=170,
    )

With `filter_weighting="abs"` (default) the aperture-mass disc check weights each pixel by the compensated filter instead of counting pixels — the masked fraction becomes $\sum_{\rm masked} |Q(\theta)| \,/\, \sum_{\rm all} |Q(\theta)|$ — so holes near the edge of the disc (where the filter carries almost no weight) no longer veto a patch, while holes at the filter peak count more. The magnitude $|Q|$ is used because compensated filters can be negative at large radii; `"signed"` uses $|\sum_{\rm masked} Q|$ instead. A custom `aperture_filter` can be supplied for the weighting (same calling convention as `preprocess`, see [Aperture filters](#aperture-filters)); the 2PCF patch-disc check always uses the raw pixel fraction.

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

Pair files are written in a consolidated layout (format version 2; version 3 with `resolution_factor`, version 4 with `pack_host_pairs`) that loads with a handful of bulk reads. Versions 2 and later are readable; **format version 1 is not** — it is rejected with an explicit error, so a pre-consolidation archive has to be regenerated.

`pack_pairs=True` stores the pair geometry on the *device* in 8 instead of 24 bytes per pair, so about three times as many pairs fit on a GPU; host arrays and pair files stay exact. `pack_host_pairs=True` applies the same packing already at pair-finding time, which cuts host RAM and the pair file by the same factor (`pair_inds` / `pair_exp2phi` are then `None`, the payload lives in `packed_pairs`):

    correlation = Correlation(..., pack_pairs=True, pack_host_pairs=True)
    correlation.preprocess()
    correlation.save_pairs("/path/to/pairs.h5")   # format version 4

Both are off by default. Packing quantises the pair rotations to $2\pi/65536$, which moves each estimate by ~1e-5 of its statistical error without biasing it — but a `pack_host_pairs` file is no longer exact and cannot be turned back into an exact one, so keep an exact file if you may want to switch the estimator later. A packed file loads into any matching mask and slices by patch like any other.

### Aperture filters

The aperture statistics $M_a$ and $M_g$ convolve the maps with a compensated filter $Q(\theta)$, evaluated for all pixels within $5\,\theta_Q$ of each patch center. The filter is modular: any callable `Q(theta, theta_Q)` (with `theta` in radians and `theta_Q` in arcminutes; a single-argument `Q(theta)` also works) can be passed as `aperture_filter` to `preprocess()`, `calculate_pairs_M_a()`, `select_patch_centers()`, and `Correlation.from_mask()`. Two tangential-shear filters $Q$ ship with the package (their convergence-space counterparts $U$, `U_crittenden` and `U_schneider`, are exported alongside them for theory work — they are not accepted as `aperture_filter`, which takes $Q$):

**`Q_crittenden` (default)** — the exponential compensated filter of [Crittenden et al. (2002)](https://arxiv.org/abs/astro-ph/0012336), as used for the i3PCF in [Halder et al. (2021)](https://arxiv.org/abs/2102.10177):

$$ Q(\theta) = \frac{\theta^2}{4\pi\theta_Q^4} \exp\left(-\frac{\theta^2}{2\theta_Q^2}\right) $$

It peaks at $\theta = \sqrt{2}\,\theta_Q$ and has decayed to below $10^{-3}$ of its peak value at the $5\,\theta_Q$ truncation radius.

**`Q_schneider`** — the polynomial ($\ell = 1$) compensated filter of [Schneider et al. (1998)](https://arxiv.org/abs/astro-ph/9708143), which has compact support:

$$ Q(\theta) = \frac{6}{\pi\theta_Q^2}\, x^2 \left(1 - x^2\right) \;\; \text{for } x = \theta/\theta_Q \le 1, \qquad Q(\theta) = 0 \;\; \text{for } \theta > \theta_Q $$

Pixels between $\theta_Q$ and the $5\,\theta_Q$ aperture disc simply receive zero weight.

Both filters are normalised to $\int Q(\theta)\, \mathrm{d}\Omega = 1$, so aperture masses measured with either are directly comparable:

    from CosmoFuse import Q_crittenden, Q_schneider

    correlation.preprocess(aperture_filter=Q_schneider)

### Measuring Correlations

The package supports 3 main probes: Cosmic Shear, Galaxy Clustering, and Galaxy-Galaxy Lensing (GGL).

#### Probes & Inputs

| Probe | Aperture Quantity | 2-Point Correlation | Inputs |
| :--- | :--- | :--- | :--- |
| **Shear** | Aperture Mass ($M_a$) | Shear 2PCF ($\xi_\pm$) | Shear maps ($g_1, g_2$), Weights ($w$) |
| **Clustering** | Aperture Count ($M_g$) | Angular Clustering ($\xi_g$) | Density maps ($\delta$ or counts), Weights ($w$) |
| **GGL** | Aperture Count ($M_g$) | Tangential Shear ($\xi_t$) | Lens density + Source shear |

#### What the measurement methods return (`return_device`)

Every measurement method takes `return_device` and it defaults to **`True`**.
On a GPU backend that means you get **cupy arrays**, not numpy — which is what
makes the device-resident pipeline work: `MapLoader` keeps the maps on the
card, `ZetaWriter` reduces on the card and copies ~9 kB instead of ~1 MB per
map-set, and nothing crosses PCIe that does not have to. Pass
`return_device=False` for numpy.

Two things worth knowing:

* **A multi-device group always returns numpy.** `MultiDeviceCorrelation`
  concatenates the per-device patch ranges on the host, so `return_device` is
  ignored there.
* **The arrays are yours.** Each call allocates its outputs, so collecting
  results in a list across realisations is safe:

  ```python
  results = [corr.get_3x2pt_tomo(...) for _ in range(n_realisations)]   # fine
  ```

  (Before 6.3.0 `get_3x2pt_tomo(return_device=True)` returned cached buffers
  that the next call overwrote in place, so that loop silently produced N
  copies of the last realisation.)

#### Which row is which pair of bins

The tomographic outputs are `(ncomb, n_patches, nbins)` and the row order is
part of the contract, not something to rederive:

```python
Correlation.tomo_combinations(4)         # xi_p / xi_m / xi_g rows: [(0,0), (0,1), ...]
Correlation.tomo_combinations(4, True)   # gc_auto_correlations_only=True: the diagonal only
Correlation.ggl_combinations(2, 4)       # xi_t rows as (lens_bin, source_bin)
Correlation.zeta_triplets(4)             # same-sample zeta rows: (z_center, z2, z3)
Correlation.zeta_cross_triplets(5, 10)   # cross-sample zeta rows: (z_center, annulus_row)
```

Note that `gc_auto_correlations_only=True` changes the length of the $\xi_g$
vector from `nz(nz+1)/2` to `nz` — a shape check will not catch a mislabelled
data vector, so index it by the accessor.

A ζ estimator uses `zeta_triplets` when its centre and its annulus are built
from the **same** sample (`zeta_a_plus`, `zeta_a_minus`, `zeta_g_g`) and
`zeta_cross_triplets` when they are not (`zeta_g_plus`, `zeta_g_minus`,
`zeta_a_g`, `zeta_g_t`, `zeta_a_t`) — see below.

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

**Overlapping uploads with compute (GPU)**:

On a GPU the host→device transfer of a map-set costs about as much as the
measurement itself. Two loaders hide it behind the previous map's kernels
(both are no-op passthroughs on CPU backends).

`MapLoader` double-buffers maps that are already in host arrays:

```python
from CosmoFuse import MapLoader

pipe = MapLoader(correlation, {"shear": (nz, 2, npix), "w": (nz, npix)})
dev = pipe.wait(pipe.stage({"shear": shear_np[0], "w": w_np[0]}))
for k in range(nmaps):
    nxt = pipe.stage({"shear": shear_np[k + 1], "w": w_np[k + 1]}) if k + 1 < nmaps else None
    results.append(correlation.get_full_tomo_shear(dev["shear"], dev["w"]))
    dev = pipe.wait(nxt)
```

`MapFileLoader` is for maps read from disk: `n_readers` threads fill a ring of
`n_slots` pinned buffers straight from the source and hand the device buffers
to you in order.

```python
from CosmoFuse import MapFileLoader

def read(source, out):                    # runs in a reader thread
    out["shear"][...] = np.load(source, mmap_mode="r")

loader = MapFileLoader(
    correlation, {"shear": (nz, 2, correlation.n_active)},
    sources=files, read_fn=read, n_slots=8, n_readers=4,
    row_pix_hash=correlation.row_pix_hash,   # optional archive/mask check
)
for k, dev in loader:
    results.append(correlation.get_full_tomo_shear(dev["shear"], w))
```

Store the maps in row space (`correlation.to_row_space(full_sky_maps)`, done
once when the archive is written) so no gather is needed per map. Fixed weight
maps should be passed as a read-only array (`w.flags.writeable = False`), which
lets CosmoFuse upload and degrade them once instead of once per map. Use enough
readers that reading keeps up with the measurement; the device arrays handed
out are only valid until the next iteration.

Note that on GPU backends `warmup()` does nothing — the CUDA kernels are
compiled on the first measurement call, so make one throwaway call before
timing a loop.

(Up to 5.0 these classes were called `PinnedMapPipeline` and
`RowSpaceMapLoader`.)

**Several GPUs**:

Passing a list of GPU ids splits the patches across those devices — one
`Correlation` per GPU, measured in parallel, outputs concatenated in patch
order. The result is identical to a single-device run, and each GPU holds only
its share of the pair geometry, which is how a geometry too large for one card
is measured. A single device remains the default.

```python
correlation = Correlation(nside, phi_center, theta_center, device=[0, 1], ...)
correlation.preprocess()          # or load_pairs(path): each device takes its own slice
M_a, xi_p, xi_m = correlation.get_full_tomo_shear(shear_maps, weights)
```

Write pair files from a single-device instance; a multi-device group only
reads them.

### Streaming zetas to a file

For long runs, `ZetaWriter` reduces each map-set to its i3PCFs and appends them to
HDF5 from a background thread, instead of holding every per-patch array in RAM until
the end:

    from CosmoFuse import ZetaWriter

    with ZetaWriter("zetas.h5", correlation, flush_every=50) as out:
        for shear, w in map_sets:
            out.submit_shear(correlation.get_full_tomo_shear(shear, w))

`submit_3x2pt()` takes a `get_3x2pt_tomo()` result, and `submit(M_a=..., xi_p=...)`
any subset of the six fields. `submit()` blocks once `depth` map-sets are queued, so
memory stays bounded and a writer that falls behind cannot go unnoticed.

This works because zeta averages over *patches*, not over maps: map-set *k* can be
reduced the moment it is measured. At the DES Y3 production geometry one map-set is
1.06 MB of per-patch arrays but only 9 kB of zetas — 118x smaller, 90 MB instead of
10.6 GB for 10,000 realisations. Pass `reduce="none"` to store the per-patch arrays
instead; zeta, a leave-one-patch-out jackknife or a different binning can all still
be derived from those.

On a GPU the reduction runs on the device before the copy, so only the 9 kB data
vector crosses PCIe and the writer thread does nothing but I/O. With a
`MultiDeviceCorrelation` the group returns host arrays, so the reduction runs on the
CPU in the writer thread — still off the measurement's critical path, and nothing
about the group changes.

The file records `n_flushed`, the number of map-sets guaranteed to be on disk, and
enough provenance (`level_table`, `row_pix_hash`, nside, bins, `resolution_factor`)
that a data vector can never be silently combined with one measured under a
different estimator. After a crash, reopen with `resume=True`: anything written past
the last flush is discarded and appending continues.

`swmr=True` lets another process follow the file as it grows, but it is **not free**
and is off by default: it forces `libver="latest"` (needs HDF5 >= 1.10 to read),
readers must pass `swmr=True`, HDF5 documents it as unreliable on NFS, and after a
crash the file keeps a stale write lock — an ordinary `h5py.File(path, "r")` then
fails until someone runs `h5clear -s`. A non-SWMR file survives the same crash and
opens normally.

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

### Mixed tomographies

The source and lens samples need not share a binning: `M_a` over 4 source bins
beside `M_g` over 5 lens bins is the ordinary 6x2pt case. An estimator whose
centre and annulus come from one sample keeps the upper-triangle layout
(`zeta_triplets`); one that crosses two samples returns a row per
`(z_center, annulus_combination)`, centre-major (`zeta_cross_triplets`).

The layout is inferred from the shapes, which is unambiguous except when the
two samples happen to have the **same number of bins** — a cross annulus then
holds exactly `nz(nz+1)/2` combinations and is read as the triangle, the
historical behaviour. Say which you meant:

```python
za_g = zeta_a_g(M_a, xi_g, symmetric=False)   # 4 x 15 rows, not the triangle

results = calculate_all_zetas(
    M_g=M_g, M_a=M_a, xi_p=xi_p, xi_g=xi_g,
    symmetric={"zeta_a_g": False, "zeta_g_plus": False},
)
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
