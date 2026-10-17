[![CI](https://github.com/D-Gebauer/CosmoFuse/actions/workflows/ci.yml/badge.svg)](https://github.com/D-Gebauer/CosmoFuse/actions/workflows/ci.yml)

[![codecov](https://codecov.io/github/D-Gebauer/CosmoFuse/graph/badge.svg?token=F4JC08UEJP)](https://codecov.io/github/D-Gebauer/CosmoFuse)



# CosmoFuse

A Python package for measuring integrated 3-point correlation functions (i3PCFs), as well as
footprint-wide two- and three-point correlation functions, on HEALPix maps, on GPU or CPU.

Three estimators share one vocabulary: methods are TreeCorr's field codes (`g` = shear,
`k` = scalar field such as an overdensity δ), and every call returns a labelled result object.

| class | measures | methods | result |
|---|---|---|---|
| `Integrated3PCF` | per-patch apertures and 2PCFs (the i3PCF ingredients) | `gg`, `kk`, `kg`, `three_by_two_pt`, `m_a`, `m_g` | `IntegratedResult` |
| `Global2PCF` | 2PCFs of the whole footprint | `gg`, `kk`, `kg`, `three_by_two_pt` | `TwoPointResult` |
| `Global3PCF` | 3PCF multipoles of the whole footprint | `ggg`, `kkk`, `kgg`, `kkg`, `measure` | `ThreePointResult` |

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
- Footprint-wide 2PCFs (ξ±, w(θ), γ_t) with per-tile sums for jackknife errors
- Footprint-wide 3PCFs (shear Γ₀…Γ₃, scalar ζ, any mixed spin-0/spin-2 triplet) by multipole decomposition
- Tomographic measurements of all probes on CPU and GPU
- Automatic patch-center selection from a survey mask (optionally weighted by the compensated filter)
- Modular compensated aperture filters (Crittenden et al. 2002 by default; Schneider et al. 1998 included)

## Installation
Install using:

    pip install git+https://github.com/D-Gebauer/CosmoFuse.git

Note: for GPU execution, install CuPy in your environment.

## USAGE

First create an Integrated3PCF object:

    from CosmoFuse import Integrated3PCF
    correlation = Integrated3PCF(
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
        pack_pairs=False,                   # 8-byte pairs on the device
        pack_host_pairs=False,              # 8-byte pairs in host RAM and in the pair file too
    )

For GPU runs use `map_precision="float32", accumulation_precision="float64"`: the maps are
handled at float32, every pair sum is accumulated at float64. Keep `map_precision="float64"`
when agreement with a float64 reference at machine precision is required.

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
    correlation = Integrated3PCF(nside, phi_center, theta_center, mask=mask, ...)

or, in one step:

    correlation = Integrated3PCF.from_mask(
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

Pair files of format version 1 (written before 5.0) are no longer readable and have to be regenerated.

`pack_pairs=True` stores the pair geometry on the device in 8 instead of 24 bytes per pair, so about
three times as many pairs fit on a GPU. `pack_host_pairs=True` applies the same packing in host RAM
and in the pair file:

    correlation = Integrated3PCF(..., pack_pairs=True, pack_host_pairs=True)
    correlation.preprocess()
    correlation.save_pairs("/path/to/pairs.h5")

Both are off by default. Packing moves each estimate by ~1e-5 of its statistical error without biasing
it, but a packed pair file cannot be turned back into an exact one, so keep an exact file if you may
want to switch later.

### Aperture filters

The aperture statistics $M_a$ and $M_g$ convolve the maps with a compensated filter $Q(\theta)$, evaluated for all pixels within $5\,\theta_Q$ of each patch center. The filter is modular: any callable `Q(theta, theta_Q)` (with `theta` in radians and `theta_Q` in arcminutes; a single-argument `Q(theta)` also works) can be passed as `aperture_filter` to `preprocess()`, `find_aperture_pairs()`, `select_patch_centers()`, and `Integrated3PCF.from_mask()`. Two tangential-shear filters $Q$ ship with the package (their convergence-space counterparts $U$, `U_crittenden` and `U_schneider`, are exported alongside them for theory work — they are not accepted as `aperture_filter`, which takes $Q$):

**`Q_crittenden` (default)** — the exponential compensated filter of [Crittenden et al. (2002)](https://arxiv.org/abs/astro-ph/0012336), as used for the i3PCF in [Halder et al. (2021)](https://arxiv.org/abs/2102.10177):

$$ Q(\theta) = \frac{\theta^2}{4\pi\theta_Q^4} \exp\left(-\frac{\theta^2}{2\theta_Q^2}\right) $$

It peaks at $\theta = \sqrt{2}\,\theta_Q$ and has decayed to below $10^{-3}$ of its peak value at the $5\,\theta_Q$ truncation radius.

**`Q_schneider`** — the polynomial ($\ell = 1$) compensated filter of [Schneider et al. (1998)](https://arxiv.org/abs/astro-ph/9708143), which has compact support:

$$ Q(\theta) = \frac{6}{\pi\theta_Q^2}\, x^2 \left(1 - x^2\right) \;\; \text{for } x = \theta/\theta_Q \le 1, \qquad Q(\theta) = 0 \;\; \text{for } \theta > \theta_Q $$

Pixels between $\theta_Q$ and the $5\,\theta_Q$ aperture disc simply receive zero weight.

Both filters are normalised to $\int Q(\theta)\, \mathrm{d}\Omega = 1$, so aperture masses measured with either are directly comparable:

    from CosmoFuse import Q_crittenden, Q_schneider

    correlation.preprocess(aperture_filter=Q_schneider)

### Measuring correlations

Method names are TreeCorr's field codes: `g` is a shear field (spin 2), `k` a
scalar field -- an overdensity map δ or a convergence map.  The class says
*what* is measured, the method *which fields*:

| `Integrated3PCF` method | measures (per patch) | result fields |
| :--- | :--- | :--- |
| `gg(shear_maps, w)` | ξ± and the aperture mass at the patch centre | `M_a`, `xi_p`, `xi_m` |
| `kk(k_maps, w)` | w(θ) (field estimator on δ) and the aperture count | `M_g`, `xi_g` |
| `kg(k_maps, shear_maps, k_w, shear_w, m_g=False, m_a=False)` | γ_t of lens δ × source shear (+ optional apertures) | `xi_t` (`M_g`, `M_a`) |
| `three_by_two_pt(shear_maps, k_maps, weights)` | all six in one fused pass | all six |
| `m_a(shear_maps, w)`, `m_g(k_maps, w)` | apertures only | array `(nz, n_patches)` |

`gg` and `kk` measure the apertures too by default; pass `apertures=False` for
the 2PCFs alone.  Inputs are full-sky
(`(nz, [2,] npix)`) or row-space (`(nz, [2,] n_active)`) arrays.

Every call returns a labelled `IntegratedResult` -- fields that were not
measured are `None`, and the rows of every 2PCF come with their tomographic
labels:

```python
res = correlation.gg(shear_maps, weights, flip_g1=True)
res.M_a.shape            # (nz, n_patches)
res.xi_p.shape           # (nz(nz+1)/2, n_patches, nbins)
res.gg_combinations      # ((0, 0), (0, 1), ...) -- which bins each row of xi_p/xi_m is
res.theta                # bin centres in arcmin

res = correlation.kg(delta_maps, shear_maps, w_lens, w_source, m_g=True)
res.xi_t, res.M_g, res.kg_combinations      # rows as (lens_bin, source_bin)

res = correlation.three_by_two_pt(
    shear_maps, delta_maps, weights={"shear": shear_w, "density": density_w},
)
zetas = res.zetas()      # every i3PCF the measured fields form
```

The TreeCorr equivalents (on a catalogue of the pixel centres, `metric="Arc"`)
are `GGCorrelation` for ξ±, `KKCorrelation` with `k = δ` for w(θ) and
`KGCorrelation` for γ_t.  With coverage-fraction weights the `kk` field
estimator is exactly Landy-Szalay with randoms at the pixel centres.

#### Probes & Inputs

| Probe | Aperture | 2PCF | Inputs |
| :--- | :--- | :--- | :--- |
| **Shear** (`gg`) | Aperture mass ($M_a$) | $\xi_\pm$ | Shear maps ($g_1, g_2$), weights |
| **Clustering** (`kk`) | Aperture count ($M_g$) | $w(\theta)$ | Overdensity maps δ, weights (coverage) |
| **GGL** (`kg`) | $M_g$ (and $M_a$) | $\gamma_t$ | Lens δ + source shear |

#### Host or device arrays (`return_device`)

`return_device` defaults to `True`: on a GPU backend the result's arrays are cupy arrays, which
is what keeps a device-resident pipeline (`MapLoader`, `ZetaWriter`) off the PCIe bus. Pass
`return_device=False` or call `res.to_numpy()` for numpy. A multi-device instance always returns
numpy. Each call allocates its own outputs, so collecting results across realisations is safe.

#### Which row is which pair of bins

Results carry their row labels (`gg_combinations`, `kk_combinations`,
`kg_combinations`).  The same orders are available without measuring:

```python
Integrated3PCF.tomo_combinations(4)                  # xi_p / xi_m / xi_g rows: [(0,0), (0,1), ...]
Integrated3PCF.tomo_combinations(4, auto_only=True)  # kk(auto_only=True): the diagonal only
Integrated3PCF.kg_combinations(2, 4)                 # xi_t rows as (lens_bin, source_bin)
Integrated3PCF.zeta_triplets(4)                      # same-sample zeta rows: (z_center, z2, z3)
Integrated3PCF.zeta_cross_triplets(5, 10)            # cross-sample zeta rows: (z_center, annulus_row)
```

`auto_only=True` changes the length of the $\xi_g$ vector from `nz(nz+1)/2` to
`nz` -- a shape check will not catch a mislabelled data vector, so use the labels.

A ζ estimator uses `zeta_triplets` when its centre and its annulus are built
from the **same** sample (`zeta_a_plus`, `zeta_a_minus`, `zeta_g_g`) and
`zeta_cross_triplets` when they are not (`zeta_g_plus`, `zeta_g_minus`,
`zeta_a_g`, `zeta_g_t`, `zeta_a_t`) -- see below.

### Overlapping uploads with compute (GPU)

On a GPU the upload of a map-set costs about as much as the measurement. Two loaders
hide it behind the previous map's measurement (both are passthroughs on CPU backends).

`MapLoader` is for maps that are already in host arrays:

```python
from CosmoFuse import MapLoader

pipe = MapLoader(correlation, {"shear": (nz, 2, npix), "w": (nz, npix)})
dev = pipe.wait(pipe.stage({"shear": shear_np[0], "w": w_np[0]}))
for k in range(nmaps):
    nxt = pipe.stage({"shear": shear_np[k + 1], "w": w_np[k + 1]}) if k + 1 < nmaps else None
    results.append(correlation.gg(dev["shear"], dev["w"]))
    dev = pipe.wait(nxt)
```

`MapFileLoader` is for maps read from disk: `n_readers` threads read into `n_slots`
buffers and hand the device arrays to you in order.

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
    results.append(correlation.gg(dev["shear"], w))
```

Store the maps in row space (`correlation.to_row_space(full_sky_maps)`, done once when the
archive is written) so no gather is needed per map. Pass fixed weight maps as a read-only array
(`w.flags.writeable = False`) so they are uploaded once. The device arrays handed out are only
valid until the next iteration.

On GPU backends `warmup()` does nothing: the CUDA kernels are compiled on the first measurement
call, so make one throwaway call before timing a loop.

### Several GPUs

Passing a list of GPU ids splits the patches across those devices and concatenates the outputs
in patch order. The result is identical to a single-device run, and each GPU holds only its share
of the pairs, which is how a geometry too large for one card is measured.

```python
correlation = Integrated3PCF(nside, phi_center, theta_center, device=[0, 1], ...)
correlation.preprocess()          # or load_pairs(path): each device takes its own slice
res = correlation.gg(shear_maps, weights)
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
            out.submit(correlation.gg(shear, w))

`submit()` takes any `Integrated3PCF` result (`gg`, `kk`, `kg`, `three_by_two_pt`) -- or an
`IntegratedResult` built by hand -- holding at least one central and one annular field. It blocks
once `depth` map-sets are queued, so memory stays bounded and a writer that falls behind cannot go
unnoticed.

Each map-set is reduced to its i3PCFs the moment it is measured (zeta averages over patches, not
over maps), so only the ~9 kB data vector per map-set is kept instead of the ~1 MB of per-patch
arrays. Pass `reduce="none"` to store the per-patch arrays instead; zeta, a leave-one-patch-out
jackknife or a different binning can all still be derived from those.

The file records `n_flushed`, the number of map-sets guaranteed to be on disk, and enough
provenance (`level_table`, `row_pix_hash`, nside, bins, `resolution_factor`) that a data vector
cannot be silently combined with one measured under a different estimator. After a crash, reopen
with `resume=True`: anything written past the last flush is discarded and appending continues.
`swmr=True` lets another process follow the file as it grows; it is off by default because a
crashed SWMR file keeps a stale write lock (`h5clear -s` releases it).

## Global (footprint-wide) estimators

Both estimators find their pairs once (per HEALPix tile, `tile_nside`) and reuse them for every
map, on CPU or GPU, with the same `preprocess` / `save_pairs` / `load_pairs` workflow as above.

### Two-point: `Global2PCF`

```python
from CosmoFuse import Global2PCF

g2 = Global2PCF(nside, mask=mask, nbins=20, theta_min=5, theta_max=250,
                resolution_factor=None, pack_pairs=False, device="auto")
g2.preprocess()                      # or g2.load_pairs("g2.h5") after g2.save_pairs(...)

gg = g2.gg(shear_maps, w, flip_g1=True)              # TwoPointResult
gg.xi_p, gg.xi_m                                      # (nz(nz+1)/2, nbins)
gg.combinations, gg.theta                             # row labels, bin centres (arcmin)
w_theta = g2.kk(delta_maps, w_cov).xi                 # KK field estimator (= LS with pixel randoms)
gamma_t = g2.kg(delta_maps, shear_maps, w_lens, w_source).xi
out = g2.three_by_two_pt(shear_maps, delta_maps, w_source, w_lens)   # {"gg", "kk", "kg"}

jk = gg.jackknife()["xi_p"]                           # delete-one-tile estimates (n_tiles, ncomb, nbins)
```

A `TwoPointResult` holds the per-tile sums; the estimate is their ratio over the whole footprint,
`jackknife()` gives the delete-one-tile estimates, and results of different tile sets or devices
add with `+`. The estimator is the ratio of sums over every pixel pair of the footprint and agrees
with TreeCorr's `GGCorrelation`, `KKCorrelation` (`k = δ`) and `KGCorrelation` on the pixel
centres (`brute=True`, `metric="Arc"`) to TreeCorr's rounding.

### Three-point: `Global3PCF`

```python
from CosmoFuse import Component, Global3PCF

g3 = Global3PCF(nside, mask=mask, nbins=12, theta_min=15, theta_max=250, n_max=20, device="auto")
g3.preprocess()

res = g3.ggg(shear_maps, w, flip_g1=True)            # ThreePointResult, labels gamma0..gamma3
res.num["gamma0"].shape    # (2 n_max + 1, nz, nz, nz, nbins, nbins): n, z_centre, z_leg2, z_leg3, b2, b3
gamma = res.binned(np.linspace(-np.pi, np.pi, 17))    # label -> (..., nbins, nbins, 16)
gamma_cen = res.to_centroid(gamma, np.linspace(-np.pi, np.pi, 17))

zeta = g3.kkk(delta_maps, w)                          # scalar 3PCF
kgg = g3.kgg(delta_maps, shear_maps, w_lens, w_source)  # <k g g>, <k g* g>
kkg = g3.kkg(delta_maps, shear_maps, w_lens, w_source)  # <k k g>
mixed = g3.measure(                                   # any spin-0 / spin-2 combination
    {"k": (delta_maps, w_lens, 0), "g": (shear_maps, w_source, 2)},
    [Component("gkk*", ("g", "k", "k"), (True, False, False))],
)
```

A triangle is seen from its centre vertex with legs in radial bins `b2`, `b3` and opening angle
`phi = phi_3 - phi_2`. The multipoles are exact for every `n`; only the conversion to
opening-angle bins (`binned()`) truncates at `n_max`. Spin-2 values are in the x-projection
(`Re` = tangential); `to_centroid` converts binned values to the centroid projection. Results
are sums, so tiles, chunks and devices add with `+`. Full resolution only; one device.

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
