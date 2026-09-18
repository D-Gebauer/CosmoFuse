# T10 results — zeta level (2026-09-18)

`t10_zeta_level.py --measure [--k K]` → `results/t10_{fine,kK}.npz`,
`--analyse --k K` → `results/T10_kK.md` (full per-triplet tables).

Stage 1 answered the level question for the **2-point** functions on Gaussian
fields. T10 is the same question for the **i3PCF data vector** on the real
measurement path: same maps, same patches, same binning, only
`resolution_factor` changed.

Setup: DES Y3 mask at nside 512, Q110 patch set (917 patches), 15–250′ in 8
log bins (edges ≥ 2R−5 dropped → 7 bins, 18′…148′ centres), 4 source bins,
`map_precision="float32"`, `accumulation_precision="float64"`,
`aperture_nside=None` (the aperture stays at full resolution, so *only* ξ±
changes). Maps: 56 baryonified N-body shear map-sets (`grid_baryonified/
4BINS/gamma_{0000,0001}.npy`, 7 realisations × 4 footprint placements each),
**noise-free** — with shape noise every σ below grows, so the numbers here
are the conservative case.

ζ_a,± = ⟨M_a ξ±⟩_patches − ⟨M_a⟩⟨ξ±⟩ for all 20 triplets × 7 bins = 140
entries per statistic. Two yardsticks:

* **σ_patch** — the patch-to-patch error of *one* map-set
  (std of the per-patch contributions / √917). This is what a single
  DES-like measurement can resolve.
* **paired scatter over the 56 map-sets** — how well the shift itself is
  measured here.

| | pairs | ms / map-set | max abs(ratio−1), ζ_a,+ | ζ_a,− | max abs(Δζ)/σ_patch, ζ_a,+ | ζ_a,− | quadrature over the 140 entries, ζ_a,+ | ζ_a,− |
|---|---|---|---|---|---|---|---|---|
| full resolution | 265.4 M | 35.9 | — | — | — | — | — | — |
| k = 2.9 | 39.0 M | 37.2 | 0.14 | 0.28 | **0.23** | **0.27** | 0.6 σ | 1.0 σ |
| k = 1.5 | 7.4 M | 35.5 | 0.33 | 0.59 | 0.29 | **0.81** | 1.2 σ | 3.7 σ |

Level tables (nside per bin):
k = 2.9 → `[512, 512, 512, 256, 256, 128, 128]` (k_eff 2.2/3.1/4.4/3.1/4.5/3.2/4.5),
k = 1.5 → `[512, 256, 256, 128, 128, 64, 64]` (k_eff 2.2/1.6/2.2/1.6/2.2/1.6/2.3).

## 1. k = 2.9 (the benchmarked setting) passes

Bins that stay at the base nside reproduce the full-resolution ζ **bitwise**
(ratio 1.000 exactly) — the expected behaviour of the degenerate level and a
useful end-to-end check of the level machinery on the real path.

In the coarsened bins the shift stays below **0.3 σ_patch** for every triplet
and every bin, in both ζ_a,+ and ζ_a,−, and it is not resolved at 3 σ by 56
noise-free map-sets (max 2.9 σ, 0 of 140 entries). The 1-D ξ± suppression at
k_eff ≈ 3 (−8…−10 % on ξ−, ≲ 1 % on ξ+, Stage 1 §1) propagates into ζ, but
the i3PCF is noisy enough per bin that this is invisible in a single
measurement.

## 2. k = 1.5 does not

ζ_a,− is suppressed coherently by 20–40 % in every coarsened bin (ratios
0.43–0.99, mostly ~0.7–0.9), reaching **0.81 σ_patch** and 3.7 σ in
quadrature; 21 of 140 entries are resolved above 3 σ against the map-set
scatter. ζ_a,+ stays harmless (≤ 0.29 σ_patch) — the same statistic-by-
statistic pattern as the 2-point result, ξ− being the one that feels the
cell window.

## 3. Consequences

* The Stage 1 recommendation carries over unchanged to the 3-point level:
  **k ≳ 3 is safe, k ≲ 1.5 is not**, and ξ−/ζ_a,− sets the limit.
* ζ_a,+ is insensitive to the level over the whole range tested.
* This does **not** license mixing resolutions: the k = 1.5 case shows the
  shift is a coherent window effect, not noise. Data, simulations and
  covariance must all be measured at the same `resolution_factor`.
* The ms/map-set figures in the table above are the *whole host loop*
  (slicing a float64 archive, casting, uploading, measuring) and are
  dominated by that host work, which is why they barely move with k. The
  measurement call itself goes 11.4 → 4.4 ms device-resident, 13.6 → 7.8 ms
  with a host row-space array (see the optimisation record in
  `CLAUDE.md`; the retired `OPTIMISATION_IDEAS.md` is in git history at
  `5b556d7`).

## 4. T11 (float16 archives), same harness

`--f16` rounds each map-set to float16 and back before measuring. Over the
same 56 map-sets at full resolution: max |Δζ| / σ_patch = **1.2e−4** for both
ζ_a,+ and ζ_a,−, with ξ± and M_ap shifting by ~2e−4 relative. A float16 map
archive is therefore free of consequence for the data vector — four orders of
magnitude below the statistical error.
