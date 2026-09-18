# Changelog

## Unreleased

### Performance
- **Batched normalisation of the tomographic wrappers.** `vectorized_shear_shear`,
  `vectorized_density_density`, `vectorized_density_shear` and the device path
  of `get_3x2pt_tomo` normalised one tomographic combination at a time, which
  issued roughly a dozen tiny elementwise kernels per combination; the GPU was
  idle waiting for launches. The whole stack is now normalised in one set of
  operations. Results are bitwise unchanged (verified on an A100 over 56
  DES-Y3-sized map-sets). On an A100 at nside 512, 917 patches, 4 source bins,
  `resolution_factor=2.9`: `vectorized_shear_shear` 4.40 → 3.03 ms and
  `get_full_tomo_shear` 5.84 → 4.40 ms per map-set with device-resident input.

### Testing
- The handful of exactness gates that dominate the suite wall time are marked
  `slow`; `pytest -m "not gpu and not slow"` runs 308 of 317 tests in ~1 min
  instead of ~5.5 min. CI still runs them (it only deselects `gpu`).

## 5.0.0 (2026-09-18)

Major release: static treecode, compact row space, combination-tiled and
packed GPU kernels for every pair statistic, ring-buffer map loader.

**Breaking / result-changing (hence 5.0):**
- cross-bin xi+- is now the ratio of the summed orientations (was the mean of
  the two orientation ratios) — see *Changed*;
- the pair search runs at float64 by default (`pair_search_precision`);
- `select_patch_centers` / `Correlation.from_mask` select patches by
  |filter|-weighted masked fraction by default;
- map arrays of the wrong length raise; pair files with virtual rows use
  format version 3 (unreadable by <= 4.20 by design);
- internal kernel contracts changed (one output row per combination; the
  GPU `kernel_3x2pt_tomo_fused` wrapper became `kernel_3x2pt_tomo_aperture` +
  `kernel_3x2pt_tomo_pairs`).

Unchanged: with explicit patch centres and `pair_search_precision="rotation"`
the auto-combination results of every public method are bit-for-bit those of
4.20.0 (verified on CPU, and on an A100 against 4.20.0 on the DES Y3
production geometry with real maps); existing full-resolution pair files
load as before.

### Added
- **Static treecode** (`Correlation(..., resolution_factor=True)` for the
  default `k = 4`, or `resolution_factor=k`; opt-in, default `None` = full
  resolution). Every
  angular bin is measured on the coarsest HEALPix level whose pixel size is
  `<= theta_lo / k`; coarse cells are built per patch (exact top-hat patch
  window), carry weighted means / weight sums, and sit at the binary-mask
  centroid of their members. Pair geometry shrinks by orders of magnitude
  (nside 2048, 5'–175', 917 DES patches, k = 2.9: 592 M pairs / 17.7 GB instead
  of ~72 G pairs). **It is a different, windowed estimator** — see
  `benchmarks/static_treecode/STAGE1_RESULTS.md` for the measured suppression
  per `k`. No CUDA kernel was changed for it:
  cells are appended as virtual rows behind the pixel rows.
- `aperture_nside=`: aperture statistics on the (weighted-mean) degraded map.
- `Correlation.level_table`: per-bin resolution provenance; written to pair
  files and pickles. `load_pairs` adopts the file's estimator and raises on an
  explicitly conflicting request.
- Preflight memory check before pair finding (`memory_budget_gb=`): raises a
  `MemoryError` naming a `resolution_factor` that fits.
- **Compact row space**: device map buffers and device indices cover only the
  unmasked pixels. Measurement methods accept full-sky *or* row-space maps
  (`Correlation.row_pix`, `n_active`, `to_row_space()`, `row_pix_hash`).
  Masked pixels are never read (they may be NaN). Read-only weight maps are
  gathered / uploaded / degraded once.
- **Combination-tiled GPU pair kernels** for xi+-, xi_g and xi_t (default;
  `kernel.tiled = False` restores the per-row kernels): one pass over the
  pairs for all tomographic combinations, shared tiles in
  `cuda/pair_tiles.cuh`. The fused 3x2pt path (`get_3x2pt_tomo`) runs its
  pair statistics in the same tiled kernels (its own kernel keeps only the
  aperture sections). Auto-only (`gc_auto_correlations_only`) and subset
  (`ggl_bin_combinations`) requests are served by the tiles too.
  `get_3x2pt_tomo` walks the pairs **once** for all three statistics
  (`cuda/tomo_tiled_3x2pt.cu`, up to 120 accumulators per thread; 8–20 %
  faster than three passes for 4 + 4 bins on an A100). xi+-:
  bit-identical auto combinations, 4.2–4.5x faster on an A100.
- **Payload packing** (`pack_pairs=True`, opt-in): 8 instead of 24 bytes per
  pair on the device (uint16 rotation angles + uint16 patch-local row indices,
  per-patch contiguous row blocks; kernels `gpu_tiled_packed_reduce_{xipm,dd,ds}`).
  A100: pair memory 6.6 → 2.3 GB (nside 512 production), 14.5 → 5.2 GB
  (nside 2048, k = 2.9), speed unchanged; GPU vs CPU 5e-15. Not bit-identical
  to unpacked: estimates move by 3e-5 (rms) of their patch scatter, unbiased.
  Serves every tomographic method (`vectorized_*`, `get_full_tomo_*`,
  `get_3x2pt_tomo`) and the aperture statistics on the GPU; the single-map
  `compute_*` methods need `pack_pairs=False`.
- `RowSpaceMapLoader`: reader threads + ring of pinned buffers + upload
  stream; results identical to the serial loop, GPU idle < 5 %.
- `pair_search_precision=`: precision of the pair search, decoupled from the
  stored rotation precision; **default `"float64"`** (no memory cost).
  `"rotation"` restores the historical search at rotation precision (float32
  by default), `"auto"` keeps that at full resolution and uses float64 with
  `resolution_factor`. A float32 search resolves separations only to
  `6e-8 / theta^2` (3 % at 5'); on the DES Y3 nside-512 production geometry it
  mis-bins pairs at the 0.05 sigma (rms) / 0.9 sigma (max) per-patch level
  against the exact result. A warning is logged when the jitter exceeds 1 %
  at `theta_min`.

### Changed
- **Cross-bin xi+- estimator.** For a tomographic combination (i, j) with
  i != j every pair contributes in both orientations (bin i at pixel a and j
  at b, and vice versa). CosmoFuse used to average the two orientation
  *ratios*, `(N_ab/W_ab + N_ba/W_ba) / 2`; it now takes the ratio of the
  summed orientations, `(N_ab + N_ba) / (W_ab + W_ba)` — the standard
  weighted estimator, TreeCorr's definition, and what xi_g and xi_t already
  did. The two coincide for equal per-bin weights; with the real per-bin DES
  Y3 weights they differ by ~3 % of max|xi+| per patch (0.02 sigma). The old
  form also halved a bin's estimate wherever one orientation had zero weight
  (bins with different masks). Auto combinations are unchanged. Applies to
  `vectorized_shear_shear`, `get_full_tomo_shear` and `get_3x2pt_tomo` on
  both backends; an explicit directional `sumofweights` is summed the same
  way. The single-map `compute_shear_shear` keeps its historical form.
- **Pair search precision default** is `"float64"` (see *Added*).
- **Patch selection default** (`select_patch_centers`, `Correlation.from_mask`):
  the filter-support masking check now uses the |filter|-weighted masked
  fraction by default (`filter_weighting="abs"`); `"signed"`/`"raw"` measures
  the deviation of the filter integral instead, `"pixels"` restores the old
  default. Always from the binary mask. New `U_crittenden`, `U_schneider`.
  Explicitly passed patch centres and existing pair files are unaffected.
- Map arrays whose last axis is neither `npix` nor `n_active` now raise a
  `ValueError` (previously undefined behaviour / out-of-bounds gathers on GPU).
- Pair file format: files **with virtual rows** are written as
  `format_version = 3` and store their index datasets under new names
  (`tc_pair_inds`, `tc_Q_inds`). CosmoFuse <= 4.20 accepts any version >= 2
  and would read virtual-row ids as pixel ids; with the new names it fails
  with a `KeyError` instead. Full-resolution files are still written as
  version 2 and remain readable by old versions. Unknown future versions are
  now rejected.

### Fixed
- Cumulative pair offsets were built as int32 and overflowed beyond 2^31
  pairs; now int64.
- `PinnedMapPipeline`: (i) a pinned host slot could be refilled while its
  previous asynchronous copy was still queued, and (ii) the upload stream did
  not wait for queued kernels that still read the device slot it overwrites.
  Both only bite when results stay on the device and the host runs ahead of
  the GPU. Covered by a CUDA stream simulator test (`tests/cuda_stream_sim.py`).

### Performance (A100 80 GB, DES Y3, 917 patches, 4 source bins, float32 maps + float64 accumulators)
| workload | 4.20.0 | 5.0.0 |
|---|---|---|
| nside 512, 15'–176', full resolution, real maps from disk (bit-identical results) | 157 ms / map-set | 26 ms |
| same, `resolution_factor=4` | — | 11.5 ms |
| same, `resolution_factor=2.9` | — | 9.1 ms |
| nside 2048, 5'–175', `resolution_factor=2.9`, `aperture_nside=512` | impossible (~1.7 TB) | 78 ms (17.7 GB) |
| `get_full_tomo_density`, nside 512, 4 lens bins, row-space host input (4.20 column: its per-row kernels) | 31.5 ms | 8.4 ms |
| `get_full_tomo_ggl`, 4 lens x 4 source bins | 96.4 ms | 23.6 ms |
| `get_3x2pt_tomo`, 4 + 4 bins | 203 ms | 38.7 ms |
| `get_3x2pt_tomo`, nside 2048, `resolution_factor=2.9`, packed | impossible | 266 ms |
