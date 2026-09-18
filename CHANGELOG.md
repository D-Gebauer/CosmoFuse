# Changelog

## 4.21.0 (unreleased)

Default behaviour is unchanged: with default arguments every public method
returns bit-for-bit the results of 4.20.0 (verified on CPU, and on an A100
against 4.20.0 on the DES Y3 production geometry with real maps).

### Added
- **Static treecode** (`Correlation(..., resolution_factor=True)` for the
  default `k = 4`, or `resolution_factor=k`; opt-in, default `None` = full
  resolution). Every
  angular bin is measured on the coarsest HEALPix level whose pixel size is
  `<= theta_lo / k`; coarse cells are built per patch (exact top-hat patch
  window), carry weighted means / weight sums, and sit at the binary-mask
  centroid of their members. Pair geometry shrinks by orders of magnitude
  (nside 2048, 5'–175', 917 DES patches, k = 2.9: 592 M pairs / 17.7 GB instead
  of ~72 G pairs). **It is a different, windowed estimator** — see the README
  for the measured suppression per `k`. No CUDA kernel was changed for it:
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
- **Combination-tiled GPU xi+- kernel** (default; `kernel.tiled = False`
  restores the per-row kernel): one pass over the pairs for all tomographic
  combinations, bit-identical results, 4.2–4.5x faster on an A100.
- **Payload packing** (`pack_pairs=True`, opt-in): 8 instead of 24 bytes per
  pair on the device (uint16 rotation angles + uint16 patch-local row indices,
  per-patch contiguous row blocks; kernel `gpu_tiled_packed_reduce_xipm`).
  A100: pair memory 6.6 → 2.3 GB (nside 512 production), 14.5 → 5.2 GB
  (nside 2048, k = 2.9), speed unchanged; GPU vs CPU 5e-15. Not bit-identical
  to unpacked: estimates move by 3e-5 (rms) of their patch scatter, unbiased.
  Shear path + aperture statistics only on GPU for now.
- `RowSpaceMapLoader`: reader threads + ring of pinned buffers + upload
  stream; results identical to the serial loop, GPU idle < 5 %.
- `pair_search_precision=`: precision of the pair search, decoupled from the
  stored rotation precision. `"auto"` keeps the historical float32 search at
  full resolution and uses float64 with `resolution_factor`. A float32 search
  resolves separations only to `6e-8 / theta^2` (3 % at 5'); a warning is
  logged when that exceeds 1 % at `theta_min`.

### Changed
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
| workload | 4.20.0 | 4.21.0 |
|---|---|---|
| nside 512, 15'–176', full resolution, real maps from disk (bit-identical results) | 157 ms / map-set | 26 ms |
| same, `resolution_factor=4` | — | 11.5 ms |
| same, `resolution_factor=2.9` | — | 9.1 ms |
| nside 2048, 5'–175', `resolution_factor=2.9`, `aperture_nside=512` | impossible (~1.7 TB) | 78 ms (17.7 GB) |
