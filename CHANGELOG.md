# Changelog

## 6.1.0 (2026-09-18)

### Measured (no library change)
- **The aperture pass: both of its levers, quantified.** `gpu_aperture_shear_tomo`
  is the largest non-pair item in `get_full_tomo_shear` (0.450 ms of 1.796 ms at
  nside 512) and the only one the treecode does not shrink. Two benchmarks
  settle what can be done about it; results in
  `benchmarks/static_treecode/APERTURE_RESULTS.md`.
  - `benchmarks/static_treecode/aperture_reuse_probe.py` (new) answers whether
    the kernel really pays for re-reading the disc geometry once per tomographic
    bin. `ncu` cannot say -- performance counters need root on the A100 node
    (`ERR_NVGPUCTRPERM`) -- so it compiles probe kernels and measures the end
    state instead. It does pay: one block per patch instead of one per
    (patch, bin) is **1.75x** at nside 512 (0.451 -> 0.258 ms) and **2.08x** at
    nside 2048, exactly the 112 -> 64 B/pixel traffic ratio, and **bitwise
    identical** to the shipped kernel in every configuration. AoS wins only at
    nside-2048-sized discs, so the scope-chosen layout stays as it is. The
    kernel itself is NOT changed here.
  - `t10_zeta_level.py` gained `--aperture-nside`, so the zeta price of a coarse
    aperture can be measured the way T10 measured the price of a coarse pair
    level. With `resolution_factor=None` only M_ap moves: base 512 -> aperture
    128 costs **0.07 sigma_patch** (aperture 256: 0.04), against the 0.27
    sigma_patch at which treecode k = 2.9 was accepted. That run is 4x more
    aggressive in pixel/theta_Q than pinning the aperture to 512 at base nside
    2048, which is therefore safe by a wide margin -- and worth **20x** on that
    pass (9.10 -> 0.455 ms). Unlike `resolution_factor` the effect is flat in
    theta, so the two knobs are not interchangeable.

### Performance
- **The row expansion writes the kernels' layout, and carries the sign
  flip** (idea #3, the remaining two sub-items). Between the degrade and
  the pair kernels sat three more passes over the map-set: a stack that
  glued the two shear components back together, a SoA -> AoS transpose
  (0.077 ms; the packed `perm` gather adds another 0.105 ms), and --
  whenever `flip_g1`/`flip_g2` was set -- a scale-and-stack copy that
  `get_full_tomo_shear` made *twice*, once per leaf (0.257 ms, **15 %** of
  the call; the production script runs `flip_g1=True`). The fused degrade
  now takes four element strides per buffer instead of one, so it writes
  the interleaved `(nz, 2, n_rows)` or AoS `(n_rows, nz, 2)` layout each
  call wants directly, and applies the sign while it copies the pixel rows
  -- the cell rows inherit it, because level 0 reads the already-signed
  rows. `get_3x2pt_tomo` no longer stages its four inputs through separate
  cached buffers at all (`ComputeContext`'s `fused_*_soa` are gone).
  Measured on the A100 (nside 512, 450 patches, 18.3 M pairs, k = 2.9,
  4 tomographic bins, float32 maps + float64 accumulators):

  | | before | after |
  |---|---|---|
  | `_expand_shear_rows` | 0.361 ms | **0.274 ms** |
  | `vectorized_shear_shear` | 1.439 ms | **1.320 ms** |
  | `vectorized_shear_shear`, `flip_g2` | 1.560 ms | **1.338 ms** |
  | `get_full_tomo_shear`, `flip_g2` | 1.973 ms | **1.796 ms** |
  | `get_3x2pt_tomo` | 4.592 ms | **4.439 ms** |
  | cost of a flip, `get_full_tomo_shear` | 0.257 ms | **0.090 ms** |
  | cost of a flip, `vectorized_shear_shear` | 0.121 ms | **0.018 ms** |

  `get_full_tomo_shear` *without* a flip is unchanged (1.716 -> 1.707 ms,
  within the run-to-run scatter): its aperture pass keeps the SoA layout,
  so its 2PCF leaf still transposes. That is deliberate and measured --
  `aperture_tomo.cu` gathers aperture discs, whose row ids are largely
  contiguous, and AoS costs it **0.450 -> 0.600 ms**, far more than the
  0.077 ms transpose it would save. The layout is therefore chosen by the
  enclosing call (`_expansion_scope(layout=...)`) so that all of its
  leaves agree and one degrade still serves the whole call; the aperture
  kernels take a second element stride so either layout *can* be passed.
  A public AoS *input* layout was rejected: it only helps at full
  resolution, and it would add a second accepted layout to
  `_coerce_map_input_array`, `MapLoader`, `ZetaWriter` and every
  row-space gate. Both changes are pure data movement, and the tests
  require **bitwise** equality with the code they replace: a sign is +-1
  so scaling is exact, the degrade is linear in the values, and a
  transpose moves floats without touching them
  (`tests/test_row_layouts.py`).
- **Fused static-treecode row degrade** (idea #3). Building the virtual
  rows was a chain of sparse matrix products over an
  `(n_active, K * n_lead)` temporary plus two transposes. Two new CUDA
  kernels (`cuda/degrade_rows.cu`) walk each level's CSR children directly
  into an `(n_lead, n_appended)` accumulation-dtype scratch and then
  normalise and scatter into the row buffers: no pair-sized temporary, no
  transpose, no atomics. The number of lanes cooperating on one cell is
  sized from the mean number of children - a treecode level halves nside,
  so a cell has exactly 4 children and a full warp per cell idled 87 % of
  its lanes (2.1x instead of 3.9x). On an A100 (nside 512, 450 patches,
  18.3 M pairs, `resolution_factor=2.9`, 4 tomographic bins, float32 maps
  + float64 accumulators): `_expand_rows` **1.071 -> 0.273 ms (3.9x)**,
  `get_full_tomo_shear` 2.533 -> 1.721 ms (-32 %), `get_3x2pt_tomo`
  6.069 -> 4.584 ms (-24 %). It is the same recursion as the sparse chain
  and differs only in the order of summation inside a cell: the weight
  rows come out bitwise identical at float32 and the value rows agree to
  one ULP (5.96e-8 at float32, **2.0e-16 at float64**). The sparse chain
  stays as the CPU path and the fallback, and `resolution_factor=None`
  reaches neither. Validated against TreeCorr run on the degraded cells
  themselves (`tests/test_degrade_treecorr.py`).

### Added
- **`ZetaWriter`**: streams each map-set's i3PCFs to HDF5 from a background
  thread instead of holding every per-patch array in RAM. zeta averages over
  patches, not over maps, so map-set *k* can be reduced as soon as it is
  measured. At the DES Y3 production geometry that is 9 kB per map-set
  instead of 1.06 MB (90 MB instead of 10.6 GB for 10,000 realisations);
  `reduce="none"` keeps the per-patch arrays, from which zeta, a jackknife
  or a different binning can still be derived. On a GPU the reduction runs
  on the device before the copy, so only the data vector crosses PCIe; with
  a `MultiDeviceCorrelation` (host arrays) it runs on the CPU in the writer
  thread. The file carries `n_flushed` plus `level_table` / `row_pix_hash`
  provenance, and `resume=True` continues after a crash. `swmr=True` is
  available but off by default — it trades a worse crash story (stale write
  lock, needs `h5clear -s`) for a live view of the file.
- The zeta reduction in `correlation_helpers` now dispatches on the array
  module, so it runs on cupy arrays without a host round-trip.

## 6.0.0 (2026-09-18)

Compatibility cleanup: everything that existed only to keep pre-5.0 code,
pickles and files working is gone. No measurement path was rewritten —
except the one estimator inconsistency listed first.

### Removed / breaking

- **`compute_shear_shear` now uses the same cross-bin estimator as
  everything else.** For a cross pair it took the mean of the two
  orientation *ratios*, `(N_ab/W_ab + N_ba/W_ba) / 2`, while every
  tomographic method moved to the ratio of the summed orientations,
  `(N_ab + N_ba) / (W_ab + W_ba)`, in 5.0. It now does the same, so the
  package measures one ξ± estimator. Auto combinations, and cross
  combinations with an explicit `sumofweights`, are unchanged; with
  different per-bin weights the estimate moves (~3 % of max|ξ+| per patch
  on DES Y3, 0.02 σ). Gated by
  `tests/test_correlations_core.py::TestCrossOrientationEstimator`.
- **ξ± no longer follows the rotation precision.** The reduced numerators
  were cast back to float32 whenever `rotation_precision="float32"`; the
  returned estimates now carry the accumulation dtype, like every other
  statistic. Values are unchanged where the cast was lossless.
- Deprecated names removed: `PinnedMapPipeline` (use `MapLoader`),
  `RowSpaceMapLoader` (`MapFileLoader`), `Correlation.precompute()`
  (`preprocess()`), `Q_T` (`Q_crittenden`).
- `select_patch_centers` / `Correlation.from_mask`: the `filter_weighted`
  argument is gone (use `filter_weighting`), as are the values `"raw"`
  (use `"signed"`) and `"pixels"`. A constant `aperture_filter` reproduces
  the plain pixel fraction exactly if you want it.
- `pair_search_precision` accepts only `"float32"` and `"float64"`;
  `"rotation"` and `"auto"` are gone. `"float64"` remains the default.
- Pair files: **format version 1** (one HDF5 group per patch) is no longer
  read; `load_pairs` raises and says to convert with 5.x. Versions 2-4 are
  unaffected.
- `Correlation.__setstate__` no longer migrates pickles written by older
  versions; it only rebuilds what `__getstate__` drops. Re-pickle with 5.x
  first if you hold old pickles. `ComputeContext.ensure_runtime_state()`
  was removed with it.
- The aperture-filter cache key for the default filter is `"Q_crittenden"`
  instead of `"Q_T"` (internal, visible in pickled state).

### Kept deliberately

Full-sky map inputs (they are a convenience, not a compatibility shim),
the per-(bin, row) GPU kernels and `kernel.tiled = False` (they are the
fallback beyond the tile's accumulator budget), the ElementwiseKernel and
CPU fallbacks (they cover missing hardware, not old versions), and the
bit-for-bit guarantee of `resolution_factor=None`.

## 5.1.0

### Added
- **Multi-GPU by patch range.** `Correlation(..., device=[0, 1, ...])` returns a
  `MultiDeviceCorrelation`: one `Correlation` per device over a contiguous
  range of patches, measurement calls run one thread per device and the
  per-patch outputs are concatenated in patch order. Results are bitwise
  identical to a single-device run, and device memory per GPU scales as
  `n_patches / n_devices`. A single device stays the default and is
  unaffected. `load_pairs()` gives each device its own slice of one pair
  file; `save_pairs()` is not supported on a group (write it from a
  single-device instance).

- **`pack_host_pairs=`** (opt-in, default `False`): applies the payload
  packing of `pack_pairs` already at pair-finding time, so the host arrays
  and the pair file hold 8 instead of 24 bytes per pair as well
  (`pair_inds` / `pair_exp2phi` become `None`; the payload is in
  `packed_pairs`). The row blocks store global ids, so a packed file is
  still mask-independent and still slices by patch. Independent of
  `pack_pairs`, and bitwise identical on the device to packing inside
  `prepare()`. The price: the geometry is quantised everywhere, so the file
  is no longer exact and cannot be turned back into one. Such files are
  written as **format version 4** and carry a `packed_pairs` attribute;
  earlier CosmoFuse versions reject them instead of misreading them.

### Changed
- `PinnedMapPipeline` is now **`MapLoader`** and `RowSpaceMapLoader` is now
  **`MapFileLoader`**. The old names still work and warn.

### Performance
- **Pair search: no more sorting.** The search kernel counted accepted pairs
  per row and the result was then put in angular-bin order with an `argsort`
  plus seven fancy-indexed gathers — at nside 2048 that was 97 % of the
  per-patch preprocessing time and ~600 MB of temporaries per patch. The
  kernel now counts per (row, bin) and writes the pairs already grouped by
  bin, in exactly the same order as the stable sort produced. Preprocessing
  is **3.3× faster at nside 512** (16.4 → 5.0 ms/patch) and **6.6× faster at
  nside 2048** (9.1 → 1.4 s/patch, i.e. ~2.3 h → ~21 min for 917 patches).
  Pair output is unchanged bitwise.
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
