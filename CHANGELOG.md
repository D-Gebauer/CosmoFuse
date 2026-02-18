# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.1]

### Changed
- Refactored tomographic galaxy clustering (`vectorized_density_density`) to use backend fused tomo kernels directly, removing Python loops over redshift bins.
- Refactored tomographic galaxy-galaxy lensing (`vectorized_density_shear`) to use backend fused tomo kernels with explicit symmetric AB+BA permutation evaluation.
- Enforced GC auto-bin handling to avoid redundant BA evaluation and preserve single-term normalization for `i == j`.
- Enforced GGL symmetric normalization over both permutations for all bin pairs, including auto-bin pairs.

### Added
- New private helpers in `Correlation` for fused scalar tomography execution:
  - `_density_density_tomo_vectorized`
  - `_density_shear_tomo_vectorized`
- Additional sum-of-weights normalization helpers for tomographic per-combination and directional inputs.

### Tests & Docs
- Updated correlation unit tests to validate helper delegation, GC auto-bin uniqueness behavior, GGL AB+BA symmetry, and matching-bin validation for symmetric GGL tomography.

## [4.0.0]

### Changed
- Consolidated all `3.3.x` changes into a major release reflecting API and behavior updates since `3.3.0`.
- Pair finding and CPU/GPU correlation execution paths are now fully fused in hot loops to reduce temporary allocations and improve throughput.
- Tomographic correlation execution now uses fused vectorized kernels and cached combination buffers, with streamlined sum-of-weights handling and reuse.
- Public 2PCF API now uses explicit physics naming:
  - `compute_shear_shear` (replacing `xipm`)
  - `compute_density_density`
  - `compute_density_shear`
- Aperture API now uses explicit field type names:
  - `get_aperture_shear` (replacing `get_M_a`)
  - `get_aperture_density`

### Added
- Optimized scalar backend kernels for Spin-0 × Spin-0 and Spin-0 × Spin-2 pair correlations, with CPU and CuPy implementations.
- Fused scalar tomographic kernels for density-density and density-shear on CPU/GPU backends.
- New 3x2pt tomography convenience API:
  - `get_3x2pt_tomo(shear_maps=None, density_maps=None, weights=None)`
  - Returns `(M_ap, N_ap, xipm, wtheta, gammat)` with map-type-dependent `None` values.
- New public vectorized helpers:
  - `vectorized_shear_shear`
  - `vectorized_density_density`
  - `vectorized_density_shear`
- `prepare(release_host_pairs=True)` host-memory release option for large runs.

### Removed
- Deprecated/public legacy interfaces removed:
  - `xipm`
  - `get_M_a`
  - `get_full_tomo(low_mem=...)` fallback mode
  - older wrapper classes and obsolete helper utilities removed across `3.3.x`

### Tests & Docs
- End-to-end test coverage now includes shear-shear (GG), density-density (GC), and density-shear (GGL) parity checks against TreeCorr.
- README and tests were updated throughout for explicit method naming and 3x2pt usage.

## [3.3.7]

### Added
- Added `Correlation.get_3x2pt_tomo(shear_maps=None, density_maps=None, weights=None)` returning a flat tuple:
  - `(M_ap, N_ap, xipm, wtheta, gammat)`
  - Outputs tied to absent map types are returned as `None`.
- Added `vectorized_shear_shear`, `vectorized_density_density`, and `vectorized_density_shear` public helper methods for tomographic bundle computation.

### Changed
- `get_3x2pt_tomo` now accepts flexible weight inputs (`None`, dict, tuple/list, or single shared 2D array when bin counts match) and normalizes to explicit shear/density weight arrays.
- Updated tests and README with direct 3x2pt tomography usage and branch coverage.

## [3.3.6]

### Changed
- **Breaking change**: renamed `Correlation.xipm` to `Correlation.compute_shear_shear` and removed the old `xipm` alias.
- Updated tests and README examples to use physics-explicit 2PCF naming.

### Added
- Added direct 2PCF methods in `Correlation`:
  - `compute_density_density` (Spin-0 × Spin-0)
  - `compute_density_shear` (Spin-0 × Spin-2)
- `compute_density_density` now calls the optimized scalar backend kernel path without passing rotation arrays.
- `compute_density_shear` now passes only source-side rotation (`exp2phi_dev[1]`) to optimized backend kernels.

## [3.3.5]

### Added
- Added highly optimized scalar correlation kernels in `backend.py` for:
  - Spin-0 × Spin-0 (`kernel_density_density`) using pure float accumulation and no rotation-array loads.
  - Spin-0 × Spin-2 (`kernel_density_shear`) loading only source-side rotations (`exp_j`) for direct tangential projection.
- Added fused-reduction tomography kernels for both scalar paths on CPU and GPU:
  - `kernel_density_density_tomo_vectorized`
  - `kernel_density_shear_tomo_vectorized`

### Changed
- Extended backend wiring so CPU/CuPy backends expose scalar pair and tomographic kernels through the unified `Backend` interface.
- Updated tests and README to cover/document the new scalar kernel paths.

## [3.3.4]

### Changed
- **Breaking change**: renamed `Correlation.get_M_a` to `Correlation.get_aperture_shear`.
- Added `Correlation.get_aperture_density` for scalar (spin-0) fields (e.g. galaxy density / convergence).
- Added dedicated backend aperture-density kernels for CPU (Numba) and GPU (CuPy) that compute weighted scalar apertures from `Q_val`, map values, and weights.

### Removed
- Removed deprecated `M_a_patch` helper from `correlation_helpers.py`.
- Removed backward compatibility alias for `get_M_a`.

## [3.3.3]

### Changed
- Replaced the tomographic CuPy RawKernel path with a true fused-reduction implementation that accumulates directly into per-bin numerators instead of materializing per-pair tomographic scratch arrays.
- Optimized fused tomographic GPU execution by removing output atomics where a single block owns each output element.
- Optimized fused tomographic GPU execution by skipping BA-orientation work for auto-bin combinations (`i == j`).
- Updated fused tomographic accumulation to write real-valued numerators directly (matching downstream normalization) instead of complex numerators.
- Added cached tomographic combination-index buffers (`comb_i`/`comb_j`) in `Correlation` to avoid rebuilding/transferring them on repeated `get_full_tomo` calls.
- `get_full_tomo` now always uses the fused tomographic kernel path and no longer accepts a `low_mem` option.

### Removed
- Removed the legacy `get_full_tomo` low-memory fallback implementation (`_xipm_auto`/`_xipm_cross` loop path).
- Removed redundant tests in `tests/test_unified.py` that duplicated backend device-selection behavior already covered in `tests/test_backend.py`.

## [3.3.2]

### Changed
- Refactored CPU `xipm` kernels (`_cpu_xipm_auto_corr_kernel*`, `_cpu_xipm_cross_corr_kernel*`) to use fused map-reduce over precomputed bin offsets, removing large per-pair intermediate arrays.
- Refactored CPU vectorized tomography kernel (`_cpu_vectorized_tomo_kernel`) to accumulate directly into per-bin/per-combination outputs.
- Updated CPU call sites in `Correlation` to pass bin-offset segments into CPU kernels and consume reduced outputs directly (no extra `_reduce_pairs` pass for CPU paths).

### Fixed
- Reduced peak CPU memory usage for large pair catalogs by eliminating `(n_pairs, ...)` and `(n_combs, n_pairs)` temporary correlation buffers in CPU execution paths.

## [3.3.1]

### Changed
- Removed redundant padding/allocation in `Correlation._reduce_pairs` by applying `add.reduceat` directly to the pair-value array.
- Enabled CuPy ElementwiseKernel fast-math compiler option (`--use_fast_math`) for fused GPU correlation kernels.
- `Correlation.prepare` now supports optional host-memory release via `release_host_pairs=True`.

### Added
- `Correlation.save_pairs` now warns and returns without writing when host pair arrays were previously released.
- Tests covering CuPy fast-math kernel options and host-pair memory-release behavior.

## [3.3.0]

### Changed
- Replaced dense O(N²) pair-distance matrix construction in `Correlation.get_pairs_patch` with an on-the-fly Numba kernel (`_compute_pairs_numba`) that computes angular distances and phase rotations per candidate pair.
- Inlined spherical-trigonometry rotation math directly in the pair kernel to avoid helper-call overhead and keep the hot loop fully JIT-optimized.
- Pair precomputation now respects the `Correlation.fastmath` setting by dispatching to a precise (`fastmath=False`) or fast (`fastmath=True`) Numba kernel variant.
- `Correlation.preprocess` no longer accepts a `threads` argument and now uses internal default pair-precompute execution paths.

### Fixed
- `pixel2RaDec` now normalizes unsigned index arrays (including `uint64`) to a Healpy-compatible integer dtype before calling `pix2ang`, fixing preprocessing failures when `index_precision='uint64'`.

### Removed
- Removed obsolete helper functions `getAngle` and `radec_to_xyz` from `correlation_helpers.py`.
- Removed unused `_compute_pairs_numba` argument `complex_dtype`.
- Removed multiprocessing configuration from `Correlation` (`multiprocessing_start_method`).
- Removed `threads` arguments from `calculate_pairs_2PCF` and `calculate_pairs_M_a`.

### Tests
- Added targeted tests for worker initialization, early-return edge paths in pair finding, and kernel edge branches.

## [3.2.0]

### Changed
- Aperture-mass computation now flattens per-patch inputs and runs in a single JIT kernel to reduce Python dispatch overhead.
- `get_full_tomo` uses fused cross-correlation kernels to avoid redundant pair traversal on GPU and CPU.
- `xipm` uses a JIT-compiled CPU kernel for pairwise accumulation.

## [3.2.1]

### Changed
- `xipm` now routes auto- vs cross-correlation paths internally and reuses fused kernels when needed.
- `get_full_tomo` now delegates auto/cross handling to the refactored `xipm` helpers without external averaging.

## [3.1.0]

### Changed
- Full-tomography (`get_full_tomo`) is now the supported high-level path for correlation measurements in tests and README examples.
- `calculate_pairs_M_a` no longer accepts a `threads` argument and now reports progress with `tqdm`.
- Precision options now support `float32`/`float64` for map and rotation arrays.
- Default `rotation_precision` changed from `float64` to `float32`.
- `preprocess()` and `load_pairs()` now call `prepare()` automatically, so manual `prepare()` is no longer required in the standard workflow.
- `xipm` now computes and caches `sumofweights` automatically when not provided, while still accepting an explicit override.

### Removed
- `Correlation.load_maps`.
- `Correlation.get_all_xipm`.
- `Correlation.calculate_2PCF`.

## [3.0.1]

### Changed
- `Correlation.preprocess` shows progressbar (tqdm) when using `threads=1`
- `Correlation.get_full_tomo` now supports automatic computation and caching of tomographic `sumofweights` when not provided.
- Repeated `get_full_tomo` calls with unchanged `w` now reuse cached `sumofweights`; changed `w` triggers recomputation.
- `get_full_tomo` continues to accept explicit `sumofweights` as a manual override.
- README tomographic usage example now shows `get_full_tomo(shear_maps, w)` as the default call.

### Removed
- deprecated wrapper classes `Correlation_GPU`, `Correlation_GPU_lowmem`, `Correlation_CPU`
- unneeded functions in `utils.py`
- outdated plotting routines from `visualisation.py` 


## [3.0.0]

### Added
- Precision configuration for core data categories in `Correlation`:
  - `map_precision` (`float16`, `float32`, `float64`)
  - `rotation_precision` (`float16`, `float32`, `float64`)
  - `index_precision` (`uint32`, `uint64`)

### Changed
- `Correlation` now uses logging in preprocessing status messages instead of direct prints.
- README usage now documents the unified `Correlation` API and `load_pairs`.

### Removed
- Deprecated `Correlation_GPU_lowmem` wrapper class.
- Package-level export of deprecated `Correlation_GPU` from `CosmoFuse.__init__`.

## [0.2.1]

### Added
- Comprehensive type hints throughout the codebase
- Proper docstrings following Google/NumPy style
- Input validation and error handling
- Logging support
- Comprehensive test suite
- Makefile for common development tasks
- Proper package structure with __init__.py exports

### Changed
- Updated pyproject.toml with proper metadata and dependencies
- Fixed naming conventions to follow PEP 8
- Improved code formatting and style
- Replaced print statements with proper logging
- Enhanced project structure and organization

### Fixed
- Import statements and package exports
- Variable naming conflicts (e.g., `bin` -> `bin_idx`, `min` -> `min_idx`)
- Code style issues and formatting

## [0.2.0]

### Added
- Initial implementation of integrated 3-point correlation functions
- GPU support via CuPy
- CPU implementation with multiprocessing support
- Visualization utilities
- Basic correlation function calculations
