# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
