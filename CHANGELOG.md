# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1]

### Added
- Comprehensive type hints throughout the codebase
- Proper docstrings following Google/NumPy style
- Input validation and error handling
- Logging support
- Comprehensive test suite
- Pre-commit hooks for code quality
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

## [0.2.0] - Previous Release

### Added
- Initial implementation of integrated 3-point correlation functions
- GPU support via CuPy
- CPU implementation with multiprocessing support
- Visualization utilities
- Basic correlation function calculations
