"""
Device buffer and cache management for correlation computations.

ComputeContext holds all GPU/CPU device-side arrays and caches needed
during correlation measurements.  It separates the mutable runtime
state from the Correlation object, making it easy to invalidate
buffers when the pair geometry changes (e.g. after a new preprocess()
call) without touching the rest of the object.

Buffer categories:
  - PREPARED: pair indices and rotation factors on device
  - WEIGHT_CACHE: cached sums-of-weights for normalisation
  - APERTURE: flattened aperture filter geometry (host + device)
  - FUSED: struct-of-arrays map data for the fused 3×2pt kernel
"""

from typing import Any, Dict, Tuple


class ComputeContext:
    """Owns runtime device buffers and lightweight caches."""

    # Pair geometry on device: pixel indices, rotation factors e^{2iφ},
    # bin offsets (CSR format), and total pair count
    _PREPARED_DEFAULTS: Tuple[Tuple[str, Any], ...] = (
        ("inds_dev", None),
        ("inds_i_dev", None),
        ("inds_j_dev", None),
        ("exp2phi_dev", None),
        ("bins_dev", None),
        ("tot_bins_dev", None),
        ("tot_bins_reduceat_dev", None),
        ("ntotpairs", 0),
    )
    # Cached weight sums Σw_i·w_j used to normalise ξ and γ_t estimators;
    # invalidated when weights change or pairs are re-prepared
    _WEIGHT_CACHE_DEFAULTS: Tuple[Tuple[str, Any], ...] = (
        ("_tomo_sumofweights_cache", None),
        ("_tomo_sumofweights_cache_w_fingerprint", None),
        ("_tomo_sumofweights_cache_prepare_version", None),
        ("_xipm_sumofweights_cache", None),
        ("_xipm_sumofweights_cache_w_fingerprint", None),
        ("_xipm_sumofweights_cache_prepare_version", None),
    )
    # Aperture filter geometry on host: pixel indices within each patch's
    # aperture, cos(2φ)/sin(2φ), filter values Q(θ), patch solid angles
    _APERTURE_DEFAULTS: Tuple[Tuple[str, Any], ...] = (
        ("Q_inds_flat", None),
        ("Q_cos_flat", None),
        ("Q_sin_flat", None),
        ("Q_val_flat", None),
        ("Q_offsets", None),
        ("Q_patch_area_flat", None),
    )
    # Same aperture data transferred to device (GPU memory)
    _APERTURE_DEVICE_DEFAULTS: Tuple[Tuple[str, Any], ...] = (
        ("Q_inds_dev", None),
        ("Q_cos_dev", None),
        ("Q_sin_dev", None),
        ("Q_val_dev", None),
        ("Q_offsets_dev", None),
        ("Q_patch_area_dev", None),
    )
    # SoA (struct-of-arrays) map buffers for the fused 3×2pt kernel:
    # density δ_g, shear (γ₁,γ₂), and their weights, pre-arranged for
    # coalesced GPU memory access
    _FUSED_INPUT_BUFFER_DEFAULTS: Tuple[Tuple[str, Any], ...] = (
        ("fused_density_soa", None),
        ("fused_shear_soa", None),
        ("fused_density_w_soa", None),
        ("fused_shear_w_soa", None),
    )
    # Pre-allocated output buffers for the fused kernel (M_ap, M_g,
    # ξ+, ξ-, ξ_g, ξ_t numerators and denominators)
    _FUSED_OUTPUT_BUFFER_DEFAULTS: Tuple[Tuple[str, Any], ...] = (
        ("fused_output_buffers", None),
    )

    def __init__(self) -> None:
        self.tomo_combination_cache: Dict[Any, Any] = {}
        self.initialize_runtime_state()

    def initialize_runtime_state(self) -> None:
        for name, value in (
            self._PREPARED_DEFAULTS
            + self._WEIGHT_CACHE_DEFAULTS
            + self._APERTURE_DEFAULTS
            + self._APERTURE_DEVICE_DEFAULTS
            + self._FUSED_INPUT_BUFFER_DEFAULTS
            + self._FUSED_OUTPUT_BUFFER_DEFAULTS
        ):
            setattr(self, name, value)
        self.prepare_version = 0

    def ensure_runtime_state(self) -> None:
        for name, value in (
            self._PREPARED_DEFAULTS
            + self._WEIGHT_CACHE_DEFAULTS
            + self._APERTURE_DEFAULTS
            + self._APERTURE_DEVICE_DEFAULTS
            + self._FUSED_INPUT_BUFFER_DEFAULTS
            + self._FUSED_OUTPUT_BUFFER_DEFAULTS
        ):
            if name not in self.__dict__:
                setattr(self, name, value)
        if "prepare_version" not in self.__dict__:
            self.prepare_version = 0
        if "tomo_combination_cache" not in self.__dict__:
            self.tomo_combination_cache = {}

    def invalidate_prepared_state(self) -> None:
        for name, value in (
            self._PREPARED_DEFAULTS
            + self._WEIGHT_CACHE_DEFAULTS
            + self._APERTURE_DEFAULTS
            + self._APERTURE_DEVICE_DEFAULTS
            + self._FUSED_INPUT_BUFFER_DEFAULTS
            + self._FUSED_OUTPUT_BUFFER_DEFAULTS
        ):
            setattr(self, name, value)
