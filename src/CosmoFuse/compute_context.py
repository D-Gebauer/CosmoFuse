from typing import Any, Dict, Tuple


class ComputeContext:
    """Owns runtime device buffers and lightweight caches."""

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
    _WEIGHT_CACHE_DEFAULTS: Tuple[Tuple[str, Any], ...] = (
        ("_tomo_sumofweights_cache", None),
        ("_tomo_sumofweights_cache_w_fingerprint", None),
        ("_tomo_sumofweights_cache_prepare_version", None),
        ("_xipm_sumofweights_cache", None),
        ("_xipm_sumofweights_cache_w_fingerprint", None),
        ("_xipm_sumofweights_cache_prepare_version", None),
    )
    _APERTURE_DEFAULTS: Tuple[Tuple[str, Any], ...] = (
        ("Q_inds_flat", None),
        ("Q_cos_flat", None),
        ("Q_sin_flat", None),
        ("Q_val_flat", None),
        ("Q_offsets", None),
        ("Q_patch_area_flat", None),
    )
    _APERTURE_DEVICE_DEFAULTS: Tuple[Tuple[str, Any], ...] = (
        ("Q_inds_dev", None),
        ("Q_cos_dev", None),
        ("Q_sin_dev", None),
        ("Q_val_dev", None),
        ("Q_offsets_dev", None),
        ("Q_patch_area_dev", None),
    )
    _FUSED_INPUT_BUFFER_DEFAULTS: Tuple[Tuple[str, Any], ...] = (
        ("fused_density_soa", None),
        ("fused_shear_soa", None),
        ("fused_density_w_soa", None),
        ("fused_shear_w_soa", None),
    )
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
