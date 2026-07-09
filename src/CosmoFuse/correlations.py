import hashlib
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import healpy as hp
import numpy as np
from numba import njit, prange
from scipy.special import binom
from tqdm import trange

from .backend import get_backend
from .compute_context import ComputeContext
from .io_handler import PairIOHandler
from .pair_geometry import PairGeometry
from .correlation_helpers import (
    Q_T,
    Q_crittenden,
    Q_schneider,
    calculate_all_zetas as _calculate_all_zetas_helper,
    zeta_a_g as _zeta_a_g_helper,
    zeta_a_minus as _zeta_a_minus_helper,
    zeta_a_plus as _zeta_a_plus_helper,
    zeta_a_t as _zeta_a_t_helper,
    zeta_g_g as _zeta_g_g_helper,
    zeta_g_minus as _zeta_g_minus_helper,
    zeta_g_plus as _zeta_g_plus_helper,
    zeta_g_t as _zeta_g_t_helper,
)
from .pair_finder import PairFinder
from .utils import pixel2RaDec, select_patch_centers

logger = logging.getLogger(__name__)

# Digest memo for read-only weight arrays (see _fingerprint_weights).
_FINGERPRINT_MEMO: Dict[Any, Any] = {}

_ALLOWED_FLOAT_PRECISIONS = {
    "float32": np.float32,
    "float64": np.float64,
}
_ROTATION_COMPLEX_PRECISION = {
    "float32": np.complex64,
    "float64": np.complex128,
}

_ANGLE_METHOD_TO_CODE = {
    "arccos": 0,
    "haversine": 1,
    "law": 2,
}


def _resolve_angle_method_code(angle_method: str) -> int:
    key = str(angle_method).lower()
    code = _ANGLE_METHOD_TO_CODE.get(key)
    if code is None:
        raise ValueError(
            "angle_method must be one of ('arccos', 'haversine', 'law'); "
            f"got {angle_method!r}"
        )
    return int(code)


def _compute_aperture_shear_all_patches(
    Q_inds: np.ndarray,
    Q_cos: np.ndarray,
    Q_sin: np.ndarray,
    Q_val: np.ndarray,
    Q_offsets: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    Q_w: np.ndarray,
    Q_patch_area: np.ndarray,
) -> np.ndarray:
    n_patches = Q_offsets.size - 1
    aperture_shear = np.zeros(n_patches, dtype=g1.dtype)
    for patch_idx in range(n_patches):
        start = Q_offsets[patch_idx]
        end = Q_offsets[patch_idx + 1]
        sum_w = g1[0] * 0.0
        sum_gtw = g1[0] * 0.0
        for i in range(start, end):
            idx = Q_inds[i]
            gt = -g1[idx] * Q_cos[i] - g2[idx] * Q_sin[i]
            weight = Q_w[idx]
            sum_w += weight
            sum_gtw += weight * gt * Q_val[i]
        aperture_shear[patch_idx] = Q_patch_area[patch_idx] * sum_gtw / sum_w
    return aperture_shear


def _compute_pairs_impl(
    patch_inds: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    binedges: np.ndarray,
    angle_method_code: int = 1,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    # Two-pass structure so the O(npts^2) work parallelises with prange:
    # pass 1 counts accepted pairs per row i (cheap: dot product + bin
    # test), an exclusive prefix sum gives per-row write offsets, and
    # pass 2 fills the exactly-sized outputs at those offsets.  No
    # atomics; output order is identical to the serial row-major loop.
    # Rows are processed in a light/heavy interleaved order so the
    # triangular row lengths balance across threads.
    npts = patch_inds.size

    cos_dec = np.cos(dec)
    x = cos_dec * np.cos(ra)
    y = cos_dec * np.sin(ra)
    z = np.sin(dec)

    # cosine-space thresholds (cos is monotonically decreasing over [0, pi])
    cos_binedges = np.cos(binedges)
    nedges = binedges.size
    cos_max = cos_binedges[0]
    cos_min = cos_binedges[nedges - 1]

    nrows = npts - 1 if npts > 1 else 0

    counts = np.zeros(npts, dtype=np.int64)
    for m in prange(nrows):
        # branchless light/heavy row interleave: even m -> row m//2,
        # odd m -> row nrows-1-m//2 (balances the triangular loop)
        half = m >> 1
        odd = m & 1
        i = half + odd * (nrows - 1 - 2 * half)
        x1 = x[i]
        y1 = y[i]
        z1 = z[i]
        row_count = 0
        for j in range(i + 1, npts):
            cos_theta = x1 * x[j] + y1 * y[j] + z1 * z[j]
            if cos_theta >= cos_max or cos_theta <= cos_min:
                continue
            # binary search over the strictly descending cos_binedges for
            # the last edge with value > cos_theta (same acceptance rule as
            # the strict-inequality linear scan it replaces)
            lo = 0
            hi = nedges
            while lo < hi:
                mid = (lo + hi) >> 1
                if cos_binedges[mid] > cos_theta:
                    lo = mid + 1
                else:
                    hi = mid
            b = lo - 1
            if b >= 0 and b < nedges - 1 and cos_theta > cos_binedges[b + 1]:
                row_count += 1
        counts[i] = row_count

    offsets = np.zeros(npts + 1, dtype=np.int64)
    for i in range(npts):
        offsets[i + 1] = offsets[i] + counts[i]
    total = offsets[npts]

    inds_a = np.empty(total, dtype=patch_inds.dtype)
    inds_b = np.empty(total, dtype=patch_inds.dtype)
    bin_indices = np.empty(total, dtype=np.int64)
    exp2phi1_real = np.empty(total, dtype=ra.dtype)
    exp2phi1_imag = np.empty(total, dtype=ra.dtype)
    exp2phi2_real = np.empty(total, dtype=ra.dtype)
    exp2phi2_imag = np.empty(total, dtype=ra.dtype)

    for m in prange(nrows):
        half = m >> 1
        odd = m & 1
        i = half + odd * (nrows - 1 - 2 * half)
        x1 = x[i]
        y1 = y[i]
        z1 = z[i]
        out_idx = offsets[i]

        for j in range(i + 1, npts):
            x2 = x[j]
            y2 = y[j]
            z2 = z[j]

            cos_theta = x1 * x2 + y1 * y2 + z1 * z2

            if cos_theta >= cos_max or cos_theta <= cos_min:
                continue
            lo = 0
            hi = nedges
            while lo < hi:
                mid = (lo + hi) >> 1
                if cos_binedges[mid] > cos_theta:
                    lo = mid + 1
                else:
                    hi = mid
            bin_idx = lo - 1
            if bin_idx < 0 or bin_idx >= nedges - 1 or not (
                cos_theta > cos_binedges[bin_idx + 1]
            ):
                continue

            # Compute C1 sine and cosine terms (unnormalized)
            sinC1 = x1 * y2 - x2 * y1
            dsq_AC1 = x1 * x1 + y1 * y1 + (z1 - 1.0) * (z1 - 1.0)
            dx12 = x1 - x2
            dy12 = y1 - y2
            dz12 = z1 - z2
            dsq_BC1 = dx12 * dx12 + dy12 * dy12 + dz12 * dz12
            dsq_AB1 = x2 * x2 + y2 * y2 + (z2 - 1.0) * (z2 - 1.0)
            cosC1 = 0.5 * (dsq_AC1 + dsq_BC1 - dsq_AB1 - 0.5 * dsq_AC1 * dsq_BC1)

            # Compute cos(2*theta1) and sin(2*theta1) directly
            # theta1 = pi/2 - C1 => 2*theta1 = pi - 2*C1
            # c1 = cos(2*theta1) = -cos(2*C1) = -(cos^2 C1 - sin^2 C1) = sin^2 C1 - cos^2 C1
            # s1 = sin(2*theta1) = sin(2*C1) = 2*sinC1*cosC1
            # Using unnormalized vectors:
            R2_C1 = sinC1 * sinC1 + cosC1 * cosC1
            if R2_C1 > 0:
                inv_R2_C1 = 1.0 / R2_C1
                c1 = (sinC1 * sinC1 - cosC1 * cosC1) * inv_R2_C1
                s1 = (2.0 * sinC1 * cosC1) * inv_R2_C1
            else:
                c1 = -1.0 # implies theta1 = pi/2 (C1=0)
                s1 = 0.0

            # Compute C2 sine and cosine terms (unnormalized)
            sinC2 = x2 * y1 - x1 * y2
            dsq_AC2 = dsq_AB1
            dsq_BC2 = dsq_BC1
            dsq_AB2 = dsq_AC1
            cosC2 = 0.5 * (dsq_AC2 + dsq_BC2 - dsq_AB2 - 0.5 * dsq_AC2 * dsq_BC2)

            # Compute cos(2*theta2) and sin(2*theta2) directly
            R2_C2 = sinC2 * sinC2 + cosC2 * cosC2
            if R2_C2 > 0:
                inv_R2_C2 = 1.0 / R2_C2
                c2 = (sinC2 * sinC2 - cosC2 * cosC2) * inv_R2_C2
                s2 = (2.0 * sinC2 * cosC2) * inv_R2_C2
            else:
                c2 = -1.0
                s2 = 0.0

            inds_a[out_idx] = patch_inds[i]
            inds_b[out_idx] = patch_inds[j]
            bin_indices[out_idx] = bin_idx
            exp2phi1_real[out_idx] = c1
            exp2phi1_imag[out_idx] = s1
            exp2phi2_real[out_idx] = c2
            exp2phi2_imag[out_idx] = s2
            out_idx += 1

    return (
        inds_a,
        inds_b,
        bin_indices,
        exp2phi1_real,
        exp2phi1_imag,
        exp2phi2_real,
        exp2phi2_imag,
    )


_compute_pairs_kernel_cache: Dict[bool, Any] = {}


def _get_pairs_numba_kernel(fastmath: bool) -> Any:
    key = bool(fastmath)
    cached = _compute_pairs_kernel_cache.get(key)
    if cached is not None:
        return cached

    kernel = njit(fastmath=key, parallel=True, cache=True)(_compute_pairs_impl)
    _compute_pairs_kernel_cache[key] = kernel
    return kernel


def _normalize_precision(
    precision: Union[str, np.dtype, type],
    allowed: Dict[str, Any],
    name: str,
) -> np.dtype:
    try:
        precision_name = np.dtype(precision).name
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be one of {list(allowed.keys())}; got {precision!r}"
        ) from exc

    if precision_name not in allowed:
        raise ValueError(
            f"{name} must be one of {list(allowed.keys())}; got {precision!r}"
        )

    return np.dtype(allowed[precision_name])


class Correlation:
    """Base class for all correlation functions.

    This class contains methods for finding pairs and their angles, as well as
    loading and saving them. It provides the foundation for calculating
    integrated 3-point correlation functions.

    Attributes:
        nside: HEALPix resolution parameter
        nbins: Number of angular bins
        theta_min: Minimum angular separation in radians
        theta_max: Maximum angular separation in radians
        binedges: Edges of angular bins in radians
        bincenters: Centers of angular bins in arcminutes
        patch_size: Size of each patch in arcminutes
        theta_Q: Size of compensated filter in arcminutes
        phi_center: Right ascension centers of patches in radians
        theta_center: Declination centers of patches in radians
        n_patches: Number of patches
        fastmath: Whether to use fastmath in JIT compiled functions
        map_inds: Indices of valid pixels in the mask
        device: Device to use for calculations ('cpu', 'gpu', 'auto', or GPU ID).
        map_precision: Float precision for map/shear/weight arrays.
        rotation_precision: Float precision for rotation/filter values.
    """

    def __init__(
        self,
        nside: int,
        phi_center: np.ndarray,
        theta_center: np.ndarray,
        nbins: int = 10,
        theta_min: float = 10,
        theta_max: float = 170,
        patch_size: float = 90,
        theta_Q: float = 90,
        mask: Optional[np.ndarray] = None,
        fastmath: bool = True,
        device: Union[str, int] = "auto",
        map_precision: Union[str, np.dtype, type] = "float64",
        rotation_precision: Union[str, np.dtype, type] = "float32",
    ) -> None:
        """Initialize the Correlation class with validation.

        Args:
            nside: HEALPix resolution parameter
            phi_center: Right ascension centers of patches in radians
            theta_center: Declination centers of patches in radians
            nbins: Number of angular bins
            theta_min: Minimum angular separation in arcminutes
            theta_max: Maximum angular separation in arcminutes
            patch_size: Size of each patch in arcminutes
            theta_Q: Size of compensated filter in arcminutes
            mask: Optional mask array
            fastmath: Whether to use fastmath in JIT compiled functions
            device: Device to use for calculations ('cpu', 'gpu', 'auto', or GPU ID).
            map_precision: One of float32/float64 for map-like arrays.
            rotation_precision: One of float32/float64 for rotation/filter arrays.

        Raises:
            ValueError: If input parameters are invalid
        """
        if nside <= 0:
            raise ValueError("nside must be positive")
        if nbins <= 0:
            raise ValueError("nbins must be positive")
        if theta_min >= theta_max:
            raise ValueError("theta_min must be less than theta_max")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if theta_Q <= 0:
            raise ValueError("theta_Q must be positive")
        if len(phi_center) != len(theta_center):
            raise ValueError("phi_center and theta_center must have the same length")
        self.map_dtype = _normalize_precision(
            map_precision, _ALLOWED_FLOAT_PRECISIONS, "map_precision"
        )
        self.rotation_dtype = _normalize_precision(
            rotation_precision, _ALLOWED_FLOAT_PRECISIONS, "rotation_precision"
        )
        npix = hp.nside2npix(nside)
        self.index_dtype = np.dtype(np.int32) if npix < 2**31 else np.dtype(np.int64)
        self.rotation_complex_dtype = np.dtype(
            _ROTATION_COMPLEX_PRECISION[self.rotation_dtype.name]
        )

        self.nside = nside
        self.nbins = nbins
        self.theta_min = theta_min / 60 / 180 * np.pi
        self.theta_max = theta_max / 60 / 180 * np.pi
        self.binedges = np.geomspace(self.theta_min, self.theta_max, self.nbins + 1)
        self.bincenters = (
            np.sqrt(self.binedges[1:] * self.binedges[:-1]) * 60 * 180 / np.pi
        )
        self.patch_size = patch_size
        self.theta_Q = theta_Q
        self.phi_center = phi_center
        self.theta_center = theta_center
        self.n_patches = len(phi_center)
        self.fastmath = fastmath
        self.aperture_shear_all_patches = njit(fastmath=fastmath, cache=True)(
            _compute_aperture_shear_all_patches
        )
        self._compute_pairs_kernel = _get_pairs_numba_kernel(fastmath)
        self._pair_finder = PairFinder(
            nbins=self.nbins,
            binedges=self.binedges,
            index_dtype=self.index_dtype,
            rotation_dtype=self.rotation_dtype,
            rotation_complex_dtype=self.rotation_complex_dtype,
            kernel=self._compute_pairs_kernel,
            resolve_angle_method_code=_resolve_angle_method_code,
        )
        self.radius_filter = 5 * self.theta_Q

        if mask is not None:
            if len(mask) != hp.nside2npix(self.nside):
                raise ValueError(
                    "Mask length must match number of pixels for given nside"
                )
            self.map_inds = np.where(mask)[0].astype(self.index_dtype, copy=False)
        else:
            self.map_inds = np.arange(hp.nside2npix(self.nside), dtype=self.index_dtype)
        self.map_mask = np.zeros(hp.nside2npix(self.nside), dtype=bool)
        self.map_mask[self.map_inds] = True

        self.backend = get_backend(device)
        self.device = device
        self.compute_context = ComputeContext()

        self.pair_inds = []
        self.pair_exp2phi = []
        self.bins = []
        self.compute_context.initialize_runtime_state()
        self._aperture_filter_active_key = "Q_T"

    @classmethod
    def from_mask(
        cls,
        nside: int,
        mask: np.ndarray,
        nside_centers: int,
        patch_size: float = 90,
        theta_Q: float = 90,
        f_mask: float = 0.2,
        f_mask_filter: Optional[float] = None,
        filter_weighted: bool = False,
        aperture_filter: Optional[Callable[..., Any]] = None,
        **kwargs: Any,
    ) -> "Correlation":
        """Construct a Correlation with patch centres selected from a mask.

        Convenience wrapper around
        :func:`CosmoFuse.utils.select_patch_centers`: candidate centres on
        an ``nside_centers`` grid are accepted when the masked fraction of
        ``mask`` within the patch disc (radius ``patch_size``) and within
        the compensated-filter support disc (radius ``5 * theta_Q``) stays
        below ``f_mask`` / ``f_mask_filter``.

        Args:
            nside: HEALPix resolution of the maps to be measured.
            mask: HEALPix mask/footprint (nonzero = observed).  May be at a
                different resolution than ``nside``; the selection runs at
                the mask's own resolution and the stored instance mask is
                regraded to ``nside`` if needed.
            nside_centers: Resolution of the candidate-centre grid
                (controls the patch oversampling density).
            patch_size: Patch radius in arcminutes.
            theta_Q: Compensated filter scale in arcminutes.
            f_mask: Maximum tolerated masked fraction inside the patch disc.
            f_mask_filter: Maximum tolerated masked fraction inside the
                filter support disc; defaults to ``f_mask``.
            filter_weighted: If ``True``, weight the filter-disc masking
                check by ``|aperture_filter(θ)|`` instead of counting
                pixels (see :func:`CosmoFuse.utils.select_patch_centers`).
            aperture_filter: Filter for the weighted check; defaults to
                the built-in ``Q_crittenden``.  Selection only — pass the
                same filter to :meth:`preprocess` for consistency.
            **kwargs: Forwarded to the constructor (``nbins``,
                ``theta_min``, ``theta_max``, ``device``, ``fastmath``,
                ``map_precision``, ``rotation_precision``).

        Raises:
            ValueError: If no candidate centre satisfies the masking
                criteria.
        """
        mask_arr = np.asarray(mask)
        phi_center, theta_center = select_patch_centers(
            mask_arr,
            nside_centers,
            patch_size=patch_size,
            theta_Q=theta_Q,
            f_mask=f_mask,
            f_mask_filter=f_mask_filter,
            filter_weighted=filter_weighted,
            aperture_filter=aperture_filter,
        )
        if phi_center.size == 0:
            raise ValueError(
                "No patch centres satisfy the masking criteria; loosen "
                "f_mask/f_mask_filter, use a finer nside_centers, or check "
                "the mask."
            )
        if mask_arr.size != hp.nside2npix(nside):
            instance_mask = hp.ud_grade(mask_arr.astype(np.float64), nside) != 0
        else:
            instance_mask = mask_arr
        return cls(
            nside,
            phi_center,
            theta_center,
            patch_size=patch_size,
            theta_Q=theta_Q,
            mask=instance_mask,
            **kwargs,
        )

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        # map_mask is rebuilt from map_inds in __setstate__; dropping it keeps
        # the full-sky boolean array out of every pickle.
        state.pop('map_mask', None)
        if 'backend' in state:
            del state['backend']
        if '_compute_pairs_kernel' in state:
            del state['_compute_pairs_kernel']
        if '_pair_finder' in state:
            del state['_pair_finder']
        if 'compute_context' in state:
            del state['compute_context']
        if '_compute_context' in state:
            del state['_compute_context']
        if '_tomo_combination_cache' in state:
            state['_tomo_combination_cache'] = {}
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        if "map_mask" not in self.__dict__:
            self.map_mask = np.zeros(hp.nside2npix(self.nside), dtype=bool)
            self.map_mask[self.map_inds] = True
        self.backend = get_backend(self.device)
        if "aperture_shear_all_patches" not in self.__dict__:
            self.aperture_shear_all_patches = njit(fastmath=self.fastmath, cache=True)(
                _compute_aperture_shear_all_patches
            )
        self._compute_pairs_kernel = _get_pairs_numba_kernel(self.fastmath)
        self._pair_finder = PairFinder(
            nbins=self.nbins,
            binedges=self.binedges,
            index_dtype=self.index_dtype,
            rotation_dtype=self.rotation_dtype,
            rotation_complex_dtype=self.rotation_complex_dtype,
            kernel=self._compute_pairs_kernel,
            resolve_angle_method_code=_resolve_angle_method_code,
        )
        self.compute_context = ComputeContext()
        legacy_context_fields = (
            "inds_dev",
            "exp2phi_dev",
            "bins_dev",
            "tot_bins_dev",
            "tot_bins_reduceat_dev",
            "ntotpairs",
            "_tomo_sumofweights_cache",
            "_tomo_sumofweights_cache_w_fingerprint",
            "_tomo_sumofweights_cache_prepare_version",
            "_xipm_sumofweights_cache",
            "_xipm_sumofweights_cache_w_fingerprint",
            "_xipm_sumofweights_cache_prepare_version",
            "Q_inds_flat",
            "Q_cos_flat",
            "Q_sin_flat",
            "Q_val_flat",
            "Q_offsets",
            "Q_patch_area_flat",
        )
        for field_name in legacy_context_fields:
            if field_name in self.__dict__:
                setattr(self.compute_context, field_name, self.__dict__.pop(field_name))
        if "_prepare_version" in self.__dict__:
            self.compute_context.prepare_version = self.__dict__.pop("_prepare_version")
        if "_tomo_combination_cache" in self.__dict__:
            self.compute_context.tomo_combination_cache = self.__dict__.pop("_tomo_combination_cache")
        self.compute_context.ensure_runtime_state()
        if "_aperture_filter_active_key" not in self.__dict__:
            self._aperture_filter_active_key = "Q_T"

    def _invalidate_prepared_state(self) -> None:
        """Clears prepared backend buffers and cached tomographic weights."""
        self.compute_context.invalidate_prepared_state()

    def warmup(self) -> None:
        """Eagerly JIT-compile the CPU kernels for this instance's dtypes.

        Moves the one-off Numba compilation cost (roughly 10-20 s in a
        fresh environment; sub-second afterwards thanks to the on-disk
        cache) out of the first map measurement.  No-op on GPU backends.
        """
        self.backend.warmup(
            map_dtype=self.map_dtype,
            rotation_dtype=self.rotation_dtype,
            rotation_complex_dtype=self.rotation_complex_dtype,
            index_dtype=self.index_dtype,
        )
        pts = np.arange(3, dtype=self.index_dtype)
        ra = np.array([0.0, 1e-3, 2e-3], dtype=self.rotation_dtype)
        dec = np.array([0.0, 1e-3, 0.0], dtype=self.rotation_dtype)
        binedges = np.asarray(self.binedges, dtype=self.rotation_dtype)
        self._compute_pairs_kernel(pts, ra, dec, binedges, 1)

    @property
    def inds_dev(self) -> Any:
        return self.compute_context.inds_dev

    @inds_dev.setter
    def inds_dev(self, value: Any) -> None:
        self.compute_context.inds_dev = value

    @property
    def exp2phi_dev(self) -> Any:
        return self.compute_context.exp2phi_dev

    @exp2phi_dev.setter
    def exp2phi_dev(self, value: Any) -> None:
        self.compute_context.exp2phi_dev = value

    @property
    def bins_dev(self) -> Any:
        return self.compute_context.bins_dev

    @bins_dev.setter
    def bins_dev(self, value: Any) -> None:
        self.compute_context.bins_dev = value

    @property
    def tot_bins_dev(self) -> Any:
        return self.compute_context.tot_bins_dev

    @tot_bins_dev.setter
    def tot_bins_dev(self, value: Any) -> None:
        self.compute_context.tot_bins_dev = value

    @property
    def tot_bins_reduceat_dev(self) -> Any:
        return self.compute_context.tot_bins_reduceat_dev

    @tot_bins_reduceat_dev.setter
    def tot_bins_reduceat_dev(self, value: Any) -> None:
        self.compute_context.tot_bins_reduceat_dev = value

    @property
    def ntotpairs(self) -> int:
        return self.compute_context.ntotpairs

    @ntotpairs.setter
    def ntotpairs(self, value: int) -> None:
        self.compute_context.ntotpairs = value

    @property
    def Q_inds_flat(self) -> Any:
        return self.compute_context.Q_inds_flat

    @Q_inds_flat.setter
    def Q_inds_flat(self, value: Any) -> None:
        self.compute_context.Q_inds_flat = value

    @property
    def Q_cos_flat(self) -> Any:
        return self.compute_context.Q_cos_flat

    @Q_cos_flat.setter
    def Q_cos_flat(self, value: Any) -> None:
        self.compute_context.Q_cos_flat = value

    @property
    def Q_sin_flat(self) -> Any:
        return self.compute_context.Q_sin_flat

    @Q_sin_flat.setter
    def Q_sin_flat(self, value: Any) -> None:
        self.compute_context.Q_sin_flat = value

    @property
    def Q_val_flat(self) -> Any:
        return self.compute_context.Q_val_flat

    @Q_val_flat.setter
    def Q_val_flat(self, value: Any) -> None:
        self.compute_context.Q_val_flat = value

    @property
    def Q_offsets(self) -> Any:
        return self.compute_context.Q_offsets

    @Q_offsets.setter
    def Q_offsets(self, value: Any) -> None:
        self.compute_context.Q_offsets = value

    @property
    def Q_patch_area_flat(self) -> Any:
        return self.compute_context.Q_patch_area_flat

    @Q_patch_area_flat.setter
    def Q_patch_area_flat(self, value: Any) -> None:
        self.compute_context.Q_patch_area_flat = value

    def _get_tomo_combination_indices(
        self, nzbins: int, nzbin_combs: int
    ) -> Tuple[Any, Any, np.ndarray]:
        cached = self.compute_context.tomo_combination_cache.get(nzbins)
        if cached is not None:
            return cached

        comb_i_np = np.zeros(nzbin_combs, dtype=np.int32)
        comb_j_np = np.zeros(nzbin_combs, dtype=np.int32)
        auto_comb_np = np.zeros(nzbin_combs, dtype=bool)
        comb_idx = 0
        for i in range(nzbins):
            for j in range(i, nzbins):
                comb_i_np[comb_idx] = i
                comb_j_np[comb_idx] = j
                auto_comb_np[comb_idx] = i == j
                comb_idx += 1

        module = self.backend.module
        comb_i_dev = module.ascontiguousarray(module.asarray(comb_i_np))
        comb_j_dev = module.ascontiguousarray(module.asarray(comb_j_np))
        cached_tuple = (comb_i_dev, comb_j_dev, auto_comb_np)
        self.compute_context.tomo_combination_cache[nzbins] = cached_tuple
        return cached_tuple

    def _get_tomo_cross_combination_indices(
        self, nlens_bins: int, nsource_bins: int
    ) -> Tuple[Any, Any]:
        cache_key = ("cross", nlens_bins, nsource_bins)
        cached = self.compute_context.tomo_combination_cache.get(cache_key)
        if cached is not None:
            return cached

        ncomb = nlens_bins * nsource_bins
        comb_i_np = np.zeros(ncomb, dtype=np.int32)
        comb_j_np = np.zeros(ncomb, dtype=np.int32)
        comb_idx = 0
        for i in range(nlens_bins):
            for j in range(nsource_bins):
                comb_i_np[comb_idx] = i
                comb_j_np[comb_idx] = j
                comb_idx += 1

        module = self.backend.module
        comb_i_dev = module.ascontiguousarray(module.asarray(comb_i_np))
        comb_j_dev = module.ascontiguousarray(module.asarray(comb_j_np))
        cached_tuple = (comb_i_dev, comb_j_dev)
        self.compute_context.tomo_combination_cache[cache_key] = cached_tuple
        return cached_tuple

    def _resolve_aperture_filter(
        self,
        aperture_filter: Optional[Callable[..., Any]] = None,
    ) -> Callable[..., Any]:
        return PairGeometry.resolve_aperture_filter(aperture_filter)

    def _aperture_filter_key(self, aperture_filter: Callable[..., Any]) -> Any:
        return PairGeometry.aperture_filter_key(aperture_filter)

    def _evaluate_aperture_filter(
        self,
        aperture_filter: Callable[..., Any],
        theta: np.ndarray,
    ) -> np.ndarray:
        return PairGeometry.evaluate_aperture_filter(self, aperture_filter, theta)

    def _ensure_aperture_pairs(
        self,
        aperture_filter: Optional[Callable[..., Any]] = None,
    ) -> None:
        PairGeometry.ensure_aperture_pairs(self, aperture_filter=aperture_filter)

    def _get_selected_tomo_density_combination_indices(
        self,
        nzbins: int,
        gc_auto_correlations_only: bool = False,
    ) -> Tuple[Any, Any, np.ndarray, int]:
        cache_key = ("density", nzbins, bool(gc_auto_correlations_only))
        cached = self.compute_context.tomo_combination_cache.get(cache_key)
        if cached is not None:
            return cached

        if gc_auto_correlations_only:
            ncomb = nzbins
            comb_i_np = np.arange(nzbins, dtype=np.int32)
            comb_j_np = np.arange(nzbins, dtype=np.int32)
            auto_comb_np = np.ones(ncomb, dtype=bool)
        else:
            ncomb = int(binom(nzbins + 1, 2))
            comb_i_np = np.zeros(ncomb, dtype=np.int32)
            comb_j_np = np.zeros(ncomb, dtype=np.int32)
            auto_comb_np = np.zeros(ncomb, dtype=bool)
            comb_idx = 0
            for i in range(nzbins):
                for j in range(i, nzbins):
                    comb_i_np[comb_idx] = i
                    comb_j_np[comb_idx] = j
                    auto_comb_np[comb_idx] = i == j
                    comb_idx += 1

        module = self.backend.module
        comb_i_dev = module.ascontiguousarray(module.asarray(comb_i_np))
        comb_j_dev = module.ascontiguousarray(module.asarray(comb_j_np))
        cached_tuple = (comb_i_dev, comb_j_dev, auto_comb_np, ncomb)
        self.compute_context.tomo_combination_cache[cache_key] = cached_tuple
        return cached_tuple

    def _get_selected_tomo_cross_combination_indices(
        self,
        nlens_bins: int,
        nsource_bins: int,
        ggl_bin_combinations: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> Tuple[Any, Any, int]:
        if ggl_bin_combinations is None:
            comb_i, comb_j = self._get_tomo_cross_combination_indices(
                nlens_bins, nsource_bins
            )
            return comb_i, comb_j, int(nlens_bins * nsource_bins)

        combinations = list(ggl_bin_combinations)
        ncomb = len(combinations)
        comb_i_np = np.empty(ncomb, dtype=np.int32)
        comb_j_np = np.empty(ncomb, dtype=np.int32)

        for idx, pair in enumerate(combinations):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(
                    "ggl_bin_combinations must be an iterable of (lens_bin, source_bin) pairs."
                )

            lens_bin = int(pair[0])
            source_bin = int(pair[1])

            if lens_bin < 0 or lens_bin >= nlens_bins:
                raise ValueError(
                    f"Lens tomographic bin index {lens_bin} out of bounds for "
                    f"{nlens_bins} lens bins."
                )
            if source_bin < 0 or source_bin >= nsource_bins:
                raise ValueError(
                    f"Source tomographic bin index {source_bin} out of bounds for "
                    f"{nsource_bins} source bins."
                )

            comb_i_np[idx] = lens_bin
            comb_j_np[idx] = source_bin

        module = self.backend.module
        comb_i_dev = module.ascontiguousarray(module.asarray(comb_i_np))
        comb_j_dev = module.ascontiguousarray(module.asarray(comb_j_np))
        return comb_i_dev, comb_j_dev, ncomb

    def get_pairs_patch(
        self,
        patch_inds: np.ndarray,
        ra: np.ndarray,
        dec: np.ndarray,
        angle_method: str = "haversine",
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        return PairGeometry.get_pairs_patch(
            self,
            patch_inds,
            ra,
            dec,
            angle_method=angle_method,
        )

    def __get_pairs_helper__(
        self,
        i: int,
        angle_method: str = "haversine",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return PairGeometry.get_pairs_helper(self, i, angle_method=angle_method)

    def calculate_pairs_2PCF(self, angle_method: str = "haversine") -> None:
        PairGeometry.calculate_pairs_2PCF(self, angle_method=angle_method)

    def get_pairs_patch_M_a(
        self,
        pixels_RA_Q_patch: np.ndarray,
        pixels_dec_Q_patch: np.ndarray,
        Q_patch_center_RA: float,
        Q_patch_center_dec: float,
        aperture_filter: Optional[Callable[..., Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return PairGeometry.get_pairs_patch_M_a(
            self,
            pixels_RA_Q_patch,
            pixels_dec_Q_patch,
            Q_patch_center_RA,
            Q_patch_center_dec,
            aperture_filter=aperture_filter,
        )

    def __get_pairs_M_a_helper__(
        self,
        i: int,
        aperture_filter: Optional[Callable[..., Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        return PairGeometry.get_pairs_M_a_helper(
            self,
            i,
            aperture_filter=aperture_filter,
        )

    def calculate_pairs_M_a(
        self,
        aperture_filter: Optional[Callable[..., Any]] = None,
    ) -> None:
        PairGeometry.calculate_pairs_M_a(self, aperture_filter=aperture_filter)

    def _prepare_aperture_flat(self) -> None:
        PairGeometry.prepare_aperture_flat(self)

    def _invalidate_aperture_device_buffers(self) -> None:
        self.compute_context.Q_inds_dev = None
        self.compute_context.Q_cos_dev = None
        self.compute_context.Q_sin_dev = None
        self.compute_context.Q_val_dev = None
        self.compute_context.Q_offsets_dev = None
        self.compute_context.Q_patch_area_dev = None

    def _prepare_aperture_device_buffers(self) -> None:
        if self.Q_inds_flat is None:
            self._invalidate_aperture_device_buffers()
            return

        module = self.backend.module
        map_backend_dtype = getattr(module, self.map_dtype.name)

        q_inds_u32 = np.asarray(self.Q_inds_flat, dtype=np.uint32)
        self.compute_context.Q_inds_dev = module.ascontiguousarray(
            self.backend.to_device(q_inds_u32)
        )
        self.compute_context.Q_cos_dev = module.ascontiguousarray(
            self.backend.to_device(np.asarray(self.Q_cos_flat, dtype=self.map_dtype)).astype(
                map_backend_dtype, copy=False
            )
        )
        self.compute_context.Q_sin_dev = module.ascontiguousarray(
            self.backend.to_device(np.asarray(self.Q_sin_flat, dtype=self.map_dtype)).astype(
                map_backend_dtype, copy=False
            )
        )
        self.compute_context.Q_val_dev = module.ascontiguousarray(
            self.backend.to_device(np.asarray(self.Q_val_flat, dtype=self.map_dtype)).astype(
                map_backend_dtype, copy=False
            )
        )
        self.compute_context.Q_offsets_dev = module.ascontiguousarray(
            self.backend.to_device(np.asarray(self.Q_offsets, dtype=np.int64))
        )
        self.compute_context.Q_patch_area_dev = module.ascontiguousarray(
            self.backend.to_device(
                np.asarray(self.Q_patch_area_flat, dtype=self.map_dtype)
            ).astype(map_backend_dtype, copy=False)
        )

    def _get_or_create_fused_input_buffers(
        self,
        npix: int,
        n_density_bins: int,
        n_shear_bins: int,
        map_backend_dtype: Any,
    ) -> Tuple[Any, Any, Any, Any]:
        ctx = self.compute_context

        if (
            ctx.fused_density_soa is None
            or ctx.fused_density_soa.shape != (npix, n_density_bins)
            or getattr(ctx.fused_density_soa, "dtype", None) != map_backend_dtype
        ):
            ctx.fused_density_soa = self.backend.zeros(
                (npix, n_density_bins), dtype=map_backend_dtype
            )

        if (
            ctx.fused_shear_soa is None
            or ctx.fused_shear_soa.shape != (npix, n_shear_bins, 2)
            or getattr(ctx.fused_shear_soa, "dtype", None) != map_backend_dtype
        ):
            ctx.fused_shear_soa = self.backend.zeros(
                (npix, n_shear_bins, 2), dtype=map_backend_dtype
            )

        if (
            ctx.fused_density_w_soa is None
            or ctx.fused_density_w_soa.shape != (npix, n_density_bins)
            or getattr(ctx.fused_density_w_soa, "dtype", None) != map_backend_dtype
        ):
            ctx.fused_density_w_soa = self.backend.zeros(
                (npix, n_density_bins), dtype=map_backend_dtype
            )

        if (
            ctx.fused_shear_w_soa is None
            or ctx.fused_shear_w_soa.shape != (npix, n_shear_bins)
            or getattr(ctx.fused_shear_w_soa, "dtype", None) != map_backend_dtype
        ):
            ctx.fused_shear_w_soa = self.backend.zeros(
                (npix, n_shear_bins), dtype=map_backend_dtype
            )

        return (
            ctx.fused_density_soa,
            ctx.fused_shear_soa,
            ctx.fused_density_w_soa,
            ctx.fused_shear_w_soa,
        )

    def _fill_fused_input_buffers(
        self,
        density_dev: Any,
        shear_dev: Any,
        density_w_dev: Any,
        shear_w_dev: Any,
        density_soa: Any,
        shear_soa: Any,
        density_w_soa: Any,
        shear_w_soa: Any,
    ) -> None:
        module = self.backend.module
        copyto = getattr(module, "copyto", np.copyto)
        swapaxes = getattr(module, "swapaxes", np.swapaxes)
        moveaxis = getattr(module, "moveaxis", np.moveaxis)

        copyto(density_soa, swapaxes(density_dev, 0, 1))
        copyto(density_w_soa, swapaxes(density_w_dev, 0, 1))
        copyto(shear_soa, moveaxis(shear_dev, 2, 0))
        copyto(shear_w_soa, swapaxes(shear_w_dev, 0, 1))

    def _get_or_create_fused_output_buffers(
        self,
        n_shear_bins: int,
        n_density_bins: int,
        n_patches: int,
        nbins_total: int,
        ss_ncomb: int,
        dd_ncomb: int,
        ds_ncomb: int,
        map_backend_dtype: Any,
    ) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
        ctx = self.compute_context
        if ctx.fused_output_buffers is None:
            ctx.fused_output_buffers = {}

        cache = ctx.fused_output_buffers
        module = self.backend.module
        empty = getattr(module, "empty", None)

        specs = (
            ("out_ma_num", (n_shear_bins, n_patches)),
            ("out_ma_den", (n_shear_bins, n_patches)),
            ("out_mg_num", (n_density_bins, n_patches)),
            ("out_mg_den", (n_density_bins, n_patches)),
            ("out_xip_num", (2 * ss_ncomb, nbins_total)),
            ("out_xim_num", (2 * ss_ncomb, nbins_total)),
            ("out_xip_den", (2 * ss_ncomb, nbins_total)),
            ("out_xig_num", (2 * dd_ncomb, nbins_total)),
            ("out_xig_den", (2 * dd_ncomb, nbins_total)),
            ("out_xit_num", (ds_ncomb, nbins_total)),
            ("out_xit_den", (ds_ncomb, nbins_total)),
        )

        buffers: Dict[str, Any] = {}
        for name, shape in specs:
            arr = cache.get(name)
            if (
                arr is None
                or getattr(arr, "shape", None) != shape
                or getattr(arr, "dtype", None) != map_backend_dtype
            ):
                if empty is not None:
                    arr = empty(shape, dtype=map_backend_dtype)
                else:
                    arr = self.backend.zeros(shape, dtype=map_backend_dtype)
                cache[name] = arr
            buffers[name] = arr

        return (
            buffers["out_ma_num"],
            buffers["out_ma_den"],
            buffers["out_mg_num"],
            buffers["out_mg_den"],
            buffers["out_xip_num"],
            buffers["out_xim_num"],
            buffers["out_xip_den"],
            buffers["out_xig_num"],
            buffers["out_xig_den"],
            buffers["out_xit_num"],
            buffers["out_xit_den"],
        )

    def _get_or_create_fused_post_buffers(
        self,
        n_shear_bins: int,
        n_density_bins: int,
        n_patches: int,
        ss_ncomb: int,
        dd_ncomb: int,
        ds_ncomb: int,
        map_backend_dtype: Any,
    ) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        ctx = self.compute_context
        if ctx.fused_output_buffers is None:
            ctx.fused_output_buffers = {}

        cache = ctx.fused_output_buffers
        module = self.backend.module
        empty = getattr(module, "empty", None)

        specs = (
            ("M_a", (n_shear_bins, n_patches)),
            ("M_g", (n_density_bins, n_patches)),
            ("xip", (ss_ncomb, n_patches, self.nbins)),
            ("xim", (ss_ncomb, n_patches, self.nbins)),
            ("xi_g", (dd_ncomb, n_patches, self.nbins)),
            ("xi_t", (ds_ncomb, n_patches, self.nbins)),
            ("tmp_a", (n_patches, self.nbins)),
            ("tmp_b", (n_patches, self.nbins)),
        )

        buffers: Dict[str, Any] = {}
        for name, shape in specs:
            cache_key = f"post_{name}"
            arr = cache.get(cache_key)
            if (
                arr is None
                or getattr(arr, "shape", None) != shape
                or getattr(arr, "dtype", None) != map_backend_dtype
            ):
                if empty is not None:
                    arr = empty(shape, dtype=map_backend_dtype)
                else:
                    arr = self.backend.zeros(shape, dtype=map_backend_dtype)
                cache[cache_key] = arr
            buffers[name] = arr

        return (
            buffers["M_a"],
            buffers["M_g"],
            buffers["xip"],
            buffers["xim"],
            buffers["xi_g"],
            buffers["xi_t"],
            buffers["tmp_a"],
            buffers["tmp_b"],
        )

    def preprocess(
        self,
        aperture_filter: Optional[Callable[..., Any]] = None,
        angle_method: str = "haversine",
        release_host_pairs: bool = False,
    ) -> None:
        """
        Calculates the pairs and their angles for all patches for 2PCF & aperture mass.
        """
        logger.info("Calculating pairs for aperture mass")
        if aperture_filter is None:
            self.calculate_pairs_M_a()
        else:
            self.calculate_pairs_M_a(aperture_filter=aperture_filter)
        logger.info("Calculating pairs for 2PCF")
        self.calculate_pairs_2PCF(angle_method=angle_method)
        logger.info("Preparing flattened pair arrays on backend device")
        if release_host_pairs:
            self.prepare(release_host_pairs=True)
        else:
            self.prepare()

    def precompute(
        self,
        aperture_filter: Optional[Callable[..., Any]] = None,
        angle_method: str = "haversine",
        release_host_pairs: bool = False,
    ) -> None:
        """Backward-compatible alias for preprocess()."""
        self.preprocess(
            aperture_filter=aperture_filter,
            angle_method=angle_method,
            release_host_pairs=release_host_pairs,
        )

    def save_pairs(self, filepath: str) -> None:
        PairIOHandler.save_pairs(self, filepath)

    def load_pairs(
        self,
        filepath: str,
        start_ind: int = 0,
        stop_ind: Optional[int] = None,
        release_host_pairs: bool = False,
    ) -> None:
        PairIOHandler.load_pairs(
            self,
            filepath,
            start_ind=start_ind,
            stop_ind=stop_ind,
            release_host_pairs=release_host_pairs,
        )

    def get_aperture_shear(
        self,
        g1: np.ndarray,
        g2: np.ndarray,
        w: np.ndarray,
        aperture_filter: Optional[Callable[..., Any]] = None,
        return_device: bool = True,
    ) -> np.ndarray:
        self._ensure_aperture_pairs(aperture_filter=aperture_filter)
        g1_arr = self._coerce_map_input_array(g1)
        g2_arr = self._coerce_map_input_array(g2)
        w_arr = self._coerce_map_input_array(w)
        kernel = getattr(self.backend, "aperture_shear_kernel", None)
        if kernel is None:
            raise RuntimeError(
                "Backend does not provide an aperture-shear kernel; use a supported backend."
            )

        if self.backend.name == "numpy":
            aperture_shear = np.zeros(self.n_patches, dtype=self.map_dtype)
            kernel(
                self.Q_inds_flat,
                self.Q_cos_flat,
                self.Q_sin_flat,
                self.Q_val_flat,
                self.Q_offsets,
                g1_arr,
                g2_arr,
                w_arr,
                self.Q_patch_area_flat,
                aperture_shear,
            )
            return aperture_shear

        module = self.backend.module
        backend_dtype = getattr(module, self.map_dtype.name)

        if self.compute_context.Q_inds_dev is None:
            # Populate (and keep) the device buffers instead of building
            # throwaway per-call copies of the full aperture geometry.
            self._prepare_aperture_device_buffers()
        Q_inds_dev = self.compute_context.Q_inds_dev
        Q_cos_dev = self.compute_context.Q_cos_dev
        Q_sin_dev = self.compute_context.Q_sin_dev
        Q_val_dev = self.compute_context.Q_val_dev
        Q_offsets_dev = self.compute_context.Q_offsets_dev
        Q_patch_area_dev = self.compute_context.Q_patch_area_dev
        g1_dev = self._to_backend_array(g1_arr, dtype=backend_dtype)
        g2_dev = self._to_backend_array(g2_arr, dtype=backend_dtype)
        w_dev = self._to_backend_array(w_arr, dtype=backend_dtype)

        weighted_num, weighted_den = self._get_aperture_scratch(
            Q_inds_dev.shape[0], backend_dtype
        )
        kernel(Q_inds_dev, Q_cos_dev, Q_sin_dev, Q_val_dev, g1_dev, g2_dev, w_dev, weighted_num, weighted_den)

        weighted_num_patch = module.add.reduceat(weighted_num, Q_offsets_dev[:-1])
        weighted_den_patch = module.add.reduceat(weighted_den, Q_offsets_dev[:-1])
        aperture_shear = Q_patch_area_dev * weighted_num_patch / weighted_den_patch
        if return_device and self.backend.name == "cupy":
            return aperture_shear
        return self.backend.to_numpy(aperture_shear)

    def get_aperture_density(
        self,
        map_values: np.ndarray,
        w: np.ndarray,
        aperture_filter: Optional[Callable[..., Any]] = None,
        return_device: bool = True,
    ) -> np.ndarray:
        self._ensure_aperture_pairs(aperture_filter=aperture_filter)

        map_values_arr = self._coerce_map_input_array(map_values)
        w_arr = self._coerce_map_input_array(w)
        kernel = getattr(self.backend, "aperture_density_kernel", None)
        if kernel is None:
            raise RuntimeError(
                "Backend does not provide an aperture-density kernel; use a supported backend."
            )

        if self.backend.name == "numpy":
            aperture_density = np.zeros(self.n_patches, dtype=self.map_dtype)
            kernel(
                self.Q_inds_flat,
                self.Q_val_flat,
                self.Q_offsets,
                map_values_arr,
                w_arr,
                self.Q_patch_area_flat,
                aperture_density,
            )
            return aperture_density

        module = self.backend.module
        backend_dtype = getattr(module, self.map_dtype.name)

        if self.compute_context.Q_inds_dev is None:
            self._prepare_aperture_device_buffers()
        Q_inds_dev = self.compute_context.Q_inds_dev
        Q_val_dev = self.compute_context.Q_val_dev
        Q_offsets_dev = self.compute_context.Q_offsets_dev
        Q_patch_area_dev = self.compute_context.Q_patch_area_dev
        map_values_dev = self._to_backend_array(map_values_arr, dtype=backend_dtype)
        w_dev = self._to_backend_array(w_arr, dtype=backend_dtype)

        weighted_num, weighted_den = self._get_aperture_scratch(
            Q_inds_dev.shape[0], backend_dtype
        )
        kernel(Q_inds_dev, Q_val_dev, map_values_dev, w_dev, weighted_num, weighted_den)

        weighted_num_patch = module.add.reduceat(weighted_num, Q_offsets_dev[:-1])
        weighted_den_patch = module.add.reduceat(weighted_den, Q_offsets_dev[:-1])
        aperture_density = Q_patch_area_dev * weighted_num_patch / weighted_den_patch
        if return_device and self.backend.name == "cupy":
            return aperture_density
        return self.backend.to_numpy(aperture_density)

    def prepare(self, release_host_pairs: bool = False) -> None:
        """Prepares pair arrays for correlation calculations on the backend device.

        Args:
            release_host_pairs:
                If ``True``, releases host-side pair arrays (``pair_inds``,
                ``pair_exp2phi``, ``bins``) after device buffers are built
                to reduce RAM usage for large runs.
        """
        host_pairs_available = (
            self.pair_inds is not None
            and self.pair_exp2phi is not None
            and self.bins is not None
        )
        if not host_pairs_available:
            if (
                self.inds_dev is not None
                and self.exp2phi_dev is not None
                and self.bins_dev is not None
                and self.tot_bins_reduceat_dev is not None
            ):
                return
            raise RuntimeError(
                "Host pair arrays were released and prepared device buffers are "
                "unavailable; reload or recompute pairs before prepare()."
            )

        size = 0
        ninds = []
        for i in range(self.n_patches):
            patchsize = np.sum(self.bins[i])
            size += patchsize
            ninds.append(patchsize)

        first_patch_ind = np.append(0, np.cumsum(ninds)).astype(int)
        # np.empty: every element is written in the loop below
        temp_inds = np.empty((2, int(size)), dtype=self.index_dtype)
        temp_exp2phi = np.empty((2, int(size)), dtype=self.rotation_complex_dtype)
        temp_bins = np.empty((self.n_patches * self.nbins), dtype=self.index_dtype)
        temp_bins_tot = np.empty((self.n_patches * self.nbins + 1), dtype=self.index_dtype)
        temp_bins_tot[0] = 0

        for i in range(self.n_patches):
            temp_inds[:, first_patch_ind[i] : first_patch_ind[i + 1]] = self.pair_inds[
                i
            ]
            temp_exp2phi[:, first_patch_ind[i] : first_patch_ind[i + 1]] = (
                self.pair_exp2phi[i]
            )
            temp_bins[i * self.nbins : (i + 1) * self.nbins] = self.bins[i]
            temp_bins_tot[1 + i * self.nbins : 1 + (i + 1) * self.nbins] = (
                first_patch_ind[i] + self.bins[i].cumsum()
            )

        self.inds_dev = self.backend.to_device(temp_inds)
        module = self.backend.module
        _index_device_dtype = getattr(module, self.index_dtype.name)
        self.compute_context.inds_i_dev = module.ascontiguousarray(
            self.inds_dev[0].astype(_index_device_dtype, copy=False)
        )
        self.compute_context.inds_j_dev = module.ascontiguousarray(
            self.inds_dev[1].astype(_index_device_dtype, copy=False)
        )
        self.exp2phi_dev = self.backend.to_device(temp_exp2phi)
        self.bins_dev = self.backend.to_device(temp_bins)
        self.tot_bins_dev = self.backend.to_device(temp_bins_tot)
        self.tot_bins_reduceat_dev = self.backend.to_device(
            temp_bins_tot.astype(np.int64, copy=False)
        )
        self._prepare_aperture_device_buffers()
        self.ntotpairs = size
        self.compute_context.prepare_version += 1
        if release_host_pairs:
            self.pair_inds = None
            self.pair_exp2phi = None
            self.bins = None

    def compute_shear_shear(
        self,
        g11: np.ndarray,
        g21: np.ndarray,
        g12: np.ndarray,
        g22: np.ndarray,
        w1: np.ndarray,
        w2: np.ndarray,
        sumofweights: Optional[Union[np.ndarray, float]] = None,
        return_device: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute xi+/xi- for one map pair.

        If ``sumofweights`` is provided explicitly, weight fingerprint/cache
        checks are bypassed.
        """
        if self.inds_dev is None:
            self.prepare()
        return_numpy = not (return_device and self.backend.name == "cupy")
        if (g11 is g12) and (g21 is g22) and (w1 is w2):
            return self._xipm_auto(g11, g21, w1, sumofweights=sumofweights, return_numpy=return_numpy)

        return self._xipm_cross(
            g11,
            g21,
            g12,
            g22,
            w1,
            w2,
            sumofweights_ab=sumofweights,
            sumofweights_ba=sumofweights,
            return_numpy=return_numpy,
        )

    def compute_density_density(
        self,
        density1: np.ndarray,
        density2: np.ndarray,
        w1: np.ndarray,
        w2: np.ndarray,
        sumofweights: Optional[Union[np.ndarray, float]] = None,
        return_device: bool = True,
    ) -> Tuple[np.ndarray]:
        """Compute scalar density-density 2PCF (w(theta)) for one map pair."""
        if self.inds_dev is None:
            self.prepare()

        density1_dev = self._to_backend_array(density1, dtype=self.map_dtype)
        density2_dev = self._to_backend_array(density2, dtype=self.map_dtype)
        w1_dev = self._to_backend_array(w1, dtype=self.map_dtype)
        w2_dev = self._to_backend_array(w2, dtype=self.map_dtype)

        if sumofweights is not None:
            sum_ab = self._normalize_xipm_sumofweights(sumofweights)
            sum_ba = sum_ab
        else:
            # CPU: denominators come out of the kernel itself (below).
            # GPU: fingerprint-cached gather+reduce.
            sum_ab = None
            sum_ba = None

        density_density_kernel = getattr(self.backend, "kernel_density_density", None)
        if density_density_kernel is None:
            raise RuntimeError(
                "Backend does not provide a density-density kernel; use a supported backend."
            )

        if self.backend.name == "numpy":
            nbins_total = int(self.tot_bins_reduceat_dev.shape[0] - 1)
            out_ab = np.empty(nbins_total, dtype=self.map_dtype)
            out_ba = np.empty(nbins_total, dtype=self.map_dtype)
            out_ab_w = np.empty(nbins_total, dtype=self.map_dtype)
            out_ba_w = np.empty(nbins_total, dtype=self.map_dtype)
            offsets = np.asarray(self.tot_bins_reduceat_dev, dtype=np.int64)
            density_density_kernel(
                density1_dev,
                density2_dev,
                w1_dev,
                w2_dev,
                self.inds_dev[0],
                self.inds_dev[1],
                offsets,
                out_ab,
                out_ba,
                out_ab_w,
                out_ba_w,
            )
            w_ab_num = out_ab
            w_ba_num = out_ba
            if sum_ab is None:
                sum_ab = out_ab_w
                sum_ba = out_ba_w
        else:
            if sum_ab is None:
                sum_ab = self._get_xipm_sumofweights(w1_dev, w2_dev)
                sum_ba = self._get_xipm_sumofweights(w2_dev, w1_dev)
            out_ab, out_ba = self._get_pair_scratch(self.map_dtype, 2)
            density_density_kernel(
                density1_dev,
                density2_dev,
                w1_dev,
                w2_dev,
                self.inds_dev[0],
                self.inds_dev[1],
                out_ab,
            )
            w_ab_num = self._reduce_pairs(out_ab)

            density_density_kernel(
                density1_dev,
                density2_dev,
                w1_dev,
                w2_dev,
                self.inds_dev[1],
                self.inds_dev[0],
                out_ba,
            )
            w_ba_num = self._reduce_pairs(out_ba)

        w_ab = self._normalize_scalar_pairs(w_ab_num, sum_ab)
        w_ba = self._normalize_scalar_pairs(w_ba_num, sum_ba)
        w_theta = 0.5 * (w_ab + w_ba)
        if return_device and self.backend.name == "cupy":
            return (self.backend.module.real(w_theta),)
        return (np.real(self.backend.to_numpy(w_theta)),)

    def compute_density_shear(
        self,
        density_lens: np.ndarray,
        g1_source: np.ndarray,
        g2_source: np.ndarray,
        w_lens: np.ndarray,
        w_source: np.ndarray,
        sumofweights: Optional[Union[np.ndarray, float]] = None,
        return_device: bool = True,
    ) -> Tuple[np.ndarray]:
        """Compute scalar-shear 2PCF (gamma_t) for one lens/source map pair."""
        if self.inds_dev is None:
            self.prepare()

        density_lens_dev = self._to_backend_array(density_lens, dtype=self.map_dtype)
        g1_source_dev = self._to_backend_array(g1_source, dtype=self.map_dtype)
        g2_source_dev = self._to_backend_array(g2_source, dtype=self.map_dtype)
        w_lens_dev = self._to_backend_array(w_lens, dtype=self.map_dtype)
        w_source_dev = self._to_backend_array(w_source, dtype=self.map_dtype)

        if sumofweights is not None:
            sumofweights_dev = self._normalize_xipm_sumofweights(sumofweights)
        else:
            sumofweights_dev = None

        density_shear_kernel = getattr(self.backend, "kernel_density_shear", None)
        if density_shear_kernel is None:
            raise RuntimeError(
                "Backend does not provide a density-shear kernel; use a supported backend."
            )

        if self.backend.name == "numpy":
            nbins_total = int(self.tot_bins_reduceat_dev.shape[0] - 1)
            out_ab = np.empty(nbins_total, dtype=self.map_dtype)
            out_ba = np.empty(nbins_total, dtype=self.map_dtype)
            out_ab_w = np.empty(nbins_total, dtype=self.map_dtype)
            out_ba_w = np.empty(nbins_total, dtype=self.map_dtype)
            offsets = np.asarray(self.tot_bins_reduceat_dev, dtype=np.int64)
            density_shear_kernel(
                density_lens_dev,
                g1_source_dev,
                g2_source_dev,
                w_lens_dev,
                w_source_dev,
                self.inds_dev[0],
                self.inds_dev[1],
                self.exp2phi_dev[0],
                self.exp2phi_dev[1],
                offsets,
                out_ab,
                out_ba,
                out_ab_w,
                out_ba_w,
            )
            gamma_num = out_ab + out_ba
            if sumofweights_dev is None:
                sumofweights_dev = out_ab_w + out_ba_w
        else:
            if sumofweights_dev is None:
                sum_ab = self._get_xipm_sumofweights(w_lens_dev, w_source_dev)
                sum_ba = self._get_xipm_sumofweights(w_source_dev, w_lens_dev)
                sumofweights_dev = sum_ab + sum_ba
            out_ab, out_ba = self._get_pair_scratch(self.map_dtype, 2)
            density_shear_kernel(
                density_lens_dev,
                g1_source_dev,
                g2_source_dev,
                w_lens_dev,
                w_source_dev,
                self.inds_dev[0],
                self.inds_dev[1],
                self.exp2phi_dev[1],
                out_ab,
            )

            density_shear_kernel(
                density_lens_dev,
                g1_source_dev,
                g2_source_dev,
                w_lens_dev,
                w_source_dev,
                self.inds_dev[1],
                self.inds_dev[0],
                self.exp2phi_dev[0],
                out_ba,
            )
            gamma_num = self._reduce_pairs(out_ab) + self._reduce_pairs(out_ba)

        gamma_t = self._normalize_scalar_pairs(gamma_num, sumofweights_dev)
        if return_device and self.backend.name == "cupy":
            return (self.backend.module.real(gamma_t),)
        return (np.real(self.backend.to_numpy(gamma_t)),)

    def _xipm_auto(
        self,
        g1: np.ndarray,
        g2: np.ndarray,
        w: np.ndarray,
        sumofweights: Optional[Union[np.ndarray, float]] = None,
        return_numpy: bool = True,
    ) -> Tuple[Any, Any]:
        g1_dev = self._to_backend_array(g1, dtype=self.map_dtype)
        g2_dev = self._to_backend_array(g2, dtype=self.map_dtype)
        w_dev = self._to_backend_array(w, dtype=self.map_dtype)

        if sumofweights is not None:
            sumofweights_dev = self._normalize_xipm_sumofweights(sumofweights)
        else:
            sumofweights_dev = None

        xipm_auto_corr_kernel = getattr(self.backend, "xipm_auto_corr_kernel", None)
        if xipm_auto_corr_kernel is None:
            raise RuntimeError(
                "Backend does not provide an xipm auto-correlation kernel; use a supported backend."
            )

        if self.backend.name == "numpy":
            nbins_total = int(self.tot_bins_reduceat_dev.shape[0] - 1)
            out_p = np.empty(nbins_total, dtype=self.map_dtype)
            out_m = np.empty(nbins_total, dtype=self.map_dtype)
            out_w = np.empty(nbins_total, dtype=self.map_dtype)
            offsets = np.asarray(self.tot_bins_reduceat_dev, dtype=np.int64)
            xipm_auto_corr_kernel(
                g1_dev,
                g2_dev,
                g1_dev,
                g2_dev,
                w_dev,
                w_dev,
                self.inds_dev[0],
                self.inds_dev[1],
                self.exp2phi_dev[0],
                self.exp2phi_dev[1],
                offsets,
                out_p,
                out_m,
                out_w,
            )
            # Preserve the historical output dtype, which follows the
            # rotation precision (float32 rotations -> float32 xi±).
            real_dtype = (
                np.float32
                if self.rotation_complex_dtype == np.dtype(np.complex64)
                else np.float64
            )
            xip_num = out_p.astype(real_dtype, copy=False)
            xim_num = out_m.astype(real_dtype, copy=False)
            if sumofweights_dev is None:
                sumofweights_dev = out_w
        else:
            complex_dtype = (
                self.backend.module.complex64
                if self.rotation_complex_dtype == np.dtype(np.complex64)
                else self.backend.module.complex128
            )
            if sumofweights_dev is None:
                sumofweights_dev = self._get_xipm_sumofweights(w_dev, w_dev)
            out_p, out_m = self._get_pair_scratch(complex_dtype, 2)

            xipm_auto_corr_kernel(
                g1_dev,
                g2_dev,
                g1_dev,
                g2_dev,
                w_dev,
                w_dev,
                self.inds_dev[0],
                self.inds_dev[1],
                self.exp2phi_dev[0],
                self.exp2phi_dev[1],
                out_p,
                out_m,
            )

            xip_num = self.backend.module.real(self._reduce_pairs(out_p))
            xim_num = self.backend.module.real(self._reduce_pairs(out_m))
        xip_dev, xim_dev = self._normalize_xipm_pairs(xip_num, xim_num, sumofweights_dev)

        if return_numpy:
            return (
                np.real(self.backend.to_numpy(xip_dev)),
                np.real(self.backend.to_numpy(xim_dev)),
            )

        return xip_dev, xim_dev

    def _xipm_cross(
        self,
        g11: np.ndarray,
        g21: np.ndarray,
        g12: np.ndarray,
        g22: np.ndarray,
        w1: np.ndarray,
        w2: np.ndarray,
        sumofweights_ab: Optional[Union[np.ndarray, float]] = None,
        sumofweights_ba: Optional[Union[np.ndarray, float]] = None,
        return_numpy: bool = True,
    ) -> Tuple[Any, Any]:
        g11_dev = self._to_backend_array(g11, dtype=self.map_dtype)
        g21_dev = self._to_backend_array(g21, dtype=self.map_dtype)
        g12_dev = self._to_backend_array(g12, dtype=self.map_dtype)
        g22_dev = self._to_backend_array(g22, dtype=self.map_dtype)
        w1_dev = self._to_backend_array(w1, dtype=self.map_dtype)
        w2_dev = self._to_backend_array(w2, dtype=self.map_dtype)

        if sumofweights_ab is not None:
            sum_ab = self._normalize_xipm_sumofweights(sumofweights_ab)
        else:
            sum_ab = None
        if sumofweights_ba is not None:
            sum_ba = self._normalize_xipm_sumofweights(sumofweights_ba)
        else:
            sum_ba = None

        xipm_cross_corr_kernel = getattr(self.backend, "xipm_cross_corr_kernel", None)
        if xipm_cross_corr_kernel is None:
            raise RuntimeError(
                "Backend does not provide an xipm cross-correlation kernel; "
                "use a supported backend."
            )

        if self.backend.name == "numpy":
            nbins_total = int(self.tot_bins_reduceat_dev.shape[0] - 1)
            out_ab_p = np.empty(nbins_total, dtype=self.map_dtype)
            out_ab_m = np.empty(nbins_total, dtype=self.map_dtype)
            out_ba_p = np.empty(nbins_total, dtype=self.map_dtype)
            out_ba_m = np.empty(nbins_total, dtype=self.map_dtype)
            out_ab_w = np.empty(nbins_total, dtype=self.map_dtype)
            out_ba_w = np.empty(nbins_total, dtype=self.map_dtype)
            offsets = np.asarray(self.tot_bins_reduceat_dev, dtype=np.int64)
            xipm_cross_corr_kernel(
                g11_dev,
                g21_dev,
                g12_dev,
                g22_dev,
                w1_dev,
                w2_dev,
                self.inds_dev[0],
                self.inds_dev[1],
                self.exp2phi_dev[0],
                self.exp2phi_dev[1],
                offsets,
                out_ab_p,
                out_ab_m,
                out_ba_p,
                out_ba_m,
                out_ab_w,
                out_ba_w,
            )

            # Preserve the historical output dtype, which follows the
            # rotation precision (float32 rotations -> float32 xi±).
            real_dtype = (
                np.float32
                if self.rotation_complex_dtype == np.dtype(np.complex64)
                else np.float64
            )
            xip_ab_num = out_ab_p.astype(real_dtype, copy=False)
            xim_ab_num = out_ab_m.astype(real_dtype, copy=False)
            xip_ba_num = out_ba_p.astype(real_dtype, copy=False)
            xim_ba_num = out_ba_m.astype(real_dtype, copy=False)
            if sum_ab is None:
                sum_ab = out_ab_w
            if sum_ba is None:
                sum_ba = out_ba_w
        else:
            complex_dtype = (
                self.backend.module.complex64
                if self.rotation_complex_dtype == np.dtype(np.complex64)
                else self.backend.module.complex128
            )
            if sum_ab is None:
                sum_ab = self._get_xipm_sumofweights(w1_dev, w2_dev)
            if sum_ba is None:
                sum_ba = self._get_xipm_sumofweights(w2_dev, w1_dev)
            out_ab_p, out_ab_m, out_ba_p, out_ba_m = self._get_pair_scratch(
                complex_dtype, 4
            )

            xipm_cross_corr_kernel(
                g11_dev,
                g21_dev,
                g12_dev,
                g22_dev,
                w1_dev,
                w2_dev,
                self.inds_dev[0],
                self.inds_dev[1],
                self.exp2phi_dev[0],
                self.exp2phi_dev[1],
                out_ab_p,
                out_ab_m,
                out_ba_p,
                out_ba_m,
            )

            xip_ab_num = self.backend.module.real(self._reduce_pairs(out_ab_p))
            xim_ab_num = self.backend.module.real(self._reduce_pairs(out_ab_m))
            xip_ba_num = self.backend.module.real(self._reduce_pairs(out_ba_p))
            xim_ba_num = self.backend.module.real(self._reduce_pairs(out_ba_m))

        xip_ab_dev, xim_ab_dev = self._normalize_xipm_pairs(
            xip_ab_num, xim_ab_num, sum_ab
        )
        xip_ba_dev, xim_ba_dev = self._normalize_xipm_pairs(
            xip_ba_num, xim_ba_num, sum_ba
        )

        xip_dev = (xip_ab_dev + xip_ba_dev) / 2
        xim_dev = (xim_ab_dev + xim_ba_dev) / 2

        if return_numpy:
            return (
                np.real(self.backend.to_numpy(xip_dev)),
                np.real(self.backend.to_numpy(xim_dev)),
            )

        return xip_dev, xim_dev

    def _normalize_by_weights(self, num: Any, den: Any) -> Any:
        """``num / den`` with zeros where ``den == 0``.

        Uses only device-side operations and array metadata — no
        device→host transfer or synchronization (``den`` may be a scalar
        or an array on either backend).
        """
        module = self.backend.module
        mask = den != 0
        safe_den = module.where(mask, den, 1)
        out = (num / safe_den) * mask
        return out.astype(num.dtype, copy=False)

    def _normalize_xipm_pairs(
        self, xip_num: Any, xim_num: Any, sumofweights_dev: Any
    ) -> Tuple[Any, Any]:
        xip = self._normalize_by_weights(xip_num, sumofweights_dev)
        xim = self._normalize_by_weights(xim_num, sumofweights_dev)
        xip = xip.reshape((self.n_patches, self.nbins))
        xim = xim.reshape((self.n_patches, self.nbins))
        return xip, xim

    def _normalize_scalar_pairs(self, num: Any, sumofweights_dev: Any) -> Any:
        val = self._normalize_by_weights(num, sumofweights_dev)
        return val.reshape((self.n_patches, self.nbins))

    def _normalize_tomo_sumofweights_per_comb(
        self, sumofweights: Union[np.ndarray, float], nzbin_combs: int
    ) -> Any:
        nbins_total = self.n_patches * self.nbins
        if self._is_backend_native_array(sumofweights):
            # Device arrays whose shape already matches an accepted layout
            # are reshaped on device — no GPU→host round-trip.
            backend_dtype = getattr(self.backend.module, self.map_dtype.name)
            s = sumofweights
            if s.ndim == 2 and s.shape == (nzbin_combs, nbins_total):
                return s.astype(backend_dtype, copy=False)
            if s.ndim == 3 and s.shape == (nzbin_combs, self.n_patches, self.nbins):
                return s.reshape(nzbin_combs, nbins_total).astype(backend_dtype, copy=False)
            if s.ndim == 2 and nzbin_combs == 1 and s.shape == (self.n_patches, self.nbins):
                return s.reshape(1, nbins_total).astype(backend_dtype, copy=False)
            if s.ndim == 1 and s.size == nzbin_combs * nbins_total:
                return s.reshape(nzbin_combs, nbins_total).astype(backend_dtype, copy=False)
            if s.ndim == 1 and nzbin_combs == 1 and s.size == nbins_total:
                return s.reshape(1, nbins_total).astype(backend_dtype, copy=False)
            # scalar / per-combination broadcasts fall through to the host path
        sum_np = np.asarray(self.backend.to_numpy(sumofweights), dtype=self.map_dtype)

        if sum_np.ndim == 0:
            expanded = np.full((nzbin_combs, nbins_total), sum_np.item(), dtype=self.map_dtype)
            return self.backend.to_device(expanded)

        if sum_np.ndim == 1:
            if sum_np.size == nzbin_combs:
                expanded = np.repeat(sum_np.reshape(nzbin_combs, 1), nbins_total, axis=1)
                return self.backend.to_device(expanded)
            if nzbin_combs == 1 and sum_np.size == nbins_total:
                return self.backend.to_device(sum_np.reshape(1, nbins_total))
            if sum_np.size == nzbin_combs * nbins_total:
                return self.backend.to_device(sum_np.reshape(nzbin_combs, nbins_total))

        if sum_np.ndim == 2:
            if sum_np.shape == (nzbin_combs, nbins_total):
                return self.backend.to_device(sum_np)
            if nzbin_combs == 1 and sum_np.shape == (self.n_patches, self.nbins):
                return self.backend.to_device(sum_np.reshape(1, nbins_total))

        if sum_np.ndim == 3 and sum_np.shape == (nzbin_combs, self.n_patches, self.nbins):
            return self.backend.to_device(sum_np.reshape(nzbin_combs, nbins_total))

        raise ValueError(
            "sumofweights must be scalar, (nzbin_combs,), "
            f"({nzbin_combs}, {self.n_patches}, {self.nbins}), "
            f"or ({nzbin_combs}, {nbins_total}); got {sum_np.shape}"
        )

    def _normalize_tomo_sumofweights_directional(
        self, sumofweights: Union[np.ndarray, float], nzbin_combs: int
    ) -> Any:
        if (
            self._is_backend_native_array(sumofweights)
            and sumofweights.ndim >= 2
            and sumofweights.shape[0] == 2
        ):
            sum_ab = self._normalize_tomo_sumofweights_per_comb(sumofweights[0], nzbin_combs)
            sum_ba = self._normalize_tomo_sumofweights_per_comb(sumofweights[1], nzbin_combs)
            return self.backend.module.stack((sum_ab, sum_ba), axis=0)
        sum_np = np.asarray(self.backend.to_numpy(sumofweights), dtype=self.map_dtype)
        if sum_np.ndim >= 2 and sum_np.shape[0] == 2:
            sum_ab = self._normalize_tomo_sumofweights_per_comb(sum_np[0], nzbin_combs)
            sum_ba = self._normalize_tomo_sumofweights_per_comb(sum_np[1], nzbin_combs)
            return self.backend.module.stack((sum_ab, sum_ba), axis=0)

        per_comb = self._normalize_tomo_sumofweights_per_comb(sum_np, nzbin_combs)
        return self.backend.module.stack((per_comb, per_comb), axis=0)

    def _is_backend_native_array(self, array: Any) -> bool:
        if self.backend.name != "cupy":
            return False

        module_ndarray = getattr(self.backend.module, "ndarray", None)
        if module_ndarray is not None and not isinstance(array, module_ndarray):
            return False
        if not hasattr(array, "device"):
            return False

        array_device = getattr(array, "device", None)
        array_device_id = getattr(array_device, "id", None)
        target_device_id = getattr(self.backend, "device_id", None)
        if (
            target_device_id is not None
            and array_device_id is not None
            and int(array_device_id) != int(target_device_id)
        ):
            return False

        return True

    def _to_backend_array(self, array: Any, dtype: Optional[Any] = None) -> Any:
        backend_array = array if self._is_backend_native_array(array) else self.backend.to_device(array)
        if dtype is not None:
            backend_array = backend_array.astype(dtype, copy=False)
        return backend_array

    def _coerce_map_input_array(self, array: Any) -> Any:
        if self._is_backend_native_array(array):
            return array.astype(self.map_dtype, copy=False)
        return np.asarray(array, dtype=self.map_dtype)

    @property
    def _index_device_dtype(self) -> Any:
        """Backend-native dtype matching self.index_dtype (e.g. cupy.int32 or numpy.int64)."""
        return getattr(self.backend.module, self.index_dtype.name)

    def _sumofweights_cache_get_or_compute(
        self, cache_attr: str, key: Any, compute: Callable[[], Any]
    ) -> Any:
        """Small LRU cache for per-weight-set sum-of-weights arrays.

        Entries are invalidated when the pair geometry changes
        (``prepare_version``) and capped so per-map weights cannot
        accumulate device arrays without bound.
        """
        ctx = self.compute_context
        entry = getattr(ctx, cache_attr, None)
        if not isinstance(entry, dict) or entry.get("__version__") != ctx.prepare_version:
            entry = {"__version__": ctx.prepare_version}
            setattr(ctx, cache_attr, entry)
        if key in entry:
            value = entry.pop(key)
            entry[key] = value  # move to end (LRU)
            return value
        value = compute()
        entry[key] = value
        while len(entry) > 33:  # 32 entries + the version marker
            oldest = next(k for k in entry if k != "__version__")
            entry.pop(oldest)
        return value

    def _get_pair_scratch(self, dtype: Any, count: int) -> Tuple[Any, ...]:
        """Reusable ntotpairs-sized device scratch buffers.

        The per-pair ElementwiseKernels overwrite every element, so the
        buffers are allocated uninitialised and reused across per-map
        calls instead of being zero-filled and reallocated each time.
        """
        ctx = self.compute_context
        if getattr(ctx, "pair_scratch", None) is None:
            ctx.pair_scratch = {}
        key = (np.dtype(dtype).str, int(self.ntotpairs))
        bufs = ctx.pair_scratch.get(key)
        if bufs is None:
            bufs = []
            ctx.pair_scratch.clear()  # geometry changed: drop stale sizes
            ctx.pair_scratch[key] = bufs
        module = self.backend.module
        empty = getattr(module, "empty", None)
        while len(bufs) < count:
            if empty is not None:
                bufs.append(empty(self.ntotpairs, dtype=dtype))
            else:
                bufs.append(self.backend.zeros(self.ntotpairs, dtype=dtype))
        return tuple(bufs[:count])

    def _get_aperture_scratch(self, size: int, dtype: Any) -> Tuple[Any, Any]:
        """Reusable scratch buffers for the aperture ElementwiseKernels."""
        ctx = self.compute_context
        cached = getattr(ctx, "aperture_scratch", None)
        key = (np.dtype(dtype).str, int(size))
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        module = self.backend.module
        empty = getattr(module, "empty", None)
        if empty is not None:
            num = empty(size, dtype=dtype)
            den = empty(size, dtype=dtype)
        else:
            num = self.backend.zeros(size, dtype=dtype)
            den = self.backend.zeros(size, dtype=dtype)
        ctx.aperture_scratch = (key, num, den)
        return num, den

    def _fingerprint_weights(
        self, w_np: Any
    ) -> Tuple[Tuple[int, ...], str, str]:
        if self._is_backend_native_array(w_np):
            shape = tuple(getattr(w_np, "shape", ()))
            dtype_str = np.dtype(getattr(w_np, "dtype", self.map_dtype)).str
            device_id = getattr(getattr(w_np, "device", None), "id", "unknown")
            data_ptr = getattr(getattr(w_np, "data", None), "ptr", None)
            if data_ptr is None:
                data_ptr = id(w_np)
            return (shape, dtype_str, f"device:{device_id};ptr:{data_ptr}")

        w_contiguous = np.ascontiguousarray(w_np)
        # Read-only arrays cannot change content, so their digest can be
        # memoized on identity: freeze weight maps with
        # ``w.flags.writeable = False`` to skip rehashing on every call.
        memo_key = None
        if w_contiguous is w_np and not w_contiguous.flags.writeable:
            memo_key = (
                id(w_np),
                w_np.__array_interface__["data"][0],
                w_np.shape,
                w_np.dtype.str,
            )
            cached = _FINGERPRINT_MEMO.get(memo_key)
            if cached is not None:
                return cached
        # blake2b accepts buffer-protocol objects: hashing a memoryview gives
        # the same digest as .tobytes() without the full byte copy.
        digest = hashlib.blake2b(memoryview(w_contiguous).cast("B")).hexdigest()
        result = (w_contiguous.shape, w_contiguous.dtype.str, digest)
        if memo_key is not None:
            if len(_FINGERPRINT_MEMO) > 64:
                _FINGERPRINT_MEMO.clear()
            _FINGERPRINT_MEMO[memo_key] = result
        return result

    def _normalize_xipm_sumofweights(
        self, sumofweights: Union[np.ndarray, float]
    ) -> Any:
        expected_size = self.n_patches * self.nbins
        if self._is_backend_native_array(sumofweights):
            # Device arrays are validated and reshaped from metadata only —
            # no GPU→host round-trip.
            backend_dtype = getattr(self.backend.module, self.map_dtype.name)
            if sumofweights.ndim == 0:
                return sumofweights.astype(backend_dtype, copy=False)
            if sumofweights.size != expected_size:
                raise ValueError(
                    "sumofweights must have shape "
                    f"({self.n_patches}, {self.nbins}) or {expected_size} elements; "
                    f"got {sumofweights.shape}"
                )
            return sumofweights.reshape(expected_size).astype(backend_dtype, copy=False)
        sumofweights_np = np.asarray(
            self.backend.to_numpy(sumofweights), dtype=self.map_dtype
        )
        if sumofweights_np.ndim == 0:
            return self.backend.to_device(sumofweights_np)
        if sumofweights_np.size != expected_size:
            raise ValueError(
                "sumofweights must have shape "
                f"({self.n_patches}, {self.nbins}) or {expected_size} elements; "
                f"got {sumofweights_np.shape}"
            )
        return self.backend.to_device(sumofweights_np.reshape(expected_size))

    def _reduce_pairs(self, values: Any) -> Any:
        """Reduce pair-valued arrays into flattened per-patch/per-bin sums."""
        starts = self.tot_bins_reduceat_dev[:-1]
        values_dev = self.backend.to_device(values)
        values_ndim = int(getattr(values_dev, "ndim", np.ndim(values_dev)))
        cached = getattr(self, "_reduce_pairs_cache", None)
        pv = self.compute_context.prepare_version
        if cached is not None and cached[0] == pv:
            starts_np = cached[1]
        else:
            starts_np = np.asarray(self.backend.to_numpy(starts), dtype=np.int64)
            self._reduce_pairs_cache = (pv, starts_np)
        nstarts = starts_np.size
        if values_ndim <= 1:
            nvals = int(getattr(values_dev, "size", np.size(values_dev)))
            valid_idx_np = np.where(starts_np < nvals)[0]
            if valid_idx_np.size == nstarts:
                reduced = self.backend.add.reduceat(values_dev, starts)
            else:
                reduced = self.backend.zeros(nstarts, dtype=values_dev.dtype)
                if valid_idx_np.size > 0:
                    starts_valid = self.backend.to_device(
                        starts_np[valid_idx_np].astype(np.int64, copy=False)
                    )
                    valid_idx = self.backend.to_device(
                        valid_idx_np.astype(np.int64, copy=False)
                    )
                    reduced_valid = self.backend.add.reduceat(values_dev, starts_valid)
                    reduced[valid_idx] = reduced_valid
            reduced[self.bins_dev == 0] = 0
            return reduced

        nvals = int(values_dev.shape[-1])
        valid_idx_np = np.where(starts_np < nvals)[0]
        output_shape = values_dev.shape[:-1] + (nstarts,)

        if valid_idx_np.size == nstarts:
            try:
                reduced = self.backend.add.reduceat(values_dev, starts, axis=-1)
            except TypeError:
                reduced = self.backend.module.stack(
                    [self.backend.add.reduceat(row, starts) for row in values_dev], axis=0
                )
        else:
            reduced = self.backend.zeros(output_shape, dtype=values_dev.dtype)
            if valid_idx_np.size > 0:
                starts_valid = self.backend.to_device(
                    starts_np[valid_idx_np].astype(np.int64, copy=False)
                )
                valid_idx = self.backend.to_device(
                    valid_idx_np.astype(np.int64, copy=False)
                )
                try:
                    reduced_valid = self.backend.add.reduceat(
                        values_dev, starts_valid, axis=-1
                    )
                except TypeError:
                    reduced_valid = self.backend.module.stack(
                        [
                            self.backend.add.reduceat(row, starts_valid)
                            for row in values_dev
                        ],
                        axis=0,
                    )
                reduced[..., valid_idx] = reduced_valid

        reduced[..., self.bins_dev == 0] = 0
        return reduced

    def _compute_xipm_sumofweights(self, w1_dev: Any, w2_dev: Any) -> Any:
        if self.inds_dev is None:
            self.prepare()

        return self._reduce_pairs(w1_dev[self.inds_dev[0]] * w2_dev[self.inds_dev[1]])

    def _get_xipm_sumofweights(self, w1_dev: Any, w2_dev: Any) -> Any:
        w_fingerprint = (
            self._fingerprint_weights(w1_dev),
            self._fingerprint_weights(w2_dev),
        )

        cache = self.compute_context._xipm_sumofweights_cache
        if not isinstance(cache, dict):
            migrated_cache: Dict[Any, Any] = {}
            if (
                cache is not None
                and self.compute_context._xipm_sumofweights_cache_w_fingerprint is not None
                and self.compute_context._xipm_sumofweights_cache_prepare_version
                == self.compute_context.prepare_version
            ):
                migrated_cache[
                    self.compute_context._xipm_sumofweights_cache_w_fingerprint
                ] = cache
            self.compute_context._xipm_sumofweights_cache = migrated_cache
            self.compute_context._xipm_sumofweights_cache_w_fingerprint = None
            cache = migrated_cache

        if (
            self.compute_context._xipm_sumofweights_cache_prepare_version
            != self.compute_context.prepare_version
        ):
            cache.clear()
            self.compute_context._xipm_sumofweights_cache_prepare_version = (
                self.compute_context.prepare_version
            )

        cached = cache.get(w_fingerprint)
        if cached is not None:
            return cached

        sumofweights_dev = self._compute_xipm_sumofweights(w1_dev, w2_dev)
        cache[w_fingerprint] = sumofweights_dev
        # Bound the cache: with per-map weights every map adds entries, which
        # would otherwise accumulate device arrays without limit.
        while len(cache) > 32:
            cache.pop(next(iter(cache)))
        return sumofweights_dev

    def _compute_tomo_sumofweights(
        self, w_dev: Any, nzbins: int, nzbin_combs: int
    ) -> Any:
        if self.inds_dev is None:
            self.prepare()

        map_backend_dtype = getattr(self.backend.module, self.map_dtype.name)
        sumofweights_dev = self.backend.zeros(
            (2, nzbin_combs, self.n_patches * self.nbins),
            dtype=map_backend_dtype,
        )

        k = 0
        for i in range(nzbins):
            for j in range(i, nzbins):
                sum_ij = self._reduce_pairs(
                    w_dev[i][self.inds_dev[0]] * w_dev[j][self.inds_dev[1]]
                )
                sumofweights_dev[0, k] = sum_ij
                if i == j:
                    sumofweights_dev[1, k] = sum_ij
                else:
                    sumofweights_dev[1, k] = self._reduce_pairs(
                        w_dev[j][self.inds_dev[0]] * w_dev[i][self.inds_dev[1]]
                    )
                k += 1

        return sumofweights_dev

    def _transpose_tomo_inputs_aos(self, shear_maps_dev: Any, w_dev: Any) -> Tuple[Any, Any]:
        """Convert tomography inputs from SoA to AoS for contiguous per-pixel loads."""
        module = self.backend.module
        shear_aos = module.ascontiguousarray(module.transpose(shear_maps_dev, (2, 0, 1)))
        weights_aos = module.ascontiguousarray(module.transpose(w_dev, (1, 0)))
        return shear_aos, weights_aos

    def _xipm_tomo_vectorized(
        self,
        shear_maps_dev: Any,
        w_dev: Any,
        sumofweights_dev: Any,
        nzbins: int,
        nzbin_combs: int,
        g1_fac: int,
        g2_fac: int,
    ) -> Optional[Tuple[Any, Any]]:
        tomo_kernel = getattr(self.backend, "xipm_tomo_vectorized_kernel", None)
        if tomo_kernel is None:
            return None

        if self.backend.name == "numpy":
            return self._xipm_tomo_vectorized_cpu(
                shear_maps_dev,
                w_dev,
                sumofweights_dev,
                nzbins,
                nzbin_combs,
                g1_fac,
                g2_fac,
            )

        if self.backend.name != "cupy":
            return None

        module = self.backend.module
        if g1_fac == 1 and g2_fac == 1:
            # No sign flip requested: skip the full-map scale-and-stack copy.
            shear_scaled = shear_maps_dev
        else:
            shear_scaled = module.stack(
                (g1_fac * shear_maps_dev[:, 0], g2_fac * shear_maps_dev[:, 1]),
                axis=1,
            )
        shear_aos, weights_aos = self._transpose_tomo_inputs_aos(shear_scaled, w_dev)

        inds_i = module.ascontiguousarray(self.inds_dev[0].astype(self._index_device_dtype, copy=False))
        inds_j = module.ascontiguousarray(self.inds_dev[1].astype(self._index_device_dtype, copy=False))
        exp_i = module.ascontiguousarray(self.exp2phi_dev[0])
        exp_j = module.ascontiguousarray(self.exp2phi_dev[1])
        bin_offsets = module.ascontiguousarray(
            self.tot_bins_reduceat_dev.astype(module.int64, copy=False)
        )
        comb_i, comb_j, auto_comb = self._get_tomo_combination_indices(
            nzbins, nzbin_combs
        )

        map_backend_dtype = getattr(module, self.map_dtype.name)
        nbins_total = int(self.n_patches * self.nbins)
        out_num = self.backend.zeros(
            (2, 2 * nzbin_combs, nbins_total), dtype=map_backend_dtype
        )

        launched = tomo_kernel(
            shear_aos,
            weights_aos,
            inds_i,
            inds_j,
            exp_i,
            exp_j,
            bin_offsets,
            comb_i,
            comb_j,
            out_num,
        )
        if not launched:
            return None

        xip_reduced = out_num[0]
        xim_reduced = out_num[1]
        xip_num = module.stack((xip_reduced[0::2], xip_reduced[1::2]), axis=0)
        xim_num = module.stack((xim_reduced[0::2], xim_reduced[1::2]), axis=0)
        xip = self.backend.zeros(
            (nzbin_combs, self.n_patches, self.nbins), dtype=map_backend_dtype
        )
        xim = self.backend.zeros(
            (nzbin_combs, self.n_patches, self.nbins), dtype=map_backend_dtype
        )

        half = self.map_dtype.type(0.5)
        for k in range(nzbin_combs):
            xip_ab, xim_ab = self._normalize_xipm_pairs(
                xip_num[0, k], xim_num[0, k], sumofweights_dev[0, k]
            )
            if auto_comb[k]:
                xip[k], xim[k] = xip_ab, xim_ab
            else:
                xip_ba, xim_ba = self._normalize_xipm_pairs(
                    xip_num[1, k], xim_num[1, k], sumofweights_dev[1, k]
                )
                xip[k] = half * (xip_ab + xip_ba)
                xim[k] = half * (xim_ab + xim_ba)

        return xip, xim

    def _xipm_tomo_vectorized_cpu(
        self,
        shear_maps_dev: Any,
        w_dev: Any,
        sumofweights_dev: Any,
        nzbins: int,
        nzbin_combs: int,
        g1_fac: int,
        g2_fac: int,
    ) -> Tuple[Any, Any]:
        if g1_fac == 1 and g2_fac == 1:
            # No sign flip requested: skip the full-map scale-and-stack copy.
            shear_scaled = shear_maps_dev
        else:
            shear_scaled = np.stack(
                (g1_fac * shear_maps_dev[:, 0], g2_fac * shear_maps_dev[:, 1]),
                axis=1,
            )
        shear_aos, weights_aos = self._transpose_tomo_inputs_aos(shear_scaled, w_dev)

        nbins_total = int(self.tot_bins_reduceat_dev.shape[0] - 1)
        out_p = np.empty((2 * nzbin_combs, nbins_total), dtype=self.map_dtype)
        out_m = np.empty((2 * nzbin_combs, nbins_total), dtype=self.map_dtype)
        out_w = np.empty((2 * nzbin_combs, nbins_total), dtype=self.map_dtype)
        comb_i, comb_j, auto_comb = self._get_tomo_combination_indices(
            nzbins, nzbin_combs
        )
        tomo_kernel = self.backend.xipm_tomo_vectorized_kernel
        offsets = np.asarray(self.tot_bins_reduceat_dev, dtype=np.int64)
        launched = tomo_kernel(
            shear_aos,
            weights_aos,
            np.ascontiguousarray(self.inds_dev[0]),
            np.ascontiguousarray(self.inds_dev[1]),
            np.ascontiguousarray(self.exp2phi_dev[0]),
            np.ascontiguousarray(self.exp2phi_dev[1]),
            offsets,
            np.ascontiguousarray(comb_i),
            np.ascontiguousarray(comb_j),
            out_p,
            out_m,
            out_w,
        )
        if launched is False:
            raise RuntimeError(
                "Backend tomography vectorized kernel unavailable for CPU backend."
            )

        if sumofweights_dev is None:
            # Denominators accumulated inside the kernel: rows 2k are the
            # A→B weight sums, rows 2k+1 the B→A sums.
            sumofweights_dev = np.stack((out_w[0::2], out_w[1::2]), axis=0)

        xip_num = np.stack((out_p[0::2], out_p[1::2]), axis=0)
        xim_num = np.stack((out_m[0::2], out_m[1::2]), axis=0)
        map_backend_dtype = getattr(self.backend.module, self.map_dtype.name)
        xip = self.backend.zeros(
            (nzbin_combs, self.n_patches, self.nbins), dtype=map_backend_dtype
        )
        xim = self.backend.zeros(
            (nzbin_combs, self.n_patches, self.nbins), dtype=map_backend_dtype
        )

        half = self.map_dtype.type(0.5)
        for k in range(nzbin_combs):
            xip_ab, xim_ab = self._normalize_xipm_pairs(
                xip_num[0, k], xim_num[0, k], sumofweights_dev[0, k]
            )
            if auto_comb[k]:
                xip[k], xim[k] = xip_ab, xim_ab
            else:
                xip_ba, xim_ba = self._normalize_xipm_pairs(
                    xip_num[1, k], xim_num[1, k], sumofweights_dev[1, k]
                )
                xip[k] = half * (xip_ab + xip_ba)
                xim[k] = half * (xim_ab + xim_ba)

        return xip, xim

    def vectorized_shear_shear(
        self,
        shear_maps: np.ndarray,
        w: np.ndarray,
        sumofweights: Optional[np.ndarray] = None,
        flip_g1: bool = False,
        flip_g2: bool = False,
        return_device: bool = True,
        _weights_fingerprint_source: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute tomographic shear 2PCFs xi+/xi-.

        If ``sumofweights`` is provided explicitly, weight fingerprint/cache
        checks are bypassed.
        """
        if self.inds_dev is None:
            self.prepare()

        nzbins = shear_maps.shape[0]
        nzbin_combs = int(binom(nzbins + 1, 2))

        shear_maps_arr = self._coerce_map_input_array(shear_maps)
        w_arr = self._coerce_map_input_array(w)
        shear_maps_dev = self._to_backend_array(shear_maps_arr, dtype=self.map_dtype)
        w_dev = self._to_backend_array(w_arr, dtype=self.map_dtype)

        if sumofweights is None and self.backend.name == "numpy":
            # CPU path: the vectorized kernel accumulates the weight sums
            # itself, so neither fingerprinting nor a separate reduction
            # pass is needed.
            sumofweights_dev = None
        elif sumofweights is None:
            w_fingerprint = self._fingerprint_weights(
                w_arr if _weights_fingerprint_source is None else _weights_fingerprint_source
            )
            cache = self.compute_context._tomo_sumofweights_cache
            cache_is_valid = (
                cache is not None
                and self.compute_context._tomo_sumofweights_cache_w_fingerprint
                == w_fingerprint
                and self.compute_context._tomo_sumofweights_cache_prepare_version
                == self.compute_context.prepare_version
                and len(cache.shape) >= 2
                and cache.shape[0] == 2
                and cache.shape[1] == nzbin_combs
            )
            if cache_is_valid:
                sumofweights_dev = cache
            else:
                sumofweights_dev = self._compute_tomo_sumofweights(
                    w_dev, nzbins, nzbin_combs
                )
                self.compute_context._tomo_sumofweights_cache = sumofweights_dev
                self.compute_context._tomo_sumofweights_cache_w_fingerprint = (
                    w_fingerprint
                )
                self.compute_context._tomo_sumofweights_cache_prepare_version = (
                    self.compute_context.prepare_version
                )
        else:
            sumofweights_arr = (
                sumofweights
                if self._is_backend_native_array(sumofweights)
                else np.asarray(sumofweights, dtype=self.map_dtype)
            )
            if sumofweights_arr.ndim < 2:
                raise ValueError(
                    "sumofweights must have at least two dimensions with "
                    "shape (2, nzbin_combs, ...)"
                )
            if sumofweights_arr.shape[0] != 2 or sumofweights_arr.shape[1] != nzbin_combs:
                raise ValueError(
                    f"sumofweights must have first dimensions (2, {nzbin_combs}); "
                    f"got {sumofweights_arr.shape}"
                )
            sumofweights_dev = self._to_backend_array(sumofweights_arr, dtype=self.map_dtype)

        g1_fac, g2_fac = 1, 1
        if flip_g1:
            g1_fac = -1
        if flip_g2:
            g2_fac = -1

        vectorized_xipm = self._xipm_tomo_vectorized(
            shear_maps_dev,
            w_dev,
            sumofweights_dev,
            nzbins,
            nzbin_combs,
            g1_fac,
            g2_fac,
        )
        if vectorized_xipm is None:
            raise RuntimeError(
                "Tomographic fused-reduction kernel unavailable for this backend."
            )

        xi_p, xi_m = vectorized_xipm
        if return_device and self.backend.name == "cupy":
            return self.backend.module.real(xi_p), self.backend.module.real(xi_m)
        return np.real(self.backend.to_numpy(xi_p)), np.real(self.backend.to_numpy(xi_m))

    def _compute_tomo_aperture_shear(
        self,
        shear_maps: np.ndarray,
        w: np.ndarray,
        aperture_filter: Optional[Callable[..., Any]] = None,
        flip_g1: bool = False,
        flip_g2: bool = False,
        return_device: bool = True,
    ) -> np.ndarray:
        shear_maps_arr = self._coerce_map_input_array(shear_maps)
        w_arr = self._coerce_map_input_array(w)
        nzbins = shear_maps_arr.shape[0]

        g1_fac, g2_fac = 1, 1
        if flip_g1:
            g1_fac = -1
        if flip_g2:
            g2_fac = -1

        self._ensure_aperture_pairs(aperture_filter=aperture_filter)

        keep_on_device = return_device and self.backend.name == "cupy"
        if keep_on_device:
            module = self.backend.module
            map_backend_dtype = getattr(module, self.map_dtype.name)
            M_a = module.zeros([nzbins, self.n_patches], dtype=map_backend_dtype)
        else:
            M_a = np.zeros([nzbins, self.n_patches], dtype=self.map_dtype)
        for i in range(nzbins):
            if g1_fac == 1 and g2_fac == 1:
                # No sign flip: pass views instead of full-map copies.
                g1_i = shear_maps_arr[i, 0]
                g2_i = shear_maps_arr[i, 1]
            else:
                g1_i = g1_fac * shear_maps_arr[i, 0]
                g2_i = g2_fac * shear_maps_arr[i, 1]
            M_a[i] = self.get_aperture_shear(
                g1_i,
                g2_i,
                w_arr[i],
                aperture_filter=None,
                return_device=keep_on_device,
            )
        return M_a

    def _compute_tomo_aperture_density(
        self,
        density_maps: np.ndarray,
        w: np.ndarray,
        aperture_filter: Optional[Callable[..., Any]] = None,
        return_device: bool = True,
    ) -> np.ndarray:
        density_arr = self._coerce_map_input_array(density_maps)
        w_arr = self._coerce_map_input_array(w)

        self._ensure_aperture_pairs(aperture_filter=aperture_filter)

        keep_on_device = return_device and self.backend.name == "cupy"
        if keep_on_device:
            module = self.backend.module
            map_backend_dtype = getattr(module, self.map_dtype.name)
            M_g = module.zeros((density_arr.shape[0], self.n_patches), dtype=map_backend_dtype)
        else:
            M_g = np.zeros((density_arr.shape[0], self.n_patches), dtype=self.map_dtype)
        for i in range(density_arr.shape[0]):
            M_g[i] = self.get_aperture_density(
                density_arr[i],
                w_arr[i],
                aperture_filter=None,
                return_device=keep_on_device,
            )
        return M_g

    def get_full_tomo_shear(
        self,
        shear_maps: np.ndarray,
        w: np.ndarray,
        sumofweights: Optional[np.ndarray] = None,
        aperture_filter: Optional[Callable[..., Any]] = None,
        flip_g1: bool = False,
        flip_g2: bool = False,
        return_device: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute full shear tomography outputs: M_a, xi_p, xi_m."""
        shear_maps_arr = self._coerce_map_input_array(shear_maps)
        w_arr = self._coerce_map_input_array(w)
        # Upload once and share the device arrays between the aperture pass
        # and the 2PCF pass (each used to re-upload the same host arrays).
        shear_dev = self._to_backend_array(shear_maps_arr, dtype=self.map_dtype)
        w_dev = self._to_backend_array(w_arr, dtype=self.map_dtype)
        M_a = self._compute_tomo_aperture_shear(
            shear_dev,
            w_dev,
            aperture_filter=aperture_filter,
            flip_g1=flip_g1,
            flip_g2=flip_g2,
            return_device=return_device,
        )
        xi_p, xi_m = self.vectorized_shear_shear(
            shear_dev,
            w_dev,
            sumofweights=sumofweights,
            flip_g1=flip_g1,
            flip_g2=flip_g2,
            return_device=return_device,
            _weights_fingerprint_source=w_arr,
        )
        return M_a, xi_p, xi_m

    def get_full_tomo_density(
        self,
        density_maps: np.ndarray,
        w: np.ndarray,
        sumofweights: Optional[np.ndarray] = None,
        gc_auto_correlations_only: bool = False,
        aperture_filter: Optional[Callable[..., Any]] = None,
        return_device: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute full density tomography outputs: M_g, xi_g."""
        density_arr = self._coerce_map_input_array(density_maps)
        w_arr = self._coerce_map_input_array(w)
        # Upload once and share the device arrays between both passes.
        density_dev = self._to_backend_array(density_arr, dtype=self.map_dtype)
        w_dev = self._to_backend_array(w_arr, dtype=self.map_dtype)
        M_g = self._compute_tomo_aperture_density(
            density_dev,
            w_dev,
            aperture_filter=aperture_filter,
            return_device=return_device,
        )
        xi_g = self.vectorized_density_density(
            density_dev,
            w_dev,
            sumofweights=sumofweights,
            gc_auto_correlations_only=gc_auto_correlations_only,
            return_device=return_device,
            _weights_fingerprint_source=w_arr,
        )
        return M_g, xi_g

    def get_full_tomo_ggl(
        self,
        density_maps: np.ndarray,
        shear_maps: np.ndarray,
        density_weights: np.ndarray,
        shear_weights: np.ndarray,
        sumofweights: Optional[np.ndarray] = None,
        ggl_bin_combinations: Optional[Sequence[Tuple[int, int]]] = None,
        aperture_filter: Optional[Callable[..., Any]] = None,
        return_N_ap: bool = False,
        return_M_ap: bool = False,
        flip_g1: bool = False,
        flip_g2: bool = False,
        return_device: bool = True,
    ) -> Union[
        np.ndarray,
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        """Compute full GGL tomography output.

        The ``flip_g1``/``flip_g2`` flags follow TreeCorr's sign convention,
        i.e. they multiply source shear components by ``-1`` before evaluating
        tangential shear.

        Returns:
            - ``xi_t`` by default.
            - ``(xi_t, M_g)`` if ``return_N_ap=True``.
            - ``(xi_t, M_a)`` if ``return_M_ap=True``.
            - ``(xi_t, M_g, M_a)`` if both flags are ``True``.
        """
        density_arr = self._coerce_map_input_array(density_maps)
        shear_arr = self._coerce_map_input_array(shear_maps)
        density_w_arr = self._coerce_map_input_array(density_weights)
        shear_w_arr = self._coerce_map_input_array(shear_weights)
        # Upload once and share the device arrays between all passes.
        density_dev = self._to_backend_array(density_arr, dtype=self.map_dtype)
        shear_dev = self._to_backend_array(shear_arr, dtype=self.map_dtype)
        density_w_dev = self._to_backend_array(density_w_arr, dtype=self.map_dtype)
        shear_w_dev = self._to_backend_array(shear_w_arr, dtype=self.map_dtype)

        xi_t = self.vectorized_density_shear(
            density_dev,
            shear_dev,
            density_w_dev,
            shear_w_dev,
            sumofweights=sumofweights,
            ggl_bin_combinations=ggl_bin_combinations,
            flip_g1=flip_g1,
            flip_g2=flip_g2,
            return_device=return_device,
            _weights_fingerprint_sources=(density_w_arr, shear_w_arr),
        )

        if not return_N_ap and not return_M_ap:
            return xi_t

        outputs: List[Any] = [xi_t]
        if return_N_ap:
            outputs.append(
                self._compute_tomo_aperture_density(
                    density_dev,
                    density_w_dev,
                    aperture_filter=aperture_filter,
                    return_device=return_device,
                )
            )
        if return_M_ap:
            outputs.append(
                self._compute_tomo_aperture_shear(
                    shear_dev,
                    shear_w_dev,
                    aperture_filter=aperture_filter,
                    flip_g1=flip_g1,
                    flip_g2=flip_g2,
                    return_device=return_device,
                )
            )

        return tuple(outputs)

    def _density_density_tomo_vectorized(
        self,
        density_maps: np.ndarray,
        weights: np.ndarray,
        sumofweights: Optional[np.ndarray],
        nzbins: int,
        gc_auto_correlations_only: bool = False,
        _weights_fingerprint_source: Optional[np.ndarray] = None,
    ) -> Any:
        if self.inds_dev is None:
            self.prepare()

        tomo_kernel = getattr(self.backend, "kernel_density_density_tomo_vectorized", None)
        if tomo_kernel is None:
            raise RuntimeError(
                "Backend does not provide a vectorized density-density tomography kernel."
            )

        module = self.backend.module
        map_backend_dtype = getattr(module, self.map_dtype.name)
        half = self.map_dtype.type(0.5)
        nbins_total = self.n_patches * self.nbins

        density_dev = self._to_backend_array(density_maps, dtype=self.map_dtype)
        w_dev = self._to_backend_array(weights, dtype=self.map_dtype)
        density_soa = module.ascontiguousarray(module.transpose(density_dev, (1, 0)))
        w_soa = module.ascontiguousarray(module.transpose(w_dev, (1, 0)))

        comb_i, comb_j, auto_comb, nzbin_combs = (
            self._get_selected_tomo_density_combination_indices(
                nzbins,
                gc_auto_correlations_only=gc_auto_correlations_only,
            )
        )

        if sumofweights is not None:
            sumofweights_dev = self._normalize_tomo_sumofweights_directional(
                sumofweights, nzbin_combs
            )
        elif self.backend.name == "numpy":
            # CPU kernel accumulates the denominators itself (below).
            sumofweights_dev = None
        else:
            fingerprint_src = (
                weights if _weights_fingerprint_source is None
                else _weights_fingerprint_source
            )
            cache_key = (
                self._fingerprint_weights(fingerprint_src),
                bool(gc_auto_correlations_only),
                int(nzbins),
            )

            def _compute_dd_sums() -> Any:
                if gc_auto_correlations_only:
                    sums = self.backend.zeros(
                        (2, nzbin_combs, nbins_total), dtype=map_backend_dtype
                    )
                    comb_i_np = np.asarray(self.backend.to_numpy(comb_i), dtype=np.int64)
                    comb_j_np = np.asarray(self.backend.to_numpy(comb_j), dtype=np.int64)
                    for k in range(nzbin_combs):
                        i = int(comb_i_np[k])
                        j = int(comb_j_np[k])
                        sum_ij = self._reduce_pairs(
                            w_dev[i][self.inds_dev[0]] * w_dev[j][self.inds_dev[1]]
                        )
                        sums[0, k] = sum_ij
                        if i == j:
                            sums[1, k] = sum_ij
                        else:
                            sums[1, k] = self._reduce_pairs(
                                w_dev[j][self.inds_dev[0]] * w_dev[i][self.inds_dev[1]]
                            )
                    return sums
                return self._compute_tomo_sumofweights(w_dev, nzbins, nzbin_combs)

            sumofweights_dev = self._sumofweights_cache_get_or_compute(
                "dd_sumofweights_cache", cache_key, _compute_dd_sums
            )

        inds_i = module.ascontiguousarray(self.inds_dev[0].astype(self._index_device_dtype, copy=False))
        inds_j = module.ascontiguousarray(self.inds_dev[1].astype(self._index_device_dtype, copy=False))
        bin_offsets = module.ascontiguousarray(
            self.tot_bins_reduceat_dev.astype(module.int64, copy=False)
        )

        out = self.backend.zeros(
            (nzbin_combs, self.n_patches, self.nbins), dtype=map_backend_dtype
        )

        if self.backend.name == "numpy":
            out_num = np.empty((nzbin_combs, nbins_total), dtype=map_backend_dtype)
            out_den = np.empty((nzbin_combs, nbins_total), dtype=map_backend_dtype)
            tomo_kernel(
                density_soa,
                w_soa,
                np.ascontiguousarray(inds_i),
                np.ascontiguousarray(inds_j),
                np.asarray(bin_offsets, dtype=np.int64),
                np.ascontiguousarray(comb_i),
                np.ascontiguousarray(comb_j),
                out_num,
                out_den,
            )

            for k in range(nzbin_combs):
                if sumofweights_dev is None:
                    # In-kernel denominator: already the symmetrised
                    # (i==j ? w_ab : (w_ab + w_ba)/2) sum.
                    sum_k = out_den[k]
                elif auto_comb[k]:
                    sum_k = sumofweights_dev[0, k]
                else:
                    sum_k = half * (sumofweights_dev[0, k] + sumofweights_dev[1, k])
                out[k] = self._normalize_scalar_pairs(out_num[k], sum_k)
            return out

        out_num = self.backend.zeros((2 * nzbin_combs, nbins_total), dtype=map_backend_dtype)
        launched = tomo_kernel(
            density_soa,
            w_soa,
            inds_i,
            inds_j,
            bin_offsets,
            comb_i,
            comb_j,
            out_num,
        )
        if not launched:
            raise RuntimeError(
                "Backend declined vectorized density-density tomography kernel launch."
            )

        num_ab = out_num[0::2]
        num_ba = out_num[1::2]
        for k in range(nzbin_combs):
            w_ab = self._normalize_scalar_pairs(num_ab[k], sumofweights_dev[0, k])
            if auto_comb[k]:
                out[k] = w_ab
            else:
                w_ba = self._normalize_scalar_pairs(num_ba[k], sumofweights_dev[1, k])
                out[k] = half * (w_ab + w_ba)

        return out

    def _density_shear_tomo_vectorized(
        self,
        density_maps: np.ndarray,
        shear_maps: np.ndarray,
        density_w: np.ndarray,
        shear_w: np.ndarray,
        sumofweights: Optional[np.ndarray],
        nlens_bins: int,
        nsource_bins: int,
        ggl_bin_combinations: Optional[Sequence[Tuple[int, int]]] = None,
        _weights_fingerprint_sources: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Any:
        if self.inds_dev is None:
            self.prepare()

        tomo_kernel = getattr(self.backend, "kernel_density_shear_tomo_vectorized", None)
        if tomo_kernel is None:
            raise RuntimeError(
                "Backend does not provide a vectorized density-shear tomography kernel."
            )

        module = self.backend.module
        map_backend_dtype = getattr(module, self.map_dtype.name)
        nbins_total = self.n_patches * self.nbins

        density_dev = self._to_backend_array(density_maps, dtype=self.map_dtype)
        shear_dev = self._to_backend_array(shear_maps, dtype=self.map_dtype)
        density_w_dev = self._to_backend_array(density_w, dtype=self.map_dtype)
        shear_w_dev = self._to_backend_array(shear_w, dtype=self.map_dtype)

        density_soa = module.ascontiguousarray(module.transpose(density_dev, (1, 0)))
        shear_soa = module.ascontiguousarray(module.transpose(shear_dev, (2, 0, 1)))
        density_w_soa = module.ascontiguousarray(module.transpose(density_w_dev, (1, 0)))
        shear_w_soa = module.ascontiguousarray(module.transpose(shear_w_dev, (1, 0)))

        comb_i_base, comb_j_base, nzbin_combs = (
            self._get_selected_tomo_cross_combination_indices(
                nlens_bins,
                nsource_bins,
                ggl_bin_combinations=ggl_bin_combinations,
            )
        )
        comb_i = module.ascontiguousarray(comb_i_base)
        comb_j = module.ascontiguousarray(comb_j_base)

        inds_i = self.compute_context.inds_i_dev
        inds_j = self.compute_context.inds_j_dev
        if inds_i is None or inds_j is None:
            inds_i = module.ascontiguousarray(self.inds_dev[0].astype(self._index_device_dtype, copy=False))
            inds_j = module.ascontiguousarray(self.inds_dev[1].astype(self._index_device_dtype, copy=False))
            self.compute_context.inds_i_dev = inds_i
            self.compute_context.inds_j_dev = inds_j
        rot_i = module.ascontiguousarray(self.exp2phi_dev[0])
        rot_j = module.ascontiguousarray(self.exp2phi_dev[1])
        bin_offsets = module.ascontiguousarray(
            self.tot_bins_reduceat_dev.astype(module.int64, copy=False)
        )

        if self.backend.name == "numpy":
            out_num = np.empty((nzbin_combs, nbins_total), dtype=map_backend_dtype)
            out_den = np.empty((nzbin_combs, nbins_total), dtype=map_backend_dtype)
            tomo_kernel(
                density_soa,
                shear_soa,
                density_w_soa,
                shear_w_soa,
                np.ascontiguousarray(inds_i),
                np.ascontiguousarray(inds_j),
                np.ascontiguousarray(rot_i),
                np.ascontiguousarray(rot_j),
                np.asarray(bin_offsets, dtype=np.int64),
                np.ascontiguousarray(comb_i),
                np.ascontiguousarray(comb_j),
                out_num,
                out_den,
            )
        else:
            out_num = self.backend.zeros((nzbin_combs, nbins_total), dtype=map_backend_dtype)
            out_den = None
            launched = tomo_kernel(
                density_soa,
                shear_soa,
                density_w_soa,
                shear_w_soa,
                inds_i,
                inds_j,
                rot_i,
                rot_j,
                bin_offsets,
                comb_i,
                comb_j,
                out_num,
            )
            if not launched:
                raise RuntimeError(
                    "Backend declined vectorized density-shear tomography kernel launch."
                )

        num_ab = out_num

        if sumofweights is None:
            if self.backend.name == "numpy":
                # In-kernel denominator: already the A→B + B→A weight sum.
                sum_total = out_den
            else:
                fingerprint_src_d = (
                    density_w if _weights_fingerprint_sources is None
                    else _weights_fingerprint_sources[0]
                )
                fingerprint_src_s = (
                    shear_w if _weights_fingerprint_sources is None
                    else _weights_fingerprint_sources[1]
                )
                if ggl_bin_combinations is None:
                    comb_key: Any = ("full", int(nlens_bins), int(nsource_bins))
                else:
                    comb_key = tuple(
                        (int(p[0]), int(p[1])) for p in ggl_bin_combinations
                    )
                cache_key = (
                    self._fingerprint_weights(fingerprint_src_d),
                    self._fingerprint_weights(fingerprint_src_s),
                    comb_key,
                )

                def _compute_ds_sums() -> Any:
                    comb_i_np = np.asarray(
                        self.backend.to_numpy(comb_i_base), dtype=np.int64
                    )
                    comb_j_np = np.asarray(
                        self.backend.to_numpy(comb_j_base), dtype=np.int64
                    )
                    sum_ab = self.backend.zeros(
                        (nzbin_combs, nbins_total), dtype=map_backend_dtype
                    )
                    sum_ba = self.backend.zeros(
                        (nzbin_combs, nbins_total), dtype=map_backend_dtype
                    )
                    for k in range(nzbin_combs):
                        i = int(comb_i_np[k])
                        j = int(comb_j_np[k])
                        sum_ab[k] = self._reduce_pairs(
                            density_w_dev[i][self.inds_dev[0]]
                            * shear_w_dev[j][self.inds_dev[1]]
                        )
                        sum_ba[k] = self._reduce_pairs(
                            density_w_dev[i][self.inds_dev[1]]
                            * shear_w_dev[j][self.inds_dev[0]]
                        )
                    return sum_ab + sum_ba

                sum_total = self._sumofweights_cache_get_or_compute(
                    "ds_sumofweights_cache", cache_key, _compute_ds_sums
                )
        else:
            if (
                self._is_backend_native_array(sumofweights)
                and sumofweights.ndim >= 2
                and sumofweights.shape[0] == 2
            ):
                sum_dir = self._normalize_tomo_sumofweights_directional(
                    sumofweights, nzbin_combs
                )
                sum_total = sum_dir[0] + sum_dir[1]
            elif self._is_backend_native_array(sumofweights):
                sum_total = self._normalize_tomo_sumofweights_per_comb(
                    sumofweights, nzbin_combs
                )
            else:
                sum_np = np.asarray(
                    self.backend.to_numpy(sumofweights), dtype=self.map_dtype
                )
                if sum_np.ndim >= 2 and sum_np.shape[0] == 2:
                    sum_dir = self._normalize_tomo_sumofweights_directional(sum_np, nzbin_combs)
                    sum_total = sum_dir[0] + sum_dir[1]
                else:
                    sum_total = self._normalize_tomo_sumofweights_per_comb(sum_np, nzbin_combs)

        out = self.backend.zeros(
            (nzbin_combs, self.n_patches, self.nbins), dtype=map_backend_dtype
        )
        for k in range(nzbin_combs):
            out[k] = self._normalize_scalar_pairs(num_ab[k], sum_total[k])
        return out

    def vectorized_density_density(
        self,
        density_maps: np.ndarray,
        w: np.ndarray,
        sumofweights: Optional[np.ndarray] = None,
        gc_auto_correlations_only: bool = False,
        return_device: bool = True,
        _weights_fingerprint_source: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute tomographic galaxy clustering w(theta).

        By default, computes upper-triangular tomographic pairs including cross-bin
        terms. If ``gc_auto_correlations_only=True``, computes only auto-correlations
        ``(i, i)`` for each tomographic bin.
        """
        density_arr = self._coerce_map_input_array(density_maps)
        w_arr = self._coerce_map_input_array(w)
        nzbins = density_arr.shape[0]
        wtheta = self._density_density_tomo_vectorized(
            density_arr,
            w_arr,
            sumofweights,
            nzbins,
            gc_auto_correlations_only,
            _weights_fingerprint_source=_weights_fingerprint_source,
        )
        if return_device and self.backend.name == "cupy":
            return self.backend.module.real(wtheta)
        return np.real(self.backend.to_numpy(wtheta))

    def vectorized_density_shear(
        self,
        density_maps: np.ndarray,
        shear_maps: np.ndarray,
        density_weights: np.ndarray,
        shear_weights: np.ndarray,
        sumofweights: Optional[np.ndarray] = None,
        ggl_bin_combinations: Optional[Sequence[Tuple[int, int]]] = None,
        flip_g1: bool = False,
        flip_g2: bool = False,
        return_device: bool = True,
        _weights_fingerprint_sources: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        """Compute tomographic density-shear correlation with directional Lens->Source ordering.

        The first argument (`density_maps`) is always treated as the lens field and the
        second argument (`shear_maps`) as the source shear field.

        Computes all cartesian tomographic combinations `(lens_bin, source_bin)` in
        row-major order: `(0,0), (0,1), ..., (0, n_source-1), (1,0), ...`.
        If ``ggl_bin_combinations`` is provided, only those pairs are computed
        and returned in the provided order.

        ``flip_g1`` and ``flip_g2`` mirror TreeCorr's ``Catalog(..., flip_g1=...)``
        and ``flip_g2`` behavior for source shears.
        """
        density_arr = self._coerce_map_input_array(density_maps)
        shear_arr = self._coerce_map_input_array(shear_maps)
        wd_arr = self._coerce_map_input_array(density_weights)
        ws_arr = self._coerce_map_input_array(shear_weights)
        if density_arr.ndim != 2:
            raise ValueError(
                "density_maps must have shape (n_lens_bins, npix); "
                f"got {density_arr.shape}"
            )
        if shear_arr.ndim != 3 or shear_arr.shape[1] != 2:
            raise ValueError(
                "shear_maps must have shape (nzbins, 2, npix); "
                f"got {shear_arr.shape}"
            )
        if wd_arr.shape != density_arr.shape:
            raise ValueError(
                "density_weights must match density_maps shape; "
                f"got {wd_arr.shape} and {density_arr.shape}"
            )
        if ws_arr.ndim != 2:
            raise ValueError(
                "shear_weights must have shape (n_source_bins, npix); "
                f"got {ws_arr.shape}"
            )
        if ws_arr.shape[0] != shear_arr.shape[0] or ws_arr.shape[1] != shear_arr.shape[2]:
            raise ValueError(
                "shear_weights must match shear_maps tomography/pixel dimensions; "
                f"got {ws_arr.shape} and {shear_arr.shape}"
            )
        if density_arr.shape[1] != shear_arr.shape[2]:
            raise ValueError(
                "density_maps and shear_maps must have the same number of pixels; "
                f"got {density_arr.shape[1]} and {shear_arr.shape[2]}"
            )

        if flip_g1 or flip_g2:
            shear_arr = shear_arr.copy()
            if flip_g1:
                shear_arr[:, 0] *= -1
            if flip_g2:
                shear_arr[:, 1] *= -1

        nlens_bins = density_arr.shape[0]
        nsource_bins = shear_arr.shape[0]
        gammat = self._density_shear_tomo_vectorized(
            density_arr,
            shear_arr,
            wd_arr,
            ws_arr,
            sumofweights,
            nlens_bins,
            nsource_bins,
            ggl_bin_combinations=ggl_bin_combinations,
            _weights_fingerprint_sources=_weights_fingerprint_sources,
        )
        if return_device and self.backend.name == "cupy":
            return self.backend.module.real(gammat)
        return np.real(self.backend.to_numpy(gammat))

    def zeta_g_plus(self, M_g: np.ndarray, xi_p: np.ndarray) -> np.ndarray:
        """Compute zeta_g_plus with g at center and xi+ on annulus."""
        return _zeta_g_plus_helper(M_g, xi_p)

    def zeta_g_minus(self, M_g: np.ndarray, xi_m: np.ndarray) -> np.ndarray:
        """Compute zeta_g_minus with g at center and xi- on annulus."""
        return _zeta_g_minus_helper(M_g, xi_m)

    def zeta_a_plus(self, M_a: np.ndarray, xi_p: np.ndarray) -> np.ndarray:
        """Compute zeta_a_plus with a at center and xi+ on annulus."""
        return _zeta_a_plus_helper(M_a, xi_p)

    def zeta_a_minus(self, M_a: np.ndarray, xi_m: np.ndarray) -> np.ndarray:
        """Compute zeta_a_minus with a at center and xi- on annulus."""
        return _zeta_a_minus_helper(M_a, xi_m)

    def zeta_g_g(self, M_g: np.ndarray, xi_g: np.ndarray) -> np.ndarray:
        """Compute zeta_g_g with g at center and galaxy auto-correlation on annulus."""
        return _zeta_g_g_helper(M_g, xi_g)

    def zeta_a_g(self, M_a: np.ndarray, xi_g: np.ndarray) -> np.ndarray:
        """Compute zeta_a_g with a at center and galaxy auto-correlation on annulus."""
        return _zeta_a_g_helper(M_a, xi_g)

    def zeta_g_t(self, M_g: np.ndarray, xi_t: np.ndarray) -> np.ndarray:
        """Compute zeta_g_t with g at center and tangential shear on annulus."""
        return _zeta_g_t_helper(M_g, xi_t)

    def zeta_a_t(self, M_a: np.ndarray, xi_t: np.ndarray) -> np.ndarray:
        """Compute zeta_a_t with a at center and tangential shear on annulus."""
        return _zeta_a_t_helper(M_a, xi_t)

    def calculate_all_zetas(
        self,
        M_g: Optional[np.ndarray] = None,
        M_a: Optional[np.ndarray] = None,
        xi_p: Optional[np.ndarray] = None,
        xi_m: Optional[np.ndarray] = None,
        xi_g: Optional[np.ndarray] = None,
        xi_t: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Calculate all supported i3PCFs with explicit center/annulus naming."""
        return _calculate_all_zetas_helper(
            M_g=M_g,
            M_a=M_a,
            xi_p=xi_p,
            xi_m=xi_m,
            xi_g=xi_g,
            xi_t=xi_t,
        )

    def _compute_3x2pt_tomo_fused(
        self,
        shear_maps: np.ndarray,
        density_maps: np.ndarray,
        shear_weights: np.ndarray,
        density_weights: np.ndarray,
        gc_auto_correlations_only: bool = False,
        ggl_bin_combinations: Optional[Sequence[Tuple[int, int]]] = None,
        aperture_filter: Optional[Callable[..., Any]] = None,
        flip_g1: bool = False,
        flip_g2: bool = False,
        return_device: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        fused_kernel = getattr(self.backend, "kernel_3x2pt_tomo_fused", None)
        if fused_kernel is None:
            raise RuntimeError("Backend does not provide a fused 3x2pt tomography kernel.")

        shear_np = self._coerce_map_input_array(shear_maps)
        density_np = self._coerce_map_input_array(density_maps)
        shear_w_np = self._coerce_map_input_array(shear_weights)
        density_w_np = self._coerce_map_input_array(density_weights)

        if density_np.ndim != 2:
            raise ValueError(
                "density_maps must have shape (n_density_bins, npix); "
                f"got {density_np.shape}"
            )
        if shear_np.ndim != 3 or shear_np.shape[1] != 2:
            raise ValueError(
                "shear_maps must have shape (n_shear_bins, 2, npix); "
                f"got {shear_np.shape}"
            )
        if density_w_np.shape != density_np.shape:
            raise ValueError(
                "density_weights must match density_maps shape; "
                f"got {density_w_np.shape} and {density_np.shape}"
            )
        if shear_w_np.ndim != 2:
            raise ValueError(
                "shear_weights must have shape (n_shear_bins, npix); "
                f"got {shear_w_np.shape}"
            )
        if shear_w_np.shape[0] != shear_np.shape[0] or shear_w_np.shape[1] != shear_np.shape[2]:
            raise ValueError(
                "shear_weights must match shear_maps tomography/pixel dimensions; "
                f"got {shear_w_np.shape} and {shear_np.shape}"
            )
        if density_np.shape[1] != shear_np.shape[2]:
            raise ValueError(
                "density_maps and shear_maps must have the same number of pixels; "
                f"got {density_np.shape[1]} and {shear_np.shape[2]}"
            )

        if flip_g1 or flip_g2:
            shear_np = shear_np.copy()
            if flip_g1:
                shear_np[:, 0] *= -1
            if flip_g2:
                shear_np[:, 1] *= -1

        if self.inds_dev is None:
            self.prepare()
        self._ensure_aperture_pairs(aperture_filter=aperture_filter)

        module = self.backend.module
        map_backend_dtype = getattr(module, self.map_dtype.name)

        density_dev = self._to_backend_array(density_np, dtype=self.map_dtype)
        shear_dev = self._to_backend_array(shear_np, dtype=self.map_dtype)
        density_w_dev = self._to_backend_array(density_w_np, dtype=self.map_dtype)
        shear_w_dev = self._to_backend_array(shear_w_np, dtype=self.map_dtype)

        n_shear_bins = int(shear_np.shape[0])
        n_density_bins = int(density_np.shape[0])
        npix = int(density_np.shape[1])
        (
            density_soa,
            shear_soa,
            density_w_soa,
            shear_w_soa,
        ) = self._get_or_create_fused_input_buffers(
            npix,
            n_density_bins,
            n_shear_bins,
            map_backend_dtype,
        )
        self._fill_fused_input_buffers(
            density_dev,
            shear_dev,
            density_w_dev,
            shear_w_dev,
            density_soa,
            shear_soa,
            density_w_soa,
            shear_w_soa,
        )

        n_patches = int(self.n_patches)
        nbins_total = int(self.n_patches * self.nbins)

        ss_ncomb = int(binom(n_shear_bins + 1, 2))
        ss_comb_i, ss_comb_j, ss_auto = self._get_tomo_combination_indices(
            n_shear_bins,
            ss_ncomb,
        )
        dd_comb_i, dd_comb_j, dd_auto, dd_ncomb = (
            self._get_selected_tomo_density_combination_indices(
                n_density_bins,
                gc_auto_correlations_only=gc_auto_correlations_only,
            )
        )
        ds_comb_i, ds_comb_j, ds_ncomb = self._get_selected_tomo_cross_combination_indices(
            n_density_bins,
            n_shear_bins,
            ggl_bin_combinations=ggl_bin_combinations,
        )

        inds_i = module.ascontiguousarray(self.inds_dev[0].astype(self._index_device_dtype, copy=False))
        inds_j = module.ascontiguousarray(self.inds_dev[1].astype(self._index_device_dtype, copy=False))
        rot_i = module.ascontiguousarray(self.exp2phi_dev[0])
        rot_j = module.ascontiguousarray(self.exp2phi_dev[1])
        pair_offsets = module.ascontiguousarray(
            self.tot_bins_reduceat_dev.astype(module.int64, copy=False)
        )

        if self.compute_context.Q_inds_dev is None:
            self._prepare_aperture_device_buffers()
        q_inds = self.compute_context.Q_inds_dev
        q_cos = self.compute_context.Q_cos_dev
        q_sin = self.compute_context.Q_sin_dev
        q_val = self.compute_context.Q_val_dev
        q_offsets = self.compute_context.Q_offsets_dev
        q_patch_area = self.compute_context.Q_patch_area_dev

        (
            out_ma_num,
            out_ma_den,
            out_mg_num,
            out_mg_den,
            out_xip_num,
            out_xim_num,
            out_xip_den,
            out_xig_num,
            out_xig_den,
            out_xit_num,
            out_xit_den,
        ) = self._get_or_create_fused_output_buffers(
            n_shear_bins=n_shear_bins,
            n_density_bins=n_density_bins,
            n_patches=n_patches,
            nbins_total=nbins_total,
            ss_ncomb=ss_ncomb,
            dd_ncomb=dd_ncomb,
            ds_ncomb=ds_ncomb,
            map_backend_dtype=map_backend_dtype,
        )

        if self.backend.name == "numpy":
            fused_kernel(
                np.ascontiguousarray(density_soa),
                np.ascontiguousarray(shear_soa),
                np.ascontiguousarray(density_w_soa),
                np.ascontiguousarray(shear_w_soa),
                np.ascontiguousarray(inds_i),
                np.ascontiguousarray(inds_j),
                np.ascontiguousarray(rot_i),
                np.ascontiguousarray(rot_j),
                np.ascontiguousarray(pair_offsets),
                np.ascontiguousarray(q_inds),
                np.ascontiguousarray(q_cos),
                np.ascontiguousarray(q_sin),
                np.ascontiguousarray(q_val),
                np.ascontiguousarray(q_offsets),
                np.ascontiguousarray(q_patch_area),
                np.ascontiguousarray(ss_comb_i),
                np.ascontiguousarray(ss_comb_j),
                np.ascontiguousarray(dd_comb_i),
                np.ascontiguousarray(dd_comb_j),
                np.ascontiguousarray(ds_comb_i),
                np.ascontiguousarray(ds_comb_j),
                out_ma_num,
                out_ma_den,
                out_mg_num,
                out_mg_den,
                out_xip_num,
                out_xim_num,
                out_xip_den,
                out_xig_num,
                out_xig_den,
                out_xit_num,
                out_xit_den,
            )
        else:
            launched = fused_kernel(
                density_soa,
                shear_soa,
                density_w_soa,
                shear_w_soa,
                inds_i,
                inds_j,
                rot_i,
                rot_j,
                pair_offsets,
                q_inds,
                q_cos,
                q_sin,
                q_val,
                q_offsets,
                q_patch_area,
                module.ascontiguousarray(ss_comb_i),
                module.ascontiguousarray(ss_comb_j),
                module.ascontiguousarray(dd_comb_i),
                module.ascontiguousarray(dd_comb_j),
                module.ascontiguousarray(ds_comb_i),
                module.ascontiguousarray(ds_comb_j),
                out_ma_num,
                out_ma_den,
                out_mg_num,
                out_mg_den,
                out_xip_num,
                out_xim_num,
                out_xip_den,
                out_xig_num,
                out_xig_den,
                out_xit_num,
                out_xit_den,
            )
            if not launched:
                raise RuntimeError("Backend declined fused 3x2pt tomography kernel launch.")

        def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
            out = np.zeros_like(num, dtype=self.map_dtype)
            valid = den != 0
            out[valid] = num[valid] / den[valid]
            return out

        def _safe_div_into_device(num: Any, den: Any, out: Any) -> Any:
            mask = den != 0
            safe_den = module.where(mask, den, self.map_dtype.type(1.0))
            module.divide(num, safe_den, out=out)
            out *= mask.astype(out.dtype, copy=False)
            return out

        if return_device and self.backend.name == "cupy":
            (
                M_a_dev,
                M_g_dev,
                xip_dev,
                xim_dev,
                xi_g_dev,
                xi_t_dev,
                tmp_a,
                tmp_b,
            ) = self._get_or_create_fused_post_buffers(
                n_shear_bins=n_shear_bins,
                n_density_bins=n_density_bins,
                n_patches=n_patches,
                ss_ncomb=ss_ncomb,
                dd_ncomb=dd_ncomb,
                ds_ncomb=ds_ncomb,
                map_backend_dtype=map_backend_dtype,
            )

            _safe_div_into_device(out_ma_num, out_ma_den, M_a_dev)
            _safe_div_into_device(out_mg_num, out_mg_den, M_g_dev)

            half = self.map_dtype.type(0.5)

            for k in range(ss_ncomb):
                ab_idx = 2 * k
                ba_idx = ab_idx + 1

                _safe_div_into_device(
                    out_xip_num[ab_idx].reshape((n_patches, self.nbins)),
                    out_xip_den[ab_idx].reshape((n_patches, self.nbins)),
                    xip_dev[k],
                )
                _safe_div_into_device(
                    out_xim_num[ab_idx].reshape((n_patches, self.nbins)),
                    out_xip_den[ab_idx].reshape((n_patches, self.nbins)),
                    xim_dev[k],
                )

                if not ss_auto[k]:
                    _safe_div_into_device(
                        out_xip_num[ba_idx].reshape((n_patches, self.nbins)),
                        out_xip_den[ba_idx].reshape((n_patches, self.nbins)),
                        tmp_a,
                    )
                    xip_dev[k] += tmp_a
                    xip_dev[k] *= half

                    _safe_div_into_device(
                        out_xim_num[ba_idx].reshape((n_patches, self.nbins)),
                        out_xip_den[ba_idx].reshape((n_patches, self.nbins)),
                        tmp_b,
                    )
                    xim_dev[k] += tmp_b
                    xim_dev[k] *= half

            for k in range(dd_ncomb):
                ab_idx = 2 * k
                ba_idx = ab_idx + 1

                _safe_div_into_device(
                    out_xig_num[ab_idx].reshape((n_patches, self.nbins)),
                    out_xig_den[ab_idx].reshape((n_patches, self.nbins)),
                    xi_g_dev[k],
                )

                if not dd_auto[k]:
                    _safe_div_into_device(
                        out_xig_num[ba_idx].reshape((n_patches, self.nbins)),
                        out_xig_den[ba_idx].reshape((n_patches, self.nbins)),
                        tmp_a,
                    )
                    xi_g_dev[k] += tmp_a
                    xi_g_dev[k] *= half

            for k in range(ds_ncomb):
                _safe_div_into_device(
                    out_xit_num[k].reshape((n_patches, self.nbins)),
                    out_xit_den[k].reshape((n_patches, self.nbins)),
                    xi_t_dev[k],
                )

            return M_a_dev, M_g_dev, xip_dev, xim_dev, xi_g_dev, xi_t_dev

        ma_num_np = np.asarray(self.backend.to_numpy(out_ma_num), dtype=self.map_dtype)
        ma_den_np = np.asarray(self.backend.to_numpy(out_ma_den), dtype=self.map_dtype)
        mg_num_np = np.asarray(self.backend.to_numpy(out_mg_num), dtype=self.map_dtype)
        mg_den_np = np.asarray(self.backend.to_numpy(out_mg_den), dtype=self.map_dtype)
        M_a = _safe_div(ma_num_np, ma_den_np)
        M_g = _safe_div(mg_num_np, mg_den_np)

        xip_num_np = np.asarray(self.backend.to_numpy(out_xip_num), dtype=self.map_dtype)
        xim_num_np = np.asarray(self.backend.to_numpy(out_xim_num), dtype=self.map_dtype)
        xip_den_np = np.asarray(self.backend.to_numpy(out_xip_den), dtype=self.map_dtype)
        xip = np.zeros((ss_ncomb, n_patches, self.nbins), dtype=self.map_dtype)
        xim = np.zeros((ss_ncomb, n_patches, self.nbins), dtype=self.map_dtype)

        half = self.map_dtype.type(0.5)
        for k in range(ss_ncomb):
            ab_idx = 2 * k
            ba_idx = ab_idx + 1
            xip_ab = _safe_div(xip_num_np[ab_idx], xip_den_np[ab_idx]).reshape((n_patches, self.nbins))
            xim_ab = _safe_div(xim_num_np[ab_idx], xip_den_np[ab_idx]).reshape((n_patches, self.nbins))
            if ss_auto[k]:
                xip[k] = xip_ab
                xim[k] = xim_ab
            else:
                xip_ba = _safe_div(xip_num_np[ba_idx], xip_den_np[ba_idx]).reshape((n_patches, self.nbins))
                xim_ba = _safe_div(xim_num_np[ba_idx], xip_den_np[ba_idx]).reshape((n_patches, self.nbins))
                xip[k] = half * (xip_ab + xip_ba)
                xim[k] = half * (xim_ab + xim_ba)

        xig_num_np = np.asarray(self.backend.to_numpy(out_xig_num), dtype=self.map_dtype)
        xig_den_np = np.asarray(self.backend.to_numpy(out_xig_den), dtype=self.map_dtype)
        xi_g = np.zeros((dd_ncomb, n_patches, self.nbins), dtype=self.map_dtype)
        for k in range(dd_ncomb):
            ab_idx = 2 * k
            ba_idx = ab_idx + 1
            xig_ab = _safe_div(xig_num_np[ab_idx], xig_den_np[ab_idx]).reshape((n_patches, self.nbins))
            if dd_auto[k]:
                xi_g[k] = xig_ab
            else:
                if self.backend.name == "numpy":
                    xi_g[k] = _safe_div(
                        xig_num_np[ab_idx] + xig_num_np[ba_idx],
                        xig_den_np[ab_idx] + xig_den_np[ba_idx],
                    ).reshape((n_patches, self.nbins))
                else:
                    xig_ba = _safe_div(xig_num_np[ba_idx], xig_den_np[ba_idx]).reshape((n_patches, self.nbins))
                    xi_g[k] = half * (xig_ab + xig_ba)

        xit_num_np = np.asarray(self.backend.to_numpy(out_xit_num), dtype=self.map_dtype)
        xit_den_np = np.asarray(self.backend.to_numpy(out_xit_den), dtype=self.map_dtype)
        xi_t = np.zeros((ds_ncomb, n_patches, self.nbins), dtype=self.map_dtype)
        for k in range(ds_ncomb):
            xi_t[k] = _safe_div(xit_num_np[k], xit_den_np[k]).reshape((n_patches, self.nbins))

        return M_a, M_g, xip, xim, xi_g, xi_t

    def get_3x2pt_tomo(
        self,
        shear_maps: Optional[np.ndarray] = None,
        density_maps: Optional[np.ndarray] = None,
        weights: Optional[Any] = None,
        gc_auto_correlations_only: bool = False,
        ggl_bin_combinations: Optional[Sequence[Tuple[int, int]]] = None,
        aperture_filter: Optional[Callable[..., Any]] = None,
        flip_g1: bool = False,
        flip_g2: bool = False,
        return_device: bool = True,
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        if shear_maps is None or density_maps is None:
            raise ValueError(
                "Both shear_maps and density_maps must be provided for get_3x2pt_tomo."
            )

        shear_arr = self._coerce_map_input_array(shear_maps)
        density_arr = self._coerce_map_input_array(density_maps)

        map_backend_dtype = getattr(self.backend.module, self.map_dtype.name)

        def _ones(shape: Tuple[int, ...]) -> Any:
            ctx = self.compute_context
            if getattr(ctx, "unit_weights_cache", None) is None:
                ctx.unit_weights_cache = {}
            key = (tuple(int(s) for s in shape), self.map_dtype.str)
            cached = ctx.unit_weights_cache.get(key)
            if cached is not None:
                return cached
            if self.backend.name == "cupy":
                arr = self.backend.module.ones(shape, dtype=map_backend_dtype)
            else:
                arr = np.ones(shape, dtype=self.map_dtype)
            ctx.unit_weights_cache.clear()
            ctx.unit_weights_cache[key] = arr
            return arr

        shear_w = None
        density_w = None
        if weights is None:
            shear_w = _ones((shear_arr.shape[0], shear_arr.shape[2]))
            density_w = _ones((density_arr.shape[0], density_arr.shape[1]))
        elif isinstance(weights, dict):
            shear_w = weights.get("shear")
            density_w = weights.get("density")
        elif isinstance(weights, (tuple, list)) and len(weights) == 2:
            shear_w, density_w = weights
        else:
            weight_arr = self._coerce_map_input_array(weights)
            if (
                weight_arr.ndim == 2
                and weight_arr.shape[0] == shear_arr.shape[0]
                and weight_arr.shape[0] == density_arr.shape[0]
            ):
                shear_w = weight_arr
                density_w = weight_arr
            else:
                raise ValueError(
                    "When both shear_maps and density_maps are provided, weights must be a dict or (shear_weights, density_weights)."
                )

        if shear_w is None:
            shear_w = _ones((shear_arr.shape[0], shear_arr.shape[2]))
        if density_w is None:
            density_w = _ones((density_arr.shape[0], density_arr.shape[1]))

        return self._compute_3x2pt_tomo_fused(
            shear_maps=shear_arr,
            density_maps=density_arr,
            shear_weights=self._coerce_map_input_array(shear_w),
            density_weights=self._coerce_map_input_array(density_w),
            gc_auto_correlations_only=gc_auto_correlations_only,
            ggl_bin_combinations=ggl_bin_combinations,
            aperture_filter=aperture_filter,
            flip_g1=flip_g1,
            flip_g2=flip_g2,
            return_device=return_device,
        )

