import hashlib
import logging
import weakref
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import healpy as hp
import numpy as np
from numba import njit, prange
from scipy.special import binom
from tqdm import trange

from .backend import get_backend, safe_divide
from .compute_context import ComputeContext
from .io_handler import PairIOHandler
from .pair_geometry import PairGeometry
from .correlation_helpers import (
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
from .packing import block_edges, decode_angles, pack_patch, unpack_rows
from .pair_finder import PairFinder
from .treecode import (
    TreecodeGeometry,
    assign_levels,
    effective_resolution_factor,
    is_power_of_two,
    level_groups,
    validate_resolution_factor,
)
from .utils import live_object_serial, pixel2RaDec, select_patch_centers

logger = logging.getLogger(__name__)

# Digest memo for read-only weight arrays (see _fingerprint_weights).
_FINGERPRINT_MEMO: Dict[Any, Any] = {}
#: Frozen row-space map memo bounds (see ``_coerce_map_input_array``).
_FROZEN_MAP_MEMO_MAX_ENTRIES = 16
_FROZEN_MAP_MEMO_MAX_BYTES = 4 << 30


def _frozen_memo_bytes(memo: Dict[Any, Any]) -> int:
    """Device/host bytes held by the frozen row-space memo, derived blocks
    included."""
    total = 0
    for _source, rows, blocks in memo.values():
        total += int(getattr(rows, "nbytes", 0))
        for block in blocks.values():
            total += int(getattr(block, "nbytes", 0))
    return total

_ALLOWED_FLOAT_PRECISIONS = {
    "float32": np.float32,
    "float64": np.float64,
}
_ROTATION_COMPLEX_PRECISION = {
    "float32": np.complex64,
    "float64": np.complex128,
}

# A buffer as the degrade kernels address it:
# (array, base, lead_stride, comp_stride, row_stride), in elements.
_Desc = Tuple[Any, int, int, int, int]

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
) -> Tuple[
    np.ndarray,
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
    nbins = nedges - 1

    # Pass 1 counts accepted pairs per (row, angular bin).  Counting per bin
    # (instead of per row only) lets pass 2 write the pairs *already grouped
    # by bin*: the measurement kernels want them in bin order, and an
    # O(npairs log npairs) argsort plus seven fancy-indexed gathers over
    # ~10^8 pairs used to dominate preprocessing at nside 2048.
    counts = np.zeros((npts, nbins), dtype=np.int64)
    for m in prange(nrows):
        # branchless light/heavy row interleave: even m -> row m//2,
        # odd m -> row nrows-1-m//2 (balances the triangular loop)
        half = m >> 1
        odd = m & 1
        i = half + odd * (nrows - 1 - 2 * half)
        x1 = x[i]
        y1 = y[i]
        z1 = z[i]
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
            if b >= 0 and b < nbins and cos_theta > cos_binedges[b + 1]:
                counts[i, b] += 1

    # Per-bin totals, then the write cursor of every (row, bin): bins in
    # order, and inside a bin the rows in ascending order -- exactly the
    # order a stable sort by bin index produced.
    bin_counts = np.zeros(nbins, dtype=np.int64)
    for b in range(nbins):
        acc = 0
        for i in range(npts):
            acc += counts[i, b]
        bin_counts[b] = acc

    cursor = np.zeros((npts, nbins), dtype=np.int64)
    running = 0
    for b in range(nbins):
        acc = running
        for i in range(npts):
            cursor[i, b] = acc
            acc += counts[i, b]
        running = acc
    total = running

    inds_a = np.empty(total, dtype=patch_inds.dtype)
    inds_b = np.empty(total, dtype=patch_inds.dtype)
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
            if bin_idx < 0 or bin_idx >= nbins or not (
                cos_theta > cos_binedges[bin_idx + 1]
            ):
                continue
            out_idx = cursor[i, bin_idx]
            cursor[i, bin_idx] = out_idx + 1

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
            exp2phi1_real[out_idx] = c1
            exp2phi1_imag[out_idx] = s1
            exp2phi2_real[out_idx] = c2
            exp2phi2_imag[out_idx] = s2

    # No per-pair bin index is returned: ``bin_counts`` already describes
    # the grouping (the kernel writes the pairs grouped by bin), and the
    # array it replaces was 8 B/pair of host memory written once per pair
    # and never read -- only its ``.size``, which is ``inds_a.size``.
    return (
        inds_a,
        inds_b,
        exp2phi1_real,
        exp2phi1_imag,
        exp2phi2_real,
        exp2phi2_imag,
        bin_counts,
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

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """``device=[0, 1, ...]`` builds a patch-parallel multi-GPU group.

        A single device (the default) is unaffected: the instance returned
        is an ordinary :class:`Correlation`.
        """
        device = kwargs.get("device", args[10] if len(args) > 10 else None)
        if isinstance(device, (list, tuple, set)):
            from .multi_device import MultiDeviceCorrelation

            devices = list(device)
            kwargs.pop("device", None)
            if len(args) > 10:
                args = args[:10]
            if len(devices) == 1:  # a one-element list is just that device
                return super().__new__(cls)
            return MultiDeviceCorrelation(*args, devices=devices, **kwargs)
        return super().__new__(cls)

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
        device: Union[str, int, Sequence[int]] = "auto",
        map_precision: Union[str, np.dtype, type] = "float64",
        rotation_precision: Union[str, np.dtype, type] = "float32",
        accumulation_precision: str = "same",
        resolution_factor: Union[None, bool, str, float] = None,
        aperture_nside: Optional[int] = None,
        memory_budget_gb: Optional[float] = None,
        pair_search_precision: str = "float64",
        pack_pairs: bool = False,
        pack_host_pairs: bool = False,
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
            accumulation_precision: "same" (default) accumulates the pair
                sums at map precision; "float64" accumulates and reduces at
                float64 even for float32 maps (recommended with
                map_precision="float32": per-pair products are computed at
                float32 but summed without float32 cancellation error).
            resolution_factor: ``None`` (default) measures every angular
                bin on the full-resolution map.  ``True`` (or
                ``"default"``) switches on the *static treecode* with the
                default factor ``k = 4``
                (:data:`CosmoFuse.DEFAULT_RESOLUTION_FACTOR`); a positive
                number sets ``k`` explicitly.  Bin ``b`` is then measured on
                the coarsest HEALPix level whose pixel size is
                ``<= theta_lo(b) / k`` (per-patch cells, weighted means at
                their centroids).  This cuts the number of pairs by orders
                of magnitude at large separations, but it is a *different
                (windowed) estimator*: use the same value for data,
                simulations and covariances.  See :attr:`level_table`.
            aperture_nside: ``None`` (default) evaluates the aperture
                statistics at the map resolution.  A power-of-two nside
                below ``nside`` evaluates them on the (globally) degraded
                map instead, keeping the aperture geometry at that size.
            memory_budget_gb: Budget for the preflight check that runs
                before pair finding.  ``None`` uses the free device memory
                on GPU backends (no check on CPU); ``float("inf")``
                disables the check.
            pack_pairs: Store the device pair geometry in 8 instead of 24
                bytes per pair (uint16 rotation angles + uint16 patch-local
                row indices; the map rows of every patch are gathered into
                contiguous blocks once per map).  Three times more pairs fit
                on the device -- a higher ``resolution_factor`` or more
                patches.  Not bit-identical to the unpacked geometry: the
                angle quantisation (<= 4.8e-5 rad) perturbs every estimate
                by ~1e-5 of its statistical error and does not bias it.
                Pair files and host arrays stay exact.  On GPU backends the
                packed geometry serves every tomographic method
                (``vectorized_*``, ``get_full_tomo_*``, ``get_3x2pt_tomo``)
                and the aperture statistics; the single-map ``compute_*``
                methods need the unpacked geometry.
            pair_search_precision: Precision of the pair *search*
                (separations, binning, position angles); the stored
                rotation factors always use ``rotation_precision``.
                ``"float64"`` (default) searches at double precision at no
                memory cost.  ``"float32"`` resolves separations only to
                ``d(theta)/theta ~ 6e-8 / theta^2`` (0.3 % at 15', 3 % at
                5') and mis-bins pairs at the 0.05 sigma (rms) / 0.9 sigma
                (max) per-patch level on the DES nside-512 production
                geometry, so it is only useful for throwaway runs.
            pack_host_pairs: Apply the same packing already at pair-finding
                time, so the *host* arrays and the pair file hold 8 instead
                of 24 bytes per pair as well (``pair_inds`` /
                ``pair_exp2phi`` become ``None``; the packed payload is in
                ``packed_pairs``).  Default ``False``, and independent of
                ``pack_pairs``: it saves host RAM and disk, not device
                memory.  The price is that the geometry is then quantised
                *everywhere* -- the pair file is no longer exact and cannot
                be turned back into one, and both backends measure the
                quantised estimator.  Pair files written this way carry
                format version 4.

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
        if accumulation_precision not in ("same", "float64"):
            raise ValueError(
                "accumulation_precision must be 'same' or 'float64', "
                f"got {accumulation_precision!r}"
            )
        self.accumulation_precision = accumulation_precision
        self.acc_dtype = (
            self.map_dtype
            if accumulation_precision == "same"
            else np.dtype(np.float64)
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
        # Truncation radius of the aperture geometry, in arcminutes.  Q
        # has decayed below 1e-3 of its peak by 5 theta_Q (Q_crittenden has
        # formally unbounded support); Q_schneider is exactly zero past
        # theta_Q.  Single source: pair_geometry builds the discs from it.
        self.radius_filter = 5 * self.theta_Q

        self.resolution_factor = validate_resolution_factor(resolution_factor)
        if pair_search_precision not in ("float32", "float64"):
            raise ValueError(
                "pair_search_precision must be 'float32' or 'float64'; got "
                f"{pair_search_precision!r}"
            )
        self.pair_search_precision = pair_search_precision
        self.pack_pairs = bool(pack_pairs)
        self.pack_host_pairs = bool(pack_host_pairs)
        self._pair_finder = self._make_pair_finder()
        if self._pair_finder.search_dtype == np.dtype(np.float32):
            jitter = 6e-8 / self.theta_min**2
            if jitter > 0.01:
                logger.warning(
                    "Pair search runs at float32: separations near theta_min="
                    "%.1f' are only resolved to ~%.1f%%, so pairs are assigned "
                    "to the wrong angular bin at that level. Pass "
                    "pair_search_precision='float64' (no memory cost).",
                    theta_min,
                    100 * jitter,
                )
        self.level_nside = assign_levels(
            self.binedges, self.nside, self.resolution_factor
        )
        if aperture_nside is not None:
            aperture_nside = int(aperture_nside)
            if (
                not is_power_of_two(aperture_nside)
                or not is_power_of_two(nside)
                or aperture_nside > nside
            ):
                raise ValueError(
                    "aperture_nside must be a power of two <= nside (and nside "
                    f"a power of two); got aperture_nside={aperture_nside}, "
                    f"nside={nside}"
                )
            if aperture_nside == nside:
                aperture_nside = None
        self.aperture_nside = aperture_nside
        if memory_budget_gb is not None and not (memory_budget_gb > 0):
            raise ValueError("memory_budget_gb must be positive")
        self.memory_budget_gb = memory_budget_gb
        # Host-side coarse-cell geometry (set by pair finding / load_pairs)
        self._treecode: Optional[TreecodeGeometry] = None

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

        if isinstance(device, (list, tuple, set)):
            # a single-element device list is just that device; several
            # devices are handled by MultiDeviceCorrelation (see __new__)
            device = list(device)[0]
        self.backend = get_backend(device)
        self.device = device
        self.compute_context = ComputeContext()

        self.pair_inds = []
        self.pair_exp2phi = []
        self.bins = []
        # Aperture geometry, filled by calculate_pairs_M_a()/load_pairs().
        # Empty rather than absent: ensure_aperture_pairs() reads Q_inds to
        # decide whether the cached geometry can be reused, so without these
        # every aperture entry point raised AttributeError on a fresh
        # instance instead of building the geometry on demand.
        self.Q_inds: List[np.ndarray] = []
        self.Q_cos: List[np.ndarray] = []
        self.Q_sin: List[np.ndarray] = []
        self.Q_val: List[np.ndarray] = []
        self.Q_patch_area: List[float] = []
        # Host-packed pair payload (pack_host_pairs=True); replaces
        # pair_inds/pair_exp2phi rather than accompanying them.
        self.packed_pairs: Optional[List[np.ndarray]] = None
        self.packed_block_ids: Optional[List[np.ndarray]] = None
        self.packed_block_sizes: Optional[List[np.ndarray]] = None
        self.compute_context.initialize_runtime_state()
        self._aperture_filter_active_key = "Q_crittenden"
        self._prepare_failed = False

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
        aperture_filter: Optional[Callable[..., Any]] = None,
        filter_weighting: str = "abs",
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
            filter_weighting: ``"abs"`` (default: masked fraction of
                ``|filter|`` weight) or ``"signed"`` (deviation of the
                filter integral, ``|Σ_masked Q| / Σ|Q|``); always from the
                binary mask, see
                :func:`CosmoFuse.utils.select_patch_centers`.
            aperture_filter: Filter for the weighted check; defaults to
                the built-in ``Q_crittenden``.  Selection only — pass the
                same filter to :meth:`preprocess` for consistency.
            **kwargs: Forwarded to the constructor (``nbins``,
                ``theta_min``, ``theta_max``, ``device``, ``fastmath``,
                ``map_precision``, ``rotation_precision``,
                ``resolution_factor``, ``aperture_nside``, ...).

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
            aperture_filter=aperture_filter,
            filter_weighting=filter_weighting,
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

    def _make_pair_finder(self) -> PairFinder:
        search_dtype = np.dtype(self.pair_search_precision)
        return PairFinder(
            nbins=self.nbins,
            binedges=self.binedges,
            index_dtype=self.index_dtype,
            rotation_dtype=self.rotation_dtype,
            rotation_complex_dtype=self.rotation_complex_dtype,
            kernel=self._compute_pairs_kernel,
            search_dtype=search_dtype,
        )

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        # map_mask is rebuilt from map_inds in __setstate__; dropping it keeps
        # the full-sky boolean array out of every pickle.
        state.pop('map_mask', None)
        state.pop('_aperture_cells_cache', None)
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
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore a pickled instance and rebuild the unpicklable state.

        Device buffers, the backend, the Numba kernels and the pair finder
        are never pickled; the prepared state is rebuilt by
        :meth:`prepare`.
        """
        self.__dict__.update(state)
        self.map_mask = np.zeros(hp.nside2npix(self.nside), dtype=bool)
        self.map_mask[self.map_inds] = True
        self.backend = get_backend(self.device)
        self.aperture_shear_all_patches = njit(fastmath=self.fastmath, cache=True)(
            _compute_aperture_shear_all_patches
        )
        self._compute_pairs_kernel = _get_pairs_numba_kernel(self.fastmath)
        self._pair_finder = self._make_pair_finder()
        self.compute_context = ComputeContext()

    def _invalidate_prepared_state(self) -> None:
        """Clears prepared backend buffers and cached tomographic weights."""
        self.compute_context.invalidate_prepared_state()

    def warmup(self) -> None:
        """Eagerly JIT-compile the CPU kernels for this instance's dtypes.

        Moves the one-off Numba compilation cost (roughly 10-20 s in a
        fresh environment; sub-second afterwards thanks to the on-disk
        cache) out of the first map measurement.

        On a GPU backend the *measurement* kernels are CUDA and are compiled
        per tomographic-bin-count template on first launch, so
        ``Backend.warmup`` is a no-op there -- but the **pair search** is
        Numba on both backends, so its kernel is compiled here regardless of
        backend.  It is compiled at ``pair_search_precision``, which is what
        ``preprocess()`` will use; compiling it at ``rotation_precision``
        (as this used to) warmed a signature the pair search never calls.
        """
        self.backend.warmup(
            map_dtype=self.map_dtype,
            rotation_dtype=self.rotation_dtype,
            rotation_complex_dtype=self.rotation_complex_dtype,
            index_dtype=self.index_dtype,
        )
        search_dtype = np.dtype(self.pair_search_precision)
        pts = np.arange(3, dtype=self.index_dtype)
        ra = np.array([0.0, 1e-3, 2e-3], dtype=search_dtype)
        dec = np.array([0.0, 1e-3, 0.0], dtype=search_dtype)
        binedges = np.asarray(self.binedges, dtype=search_dtype)
        self._compute_pairs_kernel(pts, ra, dec, binedges)

    @property
    def inds_dev(self) -> Any:
        return self.compute_context.inds_dev

    @inds_dev.setter
    def inds_dev(self, value: Any) -> None:
        # ctx.inds_i_dev / inds_j_dev are the contiguous, kernel-dtype copies
        # of exactly this array (see _pair_index_arrays).  Leaving them in
        # place would have the next measurement read the previous geometry.
        self.compute_context.inds_dev = value
        self.compute_context.inds_i_dev = None
        self.compute_context.inds_j_dev = None

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
        # The device copy is rebuilt from this by
        # _prepare_aperture_device_buffers(); drop it so it cannot be read
        # back against new host geometry.
        self._invalidate_aperture_device_buffers()

    @property
    def Q_cos_flat(self) -> Any:
        return self.compute_context.Q_cos_flat

    @Q_cos_flat.setter
    def Q_cos_flat(self, value: Any) -> None:
        self.compute_context.Q_cos_flat = value
        # The device copy is rebuilt from this by
        # _prepare_aperture_device_buffers(); drop it so it cannot be read
        # back against new host geometry.
        self._invalidate_aperture_device_buffers()

    @property
    def Q_sin_flat(self) -> Any:
        return self.compute_context.Q_sin_flat

    @Q_sin_flat.setter
    def Q_sin_flat(self, value: Any) -> None:
        self.compute_context.Q_sin_flat = value
        # The device copy is rebuilt from this by
        # _prepare_aperture_device_buffers(); drop it so it cannot be read
        # back against new host geometry.
        self._invalidate_aperture_device_buffers()

    @property
    def Q_val_flat(self) -> Any:
        return self.compute_context.Q_val_flat

    @Q_val_flat.setter
    def Q_val_flat(self, value: Any) -> None:
        self.compute_context.Q_val_flat = value
        # The device copy is rebuilt from this by
        # _prepare_aperture_device_buffers(); drop it so it cannot be read
        # back against new host geometry.
        self._invalidate_aperture_device_buffers()

    @property
    def Q_offsets(self) -> Any:
        return self.compute_context.Q_offsets

    @Q_offsets.setter
    def Q_offsets(self, value: Any) -> None:
        self.compute_context.Q_offsets = value
        # The device copy is rebuilt from this by
        # _prepare_aperture_device_buffers(); drop it so it cannot be read
        # back against new host geometry.
        self._invalidate_aperture_device_buffers()

    @property
    def Q_patch_area_flat(self) -> Any:
        return self.compute_context.Q_patch_area_flat

    @Q_patch_area_flat.setter
    def Q_patch_area_flat(self, value: Any) -> None:
        self.compute_context.Q_patch_area_flat = value
        # The device copy is rebuilt from this by
        # _prepare_aperture_device_buffers(); drop it so it cannot be read
        # back against new host geometry.
        self._invalidate_aperture_device_buffers()

    def _invalidate_aperture_device_buffers(self) -> None:
        """Drop the device aperture buffers derived from the host flats."""
        ctx = self.compute_context
        for name, value in ctx._APERTURE_DEVICE_DEFAULTS:
            setattr(ctx, name, value)

    # ---- which row of a returned data vector is which pair of bins -------

    @staticmethod
    def tomo_combinations(
        nzbins: int, gc_auto_correlations_only: bool = False
    ) -> List[Tuple[int, int]]:
        """Row order of the symmetric tomographic outputs.

        ``xi_p``, ``xi_m`` and ``xi_g`` come back as
        ``(ncomb, n_patches, nbins)``; this is what row ``k`` means.  The
        order is the upper triangle including the diagonal, row-major::

            >>> Correlation.tomo_combinations(3)
            [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]

        With ``gc_auto_correlations_only=True`` (``vectorized_density_density``
        and the ξ_g section of ``get_3x2pt_tomo``) only the diagonal is
        measured and the vector is ``nzbins`` rows, not ``nzbins(nzbins+1)/2``
        -- the silent shape change that makes mislabelling easy.

        Mislabelling a data vector is not caught by any shape check, so this
        is the accessor to index it by rather than to rederive.
        """
        if gc_auto_correlations_only:
            return [(i, i) for i in range(int(nzbins))]
        return [
            (i, j) for i in range(int(nzbins)) for j in range(i, int(nzbins))
        ]

    @staticmethod
    def ggl_combinations(
        nlens_bins: int,
        nsource_bins: int,
        ggl_bin_combinations: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> List[Tuple[int, int]]:
        """Row order of ``xi_t`` as ``(lens_bin, source_bin)``.

        The full cartesian product, lens-major, unless an explicit
        ``ggl_bin_combinations`` selection is passed -- in which case the
        rows are exactly that selection, in the order given::

            >>> Correlation.ggl_combinations(2, 3)
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        """
        if ggl_bin_combinations is not None:
            return [(int(a), int(b)) for a, b in ggl_bin_combinations]
        return [
            (i, j)
            for i in range(int(nlens_bins))
            for j in range(int(nsource_bins))
        ]

    @staticmethod
    def zeta_triplets(nzbins: int) -> List[Tuple[int, int, int]]:
        """Row order of the ζ estimators as ``(z_center, z2, z3)``.

        What :func:`~CosmoFuse.correlation_helpers.calculate_all_zetas`
        returns along axis 1, and therefore what ``ZetaWriter`` stores --
        for the estimators whose centre and annulus share one tomographic
        sample: ``zeta_a_plus``, ``zeta_a_minus`` and ``zeta_g_g``.  The
        others cross two samples and use :meth:`zeta_cross_triplets`.
        """
        import itertools

        return list(itertools.combinations_with_replacement(range(int(nzbins)), 3))

    @staticmethod
    def zeta_cross_triplets(
        n_central_bins: int, n_annulus_combinations: int
    ) -> List[Tuple[int, int]]:
        """Row order of the ζ estimators that cross two samples.

        ``(z_center, annulus_row)``, centre-major, where ``annulus_row``
        indexes the 2PCF's own combination list --
        :meth:`tomo_combinations` for ξ±/ξ_g, :meth:`ggl_combinations` for
        ξ_t.  This is the layout of ``zeta_g_plus``, ``zeta_g_minus``,
        ``zeta_a_g``, ``zeta_g_t`` and ``zeta_a_t`` whenever the centre and
        the annulus are built from different samples -- which is every
        6x2pt run with ``n_source != n_lens``, and, with an explicit
        ``symmetric=False``, also when the two counts happen to match::

            >>> Correlation.zeta_cross_triplets(2, 3)
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        """
        return [
            (c, a)
            for c in range(int(n_central_bins))
            for a in range(int(n_annulus_combinations))
        ]

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
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        return PairGeometry.get_pairs_patch(
            self,
            patch_inds,
            ra,
            dec,
        )

    def __get_pairs_helper__(
        self,
        i: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return PairGeometry.get_pairs_helper(self, i)

    def calculate_pairs_2PCF(self) -> None:
        PairGeometry.calculate_pairs_2PCF(self)

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

    def _aperture_row_inds(self) -> np.ndarray:
        """Flat aperture pixel indices in the compact row space (host, cached)."""
        ctx = self.compute_context
        cached = getattr(ctx, "Q_rows_flat", None)
        if cached is not None and cached[0] is self.Q_inds_flat:
            return cached[1]
        lut = self._global_to_row_lut()
        if lut is None:
            rows = self.Q_inds_flat
        else:
            rows = self._global_ids_to_rows(self.Q_inds_flat, lut)
            if rows.size > 0 and int(rows.min()) < 0:
                raise ValueError(
                    "Aperture indices reference pixels outside the mask "
                    "(map_inds); the aperture geometry does not belong to "
                    "this mask."
                )
        ctx.Q_rows_flat = (self.Q_inds_flat, rows)
        return rows

    def _invalidate_aperture_device_buffers(self) -> None:
        self.compute_context.Q_rows_flat = None
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

        # The filter geometry stays at rotation precision on device (the
        # kernels promote it to the map type at use, which is exact for
        # float32 -> float64): halves the persistent aperture memory and
        # its upload when rotation_precision is float32.
        q_inds_u32 = np.asarray(self._aperture_row_inds(), dtype=np.uint32)
        self.compute_context.Q_inds_dev = module.ascontiguousarray(
            self.backend.to_device(q_inds_u32)
        )
        self.compute_context.Q_cos_dev = module.ascontiguousarray(
            self.backend.to_device(np.asarray(self.Q_cos_flat, dtype=self.rotation_dtype))
        )
        self.compute_context.Q_sin_dev = module.ascontiguousarray(
            self.backend.to_device(np.asarray(self.Q_sin_flat, dtype=self.rotation_dtype))
        )
        self.compute_context.Q_val_dev = module.ascontiguousarray(
            self.backend.to_device(np.asarray(self.Q_val_flat, dtype=self.rotation_dtype))
        )
        self.compute_context.Q_offsets_dev = module.ascontiguousarray(
            self.backend.to_device(np.asarray(self.Q_offsets, dtype=np.int64))
        )
        self.compute_context.Q_patch_area_dev = module.ascontiguousarray(
            self.backend.to_device(
                np.asarray(self.Q_patch_area_flat, dtype=self.rotation_dtype)
            )
        )

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
    ) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
        ctx = self.compute_context
        if ctx.fused_output_buffers is None:
            ctx.fused_output_buffers = {}

        cache = ctx.fused_output_buffers
        module = self.backend.module
        empty = getattr(module, "empty", None)

        # Pair statistics: one row per combination, both orientations of a
        # cross combination summed (the kernels' contract on both backends).
        specs = (
            ("out_ma_num", (n_shear_bins, n_patches)),
            ("out_ma_den", (n_shear_bins, n_patches)),
            ("out_mg_num", (n_density_bins, n_patches)),
            ("out_mg_den", (n_density_bins, n_patches)),
            ("out_xipm_num", (2, ss_ncomb, nbins_total)),   # [xi+ | xi-]
            ("out_xipm_den", (ss_ncomb, nbins_total)),
            ("out_xig_num", (dd_ncomb, nbins_total)),
            ("out_xig_den", (dd_ncomb, nbins_total)),
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
            buffers["out_xipm_num"],
            buffers["out_xipm_den"],
            buffers["out_xig_num"],
            buffers["out_xig_den"],
            buffers["out_xit_num"],
            buffers["out_xit_den"],
        )

    def preprocess(
        self,
        aperture_filter: Optional[Callable[..., Any]] = None,
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
        self.calculate_pairs_2PCF()
        logger.info("Preparing flattened pair arrays on backend device")
        if release_host_pairs:
            self.prepare(release_host_pairs=True)
        else:
            self.prepare()

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
        (g1_arr, g2_arr), w_arr = self._expand_rows(
            (g1_arr, g2_arr), w_arr, blocks="aperture"
        )
        return self._aperture_shear_rows(g1_arr, g2_arr, w_arr, return_device)

    def _aperture_shear_rows(
        self, g1_arr: Any, g2_arr: Any, w_arr: Any, return_device: bool = True
    ) -> np.ndarray:
        """Aperture mass of one map whose virtual rows are already appended."""
        kernel = getattr(self.backend, "aperture_shear_kernel", None)
        if kernel is None:
            raise RuntimeError(
                "Backend does not provide an aperture-shear kernel; use a supported backend."
            )

        if self.backend.name == "numpy":
            aperture_shear = np.zeros(self.n_patches, dtype=self.map_dtype)
            kernel(
                self._aperture_row_inds(),
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

        tomo_kernel = getattr(self.backend, "aperture_tomo_shear_kernel", None)
        if tomo_kernel is not None:
            # Single launch over (patch,) blocks with ntomo=1: avoids the two
            # aperture-sized temporaries and the reduceat pass below.
            out_num = self.backend.zeros((1, self.n_patches), dtype=backend_dtype)
            out_den = self.backend.zeros((1, self.n_patches), dtype=backend_dtype)
            launched = tomo_kernel(
                g1_dev.reshape(1, -1),
                g2_dev.reshape(1, -1),
                w_dev.reshape(1, -1),
                Q_inds_dev,
                Q_cos_dev,
                Q_sin_dev,
                Q_val_dev,
                Q_offsets_dev,
                Q_patch_area_dev,
                out_num,
                out_den,
            )
            if launched:
                # Numerator is already area-scaled inside the kernel.
                aperture_shear = self._normalize_by_weights(out_num, out_den)[0]
                if return_device and self.backend.name == "cupy":
                    return aperture_shear
                return self.backend.to_numpy(aperture_shear)

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
        (map_values_arr,), w_arr = self._expand_rows(
            (map_values_arr,), w_arr, blocks="aperture"
        )
        return self._aperture_density_rows(map_values_arr, w_arr, return_device)

    def _aperture_density_rows(
        self, map_values_arr: Any, w_arr: Any, return_device: bool = True
    ) -> np.ndarray:
        """Aperture density of one map whose virtual rows are already appended."""
        kernel = getattr(self.backend, "aperture_density_kernel", None)
        if kernel is None:
            raise RuntimeError(
                "Backend does not provide an aperture-density kernel; use a supported backend."
            )

        if self.backend.name == "numpy":
            aperture_density = np.zeros(self.n_patches, dtype=self.map_dtype)
            kernel(
                self._aperture_row_inds(),
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

        tomo_kernel = getattr(self.backend, "aperture_tomo_density_kernel", None)
        if tomo_kernel is not None:
            # Single launch over (patch,) blocks with ntomo=1: avoids the two
            # aperture-sized temporaries and the reduceat pass below.
            out_num = self.backend.zeros((1, self.n_patches), dtype=backend_dtype)
            out_den = self.backend.zeros((1, self.n_patches), dtype=backend_dtype)
            launched = tomo_kernel(
                map_values_dev.reshape(1, -1),
                w_dev.reshape(1, -1),
                Q_inds_dev,
                Q_val_dev,
                Q_offsets_dev,
                Q_patch_area_dev,
                out_num,
                out_den,
            )
            if launched:
                # Numerator is already area-scaled inside the kernel.
                aperture_density = self._normalize_by_weights(out_num, out_den)[0]
                if return_device and self.backend.name == "cupy":
                    return aperture_density
                return self.backend.to_numpy(aperture_density)

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
                ``pair_exp2phi``, ``bins``, and the ``pack_host_pairs``
                payload) after device buffers are built to reduce RAM usage
                for large runs.
        """
        host_packed = self.packed_pairs is not None
        host_pairs_available = self.bins is not None and (
            host_packed
            or (self.pair_inds is not None and self.pair_exp2phi is not None)
        )
        if not host_pairs_available:
            if (
                (
                    (self.inds_dev is not None and self.exp2phi_dev is not None)
                    or self.compute_context.packed_pairs_dev is not None
                )
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
            # int(): np.sum over unsigned bins yields uint64, and under
            # legacy (numpy<2) promotion `int + uint64` silently becomes
            # float64, which is later rejected as an array shape.
            patchsize = int(np.sum(self.bins[i]))
            size += patchsize
            ninds.append(patchsize)

        first_patch_ind = np.append(0, np.cumsum(ninds)).astype(int)
        # Payload packing (8 B per pair).  On GPU backends the unpacked
        # 24 B geometry is then not built at all; the CPU kernels keep using
        # unpacked arrays, but with the *same* quantised rotation factors,
        # so both backends measure the same estimator.
        pack = bool(self.pack_pairs)
        keep_unpacked = not (pack and self.backend.name == "cupy")
        # np.empty: every element is written in the loop below
        if keep_unpacked:
            temp_inds = np.empty((2, int(size)), dtype=self.index_dtype)
            temp_exp2phi = np.empty((2, int(size)), dtype=self.rotation_complex_dtype)
        groups = level_groups(self.level_nside)
        block_of_bin = np.zeros(self.nbins, dtype=np.int64)
        for g, (_nside_g, b0, b1) in enumerate(groups):
            block_of_bin[b0:b1] = g
        if pack:
            packed = np.empty((int(size), 4), dtype=np.uint16)
            packed_row_base = np.zeros(self.n_patches * self.nbins, dtype=np.int64)
            packed_blocks: List[np.ndarray] = []
            n_packed_rows = 0
        temp_bins = np.empty((self.n_patches * self.nbins), dtype=self.index_dtype)
        # int64: the cumulative pair offsets exceed int32 beyond 2^31 pairs
        temp_bins_tot = np.empty((self.n_patches * self.nbins + 1), dtype=np.int64)
        temp_bins_tot[0] = 0

        # Device pair indices address the compact row space (host pair_inds
        # keep HEALPix ids); remapped per patch so no pair-sized temporary
        # is needed.
        lut = self._global_to_row_lut()
        for i in range(self.n_patches):
            lo, hi = first_patch_ind[i], first_patch_ind[i + 1]
            if host_packed:
                # The host payload is already packed; only the small per-group
                # row blocks have to be translated from global ids to rows.
                packed_i = self.packed_pairs[i]
                edges = block_edges(self.packed_block_sizes[i])
                ids_i = self.packed_block_ids[i]
                blocks = [
                    self._pair_ids_to_rows(ids_i[edges[g] : edges[g + 1]], lut)
                    for g in range(len(groups))
                ]
                if keep_unpacked:
                    rows_i = unpack_rows(
                        packed_i, blocks, self.bins[i], groups, self.index_dtype
                    )
                    exp_i = decode_angles(
                        packed_i[:, 2:].T, self.rotation_complex_dtype
                    )
            else:
                rows_i = self._pair_ids_to_rows(self.pair_inds[i], lut)
                exp_i = self.pair_exp2phi[i]
                if pack:
                    packed_i, blocks, _ = pack_patch(
                        np.asarray(rows_i), exp_i, self.bins[i], groups
                    )
                    if keep_unpacked:
                        exp_i = decode_angles(
                            packed_i[:, 2:].T, self.rotation_complex_dtype
                        )
            if pack:
                packed[lo:hi] = packed_i
                sizes = np.array([blk.size for blk in blocks], dtype=np.int64)
                starts = n_packed_rows + np.concatenate(([0], np.cumsum(sizes)))[:-1]
                packed_row_base[i * self.nbins : (i + 1) * self.nbins] = starts[
                    block_of_bin
                ]
                packed_blocks.extend(blocks)
                n_packed_rows += int(sizes.sum())
            if keep_unpacked:
                temp_inds[:, lo:hi] = rows_i
                temp_exp2phi[:, lo:hi] = exp_i
            temp_bins[i * self.nbins : (i + 1) * self.nbins] = self.bins[i]
            temp_bins_tot[1 + i * self.nbins : 1 + (i + 1) * self.nbins] = (
                first_patch_ind[i] + np.cumsum(self.bins[i], dtype=np.int64)
            )
        del lut

        module = self.backend.module
        ctx = self.compute_context
        # From here on the object is part-way through publishing its device
        # buffers; _ensure_prepared() refuses to measure until the tail of
        # this method clears the flag.
        self._prepare_failed = True
        if keep_unpacked:
            self.inds_dev = self.backend.to_device(temp_inds)
            _index_device_dtype = getattr(module, self.index_dtype.name)
            ctx.inds_i_dev = module.ascontiguousarray(
                self.inds_dev[0].astype(_index_device_dtype, copy=False)
            )
            ctx.inds_j_dev = module.ascontiguousarray(
                self.inds_dev[1].astype(_index_device_dtype, copy=False)
            )
            self.exp2phi_dev = self.backend.to_device(temp_exp2phi)
        else:
            self.inds_dev = None
            self.exp2phi_dev = None
            ctx.inds_i_dev = None
            ctx.inds_j_dev = None
        if pack:
            ctx.packed_pairs_dev = self.backend.to_device(packed)
            ctx.packed_row_base_dev = self.backend.to_device(packed_row_base)
            ctx.packed_perm_dev = self.backend.to_device(
                np.concatenate(packed_blocks).astype(np.int64)
                if packed_blocks
                else np.zeros(0, dtype=np.int64)
            )
            del packed, packed_blocks
        else:
            ctx.packed_pairs_dev = None
            ctx.packed_row_base_dev = None
            ctx.packed_perm_dev = None
        self.bins_dev = self.backend.to_device(temp_bins)
        if size <= np.iinfo(self.index_dtype).max:
            self.tot_bins_dev = self.backend.to_device(
                temp_bins_tot.astype(self.index_dtype)
            )
        else:
            self.tot_bins_dev = self.backend.to_device(temp_bins_tot)
        self.tot_bins_reduceat_dev = self.backend.to_device(temp_bins_tot)
        self._prepare_aperture_device_buffers()
        self.ntotpairs = size
        self.compute_context.prepare_version += 1
        # Published last, so that everything above having succeeded is what
        # clears the flag.  A failure part-way through (an OOM during the
        # aperture upload is the realistic one at nside 2048) leaves the
        # device buffers set but ntotpairs at 0, which _ensure_prepared()
        # cannot tell from a good state -- and the next measurement then
        # returns finite numbers from a torn one.
        self._prepare_failed = False
        if release_host_pairs:
            self.pair_inds = None
            self.pair_exp2phi = None
            self.bins = None
            self.packed_pairs = None
            self.packed_block_ids = None
            self.packed_block_sizes = None

    def _ensure_prepared(self) -> None:
        if getattr(self, "_prepare_failed", False):
            raise RuntimeError(
                "prepare() failed on this object and left it part-way through "
                "publishing its device buffers; it cannot measure. Fix the "
                "cause (usually device memory) and call prepare() again."
            )
        if self.inds_dev is None and self.compute_context.packed_pairs_dev is None:
            if not self._has_pair_geometry():
                raise RuntimeError(
                    "no pair geometry: call preprocess() (or load_pairs()) "
                    "before measuring."
                )
            self.prepare()

    def _has_pair_geometry(self) -> bool:
        """Whether preprocess()/load_pairs() has produced something to prepare."""
        if self.bins is None:
            # released by prepare(release_host_pairs=True) -- already prepared
            return True
        if not len(self.bins):
            return False
        if self.packed_pairs is not None:
            return True
        return bool(self.pair_inds) and bool(self.pair_exp2phi)

    def _packed_kernel_unavailable_message(
        self, statistic: str, nzbins: int, slots_per_comb: int
    ) -> str:
        """Why a packed tomographic launch was declined, and what to do.

        There is no fallback for the packed payload: the per-(bin, row)
        kernels read the 24 B geometry, which ``pack_pairs=True`` does not
        upload.  Decoding the payload to feed them would cost the memory
        packing exists to save, so the honest answer is to say where the
        limit is.
        """
        from .backend import _MAX_TILED_ACCUMULATORS

        ncomb = nzbins * (nzbins + 1) // 2
        max_bins = 0
        while (max_bins + 1) * (max_bins + 2) // 2 * slots_per_comb <= (
            _MAX_TILED_ACCUMULATORS
        ):
            max_bins += 1
        return (
            f"The packed {statistic} kernel is unavailable for {nzbins} "
            f"tomographic bins: the tiled kernel keeps "
            f"{slots_per_comb} accumulators per combination in registers and "
            f"{nzbins} bins need {slots_per_comb * ncomb} of the "
            f"{_MAX_TILED_ACCUMULATORS} available, so at most {max_bins} bins "
            f"fit. There is no packed fallback -- the per-(bin, row) kernels "
            f"read the unpacked geometry, which pack_pairs=True does not "
            f"upload. Construct the Correlation with pack_pairs=False (24 B "
            f"per pair on the device instead of 8 B)."
        )

    def _require_unpacked_pairs(self, what: str) -> None:
        """Paths without a packed kernel need the 24 B device geometry
        (the single-map ``compute_*`` methods and the explicit
        sum-of-weights precomputation; every tomographic method has one)."""
        if self.inds_dev is None:
            raise NotImplementedError(
                f"{what} is not available with pack_pairs=True on a GPU backend; "
                "use the tomographic methods (vectorized_*, get_full_tomo_*, "
                "get_3x2pt_tomo) or construct the Correlation with pack_pairs=False."
            )

    def _pair_index_arrays(self) -> Tuple[Any, Any]:
        """Contiguous device pair indices at the kernel index dtype (cached)."""
        ctx = self.compute_context
        if ctx.inds_i_dev is None or ctx.inds_j_dev is None:
            module = self.backend.module
            ctx.inds_i_dev = module.ascontiguousarray(
                self.inds_dev[0].astype(self._index_device_dtype, copy=False)
            )
            ctx.inds_j_dev = module.ascontiguousarray(
                self.inds_dev[1].astype(self._index_device_dtype, copy=False)
            )
        return ctx.inds_i_dev, ctx.inds_j_dev

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
        self._ensure_prepared()
        self._require_unpacked_pairs("compute_shear_shear")
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
        """Compute scalar density-density 2PCF (w(theta)) for one map pair.

        Ratio-of-sums over both pair orientations, as everywhere else in the
        library.  ``sumofweights``, if given, is the sum for *one*
        orientation and is applied to both -- note that
        :meth:`compute_density_shear` takes its ``sumofweights`` as the
        already-summed total instead.
        """
        self._ensure_prepared()
        self._require_unpacked_pairs("compute_density_density")

        density1_dev = self._map_to_device(density1)
        density2_dev = self._map_to_device(density2)
        w1_in = self._map_to_device(w1)
        w2_in = self._map_to_device(w2)
        (density1_dev,), w1_dev = self._expand_rows((density1_dev,), w1_in, "pairs")
        (density2_dev,), w2_dev = self._expand_rows((density2_dev,), w2_in, "pairs")

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
            # The numba kernel inherits its accumulator dtype from these arrays.
            out_ab = np.empty(nbins_total, dtype=self.acc_dtype)
            out_ba = np.empty(nbins_total, dtype=self.acc_dtype)
            out_ab_w = np.empty(nbins_total, dtype=self.acc_dtype)
            out_ba_w = np.empty(nbins_total, dtype=self.acc_dtype)
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
                sum_ab = self._get_xipm_sumofweights(w1_in, w2_in, w1_dev, w2_dev)
                sum_ba = self._get_xipm_sumofweights(w2_in, w1_in, w2_dev, w1_dev)
            out_ab, out_ba = self._get_pair_scratch(self.acc_dtype, 2)
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

        # Ratio of sums, not mean of ratios: both orientations of a cross
        # combination are summed and divided once, which is the wrapper
        # contract on both backends (TreeCorr's definition) and what
        # ``vectorized_density_density``, ``get_3x2pt_tomo`` and
        # ``compute_density_shear`` already do.  ``0.5 * (N_ab/D_ab +
        # N_ba/D_ba)`` is a different estimator whenever the two orientations
        # carry different weight, i.e. for every cross pair.  The auto case
        # and an explicit ``sumofweights`` (where D_ab == D_ba) are unchanged
        # up to the last bits.
        w_theta = self._normalize_scalar_pairs(w_ab_num + w_ba_num, sum_ab + sum_ba)
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
        self._ensure_prepared()
        self._require_unpacked_pairs("compute_density_shear")

        density_lens_dev = self._map_to_device(density_lens)
        g1_source_dev = self._map_to_device(g1_source)
        g2_source_dev = self._map_to_device(g2_source)
        w_lens_in = self._map_to_device(w_lens)
        w_source_in = self._map_to_device(w_source)
        (density_lens_dev,), w_lens_dev = self._expand_rows(
            (density_lens_dev,), w_lens_in, "pairs"
        )
        (g1_source_dev, g2_source_dev), w_source_dev = self._expand_rows(
            (g1_source_dev, g2_source_dev), w_source_in, "pairs"
        )

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
            # The numba kernel inherits its accumulator dtype from these arrays.
            out_ab = np.empty(nbins_total, dtype=self.acc_dtype)
            out_ba = np.empty(nbins_total, dtype=self.acc_dtype)
            out_ab_w = np.empty(nbins_total, dtype=self.acc_dtype)
            out_ba_w = np.empty(nbins_total, dtype=self.acc_dtype)
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
                sum_ab = self._get_xipm_sumofweights(
                    w_lens_in, w_source_in, w_lens_dev, w_source_dev
                )
                sum_ba = self._get_xipm_sumofweights(
                    w_source_in, w_lens_in, w_source_dev, w_lens_dev
                )
                sumofweights_dev = sum_ab + sum_ba
            out_ab, out_ba = self._get_pair_scratch(self.acc_dtype, 2)
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
        g1_dev = self._map_to_device(g1)
        g2_dev = self._map_to_device(g2)
        w_in = self._map_to_device(w)
        (g1_dev, g2_dev), w_dev = self._expand_rows((g1_dev, g2_dev), w_in, "pairs")

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
            # The numba kernel inherits its accumulator dtype from these arrays.
            out_p = np.empty(nbins_total, dtype=self.acc_dtype)
            out_m = np.empty(nbins_total, dtype=self.acc_dtype)
            out_w = np.empty(nbins_total, dtype=self.acc_dtype)
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
            xip_num = out_p
            xim_num = out_m
            if sumofweights_dev is None:
                sumofweights_dev = out_w
        else:
            if sumofweights_dev is None:
                sumofweights_dev = self._get_xipm_sumofweights(
                    w_in, w_in, w_dev, w_dev
                )
            # The kernel computes at map precision and emits the real
            # parts directly.
            out_p, out_m = self._get_pair_scratch(self.acc_dtype, 2)

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

            xip_num = self._reduce_pairs(out_p)
            xim_num = self._reduce_pairs(out_m)
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
        g11_dev = self._map_to_device(g11)
        g21_dev = self._map_to_device(g21)
        g12_dev = self._map_to_device(g12)
        g22_dev = self._map_to_device(g22)
        w1_in = self._map_to_device(w1)
        w2_in = self._map_to_device(w2)
        (g11_dev, g21_dev), w1_dev = self._expand_rows((g11_dev, g21_dev), w1_in, "pairs")
        (g12_dev, g22_dev), w2_dev = self._expand_rows((g12_dev, g22_dev), w2_in, "pairs")

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
            # The numba kernel inherits its accumulator dtype from these arrays.
            out_ab_p = np.empty(nbins_total, dtype=self.acc_dtype)
            out_ab_m = np.empty(nbins_total, dtype=self.acc_dtype)
            out_ba_p = np.empty(nbins_total, dtype=self.acc_dtype)
            out_ba_m = np.empty(nbins_total, dtype=self.acc_dtype)
            out_ab_w = np.empty(nbins_total, dtype=self.acc_dtype)
            out_ba_w = np.empty(nbins_total, dtype=self.acc_dtype)
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

            xip_ab_num = out_ab_p
            xim_ab_num = out_ab_m
            xip_ba_num = out_ba_p
            xim_ba_num = out_ba_m
            if sum_ab is None:
                sum_ab = out_ab_w
            if sum_ba is None:
                sum_ba = out_ba_w
        else:
            if sum_ab is None:
                sum_ab = self._get_xipm_sumofweights(w1_in, w2_in, w1_dev, w2_dev)
            if sum_ba is None:
                sum_ba = self._get_xipm_sumofweights(w2_in, w1_in, w2_dev, w1_dev)
            # The kernel computes at map precision and emits the real
            # parts directly.
            out_ab_p, out_ab_m, out_ba_p, out_ba_m = self._get_pair_scratch(
                self.acc_dtype, 4
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

            xip_ab_num = self._reduce_pairs(out_ab_p)
            xim_ab_num = self._reduce_pairs(out_ab_m)
            xip_ba_num = self._reduce_pairs(out_ba_p)
            xim_ba_num = self._reduce_pairs(out_ba_m)

        # Ratio of the summed orientations, the same weighted estimator the
        # tomographic wrappers use.  With one explicit sumofweights for both
        # orientations this is identical to normalising them separately.
        xip_dev, xim_dev = self._normalize_xipm_pairs(
            xip_ab_num + xip_ba_num,
            xim_ab_num + xim_ba_num,
            sum_ab + sum_ba,
        )

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
        or an array on either backend).  On a GPU this is one fused launch
        rather than five: see :meth:`Backend.safe_divide`.
        """
        return safe_divide(self.backend.module, num, den)

    @staticmethod
    def _align_denominator(num: Any, den: Any) -> Any:
        """Right-pad a per-combination denominator so that it broadcasts
        against a stacked numerator ``(ncomb, ...)``.  ``den`` may be one
        scalar per combination, a full ``(ncomb, nbins_total)`` array, or a
        plain scalar."""
        ndim = getattr(den, "ndim", 0)
        if ndim and ndim < num.ndim:
            return den.reshape(den.shape + (1,) * (num.ndim - ndim))
        return den

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
            f"or ({nzbin_combs}, {nbins_total}) -- or the directional form, "
            f"the same prefixed with a leading 2 for the (a->b, b->a) "
            f"orientations; got {sum_np.shape}"
        )

    def _is_per_comb_sumofweights_shape(
        self, shape: Tuple[int, ...], nzbin_combs: int
    ) -> bool:
        """Whether *shape* is one of the accepted per-combination layouts."""
        nbins_total = self.n_patches * self.nbins
        if len(shape) == 0:
            return True
        if len(shape) == 1:
            size = int(shape[0])
            return (
                size == nzbin_combs
                or (nzbin_combs == 1 and size == nbins_total)
                or size == nzbin_combs * nbins_total
            )
        if len(shape) == 2:
            return shape == (nzbin_combs, nbins_total) or (
                nzbin_combs == 1 and shape == (self.n_patches, self.nbins)
            )
        if len(shape) == 3:
            return shape == (nzbin_combs, self.n_patches, self.nbins)
        return False

    def _normalize_tomo_sumofweights_directional(
        self, sumofweights: Union[np.ndarray, float], nzbin_combs: int
    ) -> Any:
        """Accept either one sum per combination or one per orientation.

        The directional form is the per-combination form with a leading 2.
        Deciding that from ``shape[0] == 2`` alone makes the per-combination
        form unusable whenever there happen to be exactly two tomographic
        combinations -- a ``(2, nbins_total)`` array is then read as two
        orientations of a one-combination measurement.  Deciding it from the
        whole shape removes the collision: an array is directional only when
        what follows the leading 2 is itself a valid per-combination layout.
        """

        def directional(array: Any) -> bool:
            shape = tuple(getattr(array, "shape", ()))
            return (
                len(shape) >= 2
                and shape[0] == 2
                and self._is_per_comb_sumofweights_shape(shape[1:], nzbin_combs)
            )

        if self._is_backend_native_array(sumofweights) and directional(sumofweights):
            sum_ab = self._normalize_tomo_sumofweights_per_comb(sumofweights[0], nzbin_combs)
            sum_ba = self._normalize_tomo_sumofweights_per_comb(sumofweights[1], nzbin_combs)
            return self.backend.module.stack((sum_ab, sum_ba), axis=0)
        sum_np = np.asarray(self.backend.to_numpy(sumofweights), dtype=self.map_dtype)
        if directional(sum_np):
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

    # ── Compact row space ───────────────────────────────────────────────
    # Device map buffers and device pair/aperture indices live in a compact
    # "row space": row r holds footprint pixel ``row_pix[r]`` (the unmasked
    # pixels, ascending RING order).  Masked pixels are never referenced by
    # a pair or an aperture, so nothing is lost; for a partial-sky survey
    # the per-map upload and all per-map device copies shrink by the sky
    # fraction.  Host-side ``pair_inds``/``Q_inds`` keep HEALPix ids.

    @property
    def npix(self) -> int:
        """Number of full-sky HEALPix pixels at ``nside``."""
        return int(hp.nside2npix(self.nside))

    @property
    def row_pix(self) -> np.ndarray:
        """HEALPix (RING) pixel id of every row of the compact row space."""
        return self.map_inds

    @property
    def n_active(self) -> int:
        """Number of rows backed by a map pixel (= unmasked pixels)."""
        return int(self.map_inds.size)

    @property
    def row_pix_hash(self) -> str:
        """Digest of ``(nside, row_pix)``: store it next to row-space map
        archives to detect archives written for a different mask."""
        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.int64(self.nside).tobytes())
        digest.update(np.ascontiguousarray(self.map_inds, dtype=np.int64).tobytes())
        return digest.hexdigest()

    def to_row_space(self, maps: np.ndarray, dtype: Optional[Any] = None) -> np.ndarray:
        """Cut full-sky map(s) ``(..., npix)`` down to the row space
        ``(..., n_active)`` -- the form measured fastest (no mask handling,
        no gather, ``npix / n_active`` times less host→device traffic).  Do
        this once when writing a map archive, not once per measurement."""
        rows = self._gather_rows(np.asarray(maps))
        return np.ascontiguousarray(rows, dtype=dtype or self.map_dtype)

    def _aperture_cells(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Globally degraded aperture level: ``(cell_pix, indptr, child_rows)``.

        ``cell_pix`` are the RING ids (at ``aperture_nside``) of the coarse
        pixels with at least one unmasked child, ascending; the CSR arrays
        list the child *rows* of every cell.  ``None`` at base resolution.
        Global (not per-patch) degrading is right here: the aperture filter
        *is* the window.
        """
        if self.aperture_nside is None:
            return None
        cache = self.__dict__.get("_aperture_cells_cache")
        # live_object_serial, not id(): load_pairs() reassigns map_inds, and
        # the replacement can land on the old array's address -- which would
        # reuse aperture cells built for a different row space.
        key = (
            int(self.aperture_nside),
            int(self.nside),
            live_object_serial(self.map_inds),
        )
        if cache is not None and cache[0] == key:
            return cache[1]
        shift = 2 * (int(np.log2(self.nside)) - int(np.log2(self.aperture_nside)))
        nest = hp.ring2nest(self.nside, np.asarray(self.map_inds, dtype=np.int64))
        parent = hp.nest2ring(self.aperture_nside, nest >> shift)
        cell_pix, inv = np.unique(parent, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        indptr = np.zeros(cell_pix.size + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(np.bincount(inv, minlength=cell_pix.size))
        cells = (cell_pix.astype(np.int64), indptr, order.astype(np.int64))
        self._aperture_cells_cache = (key, cells, self.map_inds)
        return cells

    @property
    def n_aperture_cells(self) -> int:
        cells = self._aperture_cells()
        return 0 if cells is None else int(cells[0].size)

    @property
    def n_appended(self) -> int:
        """Number of virtual rows behind the pixel rows.

        Layout: ``[aperture level | treecode level 1 | level 2 | ...]``.
        """
        n_tree = 0 if self._treecode is None else int(self._treecode.n_cells)
        return self.n_aperture_cells + n_tree

    @property
    def n_rows(self) -> int:
        """Total number of rows of the device map buffers."""
        return self.n_active + self.n_appended

    @property
    def level_table(self) -> Dict[str, Any]:
        """Resolution used by every angular bin (read-only provenance).

        Store this next to every measured data vector: a treecode
        measurement is only comparable to measurements made with the same
        table.
        """
        edges_arcmin = np.degrees(np.asarray(self.binedges, dtype=np.float64)) * 60.0
        return {
            "resolution_factor": self.resolution_factor,
            "base_nside": int(self.nside),
            "aperture_nside": int(self.aperture_nside or self.nside),
            "theta_lo_arcmin": edges_arcmin[:-1].copy(),
            "theta_hi_arcmin": edges_arcmin[1:].copy(),
            "nside": np.asarray(self.level_nside, dtype=np.int64).copy(),
            "effective_resolution_factor": effective_resolution_factor(
                self.binedges, self.level_nside
            ),
        }

    def _global_to_row_lut(self) -> Optional[np.ndarray]:
        """HEALPix id -> row id lookup table (``None`` if rows == pixels)."""
        npix = self.npix
        if self.n_active == npix:
            return None
        lut = np.full(npix, -1, dtype=self.index_dtype)
        lut[self.map_inds] = np.arange(self.n_active, dtype=self.index_dtype)
        return lut

    def _global_ids_to_rows(
        self, ids: np.ndarray, lut: Optional[np.ndarray]
    ) -> np.ndarray:
        """Map global ids (pixel < npix <= virtual row) to device rows."""
        if lut is None:
            # rows == pixels, and virtual row a has id npix + a == n_active + a
            return ids
        if self.n_appended == 0:
            return lut[ids]
        npix = self.npix
        virtual = ids >= npix
        rows = lut[np.where(virtual, 0, ids)]
        return np.where(virtual, ids - (npix - self.n_active), rows).astype(
            self.index_dtype, copy=False
        )

    def _pair_ids_to_rows(
        self, ids: np.ndarray, lut: Optional[np.ndarray]
    ) -> np.ndarray:
        """:meth:`_global_ids_to_rows` with the mask-membership check."""
        rows = self._global_ids_to_rows(ids, lut)
        if lut is not None and rows.size > 0 and int(rows.min()) < 0:
            raise ValueError(
                "Pair indices reference pixels outside the mask (map_inds); "
                "the pair geometry does not belong to this mask."
            )
        return rows

    def _gather_rows(self, array: Any) -> Any:
        """Restrict a map-like array to the row space along its last axis.

        Accepts full-sky arrays (last axis ``npix``; gathered through
        ``row_pix``) and row-space arrays (last axis ``n_active``; returned
        unchanged).  Works for host and device arrays.
        """
        shape = getattr(array, "shape", ())
        if len(shape) == 0:
            return array
        n_last = int(shape[-1])
        n_active = self.n_active
        if n_last == n_active:
            return array
        if n_last != self.npix:
            raise ValueError(
                "map arrays must have either npix="
                f"{self.npix} (full sky) or n_active={n_active} (row space, "
                f"see Correlation.row_pix) pixels along the last axis; got {n_last}"
            )
        if self._is_backend_native_array(array):
            ctx = self.compute_context
            row_pix_dev = getattr(ctx, "row_pix_dev", None)
            if row_pix_dev is None or int(row_pix_dev.shape[0]) != n_active:
                row_pix_dev = self.backend.to_device(
                    np.asarray(self.map_inds, dtype=np.int64)
                )
                ctx.row_pix_dev = row_pix_dev
            return array[..., row_pix_dev]
        return np.take(array, self.map_inds, axis=-1)

    def _coerce_map_input_array(self, array: Any) -> Any:
        """Row-space view of a map input at map precision (host or device).

        Read-only host arrays (``arr.flags.writeable = False``) cannot change
        content, so their row-space device copy is memoised on identity:
        freeze fixed weight maps to gather and upload them only once.
        """
        if self._is_backend_native_array(array):
            return self._gather_rows(array).astype(self.map_dtype, copy=False)
        arr = np.asarray(array)
        frozen = (
            arr is array
            and arr.ndim >= 1
            and not arr.flags.writeable
            and int(arr.shape[-1]) in (self.n_active, self.npix)
        )
        if not frozen:
            return np.asarray(self._gather_rows(arr), dtype=self.map_dtype)

        ctx = self.compute_context
        memo = getattr(ctx, "frozen_map_memo", None)
        if memo is None:
            memo = ctx.frozen_map_memo = {}
        key = (
            id(arr),
            arr.__array_interface__["data"][0],
            arr.shape,
            arr.dtype.str,
            self.map_dtype.str,
        )
        entry = memo.get(key)
        if entry is None:
            rows = np.ascontiguousarray(self._gather_rows(arr), dtype=self.map_dtype)
            if self.backend.name == "cupy":
                rows = self._to_backend_array(rows, dtype=self.map_dtype)
            else:
                rows.flags.writeable = False
            # Bounded by bytes as well as by entries.  Sixteen entries is a
            # harmless count and a ruinous size: one frozen (4, 2, npix)
            # float32 shear map-set is 1.6 GB at nside 2048, so the entry
            # bound alone permits ~25 GB of VRAM for a cache whose stated
            # purpose is a handful of fixed weight maps.
            entry = (arr, rows, {})
            memo[key] = entry
            limit = self._frozen_map_memo_max_bytes()
            while len(memo) > 1 and (
                len(memo) > _FROZEN_MAP_MEMO_MAX_ENTRIES
                or _frozen_memo_bytes(memo) > limit
            ):
                memo.pop(next(iter(memo)))
        return entry[1]

    def _frozen_map_memo_max_bytes(self) -> float:
        """Byte budget for the frozen row-space memo.

        A tenth of an explicit ``memory_budget_gb`` when one is set --
        the memo is a convenience, not the working set -- and otherwise a
        fixed cap that still holds several nside-2048 weight map-sets.
        """
        budget = getattr(self, "memory_budget_gb", None)
        if budget is not None and np.isfinite(budget):
            return min(_FROZEN_MAP_MEMO_MAX_BYTES, 0.1 * float(budget) * 1e9)
        return float(_FROZEN_MAP_MEMO_MAX_BYTES)

    def release_device_memory(self) -> None:
        """Drop every cache this object holds on the device.

        The pair scratch is 32 B/pair and the frozen-map memo whole
        map-sets; both live for as long as the object does, which makes a
        long-lived ``Correlation`` hard to share a GPU with.  The pair
        geometry and the prepared device buffers are *not* touched -- the
        object still measures, it just rebuilds its scratch on the next
        call.
        """
        ctx = self.compute_context
        ctx.pair_scratch = None
        ctx.aperture_scratch = None
        ctx.fused_output_buffers = None
        memo = getattr(ctx, "frozen_map_memo", None)
        if memo is not None:
            memo.clear()
        ctx._xipm_sumofweights_cache = None
        ctx._xipm_sumofweights_cache_w_fingerprint = None
        pool = self.backend.get_memory_pool()
        if pool is not None:
            pool.free_all_blocks()

    def _map_to_device(self, array: Any) -> Any:
        """Row-space backend array at map precision for any accepted map input."""
        return self._to_backend_array(
            self._coerce_map_input_array(array), dtype=self.map_dtype
        )

    # ── Virtual rows (static treecode / coarse aperture level) ──────────────
    # Coarse cells are appended as extra rows behind the pixel rows.  A cell
    # row carries the weighted mean of its member pixels and the sum of
    # their weights, so the (unchanged) kernels reproduce
    #     W_I W_J g_I g_J = sum_i sum_j w_i w_j g_i g_j
    # exactly.  The degrade operator is a chain of child->parent sums (real,
    # all entries 1): pixels -> level 1 -> level 2 -> ...

    def _degrade_operators(self, use_cupy: bool) -> Dict[str, Any]:
        ctx = self.compute_context
        cache = getattr(ctx, "degrade_ops", None)
        if cache is None:
            cache = ctx.degrade_ops = {}
        kind = "cupy" if use_cupy else "numpy"
        ops = cache.get(kind)
        if ops is not None:
            return ops

        if use_cupy:
            import cupyx.scipy.sparse as sparse

            xp = self.backend.module
        else:
            import scipy.sparse as sparse

            xp = np
        dtype = self.acc_dtype

        def csr(indptr: np.ndarray, indices: np.ndarray, n_cols: int) -> Any:
            data = xp.ones(int(indices.size), dtype=dtype)
            return sparse.csr_matrix(
                (
                    data,
                    xp.asarray(np.asarray(indices, dtype=np.int32)),
                    xp.asarray(np.asarray(indptr, dtype=np.int32)),
                ),
                shape=(int(indptr.size - 1), int(n_cols)),
            )

        n_ap = self.n_aperture_cells
        ops = {"aperture": None, "chain": [], "ranges": []}
        ap_cells = self._aperture_cells()
        if ap_cells is not None:
            ops["aperture"] = csr(ap_cells[1], ap_cells[2], self.n_active)
        tree = self._treecode
        if tree is not None:
            lut = self._global_to_row_lut()
            starts = tree.level_starts(first=n_ap)
            n_src = self.n_active
            for level in range(tree.n_levels):
                indices = tree.child_indices[level]
                if level == 0 and lut is not None:
                    indices = lut[indices]
                    if indices.size and int(indices.min()) < 0:
                        raise ValueError(
                            "Treecode cells reference pixels outside the mask."
                        )
                ops["chain"].append(csr(tree.child_indptr[level], indices, n_src))
                ops["ranges"].append((int(starts[level]), int(starts[level + 1])))
                n_src = int(tree.cells_per_level[level])
        cache[kind] = ops
        return ops

    def _degrade_csr_device(self) -> Dict[str, Any]:
        """Device CSR children of every degrade level, for the fused kernel.

        The same arrays :meth:`_degrade_operators` wraps in sparse matrices,
        uploaded raw: ``(indptr, indices, n_cells)`` per level plus the
        appended-row range each level writes.
        """
        ctx = self.compute_context
        cached = getattr(ctx, "degrade_csr", None)
        if cached is not None:
            return cached
        xp = self.backend.module

        def dev(indptr: Any, indices: Any) -> Tuple[Any, Any, int]:
            return (
                xp.asarray(np.asarray(indptr, dtype=np.int64)),
                xp.asarray(np.asarray(indices, dtype=np.int32)),
                int(np.asarray(indptr).size - 1),
            )

        out: Dict[str, Any] = {"aperture": None, "chain": [], "ranges": []}
        ap_cells = self._aperture_cells()
        if ap_cells is not None:
            out["aperture"] = dev(ap_cells[1], ap_cells[2])
        tree = self._treecode
        if tree is not None:
            lut = self._global_to_row_lut()
            starts = tree.level_starts(first=self.n_aperture_cells)
            for level in range(tree.n_levels):
                indices = tree.child_indices[level]
                if level == 0 and lut is not None:
                    indices = lut[indices]
                    if indices.size and int(indices.min()) < 0:
                        raise ValueError(
                            "Treecode cells reference pixels outside the mask."
                        )
                out["chain"].append(dev(tree.child_indptr[level], indices))
                out["ranges"].append((int(starts[level]), int(starts[level + 1])))
        ctx.degrade_csr = out
        return out

    def _use_fused_degrade(self, weights: Any) -> bool:
        """Whether :meth:`_expand_rows_fused` applies.

        Only when the backend has the kernel *and* the maps already live on
        the device: with host inputs the sparse path keeps them on the host,
        which is what the aperture leaves want.
        """
        return self.backend.degrade_rows_kernel is not None and not isinstance(
            weights, np.ndarray
        )

    # Row-buffer layouts the fused degrade can write directly.  "soa" is
    # the historical one (one contiguous block per value); "interleaved"
    # is (n_lead, K, n_rows), i.e. a stacked shear map-set without the
    # stack; "aos" is (n_rows, n_lead, K), the layout the pair kernels
    # load, so the SoA -> AoS transpose in front of them disappears.
    _ROW_LAYOUTS = ("soa", "interleaved", "aos")

    def _row_buffers(
        self, xp: Any, n_lead: int, n_val: int, n_rows: int, layout: str
    ) -> Tuple[Any, _Desc, Any, _Desc]:
        """Allocate the row buffers of ``layout`` and their kernel descriptors.

        A descriptor is ``(array, base, lead_stride, comp_stride,
        row_stride)`` in elements; see the header of ``degrade_rows.cu``.
        """
        dt = self.map_dtype
        if layout == "aos":
            w_rows = xp.empty((n_rows, n_lead), dtype=dt)
            w_desc = (w_rows, 0, 1, 0, n_lead)
            if not n_val:
                return w_rows, w_desc, w_rows, w_desc
            v_rows = xp.empty((n_rows, n_lead, n_val), dtype=dt)
            return w_rows, w_desc, v_rows, (v_rows, 0, n_val, 1, n_lead * n_val)
        w_rows = xp.empty((n_lead, n_rows), dtype=dt)
        w_desc = (w_rows, 0, n_rows, 0, 1)
        if not n_val:
            return w_rows, w_desc, w_rows, w_desc
        if layout == "interleaved":
            v_rows = xp.empty((n_lead, n_val, n_rows), dtype=dt)
            return w_rows, w_desc, v_rows, (v_rows, 0, n_val * n_rows, n_rows, 1)
        v_rows = xp.empty((n_val * n_lead, n_rows), dtype=dt)
        return w_rows, w_desc, v_rows, (v_rows, 0, n_rows, n_lead * n_rows, 1)

    def _sign_scale_array(self, xp: Any, scales: Sequence[float], dtype: Any) -> Any:
        """Cached device array of per-component factors.

        Built once per (factors, dtype): uploading a two-element array per
        call cost more than the multiply it feeds.
        """
        key = (tuple(float(s) for s in scales), np.dtype(dtype).name,
               xp is np)
        cache = getattr(self, "_sign_scale_cache", None)
        if cache is None:
            cache = self._sign_scale_cache = {}
        arr = cache.get(key)
        if arr is None:
            arr = cache[key] = xp.asarray(list(key[0]), dtype=dtype)
        return arr

    @staticmethod
    def _scaled_copy(xp: Any, src: Any, dst: Any, scale: Any) -> None:
        """``dst[...] = scale * src`` in one pass, without a temporary.

        ``scale`` is +-1 (a sign flip), so the product is exact and the
        result is bit-identical to flipping the maps before or after the
        degrade -- the degrade is linear in the values.
        """
        if scale is None:
            dst[...] = src
        else:
            xp.multiply(src, scale, out=dst)

    def _expand_rows_fused(
        self,
        values: Sequence[Any],
        weights: Any,
        blocks: str,
        signs: Optional[Sequence[float]] = None,
        layout: str = "soa",
        stacked: Optional[Any] = None,
    ) -> Optional[Tuple[Any, Any]]:
        """One kernel per degrade level instead of the sparse chain.

        The pixel rows are written into the final row buffers first, so
        level 0 reads its children straight out of them; deeper levels
        accumulate over an ``(n_lead, n_appended)`` scratch at the
        accumulation dtype, exactly as the sparse chain does, and one
        finalize pass divides by the weight sum and scatters into the row
        buffers.  No ``(n_active, K * n_lead)`` temporary and no transpose.

        ``signs`` (one factor per value) is applied while the pixel rows
        are written, so a ``flip_g1``/``flip_g2`` costs nothing: the cell
        rows inherit it, because level 0 reads the already-signed pixel
        rows.  ``stacked`` is the same values as one ``(n_lead, K,
        n_active)`` array, which lets the "aos" layout fill its pixel rows
        with a single transposing copy.

        Returns ``(v_rows, w_rows)`` with the raw buffers of ``layout``, or
        ``None`` when the kernel is unavailable, so the caller falls back
        to the sparse path.
        """
        if not self._use_fused_degrade(weights):
            return None
        kern = self.backend.degrade_rows_kernel
        n_val = len(values)
        if n_val > 2:                      # the kernel is templated for 0..2
            return None

        xp = self.backend.module
        n_active, n_appended = self.n_active, self.n_appended
        n_rows = n_active + n_appended
        lead = tuple(int(n) for n in weights.shape[:-1])
        n_lead = int(np.prod(lead)) if lead else 1
        acc = self.acc_dtype

        w_rows, w_desc, v_rows, v_desc = self._row_buffers(
            xp, n_lead, n_val, n_rows, layout
        )
        self._fill_pixel_rows(
            xp, w_rows, v_rows, weights, values, stacked, signs,
            n_lead, n_val, n_active, layout,
        )

        w_app = xp.zeros((n_lead, n_appended), dtype=acc)
        v_app = (
            xp.zeros((n_val * n_lead, n_appended), dtype=acc) if n_val else w_app
        )

        def scratch(base: int) -> Tuple[Any, Any]:
            return (
                (w_app, base, n_appended, 0, 1),
                (v_app, base, n_appended, n_lead * n_appended, 1),
            )

        csr = self._degrade_csr_device()
        filled: List[Tuple[int, int]] = []

        def run(indptr: Any, indices: Any, n_cells: int, src: Tuple[Any, Any],
                dst_base: int, weighted: bool) -> bool:
            w_dst, v_dst = scratch(dst_base)
            return bool(
                kern.level(indptr, indices, src[0], src[1], w_dst, v_dst,
                           n_cells, n_lead, n_val, weighted)
            )

        pixel_src = (w_desc, v_desc)
        if blocks in ("aperture", "all") and csr["aperture"] is not None:
            indptr, indices, n_cells = csr["aperture"]
            if not run(indptr, indices, n_cells, pixel_src, 0, True):
                return None
            filled.append((0, self.n_aperture_cells))
        if blocks in ("pairs", "all"):
            for level, ((start, stop), level_csr) in enumerate(
                zip(csr["ranges"], csr["chain"])
            ):
                indptr, indices, n_cells = level_csr
                if level == 0:
                    src, weighted = pixel_src, True
                else:
                    src = scratch(csr["ranges"][level - 1][0])
                    weighted = False
                if not run(indptr, indices, n_cells, src, start, weighted):
                    return None
                filled.append((start, stop))

        # Blocks that were not filled must read back as zero, as they do on
        # the sparse path; when the filled ranges tile the whole appended
        # block (the usual case inside an expansion scope) that costs
        # nothing, and one finalize launch covers everything.
        filled.sort()
        contiguous = filled == [(0, n_appended)] or (
            bool(filled)
            and filled[0][0] == 0
            and filled[-1][1] == n_appended
            and all(a[1] == b[0] for a, b in zip(filled, filled[1:]))
        )
        if contiguous:
            filled = [(0, n_appended)]
        else:
            self._zero_appended_rows(w_rows, v_rows, n_active, n_val, layout)

        # Shift the destination descriptors past the pixel rows and write
        # the appended rows; `row_inner` keeps those writes coalesced.
        w_tail = (w_desc[0], n_active * w_desc[4], w_desc[2], 0, w_desc[4])
        v_tail = (v_desc[0], n_active * v_desc[4], v_desc[2], v_desc[3], v_desc[4])
        for lo, hi in filled:
            if not kern.finalize(
                w_app, v_app, n_appended, w_tail, v_tail,
                n_lead, lo, hi, n_val, row_inner=(layout != "aos"),
            ):
                return None

        return v_rows, w_rows

    def _fill_pixel_rows(
        self, xp: Any, w_rows: Any, v_rows: Any, weights: Any,
        values: Sequence[Any], stacked: Optional[Any],
        signs: Optional[Sequence[float]], n_lead: int, n_val: int,
        n_active: int, layout: str,
    ) -> None:
        """Copy the pixel rows (and any sign flip) into the row buffers."""
        w2 = weights.reshape(n_lead, n_active)
        if layout == "aos":
            w_rows[:n_active] = w2.T
        else:
            w_rows[:, :n_active] = w2
        if not n_val:
            return
        scales = None
        if signs is not None and any(float(s) != 1.0 for s in signs):
            scales = [None if float(s) == 1.0 else self.map_dtype.type(s)
                      for s in signs]
        if stacked is not None and layout in ("aos", "interleaved"):
            # One copy for the whole map-set instead of one per component,
            # reading the caller's contiguous (n_lead, K, n_active) array.
            scale = None
            if scales is not None:
                scale = self._sign_scale_array(
                    xp, [1.0 if s is None else s for s in scales],
                    self.map_dtype,
                )
            src = stacked.reshape(n_lead, n_val, n_active)
            if layout == "aos":
                # (n_lead, K, n_active) -> (n_active, n_lead, K)
                self._scaled_copy(
                    xp, xp.transpose(src, (2, 0, 1)), v_rows[:n_active], scale
                )
            else:
                self._scaled_copy(
                    xp, src, v_rows[:, :, :n_active],
                    None if scale is None else scale[:, None],
                )
            return
        if layout == "aos":
            for k, v in enumerate(values):
                self._scaled_copy(
                    xp, v.reshape(n_lead, n_active).T, v_rows[:n_active, :, k],
                    None if scales is None else scales[k],
                )
            return
        for k, v in enumerate(values):
            dst = (
                v_rows[:, k, :n_active]
                if layout == "interleaved"
                else v_rows[k * n_lead : (k + 1) * n_lead, :n_active]
            )
            self._scaled_copy(
                xp, v.reshape(n_lead, n_active), dst,
                None if scales is None else scales[k],
            )

    @staticmethod
    def _zero_appended_rows(
        w_rows: Any, v_rows: Any, n_active: int, n_val: int, layout: str
    ) -> None:
        if layout == "aos":
            w_rows[n_active:] = 0
            if n_val:
                v_rows[n_active:] = 0
            return
        w_rows[:, n_active:] = 0
        if n_val:
            if layout == "interleaved":
                v_rows[:, :, n_active:] = 0
            else:
                v_rows[:, n_active:] = 0

    def _append_block(self, X: Any, blocks: str, use_cupy: bool) -> Any:
        """Apply the degrade operators to ``X`` ``(n_active, C)`` and return
        the virtual-row block ``(n_appended, C)`` (unfilled rows are zero)."""
        xp = self.backend.module if use_cupy else np
        ops = self._degrade_operators(use_cupy)
        block = xp.zeros((self.n_appended, X.shape[1]), dtype=X.dtype)
        if blocks in ("aperture", "all") and ops["aperture"] is not None:
            block[: self.n_aperture_cells] = ops["aperture"] @ X
        if blocks in ("pairs", "all"):
            prev = X
            for (start, stop), op in zip(ops["ranges"], ops["chain"]):
                cur = op @ prev
                block[start:stop] = cur
                prev = cur
        return block

    def _expansion_scope(self, layout: str = "aos") -> Any:
        """Context manager: share virtual-row expansions between the leaf
        computations of one public call (e.g. aperture + 2PCF pass of
        ``get_full_tomo_shear``).  Inside the scope every expansion fills
        *all* blocks once and is memoised on the identity of its inputs;
        the memo is dropped on exit, so reused device buffers (e.g.
        ``MapLoader`` slots) can never produce a stale hit.

        ``layout`` is the row layout the pair leaves take, and exists so
        that they agree with the other leaves of the same call and the
        expansion really is shared.  A call whose aperture pass runs
        ``aperture_tomo.cu`` must pass ``"soa"``: that kernel walks the
        aperture discs, whose row ids are largely contiguous, so SoA
        coalesces almost perfectly and AoS puts consecutive pixels
        ``2*nz`` elements apart (measured: +0.26 ms on the A100, far more
        than the 0.08 ms transpose AoS would save).
        """
        owner = self

        class _Scope:
            def __enter__(self) -> None:
                owner._expansion_depth = getattr(owner, "_expansion_depth", 0) + 1
                if owner._expansion_depth == 1:
                    owner._expansion_memo = {}
                    owner._expansion_layout = layout

            def __exit__(self, *exc: Any) -> None:
                owner._expansion_depth -= 1
                if owner._expansion_depth == 0:
                    owner._expansion_memo = None
                    owner._expansion_layout = None

        return _Scope()

    def _pair_row_layout(self) -> str:
        """Row layout for the pair kernels: the AoS they load, unless the
        enclosing call needs its leaves to share an SoA expansion."""
        return getattr(self, "_expansion_layout", None) or "aos"

    def _weight_rows(self, weights: Any, w2: Any, blocks: str, use_cupy: bool) -> Any:
        """Virtual-row block of the weights ``(n_appended, n_lead)``.

        Cached for frozen (read-only host) weight maps, whose row-space
        device copy is memoised and immutable by contract: with fixed survey
        weights the weight rows are computed once, not once per map.
        """
        xp = self.backend.module if use_cupy else np
        frozen = None
        memo = getattr(self.compute_context, "frozen_map_memo", None)
        if memo:
            for entry in memo.values():
                if entry[1] is weights:
                    frozen = entry
                    break
        if frozen is not None and len(frozen) > 2 and frozen[2].get(blocks) is not None:
            return frozen[2][blocks]
        X = xp.ascontiguousarray(w2.T.astype(self.acc_dtype, copy=False))
        W = self._append_block(X, blocks, use_cupy)
        if frozen is not None and len(frozen) > 2:
            frozen[2][blocks] = W
        return W

    def _expand_rows(
        self,
        values: Sequence[Any],
        weights: Any,
        blocks: str = "all",
        signs: Optional[Sequence[float]] = None,
    ) -> Tuple[List[Any], Any]:
        """Append the virtual rows to row-space maps.

        Args:
            values: arrays with the same shape as ``weights``
                (``(..., n_active)``), weighted by ``weights``.
            weights: weight array ``(..., n_active)``.
            blocks: which virtual rows to fill: ``"pairs"`` (treecode
                cells), ``"aperture"`` (coarse aperture level) or ``"all"``.
                Rows that are not filled are zero and must not be used.
            signs: optional factor per value, applied on the way in.  The
                degrade is linear in the values, so scaling before it is
                bit-identical to scaling after -- and for a sign flip it is
                free, because the pixel rows are copied anyway.

        Returns ``(values_rows, weights_rows)`` with last axis ``n_rows``.
        At full resolution (no virtual rows) the inputs are returned
        untouched -- the default path pays nothing.
        """
        n_appended = self.n_appended
        scales = self._sign_scales(signs, len(values))
        if n_appended == 0 or (
            blocks == "aperture" and self.n_aperture_cells == 0
        ):
            # aperture indices address pixel rows; nothing to append
            return self._scaled_values(values, scales), weights
        n_active = self.n_active
        if int(weights.shape[-1]) != n_active:
            raise ValueError(
                f"expected row-space arrays with {n_active} rows; got {weights.shape}"
            )

        memo = getattr(self, "_expansion_memo", None)
        if memo is not None:
            blocks = "all"  # fill everything once, share between the leaves
            key = (tuple(id(v) for v in values), id(weights), scales)
            hit = memo.get(key)
            if hit is not None:
                return list(hit[0]), hit[1]

        lead = tuple(int(n) for n in weights.shape[:-1])
        n_lead = int(np.prod(lead)) if lead else 1
        n_rows = n_active + n_appended

        fused = self._expand_rows_fused(values, weights, blocks, signs=signs)
        if fused is not None:
            v_rows, w_rows = fused
            out = [
                v_rows[k * n_lead : (k + 1) * n_lead].reshape(lead + (n_rows,))
                for k in range(len(values))
            ]
            w_rows = w_rows.reshape(lead + (n_rows,))
            if memo is not None:
                memo[key] = (out, w_rows, values, weights)
            return list(out), w_rows
        use_cupy = not isinstance(weights, np.ndarray)
        xp = self.backend.module if use_cupy else np
        acc = self.acc_dtype

        # The sparse path is the fallback: scale the inputs up front (one
        # temporary per flipped component) rather than threading the sign
        # through the sparse products.
        values = self._scaled_values(values, scales)
        w2 = weights.reshape(n_lead, n_active)

        W = self._weight_rows(weights, w2, blocks, use_cupy)  # (n_appended, n_lead)
        nonzero = W != 0
        inv_W = nonzero / xp.where(nonzero, W, 1)

        w_rows = xp.empty((n_lead, n_rows), dtype=self.map_dtype)
        w_rows[:, :n_active] = w2
        w_rows[:, n_active:] = W.T
        out = []
        if values:
            # (n_active, K * n_lead), accumulated at the accumulation dtype
            X = xp.empty((n_active, len(values) * n_lead), dtype=acc)
            for k, v in enumerate(values):
                xp.multiply(
                    w2.T, v.reshape(n_lead, n_active).T,
                    out=X[:, k * n_lead : (k + 1) * n_lead],
                )
            block = self._append_block(X, blocks, use_cupy)
            del X
            for k, v in enumerate(values):
                rows = xp.empty((n_lead, n_rows), dtype=self.map_dtype)
                rows[:, :n_active] = v.reshape(n_lead, n_active)
                rows[:, n_active:] = (block[:, k * n_lead : (k + 1) * n_lead] * inv_W).T
                out.append(rows.reshape(lead + (n_rows,)))
        w_rows = w_rows.reshape(lead + (n_rows,))
        if memo is not None:
            # keep the inputs alive so their ids cannot be reused in the scope
            memo[key] = (out, w_rows, values, weights)
        return list(out), w_rows

    @staticmethod
    def _sign_scales(
        signs: Optional[Sequence[float]], n_val: int
    ) -> Optional[Tuple[float, ...]]:
        """Normalise ``signs`` to a hashable tuple, or ``None`` if trivial."""
        if signs is None:
            return None
        scales = tuple(float(s) for s in signs)
        if len(scales) != n_val:
            raise ValueError(
                f"signs must have one entry per value; got {len(scales)} for {n_val}"
            )
        return None if all(s == 1.0 for s in scales) else scales

    @staticmethod
    def _scaled_values(
        values: Sequence[Any], scales: Optional[Tuple[float, ...]]
    ) -> List[Any]:
        if scales is None:
            return list(values)
        return [v if s == 1.0 else s * v for v, s in zip(values, scales)]

    def _expand_shear_rows(
        self,
        shear: Any,
        weights: Any,
        blocks: str = "all",
        signs: Optional[Sequence[float]] = None,
        layout: str = "soa",
    ) -> Tuple[Any, Any]:
        """:meth:`_expand_rows` for planar shear ``(nz, 2, n_active)``.

        ``layout="soa"`` returns ``(nz, 2, n_rows)`` and ``(nz, n_rows)``;
        ``layout="aos"`` returns ``(n_rows, nz, 2)`` and ``(n_rows, nz)``,
        the layout the pair kernels load.  With the fused degrade the AoS
        buffers are written by the kernel itself, so neither the stack that
        used to glue g1 and g2 back together nor the SoA -> AoS transpose
        in front of the pair kernels is needed.
        """
        if layout not in ("soa", "aos"):
            raise ValueError(f"unknown row layout {layout!r}")
        use_cupy = not isinstance(weights, np.ndarray)
        xp = self.backend.module if use_cupy else np
        scales = self._sign_scales(signs, 2)
        if self.n_appended == 0 or (
            blocks == "aperture" and self.n_aperture_cells == 0
        ):
            # Nothing to append (aperture indices address pixel rows).  Not
            # memoised: the result would be wrong for a leaf that does want
            # the treecode rows.
            return self._shear_rows_no_append(xp, shear, weights, scales, layout)

        memo = getattr(self, "_expansion_memo", None)
        key = ("shear", id(shear), id(weights), scales, layout)
        if memo is not None:
            hit = memo.get(key)
            if hit is not None:
                return hit[0], hit[1]
            blocks = "all"  # fill everything once, share between the leaves

        fused = self._expand_rows_fused(
            (shear[:, 0], shear[:, 1]), weights, blocks, signs=signs,
            layout="aos" if layout == "aos" else "interleaved",
            stacked=shear,
        )
        if fused is not None:
            shear_rows, w_rows = fused
        else:
            (g1, g2), w_rows = self._expand_rows(
                (shear[:, 0], shear[:, 1]), weights, blocks=blocks, signs=signs
            )
            shear_rows = xp.stack((g1, g2), axis=1)
            if layout == "aos":
                shear_rows = xp.ascontiguousarray(
                    xp.transpose(shear_rows, (2, 0, 1))
                )
                w_rows = xp.ascontiguousarray(xp.transpose(w_rows, (1, 0)))
        if memo is not None:
            memo[key] = (shear_rows, w_rows, shear, weights)
        return shear_rows, w_rows

    def _expand_rows_aos(
        self,
        values: Sequence[Any],
        weights: Any,
        blocks: str = "all",
        signs: Optional[Sequence[float]] = None,
    ) -> Tuple[List[Any], Any]:
        """:meth:`_expand_rows` in the AoS layout the pair kernels load.

        Returns ``(values_rows, weights_rows)`` shaped ``(n_rows, n_lead)``
        each.  With the fused degrade the buffers are written in that
        layout directly, so the transpose that used to sit in front of
        every kernel launch disappears; otherwise this is the old
        expand-then-transpose, unchanged.
        """
        use_cupy = not isinstance(weights, np.ndarray)
        xp = self.backend.module if use_cupy else np
        scales = self._sign_scales(signs, len(values))

        def to_aos(vals: Sequence[Any], w: Any) -> Tuple[List[Any], Any]:
            return (
                [xp.ascontiguousarray(xp.transpose(v, (1, 0))) for v in vals],
                xp.ascontiguousarray(xp.transpose(w, (1, 0))),
            )

        if self.n_appended == 0 or (
            blocks == "aperture" and self.n_aperture_cells == 0
        ):
            return to_aos(self._scaled_values(values, scales), weights)

        memo = getattr(self, "_expansion_memo", None)
        key = ("aos", tuple(id(v) for v in values), id(weights), scales)
        if memo is not None:
            hit = memo.get(key)
            if hit is not None:
                return list(hit[0]), hit[1]
            blocks = "all"  # fill everything once, share between the leaves

        fused = None
        if len(values) <= 1:
            # With more than one value the per-component views would be
            # strided; the callers all pass a single field.
            fused = self._expand_rows_fused(
                values, weights, blocks, signs=signs, layout="aos"
            )
        if fused is not None:
            v_rows, w_rows = fused
            out = [v_rows.reshape(v_rows.shape[0], v_rows.shape[1])] if values else []
        else:
            vals, w_rows = self._expand_rows(
                values, weights, blocks=blocks, signs=signs
            )
            out, w_rows = to_aos(vals, w_rows)
        if memo is not None:
            memo[key] = (out, w_rows, values, weights)
        return list(out), w_rows

    # The pair kernels always load AoS; these two pick the cheaper route to
    # it.  Outside a scope (or in one whose other leaves are AoS too) the
    # degrade writes it directly; in an SoA scope the leaves share one SoA
    # expansion and this transposes, which is what the aperture kernels'
    # coalescing is worth.

    def _pair_shear_rows(
        self, shear: Any, weights: Any, blocks: str = "pairs",
        signs: Optional[Sequence[float]] = None,
    ) -> Tuple[Any, Any]:
        if self._pair_row_layout() == "aos":
            return self._expand_shear_rows(
                shear, weights, blocks, signs=signs, layout="aos"
            )
        rows, w_rows = self._expand_shear_rows(
            shear, weights, blocks, signs=signs
        )
        return self._transpose_tomo_inputs_aos(rows, w_rows)

    def _pair_value_rows(
        self, values: Sequence[Any], weights: Any, blocks: str = "pairs",
        signs: Optional[Sequence[float]] = None,
    ) -> Tuple[List[Any], Any]:
        if self._pair_row_layout() == "aos":
            return self._expand_rows_aos(values, weights, blocks, signs=signs)
        xp = self.backend.module if not isinstance(weights, np.ndarray) else np
        vals, w_rows = self._expand_rows(values, weights, blocks, signs=signs)
        return (
            [xp.ascontiguousarray(xp.transpose(v, (1, 0))) for v in vals],
            xp.ascontiguousarray(xp.transpose(w_rows, (1, 0))),
        )

    def _shear_rows_no_append(
        self, xp: Any, shear: Any, weights: Any,
        scales: Optional[Tuple[float, ...]], layout: str,
    ) -> Tuple[Any, Any]:
        """Full-resolution shear rows: no virtual rows to append.

        Any sign flip and the AoS transpose collapse into the single copy
        the layout needs (and into nothing at all for plain SoA input).
        """
        if layout == "soa":
            if scales is None:
                return shear, weights
            out = xp.empty_like(shear)
            for k, s in enumerate(scales):
                self._scaled_copy(
                    xp, shear[:, k], out[:, k],
                    None if s == 1.0 else self.map_dtype.type(s),
                )
            return out, weights
        nz = int(shear.shape[0])
        out = xp.empty((int(shear.shape[2]), nz, 2), dtype=shear.dtype)
        scale = None
        if scales is not None:
            scale = self._sign_scale_array(xp, scales, shear.dtype)
        self._scaled_copy(xp, xp.transpose(shear, (2, 0, 1)), out, scale)
        return out, xp.ascontiguousarray(xp.transpose(weights, (1, 0)))

    @property
    def _index_device_dtype(self) -> Any:
        """Backend-native dtype matching self.index_dtype (e.g. cupy.int32 or numpy.int64)."""
        return getattr(self.backend.module, self.index_dtype.name)

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
            # Deliberately *not* the pool pointer.  cupy's allocator hands a
            # freed pointer straight back to the next allocation, so a plain
            # ``for k: w = cp.asarray(w_host[k])`` loop gives the second map
            # the first one's address — and, with a pointer-keyed cache, the
            # first one's sum of weights.  A per-live-object serial cannot be
            # recycled while the array it names is still alive, so a hit means
            # the same array rather than the same address.
            return (
                shape,
                dtype_str,
                f"device:{device_id};obj:{live_object_serial(w_np)}",
            )

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
            # The weak reference is what makes the id safe to key on: an
            # array built and dropped once per realisation can land on the
            # recycled address of the previous one, and would otherwise
            # inherit its digest.  Holding the reference costs nothing and
            # pins nothing.
            if cached is not None and cached[0]() is w_np:
                return cached[1]
        # blake2b accepts buffer-protocol objects: hashing a memoryview gives
        # the same digest as .tobytes() without the full byte copy.
        digest = hashlib.blake2b(memoryview(w_contiguous).cast("B")).hexdigest()
        result = (w_contiguous.shape, w_contiguous.dtype.str, digest)
        if memo_key is not None:
            if len(_FINGERPRINT_MEMO) > 64:
                _FINGERPRINT_MEMO.clear()
            _FINGERPRINT_MEMO[memo_key] = (weakref.ref(w_np), result)
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
        self._ensure_prepared()
        self._require_unpacked_pairs("Precomputing sums of weights")

        product = w1_dev[self.inds_dev[0]] * w2_dev[self.inds_dev[1]]
        return self._reduce_pairs(product.astype(self.acc_dtype, copy=False))

    def _get_xipm_sumofweights(
        self,
        w1_dev: Any,
        w2_dev: Any,
        w1_rows: Optional[Any] = None,
        w2_rows: Optional[Any] = None,
    ) -> Any:
        """Cached pair weight sums.

        ``w*_dev`` identify the weight maps (cache key); ``w*_rows`` are the
        same maps with the virtual rows appended (default: ``w*_dev``).
        """
        if w1_rows is None:
            w1_rows = w1_dev
        if w2_rows is None:
            w2_rows = w2_dev
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

        sumofweights_dev = self._compute_xipm_sumofweights(w1_rows, w2_rows)
        cache[w_fingerprint] = sumofweights_dev
        # Bound the cache: with per-map weights every map adds entries, which
        # would otherwise accumulate device arrays without limit.
        while len(cache) > 32:
            cache.pop(next(iter(cache)))
        return sumofweights_dev

    def compute_sumofweights(
        self, weights: Any, nzbins: Optional[int] = None
    ) -> Any:
        """Precompute the pair weight sums the ``sumofweights=`` argument takes.

        Every tomographic method accepts ``sumofweights=`` so that a weight
        set that does not change between realisations is reduced once
        instead of once per map.  This is what builds a valid value: the
        directional ``(2, ncomb, n_patches * nbins)`` array, orientation 0
        being A→B and orientation 1 B→A (identical for auto combinations).

        Needs the unpacked geometry, so ``pack_pairs=False`` on a GPU.

        Args:
            weights: ``(nzbins, n_active)`` or ``(nzbins, npix)`` weight maps.
            nzbins: defaults to ``weights.shape[0]``.
        """
        if nzbins is None:
            nzbins = int(weights.shape[0])
        nzbin_combs = nzbins * (nzbins + 1) // 2
        return self._compute_tomo_sumofweights(
            self._map_to_device(weights), nzbins, nzbin_combs
        )

    def _compute_tomo_sumofweights(
        self, w_dev: Any, nzbins: int, nzbin_combs: int
    ) -> Any:
        self._ensure_prepared()
        self._require_unpacked_pairs("Precomputing sums of weights")

        if int(w_dev.shape[-1]) == self.n_active:
            _, w_dev = self._expand_rows((), w_dev, "pairs")

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

    def _symmetrised_denominators(
        self, out_den: Any, sumofweights_dev: Any, auto_comb: np.ndarray
    ) -> Any:
        """Per-combination weight sums ``(ncomb, nbins_total)``.

        From the kernel (both orientations of a cross combination already
        summed) or from an explicit directional ``(2, ncomb, nbins_total)``
        array, whose orientations are summed here for cross combinations
        (auto combinations carry the same sum twice).
        """
        if sumofweights_dev is None:
            return out_den
        direct = sumofweights_dev[0]
        cross = ~np.asarray(auto_comb, dtype=bool)[: direct.shape[0]]
        if not cross.any():
            return direct.copy()
        module = self.backend.module
        # one set of launches for all combinations (was one per combination)
        keep = module.asarray(cross.reshape((-1,) + (1,) * (direct.ndim - 1)))
        return module.where(keep, direct + sumofweights_dev[1], direct)

    def _finish_xipm_tomo(
        self,
        out_num: Any,
        out_den: Any,
        sumofweights_dev: Any,
        auto_comb: np.ndarray,
        nzbin_combs: int,
    ) -> Tuple[Any, Any]:
        """Normalise the ``(2, ncomb, nbins_total)`` numerators of the
        tomographic ξ± kernels (ratio of the orientation sums)."""
        den = self._symmetrised_denominators(out_den, sumofweights_dev, auto_comb)
        map_backend_dtype = getattr(self.backend.module, self.map_dtype.name)
        # Normalise both orientations of every combination in one go: the
        # per-combination loop issued ~12 tiny kernels per combination, which
        # dominated the per-call launch latency at nside 512.
        shape = (nzbin_combs, self.n_patches, self.nbins)
        den = self._align_denominator(out_num[0], den)
        xip = self._normalize_by_weights(out_num[0], den)
        xim = self._normalize_by_weights(out_num[1], den)
        return (
            xip.reshape(shape).astype(map_backend_dtype, copy=False),
            xim.reshape(shape).astype(map_backend_dtype, copy=False),
        )

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
            shear_maps_dev, w_dev = self._expand_shear_rows(
                shear_maps_dev, w_dev, blocks="pairs", signs=(g1_fac, g2_fac)
            )
            return self._xipm_tomo_vectorized_cpu(
                shear_maps_dev,
                w_dev,
                sumofweights_dev,
                nzbins,
                nzbin_combs,
            )

        if self.backend.name != "cupy":
            return None

        module = self.backend.module
        # The expansion writes the kernel's AoS layout itself and folds any
        # sign flip into the copy it makes anyway: no stack, no transpose.
        shear_aos, weights_aos = self._pair_shear_rows(
            shear_maps_dev, w_dev, "pairs", signs=(g1_fac, g2_fac)
        )

        bin_offsets = module.ascontiguousarray(
            self.tot_bins_reduceat_dev.astype(module.int64, copy=False)
        )
        comb_i, comb_j, auto_comb = self._get_tomo_combination_indices(
            nzbins, nzbin_combs
        )
        acc_backend_dtype = getattr(module, self.acc_dtype.name)
        nbins_total = int(self.n_patches * self.nbins)
        # Allocated at the accumulation dtype: the kernel reduces into them.
        out_num = self.backend.zeros((2, nzbin_combs, nbins_total), dtype=acc_backend_dtype)
        out_den = self.backend.zeros((nzbin_combs, nbins_total), dtype=acc_backend_dtype)

        ctx = self.compute_context
        if ctx.packed_pairs_dev is not None:
            packed_kernel = getattr(self.backend, "xipm_tomo_packed_kernel", None)
            # Gather every patch's rows into its contiguous block.
            perm = ctx.packed_perm_dev
            launched = packed_kernel is not None and packed_kernel(
                module.ascontiguousarray(shear_aos[perm]),
                module.ascontiguousarray(weights_aos[perm]),
                ctx.packed_pairs_dev,
                bin_offsets,
                ctx.packed_row_base_dev,
                comb_i,
                comb_j,
                out_num,
                out_den,
            )
            if not launched:
                raise RuntimeError(
                    self._packed_kernel_unavailable_message("ξ±", nzbins, 3)
                )
        else:
            inds_i, inds_j = self._pair_index_arrays()
            launched = tomo_kernel(
                shear_aos,
                weights_aos,
                inds_i,
                inds_j,
                module.ascontiguousarray(self.exp2phi_dev[0]),
                module.ascontiguousarray(self.exp2phi_dev[1]),
                bin_offsets,
                comb_i,
                comb_j,
                out_num,
                out_den,
            )
            if not launched:
                return None

        return self._finish_xipm_tomo(out_num, out_den, sumofweights_dev, auto_comb, nzbin_combs)

    def _xipm_tomo_vectorized_cpu(
        self,
        shear_maps_dev: Any,
        w_dev: Any,
        sumofweights_dev: Any,
        nzbins: int,
        nzbin_combs: int,
    ) -> Tuple[Any, Any]:
        # Any sign flip has already been folded into the expansion.
        shear_aos, weights_aos = self._transpose_tomo_inputs_aos(
            shear_maps_dev, w_dev
        )

        nbins_total = int(self.tot_bins_reduceat_dev.shape[0] - 1)
        # The numba kernel inherits its accumulator dtype from these arrays.
        out_num = np.empty((2, nzbin_combs, nbins_total), dtype=self.acc_dtype)
        out_w = np.empty((nzbin_combs, nbins_total), dtype=self.acc_dtype)
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
            out_num[0],
            out_num[1],
            out_w,
        )
        if launched is False:
            raise RuntimeError(
                "Backend tomography vectorized kernel unavailable for CPU backend."
            )
        return self._finish_xipm_tomo(out_num, out_w, sumofweights_dev, auto_comb, nzbin_combs)

    def vectorized_shear_shear(
        self,
        shear_maps: np.ndarray,
        w: np.ndarray,
        sumofweights: Optional[np.ndarray] = None,
        flip_g1: bool = False,
        flip_g2: bool = False,
        return_device: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute tomographic shear 2PCFs xi+/xi-."""
        self._ensure_prepared()

        nzbins = shear_maps.shape[0]
        nzbin_combs = int(binom(nzbins + 1, 2))

        shear_maps_arr = self._coerce_map_input_array(shear_maps)
        w_arr = self._coerce_map_input_array(w)
        shear_maps_dev = self._to_backend_array(shear_maps_arr, dtype=self.map_dtype)
        w_dev = self._to_backend_array(w_arr, dtype=self.map_dtype)

        if sumofweights is None:
            # Both backends: the vectorized kernel accumulates the weight
            # sums (denominators) in the same pass over the pairs, so no
            # separate reduction or fingerprint/cache machinery is needed.
            sumofweights_dev = None
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

        kernel = getattr(self.backend, "aperture_tomo_shear_kernel", None)
        if kernel is not None and self.backend.name == "cupy":
            module = self.backend.module
            map_backend_dtype = getattr(module, self.map_dtype.name)
            if self.compute_context.Q_inds_dev is None:
                self._prepare_aperture_device_buffers()
            ctx = self.compute_context
            shear_dev = self._to_backend_array(shear_maps_arr, dtype=self.map_dtype)
            w_dev = self._to_backend_array(w_arr, dtype=self.map_dtype)
            # SoA: this kernel gathers aperture discs, whose row ids are
            # largely contiguous.  The sign flip still rides along in the
            # copy the expansion makes anyway.
            shear_rows, w_rows = self._expand_shear_rows(
                shear_dev, w_dev, blocks="aperture", signs=(g1_fac, g2_fac)
            )
            out_num = self.backend.zeros(
                (nzbins, self.n_patches), dtype=map_backend_dtype
            )
            out_den = self.backend.zeros(
                (nzbins, self.n_patches), dtype=map_backend_dtype
            )
            launched = kernel(
                shear_rows[:, 0],  # views; both strides passed explicitly
                shear_rows[:, 1],
                w_rows,
                ctx.Q_inds_dev,
                ctx.Q_cos_dev,
                ctx.Q_sin_dev,
                ctx.Q_val_dev,
                ctx.Q_offsets_dev,
                ctx.Q_patch_area_dev,
                out_num,
                out_den,
            )
            if launched:
                # Numerator is already area-scaled inside the kernel.
                M_a = self._normalize_by_weights(out_num, out_den)
                if return_device:
                    return M_a
                return np.asarray(self.backend.to_numpy(M_a), dtype=self.map_dtype)

        keep_on_device = return_device and self.backend.name == "cupy"
        if keep_on_device:
            module = self.backend.module
            map_backend_dtype = getattr(module, self.map_dtype.name)
            M_a = module.zeros([nzbins, self.n_patches], dtype=map_backend_dtype)
        else:
            M_a = np.zeros([nzbins, self.n_patches], dtype=self.map_dtype)
        # Append the virtual rows once for all bins (shared with the 2PCF
        # pass inside an expansion scope), then run the per-bin leaf.
        shear_rows, w_rows = self._expand_shear_rows(
            shear_maps_arr, w_arr, blocks="aperture", signs=(g1_fac, g2_fac)
        )
        for i in range(nzbins):
            M_a[i] = self._aperture_shear_rows(
                shear_rows[i, 0], shear_rows[i, 1], w_rows[i],
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

        kernel = getattr(self.backend, "aperture_tomo_density_kernel", None)
        if kernel is not None and self.backend.name == "cupy":
            module = self.backend.module
            map_backend_dtype = getattr(module, self.map_dtype.name)
            if self.compute_context.Q_inds_dev is None:
                self._prepare_aperture_device_buffers()
            ctx = self.compute_context
            nzbins = int(density_arr.shape[0])
            density_dev = self._to_backend_array(density_arr, dtype=self.map_dtype)
            w_dev = self._to_backend_array(w_arr, dtype=self.map_dtype)
            (density_rows,), w_rows = self._expand_rows(
                (density_dev,), w_dev, blocks="aperture"
            )
            out_num = self.backend.zeros(
                (nzbins, self.n_patches), dtype=map_backend_dtype
            )
            out_den = self.backend.zeros(
                (nzbins, self.n_patches), dtype=map_backend_dtype
            )
            launched = kernel(
                density_rows,
                w_rows,
                ctx.Q_inds_dev,
                ctx.Q_val_dev,
                ctx.Q_offsets_dev,
                ctx.Q_patch_area_dev,
                out_num,
                out_den,
            )
            if launched:
                # Numerator is already area-scaled inside the kernel.
                M_g = self._normalize_by_weights(out_num, out_den)
                if return_device:
                    return M_g
                return np.asarray(self.backend.to_numpy(M_g), dtype=self.map_dtype)

        keep_on_device = return_device and self.backend.name == "cupy"
        if keep_on_device:
            module = self.backend.module
            map_backend_dtype = getattr(module, self.map_dtype.name)
            M_g = module.zeros((density_arr.shape[0], self.n_patches), dtype=map_backend_dtype)
        else:
            M_g = np.zeros((density_arr.shape[0], self.n_patches), dtype=self.map_dtype)
        (density_rows,), w_rows = self._expand_rows(
            (density_arr,), w_arr, blocks="aperture"
        )
        for i in range(density_arr.shape[0]):
            M_g[i] = self._aperture_density_rows(
                density_rows[i], w_rows[i], return_device=keep_on_device
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
        with self._expansion_scope(layout="soa"):  # one degrade, both passes
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
        with self._expansion_scope(layout="soa"):  # one degrade, both passes
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

        # Share one row expansion between the passes, as the other
        # ``get_full_tomo_*`` methods do.  Without this the density and shear
        # rows are degraded once for the pair pass and again for each
        # aperture pass -- four expansions where two suffice.  The scope is
        # taken only when an aperture output is requested, because it is the
        # ``aperture_tomo.cu`` leaf that fixes the layout at ``"soa"``; a
        # xi_t-only call keeps its direct AoS write.
        want_aperture = return_N_ap or return_M_ap
        scope = (
            self._expansion_scope(layout="soa") if want_aperture else nullcontext()
        )
        with scope:
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
            )

            if not want_aperture:
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
    ) -> Any:
        self._ensure_prepared()

        tomo_kernel = getattr(self.backend, "kernel_density_density_tomo_vectorized", None)
        if tomo_kernel is None:
            raise RuntimeError(
                "Backend does not provide a vectorized density-density tomography kernel."
            )

        module = self.backend.module
        map_backend_dtype = getattr(module, self.map_dtype.name)
        acc_backend_dtype = getattr(module, self.acc_dtype.name)
        nbins_total = self.n_patches * self.nbins

        density_dev = self._map_to_device(density_maps)
        w_dev = self._map_to_device(weights)
        # Expanded straight into the kernels' AoS layout (no transpose).
        (density_soa,), w_soa = self._pair_value_rows(
            (density_dev,), w_dev, "pairs"
        )

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
        else:
            # Both backends: the kernel accumulates the denominators itself.
            sumofweights_dev = None

        bin_offsets = module.ascontiguousarray(
            self.tot_bins_reduceat_dev.astype(module.int64, copy=False)
        )

        # Kernel contract (both backends): one row per combination, both
        # orientations of a cross combination summed.
        if self.backend.name == "numpy":
            # The numba kernel inherits its accumulator dtype from these arrays.
            out_num = np.empty((nzbin_combs, nbins_total), dtype=self.acc_dtype)
            out_den = np.empty((nzbin_combs, nbins_total), dtype=self.acc_dtype)
            tomo_kernel(
                density_soa,
                w_soa,
                np.ascontiguousarray(self.inds_dev[0]),
                np.ascontiguousarray(self.inds_dev[1]),
                np.asarray(bin_offsets, dtype=np.int64),
                np.ascontiguousarray(comb_i),
                np.ascontiguousarray(comb_j),
                out_num,
                out_den,
            )
        else:
            # Allocated at the accumulation dtype: the kernel reduces into them.
            out_num = self.backend.zeros((nzbin_combs, nbins_total), dtype=acc_backend_dtype)
            out_den = self.backend.zeros((nzbin_combs, nbins_total), dtype=acc_backend_dtype)
            ctx = self.compute_context
            if ctx.packed_pairs_dev is not None:
                packed_kernel = getattr(self.backend, "kernel_density_density_tomo_packed", None)
                perm = ctx.packed_perm_dev
                launched = packed_kernel is not None and packed_kernel(
                    module.ascontiguousarray(density_soa[perm]),
                    module.ascontiguousarray(w_soa[perm]),
                    ctx.packed_pairs_dev,
                    bin_offsets,
                    ctx.packed_row_base_dev,
                    comb_i,
                    comb_j,
                    out_num,
                    out_den,
                )
                if not launched:
                    raise RuntimeError(
                        "The packed density-density kernel is unavailable for this "
                        f"configuration ({nzbins} tomographic bins); use pack_pairs=False."
                    )
            else:
                inds_i, inds_j = self._pair_index_arrays()
                launched = tomo_kernel(
                    density_soa,
                    w_soa,
                    inds_i,
                    inds_j,
                    bin_offsets,
                    comb_i,
                    comb_j,
                    out_num,
                    out_den,
                )
                if not launched:
                    raise RuntimeError(
                        self._packed_kernel_unavailable_message("ξ_g", nzbins, 2)
                        if self.inds_dev is None
                        else "Backend declined vectorized density-density "
                        "tomography kernel launch."
                    )

        den = self._symmetrised_denominators(out_den, sumofweights_dev, auto_comb)
        shape = (nzbin_combs, self.n_patches, self.nbins)
        return (
            self._normalize_by_weights(out_num, self._align_denominator(out_num, den))
            .reshape(shape)
            .astype(map_backend_dtype, copy=False)
        )

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
        shear_signs: Optional[Sequence[float]] = None,
    ) -> Any:
        self._ensure_prepared()

        tomo_kernel = getattr(self.backend, "kernel_density_shear_tomo_vectorized", None)
        if tomo_kernel is None:
            raise RuntimeError(
                "Backend does not provide a vectorized density-shear tomography kernel."
            )

        module = self.backend.module
        map_backend_dtype = getattr(module, self.map_dtype.name)
        acc_backend_dtype = getattr(module, self.acc_dtype.name)
        nbins_total = self.n_patches * self.nbins

        density_dev = self._map_to_device(density_maps)
        shear_dev = self._map_to_device(shear_maps)
        density_w_dev = self._map_to_device(density_w)
        shear_w_dev = self._map_to_device(shear_w)
        # Expanded straight into the kernels' AoS layout (no transpose),
        # with any sign flip folded into the copy the expansion makes.
        (density_soa,), density_w_soa = self._pair_value_rows(
            (density_dev,), density_w_dev, "pairs"
        )
        shear_soa, shear_w_soa = self._pair_shear_rows(
            shear_dev, shear_w_dev, "pairs", signs=shear_signs
        )

        comb_i_base, comb_j_base, nzbin_combs = (
            self._get_selected_tomo_cross_combination_indices(
                nlens_bins,
                nsource_bins,
                ggl_bin_combinations=ggl_bin_combinations,
            )
        )
        comb_i = module.ascontiguousarray(comb_i_base)
        comb_j = module.ascontiguousarray(comb_j_base)

        bin_offsets = module.ascontiguousarray(
            self.tot_bins_reduceat_dev.astype(module.int64, copy=False)
        )

        if self.backend.name == "numpy":
            # The numba kernel inherits its accumulator dtype from these arrays.
            out_num = np.empty((nzbin_combs, nbins_total), dtype=self.acc_dtype)
            out_den = np.empty((nzbin_combs, nbins_total), dtype=self.acc_dtype)
            tomo_kernel(
                density_soa,
                shear_soa,
                density_w_soa,
                shear_w_soa,
                np.ascontiguousarray(self.inds_dev[0]),
                np.ascontiguousarray(self.inds_dev[1]),
                np.ascontiguousarray(self.exp2phi_dev[0]),
                np.ascontiguousarray(self.exp2phi_dev[1]),
                np.asarray(bin_offsets, dtype=np.int64),
                np.ascontiguousarray(comb_i),
                np.ascontiguousarray(comb_j),
                out_num,
                out_den,
            )
        else:
            # Allocated at the accumulation dtype: the kernel reduces into them.
            out_num = self.backend.zeros((nzbin_combs, nbins_total), dtype=acc_backend_dtype)
            out_den = self.backend.zeros((nzbin_combs, nbins_total), dtype=acc_backend_dtype)
            ctx = self.compute_context
            if ctx.packed_pairs_dev is not None:
                packed_kernel = getattr(self.backend, "kernel_density_shear_tomo_packed", None)
                perm = ctx.packed_perm_dev
                launched = packed_kernel is not None and packed_kernel(
                    module.ascontiguousarray(density_soa[perm]),
                    module.ascontiguousarray(shear_soa[perm]),
                    module.ascontiguousarray(density_w_soa[perm]),
                    module.ascontiguousarray(shear_w_soa[perm]),
                    ctx.packed_pairs_dev,
                    bin_offsets,
                    ctx.packed_row_base_dev,
                    comb_i,
                    comb_j,
                    out_num,
                    out_den,
                )
                if not launched:
                    raise RuntimeError(
                        "The packed density-shear kernel is unavailable for this "
                        f"configuration ({nlens_bins} lens x {nsource_bins} source "
                        "bins); use pack_pairs=False."
                    )
            else:
                inds_i, inds_j = self._pair_index_arrays()
                launched = tomo_kernel(
                    density_soa,
                    shear_soa,
                    density_w_soa,
                    shear_w_soa,
                    inds_i,
                    inds_j,
                    module.ascontiguousarray(self.exp2phi_dev[0]),
                    module.ascontiguousarray(self.exp2phi_dev[1]),
                    bin_offsets,
                    comb_i,
                    comb_j,
                    out_num,
                    out_den,
                )
                if not launched:
                    raise RuntimeError(
                        self._packed_kernel_unavailable_message("ξ_t", nzbins, 2)
                        if self.inds_dev is None
                        else "Backend declined vectorized density-shear "
                        "tomography kernel launch."
                    )

        num_ab = out_num

        if sumofweights is None:
            # In-kernel denominator: already the A→B + B→A weight sum
            # (both backends).
            sum_total = out_den
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

        shape = (nzbin_combs, self.n_patches, self.nbins)
        return (
            self._normalize_by_weights(
                num_ab, self._align_denominator(num_ab, sum_total)
            )
            .reshape(shape)
            .astype(map_backend_dtype, copy=False)
        )

    def vectorized_density_density(
        self,
        density_maps: np.ndarray,
        w: np.ndarray,
        sumofweights: Optional[np.ndarray] = None,
        gc_auto_correlations_only: bool = False,
        return_device: bool = True,
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

        # The sign flip rides along in the copy the row expansion makes
        # anyway, instead of a separate full-map copy here.
        shear_signs = (-1.0 if flip_g1 else 1.0, -1.0 if flip_g2 else 1.0)

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
            shear_signs=shear_signs,
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

        # Folded into the copy the row expansion makes anyway.
        shear_signs = (-1.0 if flip_g1 else 1.0, -1.0 if flip_g2 else 1.0)

        self._ensure_prepared()
        self._ensure_aperture_pairs(aperture_filter=aperture_filter)

        module = self.backend.module
        map_backend_dtype = getattr(module, self.map_dtype.name)
        on_gpu = self.backend.name == "cupy"
        if on_gpu:
            aperture_kernel = getattr(self.backend, "kernel_3x2pt_tomo_aperture", None)
            if aperture_kernel is None:
                raise RuntimeError("Backend does not provide the fused 3x2pt aperture kernel.")
        else:
            fused_kernel = getattr(self.backend, "kernel_3x2pt_tomo_fused", None)
            if fused_kernel is None:
                raise RuntimeError("Backend does not provide a fused 3x2pt tomography kernel.")

        density_dev = self._to_backend_array(density_np, dtype=self.map_dtype)
        shear_dev = self._to_backend_array(shear_np, dtype=self.map_dtype)
        density_w_dev = self._to_backend_array(density_w_np, dtype=self.map_dtype)
        shear_w_dev = self._to_backend_array(shear_w_np, dtype=self.map_dtype)
        # The row expansion writes the kernels' AoS layout directly, so the
        # four transposing copies into separate fused input buffers are
        # gone -- as is the sign flip, which rides along in the same copy.
        (density_soa,), density_w_soa = self._pair_value_rows(
            (density_dev,), density_w_dev, "all"
        )
        shear_soa, shear_w_soa = self._pair_shear_rows(
            shear_dev, shear_w_dev, "all", signs=shear_signs
        )

        n_shear_bins = int(shear_np.shape[0])
        n_density_bins = int(density_np.shape[0])
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
            out_xipm_num,
            out_xipm_den,
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
            # num/den buffers carry the accumulation dtype; the final
            # normalized results are still produced at map precision.
            map_backend_dtype=getattr(module, self.acc_dtype.name),
        )

        if not on_gpu:
            fused_kernel(
                np.ascontiguousarray(density_soa),
                np.ascontiguousarray(shear_soa),
                np.ascontiguousarray(density_w_soa),
                np.ascontiguousarray(shear_w_soa),
                np.ascontiguousarray(self.inds_dev[0]),
                np.ascontiguousarray(self.inds_dev[1]),
                np.ascontiguousarray(self.exp2phi_dev[0]),
                np.ascontiguousarray(self.exp2phi_dev[1]),
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
                out_xipm_num[0],
                out_xipm_num[1],
                out_xipm_den,
                out_xig_num,
                out_xig_den,
                out_xit_num,
                out_xit_den,
            )
        else:
            # Aperture sections (ACC accumulation) on the AoS inputs, then the
            # three pair statistics in the combination-tiled pair kernels --
            # the same kernels as the standalone tomographic methods, so the
            # packed geometry is supported here too.
            launched = aperture_kernel(
                density_soa,
                shear_soa,
                density_w_soa,
                shear_w_soa,
                q_inds,
                q_cos,
                q_sin,
                q_val,
                q_offsets,
                q_patch_area,
                out_ma_num,
                out_ma_den,
                out_mg_num,
                out_mg_den,
            )
            if not launched:
                raise RuntimeError("Backend declined fused 3x2pt aperture kernel launch.")
            self._launch_3x2pt_pair_kernels(
                density_soa, shear_soa, density_w_soa, shear_w_soa, pair_offsets,
                ss_comb_i, ss_comb_j, dd_comb_i, dd_comb_j, ds_comb_i, ds_comb_j,
                out_xipm_num, out_xipm_den, out_xig_num, out_xig_den, out_xit_num, out_xit_den,
            )

        def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
            out = np.zeros_like(num, dtype=self.map_dtype)
            valid = den != 0
            out[valid] = num[valid] / den[valid]
            return out

        def _safe_div_to_device(num: Any, den: Any) -> Any:
            """One fused launch into a *fresh* array at map precision.

            The result is what the caller keeps, so it must not be a buffer
            this call will overwrite on the next one -- see the note on
            aliasing in :meth:`get_3x2pt_tomo`.  Allocating it costs a pool
            hand-out rather than a copy, and the pool recycles the block as
            soon as the caller drops the result, so a loop that reduces and
            discards holds no more device memory than the cache did.
            """
            return safe_divide(module, num, den, dtype=map_backend_dtype)

        shape_pb = (n_patches, self.nbins)
        if return_device and on_gpu:
            # whole stacks at once: one launch per statistic
            xipm_den = out_xipm_den.reshape((ss_ncomb,) + shape_pb)
            return (
                _safe_div_to_device(out_ma_num, out_ma_den),
                _safe_div_to_device(out_mg_num, out_mg_den),
                _safe_div_to_device(
                    out_xipm_num[0].reshape((ss_ncomb,) + shape_pb), xipm_den
                ),
                _safe_div_to_device(
                    out_xipm_num[1].reshape((ss_ncomb,) + shape_pb), xipm_den
                ),
                _safe_div_to_device(
                    out_xig_num.reshape((dd_ncomb,) + shape_pb),
                    out_xig_den.reshape((dd_ncomb,) + shape_pb),
                ),
                _safe_div_to_device(
                    out_xit_num.reshape((ds_ncomb,) + shape_pb),
                    out_xit_den.reshape((ds_ncomb,) + shape_pb),
                ),
            )

        to_np = lambda arr: np.asarray(self.backend.to_numpy(arr), dtype=self.map_dtype)
        M_a = _safe_div(to_np(out_ma_num), to_np(out_ma_den))
        M_g = _safe_div(to_np(out_mg_num), to_np(out_mg_den))

        xipm_num_np = to_np(out_xipm_num)
        xipm_den_np = to_np(out_xipm_den)
        xip = np.zeros((ss_ncomb, n_patches, self.nbins), dtype=self.map_dtype)
        xim = np.zeros((ss_ncomb, n_patches, self.nbins), dtype=self.map_dtype)
        for k in range(ss_ncomb):
            xip[k] = _safe_div(xipm_num_np[0, k], xipm_den_np[k]).reshape(shape_pb)
            xim[k] = _safe_div(xipm_num_np[1, k], xipm_den_np[k]).reshape(shape_pb)

        xig_num_np = to_np(out_xig_num)
        xig_den_np = to_np(out_xig_den)
        xi_g = np.zeros((dd_ncomb, n_patches, self.nbins), dtype=self.map_dtype)
        for k in range(dd_ncomb):
            xi_g[k] = _safe_div(xig_num_np[k], xig_den_np[k]).reshape(shape_pb)

        xit_num_np = to_np(out_xit_num)
        xit_den_np = to_np(out_xit_den)
        xi_t = np.zeros((ds_ncomb, n_patches, self.nbins), dtype=self.map_dtype)
        for k in range(ds_ncomb):
            xi_t[k] = _safe_div(xit_num_np[k], xit_den_np[k]).reshape(shape_pb)

        return M_a, M_g, xip, xim, xi_g, xi_t

    def _launch_3x2pt_pair_kernels(
        self,
        density_aos: Any,
        shear_aos: Any,
        density_w_aos: Any,
        shear_w_aos: Any,
        pair_offsets: Any,
        ss_comb_i: Any,
        ss_comb_j: Any,
        dd_comb_i: Any,
        dd_comb_j: Any,
        ds_comb_i: Any,
        ds_comb_j: Any,
        out_xipm_num: Any,
        out_xipm_den: Any,
        out_xig_num: Any,
        out_xig_den: Any,
        out_xit_num: Any,
        out_xit_den: Any,
    ) -> None:
        """GPU pair statistics of the fused 3x2pt path: xi+/-, xi_g and xi_t in
        the combination-tiled pair kernels (packed geometry when prepared)."""
        module = self.backend.module
        ctx = self.compute_context
        backend = self.backend
        packed = ctx.packed_pairs_dev is not None
        if packed:
            perm = ctx.packed_perm_dev
            g = lambda arr: module.ascontiguousarray(arr[perm])
            density_aos, shear_aos = g(density_aos), g(shear_aos)
            density_w_aos, shear_w_aos = g(density_w_aos), g(shear_w_aos)
            geometry: Tuple[Any, ...] = (ctx.packed_pairs_dev, ctx.packed_row_base_dev)
        else:
            inds_i, inds_j = self._pair_index_arrays()
            geometry = (
                inds_i,
                inds_j,
                module.ascontiguousarray(self.exp2phi_dev[0]),
                module.ascontiguousarray(self.exp2phi_dev[1]),
            )

        # One walk over the pairs for as many statistics as fit the register
        # budget of the multi-statistic tile; the standalone tiles do the rest.
        done = (False, False, False)
        multi = getattr(backend, "kernel_3x2pt_tomo_pairs", None)
        if multi is not None:
            done = multi(
                density_aos, shear_aos, density_w_aos, shear_w_aos, geometry, pair_offsets,
                (ss_comb_i, ss_comb_j), (dd_comb_i, dd_comb_j), (ds_comb_i, ds_comb_j),
                out_xipm_num, out_xipm_den, out_xig_num, out_xig_den, out_xit_num, out_xit_den,
            )
        do_ss, do_dd, do_ds = (not flag for flag in done)

        if packed:
            density_p, shear_p = density_aos, shear_aos
            density_w_p, shear_w_p = density_w_aos, shear_w_aos
            pairs, row_base = geometry
            ok_ss = not do_ss or backend.xipm_tomo_packed_kernel is not None and backend.xipm_tomo_packed_kernel(
                shear_p, shear_w_p, pairs, pair_offsets, row_base,
                ss_comb_i, ss_comb_j, out_xipm_num, out_xipm_den,
            )
            ok_dd = not do_dd or backend.kernel_density_density_tomo_packed is not None and backend.kernel_density_density_tomo_packed(
                density_p, density_w_p, pairs, pair_offsets, row_base,
                dd_comb_i, dd_comb_j, out_xig_num, out_xig_den,
            )
            ok_ds = not do_ds or backend.kernel_density_shear_tomo_packed is not None and backend.kernel_density_shear_tomo_packed(
                density_p, shear_p, density_w_p, shear_w_p, pairs, pair_offsets, row_base,
                ds_comb_i, ds_comb_j, out_xit_num, out_xit_den,
            )
            if not (ok_ss and ok_dd and ok_ds):
                raise RuntimeError(
                    "A packed pair kernel is unavailable for this tomographic "
                    "configuration; use pack_pairs=False."
                )
            return

        inds_i, inds_j, rot_i, rot_j = geometry
        ok_ss = not do_ss or backend.xipm_tomo_vectorized_kernel(
            shear_aos, shear_w_aos, inds_i, inds_j, rot_i, rot_j, pair_offsets,
            ss_comb_i, ss_comb_j, out_xipm_num, out_xipm_den,
        )
        ok_dd = not do_dd or backend.kernel_density_density_tomo_vectorized(
            density_aos, density_w_aos, inds_i, inds_j, pair_offsets,
            dd_comb_i, dd_comb_j, out_xig_num, out_xig_den,
        )
        ok_ds = not do_ds or backend.kernel_density_shear_tomo_vectorized(
            density_aos, shear_aos, density_w_aos, shear_w_aos,
            inds_i, inds_j, rot_i, rot_j, pair_offsets,
            ds_comb_i, ds_comb_j, out_xit_num, out_xit_den,
        )
        if not (ok_ss and ok_dd and ok_ds):
            raise RuntimeError("Backend declined a 3x2pt pair kernel launch.")

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

