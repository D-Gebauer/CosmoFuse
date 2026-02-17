import hashlib
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import healpy as hp
import numpy as np
from numba import njit, prange
from scipy.special import binom
from tqdm import trange

from .backend import get_backend
from .correlation_helpers import Q_T, M_a_patch
from .utils import pixel2RaDec

logger = logging.getLogger(__name__)

_ALLOWED_FLOAT_PRECISIONS = {
    "float32": np.float32,
    "float64": np.float64,
}
_ALLOWED_INDEX_PRECISIONS = {
    "uint32": np.uint32,
    "uint64": np.uint64,
}
_ROTATION_COMPLEX_PRECISION = {
    "float32": np.complex64,
    "float64": np.complex128,
}


def _compute_M_a_all_patches(
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
    M_a = np.zeros(n_patches, dtype=g1.dtype)
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
        M_a[patch_idx] = Q_patch_area[patch_idx] * sum_gtw / sum_w
    return M_a


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
]:
    npts = patch_inds.size
    counts = np.zeros(npts, dtype=np.int64)

    sin_dec = np.sin(dec)
    cos_dec = np.cos(dec)

    bin_min = binedges[0]
    bin_max = binedges[binedges.size - 1]

    for i in prange(npts - 1):
        rai = ra[i]
        sdi = sin_dec[i]
        cdi = cos_dec[i]
        count_i = 0
        for j in range(i + 1, npts):
            cos_theta = sdi * sin_dec[j] + cdi * cos_dec[j] * np.cos(rai - ra[j])
            if cos_theta > 1.0:
                cos_theta = 1.0
            elif cos_theta < -1.0:
                cos_theta = -1.0
            theta = np.arccos(cos_theta)

            if theta <= bin_min or theta >= bin_max:
                continue

            valid = False
            for b in range(binedges.size - 1):
                if theta > binedges[b] and theta < binedges[b + 1]:
                    valid = True
                    break
            if not valid:
                continue
            count_i += 1
        counts[i] = count_i

    offsets = np.empty(npts + 1, dtype=np.int64)
    offsets[0] = 0
    for i in range(npts):
        offsets[i + 1] = offsets[i] + counts[i]

    ntotal = offsets[npts]

    inds_a = np.empty(ntotal, dtype=patch_inds.dtype)
    inds_b = np.empty(ntotal, dtype=patch_inds.dtype)
    bin_indices = np.empty(ntotal, dtype=np.int64)
    exp2phi1_real = np.empty(ntotal, dtype=ra.dtype)
    exp2phi1_imag = np.empty(ntotal, dtype=ra.dtype)
    exp2phi2_real = np.empty(ntotal, dtype=ra.dtype)
    exp2phi2_imag = np.empty(ntotal, dtype=ra.dtype)

    for i in prange(npts - 1):
        rai = ra[i]
        sdi = sin_dec[i]
        cdi = cos_dec[i]
        x1 = cdi * np.cos(rai)
        y1 = cdi * np.sin(rai)
        z1 = sdi
        out_idx = offsets[i]

        for j in range(i + 1, npts):
            raj = ra[j]
            sdj = sin_dec[j]
            cdj = cos_dec[j]
            x2 = cdj * np.cos(raj)
            y2 = cdj * np.sin(raj)
            z2 = sdj

            dra = raj - rai
            cos_theta = sdi * sdj + cdi * cdj * np.cos(dra)
            if cos_theta > 1.0:
                cos_theta = 1.0
            elif cos_theta < -1.0:
                cos_theta = -1.0
            theta = np.arccos(cos_theta)

            if theta <= bin_min or theta >= bin_max:
                continue

            bin_idx = -1
            for b in range(binedges.size - 1):
                if theta > binedges[b] and theta < binedges[b + 1]:
                    bin_idx = b
                    break
            if bin_idx < 0:
                continue

            sinC1 = x1 * y2 - x2 * y1
            dsq_AC1 = x1 * x1 + y1 * y1 + (z1 - 1.0) * (z1 - 1.0)
            dx12 = x1 - x2
            dy12 = y1 - y2
            dz12 = z1 - z2
            dsq_BC1 = dx12 * dx12 + dy12 * dy12 + dz12 * dz12
            dsq_AB1 = x2 * x2 + y2 * y2 + (z2 - 1.0) * (z2 - 1.0)
            cosC1 = 0.5 * (dsq_AC1 + dsq_BC1 - dsq_AB1 - 0.5 * dsq_AC1 * dsq_BC1)
            C1 = np.arctan2(sinC1, cosC1)
            theta1 = 0.5 * np.pi - C1

            sinC2 = x2 * y1 - x1 * y2
            dsq_AC2 = dsq_AB1
            dsq_BC2 = dsq_BC1
            dsq_AB2 = dsq_AC1
            cosC2 = 0.5 * (dsq_AC2 + dsq_BC2 - dsq_AB2 - 0.5 * dsq_AC2 * dsq_BC2)
            C2 = np.arctan2(sinC2, cosC2)
            theta2 = 0.5 * np.pi - C2

            c1 = np.cos(2.0 * theta1)
            s1 = np.sin(2.0 * theta1)
            c2 = np.cos(2.0 * theta2)
            s2 = np.sin(2.0 * theta2)

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


_compute_pairs_numba = njit(fastmath=True, parallel=True)(_compute_pairs_impl)
_compute_pairs_numba_precise = njit(fastmath=False, parallel=True)(
    _compute_pairs_impl
)


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
        index_precision: Integer precision for index/binned arrays.
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
        index_precision: Union[str, np.dtype, type] = "uint32",
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
            index_precision: One of uint32/uint64 for index-like arrays.

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
        self.index_dtype = _normalize_precision(
            index_precision, _ALLOWED_INDEX_PRECISIONS, "index_precision"
        )
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
        self.M_A_patch = njit(fastmath=fastmath)(M_a_patch)
        self.M_A_all_patches = njit(fastmath=fastmath)(_compute_M_a_all_patches)
        self.radius_filter = 5 * self.theta_Q

        if mask is not None:
            if len(mask) != hp.nside2npix(self.nside):
                raise ValueError(
                    "Mask length must match number of pixels for given nside"
                )
            self.map_inds = np.where(mask)[0].astype(self.index_dtype, copy=False)
        else:
            self.map_inds = np.arange(hp.nside2npix(self.nside), dtype=self.index_dtype)

        self.backend = get_backend(device)
        self.device = device

        self.pair_inds = []
        self.pair_exp2phi = []
        self.bins = []

        self.inds_dev = None
        self.exp2phi_dev = None
        self.bins_dev = None
        self.tot_bins_dev = None
        self.tot_bins_reduceat_dev = None
        self.ntotpairs = 0
        self._prepare_version = 0
        self._tomo_sumofweights_cache = None
        self._tomo_sumofweights_cache_w_fingerprint = None
        self._tomo_sumofweights_cache_prepare_version = None
        self._xipm_sumofweights_cache = None
        self._xipm_sumofweights_cache_w_fingerprint = None
        self._xipm_sumofweights_cache_prepare_version = None
        self.Q_inds_flat = None
        self.Q_cos_flat = None
        self.Q_sin_flat = None
        self.Q_val_flat = None
        self.Q_offsets = None
        self.Q_patch_area_flat = None

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        if 'backend' in state:
            del state['backend']
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.backend = get_backend(self.device)
        if "M_A_all_patches" not in self.__dict__:
            self.M_A_all_patches = njit(fastmath=self.fastmath)(_compute_M_a_all_patches)
        if "_prepare_version" not in self.__dict__:
            self._prepare_version = 0
        if "_tomo_sumofweights_cache" not in self.__dict__:
            self._tomo_sumofweights_cache = None
        if "_tomo_sumofweights_cache_w_fingerprint" not in self.__dict__:
            self._tomo_sumofweights_cache_w_fingerprint = None
        if "_tomo_sumofweights_cache_prepare_version" not in self.__dict__:
            self._tomo_sumofweights_cache_prepare_version = None
        if "_xipm_sumofweights_cache" not in self.__dict__:
            self._xipm_sumofweights_cache = None
        if "_xipm_sumofweights_cache_w_fingerprint" not in self.__dict__:
            self._xipm_sumofweights_cache_w_fingerprint = None
        if "_xipm_sumofweights_cache_prepare_version" not in self.__dict__:
            self._xipm_sumofweights_cache_prepare_version = None
        if "Q_inds_flat" not in self.__dict__:
            self.Q_inds_flat = None
        if "Q_cos_flat" not in self.__dict__:
            self.Q_cos_flat = None
        if "Q_sin_flat" not in self.__dict__:
            self.Q_sin_flat = None
        if "Q_val_flat" not in self.__dict__:
            self.Q_val_flat = None
        if "Q_offsets" not in self.__dict__:
            self.Q_offsets = None
        if "Q_patch_area_flat" not in self.__dict__:
            self.Q_patch_area_flat = None

    def _invalidate_prepared_state(self) -> None:
        """Clears prepared backend buffers and cached tomographic weights."""
        self.inds_dev = None
        self.exp2phi_dev = None
        self.bins_dev = None
        self.tot_bins_dev = None
        self.tot_bins_reduceat_dev = None
        self.ntotpairs = 0
        self._tomo_sumofweights_cache = None
        self._tomo_sumofweights_cache_w_fingerprint = None
        self._tomo_sumofweights_cache_prepare_version = None
        self._xipm_sumofweights_cache = None
        self._xipm_sumofweights_cache_w_fingerprint = None
        self._xipm_sumofweights_cache_prepare_version = None
        self.Q_inds_flat = None
        self.Q_cos_flat = None
        self.Q_sin_flat = None
        self.Q_val_flat = None
        self.Q_offsets = None
        self.Q_patch_area_flat = None

    def get_pairs_patch(
        self, patch_inds: np.ndarray, ra: np.ndarray, dec: np.ndarray
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        ra_local = np.asarray(ra, dtype=self.rotation_dtype)
        dec_local = np.asarray(dec, dtype=self.rotation_dtype)
        binedges_local = np.asarray(self.binedges, dtype=self.rotation_dtype)
        patch_inds_local = np.asarray(patch_inds, dtype=self.index_dtype)

        if patch_inds_local.size < 2:
            all_inds = [np.empty((2, 0), dtype=self.index_dtype) for _ in range(self.nbins)]
            return all_inds, np.empty((2, 0), dtype=self.rotation_complex_dtype)

        (
            inds_a,
            inds_b,
            bin_indices,
            exp2phi1_real,
            exp2phi1_imag,
            exp2phi2_real,
            exp2phi2_imag,
        ) = (
            _compute_pairs_numba if self.fastmath else _compute_pairs_numba_precise
        )(
            patch_inds_local,
            ra_local,
            dec_local,
            binedges_local,
        )

        npairs = bin_indices.size
        if npairs == 0:
            all_inds = [np.empty((2, 0), dtype=self.index_dtype) for _ in range(self.nbins)]
            return all_inds, np.empty((2, 0), dtype=self.rotation_complex_dtype)

        order = np.argsort(bin_indices, kind="stable")
        inds_a = inds_a[order]
        inds_b = inds_b[order]
        bin_indices = bin_indices[order]
        exp2phi1_real = exp2phi1_real[order]
        exp2phi1_imag = exp2phi1_imag[order]
        exp2phi2_real = exp2phi2_real[order]
        exp2phi2_imag = exp2phi2_imag[order]

        exp2phi1 = (
            exp2phi1_real.astype(self.rotation_dtype, copy=False)
            + 1j * exp2phi1_imag.astype(self.rotation_dtype, copy=False)
        ).astype(self.rotation_complex_dtype, copy=False)
        exp2phi2 = (
            exp2phi2_real.astype(self.rotation_dtype, copy=False)
            + 1j * exp2phi2_imag.astype(self.rotation_dtype, copy=False)
        ).astype(self.rotation_complex_dtype, copy=False)
        exp2phi = np.vstack((exp2phi1, exp2phi2)).astype(
            self.rotation_complex_dtype, copy=False
        )

        all_inds = []
        for bin_idx in range(self.nbins):
            in_bin = np.where(bin_indices == bin_idx)[0]
            all_inds.append(
                np.array(
                    [inds_a[in_bin], inds_b[in_bin]],
                    dtype=self.index_dtype,
                )
            )

        return all_inds, exp2phi

    def __get_pairs_helper__(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        vec = hp.ang2vec(self.theta_center[i], self.phi_center[i])
        patch_inds = hp.query_disc(
            self.nside, vec=vec, radius=np.radians(self.patch_size / 60)
        )
        pix_inds = np.intersect1d(patch_inds, self.map_inds)
        ra, dec = pixel2RaDec(pix_inds, self.nside)
        (
            inds,
            exp2theta,
        ) = self.get_pairs_patch(pix_inds, ra, dec)
        ninds = np.array([len(inds[i][0]) for i in range(self.nbins)], dtype=self.index_dtype)
        all_inds = np.zeros((2, int(ninds.sum())), dtype=self.index_dtype)
        for bin_idx in range(self.nbins):
            start_idx = np.sum(ninds[:bin_idx])
            end_idx = np.sum(ninds[: bin_idx + 1])
            all_inds[0, start_idx:end_idx] = inds[bin_idx][0]
            all_inds[1, start_idx:end_idx] = inds[bin_idx][1]

        return all_inds, exp2theta.astype(self.rotation_complex_dtype, copy=False), ninds

    def calculate_pairs_2PCF(self) -> None:
        pair_inds, pair_exp2phi, bins = [], [], []
        for i in trange(self.n_patches, desc="2PCF pairs", unit="patch"):
            result = self.__get_pairs_helper__(i)

            pair_inds.append(result[0])
            pair_exp2phi.append(result[1])
            bins.append(result[2])

        self.pair_inds = pair_inds
        self.pair_exp2phi = pair_exp2phi
        self.bins = bins
        self._invalidate_prepared_state()

    def get_pairs_patch_M_a(
        self,
        pixels_RA_Q_patch: np.ndarray,
        pixels_dec_Q_patch: np.ndarray,
        Q_patch_center_RA: float,
        Q_patch_center_dec: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cos_vartheta = np.cos(pixels_RA_Q_patch - Q_patch_center_RA) * np.cos(
            Q_patch_center_dec
        ) * np.cos(pixels_dec_Q_patch) + np.sin(Q_patch_center_dec) * np.sin(
            pixels_dec_Q_patch
        )
        vartheta = np.arccos(cos_vartheta)
        sin_vartheta = np.sqrt(1 - cos_vartheta**2)
        cos_phi = (
            np.sin(pixels_RA_Q_patch - Q_patch_center_RA)
            * np.cos(pixels_dec_Q_patch)
            / sin_vartheta
        )
        sin_phi = (
            np.cos(pixels_dec_Q_patch) * np.sin(Q_patch_center_dec)
            - np.sin(pixels_dec_Q_patch)
            * np.cos(Q_patch_center_dec)
            * np.cos(pixels_RA_Q_patch - Q_patch_center_RA)
        ) / sin_vartheta
        cos_2phi = cos_phi * cos_phi - sin_phi * sin_phi
        sin_2phi = 2 * sin_phi * cos_phi

        Q = Q_T(vartheta, self.theta_Q)

        return cos_2phi, sin_2phi, Q

    def __get_pairs_M_a_helper__(
        self, i: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        vec = hp.ang2vec(self.theta_center[i], self.phi_center[i])
        pix_center = hp.ang2pix(self.nside, self.theta_center[i], self.phi_center[i])
        patch_inds = hp.query_disc(
            self.nside, vec=vec, radius=np.radians(5 * self.theta_Q / 60)
        )
        Qpix_inds = np.intersect1d(patch_inds, self.map_inds)
        Qpix_inds = Qpix_inds[Qpix_inds != pix_center]

        ra_center, dec_center = pixel2RaDec([pix_center], self.nside)
        Q_ra, Q_dec = pixel2RaDec(Qpix_inds, self.nside)
        Q_cos, Q_sin, Q_val = self.get_pairs_patch_M_a(
            Q_ra, Q_dec, ra_center, dec_center
        )

        Q_patch_area = self.rotation_dtype.type(
            Qpix_inds.size * hp.nside2pixarea(self.nside)
        )
        return (
            Q_cos.astype(self.rotation_dtype, copy=False),
            Q_sin.astype(self.rotation_dtype, copy=False),
            Q_val.astype(self.rotation_dtype, copy=False),
            Qpix_inds.astype(self.index_dtype, copy=False),
            Q_patch_area,
        )

    def calculate_pairs_M_a(self) -> None:
        self.Q_cos, self.Q_sin, self.Q_val, self.Q_inds, self.Q_patch_area = (
            [],
            [],
            [],
            [],
            [],
        )

        for i in trange(self.n_patches, desc="M_ap pairs", unit="patch"):
            Q_cos, Q_sin, Q_val, Q_inds, Q_patch_area = self.__get_pairs_M_a_helper__(
                i
            )
            self.Q_cos.append(Q_cos)
            self.Q_sin.append(Q_sin)
            self.Q_val.append(Q_val)
            self.Q_inds.append(Q_inds)
            self.Q_patch_area.append(Q_patch_area)

        self._prepare_aperture_flat()

    def _prepare_aperture_flat(self) -> None:
        if not self.Q_inds:
            self.Q_inds_flat = None
            self.Q_cos_flat = None
            self.Q_sin_flat = None
            self.Q_val_flat = None
            self.Q_offsets = None
            self.Q_patch_area_flat = None
            return

        sizes = np.array([arr.size for arr in self.Q_inds], dtype=np.int64)
        offsets = np.zeros(len(sizes) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(sizes)
        total = int(offsets[-1])

        Q_inds_flat = np.zeros(total, dtype=self.index_dtype)
        Q_cos_flat = np.zeros(total, dtype=self.rotation_dtype)
        Q_sin_flat = np.zeros(total, dtype=self.rotation_dtype)
        Q_val_flat = np.zeros(total, dtype=self.rotation_dtype)

        for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            Q_inds_flat[start:end] = self.Q_inds[i]
            Q_cos_flat[start:end] = self.Q_cos[i]
            Q_sin_flat[start:end] = self.Q_sin[i]
            Q_val_flat[start:end] = self.Q_val[i]

        self.Q_inds_flat = Q_inds_flat
        self.Q_cos_flat = Q_cos_flat
        self.Q_sin_flat = Q_sin_flat
        self.Q_val_flat = Q_val_flat
        self.Q_offsets = offsets
        self.Q_patch_area_flat = np.asarray(self.Q_patch_area, dtype=self.rotation_dtype)

    def preprocess(self) -> None:
        """
        Calculates the pairs and their angles for all patches for 2PCF & aperture mass.
        """
        logger.info("Calculating pairs for aperture mass")
        self.calculate_pairs_M_a()
        logger.info("Calculating pairs for 2PCF")
        self.calculate_pairs_2PCF()
        logger.info("Preparing flattened pair arrays on backend device")
        self.prepare()

    def save_pairs(self, filepath: str) -> None:
        if (
            self.pair_inds is None
            or self.pair_exp2phi is None
            or self.bins is None
        ):
            warnings.warn(
                "Cannot save pairs because host pair arrays were released. "
                "Reload or recompute pairs before calling save_pairs().",
                RuntimeWarning,
            )
            return

        with h5py.File(filepath, "w") as fp:
            fp.attrs["nside"] = self.nside
            fp.attrs["nbins"] = self.nbins
            fp.attrs["theta_min"] = self.theta_min
            fp.attrs["theta_max"] = self.theta_max
            fp.attrs["patch_size"] = self.patch_size
            fp.attrs["theta_Q"] = self.theta_Q
            fp.attrs["n_patches"] = self.n_patches
            fp.create_dataset("map_inds", data=self.map_inds)
            fp.create_dataset("phi_center", data=self.phi_center)
            fp.create_dataset("theta_center", data=self.theta_center)

            for i in range(self.n_patches):
                gp = fp.create_group(f"patch_{i:02d}")

                gp.create_dataset(f"pair_inds", data=self.pair_inds[i])
                gp.create_dataset(f"pair_exp2phi", data=self.pair_exp2phi[i])
                gp.create_dataset(f"bins", data=self.bins[i])

                gp.create_dataset(f"Q_inds", data=self.Q_inds[i])
                gp.create_dataset(f"Q_cos", data=self.Q_cos[i])
                gp.create_dataset(f"Q_sin", data=self.Q_sin[i])
                gp.create_dataset(f"Q_val", data=self.Q_val[i])
                gp.create_dataset(f"Q_patch_area", data=self.Q_patch_area[i])

    def load_pairs(
        self, filepath: str, start_ind: int = 0, stop_ind: Optional[int] = None
    ) -> None:
        self._invalidate_prepared_state()
        self.pair_inds = []
        self.pair_exp2phi = []
        self.bins = []
        self.Q_inds = []
        self.Q_cos = []
        self.Q_sin = []
        self.Q_val = []
        self.Q_patch_area = []

        with h5py.File(filepath, "r") as fp:
            if stop_ind is None:
                stop_ind = fp.attrs["n_patches"]
            self.nside = fp.attrs["nside"]
            self.nbins = fp.attrs["nbins"]
            self.theta_min = fp.attrs["theta_min"]
            self.theta_max = fp.attrs["theta_max"]
            self.binedges = np.geomspace(self.theta_min, self.theta_max, self.nbins + 1)
            self.bincenters = (
                np.sqrt(self.binedges[1:] * self.binedges[:-1]) * 60 * 180 / np.pi
            )
            self.patch_size = fp.attrs["patch_size"]
            self.theta_Q = fp.attrs["theta_Q"]
            self.n_patches = stop_ind - start_ind
            self.map_inds = fp["map_inds"][:].astype(self.index_dtype, copy=False)
            self.phi_center = fp["phi_center"][start_ind:stop_ind]
            self.theta_center = fp["theta_center"][start_ind:stop_ind]

            for i in range(start_ind, stop_ind):
                gp = fp[f"patch_{i:02d}"]
                self.pair_inds.append(
                    gp["pair_inds"][:].astype(self.index_dtype, copy=False)
                )
                self.pair_exp2phi.append(
                    gp["pair_exp2phi"][:].astype(self.rotation_complex_dtype, copy=False)
                )
                self.bins.append(gp["bins"][:].astype(self.index_dtype, copy=False))
                self.Q_inds.append(gp["Q_inds"][:].astype(self.index_dtype, copy=False))
                self.Q_cos.append(gp["Q_cos"][:].astype(self.rotation_dtype, copy=False))
                self.Q_sin.append(gp["Q_sin"][:].astype(self.rotation_dtype, copy=False))
                self.Q_val.append(gp["Q_val"][:].astype(self.rotation_dtype, copy=False))
                self.Q_patch_area.append(self.rotation_dtype.type(gp["Q_patch_area"][()]))
            self._prepare_aperture_flat()
        self.prepare()

    def get_M_a(self, g1: np.ndarray, g2: np.ndarray, w: np.ndarray) -> np.ndarray:
        if self.Q_inds_flat is None:
            self._prepare_aperture_flat()
        return self.M_A_all_patches(
            self.Q_inds_flat,
            self.Q_cos_flat,
            self.Q_sin_flat,
            self.Q_val_flat,
            self.Q_offsets,
            g1,
            g2,
            w,
            self.Q_patch_area_flat,
        )

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
        temp_inds = np.zeros((2, int(size)), dtype=self.index_dtype)
        temp_exp2phi = np.zeros((2, int(size)), dtype=self.rotation_complex_dtype)
        temp_bins = np.zeros((self.n_patches * self.nbins), dtype=self.index_dtype)
        temp_bins_tot = np.zeros((self.n_patches * self.nbins), dtype=self.index_dtype)

        for i in range(self.n_patches):
            temp_inds[:, first_patch_ind[i] : first_patch_ind[i + 1]] = self.pair_inds[
                i
            ]
            temp_exp2phi[:, first_patch_ind[i] : first_patch_ind[i + 1]] = (
                self.pair_exp2phi[i]
            )
            temp_bins[i * self.nbins : (i + 1) * self.nbins] = self.bins[i]
            temp_bins_tot[i * self.nbins : (i + 1) * self.nbins] = (
                first_patch_ind[i] + self.bins[i].cumsum()
            )
        temp_bins_tot = np.concatenate(
            (np.array([0], dtype=self.index_dtype), temp_bins_tot)
        )

        self.inds_dev = self.backend.to_device(temp_inds)
        self.exp2phi_dev = self.backend.to_device(temp_exp2phi)
        self.bins_dev = self.backend.to_device(temp_bins)
        self.tot_bins_dev = self.backend.to_device(temp_bins_tot)
        self.tot_bins_reduceat_dev = self.backend.to_device(
            temp_bins_tot.astype(np.int64, copy=False)
        )
        self.ntotpairs = size
        self._prepare_version += 1
        if release_host_pairs:
            self.pair_inds = None
            self.pair_exp2phi = None
            self.bins = None

    def xipm(
        self,
        g11: np.ndarray,
        g21: np.ndarray,
        g12: np.ndarray,
        g22: np.ndarray,
        w1: np.ndarray,
        w2: np.ndarray,
        sumofweights: Optional[Union[np.ndarray, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.inds_dev is None:
            self.prepare()
        if (g11 is g12) and (g21 is g22) and (w1 is w2):
            return self._xipm_auto(g11, g21, w1, sumofweights=sumofweights)

        return self._xipm_cross(
            g11,
            g21,
            g12,
            g22,
            w1,
            w2,
            sumofweights_ab=sumofweights,
            sumofweights_ba=sumofweights,
        )

    def _xipm_auto(
        self,
        g1: np.ndarray,
        g2: np.ndarray,
        w: np.ndarray,
        sumofweights: Optional[Union[np.ndarray, float]] = None,
        return_numpy: bool = True,
    ) -> Tuple[Any, Any]:
        g1_dev = self.backend.to_device(g1).astype(self.map_dtype, copy=False)
        g2_dev = self.backend.to_device(g2).astype(self.map_dtype, copy=False)
        w_dev = self.backend.to_device(w).astype(self.map_dtype, copy=False)

        if sumofweights is None:
            sumofweights_dev = self._get_xipm_sumofweights(w_dev, w_dev)
        else:
            sumofweights_dev = self._normalize_xipm_sumofweights(sumofweights)

        xipm_auto_corr_kernel = getattr(self.backend, "xipm_auto_corr_kernel", None)
        if xipm_auto_corr_kernel is None:
            raise RuntimeError(
                "Backend does not provide an xipm auto-correlation kernel; use a supported backend."
            )

        complex_dtype = (
            self.backend.module.complex64
            if self.rotation_complex_dtype == np.dtype(np.complex64)
            else self.backend.module.complex128
        )
        out_p = self.backend.zeros(self.ntotpairs, dtype=complex_dtype)
        out_m = self.backend.zeros(self.ntotpairs, dtype=complex_dtype)

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
        g11_dev = self.backend.to_device(g11).astype(self.map_dtype, copy=False)
        g21_dev = self.backend.to_device(g21).astype(self.map_dtype, copy=False)
        g12_dev = self.backend.to_device(g12).astype(self.map_dtype, copy=False)
        g22_dev = self.backend.to_device(g22).astype(self.map_dtype, copy=False)
        w1_dev = self.backend.to_device(w1).astype(self.map_dtype, copy=False)
        w2_dev = self.backend.to_device(w2).astype(self.map_dtype, copy=False)

        if sumofweights_ab is None:
            sum_ab = self._compute_xipm_sumofweights(w1_dev, w2_dev)
        else:
            sum_ab = self._normalize_xipm_sumofweights(sumofweights_ab)

        if sumofweights_ba is None:
            sum_ba = self._compute_xipm_sumofweights(w2_dev, w1_dev)
        else:
            sum_ba = self._normalize_xipm_sumofweights(sumofweights_ba)

        xipm_cross_corr_kernel = getattr(self.backend, "xipm_cross_corr_kernel", None)
        if xipm_cross_corr_kernel is None:
            raise RuntimeError(
                "Backend does not provide an xipm cross-correlation kernel; "
                "use a supported backend."
            )

        complex_dtype = (
            self.backend.module.complex64
            if self.rotation_complex_dtype == np.dtype(np.complex64)
            else self.backend.module.complex128
        )
        out_ab_p = self.backend.zeros(self.ntotpairs, dtype=complex_dtype)
        out_ab_m = self.backend.zeros(self.ntotpairs, dtype=complex_dtype)
        out_ba_p = self.backend.zeros(self.ntotpairs, dtype=complex_dtype)
        out_ba_m = self.backend.zeros(self.ntotpairs, dtype=complex_dtype)

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

    def _normalize_xipm_pairs(
        self, xip_num: Any, xim_num: Any, sumofweights_dev: Any
    ) -> Tuple[Any, Any]:
        xip = self.backend.zeros(xip_num.shape, dtype=xip_num.dtype)
        xim = self.backend.zeros(xim_num.shape, dtype=xim_num.dtype)
        if np.ndim(self.backend.to_numpy(sumofweights_dev)) == 0:
            if self.backend.to_numpy(sumofweights_dev) != 0:
                xip = xip_num / sumofweights_dev
                xim = xim_num / sumofweights_dev
        else:
            nonzero = sumofweights_dev != 0
            xip[nonzero] = xip_num[nonzero] / sumofweights_dev[nonzero]
            xim[nonzero] = xim_num[nonzero] / sumofweights_dev[nonzero]
        xip = xip.reshape((self.n_patches, self.nbins))
        xim = xim.reshape((self.n_patches, self.nbins))
        return xip, xim

    def _fingerprint_weights(
        self, w_np: np.ndarray
    ) -> Tuple[Tuple[int, ...], str, str]:
        w_contiguous = np.ascontiguousarray(w_np)
        digest = hashlib.blake2b(w_contiguous.tobytes()).hexdigest()
        return (w_contiguous.shape, w_contiguous.dtype.str, digest)

    def _normalize_xipm_sumofweights(
        self, sumofweights: Union[np.ndarray, float]
    ) -> Any:
        sumofweights_np = np.asarray(
            self.backend.to_numpy(sumofweights), dtype=self.map_dtype
        )
        if sumofweights_np.ndim == 0:
            return self.backend.to_device(sumofweights_np)
        expected_size = self.n_patches * self.nbins
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
        reduced = self.backend.add.reduceat(values, starts)
        reduced[self.bins_dev == 0] = 0
        return reduced

    def _compute_xipm_sumofweights(self, w1_dev: Any, w2_dev: Any) -> Any:
        if self.inds_dev is None:
            self.prepare()

        return self._reduce_pairs(w1_dev[self.inds_dev[0]] * w2_dev[self.inds_dev[1]])

    def _get_xipm_sumofweights(self, w1_dev: Any, w2_dev: Any) -> Any:
        w1_np = self.backend.to_numpy(w1_dev)
        w2_np = self.backend.to_numpy(w2_dev)
        w_fingerprint = (
            self._fingerprint_weights(w1_np),
            self._fingerprint_weights(w2_np),
        )

        cache = self._xipm_sumofweights_cache
        cache_is_valid = (
            cache is not None
            and self._xipm_sumofweights_cache_w_fingerprint == w_fingerprint
            and self._xipm_sumofweights_cache_prepare_version == self._prepare_version
        )
        if cache_is_valid:
            return cache

        sumofweights_dev = self._compute_xipm_sumofweights(w1_dev, w2_dev)
        self._xipm_sumofweights_cache = sumofweights_dev
        self._xipm_sumofweights_cache_w_fingerprint = w_fingerprint
        self._xipm_sumofweights_cache_prepare_version = self._prepare_version
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

    def get_full_tomo(
        self,
        shear_maps: np.ndarray,
        w: np.ndarray,
        sumofweights: Optional[np.ndarray] = None,
        flip_g1: bool = False,
        flip_g2: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.inds_dev is None:
            self.prepare()

        nzbins = shear_maps.shape[0]
        nzbin_combs = int(binom(nzbins + 1, 2))

        shear_maps_np = np.asarray(shear_maps, dtype=self.map_dtype)
        w_np = np.asarray(w, dtype=self.map_dtype)
        w_fingerprint = self._fingerprint_weights(w_np)

        shear_maps_dev = self.backend.to_device(shear_maps_np)
        w_dev = self.backend.to_device(w_np)

        if sumofweights is None:
            cache = self._tomo_sumofweights_cache
            cache_is_valid = (
                cache is not None
                and self._tomo_sumofweights_cache_w_fingerprint == w_fingerprint
                and self._tomo_sumofweights_cache_prepare_version
                == self._prepare_version
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
                self._tomo_sumofweights_cache = sumofweights_dev
                self._tomo_sumofweights_cache_w_fingerprint = w_fingerprint
                self._tomo_sumofweights_cache_prepare_version = self._prepare_version
        else:
            sumofweights_np = np.asarray(sumofweights, dtype=self.map_dtype)
            if sumofweights_np.ndim < 2:
                raise ValueError(
                    "sumofweights must have at least two dimensions with "
                    "shape (2, nzbin_combs, ...)"
                )
            if sumofweights_np.shape[0] != 2 or sumofweights_np.shape[1] != nzbin_combs:
                raise ValueError(
                    f"sumofweights must have first dimensions (2, {nzbin_combs}); "
                    f"got {sumofweights_np.shape}"
                )
            sumofweights_dev = self.backend.to_device(sumofweights_np)
            self._tomo_sumofweights_cache = sumofweights_dev
            self._tomo_sumofweights_cache_w_fingerprint = w_fingerprint
            self._tomo_sumofweights_cache_prepare_version = self._prepare_version

        g1_fac, g2_fac = 1, 1
        if flip_g1:
            g1_fac = -1
        if flip_g2:
            g2_fac = -1

        M_ap = np.zeros([nzbins, self.n_patches], dtype=self.map_dtype)
        map_backend_dtype = getattr(self.backend.module, self.map_dtype.name)

        xim1 = self.backend.zeros(
            [nzbin_combs, self.n_patches, self.nbins], dtype=map_backend_dtype
        )
        xim2 = self.backend.zeros(
            [nzbin_combs, self.n_patches, self.nbins], dtype=map_backend_dtype
        )
        xip1 = self.backend.zeros(
            [nzbin_combs, self.n_patches, self.nbins], dtype=map_backend_dtype
        )
        xip2 = self.backend.zeros(
            [nzbin_combs, self.n_patches, self.nbins], dtype=map_backend_dtype
        )

        k = 0
        for i in range(nzbins):
            M_ap[i] = self.get_M_a(
                g1_fac * shear_maps_np[i, 0], g2_fac * shear_maps_np[i, 1], w_np[i]
            )
            for j in range(i, nzbins):
                if i == j:
                    xip1[k], xim1[k] = self._xipm_auto(
                        g1_fac * shear_maps_dev[i, 0],
                        g2_fac * shear_maps_dev[i, 1],
                        w_dev[i],
                        sumofweights=sumofweights_dev[0, k],
                        return_numpy=False,
                    )
                    xip2[k], xim2[k] = xip1[k], xim1[k]
                else:
                    xip1[k], xim1[k] = self._xipm_cross(
                        g1_fac * shear_maps_dev[i, 0],
                        g2_fac * shear_maps_dev[i, 1],
                        g1_fac * shear_maps_dev[j, 0],
                        g2_fac * shear_maps_dev[j, 1],
                        w_dev[i],
                        w_dev[j],
                        sumofweights_ab=sumofweights_dev[0, k],
                        sumofweights_ba=sumofweights_dev[1, k],
                        return_numpy=False,
                    )
                    xip2[k], xim2[k] = xip1[k], xim1[k]
                k += 1
        
        xip = (xip1 + xip2) / 2
        xim = (xim1 + xim2) / 2

        return M_ap, self.backend.to_numpy(xip), self.backend.to_numpy(xim)
