import logging
from multiprocessing import get_all_start_methods, get_context
from typing import List, Optional, Tuple, Union

import h5py
import healpy as hp
import numpy as np
from numba import njit
from scipy.special import binom

from .backend import get_backend
from .correlation_helpers import Q_T, M_a_patch, getAngle
from .utils import pixel2RaDec

logger = logging.getLogger(__name__)

_ALLOWED_FLOAT_PRECISIONS = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
}
_ALLOWED_INDEX_PRECISIONS = {
    "uint32": np.uint32,
    "uint64": np.uint64,
}
_ROTATION_COMPLEX_PRECISION = {
    "float16": np.complex64,
    "float32": np.complex64,
    "float64": np.complex128,
}


def _normalize_precision(
    precision: Union[str, np.dtype, type],
    allowed: dict,
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
        multiprocessing_start_method: multiprocessing context used when threads > 1.
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
        multiprocessing_start_method: str = "spawn",
        map_precision: Union[str, np.dtype, type] = "float64",
        rotation_precision: Union[str, np.dtype, type] = "float64",
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
            multiprocessing_start_method: Start method used by multiprocessing when
                pair calculations are parallelized. Use one of available Python start
                methods or "default".
            map_precision: One of float16/float32/float64 for map-like arrays.
            rotation_precision: One of float16/float32/float64 for rotation/filter arrays.
            index_precision: One of uint32/uint64 for index-like arrays.

        Raises:
            ValueError: If input parameters are invalid
        """
        # Validate inputs
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
        if multiprocessing_start_method != "default":
            available_start_methods = get_all_start_methods()
            if multiprocessing_start_method not in available_start_methods:
                raise ValueError(
                    "multiprocessing_start_method must be one of "
                    f"{available_start_methods} or 'default'"
                )
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
        self.multiprocessing_start_method = multiprocessing_start_method

        # CPU-side storage for pairs (lists of arrays)
        self.pair_inds = []
        self.pair_exp2phi = []
        self.bins = []

        # Attributes for prepared data (flattened and moved to device)
        self.inds_dev = None
        self.exp2phi_dev = None
        self.bins_dev = None
        self.tot_bins_dev = None
        self.tot_bins_reduceat_dev = None
        self.ntotpairs = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'backend' in state:
            del state['backend']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Re-initialize backend
        self.backend = get_backend(self.device)

    def get_pairs_patch(
        self, patch_inds: np.ndarray, ra: np.ndarray, dec: np.ndarray
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        all_inds, exp2phi1_temp, exp2phi2_temp = [], [], []
        cos_vartheta = np.cos(np.subtract.outer(ra, ra)) * np.multiply.outer(
            np.cos(dec), np.cos(dec)
        ) + np.multiply.outer(np.sin(dec), np.sin(dec))
        dist = np.arccos(np.triu(cos_vartheta, k=1))

        for bin_idx in range(self.nbins):
            inds = np.where(
                (dist > self.binedges[bin_idx]) & (dist < self.binedges[bin_idx + 1])
            )
            ra_pairs1 = ra[inds[0]]
            dec_pairs1 = dec[inds[0]]
            ra_pairs2 = ra[inds[1]]
            dec_pairs2 = dec[inds[1]]

            all_inds.append(
                np.array(
                    [patch_inds[inds[0]], patch_inds[inds[1]]], dtype=self.index_dtype
                )
            )

            for j, _ in enumerate(ra_pairs1):
                theta1 = np.pi / 2 - getAngle(
                    ra_pairs1[j], dec_pairs1[j], ra_pairs2[j], dec_pairs2[j]
                )
                theta2 = np.pi / 2 - getAngle(
                    ra_pairs2[j], dec_pairs2[j], ra_pairs1[j], dec_pairs1[j]
                )
                exp2phi1 = np.cos(2 * theta1) + 1j * np.sin(2 * theta1)
                exp2phi2 = np.cos(2 * theta2) + 1j * np.sin(2 * theta2)

                exp2phi1_temp.append(self.rotation_complex_dtype.type(exp2phi1))
                exp2phi2_temp.append(self.rotation_complex_dtype.type(exp2phi2))

        exp2phi = np.array(
            [exp2phi1_temp, exp2phi2_temp], dtype=self.rotation_complex_dtype
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

    def calculate_pairs_2PCF(self, threads: int = 1) -> None:
        if threads == 1:
            pair_inds, pair_exp2phi, bins = [], [], []
            for i in range(self.n_patches):
                result = self.__get_pairs_helper__(i)

                pair_inds.append(result[0])
                pair_exp2phi.append(result[1])
                bins.append(result[2])

        else:
            # "spawn" is safest in multi-threaded contexts, but callers can override.
            if self.multiprocessing_start_method == "default":
                context = get_context()
            else:
                context = get_context(self.multiprocessing_start_method)
            with context.Pool(threads) as p:
                results = p.map(self.__get_pairs_helper__, range(self.n_patches))
                pair_inds, pair_exp2phi, bins = list(map(list, zip(*results)))

        self.pair_inds = pair_inds
        self.pair_exp2phi = pair_exp2phi
        self.bins = bins

    def get_pairs_patch_M_a(
        self,
        pixels_RA_Q_patch,
        pixels_dec_Q_patch,
        Q_patch_center_RA,
        Q_patch_center_dec,
    ):
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

    def calculate_pairs_M_a(self, threads=1):
        self.Q_cos, self.Q_sin, self.Q_val, self.Q_inds, self.Q_patch_area = (
            [],
            [],
            [],
            [],
            [],
        )

        for i in range(self.n_patches):
            vec = hp.ang2vec(self.theta_center[i], self.phi_center[i])
            pix_center = hp.ang2pix(
                self.nside, self.theta_center[i], self.phi_center[i]
            )
            patch_inds = hp.query_disc(
                self.nside, vec=vec, radius=np.radians(5 * self.theta_Q / 60)
            )
            Qpix_inds = np.intersect1d(patch_inds, self.map_inds)
            ra_center, dec_center = pixel2RaDec([pix_center], self.nside)
            Qpix_inds = Qpix_inds[~np.isin(Qpix_inds, pix_center)]
            Q_ra, Q_dec = pixel2RaDec(Qpix_inds, self.nside)
            Q_cos, Q_sin, Q_val = self.get_pairs_patch_M_a(
                Q_ra, Q_dec, ra_center, dec_center
            )

            self.Q_cos.append(Q_cos.astype(self.rotation_dtype, copy=False))
            self.Q_sin.append(Q_sin.astype(self.rotation_dtype, copy=False))
            self.Q_val.append(Q_val.astype(self.rotation_dtype, copy=False))
            self.Q_inds.append(Qpix_inds.astype(self.index_dtype, copy=False))
            self.Q_patch_area.append(
                self.rotation_dtype.type(Qpix_inds.size * hp.nside2pixarea(self.nside))
            )

    def preprocess(self, threads=1):
        """
        Calculates the pairs and their angles for all patches for 2PCF & aperture mass.
        """
        logger.info("Calculating pairs for aperture mass")
        self.calculate_pairs_M_a(threads)
        logger.info("Calculating pairs for 2PCF")
        self.calculate_pairs_2PCF(threads)

    def save_pairs(self, filepath):
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

    def load_pairs(self, filepath, start_ind=0, stop_ind=None):
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

    def get_M_a(self, g1, g2, w):
        M_a = np.zeros(self.n_patches, dtype=self.map_dtype)
        for i in range(self.n_patches):
            M_a[i] = self.M_A_patch(
                self.Q_inds[i],
                self.Q_cos[i],
                self.Q_sin[i],
                self.Q_val[i],
                g1,
                g2,
                w,
                self.Q_patch_area[i],
            )
        return M_a

    def prepare(self):
        """Prepares arrays for correlation calculation (flattens and moves to device)."""
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

        # Move to backend device
        self.inds_dev = self.backend.to_device(temp_inds)
        self.exp2phi_dev = self.backend.to_device(temp_exp2phi)
        self.bins_dev = self.backend.to_device(temp_bins)
        self.tot_bins_dev = self.backend.to_device(temp_bins_tot)
        # reduceat requires signed integer indices on numpy/cupy
        self.tot_bins_reduceat_dev = self.backend.to_device(
            temp_bins_tot.astype(np.int64, copy=False)
        )
        self.ntotpairs = size

    def load_maps(
        self, g11, g21, g12, g22, w1, w2, sumofweights=None, flip_g1=True, flip_g2=False
    ):
        if self.inds_dev is None:
            self.prepare()

        self.g11 = self.backend.to_device(np.asarray(g11, dtype=self.map_dtype))
        self.g21 = self.backend.to_device(np.asarray(g21, dtype=self.map_dtype))
        self.g12 = self.backend.to_device(np.asarray(g12, dtype=self.map_dtype))
        self.g22 = self.backend.to_device(np.asarray(g22, dtype=self.map_dtype))
        self.w1 = self.backend.to_device(np.asarray(w1, dtype=self.map_dtype))
        self.w2 = self.backend.to_device(np.asarray(w2, dtype=self.map_dtype))

        if sumofweights is None:
            self.sumofweights = self.backend.add.reduceat(
                self.w1[self.inds_dev[0]] * self.w2[self.inds_dev[1]],
                self.tot_bins_reduceat_dev[:-1],
            )
        else:
            self.sumofweights = self.backend.to_device(
                np.asarray(sumofweights, dtype=self.map_dtype)
            )

        if flip_g1:
            self.g11 = -self.g11
            self.g12 = -self.g12
        if flip_g2:
            self.g21 = -self.g21
            self.g22 = -self.g22

    def xipm(self, g11, g21, g12, g22, w1, w2, sumofweights):
        # Arrays should already be on device if passed correctly, or we assume they are.
        # But xipm is called with slices often, which are views.
        # The arguments passed to xipm here are expected to be on the device.

        if self.inds_dev is None:
            self.prepare()

        g2 = (
            w1[self.inds_dev[0]]
            * ((g11[self.inds_dev[0]]) + 1j * g21[self.inds_dev[0]])
            * self.exp2phi_dev[0]
        )
        g1 = (
            w2[self.inds_dev[1]]
            * ((g12[self.inds_dev[1]]) + 1j * g22[self.inds_dev[1]])
            * self.exp2phi_dev[1]
        )

        xip = (
            (
                self.backend.add.reduceat(
                    g1 * self.backend.conjugate(g2), self.tot_bins_reduceat_dev[:-1]
                )
            )
            / sumofweights
        ).reshape((self.n_patches, self.nbins))
        xim = (
            (
                self.backend.add.reduceat(g1 * g2, self.tot_bins_reduceat_dev[:-1])
            )
            / sumofweights
        ).reshape((self.n_patches, self.nbins))

        # Real parts
        return xip.real, xim.real

    def get_all_xipm(self):
        xip1, xim1 = self.xipm(
            self.g11,
            self.g21,
            self.g12,
            self.g22,
            self.w1,
            self.w2,
            self.sumofweights,
        )
        xip2, xim2 = self.xipm(
            self.g12,
            self.g22,
            self.g11,
            self.g21,
            self.w2,
            self.w1,
            self.sumofweights,
        )

        return (xip1 + xip2) / 2, (xim1 + xim2) / 2

    def get_full_tomo(self, shear_maps, w, sumofweights, flip_g1=False, flip_g2=False):
        nzbins = shear_maps.shape[0]
        nzbin_combs = int(binom(nzbins + 1, 2))

        # Move inputs to device
        shear_maps_dev = self.backend.to_device(
            np.asarray(shear_maps, dtype=self.map_dtype)
        )
        w_dev = self.backend.to_device(np.asarray(w, dtype=self.map_dtype))
        sumofweights_dev = self.backend.to_device(
            np.asarray(sumofweights, dtype=self.map_dtype)
        )

        g1, g2 = 1, 1
        if flip_g1:
            g1 = -1
        if flip_g2:
            g2 = -1

        M_ap = np.zeros([nzbins, self.n_patches], dtype=self.map_dtype)
        map_backend_dtype = getattr(self.backend.module, self.map_dtype.name)

        # Results on device
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
            # M_a calculation remains on CPU as it uses numba njit on CPU arrays typically?
            # get_M_a uses self.M_A_patch which is a njit function.
            # It expects CPU arrays? Let's check get_M_a.
            # It uses self.Q_inds[i], self.Q_cos[i] etc which are lists of numpy arrays (CPU).
            # So shear_maps and w passed to it must be CPU arrays.
            # shear_maps[i, 0] is a numpy array view if shear_maps is numpy array.

            # Note: shear_maps passed to this function is likely numpy array.
            # shear_maps_dev is the device copy.

            M_ap[i] = self.get_M_a(g1 * shear_maps[i, 0], g2 * shear_maps[i, 1], w[i])
            for j in range(i, nzbins):
                if i == j:
                    xip1[k], xim1[k] = self.xipm(
                        g1 * shear_maps_dev[i, 0],
                        g2 * shear_maps_dev[i, 1],
                        g1 * shear_maps_dev[j, 0],
                        g2 * shear_maps_dev[j, 1],
                        w_dev[i],
                        w_dev[j],
                        sumofweights_dev[0, k],
                    )
                    xip2[k], xim2[k] = xip1[k], xim1[k]
                else:
                    xip1[k], xim1[k] = self.xipm(
                        g1 * shear_maps_dev[i, 0],
                        g2 * shear_maps_dev[i, 1],
                        g1 * shear_maps_dev[j, 0],
                        g2 * shear_maps_dev[j, 1],
                        w_dev[i],
                        w_dev[j],
                        sumofweights_dev[0, k],
                    )
                    xip2[k], xim2[k] = self.xipm(
                        g1 * shear_maps_dev[j, 0],
                        g2 * shear_maps_dev[j, 1],
                        g1 * shear_maps_dev[i, 0],
                        g2 * shear_maps_dev[i, 1],
                        w_dev[j],
                        w_dev[i],
                        sumofweights_dev[1, k],
                    )
                k += 1

        xip = (xip1 + xip2) / 2
        xim = (xim1 + xim2) / 2

        # Return as numpy arrays
        return M_ap, self.backend.to_numpy(xip), self.backend.to_numpy(xim)

    def calculate_2PCF(self, threads=1):
        """
        Calculates the 2PCF for the loaded maps.

        This method computes xip and xim using the loaded maps (loaded via load_maps).
        It mimics the behavior of the old Correlation_CPU.calculate_2PCF but uses the vectorized xipm implementation.
        """
        # Ensure data is prepared
        if self.inds_dev is None:
             self.prepare()

        # If maps are not loaded, this will fail.
        # But wait, calculate_2PCF is usually called after load_maps.

        xip, xim = self.get_all_xipm()
        self.xip = self.backend.to_numpy(xip)
        self.xim = self.backend.to_numpy(xim)
        return self.xip, self.xim

