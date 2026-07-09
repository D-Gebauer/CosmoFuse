"""
Pair geometry and aperture filter calculations.

Handles the spatial geometry needed for 2-point and aperture statistics:

- **2PCF pair geometry**: For each HEALPix sky patch, finds all pixel pairs
  within the configured angular separation bins and computes the rotation
  factors e^{2iφ} needed to rotate the spin-2 shear field into the pair frame.

- **Aperture geometry**: For the aperture mass M_ap, computes for each pixel
  within a patch the angular distance θ to the patch centre, the compensated
  filter Q(θ), and the cos(2φ)/sin(2φ) factors needed to project the shear
  into tangential and cross components.
"""

from typing import Any, Callable, List, Optional, Tuple, TYPE_CHECKING

import healpy as hp
import numpy as np
from tqdm import trange

from .correlation_helpers import Q_crittenden
from .utils import pixel2RaDec

if TYPE_CHECKING:
    from .correlations import Correlation


class PairGeometry:
    @staticmethod
    def resolve_aperture_filter(
        aperture_filter: Optional[Callable[..., Any]] = None,
    ) -> Callable[..., Any]:
        return Q_crittenden if aperture_filter is None else aperture_filter

    @staticmethod
    def aperture_filter_key(aperture_filter: Callable[..., Any]) -> Any:
        if aperture_filter is Q_crittenden:
            # Legacy key kept for pickle/state compatibility (Q_T is the
            # backwards-compatible alias of Q_crittenden).
            return "Q_T"
        return id(aperture_filter)

    @staticmethod
    def evaluate_aperture_filter(
        owner: "Correlation",
        aperture_filter: Callable[..., Any],
        theta: np.ndarray,
    ) -> np.ndarray:
        try:
            values = aperture_filter(theta, owner.theta_Q)
        except TypeError:
            values = aperture_filter(theta)
        return np.asarray(values, dtype=owner.rotation_dtype)

    @staticmethod
    def ensure_aperture_pairs(
        owner: "Correlation",
        aperture_filter: Optional[Callable[..., Any]] = None,
    ) -> None:
        filter_fn = PairGeometry.resolve_aperture_filter(aperture_filter)
        filter_key = PairGeometry.aperture_filter_key(filter_fn)
        if owner.Q_inds_flat is None:
            if owner.Q_inds and owner._aperture_filter_active_key == filter_key:
                owner._prepare_aperture_flat()
                return
            owner.calculate_pairs_M_a(aperture_filter=filter_fn)
            return

        if owner._aperture_filter_active_key != filter_key:
            owner.calculate_pairs_M_a(aperture_filter=filter_fn)

    @staticmethod
    def get_pairs_patch(
        owner: "Correlation",
        patch_inds: np.ndarray,
        ra: np.ndarray,
        dec: np.ndarray,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        owner._pair_finder.kernel = owner._compute_pairs_kernel
        return owner._pair_finder.get_pairs_patch(
            patch_inds,
            ra,
            dec,
        )

    @staticmethod
    def get_pairs_helper(
        owner: "Correlation",
        i: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        vec = hp.ang2vec(owner.theta_center[i], owner.phi_center[i])
        patch_inds = hp.query_disc(
            owner.nside, vec=vec, radius=np.radians(owner.patch_size / 60)
        )
        pix_inds = patch_inds[owner.map_mask[patch_inds]]
        ra, dec = pixel2RaDec(pix_inds, owner.nside)
        owner._pair_finder.kernel = owner._compute_pairs_kernel
        all_inds, exp2theta, ninds = owner._pair_finder.get_pairs_patch_flat(
            pix_inds,
            ra,
            dec,
        )
        return (
            all_inds,
            exp2theta.astype(owner.rotation_complex_dtype, copy=False),
            ninds.astype(owner.index_dtype, copy=False),
        )

    @staticmethod
    def calculate_pairs_2PCF(owner: "Correlation") -> None:
        pair_inds, pair_exp2phi, bins = [], [], []
        for i in trange(owner.n_patches, desc="2PCF pairs", unit=" patches"):
            result = owner.__get_pairs_helper__(i)

            pair_inds.append(result[0])
            pair_exp2phi.append(result[1])
            bins.append(result[2])

        owner.pair_inds = pair_inds
        owner.pair_exp2phi = pair_exp2phi
        owner.bins = bins
        owner._invalidate_prepared_state()

    @staticmethod
    def get_pairs_patch_M_a(
        owner: "Correlation",
        pixels_RA_Q_patch: np.ndarray,
        pixels_dec_Q_patch: np.ndarray,
        Q_patch_center_RA: float,
        Q_patch_center_dec: float,
        aperture_filter: Optional[Callable[..., Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the aperture geometry for pixels relative to a patch centre.

        For each pixel, calculates:
        - ϑ: angular distance to the patch centre (great-circle)
        - φ: position angle of the pixel relative to the centre
        - cos(2φ), sin(2φ): needed to project shear into tangential component
          γ_t = -γ₁·cos(2φ) - γ₂·sin(2φ)
        - Q(ϑ): the compensated aperture filter value

        Returns (cos_2phi, sin_2phi, Q).
        """
        # Hoisted trig: each of these arrays was previously recomputed
        # two to three times below (bit-identical results).
        delta_ra = pixels_RA_Q_patch - Q_patch_center_RA
        cos_delta_ra = np.cos(delta_ra)
        sin_delta_ra = np.sin(delta_ra)
        cos_dec = np.cos(pixels_dec_Q_patch)
        sin_dec = np.sin(pixels_dec_Q_patch)
        cos_dec_c = np.cos(Q_patch_center_dec)
        sin_dec_c = np.sin(Q_patch_center_dec)

        # Great-circle angular distance ϑ via spherical law of cosines
        cos_vartheta = cos_delta_ra * cos_dec_c * cos_dec + sin_dec_c * sin_dec
        vartheta = np.arccos(cos_vartheta)
        sin_vartheta = np.sqrt(1 - cos_vartheta**2)
        # Position angle φ of each pixel relative to patch centre
        # (components from the spherical bearing formula)
        cos_phi = sin_delta_ra * cos_dec / sin_vartheta
        sin_phi = (
            cos_dec * sin_dec_c - sin_dec * cos_dec_c * cos_delta_ra
        ) / sin_vartheta
        # Double-angle identities for the spin-2 shear projection
        cos_2phi = cos_phi * cos_phi - sin_phi * sin_phi
        sin_2phi = 2 * sin_phi * cos_phi

        # Evaluate the compensated aperture filter Q(ϑ)
        filter_fn = PairGeometry.resolve_aperture_filter(aperture_filter)
        Q = PairGeometry.evaluate_aperture_filter(owner, filter_fn, vartheta)

        return cos_2phi, sin_2phi, Q

    @staticmethod
    def get_pairs_M_a_helper(
        owner: "Correlation",
        i: int,
        aperture_filter: Optional[Callable[..., Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        vec = hp.ang2vec(owner.theta_center[i], owner.phi_center[i])
        pix_center = hp.ang2pix(owner.nside, owner.theta_center[i], owner.phi_center[i])
        patch_inds = hp.query_disc(
            owner.nside, vec=vec, radius=np.radians(5 * owner.theta_Q / 60)
        )
        qpix_inds = patch_inds[owner.map_mask[patch_inds]]
        qpix_inds = qpix_inds[qpix_inds != pix_center]

        ra_center, dec_center = pixel2RaDec([pix_center], owner.nside)
        q_ra, q_dec = pixel2RaDec(qpix_inds, owner.nside)
        q_cos, q_sin, q_val = PairGeometry.get_pairs_patch_M_a(
            owner,
            q_ra,
            q_dec,
            ra_center,
            dec_center,
            aperture_filter=aperture_filter,
        )

        q_patch_area = owner.rotation_dtype.type(
            qpix_inds.size * hp.nside2pixarea(owner.nside)
        )
        return (
            q_cos.astype(owner.rotation_dtype, copy=False),
            q_sin.astype(owner.rotation_dtype, copy=False),
            q_val.astype(owner.rotation_dtype, copy=False),
            qpix_inds.astype(owner.index_dtype, copy=False),
            q_patch_area,
        )

    @staticmethod
    def calculate_pairs_M_a(
        owner: "Correlation",
        aperture_filter: Optional[Callable[..., Any]] = None,
    ) -> None:
        filter_fn = PairGeometry.resolve_aperture_filter(aperture_filter)
        owner.Q_cos, owner.Q_sin, owner.Q_val, owner.Q_inds, owner.Q_patch_area = (
            [],
            [],
            [],
            [],
            [],
        )

        for i in trange(owner.n_patches, desc="M_a data", unit=" patches"):
            q_cos, q_sin, q_val, q_inds, q_patch_area = owner.__get_pairs_M_a_helper__(
                i,
                aperture_filter=filter_fn,
            )
            owner.Q_cos.append(q_cos)
            owner.Q_sin.append(q_sin)
            owner.Q_val.append(q_val)
            owner.Q_inds.append(q_inds)
            owner.Q_patch_area.append(q_patch_area)

        owner._prepare_aperture_flat()
        owner._aperture_filter_active_key = PairGeometry.aperture_filter_key(filter_fn)

    @staticmethod
    def prepare_aperture_flat(owner: "Correlation") -> None:
        if not owner.Q_inds:
            owner.Q_inds_flat = None
            owner.Q_cos_flat = None
            owner.Q_sin_flat = None
            owner.Q_val_flat = None
            owner.Q_offsets = None
            owner.Q_patch_area_flat = None
            owner._invalidate_aperture_device_buffers()
            return

        sizes = np.array([arr.size for arr in owner.Q_inds], dtype=np.int64)
        offsets = np.zeros(len(sizes) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(sizes)
        total = int(offsets[-1])

        Q_inds_flat = np.zeros(total, dtype=owner.index_dtype)
        Q_cos_flat = np.zeros(total, dtype=owner.rotation_dtype)
        Q_sin_flat = np.zeros(total, dtype=owner.rotation_dtype)
        Q_val_flat = np.zeros(total, dtype=owner.rotation_dtype)

        for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            Q_inds_flat[start:end] = owner.Q_inds[i]
            Q_cos_flat[start:end] = owner.Q_cos[i]
            Q_sin_flat[start:end] = owner.Q_sin[i]
            Q_val_flat[start:end] = owner.Q_val[i]

        owner.Q_inds_flat = Q_inds_flat
        owner.Q_cos_flat = Q_cos_flat
        owner.Q_sin_flat = Q_sin_flat
        owner.Q_val_flat = Q_val_flat
        owner.Q_offsets = offsets
        owner.Q_patch_area_flat = np.asarray(owner.Q_patch_area, dtype=owner.rotation_dtype)
        owner._invalidate_aperture_device_buffers()
