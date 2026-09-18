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
from .packing import pack_patch
from .treecode import (
    PatchCells,
    TreecodeGeometry,
    build_patch_cells,
    estimate_pairs_per_bin,
    level_groups,
)
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
    def coarse_nsides(owner: "Correlation") -> Tuple[int, ...]:
        """Distinct coarse levels used by the 2PCF bins, descending nside."""
        levels = {int(ns) for ns in owner.level_nside if int(ns) != int(owner.nside)}
        return tuple(sorted(levels, reverse=True))

    @staticmethod
    def get_pairs_helper_levels(
        owner: "Correlation",
        i: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PatchCells]:
        """Static-treecode pair search for patch ``i``.

        Every resolution level is searched only in the (contiguous) bin
        range it owns.  Coarse pair indices are returned as ``npix +
        patch-local cell index``; :meth:`calculate_pairs_2PCF` shifts them
        to global virtual-row ids once all patches are known.
        """
        vec = hp.ang2vec(owner.theta_center[i], owner.phi_center[i])
        patch_inds = hp.query_disc(
            owner.nside, vec=vec, radius=np.radians(owner.patch_size / 60)
        )
        pix_inds = patch_inds[owner.map_mask[patch_inds]]
        coarse = PairGeometry.coarse_nsides(owner)
        cells = build_patch_cells(owner.nside, pix_inds, coarse)

        npix = hp.nside2npix(owner.nside)
        owner._pair_finder.kernel = owner._compute_pairs_kernel
        all_inds, all_exp, all_ninds = [], [], []
        for nside_b, b0, b1 in level_groups(owner.level_nside):
            if nside_b == owner.nside:
                ids = pix_inds
                ra, dec = pixel2RaDec(pix_inds, owner.nside)
                shift = 0
            else:
                level = coarse.index(nside_b)
                ids = np.arange(cells.n_cells[level], dtype=np.int64)
                ra, dec = cells.ra[level], cells.dec[level]
                shift = npix
            inds, exp2theta, ninds = owner._pair_finder.get_pairs_patch_flat(
                ids, ra, dec, binedges=owner.binedges[b0 : b1 + 1]
            )
            if shift:
                inds = inds + owner.index_dtype.type(shift)
            all_inds.append(inds)
            all_exp.append(exp2theta)
            all_ninds.append(ninds)

        return (
            np.concatenate(all_inds, axis=1).astype(owner.index_dtype, copy=False),
            np.concatenate(all_exp, axis=1).astype(
                owner.rotation_complex_dtype, copy=False
            ),
            np.concatenate(all_ninds).astype(owner.index_dtype, copy=False),
            cells,
        )

    @staticmethod
    def preflight_pair_memory(owner: "Correlation") -> Optional[dict]:
        """Project the pair memory *before* pair finding and fail early.

        Uses the analytic pair-separation distribution of a disc with the
        actual number of unmasked pixels per patch.  Raises ``MemoryError``
        (naming a ``resolution_factor`` that fits) if the projection exceeds
        the budget: ``owner.memory_budget_gb`` if given, otherwise the free
        device memory on GPU backends.  Returns the projection.
        """
        budget = owner.memory_budget_gb
        if budget is None:
            if owner.backend.name != "cupy":
                return None
            try:
                free_bytes, _ = owner.backend.module.cuda.runtime.memGetInfo()
            except Exception:  # pragma: no cover - driver query failed
                return None
            budget = free_bytes / 1e9
        if not np.isfinite(budget):
            return None

        radius = np.radians(owner.patch_size / 60)
        n_pix = np.empty(owner.n_patches, dtype=np.int64)
        for i in range(owner.n_patches):
            vec = hp.ang2vec(owner.theta_center[i], owner.phi_center[i])
            disc = hp.query_disc(owner.nside, vec=vec, radius=radius)
            n_pix[i] = int(np.count_nonzero(owner.map_mask[disc]))

        bytes_per_pair = 2 * owner.index_dtype.itemsize + 2 * owner.rotation_complex_dtype.itemsize

        def projected_gb(level_nside: np.ndarray) -> float:
            pairs = estimate_pairs_per_bin(
                n_pix, owner.binedges, level_nside, owner.nside, radius
            )
            return float(pairs.sum() * bytes_per_pair / 1e9)

        from .treecode import assign_levels, is_power_of_two

        need = projected_gb(owner.level_nside)
        report = {"projected_gb": need, "budget_gb": float(budget)}
        if need <= budget:
            return report

        suggestion = None
        if is_power_of_two(owner.nside):
            current = owner.resolution_factor or np.inf
            for k in (16.0, 11.6, 8.0, 5.8, 4.0, 2.9, 2.0):
                if k >= current:
                    continue
                gb = projected_gb(assign_levels(owner.binedges, owner.nside, k))
                if gb <= budget:
                    suggestion = (k, gb)
                    break
        hint = (
            f"resolution_factor={suggestion[0]:g} would need ~{suggestion[1]:.3g} GB."
            if suggestion
            else "No resolution_factor >= 2 fits; reduce the number of patches "
            "(load_pairs(start_ind, stop_ind) chunks) or the angular range."
        )
        mode = (
            "full resolution"
            if owner.resolution_factor is None
            else f"resolution_factor={owner.resolution_factor:g}"
        )
        raise MemoryError(
            f"Projected pair geometry for {owner.n_patches} patches at nside "
            f"{owner.nside} ({mode}): ~{need:.3g} GB, budget {budget:.3g} GB. "
            f"{hint} (The static treecode is a different, windowed estimator: "
            "use the same resolution_factor for data and simulations. Set "
            "memory_budget_gb=float('inf') to skip this check.)"
        )

    @staticmethod
    def calculate_pairs_2PCF(owner: "Correlation") -> None:
        PairGeometry.preflight_pair_memory(owner)

        coarse = PairGeometry.coarse_nsides(owner)
        pair_inds, pair_exp2phi, bins = [], [], []
        patch_cells: List[PatchCells] = []
        for i in trange(owner.n_patches, desc="2PCF pairs", unit=" patches"):
            if coarse:
                result = PairGeometry.get_pairs_helper_levels(owner, i)
                patch_cells.append(result[3])
            else:
                # Full resolution: the single-level search, unchanged.
                result = owner.__get_pairs_helper__(i)

            pair_inds.append(result[0])
            pair_exp2phi.append(result[1])
            bins.append(result[2])

        treecode = None
        if coarse:
            treecode = TreecodeGeometry.from_patches(owner.nside, coarse, patch_cells)
            npix = hp.nside2npix(owner.nside)
            if (
                npix + owner.n_aperture_cells + treecode.n_cells
                > np.iinfo(owner.index_dtype).max
            ):
                raise OverflowError(
                    "pixel + coarse-cell ids exceed the index dtype range"
                )
            # patch-local cell index -> global virtual-row id
            starts = treecode.level_starts(first=owner.n_aperture_cells)
            groups = level_groups(owner.level_nside)
            for i in range(owner.n_patches):
                edges = np.concatenate(([0], np.cumsum(bins[i], dtype=np.int64)))
                for nside_b, b0, b1 in groups:
                    if nside_b == owner.nside:
                        continue
                    level = coarse.index(nside_b)
                    shift = starts[level] + treecode.cell_offsets[level, i]
                    if shift:
                        pair_inds[i][:, edges[b0] : edges[b1]] += owner.index_dtype.type(
                            shift
                        )

        owner.pair_inds = pair_inds
        owner.pair_exp2phi = pair_exp2phi
        owner.bins = bins
        owner._treecode = treecode
        owner.packed_pairs = None
        owner.packed_block_ids = None
        owner.packed_block_sizes = None
        if getattr(owner, "pack_host_pairs", False):
            PairGeometry.pack_host_pairs(owner)
        owner._invalidate_prepared_state()

    @staticmethod
    def pack_host_pairs(owner: "Correlation") -> None:
        """Replace the exact host pair geometry by its packed form (8 B/pair).

        Packing runs on the *global ids* rather than the device rows used in
        :meth:`Correlation.prepare`; both orderings agree (``map_inds`` is
        ascending and virtual rows are appended in id order), so the packed
        payload is identical and the row blocks stay mask-independent -- a
        packed pair file can still be loaded into any matching mask.  Each
        patch is released as soon as it is packed, so the exact and the
        packed geometry never coexist for the whole catalogue.
        """
        groups = level_groups(owner.level_nside)
        packed_pairs: List[np.ndarray] = []
        block_ids: List[np.ndarray] = []
        block_sizes: List[np.ndarray] = []
        empty = np.zeros(0, dtype=owner.index_dtype)
        for i in range(owner.n_patches):
            packed_i, blocks, _ = pack_patch(
                owner.pair_inds[i], owner.pair_exp2phi[i], owner.bins[i], groups
            )
            packed_pairs.append(packed_i)
            block_ids.append(
                np.concatenate(blocks).astype(owner.index_dtype, copy=False)
                if blocks
                else empty
            )
            block_sizes.append(np.array([b.size for b in blocks], dtype=np.int64))
            # free the exact arrays of this patch before packing the next
            owner.pair_inds[i] = None
            owner.pair_exp2phi[i] = None
        owner.packed_pairs = packed_pairs
        owner.packed_block_ids = block_ids
        owner.packed_block_sizes = block_sizes
        owner.pair_inds = None
        owner.pair_exp2phi = None

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
        ap_cells = owner._aperture_cells()
        if ap_cells is not None:
            # Coarse aperture level: the statistic of the (weighted-mean)
            # degraded map at ``aperture_nside``.  A coarse pixel is observed
            # if any of its children is; indices are virtual-row ids.
            return PairGeometry._get_pairs_M_a_coarse(
                owner, i, vec, ap_cells[0], aperture_filter
            )
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
    def _get_pairs_M_a_coarse(
        owner: "Correlation",
        i: int,
        vec: np.ndarray,
        cell_pix: np.ndarray,
        aperture_filter: Optional[Callable[..., Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        nside_ap = int(owner.aperture_nside)
        pix_center = hp.ang2pix(nside_ap, owner.theta_center[i], owner.phi_center[i])
        disc = hp.query_disc(
            nside_ap, vec=vec, radius=np.radians(5 * owner.theta_Q / 60)
        )
        disc = disc[disc != pix_center]
        pos = np.searchsorted(cell_pix, disc)
        pos = np.minimum(pos, max(cell_pix.size - 1, 0))
        observed = (
            cell_pix[pos] == disc if cell_pix.size else np.zeros(disc.size, dtype=bool)
        )
        qpix = disc[observed]
        cell_index = pos[observed]

        ra_center, dec_center = pixel2RaDec([pix_center], nside_ap)
        q_ra, q_dec = pixel2RaDec(qpix, nside_ap)
        q_cos, q_sin, q_val = PairGeometry.get_pairs_patch_M_a(
            owner, q_ra, q_dec, ra_center, dec_center, aperture_filter=aperture_filter
        )
        q_patch_area = owner.rotation_dtype.type(qpix.size * hp.nside2pixarea(nside_ap))
        npix = hp.nside2npix(owner.nside)
        return (
            q_cos.astype(owner.rotation_dtype, copy=False),
            q_sin.astype(owner.rotation_dtype, copy=False),
            q_val.astype(owner.rotation_dtype, copy=False),
            (cell_index + npix).astype(owner.index_dtype, copy=False),
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
