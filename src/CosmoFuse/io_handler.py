from typing import Optional, TYPE_CHECKING

import h5py
import healpy as hp
import numpy as np
import warnings

from .treecode import TreecodeGeometry, assign_levels

if TYPE_CHECKING:
    from .correlations import Correlation


class PairIOHandler:
    """HDF5 persistence for precomputed pair geometry.

    Format version 2 (written by :meth:`save_pairs`) stores consolidated
    flat datasets with per-patch offset arrays, so loading is a handful
    of bulk reads instead of ~8 small datasets per patch.  Files written
    by older versions (one group per patch) are still readable.

    Format version 3 is written only when virtual rows exist (static
    treecode and/or a coarse aperture level).  Its index datasets use
    *different names* (``tc_pair_inds``, ``tc_Q_inds``): CosmoFuse <= 4.20
    accepts any ``format_version >= 2`` and would otherwise read virtual-row
    ids as pixel ids (out-of-bounds gathers on the GPU); with the new names
    old readers fail with a ``KeyError`` instead.  Full-resolution files are
    still written as version 2 and stay readable by old versions.

    Format version 4 is written only for ``pack_host_pairs=True``: the pair
    payload is the quantised 8 B/pair form (``packed_pairs`` plus the
    per-(patch, level) row blocks in ``packed_block_ids``) instead of the
    exact indices and rotation factors.  Such a file defines a slightly
    different estimator, so it gets its own version rather than sneaking
    past a reader that expects exact geometry.
    """

    FORMAT_VERSION = 4
    PACKED_FORMAT_VERSION = 4
    TREECODE_FORMAT_VERSION = 3
    FULL_RESOLUTION_FORMAT_VERSION = 2

    @staticmethod
    def save_pairs(owner: "Correlation", filepath: str) -> None:
        packed = getattr(owner, "packed_pairs", None) is not None
        if owner.bins is None or (
            not packed and (owner.pair_inds is None or owner.pair_exp2phi is None)
        ):
            warnings.warn(
                "Cannot save pairs because host pair arrays were released. "
                "Reload or recompute pairs before calling save_pairs().",
                RuntimeWarning,
            )
            return

        n_patches = owner.n_patches
        if packed:
            pair_counts = np.array(
                [owner.packed_pairs[i].shape[0] for i in range(n_patches)],
                dtype=np.int64,
            )
        else:
            pair_counts = np.array(
                [owner.pair_inds[i].shape[1] for i in range(n_patches)], dtype=np.int64
            )
        pair_offsets = np.zeros(n_patches + 1, dtype=np.int64)
        pair_offsets[1:] = np.cumsum(pair_counts)
        total_pairs = int(pair_offsets[-1])

        q_counts = np.array(
            [np.asarray(owner.Q_inds[i]).size for i in range(n_patches)],
            dtype=np.int64,
        )
        q_offsets = np.zeros(n_patches + 1, dtype=np.int64)
        q_offsets[1:] = np.cumsum(q_counts)
        total_q = int(q_offsets[-1])

        treecode = getattr(owner, "_treecode", None)
        virtual_rows = treecode is not None or owner.aperture_nside is not None
        pair_name = "tc_pair_inds" if virtual_rows else "pair_inds"
        q_name = "tc_Q_inds" if virtual_rows else "Q_inds"

        if packed:
            version = PairIOHandler.PACKED_FORMAT_VERSION
        elif virtual_rows:
            version = PairIOHandler.TREECODE_FORMAT_VERSION
        else:
            version = PairIOHandler.FULL_RESOLUTION_FORMAT_VERSION

        with h5py.File(filepath, "w") as fp:
            fp.attrs["format_version"] = version
            fp.attrs["packed_pairs"] = bool(packed)
            # Provenance of the estimator (ignored by old readers).
            if owner.resolution_factor is not None:
                fp.attrs["resolution_factor"] = float(owner.resolution_factor)
            fp.attrs["aperture_nside"] = int(owner.aperture_nside or owner.nside)
            fp.create_dataset(
                "level_nside", data=np.asarray(owner.level_nside, dtype=np.int64)
            )
            if treecode is not None:
                gp = fp.create_group("treecode")
                gp.attrs["base_nside"] = int(treecode.base_nside)
                gp.attrs["n_aperture_cells"] = int(owner.n_aperture_cells)
                gp.create_dataset(
                    "coarse_nsides", data=np.asarray(treecode.coarse_nsides, dtype=np.int64)
                )
                gp.create_dataset("cell_offsets", data=treecode.cell_offsets)
                for level in range(treecode.n_levels):
                    gp.create_dataset(f"child_indptr_{level}", data=treecode.child_indptr[level])
                    gp.create_dataset(f"child_indices_{level}", data=treecode.child_indices[level])
                    gp.create_dataset(f"cell_ra_{level}", data=treecode.cell_ra[level])
                    gp.create_dataset(f"cell_dec_{level}", data=treecode.cell_dec[level])
            fp.attrs["nside"] = owner.nside
            fp.attrs["nbins"] = owner.nbins
            fp.attrs["theta_min"] = owner.theta_min
            fp.attrs["theta_max"] = owner.theta_max
            fp.attrs["patch_size"] = owner.patch_size
            fp.attrs["theta_Q"] = owner.theta_Q
            fp.attrs["n_patches"] = n_patches
            fp.create_dataset("map_inds", data=owner.map_inds)
            fp.create_dataset("phi_center", data=owner.phi_center)
            fp.create_dataset("theta_center", data=owner.theta_center)

            fp.create_dataset("pair_offsets", data=pair_offsets)
            fp.create_dataset("q_offsets", data=q_offsets)

            bins_arr = np.zeros((n_patches, owner.nbins), dtype=owner.index_dtype)
            for i in range(n_patches):
                bins_arr[i] = owner.bins[i]
            fp.create_dataset("bins", data=bins_arr)

            if packed:
                PairIOHandler._save_packed_pairs(
                    owner, fp, pair_offsets, total_pairs
                )
            else:
                d_inds = fp.create_dataset(
                    pair_name, shape=(2, total_pairs), dtype=owner.pair_inds[0].dtype
                )
                d_exp = fp.create_dataset(
                    "pair_exp2phi",
                    shape=(2, total_pairs),
                    dtype=owner.pair_exp2phi[0].dtype,
                )
                for i in range(n_patches):
                    start, stop = pair_offsets[i], pair_offsets[i + 1]
                    if stop > start:
                        d_inds[:, start:stop] = owner.pair_inds[i]
                        d_exp[:, start:stop] = owner.pair_exp2phi[i]

            q_inds_dtype = np.asarray(owner.Q_inds[0]).dtype if n_patches else owner.index_dtype
            q_val_dtype = np.asarray(owner.Q_val[0]).dtype if n_patches else owner.rotation_dtype
            d_qi = fp.create_dataset(q_name, shape=(total_q,), dtype=q_inds_dtype)
            d_qc = fp.create_dataset("Q_cos", shape=(total_q,), dtype=q_val_dtype)
            d_qs = fp.create_dataset("Q_sin", shape=(total_q,), dtype=q_val_dtype)
            d_qv = fp.create_dataset("Q_val", shape=(total_q,), dtype=q_val_dtype)
            for i in range(n_patches):
                start, stop = q_offsets[i], q_offsets[i + 1]
                if stop > start:
                    d_qi[start:stop] = owner.Q_inds[i]
                    d_qc[start:stop] = owner.Q_cos[i]
                    d_qs[start:stop] = owner.Q_sin[i]
                    d_qv[start:stop] = owner.Q_val[i]

            fp.create_dataset(
                "Q_patch_area",
                data=np.asarray(owner.Q_patch_area, dtype=owner.rotation_dtype),
            )

    @staticmethod
    def _save_packed_pairs(
        owner: "Correlation",
        fp: "h5py.File",
        pair_offsets: np.ndarray,
        total_pairs: int,
    ) -> None:
        """Write the ``pack_host_pairs`` payload (format version 4).

        The row blocks hold *global* ids, exactly as in memory, so a packed
        file stays independent of the row space and can be sliced by patch
        like any other.
        """
        n_patches = owner.n_patches
        if n_patches:
            block_sizes = np.stack(
                [np.asarray(s, dtype=np.int64) for s in owner.packed_block_sizes]
            )
        else:
            block_sizes = np.zeros((0, 0), dtype=np.int64)
        block_offsets = np.zeros(n_patches + 1, dtype=np.int64)
        block_offsets[1:] = np.cumsum(block_sizes.sum(axis=1))

        d_packed = fp.create_dataset(
            "packed_pairs", shape=(total_pairs, 4), dtype=np.uint16
        )
        d_blocks = fp.create_dataset(
            "packed_block_ids",
            shape=(int(block_offsets[-1]),),
            dtype=owner.index_dtype,
        )
        for i in range(n_patches):
            start, stop = int(pair_offsets[i]), int(pair_offsets[i + 1])
            if stop > start:
                d_packed[start:stop] = owner.packed_pairs[i]
            bstart, bstop = int(block_offsets[i]), int(block_offsets[i + 1])
            if bstop > bstart:
                d_blocks[bstart:bstop] = owner.packed_block_ids[i]
        fp.create_dataset("packed_block_sizes", data=block_sizes)
        fp.create_dataset("packed_block_offsets", data=block_offsets)

    @staticmethod
    def load_pairs(
        owner: "Correlation",
        filepath: str,
        start_ind: int = 0,
        stop_ind: Optional[int] = None,
        release_host_pairs: bool = False,
    ) -> None:
        owner._invalidate_prepared_state()
        # the file decides which representation the instance holds
        owner.packed_pairs = None
        owner.packed_block_ids = None
        owner.packed_block_sizes = None

        with h5py.File(filepath, "r") as fp:
            version = int(fp.attrs.get("format_version", 1))
            if version > PairIOHandler.FORMAT_VERSION:
                raise ValueError(
                    f"{filepath} has pair-file format version {version}; this "
                    f"CosmoFuse reads versions <= {PairIOHandler.FORMAT_VERSION}. "
                    "Update CosmoFuse."
                )
            if stop_ind is None:
                stop_ind = fp.attrs["n_patches"]
            owner.nside = fp.attrs["nside"]
            owner.nbins = fp.attrs["nbins"]
            owner.theta_min = fp.attrs["theta_min"]
            owner.theta_max = fp.attrs["theta_max"]
            owner.binedges = np.geomspace(owner.theta_min, owner.theta_max, owner.nbins + 1)
            owner.bincenters = (
                np.sqrt(owner.binedges[1:] * owner.binedges[:-1]) * 60 * 180 / np.pi
            )
            owner.patch_size = fp.attrs["patch_size"]
            owner.theta_Q = fp.attrs["theta_Q"]
            owner.n_patches = stop_ind - start_ind
            owner.map_inds = fp["map_inds"][:].astype(owner.index_dtype, copy=False)
            owner.map_mask = np.zeros(hp.nside2npix(owner.nside), dtype=bool)
            owner.map_mask[owner.map_inds] = True
            owner.phi_center = fp["phi_center"][start_ind:stop_ind]
            owner.theta_center = fp["theta_center"][start_ind:stop_ind]

            PairIOHandler._load_resolution(owner, fp, filepath)
            if version >= 2:
                PairIOHandler._load_pairs_v2(owner, fp, start_ind, stop_ind)
            else:
                PairIOHandler._load_pairs_legacy(owner, fp, start_ind, stop_ind)
        owner.prepare(release_host_pairs=release_host_pairs)

    @staticmethod
    def _load_resolution(owner: "Correlation", fp: "h5py.File", filepath: str) -> None:
        """Adopt the estimator definition stored in the file.

        The file is authoritative for the geometry (as for nside, bins, ...).
        An *explicitly requested* resolution that contradicts the file is an
        error; silently measuring a different estimator than asked for is
        not acceptable.
        """
        file_k = fp.attrs.get("resolution_factor", None)
        file_k = None if file_k is None else float(file_k)
        file_ap = int(fp.attrs.get("aperture_nside", owner.nside))
        file_ap = None if file_ap == int(owner.nside) else file_ap
        if "level_nside" in fp:
            file_levels = fp["level_nside"][:].astype(np.int64)
        else:
            file_levels = assign_levels(owner.binedges, owner.nside, None)

        asked_k, asked_ap = owner.resolution_factor, owner.aperture_nside
        asked_levels = assign_levels(owner.binedges, owner.nside, asked_k)
        if asked_k is not None and not np.array_equal(asked_levels, file_levels):
            raise ValueError(
                f"{filepath} was built with resolution_factor={file_k} "
                f"(nside per bin {file_levels.tolist()}), but this Correlation "
                f"asks for resolution_factor={asked_k} "
                f"(nside per bin {asked_levels.tolist()})."
            )
        if asked_ap is not None and asked_ap != file_ap:
            raise ValueError(
                f"{filepath} was built with aperture_nside="
                f"{file_ap or int(owner.nside)}, but this Correlation asks for "
                f"aperture_nside={asked_ap}."
            )
        if (asked_k is None and file_k is not None) or (
            asked_ap is None and file_ap is not None
        ):
            warnings.warn(
                f"{filepath} defines a static-treecode estimator "
                f"(resolution_factor={file_k}, aperture_nside="
                f"{file_ap or int(owner.nside)}); adopting it. Pass the same "
                "values to the constructor to silence this warning.",
                UserWarning,
                stacklevel=3,
            )
        owner.resolution_factor = file_k
        owner.aperture_nside = file_ap
        owner.level_nside = file_levels
        owner._treecode = None
        # bins / resolution may have changed: keep the pair finder in sync
        owner._pair_finder = owner._make_pair_finder()

    @staticmethod
    def _load_pairs_v2(
        owner: "Correlation", fp: "h5py.File", start_ind: int, stop_ind: int
    ) -> None:
        pair_offsets = fp["pair_offsets"][:]
        q_offsets = fp["q_offsets"][:]

        p0, p1 = int(pair_offsets[start_ind]), int(pair_offsets[stop_ind])
        q0, q1 = int(q_offsets[start_ind]), int(q_offsets[stop_ind])

        packed = "packed_pairs" in fp
        virtual_rows = "tc_pair_inds" in fp or (packed and "treecode" in fp)
        pair_name = "tc_pair_inds" if "tc_pair_inds" in fp else "pair_inds"
        q_name = "tc_Q_inds" if virtual_rows else "Q_inds"

        # Bulk reads straight into the final flat arrays
        if packed:
            PairIOHandler._load_packed_pairs(owner, fp, start_ind, stop_ind)
        else:
            pair_inds_flat = fp[pair_name][:, p0:p1].astype(
                owner.index_dtype, copy=False
            )
            if "treecode" in fp:
                pair_inds_flat = PairIOHandler._load_treecode(
                    owner, fp["treecode"], pair_inds_flat, start_ind, stop_ind
                )
            pair_exp2phi_flat = fp["pair_exp2phi"][:, p0:p1].astype(
                owner.rotation_complex_dtype, copy=False
            )
        bins_arr = fp["bins"][start_ind:stop_ind].astype(owner.index_dtype, copy=False)

        q_inds_flat = fp[q_name][q0:q1].astype(owner.index_dtype, copy=False)
        q_cos_flat = fp["Q_cos"][q0:q1].astype(owner.rotation_dtype, copy=False)
        q_sin_flat = fp["Q_sin"][q0:q1].astype(owner.rotation_dtype, copy=False)
        q_val_flat = fp["Q_val"][q0:q1].astype(owner.rotation_dtype, copy=False)
        q_patch_area = fp["Q_patch_area"][start_ind:stop_ind].astype(
            owner.rotation_dtype, copy=False
        )

        # Per-patch host lists are zero-copy views into the flat arrays,
        # keeping the same object model as the legacy path.
        if packed:
            owner.pair_inds = None
            owner.pair_exp2phi = None
        else:
            owner.pair_inds = []
            owner.pair_exp2phi = []
        owner.bins = []
        owner.Q_inds = []
        owner.Q_cos = []
        owner.Q_sin = []
        owner.Q_val = []
        owner.Q_patch_area = []
        for i in range(start_ind, stop_ind):
            ps, pe = int(pair_offsets[i]) - p0, int(pair_offsets[i + 1]) - p0
            qs, qe = int(q_offsets[i]) - q0, int(q_offsets[i + 1]) - q0
            if not packed:
                owner.pair_inds.append(pair_inds_flat[:, ps:pe])
                owner.pair_exp2phi.append(pair_exp2phi_flat[:, ps:pe])
            owner.bins.append(bins_arr[i - start_ind])
            owner.Q_inds.append(q_inds_flat[qs:qe])
            owner.Q_cos.append(q_cos_flat[qs:qe])
            owner.Q_sin.append(q_sin_flat[qs:qe])
            owner.Q_val.append(q_val_flat[qs:qe])
            owner.Q_patch_area.append(owner.rotation_dtype.type(q_patch_area[i - start_ind]))

        # The flat aperture arrays are already exactly what
        # _prepare_aperture_flat would build — set them directly instead of
        # re-copying from the per-patch views.
        local_q_offsets = (q_offsets[start_ind : stop_ind + 1] - q0).astype(np.int64)
        owner.Q_inds_flat = q_inds_flat
        owner.Q_cos_flat = q_cos_flat
        owner.Q_sin_flat = q_sin_flat
        owner.Q_val_flat = q_val_flat
        owner.Q_offsets = local_q_offsets
        owner.Q_patch_area_flat = np.asarray(q_patch_area, dtype=owner.rotation_dtype)
        owner._invalidate_aperture_device_buffers()

    @staticmethod
    def _load_packed_pairs(
        owner: "Correlation", fp: "h5py.File", start_ind: int, stop_ind: int
    ) -> None:
        """Read the packed payload of patches ``start_ind:stop_ind``.

        Only the row blocks carry global ids, so the treecode renumbering of
        a sliced load applies to them instead of to the pairs.
        """
        pair_offsets = fp["pair_offsets"][:]
        block_offsets = fp["packed_block_offsets"][:]
        p0, p1 = int(pair_offsets[start_ind]), int(pair_offsets[stop_ind])
        b0, b1 = int(block_offsets[start_ind]), int(block_offsets[stop_ind])

        packed_flat = fp["packed_pairs"][p0:p1]
        block_ids_flat = fp["packed_block_ids"][b0:b1].astype(
            owner.index_dtype, copy=False
        )
        block_sizes = fp["packed_block_sizes"][start_ind:stop_ind].astype(np.int64)
        if "treecode" in fp:
            block_ids_flat = PairIOHandler._load_treecode(
                owner, fp["treecode"], block_ids_flat, start_ind, stop_ind
            )

        owner.pack_host_pairs = True
        owner.packed_pairs = []
        owner.packed_block_ids = []
        owner.packed_block_sizes = []
        for i in range(start_ind, stop_ind):
            ps, pe = int(pair_offsets[i]) - p0, int(pair_offsets[i + 1]) - p0
            bs, be = int(block_offsets[i]) - b0, int(block_offsets[i + 1]) - b0
            owner.packed_pairs.append(packed_flat[ps:pe])
            owner.packed_block_ids.append(block_ids_flat[bs:be])
            owner.packed_block_sizes.append(block_sizes[i - start_ind])

    @staticmethod
    def _load_treecode(
        owner: "Correlation",
        gp: "h5py.Group",
        pair_inds_flat: np.ndarray,
        start_ind: int,
        stop_ind: int,
    ) -> np.ndarray:
        """Read the coarse-cell geometry of patches ``start_ind:stop_ind`` and
        renumber the virtual-row pair ids of that slice."""
        n_levels = int(gp["coarse_nsides"].shape[0])
        full = TreecodeGeometry(
            base_nside=int(gp.attrs["base_nside"]),
            coarse_nsides=tuple(int(n) for n in gp["coarse_nsides"][:]),
            cell_offsets=gp["cell_offsets"][:].astype(np.int64),
            child_indptr=[gp[f"child_indptr_{lv}"][:] for lv in range(n_levels)],
            child_indices=[gp[f"child_indices_{lv}"][:] for lv in range(n_levels)],
            cell_ra=[gp[f"cell_ra_{lv}"][:] for lv in range(n_levels)],
            cell_dec=[gp[f"cell_dec_{lv}"][:] for lv in range(n_levels)],
        )
        n_ap = int(gp.attrs["n_aperture_cells"])
        if n_ap != owner.n_aperture_cells:
            raise ValueError(
                "The aperture level of the pair file does not match this mask "
                f"({n_ap} vs {owner.n_aperture_cells} coarse aperture pixels)."
            )
        sliced = full.slice_patches(start_ind, stop_ind)
        owner._treecode = sliced
        if start_ind == 0 and stop_ind == full.n_patches:
            return pair_inds_flat

        npix = hp.nside2npix(owner.nside)
        old_starts = full.level_starts(first=n_ap)
        new_starts = sliced.level_starts(first=n_ap)
        out = pair_inds_flat.astype(np.int64)
        virtual = out >= npix + n_ap
        a = out[virtual] - npix
        level = np.searchsorted(old_starts, a, side="right") - 1
        shift = new_starts[level] - old_starts[level] - full.cell_offsets[level, start_ind]
        out[virtual] = a + shift + npix
        return out.astype(owner.index_dtype, copy=False)

    @staticmethod
    def _load_pairs_legacy(
        owner: "Correlation", fp: "h5py.File", start_ind: int, stop_ind: int
    ) -> None:
        owner.pair_inds = []
        owner.pair_exp2phi = []
        owner.bins = []
        owner.Q_inds = []
        owner.Q_cos = []
        owner.Q_sin = []
        owner.Q_val = []
        owner.Q_patch_area = []

        for i in range(start_ind, stop_ind):
            gp = fp[f"patch_{i:02d}"]
            owner.pair_inds.append(
                gp["pair_inds"][:].astype(owner.index_dtype, copy=False)
            )
            owner.pair_exp2phi.append(
                gp["pair_exp2phi"][:].astype(owner.rotation_complex_dtype, copy=False)
            )
            owner.bins.append(gp["bins"][:].astype(owner.index_dtype, copy=False))
            owner.Q_inds.append(gp["Q_inds"][:].astype(owner.index_dtype, copy=False))
            owner.Q_cos.append(gp["Q_cos"][:].astype(owner.rotation_dtype, copy=False))
            owner.Q_sin.append(gp["Q_sin"][:].astype(owner.rotation_dtype, copy=False))
            owner.Q_val.append(gp["Q_val"][:].astype(owner.rotation_dtype, copy=False))
            owner.Q_patch_area.append(owner.rotation_dtype.type(gp["Q_patch_area"][()]))
        owner._prepare_aperture_flat()
