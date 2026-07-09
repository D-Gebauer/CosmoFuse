from typing import Optional, TYPE_CHECKING

import h5py
import healpy as hp
import numpy as np
import warnings

if TYPE_CHECKING:
    from .correlations import Correlation


class PairIOHandler:
    """HDF5 persistence for precomputed pair geometry.

    Format version 2 (written by :meth:`save_pairs`) stores consolidated
    flat datasets with per-patch offset arrays, so loading is a handful
    of bulk reads instead of ~8 small datasets per patch.  Files written
    by older versions (one group per patch) are still readable.
    """

    FORMAT_VERSION = 2

    @staticmethod
    def save_pairs(owner: "Correlation", filepath: str) -> None:
        if (
            owner.pair_inds is None
            or owner.pair_exp2phi is None
            or owner.bins is None
        ):
            warnings.warn(
                "Cannot save pairs because host pair arrays were released. "
                "Reload or recompute pairs before calling save_pairs().",
                RuntimeWarning,
            )
            return

        n_patches = owner.n_patches
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

        with h5py.File(filepath, "w") as fp:
            fp.attrs["format_version"] = PairIOHandler.FORMAT_VERSION
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

            d_inds = fp.create_dataset(
                "pair_inds", shape=(2, total_pairs), dtype=owner.pair_inds[0].dtype
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
            d_qi = fp.create_dataset("Q_inds", shape=(total_q,), dtype=q_inds_dtype)
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
    def load_pairs(
        owner: "Correlation",
        filepath: str,
        start_ind: int = 0,
        stop_ind: Optional[int] = None,
        release_host_pairs: bool = False,
    ) -> None:
        owner._invalidate_prepared_state()

        with h5py.File(filepath, "r") as fp:
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

            if int(fp.attrs.get("format_version", 1)) >= 2:
                PairIOHandler._load_pairs_v2(owner, fp, start_ind, stop_ind)
            else:
                PairIOHandler._load_pairs_legacy(owner, fp, start_ind, stop_ind)
        owner.prepare(release_host_pairs=release_host_pairs)

    @staticmethod
    def _load_pairs_v2(
        owner: "Correlation", fp: "h5py.File", start_ind: int, stop_ind: int
    ) -> None:
        pair_offsets = fp["pair_offsets"][:]
        q_offsets = fp["q_offsets"][:]

        p0, p1 = int(pair_offsets[start_ind]), int(pair_offsets[stop_ind])
        q0, q1 = int(q_offsets[start_ind]), int(q_offsets[stop_ind])

        # Bulk reads straight into the final flat arrays
        pair_inds_flat = fp["pair_inds"][:, p0:p1].astype(owner.index_dtype, copy=False)
        pair_exp2phi_flat = fp["pair_exp2phi"][:, p0:p1].astype(
            owner.rotation_complex_dtype, copy=False
        )
        bins_arr = fp["bins"][start_ind:stop_ind].astype(owner.index_dtype, copy=False)

        q_inds_flat = fp["Q_inds"][q0:q1].astype(owner.index_dtype, copy=False)
        q_cos_flat = fp["Q_cos"][q0:q1].astype(owner.rotation_dtype, copy=False)
        q_sin_flat = fp["Q_sin"][q0:q1].astype(owner.rotation_dtype, copy=False)
        q_val_flat = fp["Q_val"][q0:q1].astype(owner.rotation_dtype, copy=False)
        q_patch_area = fp["Q_patch_area"][start_ind:stop_ind].astype(
            owner.rotation_dtype, copy=False
        )

        # Per-patch host lists are zero-copy views into the flat arrays,
        # keeping the same object model as the legacy path.
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
