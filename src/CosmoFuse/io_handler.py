from typing import Optional, TYPE_CHECKING

import h5py
import healpy as hp
import numpy as np
import warnings

if TYPE_CHECKING:
    from .correlations import Correlation


class PairIOHandler:
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

        with h5py.File(filepath, "w") as fp:
            fp.attrs["nside"] = owner.nside
            fp.attrs["nbins"] = owner.nbins
            fp.attrs["theta_min"] = owner.theta_min
            fp.attrs["theta_max"] = owner.theta_max
            fp.attrs["patch_size"] = owner.patch_size
            fp.attrs["theta_Q"] = owner.theta_Q
            fp.attrs["n_patches"] = owner.n_patches
            fp.create_dataset("map_inds", data=owner.map_inds)
            fp.create_dataset("phi_center", data=owner.phi_center)
            fp.create_dataset("theta_center", data=owner.theta_center)

            for i in range(owner.n_patches):
                gp = fp.create_group(f"patch_{i:02d}")

                gp.create_dataset("pair_inds", data=owner.pair_inds[i])
                gp.create_dataset("pair_exp2phi", data=owner.pair_exp2phi[i])
                gp.create_dataset("bins", data=owner.bins[i])

                gp.create_dataset("Q_inds", data=owner.Q_inds[i])
                gp.create_dataset("Q_cos", data=owner.Q_cos[i])
                gp.create_dataset("Q_sin", data=owner.Q_sin[i])
                gp.create_dataset("Q_val", data=owner.Q_val[i])
                gp.create_dataset("Q_patch_area", data=owner.Q_patch_area[i])

    @staticmethod
    def load_pairs(
        owner: "Correlation",
        filepath: str,
        start_ind: int = 0,
        stop_ind: Optional[int] = None,
        release_host_pairs: bool = False,
    ) -> None:
        owner._invalidate_prepared_state()
        owner.pair_inds = []
        owner.pair_exp2phi = []
        owner.bins = []
        owner.Q_inds = []
        owner.Q_cos = []
        owner.Q_sin = []
        owner.Q_val = []
        owner.Q_patch_area = []

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
        owner.prepare(release_host_pairs=release_host_pairs)
