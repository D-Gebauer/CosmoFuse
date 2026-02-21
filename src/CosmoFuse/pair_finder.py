from typing import Any, Callable, List, Tuple

import numpy as np


class PairFinder:
    """Geometry helper responsible for patch-level pair finding."""

    def __init__(
        self,
        *,
        nbins: int,
        binedges: np.ndarray,
        index_dtype: np.dtype,
        rotation_dtype: np.dtype,
        rotation_complex_dtype: np.dtype,
        kernel: Callable[..., Tuple[np.ndarray, ...]],
        resolve_angle_method_code: Callable[[str], int],
    ) -> None:
        self.nbins = nbins
        self.binedges = binedges
        self.index_dtype = index_dtype
        self.rotation_dtype = rotation_dtype
        self.rotation_complex_dtype = rotation_complex_dtype
        self.kernel = kernel
        self.resolve_angle_method_code = resolve_angle_method_code

    def get_pairs_patch(
        self,
        patch_inds: np.ndarray,
        ra: np.ndarray,
        dec: np.ndarray,
        angle_method: str = "haversine",
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        ra_local = np.asarray(ra, dtype=self.rotation_dtype)
        dec_local = np.asarray(dec, dtype=self.rotation_dtype)
        binedges_local = np.asarray(self.binedges, dtype=self.rotation_dtype)
        patch_inds_local = np.asarray(patch_inds, dtype=self.index_dtype)
        angle_method_code = self.resolve_angle_method_code(angle_method)

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
        ) = self.kernel(
            patch_inds_local,
            ra_local,
            dec_local,
            binedges_local,
            angle_method_code,
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
