from typing import Tuple, Union

import healpy as hp
import numpy as np


def pixel2RaDec(
    pixel_indices: Union[int, np.ndarray], NSIDE: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert pixel indices to right ascension and declination.

    Args:
        pixel_indices: Pixel indices in the HEALPix map
        NSIDE: HEALPix resolution parameter

    Returns:
        Tuple of (ra, dec) in radians
    """
    if np.isscalar(pixel_indices):
        pix_for_healpy = int(pixel_indices)
    else:
        pix_arr = np.asarray(pixel_indices)
        if np.issubdtype(pix_arr.dtype, np.unsignedinteger):
            max_int64 = np.iinfo(np.int64).max
            if pix_arr.size > 0 and np.max(pix_arr) > max_int64:
                raise ValueError("pixel indices exceed int64 range")
        pix_for_healpy = pix_arr.astype(np.int64, copy=False)

    theta, phi = hp.pixelfunc.pix2ang(NSIDE, pix_for_healpy, nest=False)
    ra = phi
    dec = np.pi / 2.0 - theta
    return ra, dec
