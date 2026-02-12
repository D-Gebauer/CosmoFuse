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
    theta, phi = hp.pixelfunc.pix2ang(NSIDE, pixel_indices, nest=False)
    ra = phi
    dec = np.pi / 2.0 - theta
    return ra, dec
