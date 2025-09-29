from typing import Any, Callable, Tuple, Union

import healpy as hp
import matplotlib.pylab as pylab
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


def set_mpl_params() -> None:
    """Set matplotlib parameters for consistent plotting style."""
    params = {
        "figure.figsize": (12, 8),
        "legend.fontsize": "20",
        "axes.labelsize": "20",
        "axes.titlesize": "24",
        "xtick.labelsize": "18",
        "ytick.labelsize": "18",
    }

    pylab.rcParams.update(params)


def eval_func_tuple(f_args: Tuple[Callable, ...]) -> Any:
    """Takes a tuple of a function and args, evaluates and returns result.

    Args:
        f_args: Tuple where first element is a callable and remaining elements are arguments

    Returns:
        Result of calling the function with the provided arguments
    """
    return f_args[0](*f_args[1:])
