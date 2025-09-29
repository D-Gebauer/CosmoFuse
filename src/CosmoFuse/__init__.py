"""
CosmoFuse: A package for efficiently measuring integrated 3-point correlation functions.

This package provides tools for calculating integrated 3-point correlation functions
on GPU/CPU, with support for shear measurements and aperture mass calculations.
"""

from .correlation_helpers import zeta
from .correlations import Correlation
from .correlations_GPU import Correlation_GPU
from .utils import pixel2RaDec, set_mpl_params
from .visualisation import contours, make_corner_plot

__version__ = "0.2.0"
__author__ = "David Gebauer"
__email__ = "git@gebauer.ai"

__all__ = [
    "Correlation",
    "Correlation_GPU",
    "zeta",
    "pixel2RaDec",
    "set_mpl_params",
    "contours",
    "make_corner_plot",
]
