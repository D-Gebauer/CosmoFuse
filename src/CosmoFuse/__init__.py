"""
CosmoFuse: A package for efficiently measuring integrated 3-point correlation functions.

This package provides tools for calculating integrated 3-point correlation functions
on GPU/CPU, with support for shear measurements and aperture mass calculations.
"""

from .correlation_helpers import (
    calculate_all_zetas,
    zeta_shear,
    zeta_clust,
    zeta_ggl,
)
from .correlations import Correlation
from .utils import pixel2RaDec

__version__ = "4.1.0"
__author__ = "David Gebauer"
__email__ = "git@gebauer.ai"

__all__ = [
    "Correlation",
    "calculate_all_zetas",
    "zeta_shear",
    "zeta_clust",
    "zeta_ggl",
    "pixel2RaDec",
]
