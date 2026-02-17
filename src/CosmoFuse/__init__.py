"""
CosmoFuse: A package for efficiently measuring integrated 3-point correlation functions.

This package provides tools for calculating integrated 3-point correlation functions
on GPU/CPU, with support for shear measurements and aperture mass calculations.
"""

from .correlation_helpers import zeta
from .correlations import Correlation
from .utils import pixel2RaDec

__version__ = "3.3.2"
__author__ = "David Gebauer"
__email__ = "git@gebauer.ai"

__all__ = [
    "Correlation",
    "zeta",
    "pixel2RaDec",
]
