"""
CosmoFuse: A package for efficiently measuring integrated 3-point correlation functions.

This package provides tools for calculating integrated 3-point correlation functions
on GPU/CPU, with support for shear measurements and aperture mass calculations.
"""

from .correlation_helpers import (
    Q_T,
    Q_crittenden,
    Q_schneider,
    calculate_all_zetas,
    zeta_a_g,
    zeta_a_minus,
    zeta_a_plus,
    zeta_a_t,
    zeta_g_g,
    zeta_g_minus,
    zeta_g_plus,
    zeta_g_t,
)
from .correlations import Correlation
from .pipeline import PinnedMapPipeline
from .utils import pixel2RaDec, select_patch_centers

__version__ = "4.20.0"
__author__ = "David Gebauer"
__email__ = "git@gebauer.ai"

__all__ = [
    "Correlation",
    "PinnedMapPipeline",
    "Q_T",
    "Q_crittenden",
    "Q_schneider",
    "calculate_all_zetas",
    "zeta_g_plus",
    "zeta_g_minus",
    "zeta_a_plus",
    "zeta_a_minus",
    "zeta_g_g",
    "zeta_a_g",
    "zeta_g_t",
    "zeta_a_t",
    "pixel2RaDec",
    "select_patch_centers",
]
