"""
CosmoFuse: GPU/CPU measurement of integrated, two- and three-point correlation functions.

This package provides tools for calculating integrated 3-point correlation functions
on GPU/CPU, with support for shear measurements and aperture mass calculations.
"""

from .correlation_helpers import (
    Q_crittenden,
    Q_schneider,
    U_crittenden,
    U_schneider,
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
from .correlations import Integrated3PCF
from .global_2pcf import Global2PCF
from .global_3pcf import Global3PCF
from .pipeline import MapFileLoader, MapLoader
from .results import Component, IntegratedResult, ThreePointResult, TwoPointResult
from .treecode import DEFAULT_RESOLUTION_FACTOR
from .utils import pixel2RaDec, select_patch_centers
from .zeta_writer import ZetaWriter

__version__ = "7.0.0"
__author__ = "David Gebauer"
__email__ = "git@gebauer.ai"

__all__ = [
    "Component",
    "Global2PCF",
    "Global3PCF",
    "Integrated3PCF",
    "IntegratedResult",
    "ThreePointResult",
    "TwoPointResult",
    "DEFAULT_RESOLUTION_FACTOR",
    "MapLoader",
    "MapFileLoader",
    "Q_crittenden",
    "Q_schneider",
    "U_crittenden",
    "U_schneider",
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
    "ZetaWriter",
]
