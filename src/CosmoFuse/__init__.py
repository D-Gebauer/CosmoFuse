"""
CosmoFuse: GPU/CPU measurement of integrated, two- and three-point correlations.

Three estimators on HEALPix maps, named after the statistic they deliver:
:class:`Integrated3PCF` (per-patch aperture masses and 2PCFs, the i3PCF
ingredients), :class:`Global2PCF` (footprint-wide 2PCFs) and
:class:`Global3PCF` (footprint-wide 3PCF multipoles).  Methods are TreeCorr
field codes (``g`` = shear, ``k`` = scalar field): ``gg``, ``kk``, ``kg``,
``three_by_two_pt``, ``ggg``, ``kkk``, ``kgg``, ``kkg``.  Every call returns a
labelled result (:class:`IntegratedResult`, :class:`TwoPointResult`,
:class:`ThreePointResult`).
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

__version__ = "7.3.0"
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
