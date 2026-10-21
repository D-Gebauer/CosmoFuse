"""
Labelled results of the three estimators.

Every measurement returns an object whose fields are named after what they
hold, together with the row labels (which tomographic bins) and the angular
binning -- instead of a positional tuple whose order and row layout the
caller has to know.

* :class:`IntegratedResult` -- per-patch outputs of
  :class:`~CosmoFuse.correlations.Integrated3PCF` (the i3PCF ingredients);
* :class:`TwoPointResult` -- per-tile sums of
  :class:`~CosmoFuse.global_2pcf.Global2PCF`;
* :class:`ThreePointResult` -- multipole sums of
  :class:`~CosmoFuse.global_3pcf.Global3PCF`.

The two global results hold *sums* (numerators and weights), so results of
different tiles, chunks or devices add with ``+``; the estimate is formed on
access.  Array fields may live on the host (numpy) or the device (cupy),
following the measurement's ``return_device``.
"""

from dataclasses import dataclass
from dataclasses import fields as dc_fields
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .backend import safe_divide
from .correlation_helpers import _xp

Combinations = Tuple[Tuple[int, int], ...]

# Field order of IntegratedResult (and of ZetaWriter datasets): the central
# aperture fields, then the annular 2PCFs.
INTEGRATED_FIELDS: Tuple[str, ...] = ("M_a", "M_g", "xi_p", "xi_m", "xi_g", "xi_t")


def _theta_arcmin(binedges: np.ndarray) -> np.ndarray:
    e = np.degrees(np.asarray(binedges, dtype=np.float64)) * 60.0
    return np.sqrt(e[1:] * e[:-1])


# ── Integrated3PCF ────────────────────────────────────────────────────────


@dataclass
class IntegratedResult:
    """Per-patch aperture masses and 2PCFs (the i3PCF ingredients).

    Fields that were not measured are ``None``.

    * ``M_a`` -- aperture mass of the shear, ``(nz_source, n_patches)``;
    * ``M_g`` -- aperture of the density, ``(nz_lens, n_patches)``;
    * ``xi_p``, ``xi_m`` -- ξ± per patch, ``(ncomb, n_patches, nbins)``, rows
      labelled by ``gg_combinations``;
    * ``xi_g`` -- w(θ) per patch, rows ``kk_combinations``;
    * ``xi_t`` -- γ_t per patch, rows ``kg_combinations`` = (lens, source).
    """

    binedges: np.ndarray
    M_a: Optional[Any] = None
    M_g: Optional[Any] = None
    xi_p: Optional[Any] = None
    xi_m: Optional[Any] = None
    xi_g: Optional[Any] = None
    xi_t: Optional[Any] = None
    gg_combinations: Optional[Combinations] = None
    kk_combinations: Optional[Combinations] = None
    kg_combinations: Optional[Combinations] = None

    @property
    def theta(self) -> np.ndarray:
        """Geometric bin centres (arcmin)."""
        return _theta_arcmin(self.binedges)

    def fields(self) -> Dict[str, Any]:
        """The measured arrays, by name, in the order of ``INTEGRATED_FIELDS``."""
        return {
            k: getattr(self, k)
            for k in INTEGRATED_FIELDS
            if getattr(self, k) is not None
        }

    def zetas(self, **kwargs: Any) -> Dict[str, Any]:
        """Every i3PCF ζ the measured fields form (see
        :func:`~CosmoFuse.correlation_helpers.calculate_all_zetas`; extra
        keyword arguments are forwarded)."""
        from .correlation_helpers import calculate_all_zetas

        batched = {k: v[None, ...] for k, v in self.fields().items()}
        return {k: v[0] for k, v in calculate_all_zetas(**batched, **kwargs).items()}

    def to_numpy(self) -> "IntegratedResult":
        """A copy with every array on the host."""
        out = {}
        for f in dc_fields(self):
            v = getattr(self, f.name)
            if f.name in INTEGRATED_FIELDS and v is not None:
                v = (
                    v.get()
                    if hasattr(v, "get") and not isinstance(v, np.ndarray)
                    else np.asarray(v)
                )
            out[f.name] = v
        return IntegratedResult(**out)


# ── Global2PCF ────────────────────────────────────────────────────────────


@dataclass
class TwoPointResult:
    """A footprint-wide 2PCF as per-tile sums.

    ``statistic``: ``"gg"`` (fields ``xi_p``, ``xi_m``), ``"kk"`` (``xi``, =
    w(θ)) or ``"kg"`` (``xi``, = γ_t).  ``tile_num[name]`` and
    ``tile_weight``: ``(ncomb, n_tiles, nbins)``; rows labelled by
    ``combinations``.  The estimate is the ratio of the sums over tiles;
    :meth:`jackknife` gives the delete-one-tile estimates.
    """

    statistic: str
    binedges: np.ndarray
    combinations: Combinations
    tile_num: Dict[str, Any]
    tile_weight: Any

    @property
    def theta(self) -> np.ndarray:
        """Geometric bin centres (arcmin)."""
        return _theta_arcmin(self.binedges)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(self.tile_num)

    @property
    def n_tiles(self) -> int:
        return int(self.tile_weight.shape[1])

    @property
    def weight(self) -> Any:
        """Sum of pair weights per (combination, bin)."""
        xp = _xp(self.tile_weight)
        return self.tile_weight.astype(xp.float64).sum(axis=1)

    def num(self, name: str) -> Any:
        xp = _xp(self.tile_num[name])
        return self.tile_num[name].astype(xp.float64).sum(axis=1)

    def estimate(self, name: str) -> Any:
        """``sum_tiles num / sum_tiles weight`` (0 where no pairs)."""
        xp = _xp(self.tile_weight)
        return safe_divide(xp, self.num(name), self.weight)

    @property
    def xi_p(self) -> Any:
        return self._field("xi_p")

    @property
    def xi_m(self) -> Any:
        return self._field("xi_m")

    @property
    def xi(self) -> Any:
        return self._field("xi")

    def _field(self, name: str) -> Any:
        if name not in self.tile_num:
            raise AttributeError(
                f"a {self.statistic!r} result holds {self.names}, not {name!r}"
            )
        return self.estimate(name)

    def jackknife(self) -> Dict[str, Any]:
        """Delete-one-tile estimates, ``name -> (n_tiles, ncomb, nbins)``."""
        xp = _xp(self.tile_weight)
        w_all = self.weight
        w_t = self.tile_weight.astype(xp.float64)
        out = {}
        for name in self.names:
            n_all = self.num(name)
            n_t = self.tile_num[name].astype(xp.float64)
            num = n_all[:, None, :] - n_t
            den = w_all[:, None, :] - w_t
            out[name] = xp.moveaxis(safe_divide(xp, num, den), 1, 0)
        return out

    def to_numpy(self) -> "TwoPointResult":
        def host(a: Any) -> np.ndarray:
            return (
                a.get()
                if hasattr(a, "get") and not isinstance(a, np.ndarray)
                else np.asarray(a)
            )

        return TwoPointResult(
            self.statistic,
            self.binedges,
            self.combinations,
            {k: host(v) for k, v in self.tile_num.items()},
            host(self.tile_weight),
        )

    def __add__(self, other: "TwoPointResult") -> "TwoPointResult":
        if (
            other.statistic != self.statistic
            or other.combinations != self.combinations
            or not np.array_equal(other.binedges, self.binedges)
        ):
            raise ValueError("can only add results of the same statistic and binning")
        xp = _xp(self.tile_weight)
        return TwoPointResult(
            self.statistic,
            self.binedges,
            self.combinations,
            {
                k: xp.concatenate((v, other.tile_num[k]), axis=1)
                for k, v in self.tile_num.items()
            },
            xp.concatenate((self.tile_weight, other.tile_weight), axis=1),
        )


# ── Global3PCF ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Component:
    """One three-point function: which field sits at which vertex, and which
    vertices are complex conjugated.  ``fields = (centre, leg 2, leg 3)``."""

    label: str
    fields: Tuple[str, str, str]
    conj: Tuple[bool, bool, bool] = (False, False, False)


@dataclass
class ThreePointResult:
    """Multipoles of one or more three-point functions.

    ``num[label]`` and ``norm[label]``: complex arrays of shape
    ``(2 n_max + 1, nz_centre, nz_leg2, nz_leg3, nbins, nbins)`` -- axis 0 is
    ``n = -n_max .. n_max``, the last two are the radial bins of legs 2 and
    3.  Sums, not ratios: results of different tiles, chunks or devices add
    (``+``), and the estimate is formed in opening-angle bins
    (:meth:`binned`).
    """

    n_max: int
    binedges: np.ndarray
    components: Tuple[Component, ...]
    spins: Dict[str, int]
    num: Dict[str, np.ndarray]
    norm: Dict[str, np.ndarray]

    @property
    def n(self) -> np.ndarray:
        return np.arange(-self.n_max, self.n_max + 1)

    @property
    def theta(self) -> np.ndarray:
        """Geometric radial bin centres (arcmin)."""
        return _theta_arcmin(self.binedges)

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(c.label for c in self.components)

    def __add__(self, other: "ThreePointResult") -> "ThreePointResult":
        if (
            other.n_max != self.n_max
            or other.components != self.components
            or not np.array_equal(other.binedges, self.binedges)
        ):
            raise ValueError("can only add multipoles of the same definition")
        return ThreePointResult(
            self.n_max,
            self.binedges,
            self.components,
            dict(self.spins),
            {k: self.num[k] + other.num[k] for k in self.num},
            {k: self.norm[k] + other.norm[k] for k in self.norm},
        )

    @staticmethod
    def _phi_coefficients(n: np.ndarray, phi_edges: np.ndarray) -> np.ndarray:
        """``c_n^k = (1/2pi) int_{bin k} exp(-i n phi) dphi``, shape (nphi, n)."""
        a = np.asarray(phi_edges[:-1], dtype=np.float64)[:, None]
        b = np.asarray(phi_edges[1:], dtype=np.float64)[:, None]
        nn = n[None, :].astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            c: np.ndarray = (np.exp(-1j * nn * a) - np.exp(-1j * nn * b)) / (
                2j * np.pi * nn
            )
        c[:, n == 0] = (b - a) / (2.0 * np.pi)
        return c

    def binned(
        self, phi_edges: Optional[np.ndarray] = None, nphi: int = 16
    ) -> Dict[str, np.ndarray]:
        """Estimate in opening-angle bins (x-projection).

        ``phi_edges`` default: ``nphi`` equal bins on ``[-pi, pi]``.  Returns
        ``label -> (nz1, nz2, nz3, nbins, nbins, nphi)``; bins without
        triangles are 0.
        """
        if phi_edges is None:
            phi_edges = np.linspace(-np.pi, np.pi, nphi + 1)
        c = self._phi_coefficients(self.n, np.asarray(phi_edges))
        out = {}
        for comp in self.components:
            # sum_n c_n Gamma_{-n}: reverse the n axis
            num = np.einsum("kn,n...->...k", c, self.num[comp.label][::-1])
            den = np.einsum("kn,n...->...k", c, self.norm[comp.label][::-1])
            with np.errstate(divide="ignore", invalid="ignore"):
                val = np.where(den != 0, num / den, 0.0)
            out[comp.label] = val
        return out

    def to_centroid(
        self,
        binned: Mapping[str, np.ndarray],
        phi_edges: Optional[np.ndarray] = None,
        nphi: int = 16,
    ) -> Dict[str, np.ndarray]:
        """Convert :meth:`binned` values from the x- to the centroid
        projection (every spin-2 value projected along the line from the
        triangle's centroid).  Flat-sky triangle at the bin centres."""
        if phi_edges is None:
            phi_edges = np.linspace(-np.pi, np.pi, nphi + 1)
        phi = 0.5 * (np.asarray(phi_edges[:-1]) + np.asarray(phi_edges[1:]))
        t = self.theta
        t2 = t[:, None, None]
        t3 = t[None, :, None]
        v1 = np.zeros((t.size, t.size, phi.size), dtype=np.complex128)
        v2 = t2 + 0j * t3 * phi
        v3 = t3 * np.exp(1j * phi)[None, None, :] + 0 * t2
        cen = (v1 + v2 + v3) / 3.0
        alpha = [np.angle(v - cen) for v in (v1, v2, v3)]
        beta = [
            np.zeros_like(alpha[0]),
            np.zeros_like(alpha[0]),
            np.broadcast_to(phi, alpha[0].shape),
        ]
        out = {}
        for comp in self.components:
            phase = np.zeros(alpha[0].shape)
            for v in range(3):
                s = self.spins[comp.fields[v]]
                sign = -1.0 if comp.conj[v] else 1.0
                phase = phase - sign * s * (alpha[v] - beta[v])
            out[comp.label] = binned[comp.label] * np.exp(1j * phase)
        return out
