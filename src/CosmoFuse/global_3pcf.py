"""
Footprint-wide three-point correlation functions by multipole decomposition.

Definition
----------
A triangle is seen from a *centre* vertex ``X1`` with two legs: ``X2`` at
separation ``theta_2`` (radial bin ``b2``) and position angle ``phi_2``,
``X3`` at ``theta_3`` (bin ``b3``) and ``phi_3``, all position angles measured
at ``X1`` from local east towards north.  ``phi = phi_3 - phi_2`` is the
opening angle.  For fields ``f1, f2, f3`` (tomographic bins of spin-0 or
spin-2 fields) and a choice of complex conjugations, the estimator's
multipoles are

    Gamma_n = sum_{X1} sum_{X2 in b2, X3 in b3, X2 != X3}
              w1 w2 w3  [f1~]^{c1} [f2~]^{c2} [f3~]^{c3}  exp(-i n phi),

    Norm_n  = the same with all fields replaced by 1 (weights only),

with the "x-projection" of every spin-2 value: ``f~ = -f exp(-2i beta)``, the
centre projected along its first leg (``beta = phi_2``) and each outer vertex
along its own leg to the centre (``beta = psi``, the direction of the centre
seen from that vertex; the minus sign makes ``Re f~`` the tangential shear).
Spin-0 values enter as they are.

For a finite set of triangles the multipoles are *exact* for every ``n``
(no truncation error in ``Gamma_n`` itself).  An estimate in opening-angle
bins is the ratio of the angle-binned sums,

    Gamma(b2, b3, phi-bin k) = sum_n c_n^k Gamma_{-n} / sum_n c_n^k Norm_{-n},
    c_n^k = (1/2pi) int_{bin k} exp(-i n phi) dphi,

which converges to the brute-force binned estimator as ``n_max`` grows
(:meth:`~CosmoFuse.results.ThreePointResult.binned`).  ``to_centroid`` converts to the
centroid projection (flat-sky triangle geometry at the bin centres).

Algorithm
---------
Step 1 (per centre, per radial bin; :mod:`CosmoFuse.threepoint_kernels`)
    ``G_m(b; X1) = sum_{X2 in b} w2 f2~ exp(-i m phi_2)`` for
    ``m = -(n_max+2) .. n_max+2`` and the pair products that remove the
    degenerate ``X2 = X3`` terms.
Step 2 (contraction over centres, batched matrix products)
    e.g. ``Gamma0_n = sum_{X1} w1 f1~ G_{2-n}(b2) G_n(b3)`` (the centre's
    projection phase ``exp(-2i phi_2)`` is absorbed into the first leg).

Both steps run on the backend device.  Pairs are *ordered* and owned by their
centre's tile (see :mod:`CosmoFuse.global_geometry`); the pairs are found and
uploaded once and reused for every map, like every other CosmoFuse estimator.

Scope
-----
Full resolution only: the static treecode suppresses multipole ``n`` by
roughly ``exp(-n^2 sigma_phi^2 / 2)`` with ``sigma_phi ~ 0.4 / k``, so its
validation for a 3PCF (which multipoles survive at which ``k``) has to come
first.  One device.
"""

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import h5py
import healpy as hp
import numpy as np

from .backend import get_backend
from .global_geometry import TileSet, default_tile_nside, ordered_pairs, radec_to_xyz
from .results import Component, ThreePointResult
from .utils import live_object_serial, pixel2RaDec

FORMAT_KIND = "CosmoFuse.Global3PCF"
FORMAT_VERSION = 1

_FLOATS = {"float32": np.float32, "float64": np.float64}


# ── the estimator ─────────────────────────────────────────────────────────


@dataclass
class _Field:
    name: str
    spin: int
    vre: Any
    vim: Any
    w: Any

    @property
    def nz(self) -> int:
        return int(self.vre.shape[0])


class Global3PCF:
    """Tomographic three-point correlation functions over a whole footprint.

    Args:
        nside: HEALPix resolution (power of two).
        mask: HEALPix map at ``nside`` (nonzero = observed); ``None`` = full sky.
        nbins, theta_min, theta_max: log-spaced radial bins of the legs
            (arcmin).
        n_max: largest opening-angle multipole.
        tile_nside: tiles of the pair search (power of two); default: the
            smallest tiles whose side is >= ``theta_max``.
        device: ``"cpu"``, ``"gpu"``, ``"auto"`` or a GPU id.
        map_precision: precision of the field and weight maps on the device.
        rotation_precision: precision of the stored pair angles.
        accumulation_precision: ``"float64"`` (default) or ``"same"``
            (= ``map_precision``): precision of the multipoles and contraction.
        pair_search_precision: precision of the pair search.
        centre_chunk: centres per step-1/step-2 batch (memory knob:
            ``~ centre_chunk * nbins * (n_fields * (2 n_max + 5) + 6 n_fields^2)``
            complex numbers on the device).
    """

    def __init__(
        self,
        nside: int,
        mask: Optional[np.ndarray] = None,
        nbins: int = 10,
        theta_min: float = 10,
        theta_max: float = 170,
        n_max: int = 20,
        tile_nside: Optional[int] = None,
        device: Union[str, int] = "auto",
        map_precision: Union[str, np.dtype, type] = "float64",
        rotation_precision: Union[str, np.dtype, type] = "float64",
        accumulation_precision: str = "float64",
        pair_search_precision: str = "float64",
        fastmath: bool = True,
        centre_chunk: int = 2048,
    ) -> None:
        if isinstance(device, (list, tuple, set)):
            raise NotImplementedError(
                "Global3PCF runs on one device; multi-device global estimators "
                "are not implemented yet."
            )
        nside = int(nside)
        if nbins <= 0:
            raise ValueError("nbins must be positive")
        if theta_min >= theta_max:
            raise ValueError("theta_min must be less than theta_max")
        if int(n_max) < 0:
            raise ValueError("n_max must be >= 0")
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.nbins = int(nbins)
        self.theta_min = np.radians(theta_min / 60.0)
        self.theta_max = np.radians(theta_max / 60.0)
        self.binedges = np.geomspace(self.theta_min, self.theta_max, self.nbins + 1)
        self.bincenters = np.sqrt(self.binedges[1:] * self.binedges[:-1]) * 60 * 180 / np.pi
        self.n_max = int(n_max)
        self.E = self.n_max + 2
        self.map_dtype = np.dtype(_FLOATS[np.dtype(map_precision).name])
        self.rotation_dtype = np.dtype(_FLOATS[np.dtype(rotation_precision).name])
        if accumulation_precision not in ("same", "float64"):
            raise ValueError("accumulation_precision must be 'same' or 'float64'")
        self.acc_dtype = (
            np.dtype(np.float64) if accumulation_precision == "float64" else self.map_dtype
        )
        self.acc_complex = np.dtype(np.complex128 if self.acc_dtype == np.float64 else np.complex64)
        if pair_search_precision not in ("float32", "float64"):
            raise ValueError("pair_search_precision must be 'float32' or 'float64'")
        self.pair_search_precision = pair_search_precision
        self.fastmath = bool(fastmath)
        self.centre_chunk = int(centre_chunk)
        if mask is not None:
            mask = np.asarray(mask)
            if mask.size != self.npix:
                raise ValueError("mask length must match nside")
            self.map_inds = np.flatnonzero(mask).astype(np.int64)
        else:
            self.map_inds = np.arange(self.npix, dtype=np.int64)
        if tile_nside is None:
            tile_nside = default_tile_nside(nside, self.theta_max)
        self.tiles = TileSet.from_rows(nside, self.map_inds, int(tile_nside))
        self.backend = get_backend(device)
        self.device = device

        # host pair geometry (CSR by (centre, bin), centres in tile order)
        self.centre_rows: Optional[np.ndarray] = None
        self.pair_offsets: Optional[np.ndarray] = None
        self.nbr_rows: Optional[np.ndarray] = None
        self.phi_c: Optional[np.ndarray] = None
        self.phi_n: Optional[np.ndarray] = None
        self._dev: Optional[Dict[str, Any]] = None
        self._norm_cache: Dict[Any, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # geometry
    # ------------------------------------------------------------------ #

    @property
    def row_pix(self) -> np.ndarray:
        return self.map_inds

    @property
    def n_active(self) -> int:
        return int(self.map_inds.size)

    @property
    def tile_nside(self) -> int:
        return int(self.tiles.tile_nside)

    @property
    def n_tiles(self) -> int:
        return self.tiles.n_tiles

    @property
    def ntotpairs(self) -> int:
        """Number of *ordered* pairs."""
        return 0 if self.nbr_rows is None else int(self.nbr_rows.size)

    @property
    def row_pix_hash(self) -> str:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.int64(self.nside).tobytes())
        digest.update(np.ascontiguousarray(self.map_inds, dtype=np.int64).tobytes())
        return digest.hexdigest()

    def to_row_space(self, maps: np.ndarray, dtype: Optional[Any] = None) -> np.ndarray:
        return np.ascontiguousarray(self._host_rows(np.asarray(maps)), dtype=dtype or self.map_dtype)

    def _host_rows(self, arr: np.ndarray) -> np.ndarray:
        n = int(arr.shape[-1])
        if n == self.n_active:
            return arr
        if n != self.npix:
            raise ValueError(
                f"map arrays must have npix={self.npix} or n_active={self.n_active} "
                f"pixels along the last axis; got {n}"
            )
        return np.take(arr, self.map_inds, axis=-1)

    def _is_device_array(self, a: Any) -> bool:
        return self.backend.name == "cupy" and type(a).__module__.split(".")[0] == "cupy"

    def _rows(self, arr: Any) -> Any:
        """Row-space backend array at map precision."""
        xp = self.backend.module
        if self._is_device_array(arr):
            n = int(arr.shape[-1])
            if n == self.npix:
                arr = xp.take(arr, self._device()["row_pix"], axis=-1)
            elif n != self.n_active:
                raise ValueError(f"map arrays must have npix or n_active pixels; got {n}")
            return arr.astype(self.map_dtype, copy=False)
        rows = np.ascontiguousarray(self._host_rows(np.asarray(arr)), dtype=self.map_dtype)
        return self.backend.to_device(rows)

    # ------------------------------------------------------------------ #
    # pairs
    # ------------------------------------------------------------------ #

    def find_pairs(self) -> None:
        """Find every ordered pair ``(centre -> neighbour)`` once."""
        ts = self.tiles
        dtype = np.dtype(self.pair_search_precision)
        pix = self.map_inds[ts.rows]
        ra, dec = pixel2RaDec(pix, self.nside)
        xyz = radec_to_xyz(ra, dec, dtype)
        rows_sorted = ts.rows
        nbr_parts: List[np.ndarray] = []
        pc_parts: List[np.ndarray] = []
        pn_parts: List[np.ndarray] = []
        cnt_parts: List[np.ndarray] = []
        for t in range(ts.n_tiles):
            a, b = int(ts.row_indptr[t]), int(ts.row_indptr[t + 1])
            reach = ts.reach(t, float(self.binedges[-1]))
            sel = np.concatenate(
                [np.arange(ts.row_indptr[u], ts.row_indptr[u + 1]) for u in reach]
            )
            nbr_pos, pc, pn, counts = ordered_pairs(
                pix[a:b], np.ascontiguousarray(xyz[:, a:b]),
                pix[sel], np.ascontiguousarray(xyz[:, sel]),
                self.binedges, fastmath=self.fastmath,
            )
            nbr_parts.append(rows_sorted[sel][nbr_pos])
            pc_parts.append(pc.astype(self.rotation_dtype))
            pn_parts.append(pn.astype(self.rotation_dtype))
            cnt_parts.append(counts.reshape(-1))
        counts_all = np.concatenate(cnt_parts) if cnt_parts else np.zeros(0, np.int64)
        self.centre_rows = rows_sorted.astype(np.int64)
        self.pair_offsets = np.concatenate(([0], np.cumsum(counts_all))).astype(np.int64)
        self.nbr_rows = (
            np.concatenate(nbr_parts).astype(np.int32) if nbr_parts else np.zeros(0, np.int32)
        )
        self.phi_c = np.concatenate(pc_parts) if pc_parts else np.zeros(0, self.rotation_dtype)
        self.phi_n = np.concatenate(pn_parts) if pn_parts else np.zeros(0, self.rotation_dtype)
        self._dev = None
        self._norm_cache.clear()

    def prepare(self) -> None:
        """Upload the pair geometry to the backend device (idempotent)."""
        if self._dev is not None:
            return
        if self.pair_offsets is None:
            raise RuntimeError("no pair geometry: call preprocess() or load_pairs() first")
        be = self.backend
        self._dev = {
            "offsets": be.to_device(self.pair_offsets),
            "nbr": be.to_device(self.nbr_rows),
            "phi_c": be.to_device(self.phi_c),
            "phi_n": be.to_device(self.phi_n),
            "centre_rows": be.to_device(self.centre_rows),
            "row_pix": be.to_device(self.map_inds),
        }

    def _device(self) -> Dict[str, Any]:
        self.prepare()
        assert self._dev is not None
        return self._dev

    def preprocess(self) -> None:
        """``find_pairs()`` + ``prepare()``."""
        self.find_pairs()
        self.prepare()

    def save_pairs(self, filepath: str) -> None:
        if self.pair_offsets is None:
            raise RuntimeError("no pair geometry to save")
        with h5py.File(filepath, "w") as fp:
            fp.attrs["kind"] = FORMAT_KIND
            fp.attrs["format_version"] = FORMAT_VERSION
            fp.attrs["nside"] = self.nside
            fp.attrs["nbins"] = self.nbins
            fp.attrs["theta_min"] = float(self.theta_min)
            fp.attrs["theta_max"] = float(self.theta_max)
            fp.attrs["tile_nside"] = self.tile_nside
            fp.attrs["pair_search_precision"] = self.pair_search_precision
            fp.create_dataset("map_inds", data=self.map_inds)
            fp.create_dataset("centre_rows", data=self.centre_rows)
            fp.create_dataset("pair_offsets", data=self.pair_offsets)
            fp.create_dataset("nbr_rows", data=self.nbr_rows)
            fp.create_dataset("phi_c", data=self.phi_c)
            fp.create_dataset("phi_n", data=self.phi_n)

    def load_pairs(self, filepath: str) -> None:
        """Read a file written by :meth:`save_pairs` (must match this
        estimator's nside, bins, mask and tiles) and prepare it."""
        with h5py.File(filepath, "r") as fp:
            if fp.attrs.get("kind", None) != FORMAT_KIND:
                raise ValueError(f"{filepath} is not a Global3PCF pair file")
            if int(fp.attrs["format_version"]) > FORMAT_VERSION:
                raise ValueError(f"{filepath}: newer Global3PCF format; update CosmoFuse")
            for name, got, want in (
                ("nside", int(fp.attrs["nside"]), self.nside),
                ("nbins", int(fp.attrs["nbins"]), self.nbins),
                ("tile_nside", int(fp.attrs["tile_nside"]), self.tile_nside),
            ):
                if got != want:
                    raise ValueError(f"{filepath}: {name}={got}, this estimator has {want}")
            if not np.allclose(
                [fp.attrs["theta_min"], fp.attrs["theta_max"]],
                [self.theta_min, self.theta_max], rtol=1e-12, atol=0.0,
            ):
                raise ValueError(f"{filepath}: angular range differs from this estimator")
            if not np.array_equal(fp["map_inds"][:], self.map_inds):
                raise ValueError(f"{filepath}: the mask (row space) differs from this estimator")
            self.centre_rows = fp["centre_rows"][:].astype(np.int64)
            self.pair_offsets = fp["pair_offsets"][:].astype(np.int64)
            self.nbr_rows = fp["nbr_rows"][:].astype(np.int32)
            self.phi_c = fp["phi_c"][:].astype(self.rotation_dtype)
            self.phi_n = fp["phi_n"][:].astype(self.rotation_dtype)
        self._dev = None
        self._norm_cache.clear()
        self.prepare()

    # ------------------------------------------------------------------ #
    # fields and descriptors
    # ------------------------------------------------------------------ #

    def _field(self, name: str, values: Any, weights: Any, spin: int,
               signs: Tuple[float, float] = (1.0, 1.0)) -> _Field:
        xp = self.backend.module
        v = self._rows(values)
        w = self._rows(weights)
        if spin == 2:
            if v.ndim != 3 or v.shape[1] != 2:
                raise ValueError(f"spin-2 field {name!r} must be (nz, 2, npix); got {v.shape}")
            vre, vim = v[:, 0], v[:, 1]
            if signs[0] != 1.0:
                vre = vre * self.map_dtype.type(signs[0])
            if signs[1] != 1.0:
                vim = vim * self.map_dtype.type(signs[1])
        elif spin == 0:
            if v.ndim != 2:
                raise ValueError(f"spin-0 field {name!r} must be (nz, npix); got {v.shape}")
            vre, vim = v, xp.zeros_like(v)
        else:
            raise ValueError(f"spin must be 0 or 2; got {spin}")
        if tuple(w.shape) != tuple(vre.shape):
            raise ValueError(f"weights of {name!r} must be (nz, npix); got {w.shape}")
        return _Field(name, int(spin), xp.ascontiguousarray(vre), xp.ascontiguousarray(vim),
                      xp.ascontiguousarray(w))

    def _descriptor(self, n_fields: int, sigmas: Sequence[int]) -> Tuple[Tuple[np.ndarray, ...], int]:
        """Output table: G_m of every field (m = -E..E), then the pair
        products ``[a_x a_y, conj(a_x) a_y] * exp(-i sigma phi)``."""
        E = self.E
        fa, fb, ca, cb, e = [], [], [], [], []
        for f in range(n_fields):
            for m in range(-E, E + 1):
                fa.append(f); fb.append(-1); ca.append(0); cb.append(0); e.append(m)
        n_g = len(fa)
        for s in sigmas:
            for typ in (0, 1):
                for x in range(n_fields):
                    for y in range(n_fields):
                        fa.append(x); fb.append(y); ca.append(typ); cb.append(0); e.append(s)
        desc = tuple(np.asarray(a, dtype=np.int32) for a in (fa, fb, ca, cb, e))
        return desc, n_g

    # ------------------------------------------------------------------ #
    # the two steps
    # ------------------------------------------------------------------ #

    def _accumulate(
        self,
        fields: Sequence[_Field],
        centre_values: Mapping[str, Tuple[Any, int]],
        jobs: Sequence[Tuple[str, str, str, str, bool, bool, int]],
        sigmas: Sequence[int],
    ) -> Dict[str, Any]:
        """Run step 1 + step 2 over all centre chunks.

        ``centre_values[key] = (values (nz, n_active) complex, field index
        offset)``; ``jobs``: ``(label, centre key, leg2 field, leg3 field, c2,
        c3, sigma1)``.  Returns ``label -> (N, nA, nB, nC, nb, nb)`` sums.
        """
        be = self.backend
        xp = be.module
        dev = self._device()
        kernel = getattr(be, "kernel_3pcf_multipoles", None)
        if kernel is None:
            raise RuntimeError("this backend has no three-point multipole kernel")
        offsets_f = {}
        off = 0
        for f in fields:
            offsets_f[f.name] = (off, off + f.nz)
            off += f.nz
        n_f = off
        vre = xp.ascontiguousarray(xp.concatenate([f.vre for f in fields]))
        vim = xp.ascontiguousarray(xp.concatenate([f.vim for f in fields]))
        wts = xp.ascontiguousarray(xp.concatenate([f.w for f in fields]))
        spin = be.to_device(np.concatenate([np.full(f.nz, f.spin, np.int32) for f in fields]))
        desc_h, n_g = self._descriptor(n_f, sigmas)
        desc = tuple(be.to_device(d) for d in desc_h)
        n_out = int(desc_h[0].size)
        E, P, nb = self.E, 2 * self.E + 1, self.nbins
        ns = np.arange(-self.n_max, self.n_max + 1)
        sig_index = {s: i for i, s in enumerate(sigmas)}
        cdt = getattr(xp, self.acc_complex.name)

        # Per component: R[n] = L3[n]^T @ A[n] with A[n] = L2[n] (x) centre,
        # i.e. one batched GEMM with every (centre bin, leg-2 bin) in its M
        # dimension (4 GEMMs of a quarter the size were 1.8x slower on an
        # A100), accumulated over chunks as (N, b3*nC, nA*b2*nB).  The
        # degenerate X2 = X3 terms do not depend on n; their sums over all
        # centres are subtracted once at the end.
        acc: Dict[str, Any] = {}
        deg: Dict[str, Any] = {}
        n_centres = int(self.centre_rows.size)
        idx_cache: Dict[Tuple[int, ...], Any] = {}

        def _idx(a: np.ndarray) -> Any:
            key = tuple(int(v) for v in a)
            if key not in idx_cache:
                idx_cache[key] = be.to_device(np.asarray(a, dtype=np.int64))
            return idx_cache[key]

        for c0 in range(0, n_centres, self.centre_chunk):
            C = min(self.centre_chunk, n_centres - c0)
            out = xp.zeros((C, nb, n_out), dtype=cdt)
            kernel(dev["offsets"], c0, C, nb, dev["nbr"], dev["phi_c"], dev["phi_n"],
                   vre, vim, wts, spin, desc, E, out)
            G = xp.ascontiguousarray(
                out[:, :, :n_g].reshape(C, nb, n_f, P).transpose(3, 0, 1, 2)
            )  # (P, C, nb, F)
            S = out[:, :, n_g:].reshape(C, nb, len(sigmas), 2, n_f, n_f)
            rows = dev["centre_rows"][c0 : c0 + C]
            legs: Dict[Tuple[Any, ...], Any] = {}

            def _leg(m: np.ndarray, conj: bool, lo: int, hi: int) -> Any:
                key = (tuple(int(v) for v in m), conj, lo, hi)
                if key not in legs:
                    L = G[_idx(m + E)][:, :, :, lo:hi]
                    if conj:
                        L = xp.conj(L)
                    legs[key] = xp.ascontiguousarray(L).reshape(ns.size, C, nb * (hi - lo))
                return legs[key]

            for label, ckey, leg2, leg3, c2, c3, sigma1 in jobs:
                vals = centre_values[ckey][0][:, rows].T.astype(cdt)  # (C, nA)
                b0, b1 = offsets_f[leg2]
                d0, d1 = offsets_f[leg3]
                L2 = _leg(ns - sigma1, True, b0, b1) if c2 else _leg(sigma1 - ns, False, b0, b1)
                L3 = _leg(-ns, True, d0, d1) if c3 else _leg(ns, False, d0, d1)
                nA = int(vals.shape[1])
                A = (L2[:, :, None, :] * vals[None, :, :, None]).reshape(ns.size, C, -1)
                R = xp.matmul(L3.transpose(0, 2, 1), A)  # (N, nb*nC, nA*nb*nB)
                if not c2 and not c3:
                    Sv = S[:, :, sig_index[sigma1], 0]
                elif c2 and not c3:
                    Sv = S[:, :, sig_index[sigma1], 1]
                elif not c2 and c3:
                    Sv = xp.conj(S[:, :, sig_index[-sigma1], 1])
                else:
                    Sv = xp.conj(S[:, :, sig_index[-sigma1], 0])
                corr = xp.tensordot(vals, Sv[:, :, b0:b1, d0:d1], axes=([0], [0]))
                if label in acc:
                    acc[label] += R
                    deg[label] += corr
                else:
                    acc[label] = R
                    deg[label] = corr

        results = {}
        for label, ckey, leg2, leg3, _c2, _c3, _s in jobs:
            nA = int(centre_values[ckey][0].shape[0])
            nB = offsets_f[leg2][1] - offsets_f[leg2][0]
            nC = offsets_f[leg3][1] - offsets_f[leg3][0]
            R6 = acc[label].reshape(ns.size, nb, nC, nA, nb, nB)  # n, b3, z3, a, b2, z2
            corr = deg[label]  # (nA, nb, nB, nC): a, b, z2, z3
            for b in range(nb):
                R6[:, b, :, :, b, :] -= corr[:, b].transpose(2, 0, 1)[None]
            out6 = xp.ascontiguousarray(R6.transpose(0, 3, 5, 2, 4, 1))  # n, a, z2, z3, b2, b3
            results[label] = np.asarray(be.to_numpy(out6)).astype(np.complex128)
        return results

    def _weights_key(self, fields: Sequence[_Field]) -> Tuple[Any, ...]:
        key = []
        for f in fields:
            if self._is_device_array(f.w):
                key.append((f.name, "dev", live_object_serial(f.w), tuple(f.w.shape)))
            else:
                w = np.ascontiguousarray(f.w)
                key.append((f.name, hashlib.blake2b(memoryview(w).cast("B")).hexdigest(), w.shape))
        return tuple(key)

    def _norms(
        self, fields: Sequence[_Field], components: Sequence[Component]
    ) -> Dict[str, np.ndarray]:
        """Weight-only multipoles of every component (cached per weights)."""
        wkey = self._weights_key(fields)
        need = sorted({c.fields for c in components})
        cache = self._norm_cache.get(wkey)
        if cache is None:
            if len(self._norm_cache) >= 4:
                self._norm_cache.pop(next(iter(self._norm_cache)))
            cache = self._norm_cache[wkey] = {}
        missing = [g for g in need if g not in cache]
        if missing:
            xp = self.backend.module
            wfields = [
                _Field(f.name, 0, xp.ones_like(f.w), xp.zeros_like(f.w), f.w) for f in fields
            ]
            cvals = {f.name: (f.w, 0) for f in fields}
            jobs = [("|".join(g), g[0], g[1], g[2], False, False, 0) for g in missing]
            res = self._accumulate(wfields, cvals, jobs, [0])
            for g in missing:
                cache[g] = res["|".join(g)]
        return {c.label: cache[c.fields] for c in components}

    def measure(
        self,
        fields: Mapping[str, Tuple[Any, Any, int]],
        components: Sequence[Component],
        flip_g1: bool = False,
        flip_g2: bool = False,
    ) -> ThreePointResult:
        """General entry point.

        ``fields``: ``name -> (values, weights, spin)`` with values ``(nz,
        npix)`` (spin 0) or ``(nz, 2, npix)`` (spin 2; ``flip_g1/2`` apply to
        them) and weights ``(nz, npix)``; full-sky or row-space.
        ``components``: which field sits at which vertex (see
        :class:`Component`).
        """
        signs = (-1.0 if flip_g1 else 1.0, -1.0 if flip_g2 else 1.0)
        flist = [
            self._field(name, v, w, int(s), signs=signs if int(s) == 2 else (1.0, 1.0))
            for name, (v, w, s) in fields.items()
        ]
        spins = {f.name: f.spin for f in flist}
        by_name = {f.name: f for f in flist}
        for comp in components:
            for name in comp.fields:
                if name not in by_name:
                    raise ValueError(f"component {comp.label!r} uses unknown field {name!r}")
        centre_values: Dict[str, Tuple[Any, int]] = {}
        jobs = []
        sigmas = {0}
        for comp in components:
            fc = by_name[comp.fields[0]]
            c1, c2, c3 = comp.conj
            if fc.spin == 2:
                val = -(fc.vre + 1j * fc.vim) * fc.w
                if c1:
                    val = self.backend.module.conj(val)
                sigma1 = -2 if c1 else 2
                sigmas |= {-2, 2}
            else:
                val = fc.vre * fc.w
                sigma1 = 0
            ckey = f"{fc.name}|{int(c1)}"
            centre_values[ckey] = (val, 0)
            jobs.append((comp.label, ckey, comp.fields[1], comp.fields[2], bool(c2), bool(c3), sigma1))
        num = self._accumulate(flist, centre_values, jobs, sorted(sigmas))
        norm = self._norms(flist, components)
        return ThreePointResult(
            self.n_max, self.binedges.copy(), tuple(components), spins, num, norm
        )

    def ggg(
        self, shear_maps: Any, w: Any, flip_g1: bool = False, flip_g2: bool = False
    ) -> ThreePointResult:
        """Shear three-point function: the natural components ``gamma0`` =
        <g g g>, ``gamma1`` = <g* g g>, ``gamma2`` = <g g* g>, ``gamma3`` =
        <g g g*> (x-projection), for all ordered tomographic triplets."""
        comps = (
            Component("gamma0", ("g", "g", "g"), (False, False, False)),
            Component("gamma1", ("g", "g", "g"), (True, False, False)),
            Component("gamma2", ("g", "g", "g"), (False, True, False)),
            Component("gamma3", ("g", "g", "g"), (False, False, True)),
        )
        return self.measure({"g": (shear_maps, w, 2)}, comps, flip_g1=flip_g1, flip_g2=flip_g2)

    def kkk(self, k_maps: Any, w: Any) -> ThreePointResult:
        """Scalar three-point function ``zeta`` of density (as a field, δ) or
        convergence maps, for all ordered tomographic triplets."""
        return self.measure({"k": (k_maps, w, 0)}, (Component("zeta", ("k", "k", "k")),))

    def kgg(
        self, k_maps: Any, shear_maps: Any, k_weights: Any, shear_weights: Any,
        flip_g1: bool = False, flip_g2: bool = False,
    ) -> ThreePointResult:
        """Scalar at the centre, shear at both legs: ``kgg`` = <k g g> and
        ``kg*g`` = <k g* g> (the two independent shear-pair products), for all
        ordered (k bin, g bin, g bin) triplets."""
        comps = (
            Component("kgg", ("k", "g", "g")),
            Component("kg*g", ("k", "g", "g"), (False, True, False)),
        )
        return self.measure(
            {"k": (k_maps, k_weights, 0), "g": (shear_maps, shear_weights, 2)},
            comps, flip_g1=flip_g1, flip_g2=flip_g2,
        )

    def kkg(
        self, k_maps: Any, shear_maps: Any, k_weights: Any, shear_weights: Any,
        flip_g1: bool = False, flip_g2: bool = False,
    ) -> ThreePointResult:
        """Scalars at the centre and leg 2, shear at leg 3: ``kkg`` = <k k g>,
        for all ordered (k bin, k bin, g bin) triplets."""
        return self.measure(
            {"k": (k_maps, k_weights, 0), "g": (shear_maps, shear_weights, 2)},
            (Component("kkg", ("k", "k", "g")),), flip_g1=flip_g1, flip_g2=flip_g2,
        )
