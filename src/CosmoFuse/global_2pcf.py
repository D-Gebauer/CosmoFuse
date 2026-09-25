"""
Footprint-wide two-point correlation functions: ``gg`` (ξ±), ``kk`` (w(θ) of
overdensity maps) and ``kg`` (γ_t), in TreeCorr's field-code vocabulary.

:class:`Global2PCF` measures the 2PCFs of the whole footprint -- every pixel
pair once -- with the same precompute-once / measure-many workflow as the
patch estimator:

1. ``preprocess()`` finds the pairs once (per tile, see
   :mod:`CosmoFuse.global_geometry`) and prepares them on the device;
2. every measurement call runs the **same pair kernels** as
   :class:`~CosmoFuse.correlations.Integrated3PCF` (CPU Numba and GPU
   tiled / packed CUDA kernels, static treecode, row space, sign flips),
   with tiles in the role of patches;
3. the result (:class:`~CosmoFuse.results.TwoPointResult`) holds the
   per-tile numerators and weight sums; the estimate is their ratio over the
   whole footprint (ratio of sums, TreeCorr's definition), and
   ``jackknife()`` gives the delete-one-tile estimates.

Internally the tile pairs are handed to an :class:`Integrated3PCF` whose
groups are tiles (:class:`_TileEngine`); it only overrides the normalisation
seam ``_finish_pair_sums`` to keep the sums.  Everything that makes the patch
path fast is therefore shared, not duplicated.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import healpy as hp
import numpy as np

from .correlations import Integrated3PCF
from .global_geometry import (
    LevelElements,
    TileSet,
    default_tile_nside,
    radec_to_xyz,
    unique_pairs,
)
from .results import TwoPointResult
from .treecode import TreecodeGeometry, build_patch_cells, level_groups
from .utils import pixel2RaDec

FORMAT_KIND = "CosmoFuse.Global2PCF"
FORMAT_VERSION = 1


class _TileEngine(Integrated3PCF):
    """A :class:`Integrated3PCF` whose groups are tiles.

    Pairs are injected by :class:`Global2PCF`; the normalisation seam keeps
    the per-tile sums (``captured``) instead of dividing per group.
    """

    captured: Optional[Tuple[Tuple[Any, ...], Any, int]] = None

    def _finish_pair_sums(
        self, nums: Sequence[Any], den: Any, nzbin_combs: int
    ) -> Tuple[Any, ...]:
        self.captured = (tuple(nums), den, int(nzbin_combs))
        return tuple(None for _ in nums)

    def find_pairs(self) -> None:  # pragma: no cover - guarded by Global2PCF
        raise RuntimeError("tile pairs are found by Global2PCF.find_pairs()")

    def take_captured(self) -> Tuple[Tuple[Any, ...], Any, int]:
        captured = self.captured
        self.captured = None
        if captured is None:
            raise RuntimeError("the pair kernel produced no sums")
        return captured


class Global2PCF:
    """Tomographic ξ±, w(θ) and γ_t over a whole footprint.

    Args:
        nside: HEALPix resolution of the maps (power of two).
        mask: HEALPix map at ``nside`` (nonzero = observed); ``None`` for the
            full sky.  Defines the row space, as for the patch estimator.
        nbins, theta_min, theta_max: log-spaced angular bins (arcmin).
        tile_nside: HEALPix resolution of the tiles (power of two, at most
            the coarsest treecode level).  Default: the smallest tiles whose
            side is still >= ``theta_max``.  With ``pack_pairs=True`` every
            (tile, level) row block, halo included, must hold <= 65536 rows;
            pass a larger ``tile_nside`` if it does not.
        device, map_precision, rotation_precision, accumulation_precision,
        resolution_factor, pair_search_precision, pack_pairs, fastmath:
            as for :class:`~CosmoFuse.correlations.Integrated3PCF`.  The
            static treecode builds its cells over the whole footprint (there
            is no patch window to preserve), from the binary mask.
    """

    def __init__(
        self,
        nside: int,
        mask: Optional[np.ndarray] = None,
        nbins: int = 10,
        theta_min: float = 10,
        theta_max: float = 170,
        tile_nside: Optional[int] = None,
        device: Union[str, int] = "auto",
        map_precision: Union[str, np.dtype, type] = "float64",
        rotation_precision: Union[str, np.dtype, type] = "float32",
        accumulation_precision: str = "same",
        resolution_factor: Union[None, bool, str, float] = None,
        pair_search_precision: str = "float64",
        pack_pairs: bool = False,
        fastmath: bool = True,
    ) -> None:
        if isinstance(device, (list, tuple, set)):
            raise NotImplementedError(
                "Global2PCF runs on one device; multi-device global estimators "
                "are not implemented yet."
            )
        if theta_min >= theta_max:
            raise ValueError("theta_min must be less than theta_max")
        theta_max_rad = np.radians(theta_max / 60.0)
        if tile_nside is None:
            tile_nside = default_tile_nside(int(nside), theta_max_rad)
            auto_tiles = True
        else:
            auto_tiles = False
        # The engine is a Integrated3PCF over *tiles*; its patch centres are
        # the tile centres and its patch_size/theta_Q are never used (no
        # patch search, no aperture).
        self._engine = _TileEngine(
            int(nside),
            np.zeros(1),
            np.zeros(1),
            nbins=nbins,
            theta_min=theta_min,
            theta_max=theta_max,
            patch_size=1.0,
            theta_Q=1.0,
            mask=mask,
            fastmath=fastmath,
            device=device,
            map_precision=map_precision,
            rotation_precision=rotation_precision,
            accumulation_precision=accumulation_precision,
            resolution_factor=resolution_factor,
            pair_search_precision=pair_search_precision,
            pack_pairs=pack_pairs,
        )
        e = self._engine
        coarse = self._coarse_nsides()
        if coarse and tile_nside > min(coarse):
            if not auto_tiles:
                raise ValueError(
                    f"tile_nside={tile_nside} is finer than the coarsest treecode "
                    f"level (nside {min(coarse)}); cells would straddle tiles. "
                    f"Use tile_nside <= {min(coarse)}."
                )
            tile_nside = int(min(coarse))
        self.tiles = TileSet.from_rows(e.nside, e.map_inds, int(tile_nside))
        phi, theta = self.tiles.centre_ang
        e.phi_center = phi
        e.theta_center = theta
        e.n_patches = self.tiles.n_tiles
        e.patch_size = float(np.degrees(hp.max_pixrad(self.tiles.tile_nside)) * 60.0)

    # ------------------------------------------------------------------ #
    # geometry
    # ------------------------------------------------------------------ #

    @property
    def nside(self) -> int:
        return int(self._engine.nside)

    @property
    def nbins(self) -> int:
        return int(self._engine.nbins)

    @property
    def binedges(self) -> np.ndarray:
        """Angular bin edges in radians."""
        return self._engine.binedges

    @property
    def bincenters(self) -> np.ndarray:
        """Geometric bin centres in arcmin."""
        return self._engine.bincenters

    @property
    def tile_nside(self) -> int:
        return int(self.tiles.tile_nside)

    @property
    def n_tiles(self) -> int:
        return self.tiles.n_tiles

    @property
    def row_pix(self) -> np.ndarray:
        """HEALPix (RING) pixel of every row of the compact row space."""
        return self._engine.row_pix

    @property
    def n_active(self) -> int:
        return self._engine.n_active

    @property
    def row_pix_hash(self) -> str:
        return self._engine.row_pix_hash

    @property
    def level_table(self) -> Dict[str, Any]:
        """Resolution of every angular bin (store it with the data vector)."""
        return self._engine.level_table

    @property
    def ntotpairs(self) -> int:
        return int(self._engine.ntotpairs)

    @property
    def backend(self) -> Any:
        return self._engine.backend

    @property
    def map_dtype(self) -> np.dtype:
        return self._engine.map_dtype

    def to_row_space(self, maps: np.ndarray, dtype: Optional[Any] = None) -> np.ndarray:
        """Cut full-sky map(s) ``(..., npix)`` to the row space."""
        return self._engine.to_row_space(maps, dtype=dtype)

    def _coarse_nsides(self) -> Tuple[int, ...]:
        e = self._engine
        levels = {int(ns) for ns in e.level_nside if int(ns) != int(e.nside)}
        return tuple(sorted(levels, reverse=True))

    def _build_treecode(self) -> Optional[TreecodeGeometry]:
        coarse = self._coarse_nsides()
        if not coarse:
            return None
        e, ts = self._engine, self.tiles
        cells = []
        for t in range(ts.n_tiles):
            rows = ts.rows[ts.row_indptr[t] : ts.row_indptr[t + 1]]
            cells.append(build_patch_cells(e.nside, e.map_inds[rows], coarse))
        return TreecodeGeometry.from_patches(e.nside, coarse, cells)

    def _level_elements(
        self, treecode: Optional[TreecodeGeometry]
    ) -> Dict[int, LevelElements]:
        """Pixels (base level) and cells (coarse levels), sorted by tile."""
        e, ts = self._engine, self.tiles
        dtype = np.dtype(e.pair_search_precision)
        pix = np.asarray(e.map_inds, dtype=np.int64)[ts.rows]
        ra, dec = pixel2RaDec(pix, e.nside)
        out = {
            int(e.nside): LevelElements(
                ids=pix, xyz=radec_to_xyz(ra, dec, dtype), indptr=ts.row_indptr
            )
        }
        if treecode is not None:
            npix = hp.nside2npix(e.nside)
            starts = treecode.level_starts(first=e.n_aperture_cells)
            for level, ns in enumerate(treecode.coarse_nsides):
                n = int(treecode.cells_per_level[level])
                out[int(ns)] = LevelElements(
                    ids=npix + starts[level] + np.arange(n, dtype=np.int64),
                    xyz=radec_to_xyz(
                        treecode.cell_ra[level], treecode.cell_dec[level], dtype
                    ),
                    indptr=treecode.cell_offsets[level].astype(np.int64),
                )
        return out

    # ------------------------------------------------------------------ #
    # pairs
    # ------------------------------------------------------------------ #

    def find_pairs(self) -> None:
        """Find every footprint pair once, grouped by tile (host arrays)."""
        e, ts = self._engine, self.tiles
        treecode = self._build_treecode()
        elements = self._level_elements(treecode)
        groups = level_groups(e.level_nside)
        pair_inds: List[np.ndarray] = []
        pair_exp2phi: List[np.ndarray] = []
        bins: List[np.ndarray] = []
        reach_cache: Dict[Tuple[int, float], np.ndarray] = {}
        for t in range(ts.n_tiles):
            inds_t, exp_t = [], []
            bins_t = np.zeros(e.nbins, dtype=np.int64)
            for nside_b, b0, b1 in groups:
                level = elements[int(nside_b)]
                radius = float(e.binedges[b1])
                key = (t, radius)
                if key not in reach_cache:
                    reach_cache[key] = ts.reach(t, radius)
                core_ids, core_xyz = level.core(t)
                cand_ids, cand_xyz = level.of_tiles(reach_cache[key])
                ids_a, ids_b, c1, s1, c2, s2, counts = unique_pairs(
                    core_ids, core_xyz, cand_ids, cand_xyz,
                    e.binedges[b0 : b1 + 1], fastmath=e.fastmath,
                )
                inds_t.append(np.stack((ids_a, ids_b)))
                exp = np.empty((2, ids_a.size), dtype=e.rotation_complex_dtype)
                exp[0].real = c1
                exp[0].imag = s1
                exp[1].real = c2
                exp[1].imag = s2
                exp_t.append(exp)
                bins_t[b0:b1] = counts
            pair_inds.append(np.concatenate(inds_t, axis=1).astype(e.index_dtype, copy=False))
            pair_exp2phi.append(np.concatenate(exp_t, axis=1))
            bins.append(bins_t.astype(e.index_dtype))
        self._install(pair_inds, pair_exp2phi, bins, treecode)

    def _install(
        self,
        pair_inds: List[np.ndarray],
        pair_exp2phi: List[np.ndarray],
        bins: List[np.ndarray],
        treecode: Optional[TreecodeGeometry],
    ) -> None:
        e = self._engine
        if treecode is not None:
            npix = hp.nside2npix(e.nside)
            if npix + treecode.n_cells > np.iinfo(e.index_dtype).max:
                raise OverflowError("pixel + coarse-cell ids exceed the index dtype range")
        e.pair_inds = pair_inds
        e.pair_exp2phi = pair_exp2phi
        e.bins = bins
        e._treecode = treecode
        e.packed_pairs = None
        e.packed_block_ids = None
        e.packed_block_sizes = None
        e._invalidate_prepared_state()

    def prepare(self, release_host_pairs: bool = False) -> None:
        """Move the pairs to the backend device (idempotent)."""
        self._engine.prepare(release_host_pairs=release_host_pairs)

    def preprocess(self, release_host_pairs: bool = False) -> None:
        """``find_pairs()`` + ``prepare()``."""
        self.find_pairs()
        self.prepare(release_host_pairs=release_host_pairs)

    def save_pairs(self, filepath: str) -> None:
        """Write the pair geometry (exact, host arrays) to HDF5."""
        e = self._engine
        if e.bins is None or e.pair_inds is None:
            raise RuntimeError(
                "host pairs were released (release_host_pairs=True); recompute "
                "them before save_pairs()."
            )
        counts = np.array([p.shape[1] for p in e.pair_inds], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(counts)))
        with h5py.File(filepath, "w") as fp:
            fp.attrs["kind"] = FORMAT_KIND
            fp.attrs["format_version"] = FORMAT_VERSION
            fp.attrs["nside"] = int(e.nside)
            fp.attrs["nbins"] = int(e.nbins)
            fp.attrs["theta_min"] = float(e.theta_min)
            fp.attrs["theta_max"] = float(e.theta_max)
            fp.attrs["tile_nside"] = int(self.tile_nside)
            fp.attrs["pair_search_precision"] = str(e.pair_search_precision)
            if e.resolution_factor is not None:
                fp.attrs["resolution_factor"] = float(e.resolution_factor)
            fp.create_dataset("map_inds", data=np.asarray(e.map_inds, dtype=np.int64))
            fp.create_dataset("tile_ids", data=self.tiles.tile_ids)
            fp.create_dataset("level_nside", data=np.asarray(e.level_nside, dtype=np.int64))
            fp.create_dataset("pair_offsets", data=offsets)
            fp.create_dataset("bins", data=np.stack([np.asarray(b, dtype=np.int64) for b in e.bins]))
            fp.create_dataset(
                "pair_inds",
                data=np.concatenate(e.pair_inds, axis=1) if e.pair_inds else np.zeros((2, 0)),
            )
            fp.create_dataset(
                "pair_exp2phi",
                data=np.concatenate(e.pair_exp2phi, axis=1) if e.pair_exp2phi else np.zeros((2, 0)),
            )
            tc = e._treecode
            if tc is not None:
                gp = fp.create_group("treecode")
                gp.attrs["base_nside"] = int(tc.base_nside)
                gp.create_dataset("coarse_nsides", data=np.asarray(tc.coarse_nsides, dtype=np.int64))
                gp.create_dataset("cell_offsets", data=tc.cell_offsets)
                for lv in range(tc.n_levels):
                    gp.create_dataset(f"child_indptr_{lv}", data=tc.child_indptr[lv])
                    gp.create_dataset(f"child_indices_{lv}", data=tc.child_indices[lv])
                    gp.create_dataset(f"cell_ra_{lv}", data=tc.cell_ra[lv])
                    gp.create_dataset(f"cell_dec_{lv}", data=tc.cell_dec[lv])

    def load_pairs(self, filepath: str, release_host_pairs: bool = False) -> None:
        """Read a pair file written by :meth:`save_pairs` and prepare it.

        The file must describe *this* estimator (nside, bins, mask, tiles,
        resolution levels); a mismatch raises instead of silently measuring
        a different estimator.
        """
        e = self._engine
        with h5py.File(filepath, "r") as fp:
            if fp.attrs.get("kind", None) != FORMAT_KIND:
                raise ValueError(f"{filepath} is not a Global2PCF pair file")
            version = int(fp.attrs["format_version"])
            if version > FORMAT_VERSION:
                raise ValueError(
                    f"{filepath} has Global2PCF format version {version}; this "
                    f"CosmoFuse reads <= {FORMAT_VERSION}"
                )
            checks = [
                ("nside", int(fp.attrs["nside"]), int(e.nside)),
                ("nbins", int(fp.attrs["nbins"]), int(e.nbins)),
                ("tile_nside", int(fp.attrs["tile_nside"]), self.tile_nside),
            ]
            for name, got, want in checks:
                if got != want:
                    raise ValueError(f"{filepath}: {name}={got}, this estimator has {want}")
            if not np.allclose(
                [fp.attrs["theta_min"], fp.attrs["theta_max"]],
                [e.theta_min, e.theta_max],
                rtol=1e-12,
                atol=0.0,
            ):
                raise ValueError(f"{filepath}: angular range differs from this estimator")
            if not np.array_equal(fp["map_inds"][:], np.asarray(e.map_inds, dtype=np.int64)):
                raise ValueError(f"{filepath}: the mask (row space) differs from this estimator")
            if not np.array_equal(fp["level_nside"][:], np.asarray(e.level_nside, dtype=np.int64)):
                raise ValueError(
                    f"{filepath}: resolution levels {fp['level_nside'][:].tolist()} differ "
                    f"from this estimator's {np.asarray(e.level_nside).tolist()} "
                    "(resolution_factor)"
                )
            offsets = fp["pair_offsets"][:]
            bins_arr = fp["bins"][:]
            inds_flat = fp["pair_inds"][:].astype(e.index_dtype, copy=False)
            exp_flat = fp["pair_exp2phi"][:].astype(e.rotation_complex_dtype, copy=False)
            treecode = None
            if "treecode" in fp:
                gp = fp["treecode"]
                n_levels = int(gp["coarse_nsides"].shape[0])
                treecode = TreecodeGeometry(
                    base_nside=int(gp.attrs["base_nside"]),
                    coarse_nsides=tuple(int(n) for n in gp["coarse_nsides"][:]),
                    cell_offsets=gp["cell_offsets"][:].astype(np.int64),
                    child_indptr=[gp[f"child_indptr_{lv}"][:] for lv in range(n_levels)],
                    child_indices=[gp[f"child_indices_{lv}"][:] for lv in range(n_levels)],
                    cell_ra=[gp[f"cell_ra_{lv}"][:] for lv in range(n_levels)],
                    cell_dec=[gp[f"cell_dec_{lv}"][:] for lv in range(n_levels)],
                )
        n = self.n_tiles
        pair_inds = [inds_flat[:, offsets[t] : offsets[t + 1]] for t in range(n)]
        pair_exp2phi = [exp_flat[:, offsets[t] : offsets[t + 1]] for t in range(n)]
        bins = [bins_arr[t].astype(e.index_dtype) for t in range(n)]
        self._install(pair_inds, pair_exp2phi, bins, treecode)
        self.prepare(release_host_pairs=release_host_pairs)

    # ------------------------------------------------------------------ #
    # measurements
    # ------------------------------------------------------------------ #

    def _result(
        self,
        statistic: str,
        names: Tuple[str, ...],
        combinations: Sequence[Tuple[int, int]],
        return_device: bool,
    ) -> TwoPointResult:
        e = self._engine
        nums, den, ncomb = e.take_captured()
        xp = e.backend.module
        shape = (ncomb, self.n_tiles, self.nbins)
        acc = getattr(xp, e.acc_dtype.name)
        tile_num = {n: xp.real(a).reshape(shape).astype(acc, copy=False) for n, a in zip(names, nums)}
        tile_weight = xp.real(den).reshape(shape).astype(acc, copy=False)
        res = TwoPointResult(
            statistic, np.asarray(self.binedges).copy(),
            tuple((int(a), int(b)) for a, b in combinations), tile_num, tile_weight,
        )
        if return_device and e.backend.name == "cupy":
            return res
        return TwoPointResult(
            res.statistic, res.binedges, res.combinations,
            {k: np.asarray(e.backend.to_numpy(v)) for k, v in tile_num.items()},
            np.asarray(e.backend.to_numpy(tile_weight)),
        )

    def gg(
        self,
        shear_maps: Any,
        w: Any,
        flip_g1: bool = False,
        flip_g2: bool = False,
        return_device: bool = True,
    ) -> TwoPointResult:
        """Tomographic ξ± of the footprint (TreeCorr ``GGCorrelation``).

        ``shear_maps``: ``(nz, 2, npix | n_active)``; ``w``: ``(nz, npix |
        n_active)``.  Returns a ``"gg"`` :class:`TwoPointResult` with
        ``.xi_p``, ``.xi_m``, rows = upper-triangle ``combinations``.
        """
        e = self._engine
        e._ensure_prepared()
        shear_arr = e._coerce_map_input_array(shear_maps)
        w_arr = e._coerce_map_input_array(w)
        if shear_arr.ndim != 3 or shear_arr.shape[1] != 2:
            raise ValueError(f"shear_maps must be (nz, 2, npix); got {shear_arr.shape}")
        if tuple(w_arr.shape) != (shear_arr.shape[0], shear_arr.shape[2]):
            raise ValueError(f"w must be (nz, npix); got {w_arr.shape}")
        nz = int(shear_arr.shape[0])
        ncomb = nz * (nz + 1) // 2
        shear_dev = e._to_backend_array(shear_arr, dtype=e.map_dtype)
        w_dev = e._to_backend_array(w_arr, dtype=e.map_dtype)
        res = e._xipm_tomo_vectorized(
            shear_dev, w_dev, None, nz, ncomb,
            -1 if flip_g1 else 1, -1 if flip_g2 else 1,
        )
        if res is None:
            raise RuntimeError("the tomographic ξ± kernel is unavailable on this backend")
        return self._result("gg", ("xi_p", "xi_m"), Integrated3PCF.tomo_combinations(nz), return_device)

    def kk(
        self,
        k_maps: Any,
        w: Any,
        auto_only: bool = False,
        return_device: bool = True,
    ) -> TwoPointResult:
        """Tomographic correlation of scalar fields -- w(θ) for overdensity
        maps: ``Σ w_i w_j δ_i δ_j / Σ w_i w_j`` (TreeCorr ``KKCorrelation``
        with ``k = δ``; equal to Landy-Szalay with randoms at the pixel
        centres when ``w`` is the pixel coverage).

        Returns a ``"kk"`` :class:`TwoPointResult` with ``.xi``; rows = upper
        triangle, or the ``nz`` auto-correlations with ``auto_only=True``.
        """
        e = self._engine
        e._ensure_prepared()
        arr = e._coerce_map_input_array(k_maps)
        if arr.ndim != 2:
            raise ValueError(f"k_maps must be (nz, npix); got {arr.shape}")
        nz = int(arr.shape[0])
        e._density_density_tomo_vectorized(arr, w, None, nz, gc_auto_correlations_only=auto_only)
        return self._result(
            "kk", ("xi",), Integrated3PCF.tomo_combinations(nz, auto_only), return_device
        )

    def kg(
        self,
        k_maps: Any,
        shear_maps: Any,
        k_weights: Any,
        shear_weights: Any,
        combinations: Optional[Sequence[Tuple[int, int]]] = None,
        flip_g1: bool = False,
        flip_g2: bool = False,
        return_device: bool = True,
    ) -> TwoPointResult:
        """Tomographic γ_t of scalar (lens δ) × shear (source) maps:
        ``Σ w_l w_s δ_l γ_t / Σ w_l w_s`` (TreeCorr ``KGCorrelation``).

        Rows: all ``(lens, source)`` combinations, lens-major, or the given
        ``combinations`` in order.  Returns a ``"kg"`` :class:`TwoPointResult`
        with ``.xi``.
        """
        e = self._engine
        e._ensure_prepared()
        d_arr = e._coerce_map_input_array(k_maps)
        s_arr = e._coerce_map_input_array(shear_maps)
        wd = e._coerce_map_input_array(k_weights)
        ws = e._coerce_map_input_array(shear_weights)
        if d_arr.ndim != 2 or wd.shape != d_arr.shape:
            raise ValueError("k_maps and k_weights must both be (n_lens, npix)")
        if s_arr.ndim != 3 or s_arr.shape[1] != 2 or tuple(ws.shape) != (s_arr.shape[0], s_arr.shape[2]):
            raise ValueError("shear_maps must be (n_source, 2, npix), shear_weights (n_source, npix)")
        nl, ns = int(d_arr.shape[0]), int(s_arr.shape[0])
        e._density_shear_tomo_vectorized(
            d_arr, s_arr, wd, ws, None, nl, ns,
            ggl_bin_combinations=combinations,
            shear_signs=(-1.0 if flip_g1 else 1.0, -1.0 if flip_g2 else 1.0),
        )
        return self._result(
            "kg", ("xi",), Integrated3PCF.kg_combinations(nl, ns, combinations), return_device
        )

    def three_by_two_pt(
        self,
        shear_maps: Any,
        k_maps: Any,
        shear_weights: Any,
        k_weights: Any,
        auto_only: bool = False,
        combinations: Optional[Sequence[Tuple[int, int]]] = None,
        flip_g1: bool = False,
        flip_g2: bool = False,
        return_device: bool = True,
    ) -> Dict[str, TwoPointResult]:
        """``{"gg": ..., "kk": ..., "kg": ...}`` in one call (the three pair
        statistics share one row expansion)."""
        e = self._engine
        with e._expansion_scope():
            return {
                "gg": self.gg(shear_maps, shear_weights, flip_g1=flip_g1, flip_g2=flip_g2,
                              return_device=return_device),
                "kk": self.kk(k_maps, k_weights, auto_only=auto_only, return_device=return_device),
                "kg": self.kg(k_maps, shear_maps, k_weights, shear_weights,
                              combinations=combinations, flip_g1=flip_g1, flip_g2=flip_g2,
                              return_device=return_device),
            }
