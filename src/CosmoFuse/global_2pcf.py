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

With ``device=[0, 1, ...]`` every device gets an engine of its own holding a
contiguous range of tiles (balanced by pair count) and the whole row space,
treecode cells included -- a pair may reach into a tile another device owns.
Measurements run in one thread per device and the per-tile sums are
concatenated in tile order, so the result is bitwise that of one device.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

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
from .multi_device import (
    balanced_ranges,
    device_list,
    require_host_arrays,
    run_on_devices,
)
from .packing import pack_host_patch
from .results import TwoPointResult
from .treecode import TreecodeGeometry, build_patch_cells, level_groups
from .utils import pixel2RaDec

FORMAT_KIND = "CosmoFuse.Global2PCF"
# Version 1: exact pairs (``pair_inds``/``pair_exp2phi``).  Version 2 is
# written only for ``pack_host_pairs=True``: the packed payload instead.
FORMAT_VERSION = 2
_EXACT_FORMAT_VERSION = 1
# packing jobs in flight behind the pair search of find_pairs()
_PACK_WORKERS = 4


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
        device: ``"cpu"``, ``"gpu"``, ``"auto"``, a GPU id, or a list of GPU
            ids.  A list splits the tiles across the devices (contiguous
            ranges with about equal pair counts, ``tile_ranges``); each device
            holds its share of the pairs and a copy of the maps.  Measurements
            then take host arrays and return host results, bitwise equal to
            a single-device run.
        map_precision, rotation_precision, accumulation_precision,
        resolution_factor, pair_search_precision, pack_pairs, fastmath:
            as for :class:`~CosmoFuse.correlations.Integrated3PCF`.  The
            static treecode builds its cells over the whole footprint (there
            is no patch window to preserve), from the binary mask.
        pack_host_pairs: pack every tile right after its pair search, so the
            host arrays and the pair file hold the 8-byte payload instead of
            the exact 24-byte pairs (as for ``Integrated3PCF``): a third of
            the host memory, and ``prepare()`` / ``load_pairs()`` skip the
            packing.  The device state is bitwise the one ``pack_pairs=True``
            builds in ``prepare()``; the file is then no longer exact.
    """

    def __init__(
        self,
        nside: int,
        mask: Optional[np.ndarray] = None,
        nbins: int = 10,
        theta_min: float = 10,
        theta_max: float = 170,
        tile_nside: Optional[int] = None,
        device: Union[str, int, Sequence[Union[str, int]]] = "auto",
        map_precision: Union[str, np.dtype, type] = "float64",
        rotation_precision: Union[str, np.dtype, type] = "float32",
        accumulation_precision: str = "same",
        resolution_factor: Union[None, bool, str, float] = None,
        pair_search_precision: str = "float64",
        pack_pairs: bool = False,
        fastmath: bool = True,
        pack_host_pairs: bool = False,
    ) -> None:
        devices: List[Any] = (
            device_list(device) if isinstance(device, (list, tuple, set)) else [device]
        )
        if theta_min >= theta_max:
            raise ValueError("theta_min must be less than theta_max")
        theta_max_rad = np.radians(theta_max / 60.0)
        if tile_nside is None:
            tile_nside = default_tile_nside(int(nside), theta_max_rad)
            auto_tiles = True
        else:
            auto_tiles = False
        # An engine is a Integrated3PCF over *tiles*; its patch centres are
        # the centres of the tiles it owns and its patch_size/theta_Q are
        # never used (no patch search, no aperture).  One engine per device.
        self._engines = [
            _TileEngine(
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
                device=dev,
                map_precision=map_precision,
                rotation_precision=rotation_precision,
                accumulation_precision=accumulation_precision,
                resolution_factor=resolution_factor,
                pair_search_precision=pair_search_precision,
                pack_pairs=pack_pairs,
            )
            for dev in devices
        ]
        self.devices = devices
        self.pack_host_pairs = bool(pack_host_pairs)
        # the engine every device-independent property is read from
        self._engine = e = self._engines[0]
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
        if self.tiles.n_tiles < len(self._engines):
            raise ValueError(
                f"{self.tiles.n_tiles} tiles cannot be split across "
                f"{len(self._engines)} devices; use a larger tile_nside"
            )
        patch_size = float(np.degrees(hp.max_pixrad(self.tiles.tile_nside)) * 60.0)
        for eng in self._engines:
            eng.patch_size = patch_size
        # (first, stop) tile of every device.  Several devices are balanced
        # by pair count, so their ranges are set when the pairs are installed.
        self.tile_ranges: Optional[List[Tuple[int, int]]] = None
        if len(self._engines) == 1:
            self._assign_tiles([(0, self.tiles.n_tiles)])

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
        return int(sum(int(e.ntotpairs) for e in self._engines))

    @property
    def backend(self) -> Any:
        """The backend of the (first) device."""
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
        """Find every footprint pair once, grouped by tile (host arrays).

        With ``pack_host_pairs=True`` every tile is packed in a worker thread
        while the next tile is searched, and its exact arrays are dropped as
        soon as it is packed.
        """
        e = self._engine
        treecode = self._build_treecode()
        elements = self._level_elements(treecode)
        groups = level_groups(e.level_nside)
        reach_cache: Dict[Tuple[int, float], np.ndarray] = {}
        bins: List[np.ndarray] = []
        if not self.pack_host_pairs:
            pair_inds: List[np.ndarray] = []
            pair_exp2phi: List[np.ndarray] = []
            for t in range(self.n_tiles):
                inds_t, exp_t, bins_t = self._search_tile(
                    t, elements, groups, reach_cache
                )
                pair_inds.append(inds_t)
                pair_exp2phi.append(exp_t)
                bins.append(bins_t)
            self._install(
                bins, treecode, pair_inds=pair_inds, pair_exp2phi=pair_exp2phi
            )
            return
        jobs: List["Future[Tuple[np.ndarray, np.ndarray, np.ndarray]]"] = []
        with ThreadPoolExecutor(max_workers=_PACK_WORKERS) as pool:
            try:
                for t in range(self.n_tiles):
                    inds_t, exp_t, bins_t = self._search_tile(
                        t, elements, groups, reach_cache
                    )
                    bins.append(bins_t)
                    jobs.append(
                        pool.submit(
                            pack_host_patch,
                            inds_t,
                            exp_t,
                            bins_t,
                            groups,
                            e.index_dtype,
                        )
                    )
                    del inds_t, exp_t
                    # bound the exact tiles waiting to be packed
                    while sum(not f.done() for f in jobs) > 2 * _PACK_WORKERS:
                        next(f for f in jobs if not f.done()).result()
                packed = [f.result() for f in jobs]
            except BaseException:
                for f in jobs:
                    f.cancel()
                raise
        self._install(
            bins,
            treecode,
            packed=(
                [p[0] for p in packed],
                [p[1] for p in packed],
                [p[2] for p in packed],
            ),
        )

    def _search_tile(
        self,
        t: int,
        elements: Dict[int, LevelElements],
        groups: Sequence[Tuple[int, int, int]],
        reach_cache: Dict[Tuple[int, float], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The unique pairs owned by tile ``t``, every resolution level."""
        e, ts = self._engine, self.tiles
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
                core_ids,
                core_xyz,
                cand_ids,
                cand_xyz,
                e.binedges[b0 : b1 + 1],
                fastmath=e.fastmath,
            )
            inds_t.append(np.stack((ids_a, ids_b)))
            exp = np.empty((2, ids_a.size), dtype=e.rotation_complex_dtype)
            exp[0].real = c1
            exp[0].imag = s1
            exp[1].real = c2
            exp[1].imag = s2
            exp_t.append(exp)
            bins_t[b0:b1] = counts
        return (
            np.concatenate(inds_t, axis=1).astype(e.index_dtype, copy=False),
            np.concatenate(exp_t, axis=1),
            bins_t.astype(e.index_dtype),
        )

    def _install(
        self,
        bins: List[np.ndarray],
        treecode: Optional[TreecodeGeometry],
        pair_inds: Optional[List[np.ndarray]] = None,
        pair_exp2phi: Optional[List[np.ndarray]] = None,
        packed: Optional[
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]
        ] = None,
    ) -> None:
        """Hand the tiles' pairs to the engines: exact (``pair_inds``,
        ``pair_exp2phi``) or the ``pack_host_pairs`` payload (``packed`` =
        per-tile payloads, block ids, block sizes)."""
        e = self._engine
        if treecode is not None:
            npix = hp.nside2npix(e.nside)
            if npix + treecode.n_cells > np.iinfo(e.index_dtype).max:
                raise OverflowError(
                    "pixel + coarse-cell ids exceed the index dtype range"
                )
        if len(self._engines) == 1:
            ranges = [(0, self.tiles.n_tiles)]
        else:
            counts = [int(np.sum(b, dtype=np.int64)) for b in bins]
            ranges = balanced_ranges(counts, len(self._engines))
        self._assign_tiles(ranges)
        for eng, (a, b) in zip(self._engines, ranges):
            # (the engine's host pair lists are None when packed or released)
            host: Any = eng
            if packed is None:
                assert pair_inds is not None and pair_exp2phi is not None
                eng.pair_inds = pair_inds[a:b]
                eng.pair_exp2phi = pair_exp2phi[a:b]
                eng.packed_pairs = None
                eng.packed_block_ids = None
                eng.packed_block_sizes = None
            else:
                host.pair_inds = None
                host.pair_exp2phi = None
                eng.packed_pairs = packed[0][a:b]
                eng.packed_block_ids = packed[1][a:b]
                eng.packed_block_sizes = packed[2][a:b]
            eng.bins = bins[a:b]
            # every engine holds every cell: its pairs reach into other tiles
            eng._treecode = treecode
            eng._invalidate_prepared_state()

    def _assign_tiles(self, ranges: List[Tuple[int, int]]) -> None:
        """Give engine ``i`` the tiles ``ranges[i]`` as its patches."""
        phi, theta = self.tiles.centre_ang
        for eng, (a, b) in zip(self._engines, ranges):
            eng.phi_center = phi[a:b]
            eng.theta_center = theta[a:b]
            eng.n_patches = b - a
        self.tile_ranges = list(ranges)

    def _on_engines(self, fn: Callable[[_TileEngine], Any]) -> List[Any]:
        """``fn(engine)`` for every engine; one thread per device if several."""
        if len(self._engines) == 1:
            return [fn(self._engine)]
        return run_on_devices(self._engines, lambda eng, _i: fn(eng))

    def prepare(self, release_host_pairs: bool = False) -> None:
        """Move the pairs to the backend device(s) (idempotent)."""
        self._on_engines(lambda eng: eng.prepare(release_host_pairs=release_host_pairs))

    def preprocess(self, release_host_pairs: bool = False) -> None:
        """``find_pairs()`` + ``prepare()``."""
        self.find_pairs()
        self.prepare(release_host_pairs=release_host_pairs)

    def save_pairs(self, filepath: str) -> None:
        """Write the pair geometry (host arrays) to HDF5: exact, or the packed
        payload when the pairs were found with ``pack_host_pairs=True``."""
        e = self._engine
        engines: List[Any] = self._engines  # host lists may be None
        packed = all(eng.packed_pairs is not None for eng in engines)
        if any(
            eng.bins is None or (eng.pair_inds is None and eng.packed_pairs is None)
            for eng in engines
        ):
            raise RuntimeError(
                "host pairs were released (release_host_pairs=True); recompute "
                "them before save_pairs()."
            )
        # the tiles of all devices, in tile order
        bins: List[np.ndarray] = []
        for eng in self._engines:
            assert eng.bins is not None  # checked above
            bins.extend(eng.bins)
        counts = np.array([int(np.sum(b, dtype=np.int64)) for b in bins], np.int64)
        offsets = np.concatenate(([0], np.cumsum(counts)))
        with h5py.File(filepath, "w") as fp:
            fp.attrs["kind"] = FORMAT_KIND
            fp.attrs["format_version"] = (
                FORMAT_VERSION if packed else _EXACT_FORMAT_VERSION
            )
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
            fp.create_dataset(
                "level_nside", data=np.asarray(e.level_nside, dtype=np.int64)
            )
            fp.create_dataset("pair_offsets", data=offsets)
            fp.create_dataset(
                "bins", data=np.stack([np.asarray(b, dtype=np.int64) for b in bins])
            )
            if packed:
                self._write_packed(fp, offsets)
            else:
                self._write_exact(fp, offsets)
            tc = e._treecode
            if tc is not None:
                gp = fp.create_group("treecode")
                gp.attrs["base_nside"] = int(tc.base_nside)
                gp.create_dataset(
                    "coarse_nsides", data=np.asarray(tc.coarse_nsides, dtype=np.int64)
                )
                gp.create_dataset("cell_offsets", data=tc.cell_offsets)
                for lv in range(tc.n_levels):
                    gp.create_dataset(f"child_indptr_{lv}", data=tc.child_indptr[lv])
                    gp.create_dataset(f"child_indices_{lv}", data=tc.child_indices[lv])
                    gp.create_dataset(f"cell_ra_{lv}", data=tc.cell_ra[lv])
                    gp.create_dataset(f"cell_dec_{lv}", data=tc.cell_dec[lv])

    def _write_exact(self, fp: h5py.File, offsets: np.ndarray) -> None:
        """Per tile into preallocated datasets (no catalogue-sized copy)."""
        e = self._engine
        pair_inds: List[np.ndarray] = []
        pair_exp2phi: List[np.ndarray] = []
        for eng in self._engines:
            assert eng.pair_inds is not None and eng.pair_exp2phi is not None
            pair_inds.extend(eng.pair_inds)
            pair_exp2phi.extend(eng.pair_exp2phi)
        n = int(offsets[-1])
        d_inds = fp.create_dataset("pair_inds", shape=(2, n), dtype=e.index_dtype)
        d_exp = fp.create_dataset(
            "pair_exp2phi", shape=(2, n), dtype=e.rotation_complex_dtype
        )
        for t, (inds, exp) in enumerate(zip(pair_inds, pair_exp2phi)):
            a, b = int(offsets[t]), int(offsets[t + 1])
            if b > a:
                d_inds[:, a:b] = inds
                d_exp[:, a:b] = exp

    def _write_packed(self, fp: h5py.File, offsets: np.ndarray) -> None:
        e = self._engine
        engines: List[Any] = self._engines  # checked by save_pairs()
        packed = [p for eng in engines for p in eng.packed_pairs]
        block_ids = [p for eng in engines for p in eng.packed_block_ids]
        block_sizes = [p for eng in engines for p in eng.packed_block_sizes]
        block_offsets = np.concatenate(
            ([0], np.cumsum([b.size for b in block_ids], dtype=np.int64))
        )
        fp.attrs["pack_host_pairs"] = True
        d_packed = fp.create_dataset(
            "packed_pairs", shape=(int(offsets[-1]), 4), dtype=np.uint16
        )
        for t, payload in enumerate(packed):
            a, b = int(offsets[t]), int(offsets[t + 1])
            if b > a:
                d_packed[a:b] = payload
        fp.create_dataset(
            "packed_block_ids",
            data=(
                np.concatenate(block_ids).astype(e.index_dtype, copy=False)
                if block_ids
                else np.zeros(0, dtype=e.index_dtype)
            ),
        )
        fp.create_dataset("packed_block_offsets", data=block_offsets)
        fp.create_dataset(
            "packed_block_sizes",
            data=np.stack([np.asarray(b, dtype=np.int64) for b in block_sizes]),
        )

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
                    raise ValueError(
                        f"{filepath}: {name}={got}, this estimator has {want}"
                    )
            if not np.allclose(
                [fp.attrs["theta_min"], fp.attrs["theta_max"]],
                [e.theta_min, e.theta_max],
                rtol=1e-12,
                atol=0.0,
            ):
                raise ValueError(
                    f"{filepath}: angular range differs from this estimator"
                )
            if not np.array_equal(
                fp["map_inds"][:], np.asarray(e.map_inds, dtype=np.int64)
            ):
                raise ValueError(
                    f"{filepath}: the mask (row space) differs from this estimator"
                )
            if not np.array_equal(
                fp["level_nside"][:], np.asarray(e.level_nside, dtype=np.int64)
            ):
                raise ValueError(
                    f"{filepath}: resolution levels "
                    f"{fp['level_nside'][:].tolist()} differ "
                    f"from this estimator's {np.asarray(e.level_nside).tolist()} "
                    "(resolution_factor)"
                )
            offsets = fp["pair_offsets"][:]
            bins_arr = fp["bins"][:]
            n = self.n_tiles
            packed = None
            pair_inds = pair_exp2phi = None
            if "packed_pairs" in fp:
                flat = fp["packed_pairs"][:]
                ids_flat = fp["packed_block_ids"][:].astype(e.index_dtype, copy=False)
                b_off = fp["packed_block_offsets"][:]
                b_sizes = fp["packed_block_sizes"][:].astype(np.int64)
                packed = (
                    [flat[offsets[t] : offsets[t + 1]] for t in range(n)],
                    [ids_flat[b_off[t] : b_off[t + 1]] for t in range(n)],
                    [b_sizes[t] for t in range(n)],
                )
            else:
                inds_flat = fp["pair_inds"][:].astype(e.index_dtype, copy=False)
                exp_flat = fp["pair_exp2phi"][:].astype(
                    e.rotation_complex_dtype, copy=False
                )
                pair_inds = [
                    inds_flat[:, offsets[t] : offsets[t + 1]] for t in range(n)
                ]
                pair_exp2phi = [
                    exp_flat[:, offsets[t] : offsets[t + 1]] for t in range(n)
                ]
            treecode = None
            if "treecode" in fp:
                gp = fp["treecode"]
                n_levels = int(gp["coarse_nsides"].shape[0])
                treecode = TreecodeGeometry(
                    base_nside=int(gp.attrs["base_nside"]),
                    coarse_nsides=tuple(int(n) for n in gp["coarse_nsides"][:]),
                    cell_offsets=gp["cell_offsets"][:].astype(np.int64),
                    child_indptr=[
                        gp[f"child_indptr_{lv}"][:] for lv in range(n_levels)
                    ],
                    child_indices=[
                        gp[f"child_indices_{lv}"][:] for lv in range(n_levels)
                    ],
                    cell_ra=[gp[f"cell_ra_{lv}"][:] for lv in range(n_levels)],
                    cell_dec=[gp[f"cell_dec_{lv}"][:] for lv in range(n_levels)],
                )
        bins = [bins_arr[t].astype(e.index_dtype) for t in range(n)]
        # The file decides what the engines hold; pack_host_pairs only says how
        # *future* pair searches behave (as for Integrated3PCF.load_pairs).
        self._install(
            bins,
            treecode,
            pair_inds=pair_inds,
            pair_exp2phi=pair_exp2phi,
            packed=packed,
        )
        self.prepare(release_host_pairs=release_host_pairs)

    # ------------------------------------------------------------------ #
    # measurements
    # ------------------------------------------------------------------ #

    def _result(
        self,
        statistic: str,
        names: Tuple[str, ...],
        combinations: Sequence[Tuple[int, int]],
        captured: Sequence[Tuple[Tuple[Any, ...], Any, int]],
        return_device: bool,
    ) -> TwoPointResult:
        """Label the captured sums of every engine, tiles in tile order."""
        parts = []
        for eng, (nums, den, ncomb) in zip(self._engines, captured):
            xp = eng.backend.module
            shape = (ncomb, eng.n_patches, self.nbins)
            acc = getattr(xp, eng.acc_dtype.name)
            tile_num = {
                n: xp.real(a).reshape(shape).astype(acc, copy=False)
                for n, a in zip(names, nums)
            }
            tile_weight = xp.real(den).reshape(shape).astype(acc, copy=False)
            parts.append((eng, tile_num, tile_weight))
        combos = tuple((int(a), int(b)) for a, b in combinations)
        binedges = np.asarray(self.binedges).copy()
        if len(parts) == 1:
            eng, tile_num, tile_weight = parts[0]
            if return_device and eng.backend.name == "cupy":
                return TwoPointResult(
                    statistic, binedges, combos, tile_num, tile_weight
                )

        def host(eng: Any, a: Any) -> np.ndarray:
            return np.asarray(eng.backend.to_numpy(a))

        return TwoPointResult(
            statistic,
            binedges,
            combos,
            {
                n: np.concatenate([host(eng, t[n]) for eng, t, _ in parts], axis=1)
                for n in names
            },
            np.concatenate([host(eng, w) for eng, _, w in parts], axis=1),
        )

    def _check_inputs(self, name: str, *arrays: Any) -> None:
        if len(self._engines) > 1:
            require_host_arrays(name, arrays)

    @staticmethod
    def _gg_sums(
        e: _TileEngine, shear_maps: Any, w: Any, flip_g1: bool, flip_g2: bool
    ) -> Tuple[Tuple[Tuple[Any, ...], Any, int], List[Tuple[int, int]]]:
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
            shear_dev,
            w_dev,
            None,
            nz,
            ncomb,
            -1 if flip_g1 else 1,
            -1 if flip_g2 else 1,
        )
        if res is None:
            raise RuntimeError(
                "the tomographic ξ± kernel is unavailable on this backend"
            )
        return e.take_captured(), Integrated3PCF.tomo_combinations(nz)

    @staticmethod
    def _kk_sums(
        e: _TileEngine, k_maps: Any, w: Any, auto_only: bool
    ) -> Tuple[Tuple[Tuple[Any, ...], Any, int], List[Tuple[int, int]]]:
        e._ensure_prepared()
        arr = e._coerce_map_input_array(k_maps)
        if arr.ndim != 2:
            raise ValueError(f"k_maps must be (nz, npix); got {arr.shape}")
        nz = int(arr.shape[0])
        e._density_density_tomo_vectorized(
            arr, w, None, nz, gc_auto_correlations_only=auto_only
        )
        return e.take_captured(), Integrated3PCF.tomo_combinations(nz, auto_only)

    @staticmethod
    def _kg_sums(
        e: _TileEngine,
        k_maps: Any,
        shear_maps: Any,
        k_weights: Any,
        shear_weights: Any,
        combinations: Optional[Sequence[Tuple[int, int]]],
        flip_g1: bool,
        flip_g2: bool,
    ) -> Tuple[Tuple[Tuple[Any, ...], Any, int], List[Tuple[int, int]]]:
        e._ensure_prepared()
        d_arr = e._coerce_map_input_array(k_maps)
        s_arr = e._coerce_map_input_array(shear_maps)
        wd = e._coerce_map_input_array(k_weights)
        ws = e._coerce_map_input_array(shear_weights)
        if d_arr.ndim != 2 or wd.shape != d_arr.shape:
            raise ValueError("k_maps and k_weights must both be (n_lens, npix)")
        if (
            s_arr.ndim != 3
            or s_arr.shape[1] != 2
            or tuple(ws.shape) != (s_arr.shape[0], s_arr.shape[2])
        ):
            raise ValueError(
                "shear_maps must be (n_source, 2, npix), shear_weights (n_source, npix)"
            )
        nl, ns = int(d_arr.shape[0]), int(s_arr.shape[0])
        e._density_shear_tomo_vectorized(
            d_arr,
            s_arr,
            wd,
            ws,
            None,
            nl,
            ns,
            ggl_bin_combinations=combinations,
            shear_signs=(-1.0 if flip_g1 else 1.0, -1.0 if flip_g2 else 1.0),
        )
        return e.take_captured(), Integrated3PCF.kg_combinations(nl, ns, combinations)

    def _labelled(
        self,
        statistic: str,
        names: Tuple[str, ...],
        per_engine: Sequence[Tuple[Any, List[Tuple[int, int]]]],
        return_device: bool,
    ) -> TwoPointResult:
        return self._result(
            statistic,
            names,
            per_engine[0][1],
            [captured for captured, _ in per_engine],
            return_device,
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
        self._check_inputs("gg", shear_maps, w)
        out = self._on_engines(
            lambda e: self._gg_sums(e, shear_maps, w, flip_g1, flip_g2)
        )
        return self._labelled("gg", ("xi_p", "xi_m"), out, return_device)

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
        self._check_inputs("kk", k_maps, w)
        out = self._on_engines(lambda e: self._kk_sums(e, k_maps, w, auto_only))
        return self._labelled("kk", ("xi",), out, return_device)

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
        self._check_inputs("kg", k_maps, shear_maps, k_weights, shear_weights)
        out = self._on_engines(
            lambda e: self._kg_sums(
                e,
                k_maps,
                shear_maps,
                k_weights,
                shear_weights,
                combinations,
                flip_g1,
                flip_g2,
            )
        )
        return self._labelled("kg", ("xi",), out, return_device)

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
        self._check_inputs(
            "three_by_two_pt", shear_maps, k_maps, shear_weights, k_weights
        )

        def run(e: _TileEngine) -> Dict[str, Any]:
            with e._expansion_scope():
                return {
                    "gg": self._gg_sums(e, shear_maps, shear_weights, flip_g1, flip_g2),
                    "kk": self._kk_sums(e, k_maps, k_weights, auto_only),
                    "kg": self._kg_sums(
                        e,
                        k_maps,
                        shear_maps,
                        k_weights,
                        shear_weights,
                        combinations,
                        flip_g1,
                        flip_g2,
                    ),
                }

        out = self._on_engines(run)
        names = {"gg": ("xi_p", "xi_m"), "kk": ("xi",), "kg": ("xi",)}
        return {
            key: self._labelled(key, names[key], [o[key] for o in out], return_device)
            for key in ("gg", "kk", "kg")
        }
