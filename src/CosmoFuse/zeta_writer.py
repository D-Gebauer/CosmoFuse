"""Streaming i3PCF reduction and HDF5 output.

A measurement loop normally keeps every per-patch result in RAM and writes
one array at the end.  At the DES Y3 production geometry (nside 512, 917
patches, 7 angular bins, 4 tomographic bins) one map-set is

    M_a (4, 917) + xi± (10, 917, 7) x 2  =  132,048 doubles  =  1.06 MB,

so a 10,000-realisation suite is ~10.6 GB held until the very end -- and
lost entirely if the run dies at realisation 9,500.

:class:`ZetaWriter` turns that into a stream.  ``zeta`` is an average over
*patches* (:func:`CosmoFuse.correlation_helpers.calculate_all_zetas`
averages over axis 2 and carries the map axis through untouched), so the
reduction of map-set *k* needs nothing but map-set *k*: it can run the
moment the measurement returns and the per-patch arrays can be dropped.
Reduced, the same map-set is 8 variants x (20 triplets, 7 bins) = **9 kB**,
118x smaller.

Two storage modes:

``reduce="zeta"``
    9 kB per map-set (90 MB for 10,000 realisations).  The patch axis is
    gone, so no jackknife and no re-binning later.
``reduce="none"``
    the raw 1.06 MB per map-set.  Everything -- zeta, a leave-one-patch-out
    jackknife, a different binning -- can still be derived from it.

(Storing *per-patch* zeta contributions is strictly worse than either:
8.2 MB per map-set, 8x the raw arrays, and it holds less information than
they do, since the jackknife is recoverable from the raw arrays exactly.)

Where the work happens
----------------------

``submit()`` runs on the calling thread and never leaves device memory
referenced by the queue -- anything on a GPU is brought to the host inside
``submit()``, after reducing on the device when that shrinks the transfer:

===========================  ==============================  ==================
input                         ``submit()`` (calling thread)   writer thread
===========================  ==============================  ==================
GPU + ``reduce="zeta"``       reduce on device, 9 kB D2H      HDF5 only
GPU + ``reduce="none"``       1.06 MB D2H                     HDF5 only
host (multi-GPU, CPU)         queue the arrays                reduce + HDF5
===========================  ==============================  ==================

:class:`~CosmoFuse.multi_device.MultiDeviceCorrelation` concatenates the
per-device outputs on the patch axis and always returns host arrays, so it
lands in the third row: the reduction runs on the CPU, in the writer
thread, off the measurement's critical path.  No device code is involved
and nothing about the group needs to change.

Both numpy and h5py release the GIL for the bulk of their work, so a plain
thread genuinely overlaps with the next map-set's kernels.
"""

import json
import queue
import threading
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .correlation_helpers import calculate_all_zetas

# Central (aperture) and annular (2PCF) fields, in the order
# get_3x2pt_tomo returns them.
CENTRAL_FIELDS: Tuple[str, ...] = ("M_a", "M_g")
ANNULAR_FIELDS: Tuple[str, ...] = ("xi_p", "xi_m", "xi_g", "xi_t")
FIELDS: Tuple[str, ...] = ("M_a", "M_g", "xi_p", "xi_m", "xi_g", "xi_t")

# Target size of one HDF5 chunk along the map axis.
_CHUNK_BYTES = 1 << 20

_PROVENANCE_ATTRS = (
    "nside",
    "nbins",
    "theta_min",
    "theta_max",
    "patch_size",
    "theta_Q",
    "n_patches",
    "resolution_factor",
    "aperture_nside",
    "pack_pairs",
    "pack_host_pairs",
    "pair_search_precision",
)


def _is_device_array(a: Any) -> bool:
    return type(a).__module__.split(".")[0] == "cupy"


def _to_host(a: Any) -> np.ndarray:
    if _is_device_array(a):
        import cupy

        return cupy.asnumpy(a)
    return np.asarray(a)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


class ZetaWriter:
    """Reduce each map-set to zeta (or keep it raw) and append it to HDF5.

    Args:
        path: Output HDF5 file.
        corr: The :class:`~CosmoFuse.correlations.Correlation` (or
            :class:`~CosmoFuse.multi_device.MultiDeviceCorrelation`) that
            produced the measurements.  Read once, for provenance.
        reduce: ``"zeta"`` (default) stores the eight i3PCF variants;
            ``"none"`` stores the raw per-patch arrays.
        flush_every: Flush to disk every this many map-sets.  A crash
            costs at most one window; ``n_flushed`` in the file records
            what is durable.
        depth: Queue depth.  ``submit()`` blocks once this many map-sets
            are waiting, which bounds memory and keeps the writer from
            falling behind unnoticed.
        swmr: Enable HDF5 single-writer/multiple-reader so another process
            can follow the file while it grows.  **Off by default, and it
            is not free** -- see the class notes below.
        resume: Open an existing file and continue appending after its
            last flushed map-set, discarding anything written past it.

    SWMR notes (measured with HDF5 1.14.6):
        * it forces ``libver="latest"`` (superblock v3), so the file needs
          HDF5 >= 1.10 to read at all;
        * readers must pass ``swmr=True``;
        * **after a crash the file keeps a stale write lock**: an ordinary
          ``h5py.File(path, "r")`` then fails with *"file is already open
          for write"* until someone runs ``h5clear -s <file>``.  A
          non-SWMR file written by this class survives the same crash and
          opens normally, because ``flush_every`` already makes the
          flushed prefix durable.
        * HDF5 documents SWMR as unreliable on NFS.

        So SWMR buys exactly one thing -- watching a run from another
        process -- and charges a worse crash story for it.  Leave it off
        unless you want the live view.
    """

    def __init__(
        self,
        path: str,
        corr: Any,
        *,
        reduce: str = "zeta",
        flush_every: int = 50,
        depth: int = 2,
        swmr: bool = False,
        resume: bool = False,
    ) -> None:
        if reduce not in ("zeta", "none"):
            raise ValueError(f"reduce must be 'zeta' or 'none'; got {reduce!r}")
        if flush_every < 1:
            raise ValueError("flush_every must be >= 1")
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.path = str(path)
        self.reduce = reduce
        self.flush_every = int(flush_every)
        self.swmr = bool(swmr)
        self._resume = bool(resume)
        self._provenance = self._collect_provenance(corr)

        self._queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(
            maxsize=depth
        )
        self._error: Optional[BaseException] = None
        self._lock = threading.Lock()
        self._n_submitted = 0
        self._n_appended = 0
        self._closed = False

        self._file: Any = None
        self._group: Any = None
        self._datasets: Dict[str, Any] = {}
        self._n_flushed_ds: Any = None

        self._thread = threading.Thread(
            target=self._drain, name="ZetaWriter", daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------ #
    # submission (calling thread)
    # ------------------------------------------------------------------ #

    def submit(self, **fields: Any) -> None:
        """Queue one map-set.

        Accepts any subset of ``M_a``, ``M_g``, ``xi_p``, ``xi_m``,
        ``xi_g``, ``xi_t``, shaped exactly as the measurement methods
        return them (no leading map axis).  Blocks when the queue is full.
        """
        self._raise_deferred()
        if self._closed:
            raise RuntimeError("ZetaWriter is closed")
        unknown = set(fields) - set(FIELDS)
        if unknown:
            raise ValueError(f"unknown field(s) {sorted(unknown)}; expected {FIELDS}")
        given = {k: v for k, v in fields.items() if v is not None}
        if not given:
            raise ValueError("submit() needs at least one field")
        if not any(k in given for k in CENTRAL_FIELDS):
            raise ValueError(f"submit() needs a central field ({CENTRAL_FIELDS})")
        if not any(k in given for k in ANNULAR_FIELDS):
            raise ValueError(f"submit() needs an annular field ({ANNULAR_FIELDS})")

        self._queue.put(self._prepare(given))
        with self._lock:
            self._n_submitted += 1

    def submit_3x2pt(self, result: Tuple[Any, ...]) -> None:
        """Queue a :meth:`Correlation.get_3x2pt_tomo` result
        ``(M_a, M_g, xi_p, xi_m, xi_g, xi_t)``."""
        if len(result) != 6:
            raise ValueError(f"expected a 6-tuple from get_3x2pt_tomo, got {len(result)}")
        self.submit(**dict(zip(FIELDS, result)))

    def submit_shear(self, result: Tuple[Any, ...]) -> None:
        """Queue a :meth:`Correlation.get_full_tomo_shear` result
        ``(M_a, xi_p, xi_m)``."""
        if len(result) != 3:
            raise ValueError(
                f"expected a 3-tuple from get_full_tomo_shear, got {len(result)}"
            )
        M_a, xi_p, xi_m = result
        self.submit(M_a=M_a, xi_p=xi_p, xi_m=xi_m)

    def _prepare(self, given: Mapping[str, Any]) -> Dict[str, Any]:
        """Add the map axis and get everything onto the host.

        Device arrays are reduced *before* the copy when that shrinks it,
        and nothing on a device is ever handed to the writer thread: the
        thread only ever touches host numpy and h5py, so no stream or
        buffer-lifetime hazard can arise.
        """
        batched = {k: v[None, ...] for k, v in given.items()}
        on_device = any(_is_device_array(v) for v in batched.values())

        if on_device and self.reduce == "zeta":
            zetas = calculate_all_zetas(**batched)
            self._require_zetas(zetas)
            return {"zeta": {k: _to_host(v) for k, v in zetas.items()}}
        if on_device:
            return {"raw": {k: _to_host(v) for k, v in batched.items()}}
        # Host input (multi-device group or CPU backend): defer the
        # reduction to the writer thread, off the critical path.
        return {"raw": {k: np.asarray(v) for k, v in batched.items()}}

    @staticmethod
    def _require_zetas(zetas: Mapping[str, Any]) -> None:
        if not zetas:
            raise ValueError(
                "the submitted fields do not form any zeta: pair a central "
                f"field {CENTRAL_FIELDS} with an annular one {ANNULAR_FIELDS}"
            )

    # ------------------------------------------------------------------ #
    # writer thread
    # ------------------------------------------------------------------ #

    def _drain(self) -> None:
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is None:
                        return
                    self._consume(item)
                finally:
                    self._queue.task_done()
        except BaseException as exc:  # surfaced on the next submit()/close()
            with self._lock:
                self._error = exc
            self._drain_remaining()

    def _drain_remaining(self) -> None:
        """Keep the queue moving after a failure so submit() cannot wedge."""
        while True:
            item = self._queue.get()
            self._queue.task_done()
            if item is None:
                return

    def _consume(self, item: Dict[str, Any]) -> None:
        if "raw" in item and self.reduce == "zeta":
            zetas = calculate_all_zetas(**item["raw"])
            self._require_zetas(zetas)
            payload = zetas
        else:
            payload = item.get("zeta", item.get("raw", {}))

        if self._file is None:
            self._open(payload)
        self._append(payload)

        self._n_appended += 1
        if self._n_appended % self.flush_every == 0:
            self._flush()

    def _open(self, payload: Mapping[str, np.ndarray]) -> None:
        import h5py

        group_name = "zeta" if self.reduce == "zeta" else "raw"
        libver = "latest" if self.swmr else None

        if self._resume:
            self._file = h5py.File(self.path, "a", libver=libver)
            self._group = self._file[group_name]
            self._n_flushed_ds = self._file["n_flushed"]
            self._datasets = {k: self._group[k] for k in payload}
            durable = int(self._n_flushed_ds[()])
            # discard anything written past the last flush: those rows may
            # be partial
            for ds in self._datasets.values():
                ds.resize(durable, axis=0)
            self._n_appended = durable
        else:
            self._file = h5py.File(self.path, "w", libver=libver)
            for key, value in self._provenance.items():
                self._file.attrs[key] = value
            self._group = self._file.create_group(group_name)
            self._n_flushed_ds = self._file.create_dataset(
                "n_flushed", shape=(), dtype=np.int64
            )
            for name, arr in payload.items():
                self._datasets[name] = self._group.create_dataset(
                    name,
                    shape=(0,) + arr.shape[1:],
                    maxshape=(None,) + arr.shape[1:],
                    chunks=self._chunks(arr),
                    dtype=arr.dtype,
                )
        # SWMR must be enabled only once every object exists.
        if self.swmr:
            self._file.swmr_mode = True

    @staticmethod
    def _chunks(arr: np.ndarray) -> Tuple[int, ...]:
        per_map = int(arr.dtype.itemsize * np.prod(arr.shape[1:], dtype=np.int64))
        n = max(1, _CHUNK_BYTES // max(per_map, 1))
        return (int(n),) + arr.shape[1:]

    def _append(self, payload: Mapping[str, np.ndarray]) -> None:
        missing = set(self._datasets) ^ set(payload)
        if missing:
            raise ValueError(
                "every map-set must carry the same fields; this one differs in "
                f"{sorted(missing)}"
            )
        row = self._n_appended
        for name, arr in payload.items():
            ds = self._datasets[name]
            if arr.shape[1:] != ds.shape[1:]:
                raise ValueError(
                    f"{name} has shape {arr.shape[1:]} but the file holds "
                    f"{ds.shape[1:]}"
                )
            ds.resize(row + 1, axis=0)
            ds[row] = arr[0]

    def _flush(self) -> None:
        self._n_flushed_ds[()] = self._n_appended
        self._file.flush()

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def _raise_deferred(self) -> None:
        with self._lock:
            exc = self._error
        if exc is not None:
            raise RuntimeError("the ZetaWriter thread failed") from exc

    @property
    def n_submitted(self) -> int:
        """Map-sets handed to :meth:`submit`."""
        with self._lock:
            return self._n_submitted

    @property
    def n_written(self) -> int:
        """Map-sets appended to the file (flushed or not)."""
        return self._n_appended

    def close(self) -> None:
        """Drain the queue, flush and close.  Idempotent."""
        if self._closed:
            self._raise_deferred()
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()
        try:
            if self._file is not None:
                try:
                    if self._error is None:
                        self._flush()
                finally:
                    self._file.close()
                    self._file = None
        finally:
            self._raise_deferred()

    def __enter__(self) -> "ZetaWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __len__(self) -> int:
        return self._n_appended

    def __repr__(self) -> str:
        return (
            f"<ZetaWriter {self.path!r} reduce={self.reduce!r} "
            f"written={self._n_appended}>"
        )

    # ------------------------------------------------------------------ #

    @staticmethod
    def _collect_provenance(corr: Any) -> Dict[str, Any]:
        """Everything needed to tell two data vectors apart.

        A treecode measurement is only comparable to another made with the
        same level table, and a row-space archive only to the mask it was
        built for -- so both go in the file.
        """
        from . import __version__

        out: Dict[str, Any] = {"cosmofuse_version": __version__}
        for name in _PROVENANCE_ATTRS:
            value = getattr(corr, name, None)
            if value is None:
                continue
            out[name] = value if not isinstance(value, bool) else bool(value)
        for name in ("level_table", "row_pix_hash"):
            try:
                value = getattr(corr, name)
            except Exception:
                continue
            out[name] = (
                value
                if isinstance(value, str)
                else json.dumps(value, default=_json_default, sort_keys=True)
            )
        return out
