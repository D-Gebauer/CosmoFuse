"""
Double-buffered map upload pipeline for GPU measurement loops.

When measuring many maps in sequence, the host→device transfer of map k+1
can overlap the kernels of map k: the copy is enqueued on a dedicated
non-blocking CUDA stream from pinned (page-locked) host memory, and the
compute (current) stream only waits on the copy event when the swapped-in
buffers are first used.  On CPU backends every operation degrades to a
plain copy, so the same driver loop runs everywhere.
"""

from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np


class PinnedMapPipeline:
    """Overlap map k+1 host→device transfer with map k compute.

    Two pinned host slots and two device slots are allocated per named
    array; ``stage()`` enqueues the async upload of the next map into the
    back slot and ``wait()`` swaps it in, making the compute stream wait
    on the copy.

    Usage::

        pipe = PinnedMapPipeline(corr, {"shear": (nz, 2, npix), "w": (nz, npix)})
        dev = pipe.wait(pipe.stage({"shear": shear_np[0], "w": w_np[0]}))
        for k in range(nmaps):
            nxt = (
                pipe.stage({"shear": shear_np[k + 1], "w": w_np[k + 1]})
                if k + 1 < nmaps
                else None
            )
            results.append(corr.get_full_tomo_shear(dev["shear"], dev["w"]))
            dev = pipe.wait(nxt)

    The measurement methods accept device arrays directly, so no API
    changes are needed on the ``Correlation`` side.  Numerically a no-op:
    ``stage()`` only copies bytes.
    """

    def __init__(
        self,
        corr: Any,
        shapes: Mapping[str, Tuple[int, ...]],
        dtype: Optional[Union[str, np.dtype, type]] = None,
    ) -> None:
        self.backend = corr.backend
        dtype = np.dtype(dtype or corr.map_dtype)
        self.stream = self.backend.create_stream(non_blocking=True)  # None on CPU
        # Two pinned host slots + two device slots per named array
        self.host = [
            {k: self.backend.alloc_pinned(s, dtype) for k, s in shapes.items()}
            for _ in range(2)
        ]
        self.dev = [
            {k: self.backend.zeros(s, dtype=dtype) for k, s in shapes.items()}
            for _ in range(2)
        ]
        self.slot = 0
        self.event: Optional[Any] = None
        self._slot_events: list = [None, None]

    def stage(self, host_arrays: Mapping[str, Any]) -> int:
        """Enqueue the async host→device copy of *host_arrays* into the
        back slot; returns the slot token to pass to :meth:`wait`."""
        back = 1 - self.slot
        if self._slot_events[back] is not None:
            # The previous asynchronous copy out of this pinned slot may
            # still be queued when the host runs ahead of the GPU; it must
            # complete before the pinned buffer is refilled.
            self._slot_events[back].synchronize()
        if self.stream is not None:
            # Kernels queued so far may still read the back slot's device
            # buffers (it held the map before the current one): the upload
            # must not overwrite them before those kernels have run.  It
            # still overlaps the kernels of the *current* map, which are
            # enqueued after this call.
            released = self.backend.module.cuda.get_current_stream().record()
            self.stream.wait_event(released)
        for k, arr in host_arrays.items():
            np.copyto(self.host[back][k], arr)  # host -> pinned (CPU-side)
            if self.stream is not None:
                with self.stream:
                    self.dev[back][k].set(self.host[back][k])  # pinned -> device, async
            else:
                np.copyto(self.dev[back][k], self.host[back][k])
        self.event = self.stream.record() if self.stream is not None else None
        self._slot_events[back] = self.event
        return back

    def wait(self, staged: Optional[int]) -> Dict[str, Any]:
        """Make the staged slot current; the compute stream waits on the
        copy.  ``wait(None)`` returns the current slot unchanged."""
        if staged is None:
            return self.dev[self.slot]
        if self.event is not None:
            self.backend.module.cuda.get_current_stream().wait_event(self.event)
        self.slot = staged
        return self.dev[self.slot]


class RowSpaceMapLoader:
    """Ring-buffer loader: keep the GPU fed while measuring many map-sets.

    ``n_readers`` background threads fill a ring of ``n_slots`` *pinned* host
    buffers straight from the map source (``read_fn`` writes into the pinned
    arrays -- no pageable staging copy), enqueue the host→device copy on a
    dedicated non-blocking stream, and hand the device buffers to the
    consumer strictly in source order.  The compute stream only waits on the
    copy event of the slot it is about to use; a slot's device buffers are
    not overwritten before the kernels that used them have finished (the
    upload stream waits on a "released" event recorded on the compute
    stream), and a slot's pinned host buffer is not refilled before its
    previous asynchronous copy has completed.  On CPU backends the same loop
    runs with plain copies.

    Usage::

        def read(source, out):               # runs in a reader thread
            out["shear"][...] = np.load(source, mmap_mode="r")

        loader = RowSpaceMapLoader(
            corr, {"shear": (nz, 2, corr.n_active)}, sources=files, read_fn=read)
        for k, dev in loader:
            results.append(corr.get_full_tomo_shear(dev["shear"], w, flip_g1=True))

    Maps should be stored in the row space of the pair geometry
    (``Correlation.row_pix`` order, see :meth:`Correlation.to_row_space`);
    pass ``row_pix_hash`` (stored next to the map archive) to fail loudly
    when the archive and the geometry do not belong together.  The device
    arrays handed out are only valid until the next iteration.
    """

    def __init__(
        self,
        corr: Any,
        shapes: Mapping[str, Tuple[int, ...]],
        sources: Any,
        read_fn: Any,
        n_slots: int = 4,
        n_readers: int = 2,
        dtype: Optional[Union[str, np.dtype, type]] = None,
        row_pix_hash: Optional[str] = None,
    ) -> None:
        if n_slots < 2:
            raise ValueError("n_slots must be >= 2")
        if n_readers < 1:
            raise ValueError("n_readers must be >= 1")
        if row_pix_hash is not None and row_pix_hash != corr.row_pix_hash:
            raise ValueError(
                "row_pix hash mismatch: the map archive was written for a "
                "different mask / row space than this pair geometry."
            )
        self.backend = corr.backend
        self.sources = list(sources)
        self.read_fn = read_fn
        self.n_readers = int(n_readers)
        dtype = np.dtype(dtype or corr.map_dtype)
        self._gpu = self.backend.name == "cupy" and hasattr(
            getattr(self.backend.module, "cuda", None), "Stream"
        )
        self.stream = self.backend.create_stream(non_blocking=True) if self._gpu else None
        self.host = [
            {k: self.backend.alloc_pinned(s, dtype) for k, s in shapes.items()}
            for _ in range(n_slots)
        ]
        self.dev = [
            {k: self.backend.zeros(s, dtype=dtype) for k, s in shapes.items()}
            for _ in range(n_slots)
        ]
        self._released = [None] * n_slots  # compute-stream events
        self._uploaded = [None] * n_slots  # upload-stream events

    def __len__(self) -> int:
        return len(self.sources)

    def __iter__(self) -> Any:
        import threading

        n = len(self.sources)
        n_slots = len(self.host)
        cond = threading.Condition()
        free = list(range(n_slots))
        ready: Dict[int, Tuple[int, Any]] = {}
        # "taken": number of maps handed to the consumer so far
        state = {"error": None, "stop": False, "next_claim": 0, "taken": 0}
        upload_lock = threading.Lock()

        def reader() -> None:
            try:
                while True:
                    with cond:
                        # Claim sources in order.  At most n_slots - 1 maps
                        # are in flight (claimed, not yet handed over) while
                        # the consumer holds one slot itself, so the map it
                        # needs next can always get a slot: no deadlock,
                        # bounded memory.
                        while not state["stop"] and (
                            state["next_claim"] >= n
                            or state["next_claim"] - state["taken"] >= n_slots - 1
                            or not free
                        ):
                            if state["next_claim"] >= n:
                                return
                            cond.wait(timeout=0.05)
                        if state["stop"]:
                            return
                        index = state["next_claim"]
                        state["next_claim"] += 1
                        slot = free.pop()
                    if self._uploaded[slot] is not None:
                        # The previous asynchronous copy out of this pinned
                        # buffer may still be queued (the CPU runs ahead of
                        # the GPU): it must complete before the buffer is
                        # refilled, or that map would be uploaded corrupted.
                        self._uploaded[slot].synchronize()
                    self.read_fn(self.sources[index], self.host[slot])
                    event = None
                    if self._gpu:
                        with upload_lock:
                            with self.backend.module.cuda.Device(self.backend.device_id):
                                with self.stream:
                                    if self._released[slot] is not None:
                                        self.stream.wait_event(self._released[slot])
                                    for k, arr in self.host[slot].items():
                                        self.dev[slot][k].set(arr)
                                    event = self.stream.record()
                                    self._uploaded[slot] = event
                    else:
                        for k, arr in self.host[slot].items():
                            np.copyto(self.dev[slot][k], arr)
                    with cond:
                        ready[index] = (slot, event)
                        cond.notify_all()
            except BaseException as exc:  # surface reader failures
                with cond:
                    state["error"] = exc
                    cond.notify_all()

        threads = [
            threading.Thread(target=reader, daemon=True) for _ in range(self.n_readers)
        ]
        for t in threads:
            t.start()
        previous = None
        try:
            for index in range(n):
                with cond:
                    while index not in ready and state["error"] is None:
                        cond.wait(timeout=0.05)
                    if state["error"] is not None:
                        raise state["error"]
                    slot, event = ready.pop(index)
                    if previous is not None:
                        # The consumer has moved on: the kernels that use the
                        # previous slot are queued on the compute stream; its
                        # device buffers may be overwritten once they ran.
                        if self._gpu:
                            self._released[previous] = (
                                self.backend.module.cuda.get_current_stream().record()
                            )
                        free.append(previous)
                    state["taken"] = index + 1
                    cond.notify_all()
                if event is not None:
                    self.backend.module.cuda.get_current_stream().wait_event(event)
                previous = slot
                yield index, self.dev[slot]
        finally:
            with cond:
                state["stop"] = True
                cond.notify_all()
            for t in threads:
                t.join()
            if self._gpu:
                self.backend.module.cuda.get_current_stream().synchronize()
