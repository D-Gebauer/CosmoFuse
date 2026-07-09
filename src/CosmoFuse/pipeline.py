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

    def stage(self, host_arrays: Mapping[str, Any]) -> int:
        """Enqueue the async host→device copy of *host_arrays* into the
        back slot; returns the slot token to pass to :meth:`wait`."""
        back = 1 - self.slot
        for k, arr in host_arrays.items():
            np.copyto(self.host[back][k], arr)  # host -> pinned (CPU-side)
            if self.stream is not None:
                with self.stream:
                    self.dev[back][k].set(self.host[back][k])  # pinned -> device, async
            else:
                np.copyto(self.dev[back][k], self.host[back][k])
        self.event = self.stream.record() if self.stream is not None else None
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
