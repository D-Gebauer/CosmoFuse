"""
Payload packing of the pair geometry (device side, opt-in).

A pair normally costs 24 bytes on the device: two int32 row indices and two
complex64 rotation factors e^{2i phi}.  Packed it costs 8 bytes:

* the rotation factors have unit modulus, so only their angle is kept, as a
  uint16 (resolution 2 pi / 65536 = 9.6e-5 rad, error <= 4.8e-5 rad): the
  nearest code to the angle of the stored value, computed at float64 for
  every rotation precision, so it does not depend on the machine;
* the row indices become uint16 indices *local to the row block of their
  (patch, resolution level)*: the map rows of every patch are gathered into
  contiguous blocks (once per map), which also makes the random gathers of a
  thread block cache-local.

By default pair files and host arrays stay exact and packing happens in
``prepare()``.  With ``pack_host_pairs=True`` the same packing is applied
already at pair-finding time, so the host arrays and the pair file hold
8 instead of 24 bytes per pair too (opt-in: the file is then no longer
exact).

The row blocks are found without sorting: the identifiers of a block are
marked in a table spanning their range, the marks are numbered in order
(= ``np.unique``'s sorted output), and each pair looks its local index up.
The result is bitwise that of ``np.unique(..., return_inverse=True)`` at
O(pairs + span) instead of O(pairs log pairs); :func:`block_local_indices`.
Every function here works on one patch and releases the GIL (Numba ``nogil``
or large numpy loops), so patches can be packed in parallel threads.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar

import numba
import numpy as np
from numba import njit

#: Device/host bytes per pair of the packed payload:
#: ``(npairs, 4) uint16`` = two local row indices + two angles.
PACKED_BYTES_PER_PAIR = 4 * np.dtype(np.uint16).itemsize
ANGLE_STEPS = 65536
ANGLE_UNIT = 2.0 * np.pi / ANGLE_STEPS  # 9.587379924285257e-05
MAX_LOCAL_ROWS = 65536
_ENCODE_CHUNK = 1 << 20  # pairs per angle-encoding pass


def encode_angles(exp2phi: np.ndarray) -> np.ndarray:
    """Unit-modulus complex rotation factors -> uint16 angles (round to nearest).

    Always at float64: the angle of a complex64 value is computed exactly
    enough there that the code is the nearest one, whereas a float32
    ``arctan2`` (whose last bit depends on numpy's SIMD path, i.e. on the CPU)
    picked the neighbouring code for ~0.07 % of real pairs (CosmoFuse <= 7.2.0).
    """
    angle = np.angle(np.asarray(exp2phi).astype(np.complex128, copy=False))
    steps = np.rint(angle / ANGLE_UNIT).astype(np.int64)
    return (steps % ANGLE_STEPS).astype(np.uint16)


def decode_angles(codes: np.ndarray, complex_dtype: np.dtype) -> np.ndarray:
    """uint16 angles -> rotation factors (same arithmetic as the CUDA kernel:
    alpha = code * 2 pi / 65536, evaluated at float64)."""
    alpha = np.asarray(codes, dtype=np.float64) * ANGLE_UNIT
    out = np.empty(alpha.shape, dtype=np.complex128)
    out.real = np.cos(alpha)
    out.imag = np.sin(alpha)
    return out.astype(complex_dtype, copy=False)


@njit(nogil=True, cache=True)
def _block_local_kernel(ids: np.ndarray, lo: int, hi: int, local: np.ndarray) -> Any:
    """Sorted unique ids of the pairs ``lo:hi`` of ``ids`` (2, npairs) and
    their local indices, written to ``local[lo:hi, :2]`` -- unless there are
    more than 65536 of them (the caller raises; ``local`` is then unchanged)."""
    if hi <= lo:
        return np.zeros(0, dtype=ids.dtype)
    mn = ids[0, lo]
    mx = mn
    for p in range(lo, hi):
        for s in range(2):
            v = ids[s, p]
            if v < mn:
                mn = v
            if v > mx:
                mx = v
    span = np.int64(mx) - np.int64(mn) + 1
    mark = np.zeros(span, dtype=np.int32)  # local index + 1
    for p in range(lo, hi):
        mark[np.int64(ids[0, p]) - mn] = 1
        mark[np.int64(ids[1, p]) - mn] = 1
    cap = 2 * (hi - lo)  # at most two new ids per pair
    uniq = np.empty(cap if cap < span else span, dtype=ids.dtype)
    k = 0
    for j in range(span):
        if mark[j] != 0:
            uniq[k] = mn + j
            k += 1
            mark[j] = k
    if k <= 65536:
        for p in range(lo, hi):
            local[p, 0] = mark[np.int64(ids[0, p]) - mn] - 1
            local[p, 1] = mark[np.int64(ids[1, p]) - mn] - 1
    return uniq[:k]


def _block_local_numpy(ids: np.ndarray, lo: int, hi: int, local: np.ndarray) -> Any:
    """Vectorised twin of :func:`_block_local_kernel` (no sort either); used
    when the JIT is disabled (the test suite) and as its reference."""
    block = ids[:, lo:hi]
    if block.size == 0:
        return np.zeros(0, dtype=ids.dtype)
    mn = block.min()
    shifted = block.astype(np.int64) - np.int64(mn)
    mark = np.zeros(int(shifted.max()) + 1, dtype=np.int64)
    mark[shifted] = 1
    uniq = (np.flatnonzero(mark) + np.int64(mn)).astype(ids.dtype)
    if uniq.size <= 65536:
        np.cumsum(mark, out=mark)
        local[lo:hi, :2] = (mark[shifted] - 1).T
    return uniq


def block_local_indices(
    ids: np.ndarray, lo: int, hi: int, local: np.ndarray
) -> np.ndarray:
    """``np.unique(ids[:, lo:hi], return_inverse=True)`` without a sort.

    Returns the sorted unique ids and writes the inverse (local indices) to
    ``local[lo:hi, :2]``; when there are more than ``MAX_LOCAL_ROWS`` of
    them the caller raises and ``local`` must not be used.
    """
    uniq: np.ndarray
    if getattr(numba.config, "DISABLE_JIT", False):
        uniq = _block_local_numpy(ids, lo, hi, local)
    else:
        uniq = _block_local_kernel(ids, int(lo), int(hi), local)
    return uniq


def pack_patch(
    rows: np.ndarray,
    exp2phi: np.ndarray,
    bin_counts: np.ndarray,
    groups: Sequence[Tuple[int, int, int]],
    out: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Pack the pairs of one patch.

    Args:
        rows: (2, npairs) row identifiers, sorted by angular bin.  Device
            row indices in :meth:`Integrated3PCF.prepare`, global ids (pixel /
            virtual-row) when packing the host arrays; the global ordering
            of the two agrees, so the packed payload is the same either way
            and only ``blocks`` changes meaning.
        exp2phi: (2, npairs) rotation factors.
        bin_counts: (nbins,) pairs per bin.
        groups: resolution levels as ``(nside, first_bin, stop_bin)``.
        out: optional (npairs, 4) uint16 array to write the payload to (a
            slice of the whole catalogue's payload, so nothing is copied).

    Returns:
        packed: (npairs, 4) uint16 ``[local_a, local_b, angle_a, angle_b]``.
        blocks: per group, the (sorted, unique) identifiers of its row block.
        block_of_bin: (nbins,) index into ``blocks`` for every bin.
    """
    npairs = int(rows.shape[1])
    if out is None:
        packed = np.empty((npairs, 4), dtype=np.uint16)
    else:
        if out.shape != (npairs, 4) or out.dtype != np.uint16:
            raise ValueError(f"out must be ({npairs}, 4) uint16; got {out.shape}")
        packed = out
    # in chunks: element-wise, so the same codes, but the float/int64
    # temporaries stay small while many patches are packed at once
    for c0 in range(0, npairs, _ENCODE_CHUNK):
        c1 = min(c0 + _ENCODE_CHUNK, npairs)
        packed[c0:c1, 2:] = encode_angles(exp2phi[:, c0:c1]).T
    edges = np.concatenate(([0], np.cumsum(np.asarray(bin_counts, dtype=np.int64))))
    blocks: List[np.ndarray] = []
    block_of_bin = np.zeros(len(bin_counts), dtype=np.int64)
    for g, (_nside, b0, b1) in enumerate(groups):
        lo, hi = int(edges[b0]), int(edges[b1])
        uniq = block_local_indices(rows, lo, hi, packed)
        if uniq.size > MAX_LOCAL_ROWS:
            raise ValueError(
                f"pair packing needs <= {MAX_LOCAL_ROWS} rows per (patch, "
                f"resolution level); found {uniq.size}. Use pack_pairs=False "
                "or smaller patches."
            )
        blocks.append(uniq)
        block_of_bin[b0:b1] = g
    return packed, blocks, block_of_bin


def pack_host_patch(
    ids: np.ndarray,
    exp2phi: np.ndarray,
    bin_counts: np.ndarray,
    groups: Sequence[Tuple[int, int, int]],
    index_dtype: np.dtype,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ``pack_host_pairs`` payload of one patch, on its *global* ids:
    ``(packed (npairs, 4) uint16, block ids (concatenated), block sizes)``."""
    packed, blocks, _ = pack_patch(ids, exp2phi, bin_counts, groups)
    block_ids = (
        np.concatenate(blocks).astype(index_dtype, copy=False)
        if blocks
        else np.zeros(0, dtype=index_dtype)
    )
    return packed, block_ids, np.array([b.size for b in blocks], dtype=np.int64)


_T = TypeVar("_T")


def map_patches(fn: Callable[[int], _T], n: int) -> List[_T]:
    """``[fn(i) for i in range(n)]`` in parallel threads.

    As many threads as Numba uses (``NUMBA_NUM_THREADS`` caps both), since
    the per-patch packing steps release the GIL.  Results keep patch order
    and the first exception is re-raised.
    """
    workers = min(max(1, int(numba.get_num_threads())), int(n))
    if workers <= 1:
        return [fn(i) for i in range(n)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, range(n)))


def block_edges(block_sizes: np.ndarray) -> np.ndarray:
    """Start/stop offsets of the per-group row blocks of one patch."""
    return np.concatenate(([0], np.cumsum(np.asarray(block_sizes, dtype=np.int64))))


def unpack_rows(
    packed: np.ndarray,
    blocks: Sequence[np.ndarray],
    bin_counts: np.ndarray,
    groups: Sequence[Tuple[int, int, int]],
    dtype: np.dtype,
) -> np.ndarray:
    """Inverse of the index half of :func:`pack_patch`.

    ``blocks[g]`` holds the identifiers of group ``g`` in whatever space the
    caller wants back (global ids or device rows).
    """
    npairs = int(packed.shape[0])
    out = np.empty((2, npairs), dtype=dtype)
    edges = np.concatenate(([0], np.cumsum(np.asarray(bin_counts, dtype=np.int64))))
    for g, (_nside, b0, b1) in enumerate(groups):
        lo, hi = int(edges[b0]), int(edges[b1])
        if hi > lo:
            out[:, lo:hi] = np.asarray(blocks[g])[packed[lo:hi, :2].T]
    return out
