"""
Payload packing of the pair geometry (device side, opt-in).

A pair normally costs 24 bytes on the device: two int32 row indices and two
complex64 rotation factors e^{2i phi}.  Packed it costs 8 bytes:

* the rotation factors have unit modulus, so only their angle is kept, as a
  uint16 (resolution 2 pi / 65536 = 9.6e-5 rad, error <= 4.8e-5 rad);
* the row indices become uint16 indices *local to the row block of their
  (patch, resolution level)*: the map rows of every patch are gathered into
  contiguous blocks (once per map), which also makes the random gathers of a
  thread block cache-local.

Pair files and host arrays stay exact; packing happens in ``prepare()``.
"""

from typing import List, Sequence, Tuple

import numpy as np

ANGLE_STEPS = 65536
ANGLE_UNIT = 2.0 * np.pi / ANGLE_STEPS  # 9.587379924285257e-05
MAX_LOCAL_ROWS = 65536


def encode_angles(exp2phi: np.ndarray) -> np.ndarray:
    """Unit-modulus complex rotation factors -> uint16 angles (round to nearest)."""
    angle = np.angle(np.asarray(exp2phi))
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


def pack_patch(
    rows: np.ndarray,
    exp2phi: np.ndarray,
    bin_counts: np.ndarray,
    groups: Sequence[Tuple[int, int, int]],
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Pack the pairs of one patch.

    Args:
        rows: (2, npairs) device row indices, sorted by angular bin.
        exp2phi: (2, npairs) rotation factors.
        bin_counts: (nbins,) pairs per bin.
        groups: resolution levels as ``(nside, first_bin, stop_bin)``.

    Returns:
        packed: (npairs, 4) uint16 ``[local_a, local_b, angle_a, angle_b]``.
        blocks: per group, the (sorted, unique) device rows of its row block.
        block_of_bin: (nbins,) index into ``blocks`` for every bin.
    """
    npairs = int(rows.shape[1])
    packed = np.empty((npairs, 4), dtype=np.uint16)
    packed[:, 2:] = encode_angles(exp2phi).T
    edges = np.concatenate(([0], np.cumsum(np.asarray(bin_counts, dtype=np.int64))))
    blocks: List[np.ndarray] = []
    block_of_bin = np.zeros(len(bin_counts), dtype=np.int64)
    for g, (_nside, b0, b1) in enumerate(groups):
        lo, hi = int(edges[b0]), int(edges[b1])
        uniq, inv = np.unique(rows[:, lo:hi], return_inverse=True)
        if uniq.size > MAX_LOCAL_ROWS:
            raise ValueError(
                f"pair packing needs <= {MAX_LOCAL_ROWS} rows per (patch, "
                f"resolution level); found {uniq.size}. Use pack_pairs=False "
                "or smaller patches."
            )
        packed[lo:hi, :2] = np.reshape(inv, (2, hi - lo)).T
        blocks.append(uniq)
        block_of_bin[b0:b1] = g
    return packed, blocks, block_of_bin
