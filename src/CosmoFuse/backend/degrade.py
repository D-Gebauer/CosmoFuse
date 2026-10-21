"""Fused static-treecode row degrade (``cuda/degrade_rows.cu``)."""

from typing import Any, Tuple

from .raw import KernelFile, _has_raw_cuda_compiler, c_real

# Row buffers are addressed through (base, lead, comp, row) element strides.
_STRIDED = ("w_src", "v_src", "w_dst", "v_dst", "w_app", "v_app", "w_rows", "v_rows")


def _build_cupy_degrade_rows_kernel(module: Any) -> Any:
    """Builder for the fused treecode row-degrade kernels.

    Two entry points behind one object: ``level()`` accumulates one CSR
    level into the accumulation-dtype scratch, ``finalize()`` normalises
    and scatters a range of appended rows into the map-dtype row buffers.
    Both return ``False`` when no raw compiler is available, so the caller
    can fall back to the sparse path.

    Every buffer is passed as a descriptor
    ``(array, base, lead_stride, comp_stride, row_stride)`` of element
    offsets, so the kernels write the SoA, interleaved or AoS layout
    directly -- see the header of ``degrade_rows.cu``.  ``comp_stride`` is
    unused for the weights.
    """
    build = KernelFile(module, "degrade_rows.cu", "Treecode degrade", strided=_STRIDED)

    def _w_args(desc: Tuple[Any, int, int, int, int]) -> Tuple[Any, ...]:
        arr, base, lead, _comp, row = desc
        return (arr, base, lead, row)

    def _v_args(desc: Tuple[Any, int, int, int, int]) -> Tuple[Any, ...]:
        arr, base, lead, comp, row = desc
        return (arr, base, lead, comp, row)

    class _DegradeKernels:
        available = True

        @staticmethod
        def level(
            indptr: Any,
            indices: Any,
            w_src: Tuple[Any, int, int, int, int],
            v_src: Tuple[Any, int, int, int, int],
            w_dst: Tuple[Any, int, int, int, int],
            v_dst: Tuple[Any, int, int, int, int],
            n_cells: int,
            n_lead: int,
            n_val: int,
            weighted: bool,
        ) -> bool:
            if not _has_raw_cuda_compiler(module) or n_cells <= 0:
                return n_cells <= 0
            src_t = c_real(w_src[0].dtype)
            acc_t = c_real(w_dst[0].dtype)
            # Lanes per cell: the next power of two at or above the mean
            # number of children, capped at a warp.  A treecode level has
            # 4 children per cell, so a full warp would idle 87 % of its
            # lanes; a coarse aperture level has many more.
            nnz = int(indices.size)
            mean_children = max(1.0, nnz / max(1, int(n_cells)))
            seg = 1
            while seg < 32 and seg < mean_children:
                seg *= 2
            kernel = build(
                "gpu_degrade_level",
                (src_t, acc_t, int(n_val), "true" if weighted else "false", seg),
            )
            if kernel is None:
                return False
            # v_src/v_dst may be unused (n_val == 0) but must still be valid
            # pointers; the caller passes the weight arrays in that case.
            threads = 256
            groups = threads // seg
            total = int(n_cells) * int(n_lead)
            blocks = (total + groups - 1) // groups
            kernel(
                (max(1, blocks),),
                (threads,),
                (
                    indptr,
                    indices,
                    *_w_args(w_src),
                    *_v_args(v_src),
                    *_w_args(w_dst),
                    *_v_args(v_dst),
                    n_cells,
                    n_lead,
                ),
            )
            return True

        @staticmethod
        def finalize(
            w_app: Any,
            v_app: Any,
            n_appended: int,
            w_rows: Tuple[Any, int, int, int, int],
            v_rows: Tuple[Any, int, int, int, int],
            n_lead: int,
            lo: int,
            hi: int,
            n_val: int,
            row_inner: bool = True,
        ) -> bool:
            if not _has_raw_cuda_compiler(module):
                return False
            if hi <= lo:
                return True
            kernel = build(
                "gpu_degrade_finalize",
                (c_real(w_rows[0].dtype), c_real(w_app.dtype), int(n_val)),
            )
            if kernel is None:
                return False
            threads = 256
            total = int(hi - lo) * int(n_lead)
            blocks = min(65535, (total + threads - 1) // threads)
            # The scratch is always SoA with a unit row stride.
            kernel(
                (max(1, blocks),),
                (threads,),
                (
                    w_app,
                    n_appended,
                    v_app,
                    n_appended,
                    int(n_lead) * int(n_appended),
                    *_w_args(w_rows),
                    *_v_args(v_rows),
                    n_lead,
                    lo,
                    hi,
                    1 if row_inner else 0,
                ),
            )
            return True

    return _DegradeKernels()
