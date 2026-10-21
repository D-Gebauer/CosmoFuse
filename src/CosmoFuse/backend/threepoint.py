"""
Step 1 of the multipole three-point estimator: per-centre, per-bin sums.

For a centre ``c`` and a radial bin ``b`` the kernel sums, over the
neighbours ``j`` of ``c`` in ``b``, a table of ``O`` outputs described by
five small integer arrays (the *descriptor*)::

    out[c, b, o] = sum_j  A(j, fa[o])^{ca[o]} * B(j, o) * exp(-i e[o] phi_cj)

    B(j, o) = A(j, fb[o])^{cb[o]}   if fb[o] >= 0   else 1

where ``x^{1}`` means complex conjugation, ``phi_cj`` is the position angle
of ``j`` seen from ``c`` (east through north) and ``A(j, f)`` is the weighted
value of field ``f`` at ``j``, projected onto the line to the centre::

    spin 0:  A = w_f(j) * v_f(j)
    spin 2:  A = -w_f(j) * (v1_f(j) + i v2_f(j)) * exp(-2i psi_j)

(``psi_j`` = position angle of the centre seen from ``j``; the minus sign
makes the real part the tangential shear).  With ``fb = -1`` an output is a
multipole ``G_e(b; c)`` of a field; with ``fb >= 0`` it is a pair product
used to remove the degenerate ``j = k`` "triangles".

Both backends implement exactly this contract: a Numba kernel on the CPU and
``cuda/threepoint_multipoles.cu`` on the GPU (numpy twin in
``tests/cuda_emulation.py``).  A sixth descriptor array, ``order``, says which
output each CUDA thread slot computes (a permutation of ``range(O)``); it
changes nothing in the contract and the CPU kernel ignores it.
"""

import math
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from numba import njit, prange

from .raw import KernelFile, _has_raw_cuda_compiler, c_real

# Threads per block of the CUDA kernel (one block per centre).
_BLOCK = 128
# Static shared memory budget of one block (bytes); decides the neighbour
# chunk NB staged in shared memory.
_SHARED_BUDGET = 40 * 1024


def _cpu_3pcf_multipoles_impl(
    offsets: np.ndarray,
    c0: int,
    n_centres: int,
    nbins: int,
    nbr: np.ndarray,
    phi_c: np.ndarray,
    phi_n: np.ndarray,
    vre: np.ndarray,
    vim: np.ndarray,
    wts: np.ndarray,
    spin: np.ndarray,
    d_fa: np.ndarray,
    d_fb: np.ndarray,
    d_ca: np.ndarray,
    d_cb: np.ndarray,
    d_e: np.ndarray,
    E: int,
    out: np.ndarray,
) -> None:
    nf = spin.size
    no = d_fa.size
    npow = 2 * E + 1
    for c in prange(n_centres):
        a = np.empty(nf, dtype=out.dtype)
        pw = np.empty(npow, dtype=out.dtype)
        acc = np.empty(no, dtype=out.dtype)
        for b in range(nbins):
            lo = offsets[(c0 + c) * nbins + b]
            hi = offsets[(c0 + c) * nbins + b + 1]
            for o in range(no):
                acc[o] = 0.0
            for p in range(lo, hi):
                j = nbr[p]
                pn2 = 2.0 * phi_n[p]
                rot = complex(math.cos(pn2), -math.sin(pn2))
                for f in range(nf):
                    # promote before multiplying (as the CUDA kernel does):
                    # float32 maps must not be combined at float32
                    w = np.float64(wts[f, j])
                    if spin[f] == 2:
                        a[f] = (
                            -w
                            * complex(np.float64(vre[f, j]), np.float64(vim[f, j]))
                            * rot
                        )
                    else:
                        a[f] = w * np.float64(vre[f, j])
                pc = phi_c[p]
                base = complex(math.cos(pc), -math.sin(pc))
                pw[E] = 1.0
                cur = complex(1.0, 0.0)
                for e in range(1, E + 1):
                    cur = cur * base
                    pw[E + e] = cur
                    pw[E - e] = cur.conjugate()
                for o in range(no):
                    x = a[d_fa[o]]
                    if d_ca[o] != 0:
                        x = x.conjugate()
                    fb = d_fb[o]
                    if fb >= 0:
                        y = a[fb]
                        if d_cb[o] != 0:
                            y = y.conjugate()
                        x = x * y
                    acc[o] += x * pw[d_e[o] + E]
            for o in range(no):
                out[c, b, o] = acc[o]


_cpu_kernel_cache: Dict[bool, Any] = {}


def cpu_3pcf_multipoles_kernel(fastmath: bool = True) -> Callable[..., None]:
    """The Numba step-1 kernel (``out`` is filled in place)."""
    k = _cpu_kernel_cache.get(bool(fastmath))
    if k is None:
        k = njit(fastmath=bool(fastmath), parallel=True, cache=True)(
            _cpu_3pcf_multipoles_impl
        )
        _cpu_kernel_cache[bool(fastmath)] = k
    return k


def _cpu_3pcf_multipoles(
    offsets: np.ndarray,
    c0: int,
    n_centres: int,
    nbins: int,
    nbr: np.ndarray,
    phi_c: np.ndarray,
    phi_n: np.ndarray,
    vre: np.ndarray,
    vim: np.ndarray,
    wts: np.ndarray,
    spin: np.ndarray,
    desc: Tuple[np.ndarray, ...],
    E: int,
    out: np.ndarray,
) -> bool:
    cpu_3pcf_multipoles_kernel(True)(
        offsets,
        int(c0),
        int(n_centres),
        int(nbins),
        nbr,
        phi_c,
        phi_n,
        vre,
        vim,
        wts,
        spin,
        *desc[:5],  # desc[5], the CUDA thread map, does not apply
        int(E),
        out,
    )
    return True


def shared_chunk(n_fields: int, E: int, acc_itemsize: int) -> int:
    """Neighbours staged in shared memory per chunk (CUDA kernel)."""
    per = (2 * n_fields + 2 * (2 * E + 1)) * acc_itemsize
    return int(max(1, min(32, _SHARED_BUDGET // per)))


def build_cupy_3pcf_multipoles_kernel(module: Any) -> Optional[Callable[..., bool]]:
    """Launch wrapper of ``gpu_3pcf_multipoles`` (same contract as the CPU
    kernel).  ``None`` when the module cannot compile raw CUDA."""
    if not _has_raw_cuda_compiler(module):
        return None
    build = KernelFile(module, "threepoint_multipoles.cu", "3PCF multipoles")

    def _kernel(
        offsets: Any,
        c0: int,
        n_centres: int,
        nbins: int,
        nbr: Any,
        phi_c: Any,
        phi_n: Any,
        vre: Any,
        vim: Any,
        wts: Any,
        spin: Any,
        desc: Tuple[Any, ...],
        E: int,
        out: Any,
    ) -> bool:
        n_fields = int(spin.shape[0])
        n_out = int(desc[0].shape[0])
        acc_real = np.dtype(out.dtype).type(0).real.dtype
        nb = shared_chunk(n_fields, int(E), acc_real.itemsize)
        k_per = -(-n_out // _BLOCK)
        kernel = build(
            "gpu_3pcf_multipoles",
            (
                c_real(vre.dtype),
                c_real(phi_c.dtype),
                c_real(acc_real),
                n_fields,
                int(E),
                n_out,
                nb,
                k_per,
            ),
        )
        if kernel is None or int(n_centres) == 0:
            return kernel is not None
        kernel(
            (int(n_centres),),
            (_BLOCK,),
            (
                offsets,
                c0,
                nbins,
                nbr,
                phi_c,
                phi_n,
                vre,
                vim,
                wts,
                vre.shape[1],
                spin,
                *desc,
                out,
            ),
        )
        return True

    return _kernel
