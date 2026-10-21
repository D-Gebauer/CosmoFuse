"""Compiling and launching the CUDA kernels in ``cuda/``.

A launch of a CuPy ``RawKernel`` is untyped: the arguments are passed as raw
bytes in the order given, so a scalar of the wrong width or an array of the
wrong dtype is not an error but a kernel that reads garbage.  This module
closes that gap without a second copy of the signatures:
:func:`parse_kernel_signatures` reads every ``__global__`` signature straight
from the ``.cu`` source, and :class:`TypedKernel` checks each launch against
it -- argument count, the dtype of every pointer argument (template type
parameters resolved from the instantiation), C-contiguity, and the integer
range of every scalar, which it also casts to the exact C type.  The checks
run on every launch, on a GPU and under the numpy emulation of the tests
alike, so a wrapper that disagrees with its kernel fails in CI on a CPU-only
machine.

:class:`KernelFile` compiles and caches the instantiations of one source
file (with the negative-caching policy described there).
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Dict, NoReturn, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

logger = logging.getLogger("CosmoFuse.backend")

# --std is mandatory: NVRTC refuses to instantiate name_expressions
# (C++ templates) without an explicit standard.
_CUPY_FASTMATH_OPTIONS = ("--use_fast_math", "--std=c++14")
_CUDA_DIR = Path(__file__).resolve().parent.parent / "cuda"
_CUDA_SOURCE_CACHE: Dict[str, str] = {}
# Sentinel distinguishing "never attempted" from "compilation failed" in the
# kernel caches, so a failing NVRTC compilation is not retried (and
# re-logged) on every subsequent call.
_KERNEL_CACHE_MISS = object()


def _load_cuda_source_file(filename: str) -> str:
    cached = _CUDA_SOURCE_CACHE.get(filename)
    if cached is not None:
        return cached
    source = (_CUDA_DIR / filename).read_text(encoding="utf-8")
    _CUDA_SOURCE_CACHE[filename] = source
    return source


_COMMON_CUDA_SOURCE = _load_cuda_source_file("common.cuh")
_PAIR_TILES_CUDA_SOURCE = _load_cuda_source_file("pair_tiles.cuh")


def _prepare_cuda_source(filename: str) -> str:
    source = _load_cuda_source_file(filename)
    source = source.replace("__COMMON_CUDA_SOURCE__", _COMMON_CUDA_SOURCE)
    return source.replace("__PAIR_TILES_CUDA_SOURCE__", _PAIR_TILES_CUDA_SOURCE)


def _has_raw_cuda_compiler(module: Any) -> bool:
    return (
        getattr(module, "RawModule", None) is not None
        or getattr(module, "RawKernel", None) is not None
    )


def _compile_raw_cuda_kernel(module: Any, source: str, name_expression: str) -> Any:
    raw_module_ctor = getattr(module, "RawModule", None)
    if raw_module_ctor is not None:
        try:
            raw_module = raw_module_ctor(
                code=source,
                options=_CUPY_FASTMATH_OPTIONS,
                name_expressions=(name_expression,),
            )
        except TypeError:
            raw_module = raw_module_ctor(
                source,
                options=_CUPY_FASTMATH_OPTIONS,
                name_expressions=(name_expression,),
            )

        get_function = getattr(raw_module, "get_function", None)
        if get_function is not None:
            return get_function(name_expression)

    raw_kernel_ctor = getattr(module, "RawKernel", None)
    if raw_kernel_ctor is None:
        raise AttributeError("Backend module does not provide RawModule or RawKernel")

    return raw_kernel_ctor(source, name_expression, options=_CUPY_FASTMATH_OPTIONS)


# ── C type names ──────────────────────────────────────────────────────────

#: The C types that appear in kernel signatures and template arguments.
_C_TO_NUMPY: Dict[str, np.dtype[Any]] = {
    "float": np.dtype(np.float32),
    "double": np.dtype(np.float64),
    "int": np.dtype(np.int32),
    "long long": np.dtype(np.int64),
    "unsigned int": np.dtype(np.uint32),
    "unsigned short": np.dtype(np.uint16),
    "cuFloatComplex": np.dtype(np.complex64),
    "cuDoubleComplex": np.dtype(np.complex128),
}


def c_real(dtype: npt.DTypeLike) -> str:
    """``float`` / ``double`` for a float32 / float64 dtype."""
    dt = np.dtype(dtype)
    if dt == np.float32:
        return "float"
    if dt == np.float64:
        return "double"
    raise TypeError(f"no CUDA real type for dtype {dt}")


def c_index(dtype: npt.DTypeLike) -> str:
    """``int`` / ``long long`` for an int32 / int64 pair-index dtype."""
    dt = np.dtype(dtype)
    if dt == np.int32:
        return "int"
    if dt == np.int64:
        return "long long"
    raise TypeError(f"no CUDA index type for dtype {dt}")


def c_complex(dtype: npt.DTypeLike) -> str:
    """``cuFloatComplex`` / ``cuDoubleComplex`` for complex64 / complex128."""
    dt = np.dtype(dtype)
    if dt == np.complex64:
        return "cuFloatComplex"
    if dt == np.complex128:
        return "cuDoubleComplex"
    raise TypeError(f"no CUDA complex type for dtype {dt}")


# ── Signatures ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class KernelParam:
    """One parameter of a ``__global__`` function."""

    name: str
    ctype: str  # base type with qualifiers removed: "T", "long long", ...
    pointer: bool


@dataclass(frozen=True)
class KernelSignature:
    """Template parameters ``(kind, name)`` and parameters of one kernel."""

    name: str
    template: Tuple[Tuple[str, str], ...]
    params: Tuple[KernelParam, ...]


_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
_KERNEL = re.compile(
    r"(?:template\s*<(?P<template>[^<>]*)>\s*)?"
    r"(?:extern\s+\"C\"\s+)?__global__\s+void\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?"
    r"(?P<name>\w+)\s*\((?P<params>[^)]*)\)",
)
_QUALIFIERS = {"const", "__restrict__", "volatile"}


def _parse_param(text: str) -> KernelParam:
    pointer = "*" in text
    tokens = [t for t in text.replace("*", " ").split() if t not in _QUALIFIERS]
    if len(tokens) < 2:
        raise ValueError(f"cannot parse kernel parameter {text!r}")
    return KernelParam(tokens[-1], " ".join(tokens[:-1]), pointer)


def parse_kernel_signatures(source: str) -> Dict[str, KernelSignature]:
    """Every ``__global__`` function of a CUDA source, by name."""
    text = _COMMENT.sub(" ", source)
    out: Dict[str, KernelSignature] = {}
    for m in _KERNEL.finditer(text):
        template: Tuple[Tuple[str, str], ...] = ()
        if m.group("template") is not None:
            template = tuple(
                (" ".join(p.split()[:-1]), p.split()[-1])
                for p in m.group("template").split(",")
                if p.strip()
            )
        params = tuple(
            _parse_param(p) for p in m.group("params").split(",") if p.strip()
        )
        out[m.group("name")] = KernelSignature(m.group("name"), template, params)
    return out


# ── Typed launches ────────────────────────────────────────────────────────


class TypedKernel:
    """A compiled kernel whose launches are checked against its signature.

    Called like a ``RawKernel``, ``kernel(grid, block, args)``.  Pointer
    arguments must be arrays of the parameter's dtype (a ``float*`` /
    ``double*`` also takes a complex64 / complex128 array, whose memory is
    the interleaved real pairs) and C-contiguous unless the parameter is
    named in ``strided`` (the kernel takes its strides explicitly).  Scalar
    arguments are cast to the exact C type after a range check.  Shapes are
    not part of a C signature; the wrappers own them.
    """

    __slots__ = ("raw", "name", "_params", "_pointers", "_integers", "_reals")

    def __init__(
        self,
        raw: Any,
        signature: KernelSignature,
        template_args: Sequence[str],
        strided: Collection[str] = (),
    ) -> None:
        if len(template_args) != len(signature.template):
            raise TypeError(
                f"{signature.name} has {len(signature.template)} template "
                f"parameters, got {len(template_args)}: {tuple(template_args)}"
            )
        bound = {
            name: value
            for (kind, name), value in zip(signature.template, template_args)
            if kind in ("typename", "class")
        }
        # Everything a launch needs is resolved here, once per
        # instantiation, and grouped by kind so that a launch runs three tight
        # loops: the checks run on every launch of host-bound calls.
        params = []  # (name, C type, dtype) by position, for the messages
        pointers = []  # (position, dtype, accepted dtypes, needs contiguity)
        integers = []  # (position, cast, lo, hi)
        reals = []  # (position, cast)
        for i, p in enumerate(signature.params):
            ctype = bound.get(p.ctype, p.ctype)
            if ctype not in _C_TO_NUMPY:
                raise TypeError(
                    f"{signature.name}: parameter {p.name} has unsupported type "
                    f"{ctype!r}"
                )
            dtype = _C_TO_NUMPY[ctype]
            params.append((p.name, ctype, dtype))
            if p.pointer:
                accepted = {dtype}
                if dtype.kind == "f":
                    accepted.add(np.result_type(dtype, np.complex64))
                pointers.append((i, dtype, frozenset(accepted), p.name not in strided))
            elif dtype.kind in "iu":
                info = np.iinfo(dtype)
                integers.append((i, dtype.type, int(info.min), int(info.max)))
            else:
                reals.append((i, dtype.type))
        self.raw = raw
        self.name = signature.name
        self._params = tuple(params)
        self._pointers = tuple(pointers)
        self._integers = tuple(integers)
        self._reals = tuple(reals)

    def __call__(
        self, grid: Tuple[int, ...], block: Tuple[int, ...], args: Sequence[Any]
    ) -> None:
        self.raw(grid, block, self.check(args))

    def check(self, args: Sequence[Any]) -> Tuple[Any, ...]:
        """The launch arguments, validated and with scalars cast.

        A few tenths of a µs per argument on a GPU host, about what the
        ``np.int64(...)`` casts the wrappers used to write by hand cost.
        """
        if len(args) != len(self._params):
            raise TypeError(
                f"{self.name} takes {len(self._params)} arguments, got {len(args)}"
            )
        out = list(args)
        for i, dtype, accepted, contiguous in self._pointers:
            a = args[i]
            try:
                adt, ndim = a.dtype, a.ndim
            except AttributeError:
                ndim = 0
            if ndim == 0:
                self._fail(i, f"an array, got {type(a).__name__}")
            # builtin dtypes are singletons: identity is the fast path
            if adt is not dtype and adt not in accepted:
                self._fail(i, f"an array of {dtype}, got {adt}")
            if contiguous:
                # cupy caches the flag; ``flags`` builds an object per call
                flag = getattr(a, "_c_contiguous", None)
                if not (a.flags.c_contiguous if flag is None else flag):
                    name = self._params[i][0]
                    raise ValueError(f"{self.name}: {name} must be C-contiguous")
        for i, cast, lo, hi in self._integers:
            a = args[i]
            if type(a) is not int:
                a = self._integer(i, a)
            if not lo <= a <= hi:
                name, ctype, _ = self._params[i]
                raise OverflowError(f"{self.name}: {name} = {a} does not fit a {ctype}")
            out[i] = cast(a)
        for i, cast in self._reals:
            if getattr(args[i], "ndim", 0) != 0:
                self._fail(i, "a scalar")
            out[i] = cast(args[i])
        return tuple(out)

    def _fail(self, i: int, what: str) -> NoReturn:
        name, ctype, _ = self._params[i]
        kind = f"{ctype}*" if any(p[0] == i for p in self._pointers) else ctype
        raise TypeError(f"{self.name}: {name} ({kind}) must be {what}")

    def _integer(self, i: int, a: Any) -> int:
        """An integer argument that is not a plain ``int`` (numpy integer,
        bool, 0-d array); floats and arrays are rejected."""
        if getattr(a, "ndim", 0) != 0:
            self._fail(i, "a scalar")
        if not isinstance(a, (bool, np.bool_, np.integer)) and not (
            hasattr(a, "dtype") and np.dtype(a.dtype).kind in "iub"
        ):
            self._fail(i, f"an integer, got {type(a).__name__}")
        return int(a)


#: Exception names that mean "this source will never compile for these
#: template arguments".  Anything else -- an IO error reading the NVRTC disk
#: cache, a transient driver or allocation failure -- may succeed next time.
_DETERMINISTIC_COMPILE_FAILURES = frozenset(
    {"CompileException", "NVRTCError", "JitifyException", "NVRTCException"}
)


def _is_deterministic_compile_failure(exc: BaseException) -> bool:
    return type(exc).__name__ in _DETERMINISTIC_COMPILE_FAILURES


class KernelFile:
    """The compiled instantiations of the kernels in one ``cuda/`` file.

    ``kernels(name, template_args)`` returns a :class:`TypedKernel`, or None
    when it cannot be compiled.  A *deterministic* compile failure is cached
    negatively (None) so it is not retried and re-logged on every call.  A
    transient one is **not**: caching it would silently pin the process to
    the slower fallback kernel for its whole lifetime, with no error and no
    way to tell from a configuration that legitimately has no tiled kernel.
    The retry is logged once per key so a genuinely broken build still says
    so, without a line per call.

    A request that does not match the source -- an unknown kernel, the
    wrong number of template arguments -- is a programming error and raises.
    """

    def __init__(
        self,
        module: Any,
        filename: str,
        what: str,
        strided: Collection[str] = (),
    ) -> None:
        self.module = module
        self.filename = filename
        self.what = what
        self.strided = frozenset(strided)
        self.signatures = parse_kernel_signatures(_prepare_cuda_source(filename))
        self._cache: Dict[Tuple[str, Tuple[str, ...]], Optional[TypedKernel]] = {}
        self._warned: set[Tuple[str, Tuple[str, ...]]] = set()

    def __call__(
        self, kernel_name: str, template_args: Sequence[Any]
    ) -> Optional[TypedKernel]:
        targs = tuple(str(a) for a in template_args)
        key = (kernel_name, targs)
        if key in self._cache:
            return self._cache[key]
        signature = self.signatures.get(kernel_name)
        if signature is None:
            raise KeyError(f"{self.filename} has no kernel {kernel_name!r}")
        name_expression = f"{kernel_name}<{', '.join(targs)}>"
        try:
            raw = _compile_raw_cuda_kernel(
                self.module, _prepare_cuda_source(self.filename), name_expression
            )
        except Exception as exc:
            if key not in self._warned:
                self._warned.add(key)
                logger.warning(
                    "%s RawKernel compilation failed (%s): %s",
                    self.what,
                    name_expression,
                    exc,
                )
            if not _is_deterministic_compile_failure(exc):
                # Retry next call rather than downgrading this process to
                # the fallback kernel for good.
                return None
            self._cache[key] = None
            return None
        kernel = TypedKernel(raw, signature, targs, self.strided)
        self._cache[key] = kernel
        return kernel
