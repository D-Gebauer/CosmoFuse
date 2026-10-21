"""The typed launch layer (``CosmoFuse.backend.raw``).

Every CUDA launch is checked against the signature parsed from its ``.cu``
source, so a wrapper and its kernel cannot disagree on argument order,
count, width or dtype without an error -- on a GPU and under the numpy
emulation alike.
"""

import unittest

import numpy as np

from CosmoFuse.backend.raw import (
    _CUDA_DIR,
    KernelFile,
    TypedKernel,
    _prepare_cuda_source,
    c_complex,
    c_index,
    c_real,
    parse_kernel_signatures,
)

from .cuda_emulation import _KERNEL_EMULATORS, EmulatedCupyModule


def _all_signatures():
    out = {}
    for path in sorted(_CUDA_DIR.glob("*.cu")):
        for name, sig in parse_kernel_signatures(
            _prepare_cuda_source(path.name)
        ).items():
            out[name] = (path.name, sig)
    return out


_SOURCE = """
/* a (comment, with) parentheses */
template<typename T, int NZ, typename I>
__global__ void k(
    const T* __restrict__ x,   /* input (npix x NZ) */
    const I* idx,
    T* out,
    const long long n,
    const int flag)
{
}
"""


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, grid, block, args):
        self.calls.append((grid, block, args))


def _kernel(strided=()):
    sig = parse_kernel_signatures(_SOURCE)["k"]
    raw = _Recorder()
    return TypedKernel(raw, sig, ("float", "4", "int"), strided), raw


def _args(**over):
    args = dict(
        x=np.zeros((3, 4), np.float32),
        idx=np.zeros(3, np.int32),
        out=np.zeros(3, np.float32),
        n=3,
        flag=1,
    )
    args.update(over)
    return tuple(args.values())


class TestSignatures(unittest.TestCase):
    def test_every_kernel_has_a_numpy_twin(self):
        """CLAUDE.md: every CUDA kernel needs a twin in cuda_emulation.py."""
        self.assertEqual(set(_all_signatures()), set(_KERNEL_EMULATORS))

    def test_every_parameter_type_is_known(self):
        for name, (filename, sig) in _all_signatures().items():
            template_types = {n for kind, n in sig.template if kind == "typename"}
            for p in sig.params:
                with self.subTest(kernel=name, param=p.name):
                    self.assertTrue(
                        p.ctype in template_types
                        or p.ctype
                        in (
                            "float",
                            "double",
                            "int",
                            "long long",
                            "unsigned int",
                            "unsigned short",
                        ),
                        f"{filename}: {name}({p.ctype} {p.name})",
                    )

    def test_parse(self):
        sig = parse_kernel_signatures(_SOURCE)["k"]
        self.assertEqual(
            sig.template, (("typename", "T"), ("int", "NZ"), ("typename", "I"))
        )
        self.assertEqual(
            [(p.name, p.ctype, p.pointer) for p in sig.params],
            [
                ("x", "T", True),
                ("idx", "I", True),
                ("out", "T", True),
                ("n", "long long", False),
                ("flag", "int", False),
            ],
        )


class TestTypedKernel(unittest.TestCase):
    def test_scalars_are_cast_to_the_c_type(self):
        k, raw = _kernel()
        k((1,), (32,), _args())
        args = raw.calls[0][2]
        self.assertIs(type(args[3]), np.int64)
        self.assertIs(type(args[4]), np.int32)

    def test_argument_count(self):
        k, _ = _kernel()
        with self.assertRaisesRegex(TypeError, "takes 5 arguments, got 4"):
            k((1,), (32,), _args()[:4])

    def test_pointer_dtype(self):
        k, _ = _kernel()
        msg = r"x \(float\*\) must be an array of float32, got float64"
        with self.assertRaisesRegex(TypeError, msg):
            k((1,), (32,), _args(x=np.zeros(3, np.float64)))
        msg = r"idx \(int\*\) must be an array of int32, got int64"
        with self.assertRaisesRegex(TypeError, msg):
            k((1,), (32,), _args(idx=np.zeros(3, np.int64)))
        with self.assertRaisesRegex(TypeError, r"out \(float\*\) must be an array"):
            k((1,), (32,), _args(out=3))
        with self.assertRaisesRegex(TypeError, "must be an array"):
            k((1,), (32,), _args(out=np.float32(3)))

    def test_complex_array_for_a_real_pointer(self):
        """A complex array is its interleaved real pairs (3PCF outputs)."""
        k, _ = _kernel()
        k((1,), (32,), _args(out=np.zeros(3, np.complex64)))
        with self.assertRaises(TypeError):
            k((1,), (32,), _args(out=np.zeros(3, np.complex128)))

    def test_contiguity(self):
        k, _ = _kernel()
        view = np.zeros((4, 3), np.float32).T
        with self.assertRaisesRegex(ValueError, "x must be C-contiguous"):
            k((1,), (32,), _args(x=view))
        strided, raw = _kernel(strided=("x",))
        strided((1,), (32,), _args(x=view))
        self.assertIs(raw.calls[0][2][0], view)

    def test_scalar_checks(self):
        k, _ = _kernel()
        with self.assertRaises(OverflowError):
            k((1,), (32,), _args(flag=2**31))
        with self.assertRaises(TypeError):
            k((1,), (32,), _args(n=2.0))
        with self.assertRaises(TypeError):
            k((1,), (32,), _args(n=np.zeros(1, np.int64)))
        k((1,), (32,), _args(flag=True, n=np.int32(7)))

    def test_template_count(self):
        sig = parse_kernel_signatures(_SOURCE)["k"]
        with self.assertRaisesRegex(TypeError, "3 template parameters, got 2"):
            TypedKernel(_Recorder(), sig, ("float", "4"))

    def test_c_type_names(self):
        self.assertEqual(c_real(np.float32), "float")
        self.assertEqual(c_real(np.float64), "double")
        self.assertEqual(c_index(np.int32), "int")
        self.assertEqual(c_index(np.int64), "long long")
        self.assertEqual(c_complex(np.complex64), "cuFloatComplex")
        self.assertEqual(c_complex(np.complex128), "cuDoubleComplex")
        for f, dt in ((c_real, np.float16), (c_index, np.uint32), (c_complex, float)):
            with self.assertRaises(TypeError):
                f(dt)


class TestKernelFile(unittest.TestCase):
    def test_unknown_kernel_raises(self):
        kernels = KernelFile(EmulatedCupyModule, "tomo_vectorized_xipm.cu", "test")
        with self.assertRaises(KeyError):
            kernels("gpu_no_such_kernel", ("float",))

    def test_cached_and_typed(self):
        kernels = KernelFile(EmulatedCupyModule, "tomo_vectorized_xipm.cu", "test")
        a = kernels("gpu_tiled_packed_reduce_xipm", ("float", 2, "double"))
        b = kernels("gpu_tiled_packed_reduce_xipm", ("float", 2, "double"))
        self.assertIs(a, b)
        self.assertIsInstance(a, TypedKernel)


if __name__ == "__main__":
    unittest.main()
