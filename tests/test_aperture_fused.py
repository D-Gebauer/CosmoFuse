"""The fused aperture kernels must be *bitwise* what they replace.

``gpu_aperture_{shear,density}_tomo_fused`` and
``gpu_3x2pt_tomo_aperture_fused`` run one block per patch with the
tomographic bins looped inside it, so the aperture disc geometry is read
once per pixel instead of once per bin (1.75x at nside 512).  They are on
the ``resolution_factor=None`` path, which must stay bit-for-bit identical
to 4.20.0 -- so the restructure is only legal because it moves the bin
loop and nothing else: the thread->pixel mapping, ``BLOCK_SIZE`` and the
reduction tree are untouched, and every bin therefore sums exactly the
same partials in exactly the same order.

Nothing here uses a tolerance.  ``np.array_equal`` is the point: a
restructure that merely agrees to 1e-15 has left the regime and must not
ship.  The CPU class runs the *real* cupy wrappers against the numpy twins
(launch grids, template dispatch, argument order, the element-stride
contract); the GPU class compiles the real kernels with the library's own
NVRTC options -- ``--use_fast_math`` included, since that is where a
contraction difference between the two forms would show up.
"""

import unittest
from unittest.mock import patch

import numpy as np
import pytest

from CosmoFuse.backend import (
    _MAX_FUSED_APERTURE_BINS,
    _build_cupy_3x2pt_tomo_aperture_kernel,
    _build_cupy_aperture_tomo_density_kernel,
    _build_cupy_aperture_tomo_shear_kernel,
    _use_fused_aperture,
)

from .cuda_emulation import LAUNCH_LOG, EmulatedCupyModule

# Enough patches that the fused grid still fills a large device, so the
# npatches guard is not what is under test here.
NPATCHES = 512
VISITS = 300          # disc pixels per patch: several BLOCK_SIZE strides
NROWS = 4096


def aperture_geometry(rng, q_dtype, npatches=NPATCHES, visits=VISITS, n_rows=NROWS):
    """A synthetic aperture-disc CSR: q_inds + the three filter arrays."""
    total = npatches * visits
    return (
        rng.integers(0, n_rows, size=total, dtype=np.uint32),
        rng.uniform(-1.0, 1.0, size=total).astype(q_dtype),      # q_cos
        rng.uniform(-1.0, 1.0, size=total).astype(q_dtype),      # q_sin
        rng.uniform(0.0, 3.0, size=total).astype(q_dtype),       # q_val
        (np.arange(npatches + 1, dtype=np.int64) * visits),      # q_offsets
        rng.uniform(0.5, 1.5, size=npatches).astype(q_dtype),    # q_patch_area
    )


def planar_maps(rng, nz, dtype, n_rows=NROWS):
    """Shear as an (nz, 2, n_rows) stack -- g1/g2 are strided views of it,
    which is what the row expansion hands the wrapper in production."""
    shear = (rng.normal(size=(nz, 2, n_rows)) * 0.3).astype(dtype)
    weights = rng.uniform(0.2, 2.0, size=(nz, n_rows)).astype(dtype)
    return shear, weights


def aos_maps(rng, n_density, n_shear, dtype, n_rows=NROWS):
    """The AoS buffers the fused 3x2pt path loads."""
    return (
        (rng.normal(size=(n_rows, n_density)) * 0.5).astype(dtype),
        (rng.normal(size=(n_rows, n_shear, 2)) * 0.3).astype(dtype),
        rng.uniform(0.2, 2.0, size=(n_rows, n_density)).astype(dtype),
        rng.uniform(0.2, 2.0, size=(n_rows, n_shear)).astype(dtype),
    )


class TestFusedApertureSelection(unittest.TestCase):
    """Which kernel the wrapper picks, and why."""

    def test_fused_is_the_default(self):
        self.assertTrue(_use_fused_aperture(EmulatedCupyModule, 4, NPATCHES))

    def test_wide_bin_sets_fall_back(self):
        """2*NZ accumulators per thread: past the budget they spill."""
        self.assertFalse(
            _use_fused_aperture(
                EmulatedCupyModule, _MAX_FUSED_APERTURE_BINS + 1, NPATCHES
            )
        )
        self.assertTrue(
            _use_fused_aperture(
                EmulatedCupyModule, _MAX_FUSED_APERTURE_BINS, NPATCHES
            )
        )

    def test_small_patch_sets_fall_back(self):
        """The fused grid is ntomo times smaller -- below 2 blocks/SM the
        per-(patch, bin) grid is the one that fills the device."""

        class FakeRuntime:
            @staticmethod
            def getDevice():
                return 0

            @staticmethod
            def getDeviceProperties(_device):
                return {"multiProcessorCount": 108}

        class FakeCuda:
            runtime = FakeRuntime

        class FakeModule:
            cuda = FakeCuda

        self.assertFalse(_use_fused_aperture(FakeModule, 4, 215))
        self.assertTrue(_use_fused_aperture(FakeModule, 4, 216))

    def test_missing_runtime_imposes_no_constraint(self):
        """A module without cuda.runtime (the emulated cupy) must not
        silently disable the fused path."""
        self.assertTrue(_use_fused_aperture(EmulatedCupyModule, 4, 1))


class FusedApertureAgreementBase:
    """Both kernels, same inputs, compared bitwise.

    Subclasses provide ``xp`` (numpy-like), ``module`` (the cupy stand-in
    passed to the builders) and ``to_host``.
    """

    NZ = (1, 4, 8)
    DTYPES = ((np.float32, np.float32), (np.float64, np.float32))

    def run_shear(self, kernel, shear, weights, geom, nz, dtype, fused):
        xp = self.xp
        out = (xp.zeros((nz, NPATCHES), dtype=dtype),
               xp.zeros((nz, NPATCHES), dtype=dtype))
        limit = _MAX_FUSED_APERTURE_BINS if fused else 0
        with patch("CosmoFuse.backend._MAX_FUSED_APERTURE_BINS", limit):
            ok = kernel(shear[:, 0], shear[:, 1], weights, *geom, *out)
        self.assertTrue(ok)
        return tuple(self.to_host(o) for o in out)

    def test_shear_fused_matches_fallback(self):
        kernel = _build_cupy_aperture_tomo_shear_kernel(self.module)
        for dtype, q_dtype in self.DTYPES:
            for nz in self.NZ:
                with self.subTest(dtype=np.dtype(dtype).name, nz=nz):
                    rng = np.random.default_rng(11 + nz)
                    shear, weights = planar_maps(rng, nz, dtype)
                    geom = aperture_geometry(rng, q_dtype)
                    shear, weights = self.to_device(shear), self.to_device(weights)
                    geom = tuple(self.to_device(g) for g in geom)
                    got = self.run_shear(kernel, shear, weights, geom, nz,
                                         dtype, fused=True)
                    ref = self.run_shear(kernel, shear, weights, geom, nz,
                                         dtype, fused=False)
                    self.assertTrue(np.array_equal(got[0], ref[0]))
                    self.assertTrue(np.array_equal(got[1], ref[1]))
                    # a non-trivial comparison, not two buffers of zeros
                    self.assertTrue(np.any(np.asarray(ref[0]) != 0))

    def run_density(self, kernel, values, weights, geom, nz, dtype, fused):
        xp = self.xp
        out = (xp.zeros((nz, NPATCHES), dtype=dtype),
               xp.zeros((nz, NPATCHES), dtype=dtype))
        q_inds, _q_cos, _q_sin, q_val, q_offsets, q_area = geom
        limit = _MAX_FUSED_APERTURE_BINS if fused else 0
        with patch("CosmoFuse.backend._MAX_FUSED_APERTURE_BINS", limit):
            ok = kernel(values, weights, q_inds, q_val, q_offsets, q_area, *out)
        self.assertTrue(ok)
        return tuple(self.to_host(o) for o in out)

    def test_density_fused_matches_fallback(self):
        kernel = _build_cupy_aperture_tomo_density_kernel(self.module)
        for dtype, q_dtype in self.DTYPES:
            for nz in self.NZ:
                with self.subTest(dtype=np.dtype(dtype).name, nz=nz):
                    rng = np.random.default_rng(23 + nz)
                    values = (rng.normal(size=(nz, NROWS)) * 0.5).astype(dtype)
                    weights = rng.uniform(0.2, 2.0, size=(nz, NROWS)).astype(dtype)
                    geom = aperture_geometry(rng, q_dtype)
                    values, weights = self.to_device(values), self.to_device(weights)
                    geom = tuple(self.to_device(g) for g in geom)
                    got = self.run_density(kernel, values, weights, geom, nz,
                                           dtype, fused=True)
                    ref = self.run_density(kernel, values, weights, geom, nz,
                                           dtype, fused=False)
                    self.assertTrue(np.array_equal(got[0], ref[0]))
                    self.assertTrue(np.array_equal(got[1], ref[1]))
                    self.assertTrue(np.any(np.asarray(ref[0]) != 0))

    def run_3x2pt(self, kernel, maps, geom, n_density, n_shear, acc, fused):
        xp = self.xp
        out = (xp.zeros((n_shear, NPATCHES), dtype=acc),
               xp.zeros((n_shear, NPATCHES), dtype=acc),
               xp.zeros((n_density, NPATCHES), dtype=acc),
               xp.zeros((n_density, NPATCHES), dtype=acc))
        limit = _MAX_FUSED_APERTURE_BINS if fused else 0
        with patch("CosmoFuse.backend._MAX_FUSED_APERTURE_BINS", limit):
            ok = kernel(*maps, *geom, *out)
        self.assertTrue(ok)
        self.sync()
        return tuple(self.to_host(o) for o in out)

    def test_3x2pt_aperture_fused_matches_fallback(self):
        kernel = _build_cupy_3x2pt_tomo_aperture_kernel(self.module)
        for dtype, q_dtype in self.DTYPES:
            # float32 maps accumulate at float64 in this path
            acc = np.float64
            for n_density, n_shear in ((1, 1), (2, 4), (5, 3)):
                with self.subTest(dtype=np.dtype(dtype).name,
                                  nd=n_density, ns=n_shear):
                    rng = np.random.default_rng(37 + n_shear)
                    maps = tuple(self.to_device(m) for m in
                                 aos_maps(rng, n_density, n_shear, dtype))
                    geom = tuple(self.to_device(g)
                                 for g in aperture_geometry(rng, q_dtype))
                    got = self.run_3x2pt(kernel, maps, geom, n_density,
                                         n_shear, acc, fused=True)
                    ref = self.run_3x2pt(kernel, maps, geom, n_density,
                                         n_shear, acc, fused=False)
                    for a, b in zip(got, ref):
                        self.assertTrue(np.array_equal(a, b))
                    self.assertTrue(np.any(np.asarray(ref[0]) != 0))


class TestFusedApertureEmulated(FusedApertureAgreementBase, unittest.TestCase):
    """Through the real wrappers, against the numpy twins.

    This pins the launch contract -- grid, template arguments, argument
    order, the element strides the twins re-derive from the views -- on
    every machine.  The arithmetic gate is the GPU class below.
    """

    xp = np
    module = EmulatedCupyModule

    @staticmethod
    def to_device(arr):
        return arr

    @staticmethod
    def to_host(arr):
        return arr

    @staticmethod
    def sync():
        return None

    def test_launch_log_names_the_fused_kernels(self):
        """The default must really be the fused kernel: if the selection
        quietly reverted, every agreement test above would still pass."""
        rng = np.random.default_rng(5)
        shear, weights = planar_maps(rng, 4, np.float32)
        geom = aperture_geometry(rng, np.float32)
        kernel = _build_cupy_aperture_tomo_shear_kernel(self.module)
        LAUNCH_LOG.clear()
        self.run_shear(kernel, shear, weights, geom, 4, np.float32, fused=True)
        self.assertEqual(LAUNCH_LOG, ["gpu_aperture_shear_tomo_fused"])
        LAUNCH_LOG.clear()
        self.run_shear(kernel, shear, weights, geom, 4, np.float32, fused=False)
        self.assertEqual(LAUNCH_LOG, ["gpu_aperture_shear_tomo"])
        LAUNCH_LOG.clear()


@pytest.mark.gpu
class TestFusedApertureOnDevice(FusedApertureAgreementBase, unittest.TestCase):
    """The arithmetic gate: the real kernels, the library's own NVRTC
    options (``--use_fast_math``), on a real device."""

    @classmethod
    def setUpClass(cls):
        cupy = pytest.importorskip("cupy")
        if cupy.cuda.runtime.getDeviceCount() < 1:  # pragma: no cover
            raise unittest.SkipTest("no CUDA device")
        cls.xp = cupy
        cls.module = cupy

    @classmethod
    def to_device(cls, arr):
        return cls.xp.asarray(arr)

    @classmethod
    def to_host(cls, arr):
        return cls.xp.asnumpy(arr)

    @classmethod
    def sync(cls):
        cls.xp.cuda.runtime.deviceSynchronize()
