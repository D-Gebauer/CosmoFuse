"""End-to-end parity of the GPU code paths against the CPU backend,
using numpy emulations of the CUDA kernels.

The emulators in ``cuda_emulation.py`` are compiled through the *real*
``_build_cupy_*`` wrapper builders (template dispatch, launch grids,
argument order) and installed on the backend, so the *real* GPU
orchestrator branches in ``correlations.py`` run end-to-end.  Their
results are compared against the independently implemented CPU Numba
kernels, which are validated against treecorr by the end-to-end tests.

This is the strongest GPU-correctness check available without a GPU: it
verifies kernel math transcription, output buffer layouts, launch grid
coverage, and the orchestrators' normalization of the new in-kernel
denominators.  Run ``benchmarks/bench_gpu_parity.py`` on a GPU machine for
the definitive check.
"""

import unittest
import types
from contextlib import contextmanager
from pathlib import Path

import healpy as hp
import numpy as np

from CosmoFuse.backend import (
    _build_cupy_3x2pt_tomo_aperture_kernel,
    _build_cupy_3x2pt_tomo_pairs_kernel,
    _build_cupy_aperture_tomo_density_kernel,
    _build_cupy_aperture_tomo_shear_kernel,
    _build_cupy_density_density_tomo_packed_kernel,
    _build_cupy_density_density_tomo_vectorized_kernel,
    _build_cupy_density_shear_tomo_packed_kernel,
    _build_cupy_density_shear_tomo_vectorized_kernel,
    _build_cupy_degrade_rows_kernel,
    _build_cupy_tomo_packed_kernel,
    _build_cupy_tomo_vectorized_kernel,
)
from CosmoFuse.correlations import Correlation

from .cuda_emulation import LAUNCH_LOG, EmulatedCupyModule

_BUILDERS = {
    "xipm_tomo_vectorized_kernel": _build_cupy_tomo_vectorized_kernel,
    "xipm_tomo_packed_kernel": _build_cupy_tomo_packed_kernel,
    "kernel_density_density_tomo_vectorized": _build_cupy_density_density_tomo_vectorized_kernel,
    "kernel_density_density_tomo_packed": _build_cupy_density_density_tomo_packed_kernel,
    "kernel_density_shear_tomo_vectorized": _build_cupy_density_shear_tomo_vectorized_kernel,
    "kernel_density_shear_tomo_packed": _build_cupy_density_shear_tomo_packed_kernel,
    "aperture_tomo_shear_kernel": _build_cupy_aperture_tomo_shear_kernel,
    "aperture_tomo_density_kernel": _build_cupy_aperture_tomo_density_kernel,
    "kernel_3x2pt_tomo_aperture": _build_cupy_3x2pt_tomo_aperture_kernel,
    "kernel_3x2pt_tomo_pairs": _build_cupy_3x2pt_tomo_pairs_kernel,
    "degrade_rows_kernel": _build_cupy_degrade_rows_kernel,
}
_EMULATED_KERNEL_ATTRS = tuple(_BUILDERS)


@contextmanager
def emulated_gpu(corr):
    """Temporarily turn *corr*'s CPU backend into an emulated-cupy backend.

    Yields a dict of per-kernel call counters so tests can assert the
    emulated kernels (not a fallback path) produced the results.
    """
    backend = corr.backend
    saved = {name: getattr(backend, name) for name in _EMULATED_KERNEL_ATTRS}
    saved["name"] = backend.name
    saved["to_device"] = backend.to_device
    saved["to_numpy"] = backend.to_numpy

    calls = {name: 0 for name in _EMULATED_KERNEL_ATTRS}

    def _counting(name, wrapped):
        if not isinstance(wrapped, types.FunctionType):
            # a bundle of entry points (degrade_rows_kernel): count each
            proxy = types.SimpleNamespace()
            for attr in dir(wrapped):
                if attr.startswith("_"):
                    continue
                value = getattr(wrapped, attr)
                setattr(
                    proxy,
                    attr,
                    _counting(f"{name}.{attr}", value) if callable(value) else value,
                )
            return proxy

        def _call(*args, **kwargs):
            calls[name] = calls.get(name, 0) + 1
            return wrapped(*args, **kwargs)

        return _call

    backend.name = "cupy"
    backend.to_device = lambda arr, stream=None: np.asarray(arr)
    backend.to_numpy = lambda arr, stream=None: np.asarray(arr)
    built = {}
    for name, builder in _BUILDERS.items():
        built[name] = builder(EmulatedCupyModule)
        setattr(backend, name, _counting(name, built[name]))
    calls["_built"] = built  # the raw wrapper callables (e.g. to toggle .tiled)
    try:
        yield calls
    finally:
        for name, value in saved.items():
            setattr(backend, name, value)


def _load_test_setup(rotation_precision):
    data_dir = Path(__file__).parent / "data"
    mask_path = data_dir / "hp_inds.npy"
    shear_path = data_dir / "shear_maps.npy"
    density_path = data_dir / "density_maps.npy"
    if not (mask_path.exists() and shear_path.exists() and density_path.exists()):
        raise unittest.SkipTest(f"Test data files not found in {data_dir}")

    nside = 256
    npix = hp.nside2npix(nside)
    map_inds = np.load(mask_path)
    mask = np.zeros(npix)
    mask[map_inds] = 1
    phi_center = np.array([0.44178647, 0.73631078, 0.85902924, 0.71176709])
    theta_center = np.array([1.54996149, 1.80201781, 1.9551931, 2.04691539])

    corr = Correlation(
        nside,
        phi_center,
        theta_center,
        nbins=5,
        theta_min=30,
        theta_max=120,
        patch_size=90,
        mask=mask,
        fastmath=False,
        device="cpu",
        map_precision="float64",
        rotation_precision=rotation_precision,
    )
    corr.preprocess()

    shear_maps = np.zeros((2, 2, npix))
    shear_maps[:, :, map_inds] = np.load(shear_path)
    density_maps = np.zeros((2, npix))
    density_maps[:, map_inds] = np.load(density_path)

    # Independent per-bin weights everywhere: cross-bin xi_g must agree
    # between the backends even when the directional weight sums differ
    # (both symmetrise as the ratio of summed orientations).
    rng = np.random.default_rng(2026)
    shear_w = 0.5 + rng.random((2, npix))
    density_w = 0.5 + rng.random((2, npix))

    return corr, {
        "shear": shear_maps,
        "density": density_maps,
        "w_shear": shear_w,
        "w_density": density_w,
    }


class TestEmulatedGpuParityFloat64Rotations(unittest.TestCase):
    """map float64 + rotation float64: results must match the CPU backend
    to reduction-order roundoff."""

    rtol = 1e-9
    atol = 1e-14

    @classmethod
    def setUpClass(cls):
        cls.corr, cls.maps = _load_test_setup("float64")
        corr, m = cls.corr, cls.maps
        cls.ref_Ma, cls.ref_xip, cls.ref_xim = corr.get_full_tomo_shear(
            m["shear"], m["w_shear"], return_device=False
        )
        cls.ref_Mg, cls.ref_xig = corr.get_full_tomo_density(
            m["density"], m["w_density"], return_device=False
        )
        cls.ref_xit = corr.get_full_tomo_ggl(
            m["density"], m["shear"], m["w_density"], m["w_shear"], return_device=False
        )
        cls.ref_fused = corr.get_3x2pt_tomo(
            shear_maps=m["shear"],
            density_maps=m["density"],
            weights={"shear": m["w_shear"], "density": m["w_density"]},
            return_device=False,
        )
        cls.ref_ap_shear = np.asarray(
            corr.get_aperture_shear(
                m["shear"][0, 0], m["shear"][0, 1], m["w_shear"][0], return_device=False
            )
        )
        cls.ref_ap_density = np.asarray(
            corr.get_aperture_density(m["density"][0], m["w_density"][0], return_device=False)
        )
        cls.ref_Ma_flipped, cls.ref_xip_flipped, _ = corr.get_full_tomo_shear(
            m["shear"], m["w_shear"], flip_g1=True, return_device=False
        )

    def _allclose(self, actual, ref):
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(ref), rtol=self.rtol, atol=self.atol
        )

    def test_full_tomo_shear(self):
        m = self.maps
        with emulated_gpu(self.corr) as calls:
            M_a, xip, xim = self.corr.get_full_tomo_shear(
                m["shear"], m["w_shear"], return_device=False
            )
        self.assertGreaterEqual(calls["aperture_tomo_shear_kernel"], 1)
        self.assertGreaterEqual(calls["xipm_tomo_vectorized_kernel"], 1)
        self._allclose(M_a, self.ref_Ma)
        self._allclose(xip, self.ref_xip)
        self._allclose(xim, self.ref_xim)

    def test_tiled_and_per_row_xipm_kernels_agree(self):
        """The combination-tiled kernel (default) and the original
        per-(bin, row) kernel share one launch contract and one result."""
        m = self.maps
        results = {}
        for tiled in (True, False):
            with emulated_gpu(self.corr) as calls:
                wrapper = calls["_built"]["xipm_tomo_vectorized_kernel"]
                self.assertTrue(wrapper.tiled)  # tiled is the default
                wrapper.tiled = tiled
                del LAUNCH_LOG[:]
                results[tiled] = self.corr.vectorized_shear_shear(
                    m["shear"], m["w_shear"], return_device=False
                )
                self.assertEqual(calls["xipm_tomo_vectorized_kernel"], 1)
                self.assertEqual(
                    LAUNCH_LOG,
                    ["gpu_tiled_tomo_reduce_xipm" if tiled else "gpu_fused_tomo_reduce_xipm"],
                )
        for a, b in zip(results[True], results[False]):
            self._allclose(a, b)
        self._allclose(results[True][0], self.ref_xip)
        self._allclose(results[True][1], self.ref_xim)

    def test_full_tomo_shear_flip_g1(self):
        m = self.maps
        with emulated_gpu(self.corr):
            M_a, xip, _xim = self.corr.get_full_tomo_shear(
                m["shear"], m["w_shear"], flip_g1=True, return_device=False
            )
        self._allclose(M_a, self.ref_Ma_flipped)
        self._allclose(xip, self.ref_xip_flipped)

    def test_full_tomo_density(self):
        m = self.maps
        with emulated_gpu(self.corr) as calls:
            M_g, xi_g = self.corr.get_full_tomo_density(
                m["density"], m["w_density"], return_device=False
            )
        self.assertGreaterEqual(calls["aperture_tomo_density_kernel"], 1)
        self.assertGreaterEqual(calls["kernel_density_density_tomo_vectorized"], 1)
        self._allclose(M_g, self.ref_Mg)
        self._allclose(xi_g, self.ref_xig)

    def test_full_tomo_ggl(self):
        m = self.maps
        with emulated_gpu(self.corr) as calls:
            xi_t = self.corr.get_full_tomo_ggl(
                m["density"], m["shear"], m["w_density"], m["w_shear"],
                return_device=False,
            )
        self.assertGreaterEqual(calls["kernel_density_shear_tomo_vectorized"], 1)
        self._allclose(xi_t, self.ref_xit)

    def test_fused_3x2pt(self):
        m = self.maps
        with emulated_gpu(self.corr) as calls:
            results = self.corr.get_3x2pt_tomo(
                shear_maps=m["shear"],
                density_maps=m["density"],
                weights={"shear": m["w_shear"], "density": m["w_density"]},
                return_device=False,
            )
        self.assertGreaterEqual(calls["kernel_3x2pt_tomo_aperture"], 1)
        # All three pair statistics in one walk over the pairs.
        self.assertEqual(calls["kernel_3x2pt_tomo_pairs"], 1)
        self.assertEqual(calls["xipm_tomo_vectorized_kernel"], 0)
        self.assertEqual(calls["kernel_density_density_tomo_vectorized"], 0)
        self.assertEqual(calls["kernel_density_shear_tomo_vectorized"], 0)
        for actual, ref, label in zip(
            results, self.ref_fused, ("M_a", "M_g", "xip", "xim", "xi_g", "xi_t")
        ):
            with self.subTest(output=label):
                self._allclose(actual, ref)

    def test_fused_3x2pt_single_pass_equals_three_tiles(self):
        """Single-pass tile == the three standalone tiles, also for auto-only
        / subset requests; beyond the register budget (or mode "off") the
        standalone tiles take over."""
        m = self.maps
        kwargs = dict(
            shear_maps=m["shear"], density_maps=m["density"],
            weights={"shear": m["w_shear"], "density": m["w_density"]}, return_device=False,
        )
        variants = (
            {},
            {"gc_auto_correlations_only": True},
            {"ggl_bin_combinations": [(1, 0), (0, 1)]},
        )
        for extra in variants:
            out = {}
            for mode in ("auto", "off", "budget"):
                with emulated_gpu(self.corr) as calls:
                    wrapper = calls["_built"]["kernel_3x2pt_tomo_pairs"]
                    self.assertEqual(wrapper.mode, "auto")
                    if mode == "off":
                        wrapper.mode = "off"
                    if mode == "budget":
                        wrapper.max_accumulators = 1
                    del LAUNCH_LOG[:]
                    out[mode] = self.corr.get_3x2pt_tomo(**kwargs, **extra)
                    single = "gpu_tiled_tomo_reduce_3x2pt" in LAUNCH_LOG
                    self.assertEqual(single, mode == "auto")
                    self.assertEqual(
                        calls["kernel_density_shear_tomo_vectorized"], 0 if single else 1)
            # (on a real GPU xi+- / xi_g are bit-identical and xi_t agrees to
            # one rounding of the map precision, see pair_tiles.cuh)
            for a, b, c in zip(out["auto"], out["off"], out["budget"]):
                np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-13, atol=0)
                np.testing.assert_array_equal(np.asarray(b), np.asarray(c))

    def test_tiled_and_per_row_dd_ds_kernels_agree(self):
        """The tiled xi_g / xi_t kernels and the per-row fallbacks give one
        result; auto-only and subset combination requests go through the
        tiles too (canonical rows gathered)."""
        m = self.maps
        for tiled in (True, False):
            with emulated_gpu(self.corr) as calls:
                calls["_built"]["kernel_density_density_tomo_vectorized"].tiled = tiled
                calls["_built"]["kernel_density_shear_tomo_vectorized"].tiled = tiled
                del LAUNCH_LOG[:]
                xi_g = self.corr.vectorized_density_density(
                    m["density"], m["w_density"], return_device=False
                )
                xi_g_auto = self.corr.vectorized_density_density(
                    m["density"], m["w_density"], gc_auto_correlations_only=True,
                    return_device=False,
                )
                xi_t = self.corr.vectorized_density_shear(
                    m["density"], m["shear"], m["w_density"], m["w_shear"],
                    return_device=False,
                )
                xi_t_sub = self.corr.vectorized_density_shear(
                    m["density"], m["shear"], m["w_density"], m["w_shear"],
                    ggl_bin_combinations=[(1, 0), (0, 0)], return_device=False,
                )
                expected = (
                    ["gpu_tiled_tomo_reduce_dd"] * 2 + ["gpu_tiled_tomo_reduce_ds"] * 2
                    if tiled
                    else ["gpu_fused_tomo_reduce_dd"] * 2 + ["gpu_fused_tomo_reduce_ds"] * 2
                )
                self.assertEqual(LAUNCH_LOG, expected)
            self._allclose(xi_g, self.ref_xig)
            self._allclose(xi_g_auto, self.ref_xig[[0, 2]])   # (0,0), (1,1)
            self._allclose(xi_t, self.ref_xit)
            self._allclose(xi_t_sub, self.ref_xit[[2, 0]])    # (1,0), (0,0)

    def test_single_map_aperture_routes_through_tomo_kernel(self):
        m = self.maps
        with emulated_gpu(self.corr) as calls:
            ap_shear = self.corr.get_aperture_shear(
                m["shear"][0, 0], m["shear"][0, 1], m["w_shear"][0], return_device=False
            )
            ap_density = self.corr.get_aperture_density(
                m["density"][0], m["w_density"][0], return_device=False
            )
        self.assertGreaterEqual(calls["aperture_tomo_shear_kernel"], 1)
        self.assertGreaterEqual(calls["aperture_tomo_density_kernel"], 1)
        self._allclose(ap_shear, self.ref_ap_shear)
        self._allclose(ap_density, self.ref_ap_density)

    def test_explicit_sumofweights_still_respected(self):
        """An explicit sumofweights bypasses the in-kernel denominators."""
        m = self.maps
        nzbin_combs = 3
        nbins_total = self.corr.n_patches * self.corr.nbins
        explicit = np.full((2, nzbin_combs, nbins_total), 2.0)
        ref_xip, ref_xim = self.corr.vectorized_shear_shear(
            m["shear"], m["w_shear"], sumofweights=explicit, return_device=False
        )
        with emulated_gpu(self.corr):
            xip, xim = self.corr.vectorized_shear_shear(
                m["shear"], m["w_shear"], sumofweights=explicit, return_device=False
            )
        self._allclose(xip, ref_xip)
        self._allclose(xim, ref_xim)


class TestEmulatedGpuParityFloat32Rotations(unittest.TestCase):
    """rotation float32: the standalone xi+/- kernel rotates in complex64
    (cuCmulf), so parity with the float64-rotation CPU math is limited by
    float32 roundoff."""

    rtol = 2e-4
    atol = 1e-9

    @classmethod
    def setUpClass(cls):
        cls.corr, cls.maps = _load_test_setup("float32")

    def _allclose(self, actual, ref):
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(ref), rtol=self.rtol, atol=self.atol
        )

    def test_full_tomo_shear(self):
        m = self.maps
        ref_Ma, ref_xip, ref_xim = self.corr.get_full_tomo_shear(
            m["shear"], m["w_shear"], return_device=False
        )
        with emulated_gpu(self.corr):
            M_a, xip, xim = self.corr.get_full_tomo_shear(
                m["shear"], m["w_shear"], return_device=False
            )
        self._allclose(M_a, ref_Ma)
        self._allclose(xip, ref_xip)
        self._allclose(xim, ref_xim)

    def test_fused_3x2pt(self):
        m = self.maps
        ref = self.corr.get_3x2pt_tomo(
            shear_maps=m["shear"],
            density_maps=m["density"],
            weights={"shear": m["w_shear"], "density": m["w_density"]},
            return_device=False,
        )
        with emulated_gpu(self.corr):
            results = self.corr.get_3x2pt_tomo(
                shear_maps=m["shear"],
                density_maps=m["density"],
                weights={"shear": m["w_shear"], "density": m["w_density"]},
                return_device=False,
            )
        for actual, expected, label in zip(
            results, ref, ("M_a", "M_g", "xip", "xim", "xi_g", "xi_t")
        ):
            with self.subTest(output=label):
                self._allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
