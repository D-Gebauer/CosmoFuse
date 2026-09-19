"""Fused treecode row degrade (idea #3): gpu_degrade_level / _finalize.

The sparse chain (cupyx.scipy.sparse) stays as the CPU path and the
fallback, so it is the reference: the fused kernels must reproduce it for
every public method, every block selection and every precision pairing.

The kernels are exercised through the *real* cupy wrappers against the
numpy twins in tests/cuda_emulation.py, so the launch geometry, the flat
index arithmetic and the template dispatch are all under test on a
CPU-only machine.  `_use_fused_degrade` is patched on, because under
emulation the arrays are numpy and the production gate deliberately keeps
host arrays on the host.
"""

import unittest
from unittest.mock import patch

import healpy as hp
import numpy as np
import pytest

from CosmoFuse import Correlation

from .cuda_emulation import LAUNCH_LOG
from .test_gpu_kernel_emulation import emulated_gpu

NSIDE = 64
NPIX = hp.nside2npix(NSIDE)


#: ``make_corr`` is ~13.5 s of pair finding and ``setUpClass`` runs once per
#: *subclass* -- six of them across this file and test_row_layouts.py, for
#: ~68 s of a 775 s suite spent rebuilding the same geometry.  The geometry
#: is a pure function of the arguments, so it is built once per distinct
#: configuration and the per-test mutable state is reset instead.
_CORR_CACHE = {}


def reset_corr(corr):
    """Drop the device-side state the degrade tests mutate.

    ``fused()`` already clears these after each run; doing it again at
    ``setUpClass`` is what makes sharing one instance between subclasses
    equivalent to building a fresh one.
    """
    corr.compute_context.Q_inds_dev = None
    corr.compute_context.degrade_csr = None
    corr.compute_context.pair_scratch = None
    corr.compute_context.fused_output_buffers = None
    memo = getattr(corr.compute_context, "frozen_map_memo", None)
    if memo is not None:
        memo.clear()
    corr._expansion_memo = None
    corr._expansion_layout = None
    corr._expansion_depth = 0
    return corr


def make_corr(k=2.0, aperture_nside=None, map_precision="float64",
              accumulation_precision="same", n_side_centers=4):
    key = (k, aperture_nside, map_precision, accumulation_precision,
           n_side_centers)
    cached = _CORR_CACHE.get(key)
    if cached is not None:
        return reset_corr(cached)
    _CORR_CACHE[key] = corr = _build_corr(*key)
    return corr


def _build_corr(k, aperture_nside, map_precision, accumulation_precision,
                n_side_centers):
    theta, phi = hp.pix2ang(NSIDE, np.arange(NPIX))
    mask = (np.degrees(phi) < 200.0) & (np.abs(90 - np.degrees(theta)) < 55)
    rng = np.random.default_rng(1)
    mask[rng.choice(NPIX, NPIX // 30, replace=False)] = False
    corr = Correlation.from_mask(
        NSIDE, mask, n_side_centers, patch_size=600, theta_Q=200, f_mask=0.3,
        nbins=4, theta_min=120, theta_max=1100, device="cpu",
        map_precision=map_precision, rotation_precision="float64",
        accumulation_precision=accumulation_precision,
        resolution_factor=k, aperture_nside=aperture_nside,
    )
    corr.preprocess()
    return corr


class FusedDegradeBase(unittest.TestCase):
    """Runs a callable twice: sparse chain vs fused kernels."""

    @classmethod
    def setUpClass(cls):
        cls.corr = make_corr()
        rng = np.random.default_rng(5)
        n = cls.corr.n_active
        cls.shear = rng.normal(size=(2, 2, n)) * 0.3
        cls.dens = rng.normal(size=(2, n))
        cls.w = rng.uniform(0.2, 2.0, size=(2, n))

    def fused(self, corr, fn):
        """Run `fn(corr)` with the fused kernels active."""
        with emulated_gpu(corr) as calls:
            corr.compute_context.Q_inds_dev = None
            with patch.object(Correlation, "_use_fused_degrade",
                              lambda self, w: True):
                del LAUNCH_LOG[:]
                out = fn(corr)
        corr.compute_context.Q_inds_dev = None
        corr.compute_context.degrade_csr = None
        return out, calls

    def sparse(self, corr, fn):
        corr.compute_context.degrade_ops = None
        return fn(corr)

    def assert_close(self, got, want, name, atol_scale=1e-13):
        got = [np.asarray(a) for a in (got if isinstance(got, tuple) else (got,))]
        want = [np.asarray(a) for a in (want if isinstance(want, tuple) else (want,))]
        self.assertEqual(len(got), len(want), name)
        for i, (a, b) in enumerate(zip(got, want)):
            self.assertEqual(a.shape, b.shape, f"{name}[{i}]")
            scale = np.max(np.abs(b))
            self.assertGreater(scale, 0, f"{name}[{i}] is all zeros")
            np.testing.assert_allclose(
                a, b, rtol=0, atol=atol_scale * scale, err_msg=f"{name}[{i}]"
            )


class TestRowsMatchTheSparseChain(FusedDegradeBase):
    def test_kernels_are_actually_used(self):
        corr = self.corr
        _, calls = self.fused(
            corr, lambda c: c._expand_rows(
                (self.shear[0], self.shear[1]), self.w, "pairs")
        )
        self.assertIn("gpu_degrade_level", LAUNCH_LOG)
        self.assertIn("gpu_degrade_finalize", LAUNCH_LOG)
        self.assertGreater(corr._treecode.n_levels, 1)  # a real chain

    def test_expand_rows_matches(self):
        corr = self.corr
        for name, values in (
            ("weights only", ()),
            ("one value", (self.dens,)),
            ("two values", (self.shear[0], self.shear[1])),
        ):
            for blocks in ("pairs", "all"):
                with self.subTest(values=name, blocks=blocks):
                    call = lambda c: c._expand_rows(values, self.w, blocks)
                    (v_f, w_f), _ = self.fused(corr, call)
                    v_s, w_s = self.sparse(corr, call)
                    self.assert_close(tuple(v_f), tuple(v_s), f"{name}/{blocks}")
                    self.assert_close(w_f, w_s, f"{name}/{blocks}/w")

    def test_unrequested_blocks_stay_zero(self):
        """The contract: rows outside the requested blocks are zero."""
        corr = make_corr(aperture_nside=32)
        n_ap = corr.n_aperture_cells
        self.assertGreater(n_ap, 0)
        n_active = corr.n_active
        call = lambda c: c._expand_rows((self.dens,), self.w, "aperture")
        (v_f, w_f), _ = self.fused(corr, call)
        # aperture rows filled, treecode rows untouched -> exactly zero
        self.assertTrue(np.any(np.asarray(w_f)[:, n_active:n_active + n_ap] != 0))
        self.assertTrue(np.all(np.asarray(w_f)[:, n_active + n_ap:] == 0))
        self.assertTrue(np.all(np.asarray(v_f[0])[:, n_active + n_ap:] == 0))

    def test_aperture_level_matches(self):
        corr = make_corr(aperture_nside=32)
        for blocks in ("aperture", "all"):
            with self.subTest(blocks=blocks):
                call = lambda c: c._expand_rows(
                    (self.shear[0], self.shear[1]), self.w, blocks)
                (v_f, w_f), _ = self.fused(corr, call)
                v_s, w_s = self.sparse(corr, call)
                self.assert_close(tuple(v_f), tuple(v_s), f"aperture/{blocks}")
                self.assert_close(w_f, w_s, f"aperture/{blocks}/w")

    def test_cell_rows_are_weighted_means_and_weight_sums(self):
        """T3, independently of either implementation: a level-0 cell must
        hold sum(w) and sum(w*v)/sum(w) over its member pixels."""
        corr = self.corr
        call = lambda c: c._expand_rows((self.dens,), self.w, "pairs")
        (v_f, w_f), _ = self.fused(corr, call)
        v_f, w_f = np.asarray(v_f[0]), np.asarray(w_f)
        n_active, n_ap = corr.n_active, corr.n_aperture_cells
        tree = corr._treecode
        lut = corr._global_to_row_lut()
        indptr = tree.child_indptr[0]
        indices = tree.child_indices[0]
        if lut is not None:
            indices = lut[indices]
        start = int(tree.level_starts(first=n_ap)[0])
        checked = 0
        for cell in range(0, min(400, int(indptr.size - 1))):
            child = indices[int(indptr[cell]):int(indptr[cell + 1])].astype(np.int64)
            if child.size == 0:
                continue
            for lead in range(self.w.shape[0]):
                wsum = self.w[lead][child].sum()
                mean = (self.w[lead][child] * self.dens[lead][child]).sum() / wsum
                row = n_active + start + cell
                self.assertAlmostEqual(w_f[lead, row], wsum, places=10)
                self.assertAlmostEqual(v_f[lead, row], mean, places=10)
            checked += 1
        self.assertGreater(checked, 50)

    def test_deterministic(self):
        corr = self.corr
        call = lambda c: c._expand_rows(
            (self.shear[0], self.shear[1]), self.w, "all")
        (a, wa), _ = self.fused(corr, call)
        (b, wb), _ = self.fused(corr, call)
        self.assertTrue(np.array_equal(np.asarray(a[0]), np.asarray(b[0])))
        self.assertTrue(np.array_equal(np.asarray(wa), np.asarray(wb)))

    def test_float32_maps_keep_a_float64_chain(self):
        """With map_precision=float32 and accumulation_precision=float64 the
        level chain must run at float64, as the sparse path's acc-dtype
        intermediates do -- not at the map dtype."""
        corr = make_corr(map_precision="float32",
                         accumulation_precision="float64")
        w32 = self.w.astype(np.float32)
        d32 = self.dens.astype(np.float32)
        call = lambda c: c._expand_rows((d32,), w32, "pairs")
        (v_f, w_f), _ = self.fused(corr, call)
        v_s, w_s = self.sparse(corr, call)
        self.assertEqual(np.asarray(w_f).dtype, np.float32)
        self.assert_close(np.asarray(v_f[0]), np.asarray(v_s[0]),
                          "float32 maps", atol_scale=1e-6)
        self.assert_close(np.asarray(w_f), np.asarray(w_s),
                          "float32 weights", atol_scale=1e-6)


@pytest.mark.slow
class TestPublicMethodsMatch(FusedDegradeBase):
    """Every measurement that degrades rows, end to end."""

    def test_methods(self):
        corr = self.corr
        cases = {
            "get_full_tomo_shear": lambda c: c.get_full_tomo_shear(
                self.shear, self.w, return_device=False),
            "vectorized_shear_shear": lambda c: c.vectorized_shear_shear(
                self.shear, self.w, return_device=False),
            "get_full_tomo_density": lambda c: c.get_full_tomo_density(
                self.dens, self.w, return_device=False),
            "get_full_tomo_ggl": lambda c: c.get_full_tomo_ggl(
                self.dens, self.shear, self.w, self.w, return_device=False),
            "get_3x2pt_tomo": lambda c: c.get_3x2pt_tomo(
                shear_maps=self.shear, density_maps=self.dens,
                weights={"shear": self.w, "density": self.w},
                return_device=False),
            "get_aperture_shear": lambda c: c.get_aperture_shear(
                self.shear[0, 0], self.shear[0, 1], self.w[0],
                return_device=False),
        }
        for name, call in cases.items():
            with self.subTest(method=name):
                got, _ = self.fused(corr, call)
                want = self.sparse(corr, call)
                self.assert_close(got, want, name)


if __name__ == "__main__":
    unittest.main()
