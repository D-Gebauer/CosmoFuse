"""Row-buffer layouts and the folded sign flip (idea #3, sub-items 2 and 3).

The fused degrade writes the layout each leaf wants -- SoA, the
interleaved ``(nz, 2, n_rows)`` shear stack, or the ``(n_rows, nz, 2)``
AoS buffers the pair kernels load -- and applies ``flip_g1``/``flip_g2``
while it copies the pixel rows.  Both are pure data movement, so the
requirement is *bitwise* equality with the layout-and-flip-afterwards code
they replace:

  * a sign is +-1, so scaling is exact, and the degrade is linear in the
    values -- scaling before it is the same float as scaling after;
  * a transpose moves floats without touching them.

Anything weaker would hide a real mistake, so nothing here uses a
tolerance.  The kernels run through the real cupy wrappers against the
numpy twins in tests/cuda_emulation.py, which also assert that the
aperture kernels are handed the element strides their views really have.
"""

import unittest
from unittest.mock import patch

import numpy as np
import pytest

from CosmoFuse import Correlation

from .test_degrade_kernel import FusedDegradeBase, make_corr
from .test_gpu_kernel_emulation import emulated_gpu


def arr(x):
    return np.asarray(x)


class TestLayoutsAgree(FusedDegradeBase):
    """Every layout holds the same numbers, bit for bit."""

    def test_aos_is_the_transpose_of_soa(self):
        corr = self.corr
        for blocks in ("pairs", "all"):
            with self.subTest(blocks=blocks):
                soa, _ = self.fused(corr, lambda c: c._expand_shear_rows(
                    self.shear, self.w, blocks=blocks))
                aos, _ = self.fused(corr, lambda c: c._expand_shear_rows(
                    self.shear, self.w, blocks=blocks, layout="aos"))
                self.assertEqual(arr(aos[0]).shape,
                                 (corr.n_rows, self.shear.shape[0], 2))
                self.assertTrue(np.array_equal(
                    arr(aos[0]), np.transpose(arr(soa[0]), (2, 0, 1))))
                self.assertTrue(np.array_equal(
                    arr(aos[1]), np.transpose(arr(soa[1]), (1, 0))))

    def test_aos_scalar_values_match(self):
        corr = self.corr
        soa, _ = self.fused(corr, lambda c: c._expand_rows(
            (self.dens,), self.w, "pairs"))
        aos, _ = self.fused(corr, lambda c: c._expand_rows_aos(
            (self.dens,), self.w, "pairs"))
        self.assertTrue(np.array_equal(
            arr(aos[0][0]), np.transpose(arr(soa[0][0]), (1, 0))))
        self.assertTrue(np.array_equal(
            arr(aos[1]), np.transpose(arr(soa[1]), (1, 0))))

    def test_interleaved_shear_is_the_stack_of_the_components(self):
        """`_expand_shear_rows` used to stack the two component buffers;
        the interleaved layout is that stack, written in place."""
        corr = self.corr
        stacked, _ = self.fused(corr, lambda c: c._expand_shear_rows(
            self.shear, self.w, blocks="pairs"))
        parts, _ = self.fused(corr, lambda c: c._expand_rows(
            (self.shear[:, 0], self.shear[:, 1]), self.w, "pairs"))
        self.assertTrue(np.array_equal(
            arr(stacked[0]), np.stack([arr(parts[0][0]), arr(parts[0][1])],
                                      axis=1)))

    def test_aos_buffers_are_contiguous(self):
        """The pair kernels index them as flat AoS memory."""
        corr = self.corr
        aos, _ = self.fused(corr, lambda c: c._expand_shear_rows(
            self.shear, self.w, blocks="pairs", layout="aos"))
        self.assertTrue(arr(aos[0]).flags["C_CONTIGUOUS"])
        self.assertTrue(arr(aos[1]).flags["C_CONTIGUOUS"])
        dens, _ = self.fused(corr, lambda c: c._expand_rows_aos(
            (self.dens,), self.w, "pairs"))
        self.assertTrue(arr(dens[0][0]).flags["C_CONTIGUOUS"])

    def test_sparse_fallback_produces_the_same_layouts(self):
        """The AoS layout is not a fused-only feature: the sparse chain
        reaches it by transposing, and must land on the same bits."""
        corr = self.corr
        fused, _ = self.fused(corr, lambda c: c._expand_shear_rows(
            self.shear, self.w, blocks="pairs", layout="aos"))
        sparse = self.sparse(corr, lambda c: c._expand_shear_rows(
            self.shear, self.w, blocks="pairs", layout="aos"))
        self.assertEqual(arr(fused[0]).shape, arr(sparse[0]).shape)
        np.testing.assert_allclose(arr(fused[0]), arr(sparse[0]),
                                   rtol=0, atol=1e-13 * np.abs(arr(sparse[0])).max())


class TestSignsAreFoldedIn(FusedDegradeBase):
    """A folded sign must equal flipping the maps, exactly."""

    def signed(self, corr, signs, **kw):
        out, _ = self.fused(corr, lambda c: c._expand_shear_rows(
            self.shear, self.w, blocks="pairs", signs=signs, **kw))
        return out

    def flipped_input(self, corr, signs, **kw):
        shear = self.shear * np.asarray(signs).reshape(1, 2, 1)
        out, _ = self.fused(corr, lambda c: c._expand_shear_rows(
            shear, self.w, blocks="pairs", **kw))
        return out

    def test_shear_signs_match_flipping_the_maps(self):
        corr = self.corr
        for signs in ((-1, 1), (1, -1), (-1, -1), (1, 1)):
            for layout in ("soa", "aos"):
                with self.subTest(signs=signs, layout=layout):
                    got = self.signed(corr, signs, layout=layout)
                    want = self.flipped_input(corr, signs, layout=layout)
                    self.assertTrue(np.array_equal(arr(got[0]), arr(want[0])))
                    self.assertTrue(np.array_equal(arr(got[1]), arr(want[1])))

    def test_scalar_signs_match_flipping_the_maps(self):
        corr = self.corr
        got, _ = self.fused(corr, lambda c: c._expand_rows(
            (self.dens,), self.w, "pairs", signs=(-1,)))
        want, _ = self.fused(corr, lambda c: c._expand_rows(
            (-self.dens,), self.w, "pairs"))
        self.assertTrue(np.array_equal(arr(got[0][0]), arr(want[0][0])))

    def test_the_cell_rows_carry_the_sign_too(self):
        """Not just the pixel rows: level 0 reads the already-signed rows,
        so every virtual row must come out negated as well."""
        corr = self.corr
        plain, _ = self.fused(corr, lambda c: c._expand_shear_rows(
            self.shear, self.w, blocks="pairs"))
        flipped, _ = self.fused(corr, lambda c: c._expand_shear_rows(
            self.shear, self.w, blocks="pairs", signs=(-1, -1)))
        n_active = corr.n_active
        cells_plain = arr(plain[0])[:, :, n_active:]
        cells_flipped = arr(flipped[0])[:, :, n_active:]
        self.assertGreater(cells_plain.size, 0)
        self.assertTrue(np.any(cells_plain != 0))
        self.assertTrue(np.array_equal(cells_flipped, -cells_plain))
        # the weights are untouched by a value sign
        self.assertTrue(np.array_equal(arr(plain[1]), arr(flipped[1])))

    def test_sparse_fallback_applies_the_signs(self):
        corr = self.corr
        got = self.sparse(corr, lambda c: c._expand_rows(
            (self.dens,), self.w, "pairs", signs=(-1,)))
        want = self.sparse(corr, lambda c: c._expand_rows(
            (-self.dens,), self.w, "pairs"))
        self.assertTrue(np.array_equal(arr(got[0][0]), arr(want[0][0])))

    def test_trivial_signs_do_not_copy(self):
        """signs=(1, 1) must be indistinguishable from no signs at all."""
        corr = self.corr
        self.assertIsNone(corr._sign_scales((1, 1), 2))
        self.assertIsNone(corr._sign_scales(None, 2))
        self.assertEqual(corr._sign_scales((1, -1), 2), (1.0, -1.0))
        with self.assertRaises(ValueError):
            corr._sign_scales((1,), 2)


class TestOneExpansionPerCall(FusedDegradeBase):
    """One degrade per public call -- and in the layout that call's leaves
    can all use.

    A call whose aperture pass runs `aperture_tomo.cu` shares an SoA
    expansion and lets the pair leaf transpose: that kernel gathers
    aperture discs, whose row ids are largely contiguous, and AoS would
    cost it far more than the transpose saves.  A pair-only call (or one
    whose aperture pass is the fused 3x2pt kernel, which is AoS anyway)
    takes AoS straight from the degrade.
    """

    def count_expansions(self, corr, fn):
        seen = []
        real = Correlation._expand_rows_fused

        def counting(self, *a, **kw):
            out = real(self, *a, **kw)
            seen.append(kw.get("layout", "soa"))
            return out

        with patch.object(Correlation, "_expand_rows_fused", counting):
            self.fused(corr, fn)
        return seen

    def test_get_full_tomo_shear_degrades_once_in_soa(self):
        corr = self.corr
        seen = self.count_expansions(corr, lambda c: c.get_full_tomo_shear(
            self.shear, self.w, return_device=False))
        self.assertEqual(seen, ["interleaved"])

    def test_get_full_tomo_density_degrades_once_in_soa(self):
        corr = self.corr
        seen = self.count_expansions(corr, lambda c: c.get_full_tomo_density(
            self.dens, self.w, return_device=False))
        self.assertEqual(seen, ["soa"])

    def test_a_flip_does_not_cost_a_second_expansion(self):
        corr = self.corr
        seen = self.count_expansions(corr, lambda c: c.get_full_tomo_shear(
            self.shear, self.w, flip_g1=True, return_device=False))
        self.assertEqual(seen, ["interleaved"])

    def test_pair_only_calls_take_aos_from_the_degrade(self):
        corr = self.corr
        seen = self.count_expansions(corr, lambda c: c.vectorized_shear_shear(
            self.shear, self.w, return_device=False))
        self.assertEqual(seen, ["aos"])
        seen = self.count_expansions(corr, lambda c: c.get_3x2pt_tomo(
            shear_maps=self.shear, density_maps=self.dens,
            weights={"shear": self.w, "density": self.w},
            return_device=False))
        self.assertEqual(sorted(seen), ["aos", "aos"])

    def test_the_scope_decides_the_pair_layout(self):
        corr = self.corr
        self.assertEqual(corr._pair_row_layout(), "aos")
        with corr._expansion_scope(layout="soa"):
            self.assertEqual(corr._pair_row_layout(), "soa")
        self.assertEqual(corr._pair_row_layout(), "aos")

    def test_an_soa_scope_still_hands_the_pair_kernels_aos(self):
        """Only the route changes, never what the kernels get."""
        corr = self.corr
        with emulated_gpu(corr):
            with patch.object(Correlation, "_use_fused_degrade",
                              lambda self, w: True):
                direct = corr._pair_shear_rows(self.shear, self.w, "pairs")
                with corr._expansion_scope(layout="soa"):
                    via_soa = corr._pair_shear_rows(self.shear, self.w, "pairs")
        corr.compute_context.degrade_csr = None
        self.assertEqual(arr(direct[0]).shape, arr(via_soa[0]).shape)
        np.testing.assert_allclose(
            arr(direct[0]), arr(via_soa[0]), rtol=0,
            atol=1e-13 * np.abs(arr(via_soa[0])).max())
        self.assertTrue(np.array_equal(arr(direct[1]), arr(via_soa[1])))


@pytest.mark.slow
class TestPublicFlipsUnchanged(FusedDegradeBase):
    """Folding the flip into the expansion must not change any output."""

    def cases(self, corr, shear, flip_g1, flip_g2):
        return {
            "get_full_tomo_shear": lambda c: c.get_full_tomo_shear(
                shear, self.w, flip_g1=flip_g1, flip_g2=flip_g2,
                return_device=False),
            "vectorized_shear_shear": lambda c: c.vectorized_shear_shear(
                shear, self.w, flip_g1=flip_g1, flip_g2=flip_g2,
                return_device=False),
            "get_full_tomo_ggl": lambda c: c.get_full_tomo_ggl(
                self.dens, shear, self.w, self.w, flip_g1=flip_g1,
                flip_g2=flip_g2, return_device=False),
            "get_3x2pt_tomo": lambda c: c.get_3x2pt_tomo(
                shear_maps=shear, density_maps=self.dens,
                weights={"shear": self.w, "density": self.w},
                flip_g1=flip_g1, flip_g2=flip_g2, return_device=False),
        }

    def test_flip_flags_equal_pre_flipped_maps(self):
        corr = self.corr
        for flip_g1, flip_g2 in ((True, False), (False, True), (True, True)):
            signs = np.array([-1.0 if flip_g1 else 1.0,
                              -1.0 if flip_g2 else 1.0]).reshape(1, 2, 1)
            pre = self.shear * signs
            flagged = self.cases(corr, self.shear, flip_g1, flip_g2)
            plain = self.cases(corr, pre, False, False)
            for name in flagged:
                with self.subTest(method=name, flip=(flip_g1, flip_g2)):
                    got, _ = self.fused(corr, flagged[name])
                    want, _ = self.fused(corr, plain[name])
                    self.assert_close(got, want, name, atol_scale=1e-14)


@pytest.mark.slow
class TestFullResolutionPath(unittest.TestCase):
    """resolution_factor=None has no virtual rows: the layouts and the
    sign flip must still be right, and must cost one copy, not two."""

    @classmethod
    def setUpClass(cls):
        cls.corr = make_corr(k=None)
        rng = np.random.default_rng(7)
        n = cls.corr.n_active
        cls.shear = rng.normal(size=(2, 2, n)) * 0.3
        cls.w = rng.uniform(0.2, 2.0, size=(2, n))

    def test_no_virtual_rows(self):
        self.assertEqual(self.corr.n_appended, 0)

    def test_aos_and_signs_without_appended_rows(self):
        corr = self.corr
        with emulated_gpu(corr):
            aos, w_aos = corr._expand_shear_rows(
                self.shear, self.w, blocks="pairs", signs=(-1, 1),
                layout="aos")
        want = np.transpose(self.shear * np.array([-1.0, 1.0]).reshape(1, 2, 1),
                            (2, 0, 1))
        self.assertTrue(np.array_equal(arr(aos), want))
        self.assertTrue(np.array_equal(arr(w_aos), np.transpose(self.w, (1, 0))))
        self.assertTrue(arr(aos).flags["C_CONTIGUOUS"])

    def test_soa_without_signs_returns_the_input_untouched(self):
        corr = self.corr
        with emulated_gpu(corr):
            rows, w_rows = corr._expand_shear_rows(
                self.shear, self.w, blocks="pairs")
        self.assertIs(rows, self.shear)
        self.assertIs(w_rows, self.w)


if __name__ == "__main__":
    unittest.main()
