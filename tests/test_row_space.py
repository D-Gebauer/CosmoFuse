"""T5 — compact row space.

Device map buffers and device pair/aperture indices address the unmasked
pixels only ("row space").  These are machine-precision gates:

* full-sky map input  ==  row-space map input           (bitwise)
* compact row space   ==  legacy full-sky device layout (bitwise)

for every public measurement method, on the CPU backend and through the
real GPU orchestrator branches (numpy-emulated CUDA kernels).
"""

import unittest

import healpy as hp
import numpy as np

from CosmoFuse.correlations import Correlation

from .test_gpu_kernel_emulation import emulated_gpu

NSIDE = 32
NPIX = hp.nside2npix(NSIDE)


def _make_mask() -> np.ndarray:
    theta, phi = hp.pix2ang(NSIDE, np.arange(NPIX))
    dec = 90.0 - np.degrees(theta)
    mask = (dec > -50.0) & (dec < 25.0) & (np.degrees(phi) < 120.0)
    # a hole inside the footprint
    hole = hp.query_disc(NSIDE, hp.ang2vec(np.radians(100.0), np.radians(40.0)), np.radians(4.0))
    mask[hole] = False
    return mask


def _make_corr(mask, **kwargs) -> Correlation:
    phi_c = np.radians([30.0, 42.0, 80.0])
    theta_c = np.radians([95.0, 102.0, 70.0])
    corr = Correlation(
        NSIDE,
        phi_c,
        theta_c,
        nbins=4,
        theta_min=150,
        theta_max=1100,
        patch_size=600,
        theta_Q=150,
        mask=mask,
        device="cpu",
        map_precision="float64",
        rotation_precision="float64",
        **kwargs,
    )
    corr.calculate_pairs_M_a()
    corr.calculate_pairs_2PCF()
    return corr


def _legacy_layout(corr: Correlation) -> Correlation:
    """Same geometry, but with the pre-row-space full-sky device layout."""
    legacy = _make_corr(np.ones(NPIX, dtype=bool))
    legacy.pair_inds = corr.pair_inds
    legacy.pair_exp2phi = corr.pair_exp2phi
    legacy.bins = corr.bins
    legacy.Q_inds, legacy.Q_cos, legacy.Q_sin = corr.Q_inds, corr.Q_cos, corr.Q_sin
    legacy.Q_val, legacy.Q_patch_area = corr.Q_val, corr.Q_patch_area
    legacy._prepare_aperture_flat()
    legacy.prepare()
    assert legacy.n_active == NPIX
    return legacy


class TestRowSpace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mask = _make_mask()
        cls.corr = _make_corr(cls.mask)
        cls.corr.prepare()
        cls.legacy = _legacy_layout(cls.corr)

        rng = np.random.default_rng(7)
        nz = 3
        cls.shear = rng.normal(size=(nz, 2, NPIX)) * 0.02
        cls.dens = rng.normal(size=(2, NPIX))
        cls.w_s = rng.uniform(0.2, 2.0, size=(nz, NPIX))
        cls.w_d = rng.uniform(0.2, 2.0, size=(2, NPIX))
        # masked pixels carry garbage that must never be touched
        for arr in (cls.shear, cls.dens, cls.w_s, cls.w_d):
            arr[..., ~cls.mask] = np.nan

    # -- helpers ------------------------------------------------------------
    def rows(self, arr):
        return np.ascontiguousarray(arr[..., self.corr.row_pix])

    def legacy_maps(self, arr):
        # The legacy single-map CPU kernels seed their accumulators from
        # element 0 of the (full-sky) map, so masked pixels must be finite
        # there -- as in production maps, which are zero outside the mask.
        return np.nan_to_num(arr, nan=0.0)

    def assert_same(self, a, b):
        a = a if isinstance(a, tuple) else (a,)
        b = b if isinstance(b, tuple) else (b,)
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            x, y = np.asarray(x), np.asarray(y)
            self.assertEqual(x.shape, y.shape)
            self.assertTrue(np.all(np.isfinite(x)))
            self.assertTrue(np.array_equal(x, y), f"max|d|={np.max(np.abs(x - y))}")

    def check(self, call):
        """call(corr, f) with f mapping a full-sky array to the input form."""
        full = call(self.corr, lambda a: a)
        rows = call(self.corr, self.rows)
        legacy = call(self.legacy, self.legacy_maps)
        self.assert_same(full, rows)
        self.assert_same(full, legacy)
        return full

    # -- geometry -----------------------------------------------------------
    def test_row_space_is_the_footprint(self):
        corr = self.corr
        self.assertTrue(np.array_equal(corr.row_pix, np.flatnonzero(self.mask)))
        self.assertEqual(corr.n_active, int(self.mask.sum()))
        self.assertLess(corr.n_active, NPIX)
        # device indices are rows, host indices stay HEALPix ids
        inds = np.asarray(corr.inds_dev)
        self.assertLess(int(inds.max()), corr.n_active)
        host = np.concatenate([p for p in corr.pair_inds], axis=1)
        self.assertTrue(np.array_equal(corr.row_pix[inds], host))
        q_rows = np.asarray(corr._aperture_row_inds())
        self.assertLess(int(q_rows.max()), corr.n_active)
        self.assertTrue(np.array_equal(corr.row_pix[q_rows], corr.Q_inds_flat))

    def test_unmasked_row_space_is_identity(self):
        self.assertIsNone(self.legacy._global_to_row_lut())
        self.assertTrue(np.array_equal(self.legacy.row_pix, np.arange(NPIX)))

    def test_wrong_length_maps_raise(self):
        bad = np.ones((3, 2, NPIX // 4))
        with self.assertRaisesRegex(ValueError, "map arrays must have either npix"):
            self.corr.vectorized_shear_shear(bad, np.ones((3, NPIX // 4)), return_device=False)

    def test_geometry_outside_mask_raises(self):
        other = _make_corr(self.mask)
        shrunk = self.mask.copy()
        shrunk[other.pair_inds[0][0, 0]] = False
        other.map_inds = np.flatnonzero(shrunk).astype(other.index_dtype)
        with self.assertRaisesRegex(ValueError, "outside the mask"):
            other.prepare()

    # -- every public measurement method ------------------------------------
    def test_aperture_single_map(self):
        self.check(lambda c, f: c.get_aperture_shear(
            f(self.shear[0, 0]), f(self.shear[0, 1]), f(self.w_s[0]), return_device=False))
        self.check(lambda c, f: c.get_aperture_density(
            f(self.dens[0]), f(self.w_d[0]), return_device=False))

    def test_single_pair_2pcf(self):
        s, w = self.shear, self.w_s
        self.check(lambda c, f: c.compute_shear_shear(
            *(lambda g1, g2, ww: (g1, g2, g1, g2, ww, ww))(f(s[0, 0]), f(s[0, 1]), f(w[0])),
            return_device=False))
        self.check(lambda c, f: c.compute_shear_shear(
            f(s[0, 0]), f(s[0, 1]), f(s[1, 0]), f(s[1, 1]), f(w[0]), f(w[1]), return_device=False))
        self.check(lambda c, f: c.compute_density_density(
            f(self.dens[0]), f(self.dens[1]), f(self.w_d[0]), f(self.w_d[1]), return_device=False))
        self.check(lambda c, f: c.compute_density_shear(
            f(self.dens[0]), f(s[1, 0]), f(s[1, 1]), f(self.w_d[0]), f(w[1]), return_device=False))

    def test_vectorized_tomography(self):
        for flips in ({}, {"flip_g1": True}, {"flip_g2": True}):
            self.check(lambda c, f: c.vectorized_shear_shear(
                f(self.shear), f(self.w_s), return_device=False, **flips))
            self.check(lambda c, f: c.vectorized_density_shear(
                f(self.dens), f(self.shear), f(self.w_d), f(self.w_s), return_device=False, **flips))
        self.check(lambda c, f: c.vectorized_density_density(
            f(self.dens), f(self.w_d), return_device=False))
        self.check(lambda c, f: c.vectorized_density_density(
            f(self.dens), f(self.w_d), gc_auto_correlations_only=True, return_device=False))

    def test_full_tomo_and_fused(self):
        self.check(lambda c, f: c.get_full_tomo_shear(
            f(self.shear), f(self.w_s), flip_g1=True, return_device=False))
        self.check(lambda c, f: c.get_full_tomo_density(
            f(self.dens), f(self.w_d), return_device=False))
        self.check(lambda c, f: c.get_full_tomo_ggl(
            f(self.dens), f(self.shear), f(self.w_d), f(self.w_s),
            return_N_ap=True, return_M_ap=True, flip_g2=True, return_device=False))
        for flips in ({}, {"flip_g1": True, "flip_g2": True}):
            self.check(lambda c, f: tuple(np.array(o, copy=True) for o in c.get_3x2pt_tomo(
                shear_maps=f(self.shear), density_maps=f(self.dens),
                weights=(f(self.w_s), f(self.w_d)), return_device=False, **flips)))
        # unit weights are created in row space
        self.check(lambda c, f: tuple(np.array(o, copy=True) for o in c.get_3x2pt_tomo(
            shear_maps=f(self.shear), density_maps=f(self.dens), return_device=False)))

    def test_gpu_orchestrator_paths(self):
        cpu = {
            "shear": self.corr.get_full_tomo_shear(self.shear, self.w_s, flip_g1=True, return_device=False),
            "dd": self.corr.vectorized_density_density(self.dens, self.w_d, return_device=False),
            "ds": self.corr.vectorized_density_shear(
                self.dens, self.shear, self.w_d, self.w_s, return_device=False),
        }
        for corr, to_input in ((self.corr, self.rows), (self.corr, lambda a: a), (self.legacy, lambda a: a)):
            with emulated_gpu(corr) as calls:
                corr.compute_context.Q_inds_dev = None
                gpu = {
                    "shear": corr.get_full_tomo_shear(
                        to_input(self.shear), to_input(self.w_s), flip_g1=True, return_device=False),
                    "dd": corr.vectorized_density_density(
                        to_input(self.dens), to_input(self.w_d), return_device=False),
                    "ds": corr.vectorized_density_shear(
                        to_input(self.dens), to_input(self.shear), to_input(self.w_d),
                        to_input(self.w_s), return_device=False),
                }
                self.assertGreater(calls["xipm_tomo_vectorized_kernel"], 0)
                self.assertGreater(calls["aperture_tomo_shear_kernel"], 0)
            corr.compute_context.Q_inds_dev = None
            for key in cpu:
                a = cpu[key] if isinstance(cpu[key], tuple) else (cpu[key],)
                b = gpu[key] if isinstance(gpu[key], tuple) else (gpu[key],)
                for x, y in zip(a, b):
                    np.testing.assert_allclose(y, x, rtol=1e-12, atol=1e-18)

    # -- frozen (read-only) inputs are gathered/uploaded once -----------------
    def test_frozen_weights_are_memoised(self):
        corr = _make_corr(self.mask)
        corr.prepare()
        w = self.w_s.copy()
        ref = corr.vectorized_shear_shear(self.shear, w, return_device=False)
        w.flags.writeable = False
        first = corr._coerce_map_input_array(w)
        self.assertIs(corr._coerce_map_input_array(w), first)
        self.assertEqual(first.shape[-1], corr.n_active)
        self.assert_same(corr.vectorized_shear_shear(self.shear, w, return_device=False), ref)
        # writable arrays are never memoised
        w2 = self.w_s.copy()
        self.assertIsNot(corr._coerce_map_input_array(w2), corr._coerce_map_input_array(w2))

    def test_pickle_roundtrip_keeps_row_space(self):
        import pickle

        clone = pickle.loads(pickle.dumps(self.corr))
        clone.prepare()
        self.assertTrue(np.array_equal(clone.row_pix, self.corr.row_pix))
        self.assert_same(
            clone.vectorized_shear_shear(self.rows(self.shear), self.rows(self.w_s), return_device=False),
            self.corr.vectorized_shear_shear(self.shear, self.w_s, return_device=False),
        )


if __name__ == "__main__":
    unittest.main()
