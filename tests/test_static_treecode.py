"""Static treecode (per-bin resolution levels) -- machine-precision gates.

T1   full resolution (the default) is untouched and pays nothing
T1b  option plumbing, provenance, preflight memory check
T2   brute-force parent reference: every fine pair, binned and rotated with
     its parents' geometry, equals the treecode result to <= 1e-13
T3   degrade operator (ud_grade equality, empty cells, patch edge)
T4   pair-count conservation per level
T6   IO round trip, patch slicing, format compatibility
T7   CPU kernels vs emulated-GPU orchestrator paths
"""

import os
import pickle
import tempfile
import unittest
import warnings

import h5py
import healpy as hp
import numpy as np
import pytest

from CosmoFuse.correlations import Correlation, _compute_pairs_impl
from CosmoFuse.treecode import assign_levels, level_groups

from .test_gpu_kernel_emulation import emulated_gpu

NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
# nside 64/32/16 pixel sizes: 55' / 110' / 220'.  With k = 2:
#   bins starting at 120', 190' -> nside 64;  301' -> 32;  477', 757' -> 16
BIN_ARGS = dict(nbins=5, theta_min=120, theta_max=1200)
EXPECTED_LEVELS = [64, 64, 32, 16, 16]
PHI_C = np.radians([40.0, 47.0, 80.0])
THETA_C = np.radians([95.0, 100.0, 60.0])


def make_mask() -> np.ndarray:
    theta, phi = hp.pix2ang(NSIDE, np.arange(NPIX))
    mask = (np.degrees(phi) > 20.0) & (np.degrees(phi) < 110.0)
    rng = np.random.default_rng(3)
    # holes of several sizes, some inside the patches
    for lon, lat, rad in ((42.0, -6.0, 2.5), (50.0, -12.0, 1.0), (78.0, 31.0, 3.0)):
        hole = hp.query_disc(NSIDE, hp.ang2vec(np.radians(90 - lat), np.radians(lon)), np.radians(rad))
        mask[hole] = False
    mask[rng.choice(NPIX, NPIX // 40, replace=False)] = False  # pepper
    return mask


def make_corr(mask, resolution_factor=None, **kwargs) -> Correlation:
    args = dict(
        patch_size=700,
        theta_Q=200,
        mask=mask,
        device="cpu",
        map_precision="float64",
        rotation_precision="float64",
        resolution_factor=resolution_factor,
    )
    args.update(BIN_ARGS)
    args.update(kwargs)
    return Correlation(NSIDE, PHI_C, THETA_C, **args)


def make_maps(mask, seed=11, nz=2):
    rng = np.random.default_rng(seed)
    shear = rng.normal(size=(nz, 2, NPIX)) * 0.03
    dens = rng.normal(size=(nz, NPIX))
    w_s = rng.uniform(0.1, 3.0, size=(nz, NPIX))
    w_d = rng.uniform(0.1, 3.0, size=(nz, NPIX))
    # some observed pixels without sources
    w_s[:, rng.choice(NPIX, NPIX // 10, replace=False)] = 0.0
    for arr in (shear, dens, w_s, w_d):
        arr[..., ~mask] = np.nan  # must never be touched
    return shear, dens, w_s, w_d


def patch_pixels(corr, i):
    vec = hp.ang2vec(corr.theta_center[i], corr.phi_center[i])
    disc = hp.query_disc(corr.nside, vec, np.radians(corr.patch_size / 60))
    return disc[corr.map_mask[disc]]


def parents_and_centroids(pix, nside_c):
    """Independent (loop-based) parent relation and binary-mask centroids."""
    shift = 2 * (int(np.log2(NSIDE)) - int(np.log2(nside_c)))
    parent = hp.ring2nest(NSIDE, pix) >> shift
    vec = np.array(hp.pix2vec(NSIDE, pix)).T
    ids = sorted(set(parent.tolist()))
    centroids = []
    for pid in ids:
        v = vec[parent == pid].sum(axis=0)
        centroids.append(v / np.linalg.norm(v))
    centroids = np.array(centroids)
    theta, phi = hp.vec2ang(centroids)
    index = np.array([ids.index(p) for p in parent.tolist()])
    return index, phi, np.pi / 2 - theta, centroids


def brute_force_reference(corr, shear, dens, w_s, w_d):
    """For every fine pair: parents' bin + parents' rotation, fine values.

    Returns dict of (n_patches, nbins) arrays: numerators and denominators of
    xi+/xi- (tomo 0 x tomo 1, both orientations summed), xi_g, xi_t (lens 0,
    source 1).
    """
    nb = corr.nbins
    keys = ("xip", "xim", "ss_den", "xig", "dd_den", "xit", "ds_den", "npairs")
    out = {k: np.zeros((corr.n_patches, nb)) for k in keys}
    g = shear[:, 0] + 1j * shear[:, 1]
    edges = np.asarray(corr.binedges, dtype=np.float64)
    for ip in range(corr.n_patches):
        pix = patch_pixels(corr, ip)
        for nside_b, b0, b1 in level_groups(np.array(EXPECTED_LEVELS)):
            if nside_b == NSIDE:
                index = np.arange(pix.size)
                theta, phi = hp.pix2ang(NSIDE, pix)
                ra, dec = phi, np.pi / 2 - theta
            else:
                index, ra, dec, _ = parents_and_centroids(pix, nside_b)
            ids = np.arange(ra.size, dtype=np.int64)
            I, J, bins, c1, s1, c2, s2, _ = _compute_pairs_impl(
                ids, ra, dec, edges[b0 : b1 + 1]
            )
            members = [pix[index == c] for c in range(ra.size)]
            for a, b, bb, e1, e2 in zip(I, J, bins + b0, c1 + 1j * s1, c2 + 1j * s2):
                pa, pb = members[a], members[b]
                # shear-shear, tomo 0 x tomo 1: (0 at a, 1 at b) + (1 at a, 0 at b)
                for ta, tb in ((0, 1), (1, 0)):
                    wa, wb = w_s[ta][pa], w_s[tb][pb]
                    ga, gb = g[ta][pa] * e1, g[tb][pb] * e2
                    out["xip"][ip, bb] += np.real(np.sum(wb * gb) * np.conj(np.sum(wa * ga)))
                    out["xim"][ip, bb] += np.real(np.sum(wb * gb) * np.sum(wa * ga))
                    out["ss_den"][ip, bb] += wa.sum() * wb.sum()
                # density-density, tomo 0 x tomo 1 (both orientations summed)
                da, db = w_d[0][pa] * dens[0][pa], w_d[1][pb] * dens[1][pb]
                da2, db2 = w_d[1][pa] * dens[1][pa], w_d[0][pb] * dens[0][pb]
                out["xig"][ip, bb] += da.sum() * db.sum() + da2.sum() * db2.sum()
                out["dd_den"][ip, bb] += (
                    w_d[0][pa].sum() * w_d[1][pb].sum() + w_d[1][pa].sum() * w_d[0][pb].sum()
                )
                # tangential shear: lens 0, source 1, both orientations
                gt_b = -np.real(g[1][pb] * e2)
                gt_a = -np.real(g[1][pa] * e1)
                out["xit"][ip, bb] += np.sum(w_d[0][pa] * dens[0][pa]) * np.sum(w_s[1][pb] * gt_b)
                out["xit"][ip, bb] += np.sum(w_d[0][pb] * dens[0][pb]) * np.sum(w_s[1][pa] * gt_a)
                out["ds_den"][ip, bb] += (
                    w_d[0][pa].sum() * w_s[1][pb].sum() + w_d[0][pb].sum() * w_s[1][pa].sum()
                )
                out["npairs"][ip, bb] += pa.size * pb.size
    return out


def clean(arr, mask):
    return np.where(mask, arr, 0.0)


class TreecodeBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mask = make_mask()
        cls.shear, cls.dens, cls.w_s, cls.w_d = make_maps(cls.mask)
        cls.tree = make_corr(cls.mask, resolution_factor=2.0)
        cls.tree.preprocess()


class TestFullResolutionDefault(unittest.TestCase):
    """T1: full resolution is the default and is left untouched."""

    @classmethod
    def setUpClass(cls):
        cls.mask = make_mask()
        cls.maps = make_maps(cls.mask)
        cls.default = make_corr(cls.mask)  # no argument passed
        cls.default.preprocess()
        cls.k_inf = make_corr(cls.mask, resolution_factor=1e9)
        cls.k_inf.preprocess()

    def test_default_is_full_resolution(self):
        for corr in (self.default, self.k_inf):
            self.assertTrue(np.all(corr.level_nside == NSIDE))
            self.assertIsNone(corr._treecode)
            self.assertEqual(corr.n_appended, 0)
            self.assertEqual(corr.n_rows, corr.n_active)
        self.assertIsNone(self.default.resolution_factor)

    def test_geometry_identical(self):
        for a, b in zip(self.default.pair_inds, self.k_inf.pair_inds):
            self.assertTrue(np.array_equal(a, b))
        for a, b in zip(self.default.pair_exp2phi, self.k_inf.pair_exp2phi):
            self.assertTrue(np.array_equal(a, b))
        for a, b in zip(self.default.bins, self.k_inf.bins):
            self.assertTrue(np.array_equal(a, b))

    def test_default_path_does_no_expansion(self):
        corr = self.default
        w = np.ones((2, corr.n_active))
        v = np.ones((2, corr.n_active))
        (v_out,), w_out = corr._expand_rows((v,), w)
        self.assertIs(v_out, v)
        self.assertIs(w_out, w)
        shear = np.ones((2, 2, corr.n_active))
        s_out, w_out = corr._expand_shear_rows(shear, w)
        self.assertIs(s_out, shear)
        self.assertIsNone(getattr(corr.compute_context, "degrade_ops", None))

    @pytest.mark.slow
    def test_results_bitwise_equal(self):
        shear, dens, w_s, w_d = self.maps
        for name, call in (
            ("shear", lambda c: c.get_full_tomo_shear(shear, w_s, return_device=False)),
            ("dens", lambda c: c.get_full_tomo_density(dens, w_d, return_device=False)),
            ("ggl", lambda c: c.get_full_tomo_ggl(dens, shear, w_d, w_s, return_device=False)),
            ("fused", lambda c: tuple(np.array(o) for o in c.get_3x2pt_tomo(
                shear_maps=shear, density_maps=dens, weights=(w_s, w_d), return_device=False))),
        ):
            a, b = call(self.default), call(self.k_inf)
            a = a if isinstance(a, tuple) else (a,)
            b = b if isinstance(b, tuple) else (b,)
            for x, y in zip(a, b):
                self.assertTrue(np.all(np.isfinite(x)), name)
                self.assertTrue(np.array_equal(x, y), name)


class TestOptionPlumbing(TreecodeBase):
    """T1b"""

    def test_invalid_values_raise(self):
        for bad in (0, -1.0, float("nan"), "2", "auto", {"xi_m": 4.0}):
            with self.assertRaises(ValueError, msg=repr(bad)):
                make_corr(self.mask, resolution_factor=bad)
        with self.assertRaisesRegex(ValueError, "power-of-two nside"):
            Correlation(48, PHI_C, THETA_C, resolution_factor=2.0, device="cpu")
        for bad in (3, 128, 48):
            with self.assertRaisesRegex(ValueError, "aperture_nside"):
                make_corr(self.mask, aperture_nside=bad)
        with self.assertRaises(ValueError):
            make_corr(self.mask, memory_budget_gb=0)

    def test_default_factor_when_switched_on(self):
        import CosmoFuse

        self.assertEqual(CosmoFuse.DEFAULT_RESOLUTION_FACTOR, 4.0)
        for on in (True, "default", "DEFAULT"):
            corr = make_corr(self.mask, resolution_factor=on)
            self.assertEqual(corr.resolution_factor, 4.0)
            self.assertEqual(
                corr.level_nside.tolist(),
                make_corr(self.mask, resolution_factor=4.0).level_nside.tolist(),
            )
        for off in (None, False):
            corr = make_corr(self.mask, resolution_factor=off)
            self.assertIsNone(corr.resolution_factor)
            self.assertTrue(np.all(corr.level_nside == NSIDE))
        self.assertIsNone(make_corr(self.mask).resolution_factor)  # default: full

    def test_level_table(self):
        table = self.tree.level_table
        self.assertEqual(table["nside"].tolist(), EXPECTED_LEVELS)
        self.assertEqual(table["resolution_factor"], 2.0)
        self.assertEqual(table["base_nside"], NSIDE)
        self.assertTrue(np.all(table["effective_resolution_factor"] >= 2.0))
        self.assertTrue(np.all(table["effective_resolution_factor"] < 4.0))
        self.assertEqual(
            assign_levels(self.tree.binedges, NSIDE, 2.0).tolist(), EXPECTED_LEVELS
        )
        # read-only: mutating the returned table does not touch the object
        table["nside"][:] = 1
        self.assertEqual(self.tree.level_table["nside"].tolist(), EXPECTED_LEVELS)

    def test_from_mask_forwards_option(self):
        corr = Correlation.from_mask(
            NSIDE, self.mask, 4, patch_size=700, theta_Q=200, f_mask=0.6,
            resolution_factor=2.0, device="cpu", **BIN_ARGS,
        )
        self.assertEqual(corr.level_table["nside"].tolist(), EXPECTED_LEVELS)

    def test_pickle_roundtrip(self):
        clone = pickle.loads(pickle.dumps(self.tree))
        self.assertEqual(clone.resolution_factor, 2.0)
        self.assertEqual(clone.level_table["nside"].tolist(), EXPECTED_LEVELS)
        self.assertEqual(clone.n_appended, self.tree.n_appended)
        clone.prepare()
        a = clone.vectorized_shear_shear(self.shear, self.w_s, return_device=False)
        b = self.tree.vectorized_shear_shear(self.shear, self.w_s, return_device=False)
        for x, y in zip(a, b):
            self.assertTrue(np.array_equal(x, y))

    def test_old_pickle_state_defaults_to_full_resolution(self):
        state = make_corr(self.mask).__getstate__()
        for key in ("resolution_factor", "level_nside", "aperture_nside",
                    "memory_budget_gb", "_treecode"):
            state.pop(key)
        old = Correlation.__new__(Correlation)
        old.__setstate__(state)
        self.assertIsNone(old.resolution_factor)
        self.assertTrue(np.all(old.level_nside == NSIDE))
        self.assertEqual(old.n_appended, 0)

    def test_preflight_fires_before_pair_finding(self):
        from CosmoFuse.pair_geometry import PairGeometry

        # a budget between the full-resolution and the k=2 projection
        need_full = PairGeometry.preflight_pair_memory(
            make_corr(self.mask, memory_budget_gb=1e6))["projected_gb"]
        need_k2 = PairGeometry.preflight_pair_memory(
            make_corr(self.mask, resolution_factor=2.0, memory_budget_gb=1e6))["projected_gb"]
        self.assertLess(need_k2, 0.5 * need_full)
        corr = make_corr(self.mask, memory_budget_gb=0.5 * (need_full + need_k2))
        called = []
        corr._pair_finder.get_pairs_patch_flat = lambda *a, **k: called.append(1)
        with self.assertRaisesRegex(MemoryError, r"resolution_factor=\d") as ctx:
            corr.calculate_pairs_2PCF()
        self.assertEqual(called, [])
        self.assertIn("different, windowed estimator", str(ctx.exception))

    def test_preflight_projection_is_accurate(self):
        from CosmoFuse.pair_geometry import PairGeometry

        for corr in (make_corr(self.mask, memory_budget_gb=1e6), ):
            report = PairGeometry.preflight_pair_memory(corr)
            corr.calculate_pairs_2PCF()
            bytes_per_pair = 2 * corr.index_dtype.itemsize + 2 * corr.rotation_complex_dtype.itemsize
            actual = sum(int(np.sum(b)) for b in corr.bins) * bytes_per_pair / 1e9
            # masked patches: the uniform-disc model is an upper-ish estimate
            self.assertLess(abs(report["projected_gb"] / actual - 1.0), 0.35)


class TestPairSearchPrecision(unittest.TestCase):
    """The pair search precision is decoupled from the stored rotations."""

    def test_modes(self):
        mask = make_mask()
        default = make_corr(mask, rotation_precision="float32")
        self.assertEqual(default._pair_finder.search_dtype, np.float64)  # default
        legacy = make_corr(mask, rotation_precision="float32", pair_search_precision="rotation")
        self.assertEqual(legacy._pair_finder.search_dtype, np.float32)  # historical
        self.assertEqual(
            make_corr(mask, rotation_precision="float32", pair_search_precision="auto")
            ._pair_finder.search_dtype, np.float32)  # auto = historical at full resolution
        tree32 = make_corr(mask, resolution_factor=2.0, rotation_precision="float32",
                           pair_search_precision="auto")
        self.assertEqual(tree32._pair_finder.search_dtype, np.float64)  # auto + treecode
        with self.assertRaisesRegex(ValueError, "pair_search_precision"):
            make_corr(mask, pair_search_precision="float16")

        # float64 search + float32 storage: same pairs as the float64 geometry,
        # rotation factors rounded to float32
        tree64 = make_corr(mask, resolution_factor=2.0)
        tree32.calculate_pairs_2PCF()
        tree64.calculate_pairs_2PCF()
        for a, b in zip(tree32.pair_inds, tree64.pair_inds):
            self.assertTrue(np.array_equal(a, b))
        for a, b in zip(tree32.pair_exp2phi, tree64.pair_exp2phi):
            self.assertEqual(a.dtype, np.complex64)
            self.assertTrue(np.array_equal(a, b.astype(np.complex64)))

        clone = pickle.loads(pickle.dumps(tree32))
        self.assertEqual(clone._pair_finder.search_dtype, np.float64)

    def test_small_scale_float32_search_warns(self):
        with self.assertLogs("CosmoFuse.correlations", level="WARNING") as logs:
            Correlation(64, PHI_C, THETA_C, nbins=3, theta_min=5, theta_max=50, device="cpu",
                        pair_search_precision="rotation")
        self.assertIn("pair_search_precision='float64'", logs.output[0])
        with self.assertNoLogs("CosmoFuse.correlations", level="WARNING"):
            Correlation(64, PHI_C, THETA_C, nbins=3, theta_min=15, theta_max=50, device="cpu",
                        pair_search_precision="rotation")
            Correlation(64, PHI_C, THETA_C, nbins=3, theta_min=5, theta_max=50, device="cpu")


class TestBruteForceReference(TreecodeBase):
    """T2 -- the central correctness test."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.ref = brute_force_reference(
            cls.tree,
            *(clean(a, cls.mask) for a in (cls.shear, cls.dens, cls.w_s, cls.w_d)),
        )

    def assert_close(self, got, want, name):
        scale = np.max(np.abs(want))
        self.assertGreater(scale, 0, name)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-13 * scale, err_msg=name)

    def test_levels_are_exercised(self):
        self.assertEqual(self.tree._treecode.coarse_nsides, (32, 16))
        per_bin = np.sum(np.asarray(self.tree.bins, dtype=np.int64), axis=0)
        self.assertTrue(np.all(per_bin > 0), per_bin)

    def test_denominators(self):
        corr = self.tree
        rows = lambda a: clean(a, self.mask)[..., corr.row_pix]
        _, ws = corr._expand_rows((), rows(self.w_s), "pairs")
        _, wd = corr._expand_rows((), rows(self.w_d), "pairs")
        shape = (corr.n_patches, corr.nbins)
        ss = (
            np.asarray(corr._compute_xipm_sumofweights(ws[0], ws[1]))
            + np.asarray(corr._compute_xipm_sumofweights(ws[1], ws[0]))
        ).reshape(shape)
        self.assert_close(ss, self.ref["ss_den"], "ss_den")
        dd = (
            np.asarray(corr._compute_xipm_sumofweights(wd[0], wd[1]))
            + np.asarray(corr._compute_xipm_sumofweights(wd[1], wd[0]))
        ).reshape(shape)
        self.assert_close(dd, self.ref["dd_den"], "dd_den")
        ds = (
            np.asarray(corr._compute_xipm_sumofweights(wd[0], ws[1]))
            + np.asarray(corr._compute_xipm_sumofweights(ws[1], wd[0]))
        ).reshape(shape)
        self.assert_close(ds, self.ref["ds_den"], "ds_den")

    def check_all_paths(self, corr):
        ref = self.ref
        shear, dens, w_s, w_d = self.shear, self.dens, self.w_s, self.w_d
        with np.errstate(invalid="ignore", divide="ignore"):
            xip_ref = ref["xip"] / ref["ss_den"]
            xim_ref = ref["xim"] / ref["ss_den"]
            xig_ref = ref["xig"] / ref["dd_den"]
            xit_ref = ref["xit"] / ref["ds_den"]

        # single-pair API (cross: tomo 0 x tomo 1), both orientations averaged
        # by the library -> compare the A->B numerators through the vectorized
        # kernel instead, and use the single-pair API for xi_g / xi_t.
        (xig,) = corr.compute_density_density(dens[0], dens[1], w_d[0], w_d[1], return_device=False)
        (xit,) = corr.compute_density_shear(
            dens[0], shear[1, 0], shear[1, 1], w_d[0], w_s[1], return_device=False)
        self.assert_close(xit, xit_ref, "single xi_t")
        # compute_density_density averages the two orientation *ratios*
        ab = np.asarray(corr.compute_density_density(
            dens[0], dens[1], w_d[0], w_d[1], return_device=False)[0])
        self.assertTrue(np.all(np.isfinite(ab)))

        # vectorized tomography
        xi_g = corr.vectorized_density_density(dens, w_d, return_device=False)
        self.assert_close(xi_g[1], xig_ref, "vectorized xi_g (0,1)")
        xi_t = corr.vectorized_density_shear(dens, shear, w_d, w_s, return_device=False)
        self.assert_close(xi_t[1], xit_ref, "vectorized xi_t (0,1)")

        # fused kernel: xi_g, xi_t as above; the xi+- cross term is the ratio
        # of the summed orientations -> the raw fused numerators / denominators
        M_a, M_g, xip, xim, xi_g2, xi_t2 = (
            np.array(o) for o in corr.get_3x2pt_tomo(
                shear_maps=shear, density_maps=dens, weights=(w_s, w_d), return_device=False)
        )
        self.assert_close(xi_g2[1], xig_ref, "fused xi_g (0,1)")
        self.assert_close(xi_t2[1], xit_ref, "fused xi_t (0,1)")
        self.assert_close(xip[1], xip_ref, "fused xi+ (0,1)")
        self.assert_close(xim[1], xim_ref, "fused xi- (0,1)")
        buf = corr.compute_context.fused_output_buffers
        shape = (corr.n_patches, corr.nbins)
        self.assert_close(np.asarray(buf["out_xipm_num"][0, 1]).reshape(shape), ref["xip"], "fused xi+ num")
        self.assert_close(np.asarray(buf["out_xipm_num"][1, 1]).reshape(shape), ref["xim"], "fused xi- num")
        self.assert_close(np.asarray(buf["out_xipm_den"][1]).reshape(shape), ref["ss_den"], "fused den")
        self.assert_close(np.asarray(buf["out_xit_num"][1]).reshape(shape), ref["xit"], "fused xi_t num")
        self.assert_close(np.asarray(buf["out_xit_den"][1]).reshape(shape), ref["ds_den"], "fused xi_t den")
        self.assert_close(np.asarray(buf["out_xig_num"][1]).reshape(shape), ref["xig"], "fused xi_g num")

        # vectorized xi+- must agree with the fused kernel bit-for-bit-ish
        xip_v, xim_v = corr.vectorized_shear_shear(shear, w_s, return_device=False)
        np.testing.assert_allclose(xip_v, xip, rtol=0, atol=1e-13 * np.max(np.abs(xip)))
        np.testing.assert_allclose(xim_v, xim, rtol=0, atol=1e-13 * np.max(np.abs(xim)))
        self.assert_close(xip_v[1], xip_ref, "vectorized xi+ (0,1)")
        # auto-correlation through the single-pair API vs the vectorized path
        xip_s, xim_s = corr.compute_shear_shear(
            shear[0, 0], shear[0, 1], shear[0, 0], shear[0, 1], w_s[0], w_s[0], return_device=False)
        np.testing.assert_allclose(xip_s, xip_v[0], rtol=0, atol=1e-13 * np.max(np.abs(xip_v)))
        np.testing.assert_allclose(xim_s, xim_v[0], rtol=0, atol=1e-13 * np.max(np.abs(xim_v)))
        # the single-map cross API keeps the historical mean of the two
        # orientation ratios: equal to the ratio of sums only for equal weights
        xip_c, _ = corr.compute_shear_shear(
            shear[0, 0], shear[0, 1], shear[1, 0], shear[1, 1], w_s[0], w_s[0], return_device=False)
        xip_e, _ = corr.vectorized_shear_shear(shear, np.stack((w_s[0], w_s[0])), return_device=False)
        np.testing.assert_allclose(xip_c, xip_e[1], rtol=0, atol=1e-13 * np.max(np.abs(xip_e)))
        return xip_v, xim_v

    def test_cpu_paths_match_reference(self):
        self.check_all_paths(self.tree)

    def test_row_space_input_is_bitwise_equal(self):
        corr = self.tree
        rows = lambda a: np.ascontiguousarray(a[..., corr.row_pix])
        a = corr.vectorized_shear_shear(self.shear, self.w_s, return_device=False)
        b = corr.vectorized_shear_shear(rows(self.shear), rows(self.w_s), return_device=False)
        for x, y in zip(a, b):
            self.assertTrue(np.array_equal(x, y))

    def test_emulated_gpu_paths_match_cpu(self):
        """T7 without a GPU: real GPU orchestrator branches, numpy kernels."""
        corr = self.tree
        cpu = {
            "ss": corr.get_full_tomo_shear(self.shear, self.w_s, flip_g1=True, return_device=False),
            "dd": corr.vectorized_density_density(self.dens, self.w_d, return_device=False),
            "ds": corr.vectorized_density_shear(
                self.dens, self.shear, self.w_d, self.w_s, return_device=False),
        }
        with emulated_gpu(corr) as calls:
            corr.compute_context.Q_inds_dev = None
            gpu = {
                "ss": corr.get_full_tomo_shear(self.shear, self.w_s, flip_g1=True, return_device=False),
                "dd": corr.vectorized_density_density(self.dens, self.w_d, return_device=False),
                "ds": corr.vectorized_density_shear(
                    self.dens, self.shear, self.w_d, self.w_s, return_device=False),
            }
            self.assertGreater(calls["xipm_tomo_vectorized_kernel"], 0)
        corr.compute_context.Q_inds_dev = None
        for key in cpu:
            a = cpu[key] if isinstance(cpu[key], tuple) else (cpu[key],)
            b = gpu[key] if isinstance(gpu[key], tuple) else (gpu[key],)
            for x, y in zip(a, b):
                np.testing.assert_allclose(y, x, rtol=0, atol=1e-13 * np.max(np.abs(x)))

    def test_differs_from_full_resolution(self):
        """Sanity: the treecode is a different estimator in the coarse bins
        and identical in the base-level bins."""
        full = make_corr(self.mask)
        full.preprocess()
        a = full.vectorized_shear_shear(self.shear, self.w_s, return_device=False)[0]
        b = self.tree.vectorized_shear_shear(self.shear, self.w_s, return_device=False)[0]
        self.assertTrue(np.array_equal(a[..., :2], b[..., :2]))
        self.assertFalse(np.allclose(a[..., 2:], b[..., 2:], rtol=1e-6, atol=0))


class TestDegradeOperator(TreecodeBase):
    """T3"""

    def expanded(self, blocks="pairs"):
        corr = self.tree
        rows = lambda a: clean(a, self.mask)[..., corr.row_pix]
        shear_rows, w_rows = corr._expand_shear_rows(rows(self.shear), rows(self.w_s), blocks)
        return shear_rows, w_rows

    def test_interior_cells_equal_ud_grade(self):
        """(a) + (d): RING input maps, NESTED parents."""
        corr = self.tree
        tree = corr._treecode
        shear_rows, w_rows = self.expanded()
        w = clean(self.w_s, self.mask)[0]
        g1 = clean(self.shear, self.mask)[0, 0]
        starts = tree.level_starts(first=corr.n_aperture_cells)
        checked = 0
        for level, nside_c in enumerate(tree.coarse_nsides):
            n_child = (NSIDE // nside_c) ** 2
            num_c = hp.ud_grade(w * g1, nside_c, order_in="RING", order_out="NESTED") * n_child
            w_c = hp.ud_grade(w, nside_c, order_in="RING", order_out="NESTED") * n_child
            members = tree.members_per_cell(level)
            for ip in range(corr.n_patches):
                pix = patch_pixels(corr, ip)
                shift = 2 * (int(np.log2(NSIDE)) - int(np.log2(nside_c)))
                cell_ids = np.unique(hp.ring2nest(NSIDE, pix) >> shift)
                a, b = tree.cell_offsets[level, ip], tree.cell_offsets[level, ip + 1]
                self.assertEqual(b - a, cell_ids.size)
                interior = members[a:b] == n_child
                self.assertTrue(interior.any())
                row = corr.n_active + starts[level] + np.arange(a, b)[interior]
                np.testing.assert_allclose(w_rows[0, row], w_c[cell_ids[interior]], rtol=1e-13)
                ok = w_c[cell_ids[interior]] > 0
                np.testing.assert_allclose(
                    shear_rows[0, 0, row][ok],
                    (num_c[cell_ids[interior]] / w_c[cell_ids[interior]])[ok],
                    rtol=1e-12, atol=1e-16,
                )
                checked += int(interior.sum())
        self.assertGreater(checked, 50)

    def test_empty_cells(self):
        """(b): W_I = 0 -> g_I = 0, no NaN/inf, no contribution."""
        corr = self.tree
        rows = lambda a: clean(a, self.mask)[..., corr.row_pix]
        w = rows(self.w_s).copy()
        pix = patch_pixels(corr, 0)
        lut = corr._global_to_row_lut()
        w[:, lut[pix[: pix.size // 2]]] = 0.0  # empty half of patch 0
        shear_rows, w_rows = corr._expand_shear_rows(rows(self.shear), w, "pairs")
        self.assertTrue(np.all(np.isfinite(shear_rows)))
        empty = w_rows[0, corr.n_active:] == 0
        self.assertTrue(empty.any())
        self.assertTrue(np.all(shear_rows[0, :, corr.n_active:][:, empty] == 0))
        full_w = np.zeros((2, NPIX)); full_w[:, corr.row_pix] = w
        xip, xim = corr.vectorized_shear_shear(clean(self.shear, self.mask), full_w, return_device=False)
        self.assertTrue(np.all(np.isfinite(xip)) and np.all(np.isfinite(xim)))
        # all weights zero -> zeros, not NaN
        xip0, _ = corr.vectorized_shear_shear(
            clean(self.shear, self.mask), np.zeros_like(full_w), return_device=False)
        self.assertTrue(np.all(xip0 == 0))

    def test_cells_contain_only_in_patch_members(self):
        """(c): membership vs an explicit disc query."""
        corr = self.tree
        tree = corr._treecode
        for ip in range(corr.n_patches):
            pix = set(patch_pixels(corr, ip).tolist())
            a, b = tree.cell_offsets[0, ip], tree.cell_offsets[0, ip + 1]
            lo, hi = tree.child_indptr[0][a], tree.child_indptr[0][b]
            members = tree.child_indices[0][lo:hi]
            self.assertEqual(set(members.tolist()), pix)
            self.assertEqual(members.size, len(pix))  # every pixel exactly once
            # a straddling cell exists and has fewer members than a full cell
            counts = tree.members_per_cell(0)[a:b]
            self.assertTrue(np.any(counts < (NSIDE // tree.coarse_nsides[0]) ** 2))
            # deeper levels: same member set, regrouped
            counts2 = tree.members_per_cell(1)[tree.cell_offsets[1, ip]: tree.cell_offsets[1, ip + 1]]
            self.assertEqual(int(counts2.sum()), len(pix))

    def test_centroids_are_binary_mask_centroids(self):
        corr = self.tree
        tree = corr._treecode
        for level, nside_c in enumerate(tree.coarse_nsides):
            a, b = tree.cell_offsets[level, 1], tree.cell_offsets[level, 2]
            _, ra, dec, _ = parents_and_centroids(patch_pixels(corr, 1), nside_c)
            np.testing.assert_allclose(tree.cell_ra[level][a:b], ra, rtol=0, atol=1e-13)
            np.testing.assert_allclose(tree.cell_dec[level][a:b], dec, rtol=0, atol=1e-13)


class TestExpansionSharing(unittest.TestCase):
    """The degrade runs once per public call; weight rows of frozen weights
    are computed once per weight set."""

    @classmethod
    def setUpClass(cls):
        cls.mask = make_mask()
        cls.shear, cls.dens, cls.w_s, cls.w_d = make_maps(cls.mask)
        cls.corr = make_corr(cls.mask, resolution_factor=2.0, aperture_nside=16)
        cls.corr.preprocess()

    def count_appends(self, fn):
        corr = self.corr
        calls = []
        original = corr._append_block
        corr._append_block = lambda X, blocks, use_cupy: (
            calls.append((blocks, X.shape[1])), original(X, blocks, use_cupy))[1]
        try:
            out = fn()
        finally:
            del corr._append_block
        return out, calls

    def test_one_degrade_per_call_and_frozen_weights(self):
        corr = self.corr
        nz = self.shear.shape[0]
        call = lambda w: corr.get_full_tomo_shear(self.shear, w, return_device=False)

        ref, calls = self.count_appends(lambda: call(self.w_s))
        # one pass for the weights, one for w*g1 and w*g2 -- shared by the
        # aperture and the 2PCF leaf
        self.assertEqual(calls, [("all", nz), ("all", 2 * nz)])
        self.assertIsNone(corr._expansion_memo)  # nothing survives the call

        frozen = self.w_s.copy()
        frozen.flags.writeable = False
        out1, calls1 = self.count_appends(lambda: call(frozen))
        out2, calls2 = self.count_appends(lambda: call(frozen))
        self.assertEqual(calls1, [("all", nz), ("all", 2 * nz)])
        self.assertEqual(calls2, [("all", 2 * nz)])  # weight rows cached
        for a, b, c in zip(ref, out1, out2):
            self.assertTrue(np.array_equal(a, b))
            self.assertTrue(np.array_equal(a, c))

        # a *writable* array that changes between calls is never stale
        w = self.w_s.copy()
        first = call(w)
        w[:, patch_pixels(corr, 0)[::2]] *= 3.0  # pixels inside patch 0
        second = call(w)
        self.assertFalse(np.array_equal(first[1], second[1]))
        fresh = make_corr(self.mask, resolution_factor=2.0, aperture_nside=16)
        fresh.preprocess()
        for a, b in zip(second, fresh.get_full_tomo_shear(self.shear, w, return_device=False)):
            self.assertTrue(np.array_equal(a, b))

    def test_reused_buffer_is_not_stale(self):
        """Same array object, new content (what PinnedMapPipeline does)."""
        corr = self.corr
        buf = np.ascontiguousarray(clean(self.shear, self.mask)[..., corr.row_pix])
        w = np.ascontiguousarray(clean(self.w_s, self.mask)[..., corr.row_pix])
        a = corr.get_full_tomo_shear(buf, w, return_device=False)
        buf *= -2.0
        b = corr.get_full_tomo_shear(buf, w, return_device=False)
        np.testing.assert_allclose(b[0], -2.0 * a[0], rtol=1e-12)
        np.testing.assert_allclose(b[1], 4.0 * a[1], rtol=1e-12)


class TestPairCountConservation(TreecodeBase):
    """T4"""

    def test_pair_counts_per_level(self):
        corr = self.tree
        tree = corr._treecode
        npix = corr.npix
        starts = tree.level_starts(first=corr.n_aperture_cells)
        edges = np.asarray(corr.binedges)
        for ip in range(corr.n_patches):
            pix = patch_pixels(corr, ip)
            bin_edges = np.concatenate(([0], np.cumsum(corr.bins[ip], dtype=np.int64)))
            for nside_b, b0, b1 in level_groups(corr.level_nside):
                if nside_b == NSIDE:
                    for b in range(b0, b1):
                        ids = corr.pair_inds[ip][:, bin_edges[b]: bin_edges[b + 1]]
                        self.assertTrue(np.all(ids < npix))
                    continue
                level = tree.level_index(nside_b)
                members = tree.members_per_cell(level)
                index, ra, dec, cvec = parents_and_centroids(pix, nside_b)
                # independent great-circle separations of the parents
                cosang = np.clip(cvec[index] @ cvec[index].T, -1, 1)
                sep = np.arccos(cosang)
                iu = np.triu_indices(pix.size, k=1)
                different_parent = index[iu[0]] != index[iu[1]]
                for b in range(b0, b1):
                    ids = corr.pair_inds[ip][:, bin_edges[b]: bin_edges[b + 1]].astype(np.int64)
                    self.assertTrue(np.all(ids >= npix + starts[level]))
                    self.assertTrue(np.all(ids < npix + starts[level + 1]))
                    cells = ids - npix - starts[level]
                    coarse_count = int(np.sum(members[cells[0]] * members[cells[1]]))
                    in_bin = (sep[iu] > edges[b]) & (sep[iu] < edges[b + 1]) & different_parent
                    self.assertEqual(coarse_count, int(in_bin.sum()), (ip, b))


class TestTreecodeIO(TreecodeBase):
    """T6"""

    def measure(self, corr):
        return corr.get_full_tomo_shear(self.shear, self.w_s, return_device=False)

    def test_roundtrip_and_slicing(self):
        ref = self.measure(self.tree)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pairs.h5")
            self.tree.save_pairs(path)
            with h5py.File(path, "r") as fp:
                self.assertEqual(int(fp.attrs["format_version"]), 3)
                # structurally unreadable for CosmoFuse <= 4.20
                self.assertNotIn("pair_inds", fp)
                self.assertNotIn("Q_inds", fp) if self.tree.aperture_nside else None
                self.assertEqual(fp["level_nside"][:].tolist(), EXPECTED_LEVELS)

            loaded = make_corr(self.mask, resolution_factor=2.0)
            loaded.load_pairs(path)
            self.assertEqual(loaded.level_table["nside"].tolist(), EXPECTED_LEVELS)
            for x, y in zip(self.measure(loaded), ref):
                self.assertTrue(np.array_equal(x, y))

            for start, stop in ((0, 1), (1, 3), (2, 3)):
                part = make_corr(self.mask, resolution_factor=2.0)
                part.load_pairs(path, start_ind=start, stop_ind=stop)
                self.assertEqual(part.n_patches, stop - start)
                for x, y in zip(self.measure(part), ref):
                    self.assertTrue(np.array_equal(x, y[..., start:stop, :] if y.ndim == 3 else y[:, start:stop]))

            # default constructor adopts the file's estimator, loudly
            adopt = make_corr(self.mask)
            with self.assertWarnsRegex(UserWarning, "static-treecode estimator"):
                adopt.load_pairs(path)
            self.assertEqual(adopt.resolution_factor, 2.0)
            for x, y in zip(self.measure(adopt), ref):
                self.assertTrue(np.array_equal(x, y))

            # explicit conflicting request raises
            with self.assertRaisesRegex(ValueError, "resolution_factor"):
                make_corr(self.mask, resolution_factor=8.0).load_pairs(path)

            # unknown future format raises
            with h5py.File(path, "a") as fp:
                fp.attrs["format_version"] = 99
            with self.assertRaisesRegex(ValueError, "format version 99"):
                make_corr(self.mask, resolution_factor=2.0).load_pairs(path)

    @pytest.mark.slow
    def test_full_resolution_files_stay_version_2(self):
        full = make_corr(self.mask)
        full.preprocess()
        ref = self.measure(full)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pairs.h5")
            full.save_pairs(path)
            with h5py.File(path, "r") as fp:
                self.assertEqual(int(fp.attrs["format_version"]), 2)
                self.assertIn("pair_inds", fp)
                self.assertNotIn("treecode", fp)
            again = make_corr(self.mask)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                again.load_pairs(path)
            self.assertIsNone(again.resolution_factor)
            for x, y in zip(self.measure(again), ref):
                self.assertTrue(np.array_equal(x, y))
            # a treecode request against a full-resolution file raises
            with self.assertRaisesRegex(ValueError, "resolution_factor"):
                make_corr(self.mask, resolution_factor=2.0).load_pairs(path)

            # files written before the level table existed load as full resolution
            with h5py.File(path, "a") as fp:
                del fp["level_nside"]
                del fp.attrs["aperture_nside"]
            old = make_corr(self.mask)
            old.load_pairs(path)
            self.assertTrue(np.all(old.level_nside == NSIDE))
            for x, y in zip(self.measure(old), ref):
                self.assertTrue(np.array_equal(x, y))


@pytest.mark.slow
class TestCoarseApertureLevel(unittest.TestCase):
    """aperture_nside: the aperture statistic of the degraded map."""

    @classmethod
    def setUpClass(cls):
        cls.mask = make_mask()
        cls.shear, cls.dens, cls.w_s, cls.w_d = make_maps(cls.mask)

    def degraded(self, nside_c):
        n_child = (NSIDE // nside_c) ** 2
        ud = lambda m: hp.ud_grade(m, nside_c) * n_child
        mask_c = ud(self.mask.astype(float)) > 0
        w_s, w_d = clean(self.w_s, self.mask), clean(self.w_d, self.mask)
        shear, dens = clean(self.shear, self.mask), clean(self.dens, self.mask)
        W_s = np.array([ud(w) for w in w_s])
        W_d = np.array([ud(w) for w in w_d])
        with np.errstate(invalid="ignore", divide="ignore"):
            sh = np.array([[np.where(W_s[z] > 0, ud(w_s[z] * shear[z, c]) / W_s[z], 0.0)
                            for c in range(2)] for z in range(shear.shape[0])])
            de = np.array([np.where(W_d[z] > 0, ud(w_d[z] * dens[z]) / W_d[z], 0.0)
                           for z in range(dens.shape[0])])
        return mask_c, sh, de, W_s, W_d

    def test_equals_aperture_statistics_of_degraded_maps(self):
        nside_c = 16
        mask_c, sh, de, W_s, W_d = self.degraded(nside_c)
        coarse = Correlation(
            nside_c, PHI_C, THETA_C, patch_size=700, theta_Q=200, mask=mask_c,
            device="cpu", map_precision="float64", rotation_precision="float64", **BIN_ARGS)
        coarse.calculate_pairs_M_a()
        ref_a = coarse._compute_tomo_aperture_shear(sh, W_s, return_device=False)
        ref_g = coarse._compute_tomo_aperture_density(de, W_d, return_device=False)

        for k in (None, 2.0):
            fine = make_corr(self.mask, resolution_factor=k, aperture_nside=nside_c)
            fine.preprocess()
            self.assertGreater(fine.n_aperture_cells, 0)
            self.assertEqual(fine.level_table["aperture_nside"], nside_c)
            M_a, xip, xim = fine.get_full_tomo_shear(self.shear, self.w_s, return_device=False)
            M_g, xig = fine.get_full_tomo_density(self.dens, self.w_d, return_device=False)
            np.testing.assert_allclose(M_a, ref_a, rtol=0, atol=1e-13 * np.max(np.abs(ref_a)))
            np.testing.assert_allclose(M_g, ref_g, rtol=0, atol=1e-13 * np.max(np.abs(ref_g)))
            # single-map API and fused kernel see the same aperture level
            one = fine.get_aperture_shear(
                self.shear[0, 0], self.shear[0, 1], self.w_s[0], return_device=False)
            np.testing.assert_allclose(one, ref_a[0], rtol=0, atol=1e-13 * np.max(np.abs(ref_a)))
            fused = fine.get_3x2pt_tomo(
                shear_maps=self.shear, density_maps=self.dens,
                weights=(self.w_s, self.w_d), return_device=False)
            np.testing.assert_allclose(fused[0], ref_a, rtol=0, atol=1e-13 * np.max(np.abs(ref_a)))
            np.testing.assert_allclose(fused[1], ref_g, rtol=0, atol=1e-13 * np.max(np.abs(ref_g)))
            # the 2PCF is not affected by the aperture level
            same = make_corr(self.mask, resolution_factor=k)
            same.preprocess()
            xip2, xim2 = same.vectorized_shear_shear(self.shear, self.w_s, return_device=False)
            np.testing.assert_allclose(xip, xip2, rtol=0, atol=1e-14 * np.max(np.abs(xip2)))

            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "pairs.h5")
                fine.save_pairs(path)
                again = make_corr(self.mask, resolution_factor=k, aperture_nside=nside_c)
                again.load_pairs(path, start_ind=1, stop_ind=3)
                M_a2, _, _ = again.get_full_tomo_shear(self.shear, self.w_s, return_device=False)
                self.assertTrue(np.array_equal(M_a2, M_a[:, 1:3]))
                with self.assertRaisesRegex(ValueError, "aperture_nside"):
                    make_corr(self.mask, resolution_factor=k, aperture_nside=32).load_pairs(path)


if __name__ == "__main__":
    unittest.main()
