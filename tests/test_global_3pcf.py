"""Gates of the footprint-wide three-point estimator (Global3PCF).

G1  multipoles = brute-force triplet sums with independent geometry, every
    component (shear Gamma_0..3, scalar zeta, mixed spin-0/spin-2 with
    conjugated legs), numerators and normalisations, <= 1e-12.
G2  invariance under the tiling and the centre chunking.
G3  input forms: row space == full sky (bitwise); flip_g1 == negated g1.
G4  pair files round-trip bitwise and refuse a different estimator.
G5  opening-angle binning: the Fourier coefficients and the convergence of
    the binned estimate to brute-force angle-binned sums.
G6  the GPU launch wrapper + numpy twin of gpu_3pcf_multipoles == CPU.
"""

import unittest
import warnings

import healpy as hp
import numpy as np
import pytest

from CosmoFuse import Component, Global3PCF, ThreePointMultipoles
from CosmoFuse.threepoint_kernels import build_cupy_3pcf_multipoles_kernel

from .cuda_emulation import LAUNCH_LOG, EmulatedCupyModule
from .threepoint_reference import brute_force, position_angle, _frame

NSIDE = 16
N_MAX = 3


def _setup():
    npix = hp.nside2npix(NSIDE)
    disc = hp.query_disc(NSIDE, hp.ang2vec(np.radians(70), np.radians(40)), np.radians(16))
    mask = np.zeros(npix, bool)
    mask[disc] = True
    rng = np.random.default_rng(3)
    g = rng.normal(size=(2, 2, npix))
    w = rng.uniform(0.5, 1.5, size=(2, npix))
    d = rng.normal(size=(2, npix))
    return mask, g, w, d


def _estimator(mask, **kw):
    kw.setdefault("tile_nside", 2)
    kw.setdefault("n_max", N_MAX)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = Global3PCF(
            NSIDE, mask=mask, nbins=3, theta_min=240, theta_max=900, device="cpu", **kw
        )
    est.preprocess()
    return est


def _rel(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


class TestBruteForce(unittest.TestCase):
    """G1"""

    @classmethod
    def setUpClass(cls):
        cls.mask, cls.g, cls.w, cls.d = _setup()
        cls.pix = np.flatnonzero(cls.mask)
        cls.est = _estimator(cls.mask, centre_chunk=17)

    def _check(self, res, fields):
        comps = [(c.label, c.fields, c.conj) for c in res.components]
        num, norm = brute_force(NSIDE, self.pix, fields, comps, self.est.binedges, N_MAX)
        for c in res.components:
            with self.subTest(component=c.label):
                self.assertLess(_rel(res.num[c.label], num[c.label]), 1e-12)
                self.assertLess(_rel(res.norm[c.label], norm[c.label]), 1e-12)

    def test_shear(self):
        res = self.est.shear(self.g, self.w)
        self.assertEqual([c.label for c in res.components], ["gamma0", "gamma1", "gamma2", "gamma3"])
        self.assertEqual(res.num["gamma0"].shape, (2 * N_MAX + 1, 2, 2, 2, 3, 3))
        self._check(res, {"g": (self.g[:, :, self.pix], self.w[:, self.pix], 2)})

    def test_density(self):
        res = self.est.density(self.d, self.w)
        self._check(res, {"d": (self.d[:, self.pix], self.w[:, self.pix], 0)})

    def test_mixed_fields(self):
        comps = [
            Component("dgg", ("d", "g", "g")),
            Component("g*dd", ("g", "d", "d"), (True, False, False)),
            Component("dgg*", ("d", "g", "g"), (False, False, True)),
            Component("gg*d", ("g", "g", "d"), (False, True, False)),
        ]
        w2 = self.w[::-1].copy()  # different weights per field
        res = self.est.measure({"d": (self.d, self.w, 0), "g": (self.g, w2, 2)}, comps)
        self._check(
            res,
            {
                "d": (self.d[:, self.pix], self.w[:, self.pix], 0),
                "g": (self.g[:, :, self.pix], w2[:, self.pix], 2),
            },
        )

    def test_position_angle_convention(self):
        """The pair finder's full angles equal the tangent-frame angles."""
        est = self.est
        P, Ee, Nn = _frame(NSIDE, est.map_inds)
        for c in range(0, est.centre_rows.size, 7):
            i = int(est.centre_rows[c])
            lo, hi = est.pair_offsets[c * est.nbins], est.pair_offsets[(c + 1) * est.nbins]
            for p in range(lo, hi):
                j = int(est.nbr_rows[p])
                ref_c = position_angle(P[i], Ee[i], Nn[i], P[j])
                ref_n = position_angle(P[j], Ee[j], Nn[j], P[i])
                self.assertAlmostEqual(np.angle(np.exp(1j * (est.phi_c[p] - ref_c))), 0.0, places=10)
                self.assertAlmostEqual(np.angle(np.exp(1j * (est.phi_n[p] - ref_n))), 0.0, places=10)


class TestInvariance(unittest.TestCase):
    """G2 + G3"""

    @classmethod
    def setUpClass(cls):
        cls.mask, cls.g, cls.w, cls.d = _setup()
        cls.ref = _estimator(cls.mask, tile_nside=1, centre_chunk=4096).shear(cls.g, cls.w)

    def test_tiles_and_chunks(self):
        for tile_nside, chunk in ((2, 5), (4, 1000), (8, 33)):
            with self.subTest(tile_nside=tile_nside, chunk=chunk):
                est = _estimator(self.mask, tile_nside=tile_nside, centre_chunk=chunk)
                res = est.shear(self.g, self.w)
                for label in res.num:
                    self.assertLess(_rel(res.num[label], self.ref.num[label]), 1e-13)
                    self.assertLess(_rel(res.norm[label], self.ref.norm[label]), 1e-13)

    def test_row_space_input(self):
        est = _estimator(self.mask)
        full = est.shear(self.g, self.w)
        rows = est.shear(est.to_row_space(self.g), est.to_row_space(self.w))
        for label in full.num:
            np.testing.assert_array_equal(full.num[label], rows.num[label])

    def test_flip_g1(self):
        est = _estimator(self.mask)
        flipped = est.shear(self.g, self.w, flip_g1=True)
        g = self.g.copy()
        g[:, 0] *= -1
        ref = est.shear(g, self.w)
        for label in ref.num:
            np.testing.assert_array_equal(flipped.num[label], ref.num[label])

    def test_norm_cache_follows_weights(self):
        est = _estimator(self.mask)
        a = est.shear(self.g, self.w)
        b = est.shear(self.g, self.w * 2.0)
        np.testing.assert_allclose(b.norm["gamma0"], 8.0 * a.norm["gamma0"], rtol=1e-13)


class TestPairFiles(unittest.TestCase):
    """G4"""

    def test_round_trip(self):
        import tempfile, os

        mask, g, w, _ = _setup()
        est = _estimator(mask)
        ref = est.shear(g, w)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p3.h5")
            est.save_pairs(path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                other = Global3PCF(NSIDE, mask=mask, nbins=3, theta_min=240, theta_max=900,
                                   n_max=N_MAX, tile_nside=2, device="cpu")
            other.load_pairs(path)
            res = other.shear(g, w)
            for label in ref.num:
                np.testing.assert_array_equal(res.num[label], ref.num[label])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wrong = Global3PCF(NSIDE, mask=mask, nbins=4, theta_min=240, theta_max=900,
                                   tile_nside=2, device="cpu")
            with self.assertRaises(ValueError):
                wrong.load_pairs(path)


class TestBinning(unittest.TestCase):
    """G5"""

    def test_coefficients(self):
        n = np.arange(-5, 6)
        edges = np.array([-2.0, 0.3, 1.7])
        c = ThreePointMultipoles._phi_coefficients(n, edges)
        phi = np.linspace(edges[0], edges[1], 200001)
        for i, m in enumerate(n):
            ref = np.trapezoid(np.exp(-1j * m * phi), phi) / (2 * np.pi)
            self.assertAlmostEqual(c[0, i], ref, places=9)

    def test_converges_to_binned_sums(self):
        """Synthetic triangles: the binned series approaches the direct
        angle-binned ratio as n_max grows (away from the bin edges)."""
        rng = np.random.default_rng(0)
        edges = np.linspace(-np.pi, np.pi, 5)
        centres = 0.5 * (edges[:-1] + edges[1:])
        phi_t = (centres[rng.integers(0, 4, 400)] + rng.uniform(-0.6, 0.6, 400))
        vals = rng.normal(size=400) + 1j * rng.normal(size=400)
        wts = rng.uniform(0.5, 1.5, size=400)
        k = np.digitize(phi_t, edges) - 1
        direct = np.array([np.sum(vals[k == i]) / np.sum(wts[k == i]) for i in range(4)])
        errs = []
        for n_max in (20, 80, 320):
            n = np.arange(-n_max, n_max + 1)
            num = np.sum(vals[None] * np.exp(-1j * n[:, None] * phi_t[None]), axis=1)
            den = np.sum(wts[None] * np.exp(-1j * n[:, None] * phi_t[None]), axis=1)
            shape = (n.size, 1, 1, 1, 1, 1)
            res = ThreePointMultipoles(
                n_max, np.array([1.0, 2.0]), (Component("x", ("a", "a", "a")),), {"a": 0},
                {"x": num.reshape(shape)}, {"x": den.reshape(shape)},
            )
            errs.append(np.abs(res.binned(edges)["x"].ravel() - direct).max())
        self.assertLess(errs[2], errs[0])
        self.assertLess(errs[2], 0.02 * np.abs(direct).max())

    def test_centroid_is_a_phase(self):
        mask, g, w, d = _setup()
        est = _estimator(mask)
        res = est.shear(g, w)
        edges = np.linspace(-np.pi, np.pi, 9)
        b = res.binned(edges)
        cen = res.to_centroid(b, edges)
        for label in b:
            np.testing.assert_allclose(np.abs(cen[label]), np.abs(b[label]), rtol=1e-12, atol=0)
        dres = est.density(d, w)
        db = dres.binned(edges)
        np.testing.assert_array_equal(dres.to_centroid(db, edges)["zeta"], db["zeta"])

    def test_sum_of_results(self):
        mask, g, w, _ = _setup()
        est = _estimator(mask)
        a = est.shear(g, w)
        s = a + a
        np.testing.assert_array_equal(s.num["gamma1"], 2 * a.num["gamma1"])


class TestEmulatedGPU(unittest.TestCase):
    """G6: the real launch wrapper against the numpy twin of the kernel."""

    def test_matches_cpu(self):
        mask, g, w, d = _setup()
        for acc in ("float64", "same"):
            with self.subTest(accumulation_precision=acc):
                est = _estimator(mask, map_precision="float32", rotation_precision="float32",
                                 accumulation_precision=acc)
                cpu = est.shear(g, w)
                cpu_d = est.density(d, w)
                be = est.backend
                saved = (be.name, be.kernel_3pcf_multipoles, be.to_device, be.to_numpy)
                be.name = "cupy"
                be.kernel_3pcf_multipoles = build_cupy_3pcf_multipoles_kernel(EmulatedCupyModule)
                be.to_device = lambda a, stream=None: np.asarray(a)
                be.to_numpy = lambda a, stream=None: np.asarray(a)
                est._norm_cache.clear()
                LAUNCH_LOG.clear()
                try:
                    gpu = est.shear(g, w)
                    gpu_d = est.density(d, w)
                finally:
                    be.name, be.kernel_3pcf_multipoles, be.to_device, be.to_numpy = saved
                self.assertIn("gpu_3pcf_multipoles", LAUNCH_LOG)
                tol = 1e-12 if acc == "float64" else 1e-5
                for label in cpu.num:
                    self.assertLess(_rel(gpu.num[label], cpu.num[label]), tol)
                    self.assertLess(_rel(gpu.norm[label], cpu.norm[label]), tol)
                self.assertLess(_rel(gpu_d.num["zeta"], cpu_d.num["zeta"]), tol)


@pytest.mark.gpu
class TestRealGPU(unittest.TestCase):
    def test_gpu_matches_cpu(self):
        mask, g, w, d = _setup()
        for mp, acc, tol in (("float64", "float64", 1e-12), ("float32", "float64", 1e-5),
                             ("float32", "same", 1e-4)):
            with self.subTest(map_precision=mp, accumulation_precision=acc):
                kw = dict(map_precision=mp, rotation_precision=mp, accumulation_precision=acc)
                cpu = _estimator(mask, **kw)
                gpu = Global3PCF(NSIDE, mask=mask, nbins=3, theta_min=240, theta_max=900,
                                 n_max=N_MAX, tile_nside=2, device="gpu", **kw)
                gpu.preprocess()
                for fn, args in (("shear", (g, w)), ("density", (d, w))):
                    a = getattr(cpu, fn)(*args)
                    b = getattr(gpu, fn)(*args)
                    for label in a.num:
                        self.assertLess(_rel(b.num[label], a.num[label]), tol)
                        self.assertLess(_rel(b.norm[label], a.norm[label]), tol)
