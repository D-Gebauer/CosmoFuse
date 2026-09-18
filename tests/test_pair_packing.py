"""Payload packing (pack_pairs=True): 8 bytes per pair on the device.

* codec: angle round trip within pi/65536, local indices reconstruct the rows
* the CPU backend and the (emulated) packed GPU kernel measure the *same*
  quantised estimator (<= 1e-13)
* the quantisation errors average out: they are ~1e-5 of the per-patch
  scatter, unbiased, and do not accumulate in patch averages or zetas
"""

import unittest

import healpy as hp
import numpy as np

from CosmoFuse import Correlation, calculate_all_zetas
from CosmoFuse import packing

from .cuda_emulation import LAUNCH_LOG
from .test_gpu_kernel_emulation import emulated_gpu

NSIDE = 64
NPIX = hp.nside2npix(NSIDE)


def make_setup(pack, k=2.0, n_side_centers=4):
    theta, phi = hp.pix2ang(NSIDE, np.arange(NPIX))
    mask = (np.degrees(phi) < 200.0) & (np.abs(90 - np.degrees(theta)) < 55)
    rng = np.random.default_rng(1)
    mask[rng.choice(NPIX, NPIX // 30, replace=False)] = False
    corr = Correlation.from_mask(
        NSIDE, mask, n_side_centers, patch_size=600, theta_Q=200, f_mask=0.3,
        nbins=5, theta_min=120, theta_max=1100, device="cpu",
        map_precision="float64", rotation_precision="float64",
        resolution_factor=k, pack_pairs=pack,
    )
    corr.preprocess()
    return corr, mask


class TestCodec(unittest.TestCase):
    def test_angle_round_trip(self):
        rng = np.random.default_rng(0)
        alpha = rng.uniform(-np.pi, np.pi, 200000)
        alpha[:4] = [np.pi, -np.pi, 0.0, -1e-9]
        z = np.exp(1j * alpha)
        codes = packing.encode_angles(z)
        self.assertEqual(codes.dtype, np.uint16)
        back = packing.decode_angles(codes, np.complex128)
        np.testing.assert_allclose(np.abs(back), 1.0, rtol=0, atol=1e-15)
        err = np.angle(back * np.conj(z))
        self.assertLessEqual(np.max(np.abs(err)), np.pi / 65536 * (1 + 1e-9))
        # round-to-nearest: unbiased, uniform error
        self.assertLess(abs(err.mean()), 4 * (2 * np.pi / 65536) / np.sqrt(12 * err.size))
        self.assertEqual(packing.decode_angles(codes, np.complex64).dtype, np.complex64)

    def test_pack_patch_reconstructs_rows(self):
        rng = np.random.default_rng(2)
        counts = np.array([5, 0, 7, 3])
        rows = np.concatenate(
            [rng.integers(100, 140, (2, 12)), rng.integers(9000, 9010, (2, 3))], axis=1)
        exp = np.exp(1j * rng.uniform(-3, 3, (2, 15)))
        groups = [(64, 0, 3), (32, 3, 4)]
        packed, blocks, block_of_bin = packing.pack_patch(rows, exp, counts, groups)
        self.assertEqual(packed.shape, (15, 4))
        self.assertEqual(block_of_bin.tolist(), [0, 0, 0, 1])
        np.testing.assert_array_equal(blocks[0][packed[:12, :2].T], rows[:, :12])
        np.testing.assert_array_equal(blocks[1][packed[12:, :2].T], rows[:, 12:])
        for blk in blocks:
            self.assertTrue(np.all(np.diff(blk) > 0))

    def test_too_many_local_rows_raise(self):
        old = packing.MAX_LOCAL_ROWS
        packing.MAX_LOCAL_ROWS = 4
        try:
            with self.assertRaisesRegex(ValueError, "pack_pairs=False"):
                packing.pack_patch(
                    np.arange(20).reshape(2, 10), np.ones((2, 10), complex),
                    np.array([10]), [(64, 0, 1)])
        finally:
            packing.MAX_LOCAL_ROWS = old


class TestPackedMeasurement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exact, cls.mask = make_setup(pack=False)
        cls.packed, _ = make_setup(pack=True)
        rng = np.random.default_rng(3)
        nz = 2          # pure-Python kernels under NUMBA_DISABLE_JIT: keep it small
        cls.nmaps = 4
        cls.shear = rng.normal(size=(cls.nmaps, nz, 2, NPIX)) * 0.3   # shape-noise like
        cls.w = rng.uniform(0.2, 2.0, size=(nz, NPIX))

    def test_geometry(self):
        a, b = self.exact, self.packed
        self.assertGreater(a.n_patches, 20)
        ctx = b.compute_context
        self.assertEqual(ctx.packed_pairs_dev.shape, (b.ntotpairs, 4))
        self.assertEqual(ctx.packed_pairs_dev.dtype, np.uint16)
        # local indices + block base + gather reproduce the device rows
        offsets = np.asarray(b.tot_bins_reduceat_dev)
        block = np.repeat(np.arange(offsets.size - 1), np.diff(offsets))
        rows = ctx.packed_perm_dev[
            ctx.packed_row_base_dev[block][:, None] + ctx.packed_pairs_dev[:, :2]]
        np.testing.assert_array_equal(rows.T, np.asarray(a.inds_dev))
        # CPU kernels use the same quantised rotations
        np.testing.assert_array_equal(
            np.asarray(b.exp2phi_dev),
            packing.decode_angles(ctx.packed_pairs_dev[:, 2:].T, np.complex128))
        err = np.angle(np.asarray(b.exp2phi_dev) * np.conj(np.asarray(a.exp2phi_dev)))
        self.assertLessEqual(np.max(np.abs(err)), np.pi / 65536 * (1 + 1e-6))
        self.assertIsNone(a.compute_context.packed_pairs_dev)

    def test_emulated_gpu_packed_kernel_equals_cpu(self):
        corr = self.packed
        cpu = corr.get_full_tomo_shear(self.shear[0], self.w, flip_g1=True, return_device=False)
        with emulated_gpu(corr) as calls:
            corr.compute_context.Q_inds_dev = None
            del LAUNCH_LOG[:]
            gpu = corr.get_full_tomo_shear(self.shear[0], self.w, flip_g1=True, return_device=False)
            self.assertEqual(calls["xipm_tomo_packed_kernel"], 1)
            self.assertIn("gpu_tiled_packed_reduce_xipm", LAUNCH_LOG)
            self.assertNotIn("gpu_tiled_tomo_reduce_xipm", LAUNCH_LOG)
            self.assertNotIn("gpu_fused_tomo_reduce_xipm", LAUNCH_LOG)
        corr.compute_context.Q_inds_dev = None
        for x, y in zip(cpu, gpu):
            np.testing.assert_allclose(y, x, rtol=0, atol=1e-13 * np.max(np.abs(x)))

    def test_emulated_gpu_packed_density_ggl_3x2pt_equal_cpu(self):
        """Every tomographic method runs on the packed geometry (auto-only
        and subset combination requests included)."""
        corr = self.packed
        dens = np.asarray(self.shear[1, :, 0]) * 3.0
        wd = np.asarray(self.w[::-1])
        calls_ref = dict(
            dd=lambda: corr.get_full_tomo_density(dens, wd, return_device=False),
            dd_auto=lambda: corr.vectorized_density_density(
                dens, wd, gc_auto_correlations_only=True, return_device=False),
            ggl=lambda: corr.get_full_tomo_ggl(
                dens, self.shear[0], wd, self.w, return_N_ap=True, return_device=False),
            ggl_sub=lambda: corr.vectorized_density_shear(
                dens, self.shear[0], wd, self.w, ggl_bin_combinations=[(1, 0)],
                return_device=False),
            fused=lambda: corr.get_3x2pt_tomo(
                shear_maps=self.shear[0], density_maps=dens,
                weights={"shear": self.w, "density": wd}, flip_g1=True, return_device=False),
        )
        cpu = {k: f() for k, f in calls_ref.items()}
        with emulated_gpu(corr) as calls:
            corr.compute_context.Q_inds_dev = None
            del LAUNCH_LOG[:]
            gpu = {k: f() for k, f in calls_ref.items()}
        corr.compute_context.Q_inds_dev = None
        self.assertEqual(calls["kernel_density_density_tomo_packed"], 3)
        self.assertEqual(calls["kernel_density_shear_tomo_packed"], 3)
        self.assertEqual(calls["xipm_tomo_packed_kernel"], 1)
        self.assertEqual(calls["kernel_density_density_tomo_vectorized"], 0)
        self.assertEqual(calls["kernel_density_shear_tomo_vectorized"], 0)
        self.assertEqual(calls["xipm_tomo_vectorized_kernel"], 0)
        for name in ("gpu_tiled_packed_reduce_dd", "gpu_tiled_packed_reduce_ds",
                     "gpu_tiled_packed_reduce_xipm", "gpu_3x2pt_tomo_aperture"):
            self.assertIn(name, LAUNCH_LOG)
        for key in cpu:
            for x, y in zip(cpu[key], gpu[key]):
                x, y = np.asarray(x), np.asarray(y)
                np.testing.assert_allclose(y, x, rtol=0, atol=1e-13 * np.max(np.abs(x)), err_msg=key)

    def test_gpu_without_unpacked_geometry(self):
        """On a real GPU the 24 B geometry is not uploaded: no tomographic
        method may need it; the single-map compute_* methods refuse clearly."""
        corr, _ = make_setup(pack=True)
        dens = np.ones((2, NPIX)) + 0.1 * self.shear[1, :, 0]
        ref = dict(
            ss=corr.vectorized_shear_shear(self.shear[0], self.w, return_device=False),
            dd=corr.vectorized_density_density(dens, self.w, return_device=False),
            ds=corr.vectorized_density_shear(dens, self.shear[0], self.w, self.w, return_device=False),
            fused=corr.get_3x2pt_tomo(shear_maps=self.shear[0], density_maps=dens, return_device=False),
        )
        with emulated_gpu(corr):
            corr.inds_dev = None
            corr.exp2phi_dev = None
            corr.compute_context.inds_i_dev = corr.compute_context.inds_j_dev = None
            corr.compute_context.Q_inds_dev = None
            got = dict(
                ss=corr.vectorized_shear_shear(self.shear[0], self.w, return_device=False),
                dd=corr.vectorized_density_density(dens, self.w, return_device=False),
                ds=corr.vectorized_density_shear(dens, self.shear[0], self.w, self.w, return_device=False),
                fused=corr.get_3x2pt_tomo(shear_maps=self.shear[0], density_maps=dens, return_device=False),
            )
            as_list = lambda r: list(r) if isinstance(r, tuple) else [r]
            for key in ref:
                for x, y in zip(as_list(ref[key]), as_list(got[key])):
                    x, y = np.asarray(x), np.asarray(y)
                    np.testing.assert_allclose(y, x, rtol=0, atol=1e-13 * np.max(np.abs(x)), err_msg=key)
            with self.assertRaisesRegex(NotImplementedError, "pack_pairs=False"):
                corr.compute_shear_shear(
                    self.shear[0, 0, 0], self.shear[0, 0, 1], self.shear[0, 0, 0],
                    self.shear[0, 0, 1], self.w[0], self.w[0])

    def test_quantisation_errors_average_out(self):
        exact = [self.exact.get_full_tomo_shear(m, self.w, return_device=False) for m in self.shear]
        packed = [self.packed.get_full_tomo_shear(m, self.w, return_device=False) for m in self.shear]
        for idx, name in ((1, "xi+"), (2, "xi-")):
            e = np.array([r[idx] for r in exact])          # (maps, comb, patch, bin)
            p = np.array([r[idx] for r in packed])
            self.assertFalse(np.array_equal(e, p))         # it *is* a different rounding
            sigma = e.std(axis=(0, 2), keepdims=True)      # scatter over maps x patches
            z = (p - e) / sigma
            # every single estimate moves by a tiny fraction of its own noise
            self.assertLess(np.max(np.abs(z)), 3e-4, name)
            self.assertLess(z.std(), 5e-5, name)
            # unbiased: the mean shift is consistent with zero ...
            n = z.size
            self.assertLess(abs(z.mean()), 5 * z.std() / np.sqrt(n), name)
            # ... and errors do not accumulate in averages over patches:
            # the shift of the patch mean shrinks like 1/sqrt(n_patches)
            shift_mean = (p.mean(axis=2) - e.mean(axis=2)) / (sigma[:, :, 0] / np.sqrt(e.shape[2]))
            self.assertLess(np.max(np.abs(shift_mean)), 3e-4, name)
        # aperture mass does not use pair rotations at all
        for a, b in zip(exact, packed):
            self.assertTrue(np.array_equal(a[0], b[0]))
        # i3PCF level
        ze = calculate_all_zetas(M_a=np.array([r[0] for r in exact]),
                                 xi_p=np.array([r[1] for r in exact]),
                                 xi_m=np.array([r[2] for r in exact]))
        zp = calculate_all_zetas(M_a=np.array([r[0] for r in packed]),
                                 xi_p=np.array([r[1] for r in packed]),
                                 xi_m=np.array([r[2] for r in packed]))
        for key in ze:
            if ze[key] is None:
                continue
            scatter = np.std(ze[key], axis=0)
            self.assertLess(np.max(np.abs(zp[key] - ze[key]) / scatter), 3e-4, key)

    def test_full_resolution_and_pickle(self):
        import pickle

        corr, _ = make_setup(pack=True, k=None, n_side_centers=2)
        ref, _ = make_setup(pack=False, k=None, n_side_centers=2)
        a = corr.vectorized_shear_shear(self.shear[0], self.w, return_device=False)
        b = ref.vectorized_shear_shear(self.shear[0], self.w, return_device=False)
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, rtol=0, atol=2e-4 * np.max(np.abs(y)))
        clone = pickle.loads(pickle.dumps(corr))
        self.assertTrue(clone.pack_pairs)
        clone.prepare()
        for x, y in zip(a, clone.vectorized_shear_shear(self.shear[0], self.w, return_device=False)):
            self.assertTrue(np.array_equal(x, y))


if __name__ == "__main__":
    unittest.main()
