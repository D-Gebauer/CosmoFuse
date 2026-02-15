import pickle
import unittest
import numpy as np
import sys
import warnings
from pathlib import Path
from unittest.mock import patch

# Add src to path for testing
sys.path.insert(1, str(Path(__file__).parent.parent / "src"))

from CosmoFuse.correlations import Correlation


class TestUnifiedCorrelation(unittest.TestCase):
    def setUp(self):
        self.nside = 32
        self.phi_center = np.array([0.0])
        self.theta_center = np.array([np.pi / 2])
        self.nbins = 2
        self.theta_min = 10.0
        self.theta_max = 100.0
        self.patch_size = 60.0
        self.theta_Q = 30.0

    def test_cpu_device(self):
        corr = Correlation(
            nside=self.nside,
            phi_center=self.phi_center,
            theta_center=self.theta_center,
            nbins=self.nbins,
            device='cpu'
        )
        self.assertEqual(corr.device, 'cpu')
        self.assertEqual(corr.backend.name, 'numpy')

    def test_auto_device_fallback(self):
        # Without cupy, auto should fallback to cpu
        with patch.dict('sys.modules', {'cupy': None}):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                corr = Correlation(
                    nside=self.nside,
                    phi_center=self.phi_center,
                    theta_center=self.theta_center,
                    nbins=self.nbins,
                    device='auto'
                )
                # Depending on implementation, device attribute might remain 'auto' but backend is numpy
                self.assertEqual(corr.backend.name, 'numpy')

    def test_gpu_device_missing_cupy(self):
        # Should raise ImportError or warning+fallback?
        # My implementation raises ImportError if explicit 'gpu' requested and no cupy.
        with patch.dict('sys.modules', {'cupy': None}):
             with self.assertRaises(ImportError):
                 Correlation(
                    nside=self.nside,
                    phi_center=self.phi_center,
                    theta_center=self.theta_center,
                    nbins=self.nbins,
                    device='gpu'
                )

    def test_xipm_scalar_sumofweights_and_normalization(self):
        corr = Correlation(
            nside=self.nside,
            phi_center=self.phi_center,
            theta_center=self.theta_center,
            nbins=self.nbins,
            device="cpu",
        )

        n_pairs = 6
        corr.pair_inds = [np.zeros((2, n_pairs), dtype=np.uint32)]
        corr.pair_exp2phi = [np.ones((2, n_pairs), dtype=np.complex128)]
        corr.bins = [np.array([3, 3], dtype=np.uint32)]
        corr.prepare()

        npix = 12 * self.nside**2
        g11 = np.ones(npix, dtype=np.float64)
        g21 = np.ones(npix, dtype=np.float64)
        g12 = np.ones(npix, dtype=np.float64)
        g22 = np.ones(npix, dtype=np.float64)
        w1 = np.ones(npix, dtype=np.float64)
        w2 = np.ones(npix, dtype=np.float64)

        xip_zero, xim_zero = corr.xipm(
            g11, g21, g12, g22, w1, w2, sumofweights=0.0
        )
        self.assertTrue(np.all(xip_zero == 0))
        self.assertTrue(np.all(xim_zero == 0))

        xip_one, xim_one = corr.xipm(
            g11, g21, g12, g22, w1, w2, sumofweights=1.0
        )
        self.assertTrue(np.all(np.isfinite(xip_one)))
        self.assertTrue(np.all(np.isfinite(xim_one)))

        sumofweights_dev = corr._normalize_xipm_sumofweights(1.0)
        self.assertEqual(np.ndim(corr.backend.to_numpy(sumofweights_dev)), 0)

        with self.assertRaises(ValueError):
            corr._normalize_xipm_sumofweights(np.ones(3))

    def test_get_full_tomo_sumofweights_validation(self):
        corr = Correlation(
            nside=self.nside,
            phi_center=self.phi_center,
            theta_center=self.theta_center,
            nbins=self.nbins,
            device="cpu",
        )

        n_pairs = 6
        corr.pair_inds = [np.zeros((2, n_pairs), dtype=np.uint32)]
        corr.pair_exp2phi = [np.ones((2, n_pairs), dtype=np.complex128)]
        corr.bins = [np.array([3, 3], dtype=np.uint32)]
        corr.prepare()

        corr.Q_inds = [np.array([0, 1, 2], dtype=np.uint32)]
        corr.Q_cos = [np.array([1.0, 1.0, 1.0], dtype=np.float64)]
        corr.Q_sin = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]
        corr.Q_val = [np.array([1.0, 1.0, 1.0], dtype=np.float64)]
        corr.Q_patch_area = [3.0]

        npix = 12 * self.nside**2
        shear_maps = np.ones((1, 2, npix), dtype=np.float64)
        w = np.ones((1, npix), dtype=np.float64)

        with self.assertRaises(ValueError):
            corr.get_full_tomo(shear_maps, w, sumofweights=1.0)

        with self.assertRaises(ValueError):
            corr.get_full_tomo(shear_maps, w, sumofweights=np.ones((1, 1, 2)))

    def test_prepare_on_demand_paths(self):
        corr = Correlation(
            nside=self.nside,
            phi_center=self.phi_center,
            theta_center=self.theta_center,
            nbins=self.nbins,
            device="cpu",
        )

        n_pairs = 6
        corr.pair_inds = [np.zeros((2, n_pairs), dtype=np.uint32)]
        corr.pair_exp2phi = [np.ones((2, n_pairs), dtype=np.complex128)]
        corr.bins = [np.array([3, 3], dtype=np.uint32)]

        npix = 12 * self.nside**2
        g11 = np.ones(npix, dtype=np.float64)
        g21 = np.ones(npix, dtype=np.float64)
        g12 = np.ones(npix, dtype=np.float64)
        g22 = np.ones(npix, dtype=np.float64)
        w1 = np.ones(npix, dtype=np.float64)
        w2 = np.ones(npix, dtype=np.float64)

        # xipm should trigger prepare() when inds_dev is None.
        xip, xim = corr.xipm(g11, g21, g12, g22, w1, w2, sumofweights=1.0)
        self.assertEqual(xip.shape, (1, self.nbins))
        self.assertEqual(xim.shape, (1, self.nbins))

        corr2 = Correlation(
            nside=self.nside,
            phi_center=self.phi_center,
            theta_center=self.theta_center,
            nbins=self.nbins,
            device="cpu",
        )
        corr2.pair_inds = [np.zeros((2, n_pairs), dtype=np.uint32)]
        corr2.pair_exp2phi = [np.ones((2, n_pairs), dtype=np.complex128)]
        corr2.bins = [np.array([3, 3], dtype=np.uint32)]

        w1_dev = corr2.backend.to_device(w1)
        w2_dev = corr2.backend.to_device(w2)
        sum_w = corr2._compute_xipm_sumofweights(w1_dev, w2_dev)
        self.assertEqual(sum_w.shape[0], corr2.n_patches * corr2.nbins)

        corr3 = Correlation(
            nside=self.nside,
            phi_center=self.phi_center,
            theta_center=self.theta_center,
            nbins=self.nbins,
            device="cpu",
        )
        corr3.pair_inds = [np.zeros((2, n_pairs), dtype=np.uint32)]
        corr3.pair_exp2phi = [np.ones((2, n_pairs), dtype=np.complex128)]
        corr3.bins = [np.array([3, 3], dtype=np.uint32)]

        w_dev = corr3.backend.to_device(np.ones((1, npix), dtype=np.float64))
        sum_w_tomo = corr3._compute_tomo_sumofweights(w_dev, 1, 1)
        self.assertEqual(sum_w_tomo.shape, (2, 1, corr3.n_patches * corr3.nbins))

    def test_precision_applied_to_internal_arrays(self):
        corr = Correlation(
            nside=self.nside,
            phi_center=self.phi_center,
            theta_center=self.theta_center,
            nbins=self.nbins,
            device='cpu',
            map_precision='float32',
            rotation_precision='float32',
            index_precision='uint64'
        )

        n_pairs = 6
        corr.pair_inds = [np.zeros((2, n_pairs), dtype=np.uint64)]
        corr.pair_exp2phi = [np.ones((2, n_pairs), dtype=np.complex64)]
        corr.bins = [np.array([3, 3], dtype=np.uint64)]

        corr.prepare()
        self.assertEqual(corr.inds_dev.dtype, np.uint64)
        self.assertEqual(corr.exp2phi_dev.dtype, np.complex64)
        self.assertEqual(corr.bins_dev.dtype, np.uint64)
        self.assertEqual(corr.tot_bins_dev.dtype, np.uint64)

        corr.Q_inds = [np.array([0, 1, 2], dtype=np.uint64)]
        corr.Q_cos = [np.array([1.0, 1.0, 1.0], dtype=np.float32)]
        corr.Q_sin = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]
        corr.Q_val = [np.array([1.0, 1.0, 1.0], dtype=np.float32)]
        corr.Q_patch_area = [3.0]

        npix = 12 * self.nside**2
        shear_maps = np.ones((2, 2, npix), dtype=np.float64)
        w = np.ones((2, npix), dtype=np.float64)

        M_ap, xip, xim = corr.get_full_tomo(shear_maps, w)

        self.assertEqual(M_ap.dtype, np.float32)
        self.assertEqual(xip.dtype, np.float32)
        self.assertEqual(xim.dtype, np.float32)
        self.assertEqual(corr._tomo_sumofweights_cache.dtype, np.float32)

if __name__ == '__main__':
    unittest.main()
