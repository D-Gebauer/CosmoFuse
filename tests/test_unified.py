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

    def test_prepare_and_get_full_tomo_cpu(self):
        corr = Correlation(
            nside=self.nside,
            phi_center=self.phi_center,
            theta_center=self.theta_center,
            nbins=self.nbins,
            device='cpu'
        )

        # Manually populate pair_inds to simulate preprocess
        # 1 patch, 2 bins.
        # pair_inds[0] is array of shape (2, N_pairs)
        # We need N_pairs > 0
        N_pairs = 10
        corr.pair_inds = [np.zeros((2, N_pairs), dtype=np.uint32)]
        corr.pair_exp2phi = [np.ones((2, N_pairs), dtype=np.complex128)]
        corr.bins = [np.array([5, 5], dtype=np.uint32)] # 2 bins, 5 pairs each

        # Prepare
        corr.prepare()

        self.assertIsNotNone(corr.inds_dev)
        self.assertEqual(corr.inds_dev.shape, (2, 10))
        self.assertTrue(isinstance(corr.inds_dev, np.ndarray))

        corr.Q_inds = [np.array([0, 1, 2], dtype=np.uint32)]
        corr.Q_cos = [np.array([1.0, 1.0, 1.0], dtype=np.float64)]
        corr.Q_sin = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]
        corr.Q_val = [np.array([1.0, 1.0, 1.0], dtype=np.float64)]
        corr.Q_patch_area = [3.0]

        # Create dummy tomographic maps (one redshift bin)
        npix = 12 * self.nside**2
        shear_maps = np.ones((1, 2, npix), dtype=np.float64)
        w = np.ones((1, npix), dtype=np.float64)

        _, xip, xim = corr.get_full_tomo(shear_maps, w)

        self.assertEqual(xip.shape, (1, 1, 2))
        self.assertEqual(xim.shape, (1, 1, 2))

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
