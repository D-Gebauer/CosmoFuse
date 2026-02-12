import unittest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path for testing
sys.path.insert(1, str(Path(__file__).parent.parent / "src"))

from CosmoFuse.correlations import Correlation
from CosmoFuse.backend import Backend

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

    def test_prepare_and_xipm_cpu(self):
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

        # Create dummy maps
        npix = 12 * self.nside**2
        g1 = np.ones(npix)
        g2 = np.ones(npix)
        w = np.ones(npix)

        # Load maps
        corr.load_maps(g1, g2, g1, g2, w, w)

        # Calculate xipm
        # We need to pass arguments to xipm as backend arrays (which load_maps handles for stored ones)
        # But public xipm method takes arguments.
        # Wait, public xipm takes arguments g11, g21...
        # But usually we call calculate_2PCF() or get_full_tomo().

        # Let's test get_all_xipm which uses stored maps
        xip, xim = corr.get_all_xipm()

        self.assertEqual(xip.shape, (1, 2)) # 1 patch, 2 bins
        self.assertEqual(xim.shape, (1, 2))

if __name__ == '__main__':
    unittest.main()
