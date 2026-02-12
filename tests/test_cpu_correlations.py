import sys
import unittest
from pathlib import Path

import healpy as hp
import numpy as np
import treecorr

# Add src to path for testing
sys.path.insert(1, str(Path(__file__).parent.parent / "src"))

from CosmoFuse.correlations import Correlation
from CosmoFuse.utils import pixel2RaDec


class TestCPUCorrelation(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.nside = 512
        self.radius_patch = 90
        self.theta_min = 20
        self.theta_max = 170
        self.nbins = 10

        data_dir = Path(__file__).parent / "data"
        mask_path = data_dir / "DESY3_Mask.fits"
        phi_path = data_dir / "patch_center_phi.dat"
        theta_path = data_dir / "patch_center_theta.dat"
        shear_path = data_dir / "shear_maps.npy"

        if not (mask_path.exists() and phi_path.exists() and theta_path.exists() and shear_path.exists()):
            raise unittest.SkipTest(f"Test data files not found in {data_dir}")

        self.des_map = hp.read_map(str(mask_path))
        self.map_inds = np.where(self.des_map != 0)[0]
        self.phi_center = np.loadtxt(phi_path)
        self.theta_center = np.loadtxt(theta_path)

        self.corr = Correlation(
            self.nside,
            self.phi_center,
            self.theta_center,
            nbins=self.nbins,
            patch_size=self.radius_patch,
            theta_min=self.theta_min,
            theta_max=self.theta_max,
            mask=self.des_map,
            fastmath=False,
        )

        self.shear_maps = np.zeros((2, 2, hp.nside2npix(self.nside)))
        self.shear_maps[:, :, self.map_inds] = np.load(shear_path)

        self.w1 = np.ones(len(self.shear_maps[0, 0]))
        self.w2 = self.w1

        self.xip_treecorr_auto = np.zeros((len(self.theta_center), self.nbins))
        self.xim_treecorr_auto = np.zeros((len(self.theta_center), self.nbins))
        self.xip_treecorr_cross = np.zeros((len(self.theta_center), self.nbins))
        self.xim_treecorr_cross = np.zeros((len(self.theta_center), self.nbins))
        self.npairs_treecorr_auto = np.zeros((len(self.theta_center), self.nbins))
        self.npairs_treecorr_cross = np.zeros((len(self.theta_center), self.nbins))

        correlation = treecorr.GGCorrelation(
            nbins=self.nbins,
            min_sep=self.theta_min,
            max_sep=self.theta_max,
            sep_units="arcmin",
            brute=True,
            metric="Arc",
            bin_slop=0.0,
            ang_slop=0.0,
        )

        for i in range(len(self.theta_center)):
            vec = hp.ang2vec(self.theta_center[i], self.phi_center[i])
            patch_inds = hp.query_disc(
                self.nside, vec=vec, radius=np.radians(self.radius_patch / 60)
            )
            pix_inds = np.intersect1d(patch_inds, self.map_inds)
            g11 = self.shear_maps[0, 0, pix_inds]
            g21 = self.shear_maps[0, 1, pix_inds]
            g12 = self.shear_maps[1, 0, pix_inds]
            g22 = self.shear_maps[1, 1, pix_inds]
            ra, dec = pixel2RaDec(pix_inds, self.nside)
            catalog1 = treecorr.Catalog(
                ra=ra,
                dec=dec,
                g1=g11,
                g2=g21,
                w=self.w1[pix_inds],
                ra_units="rad",
                dec_units="rad",
                flip_g1=True,
            )
            catalog2 = treecorr.Catalog(
                ra=ra,
                dec=dec,
                g1=g12,
                g2=g22,
                w=self.w2[pix_inds],
                ra_units="rad",
                dec_units="rad",
                flip_g1=True,
            )
            correlation.process(catalog2)
            self.xip_treecorr_auto[i, :] = correlation.xip
            self.xim_treecorr_auto[i, :] = correlation.xim
            self.npairs_treecorr_auto[i, :] = correlation.npairs

            correlation.process(catalog1, catalog2)
            self.xip_treecorr_cross[i, :] = correlation.xip
            self.xim_treecorr_cross[i, :] = correlation.xim
            self.npairs_treecorr_cross[i, :] = correlation.npairs
        self.rnom = correlation.rnom
        # Setup complete

    def find_pairs_single(self):
        self.corr.calculate_pairs_2PCF(threads=1)
        self.assertTrue(
            (
                np.array(self.npairs_treecorr_auto).astype("int")
                == np.array(self.corr.bins)
            ).all()
        )
        self.assertTrue(
            (
                np.array(self.npairs_treecorr_cross).astype("int")
                == 2 * np.array(self.corr.bins)
            ).all()
        )

    def find_pairs_multi(self):
        self.corr.calculate_pairs_2PCF(threads=5)
        self.assertTrue(
            (
                np.array(self.npairs_treecorr_auto).astype("int")
                == np.array(self.corr.bins)
            ).all()
        )
        self.assertTrue(
            (
                np.array(self.npairs_treecorr_cross).astype("int")
                == 2 * np.array(self.corr.bins)
            ).all()
        )

    def get_auto_correlation_single(self):
        self.corr.load_maps(
            self.shear_maps[1, 0],
            self.shear_maps[1, 1],
            self.shear_maps[1, 0],
            self.shear_maps[1, 1],
            self.w1,
            self.w2,
            flip_g1=True,
        )
        xip, xim = self.corr.calculate_2PCF(threads=1)

        self.assertAlmostEqual(
            np.abs(1 - (xip / self.xip_treecorr_auto)).max(), 0.0, delta=1e-6
        )
        self.assertAlmostEqual(
            np.abs(1 - (xim / self.xim_treecorr_auto)).max(), 0.0, delta=1e-6
        )
        self.assertAlmostEqual(
            np.abs(xip - self.xip_treecorr_auto).max(), 0.0, delta=1e-10
        )
        self.assertAlmostEqual(
            np.abs(xim - self.xim_treecorr_auto).max(), 0.0, delta=1e-10
        )

    def get_auto_correlation_mult(self):
        self.corr.load_maps(
            self.shear_maps[1, 0],
            self.shear_maps[1, 1],
            self.shear_maps[1, 0],
            self.shear_maps[1, 1],
            self.w1,
            self.w2,
            flip_g1=True,
        )
        xip, xim = self.corr.calculate_2PCF(threads=5)

        self.assertAlmostEqual(
            np.abs(1 - (xip / self.xip_treecorr_auto)).max(), 0.0, delta=1e-6
        )
        self.assertAlmostEqual(
            np.abs(1 - (xim / self.xim_treecorr_auto)).max(), 0.0, delta=1e-6
        )
        self.assertAlmostEqual(
            np.abs(xip - self.xip_treecorr_auto).max(), 0.0, delta=1e-10
        )
        self.assertAlmostEqual(
            np.abs(xim - self.xim_treecorr_auto).max(), 0.0, delta=1e-10
        )

    def get_cross_correlation_single(self):
        self.corr.load_maps(
            self.shear_maps[0, 0],
            self.shear_maps[0, 1],
            self.shear_maps[1, 0],
            self.shear_maps[1, 1],
            self.w1,
            self.w2,
            flip_g1=True,
        )
        xip, xim = self.corr.calculate_2PCF(threads=1)

        self.assertAlmostEqual(
            np.abs(1 - (xip / self.xip_treecorr_cross)).max(), 0.0, delta=1e-6
        )
        self.assertAlmostEqual(
            np.abs(1 - (xim / self.xim_treecorr_cross)).max(), 0.0, delta=1e-6
        )
        self.assertAlmostEqual(
            np.abs(xip - self.xip_treecorr_cross).max(), 0.0, delta=1e-10
        )
        self.assertAlmostEqual(
            np.abs(xim - self.xim_treecorr_cross).max(), 0.0, delta=1e-10
        )

    def get_cross_correlation_mult(self):
        self.corr.load_maps(
            self.shear_maps[0, 0],
            self.shear_maps[0, 1],
            self.shear_maps[1, 0],
            self.shear_maps[1, 1],
            self.w1,
            self.w2,
            flip_g1=True,
        )
        xip, xim = self.corr.calculate_2PCF(threads=5)

        self.assertAlmostEqual(
            np.abs(1 - (xip / self.xip_treecorr_cross)).max(), 0.0, delta=1e-6
        )
        self.assertAlmostEqual(
            np.abs(1 - (xim / self.xim_treecorr_cross)).max(), 0.0, delta=1e-6
        )
        self.assertAlmostEqual(
            np.abs(xip - self.xip_treecorr_cross).max(), 0.0, delta=1e-10
        )
        self.assertAlmostEqual(
            np.abs(xim - self.xim_treecorr_cross).max(), 0.0, delta=1e-10
        )

    def test_single_core(self):
        self.find_pairs_single()
        self.get_auto_correlation_single()
        self.get_cross_correlation_single()

    def test_multi_core(self):
        self.find_pairs_multi()
        self.get_auto_correlation_mult()
        self.get_cross_correlation_mult()

    def test_patch_xip_xim_cosmofuse_vs_treecorr(self):
        """End-to-end validation of patch-wise xip/xim against TreeCorr."""
        self.corr.calculate_pairs_2PCF(threads=1)

        # Auto-correlation comparison
        self.corr.load_maps(
            self.shear_maps[1, 0],
            self.shear_maps[1, 1],
            self.shear_maps[1, 0],
            self.shear_maps[1, 1],
            self.w1,
            self.w2,
            flip_g1=True,
        )
        xip_auto, xim_auto = self.corr.calculate_2PCF(threads=1)

        np.testing.assert_allclose(
            xip_auto, self.xip_treecorr_auto, rtol=1e-6, atol=1e-10
        )
        np.testing.assert_allclose(
            xim_auto, self.xim_treecorr_auto, rtol=1e-6, atol=1e-10
        )

        # Cross-correlation comparison
        self.corr.load_maps(
            self.shear_maps[0, 0],
            self.shear_maps[0, 1],
            self.shear_maps[1, 0],
            self.shear_maps[1, 1],
            self.w1,
            self.w2,
            flip_g1=True,
        )
        xip_cross, xim_cross = self.corr.calculate_2PCF(threads=1)

        np.testing.assert_allclose(
            xip_cross, self.xip_treecorr_cross, rtol=1e-6, atol=1e-10
        )
        np.testing.assert_allclose(
            xim_cross, self.xim_treecorr_cross, rtol=1e-6, atol=1e-10
        )


if __name__ == "__main__":
    unittest.main()
