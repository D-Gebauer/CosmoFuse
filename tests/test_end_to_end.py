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


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.nside = 256
        self.radius_patch = 90
        self.theta_min = 30
        self.theta_max = 120
        self.nbins = 5

        data_dir = Path(__file__).parent / "data"
        mask_path = data_dir / "hp_inds.npy"
        shear_path = data_dir / "shear_maps.npy"
        density_path = data_dir / "density_maps.npy"


        if not (mask_path.exists() and shear_path.exists() and density_path.exists()):
            raise unittest.SkipTest(f"Test data files not found in {data_dir}")

        self.map_inds = np.load(mask_path)
        self.des_map = np.zeros(hp.nside2npix(self.nside))
        self.des_map[self.map_inds] = 1
        self.phi_center = np.array([0.44178647, 0.73631078, 0.85902924, 0.71176709, 0.17180585, 1.07992247])
        self.theta_center = np.array([1.54996149, 1.80201781, 1.9551931 , 2.04691539, 2.1432149 , 2.21951601])

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
            rotation_precision="float64",
            map_precision="float64",
            index_precision="uint32",
        )

        self.shear_maps = np.zeros((2, 2, hp.nside2npix(self.nside)))
        self.shear_maps[:, :, self.map_inds] = np.load(shear_path)

        self.density_maps = np.zeros((2, hp.nside2npix(self.nside)))
        self.density_maps[:, self.map_inds] = np.load(density_path)

        self.w1 = np.ones(len(self.shear_maps[0, 0]))
        self.w2 = self.w1

        npatches = len(self.theta_center)
        self.xip_treecorr_auto = np.zeros((npatches, self.nbins))
        self.xim_treecorr_auto = np.zeros((npatches, self.nbins))
        self.xip_treecorr_cross = np.zeros((npatches, self.nbins))
        self.xim_treecorr_cross = np.zeros((npatches, self.nbins))
        self.npairs_treecorr_auto = np.zeros((npatches, self.nbins))
        self.npairs_treecorr_cross = np.zeros((npatches, self.nbins))
        self.wtheta_treecorr_cross = np.zeros((npatches, self.nbins))
        self.gammat_treecorr_cross = np.zeros((npatches, self.nbins))

        gg = treecorr.GGCorrelation(
            nbins=self.nbins,
            min_sep=self.theta_min,
            max_sep=self.theta_max,
            sep_units="arcmin",
            brute=True,
            metric="Arc",
            bin_slop=0.0,
            angle_slop=0.0,
        )
        kk = treecorr.KKCorrelation(
            nbins=self.nbins,
            min_sep=self.theta_min,
            max_sep=self.theta_max,
            sep_units="arcmin",
            brute=True,
            metric="Arc",
            bin_slop=0.0,
            angle_slop=0.0,
        )
        kg = treecorr.KGCorrelation(
            nbins=self.nbins,
            min_sep=self.theta_min,
            max_sep=self.theta_max,
            sep_units="arcmin",
            brute=True,
            metric="Arc",
            bin_slop=0.0,
            angle_slop=0.0,
        )

        for i in range(npatches):
            vec = hp.ang2vec(self.theta_center[i], self.phi_center[i])
            patch_inds = hp.query_disc(
                self.nside, vec=vec, radius=np.radians(self.radius_patch / 60)
            )
            pix_inds = np.intersect1d(patch_inds, self.map_inds)
            ra, dec = pixel2RaDec(pix_inds, self.nside)

            shear_1 = treecorr.Catalog(
                ra=ra,
                dec=dec,
                g1=self.shear_maps[0, 0, pix_inds],
                g2=self.shear_maps[0, 1, pix_inds],
                w=self.w1[pix_inds],
                ra_units="rad",
                dec_units="rad",
                flip_g1=True,
            )
            shear_2 = treecorr.Catalog(
                ra=ra,
                dec=dec,
                g1=self.shear_maps[1, 0, pix_inds],
                g2=self.shear_maps[1, 1, pix_inds],
                w=self.w2[pix_inds],
                ra_units="rad",
                dec_units="rad",
                flip_g1=True,
            )

            gg.process(shear_2)
            self.xip_treecorr_auto[i, :] = gg.xip
            self.xim_treecorr_auto[i, :] = gg.xim
            self.npairs_treecorr_auto[i, :] = gg.npairs

            gg.process(shear_1, shear_2)
            self.xip_treecorr_cross[i, :] = gg.xip
            self.xim_treecorr_cross[i, :] = gg.xim
            self.npairs_treecorr_cross[i, :] = gg.npairs

            density_1 = treecorr.Catalog(
                ra=ra,
                dec=dec,
                k=self.density_maps[0, pix_inds],
                w=self.w1[pix_inds],
                ra_units="rad",
                dec_units="rad",
            )
            density_2 = treecorr.Catalog(
                ra=ra,
                dec=dec,
                k=self.density_maps[1, pix_inds],
                w=self.w2[pix_inds],
                ra_units="rad",
                dec_units="rad",
            )

            kk.process(density_1, density_2)
            self.wtheta_treecorr_cross[i, :] = kk.xi

            kg.process(density_1, shear_2)
            self.gammat_treecorr_cross[i, :] = kg.xi

        self.rnom = gg.rnom

    def find_pairs(self):
        self.corr.calculate_pairs_2PCF()
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

    def _ensure_mock_maperture_pairs(self):
        if hasattr(self.corr, "Q_inds"):
            return

        npatches = len(self.theta_center)
        self.corr.Q_inds = [np.array([0], dtype=np.uint32) for _ in range(npatches)]
        self.corr.Q_cos = [np.array([1.0], dtype=np.float64) for _ in range(npatches)]
        self.corr.Q_sin = [np.array([0.0], dtype=np.float64) for _ in range(npatches)]
        self.corr.Q_val = [np.array([1.0], dtype=np.float64) for _ in range(npatches)]
        self.corr.Q_patch_area = [1.0 for _ in range(npatches)]

    def get_auto_correlation(self):
        self._ensure_mock_maperture_pairs()
        shear_maps = self.shear_maps[1:2]
        w = np.stack((self.w1,), axis=0)
        _, xip_full, xim_full = self.corr.vectorized_shear_shear(
            shear_maps,
            w,
            flip_g1=True,
        )
        xip = xip_full[0]
        xim = xim_full[0]

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

    def get_cross_correlation(self):
        self._ensure_mock_maperture_pairs()
        w = np.stack((self.w1, self.w2), axis=0)
        _, xip_full, xim_full = self.corr.vectorized_shear_shear(
            self.shear_maps,
            w,
            flip_g1=True,
        )
        xip = xip_full[1]
        xim = xim_full[1]

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

    def get_gc_correlation(self):
        wtheta, = self.corr.compute_density_density(
            self.density_maps[0],
            self.density_maps[1],
            self.w1,
            self.w2,
        )

        self.assertAlmostEqual(
            np.abs(1 - (wtheta / self.wtheta_treecorr_cross)).max(), 0.0, delta=1e-6
        )
        self.assertAlmostEqual(
            np.abs(wtheta - self.wtheta_treecorr_cross).max(), 0.0, delta=1e-10
        )

    def get_ggl_correlation(self):
        gamma_t, = self.corr.compute_density_shear(
            self.density_maps[0],
            self.shear_maps[1, 0],
            self.shear_maps[1, 1],
            self.w1,
            self.w2,
        )

        self.assertAlmostEqual(
            np.abs(1 - (gamma_t / self.gammat_treecorr_cross)).max(), 0.0, delta=1e-6
        )
        self.assertAlmostEqual(
            np.abs(gamma_t - self.gammat_treecorr_cross).max(), 0.0, delta=1e-10
        )

    def test_end_to_end_correlations(self):
        self.find_pairs()
        self.get_auto_correlation()
        self.get_cross_correlation()
        self.get_gc_correlation()
        self.get_ggl_correlation()


if __name__ == "__main__":
    unittest.main()
