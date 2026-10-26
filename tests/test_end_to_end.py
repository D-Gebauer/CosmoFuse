import unittest
from pathlib import Path

import healpy as hp
import numpy as np
import treecorr

from CosmoFuse import Q_crittenden, U_crittenden, U_tophat
from CosmoFuse.correlations import Integrated3PCF
from CosmoFuse.utils import pixel2RaDec

from ._results import fields_of


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
        self.phi_center = np.array(
            [0.44178647, 0.73631078, 0.85902924, 0.71176709, 0.17180585, 1.07992247]
        )
        self.theta_center = np.array(
            [1.54996149, 1.80201781, 1.9551931, 2.04691539, 2.1432149, 2.21951601]
        )

        self.corr = Integrated3PCF(
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
        self.corr.find_pairs()
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
        xip_full, xim_full = fields_of(
            self.corr.gg(shear_maps, w, flip_g1=True, apertures=False), "xi_p", "xi_m"
        )
        xip = self.corr.backend.to_numpy(xip_full[0])
        xim = self.corr.backend.to_numpy(xim_full[0])

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
        xip_full, xim_full = fields_of(
            self.corr.gg(self.shear_maps, w, flip_g1=True, apertures=False),
            "xi_p",
            "xi_m",
        )
        xip = self.corr.backend.to_numpy(xip_full[1])
        xim = self.corr.backend.to_numpy(xim_full[1])

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
        wtheta_full = self.corr.kk(
            self.density_maps, np.stack((self.w1, self.w2), axis=0), apertures=False
        ).xi_g
        wtheta = self.corr.backend.to_numpy(wtheta_full[1])

        self.assertAlmostEqual(
            np.abs(1 - (wtheta / self.wtheta_treecorr_cross)).max(), 0.0, delta=1e-6
        )
        self.assertAlmostEqual(
            np.abs(wtheta - self.wtheta_treecorr_cross).max(), 0.0, delta=1e-10
        )

    def get_ggl_correlation(self):
        gamma_t_full = self.corr.kg(
            self.density_maps,
            self.shear_maps,
            np.stack((self.w1, self.w2), axis=0),
            np.stack((self.w1, self.w2), axis=0),
            flip_g1=True,
        ).xi_t
        gamma_t = self.corr.backend.to_numpy(gamma_t_full[1])

        self.assertAlmostEqual(
            np.abs(1 - (gamma_t / self.gammat_treecorr_cross)).max(), 0.0, delta=1e-6
        )
        self.assertAlmostEqual(
            np.abs(gamma_t - self.gammat_treecorr_cross).max(), 0.0, delta=1e-10
        )

    @staticmethod
    def _q_t_numpy(theta: np.ndarray, theta_q_arcmin: float) -> np.ndarray:
        theta_q = np.radians(theta_q_arcmin / 60)
        return (
            theta**2 / (4 * np.pi * theta_q**4) * np.exp(-(theta**2) / (2 * theta_q**2))
        )

    def _get_aperture_patch_data(self, patch_idx: int):
        vec = hp.ang2vec(self.theta_center[patch_idx], self.phi_center[patch_idx])
        pix_center = hp.ang2pix(
            self.nside,
            self.theta_center[patch_idx],
            self.phi_center[patch_idx],
        )
        patch_inds = hp.query_disc(
            self.nside,
            vec=vec,
            radius=np.radians(5 * self.corr.theta_Q / 60),
        )
        qpix_inds = np.intersect1d(patch_inds, self.map_inds)
        qpix_inds = qpix_inds[qpix_inds != pix_center]

        center_ra, center_dec = pixel2RaDec([pix_center], self.nside)
        q_ra, q_dec = pixel2RaDec(qpix_inds, self.nside)
        q_patch_area = hp.nside2pixarea(self.nside) * qpix_inds.size

        return (
            qpix_inds,
            q_ra,
            q_dec,
            float(center_ra[0]),
            float(center_dec[0]),
            q_patch_area,
        )

    def _reference_maperture_shear_numpy(
        self, g1: np.ndarray, g2: np.ndarray, w: np.ndarray
    ) -> np.ndarray:
        m_a = np.zeros(self.corr.n_patches, dtype=np.float64)

        for patch_idx in range(self.corr.n_patches):
            qpix_inds, q_ra, q_dec, center_ra, center_dec, q_patch_area = (
                self._get_aperture_patch_data(patch_idx)
            )

            cos_vartheta = np.cos(q_ra - center_ra) * np.cos(center_dec) * np.cos(
                q_dec
            ) + np.sin(center_dec) * np.sin(q_dec)
            cos_vartheta = np.clip(cos_vartheta, -1.0, 1.0)
            vartheta = np.arccos(cos_vartheta)

            # Projection from vector geometry, not from the bearing formula
            # the package uses: d = direction of the great circle to the
            # centre in the tangent plane of the pixel, in (east, north).
            v = np.stack(hp.pix2vec(self.nside, qpix_inds), axis=1)
            c = np.asarray(hp.ang2vec(np.pi / 2 - center_dec, center_ra))
            d = c[None, :] - (v @ c)[:, None] * v
            theta, phi = hp.pix2ang(self.nside, qpix_inds)
            east = np.stack([-np.sin(phi), np.cos(phi), 0 * phi], axis=1)
            north = np.stack(
                [
                    -np.cos(theta) * np.cos(phi),
                    -np.cos(theta) * np.sin(phi),
                    np.sin(theta),
                ],
                axis=1,
            )
            d_e = np.sum(d * east, axis=1)
            d_n = np.sum(d * north, axis=1)
            d2 = d_e**2 + d_n**2
            cos_2phi = (d_e**2 - d_n**2) / d2
            sin_2phi = -2 * d_e * d_n / d2
            gt = -g1[qpix_inds] * cos_2phi - g2[qpix_inds] * sin_2phi
            q_vals = self._q_t_numpy(vartheta, self.corr.theta_Q)

            m_a[patch_idx] = (
                q_patch_area * np.sum(w[qpix_inds] * gt * q_vals) / np.sum(w[qpix_inds])
            )

        return m_a

    @staticmethod
    def _u_tophat_numpy(theta: np.ndarray, theta_q_arcmin: float) -> np.ndarray:
        theta_q = np.radians(theta_q_arcmin / 60)
        cap_area = 2 * np.pi * (1 - np.cos(theta_q))
        return np.where(theta <= theta_q, 1.0 / cap_area, 0.0)

    @staticmethod
    def _u_crittenden_numpy(theta: np.ndarray, theta_q_arcmin: float) -> np.ndarray:
        theta_q = np.radians(theta_q_arcmin / 60)
        x2 = theta**2 / (2 * theta_q**2)
        return (1 - x2) * np.exp(-x2) / (2 * np.pi * theta_q**2)

    def _reference_maperture_density_numpy(
        self, density: np.ndarray, w: np.ndarray, filter_numpy
    ) -> np.ndarray:
        """Scalar aperture with a filter on the 5 theta_Q aperture discs."""
        m_g = np.zeros(self.corr.n_patches, dtype=np.float64)

        for patch_idx in range(self.corr.n_patches):
            qpix_inds, q_ra, q_dec, center_ra, center_dec, q_patch_area = (
                self._get_aperture_patch_data(patch_idx)
            )

            cos_vartheta = np.cos(q_ra - center_ra) * np.cos(center_dec) * np.cos(
                q_dec
            ) + np.sin(center_dec) * np.sin(q_dec)
            cos_vartheta = np.clip(cos_vartheta, -1.0, 1.0)
            vartheta = np.arccos(cos_vartheta)
            q_vals = filter_numpy(vartheta, self.corr.theta_Q)

            m_g[patch_idx] = (
                q_patch_area
                * np.sum(w[qpix_inds] * density[qpix_inds] * q_vals)
                / np.sum(w[qpix_inds])
            )

        return m_g

    def test_end_to_end_wl_correlations(self):
        self.find_pairs()
        self.get_auto_correlation()
        self.get_cross_correlation()

    def test_end_to_end_gc_correlation(self):
        self.find_pairs()
        self.get_gc_correlation()

    def test_end_to_end_ggl_correlation(self):
        self.find_pairs()
        self.get_ggl_correlation()

    def test_end_to_end_maperture_shear_matches_pure_numpy(self):
        self.corr.find_aperture_pairs()

        g1 = self.shear_maps[1, 0]
        g2 = self.shear_maps[1, 1]
        w = self.w2

        m_a_ref = self._reference_maperture_shear_numpy(g1, g2, w)
        m_a_corr = self.corr.backend.to_numpy(self.corr._m_a_single(g1, g2, w))

        np.testing.assert_allclose(m_a_corr, m_a_ref, rtol=1e-8, atol=1e-10)

    def test_end_to_end_maperture_density_matches_pure_numpy(self):
        self.corr.find_aperture_pairs()

        density = self.density_maps[1]
        w = self.w2

        # default: the weighted mean over the patch disc (Halder et al. 2023)
        m_g_ref = np.zeros(self.corr.n_patches)
        for patch_idx in range(self.corr.n_patches):
            vec = hp.ang2vec(self.theta_center[patch_idx], self.phi_center[patch_idx])
            disc = hp.query_disc(
                self.nside, vec=vec, radius=np.radians(self.corr.patch_size / 60)
            )
            disc = disc[np.isin(disc, self.map_inds)]
            m_g_ref[patch_idx] = np.sum(w[disc] * density[disc]) / np.sum(w[disc])
        m_g_corr = self.corr.backend.to_numpy(self.corr._m_g_single(density, w))
        np.testing.assert_allclose(m_g_corr, m_g_ref, rtol=1e-8, atol=1e-10)
        self.assertTrue(np.any(m_g_ref != 0))

        # explicit filters on the aperture discs: Q (the 7.6.0 default), the
        # compensated U and the disc-normalised top hat (the 7.6.1 default)
        for filter_fn, filter_numpy in (
            (Q_crittenden, self._q_t_numpy),
            (U_crittenden, self._u_crittenden_numpy),
            (U_tophat, self._u_tophat_numpy),
        ):
            with self.subTest(filter=filter_fn.__name__):
                m_g_ref = self._reference_maperture_density_numpy(
                    density, w, filter_numpy
                )
                m_g_corr = self.corr.backend.to_numpy(
                    self.corr._m_g_single(density, w, aperture_filter=filter_fn)
                )
                np.testing.assert_allclose(m_g_corr, m_g_ref, rtol=1e-8, atol=1e-10)

    def test_patch_mean_of_a_constant_is_the_constant(self):
        """The default M_g is a weighted mean: exactly the constant for any
        weights and mask."""
        npix = hp.nside2npix(self.nside)
        m_g = self.corr.backend.to_numpy(
            self.corr._m_g_single(np.full(npix, 0.25), self.w2)
        )
        np.testing.assert_allclose(m_g, 0.25, rtol=1e-12)

    def test_tophat_filter_m_g_is_the_cap_mean(self):
        """Unit weights and a constant field: the U_tophat filter gives
        pixarea * N_in / cap area, N_in the observed pixels in the cap -- 1 up
        to the pixelisation of the cap where the cap is fully observed (a
        masked pixel lowers M_g; it is not renormalised away)."""
        self.corr.find_aperture_pairs()
        ones = np.ones(hp.nside2npix(self.nside))
        m_g = self.corr.backend.to_numpy(
            self.corr._m_g_single(ones, ones, aperture_filter=U_tophat)
        )
        theta_q = np.radians(self.corr.theta_Q / 60)
        cap_area = 2 * np.pi * (1 - np.cos(theta_q))
        n_full = 0
        for patch_idx in range(self.corr.n_patches):
            vec = hp.ang2vec(self.theta_center[patch_idx], self.phi_center[patch_idx])
            pix_center = hp.ang2pix(
                self.nside, self.theta_center[patch_idx], self.phi_center[patch_idx]
            )
            qpix_inds, *_ = self._get_aperture_patch_data(patch_idx)
            vartheta = hp.rotator.angdist(
                hp.pix2vec(self.nside, pix_center), hp.pix2vec(self.nside, qpix_inds)
            )
            n_in = int(np.count_nonzero(vartheta <= theta_q))
            expected = hp.nside2pixarea(self.nside) * n_in / cap_area
            np.testing.assert_allclose(m_g[patch_idx], expected, rtol=1e-8)
            cap = hp.query_disc(self.nside, vec, theta_q)
            if np.isin(cap, self.map_inds).all():
                n_full += 1
                np.testing.assert_allclose(m_g[patch_idx], 1.0, rtol=0.05)
        self.assertGreater(n_full, 0)

    def test_end_to_end_maperture_shear_matches_treecorr(self):
        """M_a with TreeCorr's tangential projection: a brute-force
        NGCorrelation with the centre pixel as the only lens, the filter
        folded into the source weights and one radial bin, so
        sum(w Q g_t) = xi * weight."""
        self.corr.find_aperture_pairs()
        g1 = self.shear_maps[1, 0]
        g2 = self.shear_maps[1, 1]
        w = self.w2
        m_a_corr = self.corr.backend.to_numpy(self.corr._m_a_single(g1, g2, w))

        m_a_tc = np.zeros(self.corr.n_patches)
        for patch_idx in range(self.corr.n_patches):
            qpix_inds, q_ra, q_dec, center_ra, center_dec, q_patch_area = (
                self._get_aperture_patch_data(patch_idx)
            )
            vartheta = hp.rotator.angdist(
                hp.ang2vec(np.pi / 2 - center_dec, center_ra),
                hp.pix2vec(self.nside, qpix_inds),
            )
            q_vals = self._q_t_numpy(vartheta, self.corr.theta_Q)
            lens = treecorr.Catalog(
                ra=[center_ra], dec=[center_dec], ra_units="rad", dec_units="rad"
            )
            src = treecorr.Catalog(
                ra=q_ra,
                dec=q_dec,
                g1=g1[qpix_inds],
                g2=g2[qpix_inds],
                w=w[qpix_inds] * q_vals,
                ra_units="rad",
                dec_units="rad",
            )
            ng = treecorr.NGCorrelation(
                nbins=1,
                min_sep=1e-3,
                max_sep=5 * self.corr.theta_Q + 3 * hp.nside2resol(self.nside, True),
                sep_units="arcmin",
                metric="Arc",
                brute=True,
            )
            ng.process(lens, src)
            m_a_tc[patch_idx] = (
                q_patch_area * ng.xi[0] * ng.weight[0] / np.sum(w[qpix_inds])
            )

        self.assertGreater(np.abs(m_a_tc).max(), 0)
        # TreeCorr's own angle arithmetic limits the agreement (~1e-8)
        np.testing.assert_allclose(m_a_corr, m_a_tc, rtol=1e-6, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
