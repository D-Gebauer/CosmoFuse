"""M_ap must not depend on resolution_factor when the aperture level is the base."""
import h5py, healpy as hp, numpy as np
from CosmoFuse.correlations import Correlation
SBI = "/e/ocean1/users/dgebauer/sbi"
nside = 512
des_map = hp.ud_grade(hp.read_map("/home/moon/dgebauer/research/lfi/local/DESY3_Mask.fits"), nside)
pair_file = f"{SBI}/CosmoFuse/Q110/2PCF_pairs_512_15_250_8.h5"
with h5py.File(pair_file, "r") as fp:
    attrs = dict(fp.attrs)
phi = np.loadtxt(f"{SBI}/CosmoFuse/Q110/patch_center_original_phi.dat")[:60]
theta = np.loadtxt(f"{SBI}/CosmoFuse/Q110/patch_center_original_theta.dat")[:60]
geo = dict(nbins=int(attrs["nbins"]), theta_min=float(np.degrees(attrs["theta_min"]) * 60),
           theta_max=float(np.degrees(attrs["theta_max"]) * 60), patch_size=110., theta_Q=110., mask=des_map)
cut = np.load(f"{SBI}/shear_maps/grid_baryonified/4BINS/gamma_0000.npy").reshape(-1, 4, 2, 369190)[0].astype(np.float32)
w = np.load(f"{SBI}/shear_maps/SumOfWeights_512.npy")
res = {}
for name, k, load in (("file", None, True), ("fresh_full", None, False), ("k2.9", 2.9, False)):
    corr = Correlation(nside, phi, theta, device="gpu", map_precision="float32", accumulation_precision="float64", resolution_factor=k, **geo)
    if load:
        corr.load_pairs(pair_file, stop_ind=60)
    else:
        corr.preprocess()
    ww = np.ascontiguousarray(w[:, corr.row_pix], dtype=np.float32)
    res[name] = [np.asarray(o, dtype=np.float64) for o in corr.get_full_tomo_shear(cut, ww, flip_g1=True, return_device=False)]
for a, b in (("fresh_full", "k2.9"), ("file", "fresh_full")):
    print(a, "vs", b, "M_a identical:", np.array_equal(res[a][0], res[b][0]), " max rel diff M_a: %.2e" % (np.max(np.abs(res[a][0]-res[b][0]))/np.max(np.abs(res[b][0]))),
          "| xi+ base bins identical:", np.array_equal(res[a][1][..., :3], res[b][1][..., :3]), " max rel diff: %.2e" % (np.max(np.abs(res[a][1][..., :3]-res[b][1][..., :3]))/np.max(np.abs(res[b][1][..., :3]))))
