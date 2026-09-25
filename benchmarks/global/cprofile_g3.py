import sys, cProfile, pstats
import healpy as hp, numpy as np, cupy
from CosmoFuse import Global3PCF
nside = int(sys.argv[1])
m = hp.read_map(sys.argv[2], dtype=np.float64)
mask = hp.ud_grade(m, nside) >= 0.5
rng = np.random.default_rng(0); n = int(mask.sum())
shear = cupy.asarray((rng.normal(size=(4, 2, n)) * 0.3).astype(np.float32))
w = cupy.asarray(rng.uniform(0.5, 1.5, size=(4, n)).astype(np.float32))
g3 = Global3PCF(nside, mask=mask, nbins=12, theta_min=15, theta_max=250, n_max=20, device="gpu",
                map_precision="float32", rotation_precision="float32", centre_chunk=4096)
g3.find_pairs(); g3.prepare(); g3.shear(shear, w)
cupy.cuda.Device().synchronize()
pr = cProfile.Profile(); pr.enable()
for _ in range(3):
    g3.shear(shear, w)
cupy.cuda.Device().synchronize()
pr.disable()
pstats.Stats(pr).sort_stats("tottime").print_stats(14)
