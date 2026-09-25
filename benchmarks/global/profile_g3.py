"""Split one Global3PCF.shear call into step 1 (kernel) and step 2 (contraction)."""
import sys, time
import healpy as hp, numpy as np, cupy
from CosmoFuse import Global3PCF
import CosmoFuse.global_3pcf as g3m

nside = int(sys.argv[1]); mask_path = sys.argv[2]; chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
m = hp.read_map(mask_path, dtype=np.float64)
mask = hp.ud_grade(m, nside) >= 0.5
rng = np.random.default_rng(0)
n = int(mask.sum())
shear = cupy.asarray((rng.normal(size=(4, 2, n)) * 0.3).astype(np.float32))
w = cupy.asarray(rng.uniform(0.5, 1.5, size=(4, n)).astype(np.float32))
g3 = Global3PCF(nside, mask=mask, nbins=12, theta_min=15, theta_max=250, n_max=20, device="gpu",
                map_precision="float32", rotation_precision="float32", accumulation_precision="float64",
                centre_chunk=chunk)
g3.find_pairs(); g3.prepare()
g3.shear(shear, w)  # warm up + norm
kern = g3.backend.kernel_3pcf_multipoles
tk = [0.0]
def timed_kernel(*a, **k):
    cupy.cuda.Device().synchronize(); t = time.perf_counter()
    r = kern(*a, **k)
    cupy.cuda.Device().synchronize(); tk[0] += time.perf_counter() - t
    return r
g3.backend.kernel_3pcf_multipoles = timed_kernel
for _ in range(3):
    tk[0] = 0.0
    cupy.cuda.Device().synchronize(); t = time.perf_counter()
    g3.shear(shear, w)
    cupy.cuda.Device().synchronize(); tot = time.perf_counter() - t
    print(f"nside {nside} chunk {chunk}: total {tot:.3f} s, step-1 kernel {tk[0]:.3f} s, rest {tot - tk[0]:.3f} s, pairs {g3.ntotpairs:.3e}")
