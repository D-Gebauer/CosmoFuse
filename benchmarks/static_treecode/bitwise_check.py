import numpy as np, sys
sys.path.insert(0, "benchmarks/static_treecode")
from stage7_gpu_tiles import geometry, make_maps, set_tiled, methods, as_list
from CosmoFuse.correlations import Correlation
phi, theta, geo, pair_file = geometry(110)
n = 48
res = {}
for tiled in (True, False):
    corr = Correlation(512, phi[:n], theta[:n], device="gpu", map_precision="float32",
                       accumulation_precision="float64", **geo)
    corr.preprocess()
    set_tiled(corr, tiled)
    shear, w, dens, wd = make_maps(corr, np.random.default_rng(1))
    res[tiled] = {k: as_list(f()) for k, f in methods(corr, shear, w, dens, wd).items()}
    del corr
auto4 = [0, 4, 7, 9]   # (0,0) (1,1) (2,2) (3,3) in the 4-bin upper triangle
def eq(a, b): return bool(np.array_equal(a, b))
print("M_a bitwise:", eq(res[True]["shear"][0], res[False]["shear"][0]))
print("xi+ auto bitwise:", eq(res[True]["shear"][1][auto4], res[False]["shear"][1][auto4]))
print("xi- auto bitwise:", eq(res[True]["shear"][2][auto4], res[False]["shear"][2][auto4]))
print("xi_g auto bitwise:", eq(res[True]["density"][1][auto4], res[False]["density"][1][auto4]))
print("xi_t bitwise:", eq(res[True]["ggl"][0], res[False]["ggl"][0]))
x = res[True]["shear"][1]; y = res[False]["shear"][1]
print("xi+ cross max rel:", float(np.max(np.abs(x - y)) / np.max(np.abs(y))))
f = res[True]["3x2pt"]; g = res[False]["3x2pt"]
print("3x2pt bitwise per output:", [eq(a, b) for a, b in zip(f, g)])
print("3x2pt xi+ auto bitwise:", eq(f[2][auto4], g[2][auto4]), "xi_g auto:", eq(f[4][auto4], g[4][auto4]))
