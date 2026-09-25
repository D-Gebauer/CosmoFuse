"""GPU float64 (single-pass 3x2pt tile, the default) vs CPU float64, packed
and unpacked, 24 production patches; also single pass vs three tiles."""
import numpy as np
from stage7_gpu_tiles import geometry, make_maps, rel
from CosmoFuse.correlations import Integrated3PCF


def fields_of(result, *names):
    return tuple(getattr(result, n) for n in names)


phi, theta, geo, _ = geometry(110)
res = {}
for name, dev, pack, mode in (("cpu", "cpu", False, None), ("cpu_packed", "cpu", True, None),
                              ("gpu", "gpu", False, "auto"), ("gpu_packed", "gpu", True, "auto"),
                              ("gpu_three_tiles", "gpu", False, "off")):
    corr = Integrated3PCF(512, phi[:24], theta[:24], device=dev, map_precision="float64",
                       rotation_precision="float64", pack_pairs=pack, **geo)
    corr.preprocess()
    if mode:
        corr.backend.kernel_3x2pt_tomo_pairs.mode = mode
    shear, w, dens, wd = (x.astype(np.float64) for x in make_maps(corr, np.random.default_rng(1)))
    res[name] = [np.asarray(x, dtype=np.float64) for x in fields_of(corr.three_by_two_pt(
        shear_maps=shear, k_maps=dens, weights={"shear": w, "density": wd},
        flip_g1=True, return_device=False), "M_a", "M_g", "xi_p", "xi_m", "xi_g", "xi_t")]
for a, b in (("gpu", "cpu"), ("gpu_packed", "cpu_packed"), ("gpu", "gpu_three_tiles")):
    print(a, "vs", b, ["%.1e" % rel(x, y) for x, y in zip(res[a], res[b])])
