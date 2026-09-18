"""Stage 5 gate: MapFileLoader on the A100.

(A) production nside 512: real archive files (float64 footprint cut-outs, 28
    map-sets per file) -> serial production-style loop vs loader; results must
    be bitwise identical; report steady-state ms per map-set and the GPU idle
    fraction (1 - device-only call time / wall per map-set).
(B) nside 2048 treecode geometry, maps served from RAM (isolates the upload
    overlap): same metrics.
"""
import argparse, json, time
import h5py, healpy as hp, numpy as np
from CosmoFuse import Correlation, MapFileLoader

SBI = "/e/ocean1/users/dgebauer/sbi"
MASK = "/home/moon/dgebauer/research/lfi/local/DESY3_Mask.fits"


def sync(corr):
    corr.backend.module.cuda.runtime.deviceSynchronize()


def device_only_ms(corr, shape, w, n=10):
    xp = corr.backend.module
    g = xp.asarray(np.random.default_rng(0).normal(size=shape).astype(np.float32) * 0.1)
    corr.get_full_tomo_shear(g, w, flip_g1=True); sync(corr)
    ts = []
    for _ in range(n):
        sync(corr); t0 = time.perf_counter()
        out = corr.get_full_tomo_shear(g, w, flip_g1=True)
        [o.get() for o in out]; ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3)


def run(corr, shape, sources, read_fn, w, n_slots, n_readers, serial_fn):
    # serial reference loop (what a production script does today)
    t0 = time.perf_counter(); serial = [serial_fn(s) for s in sources]; sync(corr)
    t_serial = (time.perf_counter() - t0) / len(sources) * 1e3
    loader = MapFileLoader(corr, {"shear": shape}, sources, read_fn, n_slots=n_slots,
                               n_readers=n_readers, row_pix_hash=corr.row_pix_hash)
    t0 = time.perf_counter(); got = []
    for k, dev in loader:
        out = corr.get_full_tomo_shear(dev["shear"], w, flip_g1=True)   # stays on device
        got.append([o.copy() for o in out])
    got = [[o.get() for o in r] for r in got]; sync(corr)
    t_loader = (time.perf_counter() - t0) / len(sources) * 1e3
    same = all(np.array_equal(a, b) for r, s in zip(got, serial) for a, b in zip(r, s))
    return t_serial, t_loader, same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="AB")
    ap.add_argument("--nfiles", type=int, default=3)
    ap.add_argument("--n2048", type=int, default=40)
    ap.add_argument("--k512", type=float, default=None)
    ap.add_argument("--pairs2048", default="benchmarks/static_treecode/results/pairs_2048_k2.9_Q110.h5")
    args = ap.parse_args()
    out = {}
    if "A" in args.parts:
        nside = 512
        des_map = hp.ud_grade(hp.read_map(MASK), nside)
        pair_file = f"{SBI}/CosmoFuse/Q110/2PCF_pairs_512_15_250_8.h5"
        with h5py.File(pair_file, "r") as fp:
            attrs = dict(fp.attrs)
        phi = np.loadtxt(f"{SBI}/CosmoFuse/Q110/patch_center_original_phi.dat")
        theta = np.loadtxt(f"{SBI}/CosmoFuse/Q110/patch_center_original_theta.dat")
        corr = Correlation(nside, phi, theta, nbins=int(attrs["nbins"]),
                           theta_min=float(np.degrees(attrs["theta_min"]) * 60),
                           theta_max=float(np.degrees(attrs["theta_max"]) * 60),
                           patch_size=110., theta_Q=110., mask=des_map, device="gpu",
                           map_precision="float32", accumulation_precision="float64",
                           resolution_factor=args.k512)
        if args.k512 is None:
            corr.load_pairs(pair_file, release_host_pairs=True)
        else:
            corr.preprocess(release_host_pairs=True)
        w = np.ascontiguousarray(np.load(f"{SBI}/shear_maps/SumOfWeights_512.npy")[:, corr.row_pix], dtype=np.float32)
        w.flags.writeable = False
        files = [f"{SBI}/shear_maps/grid_baryonified/4BINS/gamma_{i:04d}.npy" for i in range(args.nfiles)]
        sources = [(f, r, c) for f in files for r in range(7) for c in range(4)]
        shape = (4, 2, corr.n_active)

        def read(src, o):
            f, r, c = src
            o["shear"][...] = np.load(f, mmap_mode="r")[r, c]

        def serial(src):
            f, r, c = src
            rows = np.load(f, mmap_mode="r")[r, c].astype(np.float32)
            return [np.asarray(x) for x in corr.get_full_tomo_shear(rows, w, flip_g1=True, return_device=False)]

        for f in files:  # warm the page cache so both loops see the same disk state
            np.load(f, mmap_mode="r").sum()
        dev_ms = device_only_ms(corr, shape, corr._coerce_map_input_array(w))
        for n_readers in (1, 2, 4):
            t_serial, t_loader, same = run(corr, shape, sources, read, w, 6, n_readers, serial)
            key = f"A_512_k{args.k512}_readers{n_readers}"
            out[key] = {"serial_ms": t_serial, "loader_ms": t_loader, "device_only_ms": dev_ms,
                        "gpu_idle_frac": max(0.0, 1 - dev_ms / t_loader), "bitwise_equal": bool(same)}
            print(key, json.dumps(out[key]), flush=True)
        del corr
    if "B" in args.parts:
        nside = 2048
        mask = hp.read_map(MASK)
        if hp.npix2nside(mask.size) != nside:
            mask = hp.ud_grade(mask.astype(np.float64), nside)
        mask = mask != 0
        phi = np.loadtxt(f"{SBI}/CosmoFuse/Q110/patch_center_original_phi.dat")
        theta = np.loadtxt(f"{SBI}/CosmoFuse/Q110/patch_center_original_theta.dat")
        edges = np.geomspace(5.0, 250.0, 12); edges = edges[edges < 215]
        corr = Correlation(nside, phi, theta, nbins=len(edges) - 1, theta_min=edges[0], theta_max=edges[-1],
                           patch_size=110, theta_Q=110, mask=mask, device="gpu", map_precision="float32",
                           accumulation_precision="float64", resolution_factor=2.9, aperture_nside=512)
        corr.load_pairs(args.pairs2048, release_host_pairs=True)
        rng = np.random.default_rng(1)
        shape = (4, 2, corr.n_active)
        pool = [rng.normal(size=shape).astype(np.float32) * 0.1 for _ in range(4)]
        w = rng.uniform(0.5, 2.0, size=(4, corr.n_active)).astype(np.float32); w.flags.writeable = False
        sources = list(range(args.n2048))

        def read(src, o):
            np.copyto(o["shear"], pool[src % 4])

        def serial(src):
            return [np.asarray(x) for x in corr.get_full_tomo_shear(pool[src % 4], w, flip_g1=True, return_device=False)]

        dev_ms = device_only_ms(corr, shape, corr._coerce_map_input_array(w))
        for n_readers in (1, 2):
            t_serial, t_loader, same = run(corr, shape, sources, read, w, 4, n_readers, serial)
            key = f"B_2048_k2.9_readers{n_readers}"
            out[key] = {"serial_ms": t_serial, "loader_ms": t_loader, "device_only_ms": dev_ms,
                        "gpu_idle_frac": max(0.0, 1 - dev_ms / t_loader), "bitwise_equal": bool(same)}
            print(key, json.dumps(out[key]), flush=True)
    with open("benchmarks/static_treecode/results/stage5_loader.json", "w") as fp:
        json.dump(out, fp, indent=1)


if __name__ == "__main__":
    main()
