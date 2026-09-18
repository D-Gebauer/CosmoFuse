"""Where the non-kernel time of a tomographic call goes (idea #3, sub-items 2+3).

Times, on the real A100 path, the three per-map passes that sit between
`_expand_rows` and the pair kernel:

  * the sign-flip stack   (`flip_g1`/`flip_g2`; skipped entirely when unset)
  * the SoA -> AoS transpose (`_transpose_tomo_inputs_aos`)
  * the packed `perm` gather (pack_pairs=True only)

and compares them against the whole device-resident call, so we can decide
whether folding them into the degrade kernel's write is worth anything.

Usage (on seitz1):
    PYTHONPATH=src python benchmarks/static_treecode/layout_overhead.py \
        --nside 512 --k 2.9 --nz 4
"""

import argparse
import json

import numpy as np

from degrade_kernel_gate import build, timeit


def sizes_mb(arrs):
    return sum(a.nbytes for a in arrs) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nside", type=int, default=512)
    ap.add_argument("--k", type=float, default=2.9)
    ap.add_argument("--nz", type=int, default=4)
    ap.add_argument("--nbins", type=int, default=7)
    ap.add_argument("--theta-min", type=float, default=18.0)
    ap.add_argument("--theta-max", type=float, default=148.0)
    ap.add_argument("--patch-size", type=float, default=110.0)
    ap.add_argument("--theta-Q", type=float, default=110.0)
    ap.add_argument("--nside-centers", type=int, default=16)
    ap.add_argument("--map-precision", default="float32")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--pack-pairs", action="store_true")
    args = ap.parse_args()

    import cupy

    corr = build(args)
    corr.pack_pairs = bool(args.pack_pairs)
    corr.prepare()
    ctx = corr.compute_context
    packed = ctx.packed_pairs_dev is not None
    print(f"patches={corr.n_patches} n_active={corr.n_active} "
          f"n_appended={corr.n_appended} n_rows={corr.n_rows} "
          f"pairs={corr.ntotpairs} packed={packed}")

    rng = np.random.default_rng(3)
    shear = cupy.asarray(
        rng.normal(size=(args.nz, 2, corr.n_active)).astype(corr.map_dtype) * 0.3)
    dens = cupy.asarray(
        rng.normal(size=(args.nz, corr.n_active)).astype(corr.map_dtype))
    w = cupy.asarray(
        rng.uniform(0.2, 2.0, size=(args.nz, corr.n_active)).astype(corr.map_dtype))

    module = corr.backend.module
    rep = args.repeats
    report = {"config": vars(args), "packed": packed,
              "n_rows": int(corr.n_rows), "n_active": int(corr.n_active),
              "ntotpairs": int(corr.ntotpairs)}

    # ---- the expanded row buffers the pair pass actually starts from ----
    shear_rows, w_rows = corr._expand_shear_rows(shear, w, blocks="pairs")
    report["mapset_mb"] = sizes_mb((shear_rows, w_rows))

    # 1. the sign-flip stack, in isolation and end to end
    t_flip = timeit(
        lambda: module.stack((-shear_rows[:, 0], -shear_rows[:, 1]), axis=1),
        cupy, n=rep)
    report["flip_stack_ms"] = t_flip

    # 2. the SoA -> AoS transpose
    t_tr = timeit(lambda: corr._transpose_tomo_inputs_aos(shear_rows, w_rows),
                  cupy, n=rep)
    report["transpose_aos_ms"] = t_tr

    # 3. the packed perm gather, on top of the transpose
    if packed:
        aos, w_aos = corr._transpose_tomo_inputs_aos(shear_rows, w_rows)
        perm = ctx.packed_perm_dev
        report["n_packed_rows"] = int(perm.size)
        t_gather = timeit(
            lambda: (module.ascontiguousarray(aos[perm]),
                     module.ascontiguousarray(w_aos[perm])), cupy, n=rep)
        report["perm_gather_ms"] = t_gather

    # 4. the expansion itself, in both layouts
    report["expand_rows_ms"] = timeit(
        lambda: corr._expand_shear_rows(shear, w, blocks="pairs"), cupy, n=rep)
    report["expand_rows_aos_ms"] = timeit(
        lambda: corr._expand_shear_rows(shear, w, blocks="pairs", layout="aos"),
        cupy, n=rep)
    report["expand_rows_aos_flipped_ms"] = timeit(
        lambda: corr._expand_shear_rows(shear, w, blocks="pairs",
                                        signs=(-1.0, 1.0), layout="aos"),
        cupy, n=rep)

    # 5. the aperture kernel in both layouts -- this is why the pair
    #    layout is decided by the enclosing call and not per leaf.
    corr._compute_tomo_aperture_shear(shear, w, return_device=True)  # warm
    ctx = corr.compute_context
    kern = corr.backend.aperture_tomo_shear_kernel
    map_dt = getattr(module, corr.map_dtype.name)
    out_num = corr.backend.zeros((args.nz, corr.n_patches), dtype=map_dt)
    out_den = corr.backend.zeros((args.nz, corr.n_patches), dtype=map_dt)
    g_soa, w_soa = corr._expand_shear_rows(shear, w, blocks="aperture")
    g_aos, w_aosr = corr._expand_shear_rows(shear, w, blocks="aperture",
                                            layout="aos")
    geom = (ctx.Q_inds_dev, ctx.Q_cos_dev, ctx.Q_sin_dev, ctx.Q_val_dev,
            ctx.Q_offsets_dev, ctx.Q_patch_area_dev, out_num, out_den)
    report["aperture_kernel_ms"] = {
        "soa": timeit(lambda: kern(g_soa[:, 0], g_soa[:, 1], w_soa, *geom),
                      cupy, n=rep),
        "aos": timeit(lambda: kern(g_aos[:, :, 0].T, g_aos[:, :, 1].T,
                                   w_aosr.T, *geom), cupy, n=rep),
    }

    # 6. whole calls, with and without the flip
    for name, call in (
        ("get_full_tomo_shear", lambda flip: corr.get_full_tomo_shear(
            shear, w, flip_g2=flip, return_device=True)),
        ("vectorized_shear_shear", lambda flip: corr.vectorized_shear_shear(
            shear, w, flip_g2=flip, return_device=True)),
        ("get_3x2pt_tomo", lambda flip: corr.get_3x2pt_tomo(
            shear_maps=shear, density_maps=dens,
            weights={"shear": w, "density": w},
            flip_g2=flip, return_device=True)),
    ):
        try:
            noflip = timeit(lambda: call(False), cupy, n=rep)
            withflip = timeit(lambda: call(True), cupy, n=rep)
        except Exception as exc:  # noqa: BLE001 - report, do not hide
            report[name] = {"error": repr(exc)}
            continue
        report[name] = {"ms": noflip, "ms_flip_g2": withflip,
                        "flip_cost_ms": withflip - noflip}

    print(json.dumps(report, indent=2))
    out = "benchmarks/static_treecode/layout_overhead"
    out += "_packed.json" if packed else ".json"
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
