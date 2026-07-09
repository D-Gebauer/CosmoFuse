"""Compile-and-launch smoke test for every CUDA RawKernel wrapper.

Run this on the GPU machine after every .cu edit so NVRTC errors surface
immediately with a stack trace instead of a logged warning followed by a
RuntimeError deep inside a measurement:

    python scripts/compile_smoke.py [device_id]

Each kernel family is launched once per production (map dtype, rotation
dtype, filter dtype, tomo-bin count) combination with 1-pair / 1-bin
inputs.  A wrapper returning False (compilation refused/failed) is an
error here.

For CUDA-syntax checking on a machine WITHOUT a GPU, use
scripts/compile_check_nvrtc.py instead.
"""

import sys

import numpy as np


def _inputs(cupy, map_dtype, complex_dtype, q_dtype, nz):
    npix = 4
    npairs = 3
    nbins = 2
    npatches = 2
    ncomb = nz * (nz + 1) // 2
    rng = np.random.default_rng(0)

    def dev(arr):
        return cupy.asarray(arr)

    comb_i = np.zeros(ncomb, dtype=np.int32)
    comb_j = np.zeros(ncomb, dtype=np.int32)
    k = 0
    for i in range(nz):
        for j in range(i, nz):
            comb_i[k], comb_j[k] = i, j
            k += 1

    return {
        "shear": dev(rng.standard_normal((npix, nz, 2)).astype(map_dtype)),
        "density": dev(rng.standard_normal((npix, nz)).astype(map_dtype)),
        "weights": dev((0.5 + rng.random((npix, nz))).astype(map_dtype)),
        "ind_i": dev(rng.integers(0, npix, npairs).astype(np.int32)),
        "ind_j": dev(rng.integers(0, npix, npairs).astype(np.int32)),
        "rot_i": dev(np.exp(2j * rng.random(npairs)).astype(complex_dtype)),
        "rot_j": dev(np.exp(2j * rng.random(npairs)).astype(complex_dtype)),
        "offsets": dev(np.array([0, 2, npairs], dtype=np.int64)),
        "comb_i": dev(comb_i),
        "comb_j": dev(comb_j),
        "q_inds": dev(rng.integers(0, npix, npairs).astype(np.uint32)),
        "q_cos": dev(rng.random(npairs).astype(q_dtype)),
        "q_sin": dev(rng.random(npairs).astype(q_dtype)),
        "q_val": dev(rng.random(npairs).astype(q_dtype)),
        "q_offsets": dev(np.array([0, 1, npairs], dtype=np.int64)),
        "q_area": dev(np.ones(npatches, dtype=q_dtype)),
        "nz": nz,
        "ncomb": ncomb,
        "nbins": nbins,
        "npatches": npatches,
        "map_dtype": map_dtype,
    }


def _zeros(cupy, shape, dtype):
    return cupy.zeros(shape, dtype=dtype)


def smoke(device_id=0):
    import cupy

    from CosmoFuse.backend import get_backend

    backend = get_backend(device_id)
    combos = [
        (np.float64, np.complex64, np.float32),
        (np.float64, np.complex128, np.float64),
        (np.float32, np.complex64, np.float32),
    ]
    for map_dtype, complex_dtype, q_dtype in combos:
        for nz in (1, 2, 5):
            label = f"map={np.dtype(map_dtype).name} rot={np.dtype(complex_dtype).name} q={np.dtype(q_dtype).name} nz={nz}"
            d = _inputs(cupy, map_dtype, complex_dtype, q_dtype, nz)
            nbins, ncomb, npatches = d["nbins"], d["ncomb"], d["npatches"]

            ok = backend.xipm_tomo_vectorized_kernel(
                d["shear"], d["weights"], d["ind_i"], d["ind_j"],
                d["rot_i"], d["rot_j"], d["offsets"], d["comb_i"], d["comb_j"],
                _zeros(cupy, (2, 2 * ncomb, nbins), map_dtype),
                _zeros(cupy, (2 * ncomb, nbins), map_dtype),
            )
            assert ok, f"xipm wrapper declined: {label}"

            ok = backend.kernel_density_density_tomo_vectorized(
                d["density"], d["weights"], d["ind_i"], d["ind_j"],
                d["offsets"], d["comb_i"], d["comb_j"],
                _zeros(cupy, (2 * ncomb, nbins), map_dtype),
                _zeros(cupy, (2 * ncomb, nbins), map_dtype),
            )
            assert ok, f"dd wrapper declined: {label}"

            ok = backend.kernel_density_shear_tomo_vectorized(
                d["density"], d["shear"], d["weights"], d["weights"],
                d["ind_i"], d["ind_j"], d["rot_i"], d["rot_j"],
                d["offsets"], d["comb_i"], d["comb_j"],
                _zeros(cupy, (ncomb, nbins), map_dtype),
                _zeros(cupy, (ncomb, nbins), map_dtype),
            )
            assert ok, f"ds wrapper declined: {label}"

            shear_planar = cupy.ascontiguousarray(cupy.transpose(d["shear"], (1, 2, 0)))
            weights_planar = cupy.ascontiguousarray(cupy.transpose(d["weights"], (1, 0)))
            ok = backend.aperture_tomo_shear_kernel(
                shear_planar[:, 0], shear_planar[:, 1], weights_planar,
                d["q_inds"], d["q_cos"], d["q_sin"], d["q_val"],
                d["q_offsets"], d["q_area"],
                _zeros(cupy, (nz, npatches), map_dtype),
                _zeros(cupy, (nz, npatches), map_dtype),
            )
            assert ok, f"aperture-shear wrapper declined: {label}"

            density_planar = cupy.ascontiguousarray(cupy.transpose(d["density"], (1, 0)))
            ok = backend.aperture_tomo_density_kernel(
                density_planar, weights_planar,
                d["q_inds"], d["q_val"], d["q_offsets"], d["q_area"],
                _zeros(cupy, (nz, npatches), map_dtype),
                _zeros(cupy, (nz, npatches), map_dtype),
            )
            assert ok, f"aperture-density wrapper declined: {label}"

            ok = backend.kernel_3x2pt_tomo_fused(
                d["density"], d["shear"], d["weights"], d["weights"],
                d["ind_i"], d["ind_j"], d["rot_i"], d["rot_j"], d["offsets"],
                d["q_inds"], d["q_cos"], d["q_sin"], d["q_val"],
                d["q_offsets"], d["q_area"],
                d["comb_i"], d["comb_j"], d["comb_i"], d["comb_j"],
                d["comb_i"], d["comb_j"],
                _zeros(cupy, (nz, npatches), map_dtype),
                _zeros(cupy, (nz, npatches), map_dtype),
                _zeros(cupy, (nz, npatches), map_dtype),
                _zeros(cupy, (nz, npatches), map_dtype),
                _zeros(cupy, (2 * ncomb, nbins), map_dtype),
                _zeros(cupy, (2 * ncomb, nbins), map_dtype),
                _zeros(cupy, (2 * ncomb, nbins), map_dtype),
                _zeros(cupy, (2 * ncomb, nbins), map_dtype),
                _zeros(cupy, (2 * ncomb, nbins), map_dtype),
                _zeros(cupy, (ncomb, nbins), map_dtype),
                _zeros(cupy, (ncomb, nbins), map_dtype),
            )
            assert ok, f"fused wrapper declined: {label}"

            cupy.cuda.runtime.deviceSynchronize()
            print(f"ok: {label}")

    print("All kernel families compiled and launched.")


if __name__ == "__main__":
    smoke(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
