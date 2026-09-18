# Benchmark against TreeCorr (2026-09-18)

`benchmark_treecorr.py` / `summarise_benchmark.py`; full tables in
`results/benchmark_treecorr_summary.txt`, raw numbers in
`results/benchmark_treecorr_*.json`. seitz1: A100 80 GB (CosmoFuse), 48–64 CPU
processes (TreeCorr 5.0.2). DES Y3 mask, the 992 production Q90 patches
(R = θ_Q = 90′), 4 tomographic bins (10 combinations), **one map-set per case**.
CosmoFuse: float32 maps + float64 accumulators, `pair_search_precision="float64"`.

Reference = TreeCorr `brute=True` on the pixel catalogue of every patch.
`bin_slop=0, angle_slop=0` is identical to it (0.0 difference); `bin_slop=0`
*alone* is not exact (TreeCorr ≥ 5 still approximates the projection angles:
ξ− off by ~1e-2 σ). CosmoFuse agrees with an independent brute-force
calculation to 1e-16, TreeCorr brute to 1e-10.

σ = patch-to-patch scatter of the reference in that bin. "per patch" = the 992
× bins × 4 auto-combination estimates; "patch mean" = average over patches.

## Timings (one map-set)

| case | TreeCorr brute | TreeCorr default slop | CosmoFuse full res | CosmoFuse k = 4 | k = 4 packed |
|---|---|---|---|---|---|
| nside 512, 15–124′, 6 bins | 57.7 s × 48 proc (2771 core-s) | 57.4 s × 48 | **12.7 ms** (113 M pairs; 14 s one-off) | **9.8 ms** (48 M pairs; 10 s) | 10.1 ms (13 s) |
| nside 2048, 5–123′, 9 bins | 774 s × 64 proc (49.5 k core-s) | 105 s × 64 | 488 ms for 48 patches in 4 chunks (does not fit for 992) | **137 ms** (940 M pairs; 153 s one-off) | 129 ms (220 s) |

(TreeCorr at nside 512 is dominated by per-patch catalogue/tree overhead,
hence identical times for all modes.) Pair memory on the device, measured in
`stage6_gpu_packing.py`: 24 B → 8 B per pair (nside 2048, k = 2.9, 917 patches:
14.5 → 5.2 GB).

## Differences to exact (auto-combinations)

### nside 512 — real baryonified production map, real DES weights (noisy)
| method | ξ+ per patch: abs rms / max | in σ: rms / max | ξ− per patch: abs rms / max | in σ: rms / max |
|---|---|---|---|---|
| CosmoFuse full resolution | 1.7e-12 / 1.5e-11 | 8e-8 / 4e-7 | 1.6e-12 / 1.3e-11 | 9e-8 / 5e-7 |
| CosmoFuse full res, **float32 pair search (= 4.20.0 default)** | 1.4e-6 / 3.1e-5 | **0.050 / 0.84** | 1.4e-6 / 3.4e-5 | **0.056 / 0.95** |
| TreeCorr default bin_slop | 3.8e-7 / 5.8e-6 | 0.026 / 0.36 | 3.9e-7 / 6.3e-6 | 0.030 / 0.42 |
| CosmoFuse k = 4 (= packed) | 2.7e-6 / 2.7e-5 | 0.24 / 2.5 | 3.0e-6 / 2.9e-5 | 0.29 / 2.2 |

typical |ξ| per patch: 2.2e-5 (ξ+), 1.9e-5 (ξ−). Only the two largest bins are
coarse at k = 4 (nside 256); the four base-level bins are identical to full
resolution (≤ 5e-8 relative). A single noisy map cannot show the signal-level
effect (the patch means are themselves ~1σ) — see the noise-free case.

### nside 512 — CosmoGrid N-body, noise-free (signal-level effect)
relative difference of the patch-mean ξ per bin (15, 21, 30, 43, 61, 87′):

| method | ξ+ | ξ− |
|---|---|---|
| CosmoFuse full resolution | ≤ 1e-9 everywhere | ≤ 1e-9 |
| TreeCorr default | ≤ 7e-5 | ≤ 8e-5 |
| CosmoFuse k = 4 | 0, 0, 0, 0, **−0.57 %, −0.30 %** | 0, 0, 0, 0, **−3.8 %, −2.4 %** |
| k = 4 packed − k = 4 | ≤ 3e-7 | ≤ 4e-7 |

### nside 2048 — CosmoGrid N-body, noise-free
bins 5.0, 7.1, 10.2 | 14.5, 20.7 | 29.6, 42.2 | 60.3, 86.0′ at nside 2048 | 1024 | 512 | 256:

| method | ξ+ patch mean, rel. diff per bin | ξ− |
|---|---|---|
| CosmoFuse full resolution (48 patches) | ≤ 4e-9; per patch 5e-8 σ | ≤ 4e-9; 8e-8 σ |
| TreeCorr default | +0.04, −0.35, +0.07, +0.18, +0.07, +0.13, +0.03, +0.01, +0.07 % | 0.00, −1.0, −0.65, −1.1, −1.3, −0.81, −0.83, −0.64, −0.31 % |
| CosmoFuse k = 4 | 0, 0, 0, −0.09, +0.07, −0.63, +0.11, −0.62, +0.06 % | 0, 0, 0, **−4.3, −1.9, −5.2, −1.9, −5.1, −2.0 %** |
| k = 4 packed − k = 4 | ≤ 2e-7 | ≤ 1.5e-7 |

per-patch, in σ (rms / max): k = 4: ξ+ 0.032 / 0.35, ξ− 0.13 / 2.3; TreeCorr
default: ξ+ 0.013 / 0.24, ξ− 0.045 / 0.45. Absolute: k = 4 ξ+ 2.2e-7 rms (typ.
|ξ+| 2.2e-5), ξ− 3.5e-7 rms (typ. 1.0e-5).

### nside 2048 — same map + DES-like shape noise (σ = 0.128 per pixel and component)
| method | ξ+ per patch: abs rms | in σ: rms / max | ξ− abs rms | in σ: rms / max |
|---|---|---|---|---|
| CosmoFuse full resolution (48 patches) | 2.2e-12 | 6e-8 / 2e-7 | 2.1e-12 | 7e-8 / 3e-7 |
| TreeCorr default bin_slop | 8.7e-6 | 0.29 / 1.6 | 9.1e-6 | 0.35 / 2.1 |
| CosmoFuse k = 4 (= packed to 3e-5 σ) | 6.9e-6 | 0.35 / 1.9 | 7.6e-6 | 0.46 / 2.5 |

With noise the k = 4 treecode differs from the exact estimator by about as
much as TreeCorr's *default* settings do — both re-bin pairs near bin edges,
which decorrelates the shape noise without changing its variance.

## ζ level (T10), auto-combinations, exact ξ from TreeCorr brute + M_ap from CosmoFuse
|Δζ| in units of the patch-scatter error of ζ for that single map (rms / max over bins × z):

| case | ζ+ | ζ− |
|---|---|---|
| nside 512 real, full resolution | 7e-8 / 2e-7 | 6e-8 / 1.5e-7 |
| nside 512 real, float32 pair search (4.20.0 default) | 0.06 / 0.18 | 0.04 / 0.11 |
| nside 512 real, k = 4 | 0.18 / 0.65 | 0.24 / 0.93 |
| nside 2048 noise-free, k = 4 | 0.034 / 0.10 | 0.39 / 1.0 |
| nside 2048 noisy, k = 4 | 0.47 / 1.3 | 0.47 / 1.9 |

Packed and unpacked are indistinguishable at this level everywhere.

## Cross-combinations
CosmoFuse's cross-bin ξ± is the mean of the two orientation ratios
½(N_AB/W_AB + N_BA/W_BA); TreeCorr's is (N_AB + N_BA)/(W_AB + W_BA). With the
real per-bin DES weights they differ by ~3 % of max|ξ+| per patch (0.024 σ rms);
recombining CosmoFuse's raw numerators/denominators TreeCorr-style agrees to
5e-8. With equal weights per bin the two definitions coincide (≤ 5e-9).
