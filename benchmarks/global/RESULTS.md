# Global estimators — first measurements (2026-09-25)

`bench_global.py` / `profile_g3.py` / `profile_step2.py`, seitz1 A100 80 GB PCIe
(idle), cupy 13.3, DES Y3 mask (`hp.ud_grade(mask) >= 0.5`), random maps,
4 source bins, float32 maps + float64 accumulation, `flip_g1=True`,
device-resident inputs, synchronised wall time, median.

## Global2PCF (ξ±, 20 log bins 15–250′, full resolution, `pack_pairs=True`)

| nside | footprint rows | unique pairs | tiles (nside) | find | prepare | `gg()` |
|---|---|---|---|---|---|---|
| 256 | 91 k | — | 8 | 5.6 s | 4.8 s | **2.5 ms** |
| 512 | 365 k | 6.72 × 10⁸ | 135 (8) | 42 s | 158 s | **33 ms** |

`prepare` at nside 512 is dominated by the host-side packing (`np.unique` of
the row block of every (tile, level)); a one-off, but the next thing to make
faster before nside 2048.

### Update 2026-10-21 (7.2.0): `prepare` / `load_pairs`

Profiled before changing anything (laptop, one core, GPU code path with an
identity upload, 1.8 × 10⁸ pairs at nside 512): 90 ns/pair, of which
`np.unique` 54 % (its argsort), `encode_angles` 32 % (`np.angle` alone
20 %), id → row translation 5 %. 7.2.0 replaces the sort by a marker table
(3.9 ns/pair, bitwise `np.unique`) and packs the tiles in parallel threads.
`bench_prepare.py` on seitz1 (A100, 160 threads; another user held ~12 cores),
DES Y3, 20 bins 15–250′, `pack_pairs=True`, both trees in one session:

| nside | k | pairs | find | prepare 7.1.0 → 7.2.0 | load exact 7.1.0 → 7.2.0 | packed prepare / load / file | `gg()` |
|---|---|---|---|---|---|---|---|
| 512 | — | 6.72 × 10⁸ | 26–29 s | 119.8 → 8.7 s | 117.2 → 9.5 s | 1.2 s / 3.3 s / 5.4 GB (exact 16.1) | 36 ms |
| 1024 | 2.9 | 6.98 × 10⁷ | 29 s | 9.9 → 1.1 s | 9.5 → 1.2 s | 0.33 s / 0.55 s / 0.61 GB | 21 ms |
| 2048 | 2.9 | 6.98 × 10⁷ | 31 s | 9.7 → 1.2 s | 9.8 → 1.4 s | 0.36 s / 0.67 s / 0.72 GB | 69 ms |

Device state and `gg` output hash identically for both trees and all paths.
The 158 s of September was measured before the sortless pair search; 7.1.0
measured 120 s in this session.  `find` is now the slowest step: at nside 1024
(laptop profile) 37 of 39 s are inside `unique_pairs`, i.e. the brute-force
candidate tests of the fine levels, not the pairs kept.

## Global3PCF (shear Γ₀…Γ₃, 12 log bins 15–250′, n_max = 20, all 64 ordered z-triplets)

| nside | ordered pairs | find | device memory | first call (+ normalisation) | `ggg()` |
|---|---|---|---|---|---|
| 256 | 8.46 × 10⁷ | 4.2 s | 2.5 GB | 1.7 s | **0.29 s** |
| 512 | 1.34 × 10⁹ | 129 s | 19.1 GB | 3.7 s | **1.94 s** |

Split at nside 256: step 1 (CUDA kernel) 77 ms, step 2 (contraction) 210 ms.
Step 1 scales with the pairs (×16 per nside doubling), step 2 with the
centres (×4), so at nside 512 step 1 dominates (~1.2 s vs ~0.8 s).

* Step 2 was rewritten from one GEMM per centre bin (0.42 s total at nside
  256) to one batched GEMM per component, `L3ᵀ @ (L2 ⊗ centre)` (0.29 s): the
  complex128 GEMMs run at ~9 TFLOP/s, i.e. at the A100's FP64 rate — the
  remaining levers are fewer flops (symmetries of the components / z-triplets)
  or complex64 with `accumulation_precision="same"`.
* Step 1 runs at ~2.5 TFLOP/s (1.34 × 10⁹ pairs × 276 complex MACs per pair in
  1.2 s), a quarter of the FP64 rate: it is limited by the shared-memory
  reads of the staged neighbours, not by arithmetic.  The obvious next lever.
* The earlier back-of-envelope estimate for nside 512 (0.15–0.4 s per map)
  assumed 10–30 TFLOP/s sustained over both steps; it was ~5× too optimistic.
  10⁵ maps at nside 512 cost ~55 GPU-hours as measured.

Correctness on the same machine: `pytest -m gpu` (9 passed, including the
Global2PCF and Global3PCF GPU-vs-CPU gates at 1e-12 in float64).
