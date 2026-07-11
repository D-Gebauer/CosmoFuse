# First GPU session — Stage V validation and gate decisions (2026-07-11)

Hardware: NVIDIA A100 80GB PCIe (driver 580.173.02 / CUDA 13.0), conda env
`cosmo` (Python 3.12, numpy 2.0, CuPy 13.3 cu11x wheels, treecorr 5.0.2).
Harness: `benchmarks/bench_gpu_parity.py`, shared pair file, interleaved
baseline/candidate runs (two rounds each, min taken).

## Blockers found and fixed before anything would run (commit bccb6ef)

* NVRTC needs an explicit `--std` to instantiate template name
  expressions — without it **every** RawKernel silently fell back to the
  legacy paths. Baselines benchmarked for comparison need this fix
  backported or they measure the fallback, not the old kernels.
* `<cuComplex.h>` is not reliably locatable from conda/pip CuPy installs
  (this node's `/usr/local/cuda` is CUDA 10.1); the needed subset is now
  inlined in `common.cuh`.
* Two numpy<2 promotion bugs (`ntotpairs` becoming float64, uint64
  overflow guard) — the tests were authored on a numpy 2.x machine.

## Pre-existing precision bug fixed (commit 17b9204)

With the default `rotation_precision="float32"`, the standalone ξ± GPU
kernels multiplied float64 shear at float32, diverging from the CPU
reference by 6e-7 (vectorized) / 3e-5 (single-map ElementwiseKernels) —
identically at baseline a438504, so not a regression of the merged work.
All pair math now runs at map precision (rotation components promoted at
load), like the fused kernel and the CPU kernels always did.

## Stage V results (all criteria met after the fixes above)

Parity: all 19 harness rows ≤ 2.5e-14 scale-relative (float64).

GPU timings vs baseline a438504 (+env-fix patch), nside 512, 56 patches,
5 tomo bins, 10 angular bins:

| path | a438504 | merged | speedup |
|---|---|---|---|
| vectorized_shear_shear | 255 ms | 82 ms | 3.1× |
| vectorized_density_density | 229 ms | 53 ms | 4.3× |
| vectorized_density_shear | 476 ms | 129 ms | 3.7× |
| get_full_tomo_shear | 260 ms | 83 ms | 3.2× |
| get_3x2pt_tomo | 141 ms | 141 ms | 1.0× |
| single-map compute_*/aperture | — | — | ~1.0× |

* **G1** pass (the 3–4× rows above).
* **G2** pass via the M_a share of `get_full_tomo_shear`; the standalone
  `get_aperture_*` timings are upload-dominated (~17 ms ≈ H2D of three
  npix arrays), so the µs-scale kernel gain is invisible there.
* **G3** wall-neutral: each fused section already saturates the A100, so
  stream overlap adds no throughput at this size. Kept (enables
  exactly-sized launches; no regression).
* **G6** +8% on a two-map `get_full_tomo_shear` loop (90.3 → 83.0 ms/map).
  Most of the H2D win is eaten by the host-side copy into the pinned
  staging buffer; revisit only if maps arrive already pinned.
* **G7** persistent aperture+rotation device memory 341 MB → 204 MB with
  float32 rotation precision; numerically exact (parity above).

## Gate decisions for the remaining stages

`ncu` is unavailable to non-admin users on this node (ERR_NVGPUCTRPERM),
so the DRAM-read gates were evaluated with `nsys` (kernel/memcpy stats)
plus wall-time keep-ifs.

nsys over a full harness pass (4 map-set measurements × 4 reps):
**HtoD memcpys total 633 ms (7.75 GB); all kernels combined ~103 ms.**
Per map-set, the pair kernels are 3.8 ms (ξ±), 6.9 ms (ds), 1.5 ms (dd),
13 ms (fused, 5 sections) of 82–141 ms wall.

* **G4 (comb tiling): REJECTED** — entry gate 3 fails: the pair kernels
  are ~5–13% of per-map wall time; geometry re-streaming cannot matter.
* **G5 (payload packing): REJECTED** — same profile; the gathers are not
  on the critical path.
* **G8 (f32 payload, f64 accumulators): IMPLEMENTED** (commit 249430d) —
  attacks the measured upload/staging bottleneck directly.
  Keep-if met: 1.8–2.6× on every measurement path at ≤ 9.5e-7 parity vs
  the float64 CPU reference (tolerance 2e-4); float64 configs unchanged
  at ≤ 2.5e-14. With `accumulation_precision="same"` the single-map rows
  sat at ~5e-3 — the float64 accumulators are what make float32 usable.
* **G9/G10**: unchanged, workload-driven; note for G9 that per-map H2D —
  not geometry streaming — is the residual cost, so `PinnedMapPipeline`
  batching (+ float32 maps) is the relevant lever.

## Real-world workload (DES Y3 i3PCF geometry)

Production Q-patch pair files
(`/e/ocean1/users/dgebauer/sbi/CosmoFuse/Q*/2PCF_pairs_512_15_250_8.h5`),
real baryonified shear maps (`gamma_0000.npy`, footprint 0), 4 tomo
bins, `get_full_tomo_shear(..., flip_g1=True)`, DES Y3 mask at
nside 512. Interleaved runs (two rounds, median of 8 calls each).

| config | Q50 (1358 patches) | Q110 (917 patches) |
|---|---|---|
| a438504 + env-fix, float64 | 203 ms/call | 213 ms/call |
| merged, float64 | 62 ms/call (3.3×) | 60 ms/call (3.6×) |
| merged, float32 + float64 acc | 36 ms/call (5.6×) | 36 ms/call (5.9×) |

Per-call time is nearly independent of Q — it is dominated by the
per-map H2D upload + staging (same maps regardless of patch set),
which is exactly what the nsys profile predicted. For reference, the
Aug 2025 production log (`~/research/lfi/correlations/3PCF/log.txt`,
older code *and* different hardware) worked out to ~3.4–4.3 s per
call.

Correctness on the real Q110 data vs the float64 CPU reference
(max|Δ|/rms over all patches/bins):

| config | M_a | ξ+ | ξ− |
|---|---|---|---|
| a438504 baseline | 6.6e-15 | 9.6e-08 | 1.9e-07 |
| merged, float64 | 3.5e-15 | 7.1e-14 | 1.1e-13 |
| merged, float32 + f64 acc | 4.1e-07 | 4.0e-07 | 4.2e-07 |

(The baseline's 1e-7 ξ± rows are the pre-existing rotation-precision
bug fixed in commit 17b9204, visible on real data.) Q50 shows the
same pattern (merged float64: ≤ 3.4e-14).
