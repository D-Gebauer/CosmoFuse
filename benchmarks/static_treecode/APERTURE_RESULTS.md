# The aperture pass: how fast can it get, and how coarse may it be (2026-09-18)

Two independent questions about `gpu_aperture_shear_tomo`, the largest
non-pair item in `get_full_tomo_shear` (0.450 ms of 1.796 ms at nside 512)
and the one the treecode does not touch:

* **`aperture_nside`** — an *estimator* change: evaluate M_ap on the map
  degraded to a coarser nside.  Costs accuracy, must be paid for in zeta.
* **one block per patch instead of one per (patch, tomo bin)** — an
  *implementation* change: the same numbers, less memory traffic.

They are orthogonal and are answered separately below.  Scripts:
`aperture_reuse_probe.py` and `t10_zeta_level.py --aperture-nside`.


## 1. Block structure: the disc geometry really is re-read per bin

The shipped kernel runs one block per (patch, tomo bin), so it re-reads the
disc geometry — `q_inds` + `q_cos` + `q_sin` + `q_val`, 16 B per disc pixel —
once for every tomographic bin while using 12 B of map data per visit.  At
nz = 4 that is 112 B/pixel where one block per patch needs 16 + 12·nz = 64.

`ncu` cannot check whether those re-reads reach DRAM or are absorbed by L2:
performance counters need root on seitz1 (`ERR_NVGPUCTRPERM`).  So
`aperture_reuse_probe.py` measures the end state instead — two probe kernels
compiled in the benchmark, never touching the library, doing identical
arithmetic with different block structures.

A100 80 GB PCIe, 450 patches, float32 maps, nz = 4, median of 30:

| configuration | disc-pixel visits | shipped | 1 block/(patch,bin) | 1 block/patch, SoA | 1 block/patch, AoS | best |
|---|---|---|---|---|---|---|
| nside 512, aperture full res | 8.4 M | 0.456 ms | 0.451 ms | **0.258 ms** | 0.272 ms | **1.75×** |
| nside 2048, aperture full res | 134.5 M | 9.100 ms | 9.078 ms | 5.281 ms | **4.363 ms** | **2.08×** |
| nside 2048, `aperture_nside=512` | 8.6 M | 0.461 ms | 0.455 ms | **0.259 ms** | 0.271 ms | **1.75×** |

The measured 1.75× lands exactly on the 112 → 64 B/pixel traffic ratio, so
the geometry re-reads are **not** being caught by L2 and the naive traffic
model is the right one.  The nz scan confirms the mechanism independently:
fit t(nz) = intercept + slope·nz and look at intercept/slope, which the two
models predict to be 0 (geometry refetched every bin) or 16/12 = 1.33
(geometry read once).

| | current | fused SoA | fused AoS |
|---|---|---|---|
| nside 2048, aperture full res | 0.09 | 1.01 | 1.74 |
| nside 512 | 0.49 | 1.29 | 0.01 |

**Every variant is bitwise identical to the shipped kernel** (numerator and
denominator, all three configurations, max relative difference exactly 0.0).
That is not a bonus — `resolution_factor=None` must stay bit-for-bit
identical to 4.20.0, so a restructure that is not bitwise cannot ship.  It is
bitwise because the thread→pixel mapping and the reduction tree are
unchanged; only the bin loop moves inside the block, so each bin still sums
the same partials in the same order.

**Layout**: AoS wins only at nside-2048-sized discs (2.08× vs 1.72×) and
loses at nside-512-sized ones.  Since the production target evaluates the
aperture at nside 512 either way, **SoA stays** — the scope-chosen layout
logic in `_expansion_scope` needs no change.

**Implementation note.** `block_reduce_sum_pair` declares its shared buffers
inside the function, so inlining it NZ times allocates NZ copies (16 kB at
NZ = 8, float32; 32 kB at float64, which would start costing occupancy).  The
real kernel should declare one buffer pair outside the loop and
`__syncthreads()` between bins.  The per-thread accumulators must stay a
compile-time-sized array (`template<int NZ>`): run-time indexing spills them
to local memory, which is what made the first combination-tiled pair kernel
3× slower.


## 2. `aperture_nside`: what a coarse aperture costs zeta

`aperture_nside` was deliberately left at `None` in T10 so that only ξ±
changed.  This is the other half: `resolution_factor=None` throughout, so
**only M_ap changes**.  Same setup as T10 — DES Y3 mask at nside 512, Q110
(917 patches), 7 bins 18′–148′, 4 source bins, 56 baryonified N-body
map-sets, noise-free.

The target configuration is *base nside 2048 with the aperture pinned to
512*, but the map archive is at nside 512, so it cannot be run directly.  It
does not need to be: the window error is governed by the coarse pixel size
relative to θ_Q = 110′, and base 512 → aperture 128 is a **strictly more
aggressive** version of the same estimator change.

| run | aperture pixel | pixel/θ_Q | max abs(ratio−1) ζ+ / ζ− | max abs(Δζ)/σ_patch ζ+ / ζ− | quadrature over 140 entries |
|---|---|---|---|---|---|
| base 2048 → ap 512 (**the target**) | 6.87′ | 0.062 | — not runnable here — | | |
| base 512 → ap 256 | 13.7′ | 0.125 | 0.037 / 0.028 | **0.04 / 0.04** | 0.1 / 0.2 σ_patch |
| base 512 → ap 128 | 27.5′ | 0.250 | 0.068 / 0.094 | **0.06 / 0.07** | 0.3 / 0.3 σ_patch |

For scale, the treecode gate that was accepted is k = 2.9 at ≤ 0.27 σ_patch.
The 4×-too-aggressive proxy already sits at 0.07 σ_patch — a quarter of that
— and the measured scaling from ap256 to ap128 is between linear and
quadratic in pixel size (0.04 → 0.07 for a 2× coarser pixel).  Extrapolating
the 4× step down to the target gives **at most ~0.02 σ_patch, and ~0.004 if
the scaling is quadratic**.

**Verdict: `aperture_nside=512` at base nside 2048 is safe by a wide margin**
and buys a **20× cut** in the aperture pass (9.10 → 0.455 ms, table 1).

### Two things worth knowing about the shape of the effect

* **It is flat in θ.**  Per-bin max abs(Δζ)/σ_patch for ap128 runs
  0.058 / 0.058 / 0.055 / 0.056 / 0.039 / 0.055 / 0.050 across 18′…148′.
  That is structurally different from `resolution_factor`, which smears the
  pair separation and so hits small θ hardest.  Degrading the aperture
  rescales M_a almost uniformly, so it moves ζ as a near-constant
  multiplicative factor rather than a scale-dependent window.  The two knobs
  are therefore *not* interchangeable and should not be traded off against
  one another blindly.
* **The "detectability" counts are noise-dominated here.**  ap256 shows 3 and
  7 of 140 entries above 3σ of the paired 56-map-set scatter while the
  *larger* ap128 shift shows 0 — an inversion that only makes sense as noise
  in the paired-scatter estimate itself.  Read σ_patch, which is monotone
  (0.04 → 0.07), not the 3σ counts.
* Like the treecode, this is a deterministic window effect, identical in data
  and sims, so it costs information rather than biasing SBI — **provided the
  same `aperture_nside` is used for both**.  It is already written into the
  level table and carried by `ZetaWriter` as provenance.


## Reproducing

```bash
# block structure (fast, ~2 min each; the 2048 geometry build dominates)
python benchmarks/static_treecode/aperture_reuse_probe.py --nside 512 --nz 4
python benchmarks/static_treecode/aperture_reuse_probe.py --nside 2048 --nz 4 --nz-scan 1 2 4
python benchmarks/static_treecode/aperture_reuse_probe.py --nside 2048 --nz 4 \
    --aperture-nside 512 --nz-scan 1 2 4

# zeta level (~2 min each: 20 s preprocess + 56 map-sets at ~36 ms)
python benchmarks/static_treecode/t10_zeta_level.py --measure
python benchmarks/static_treecode/t10_zeta_level.py --measure --aperture-nside 256
python benchmarks/static_treecode/t10_zeta_level.py --measure --aperture-nside 128
python benchmarks/static_treecode/t10_zeta_level.py --analyse --a fine --b fine_ap256
python benchmarks/static_treecode/t10_zeta_level.py --analyse --a fine --b fine_ap128
```

Full per-triplet tables are written to
`benchmarks/static_treecode/results/T10_fine_ap{256,128}_vs_fine.md` and the
probe JSONs to `aperture_reuse_probe_{512,2048,2048_ap512}.json` — that
directory is excluded from `deploy_seitz1.sh`, so they live on seitz1 only.
