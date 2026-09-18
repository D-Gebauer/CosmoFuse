# Stage 0 results (2026-09-17)

Scripts: `stage0_geometry.py` (local CPU, DES Y3 mask), `stage0_gpu_baseline.py`
(seitz1, A100 80GB PCIe, shared with a light NLE training job, conda env
`cosmo`: CuPy 13.3, numpy 2.0). Deployed tree: `seitz1:research/CosmoFuse-static-treecode`
(the user's own checkout on seitz1 is untouched).

## Geometry (real pair finder, DES Y3 mask)

| configuration | patch pixels | pairs/patch | per 1000 patches |
|---|---|---|---|
| nside 512, 5–250′ × 11, R = 110′ | 810 | 0.328 M | 7.9 GB |
| nside 1024, same | 3 195 | 5.09 M | 122 GB |
| nside 2048, same | 12 756 | **81.1 M** | **1.95 TB** |

The guide's §2.1 estimates (0.33 M / 83 M / ~2 TB) are confirmed.

Production file `Q110/2PCF_pairs_512_15_250_8.h5` (legacy format, 7 bins —
the 8th edge is cut by `< 2·θ_Q − 5`): 917 patches, **265.4 M pairs**,
pairs/bin = 5.0, 9.5, 17.8, 30.3, 52.6, 75.4, 74.9 M. The two largest bins
hold 57 % of all pairs.

Local reconstruction with `from_mask(nside_centers=32, f_mask=0.5)` and a
`>= 0.5` degraded mask gives 1353 patches / 393 M pairs / 9.4 GB; footprint
364 790 px, pixels referenced by pairs 345 237, by apertures 354 813 (97 % of
the footprint). Aperture geometry: 23 M entries, 0.37 GB.

**Design consequence:** "active pixels" ≈ footprint to within 3 %. The row
space is therefore defined as the footprint (`map_inds`, ascending RING) —
known before pair finding, independent of `start_ind/stop_ind` slicing, and
*identical to the ordering of the existing production map archive*
(`gamma_XXXX.npy` has shape `(7, 4, nz, 2, 369190)` = footprint cut-outs).

## Reference workload timing (Q110, 4 tomo bins, float32 maps + float64 acc)

`get_full_tomo_shear(full_sky_maps, w, flip_g1=True, return_device=False)`,
device-synchronised, median of 28 real map-sets:

| component | ms / map-set |
|---|---|
| disk read (`np.load`, amortised over 28 sets/file, float64 on disk) | 62 |
| host scatter footprint → full sky | 7.6 |
| call, host inputs | **149** |
| call, device-resident inputs (= device prep + kernels + D2H) | **124** |
| ⇒ H2D + staging | 25 |

nsys, per call: `gpu_fused_tomo_reduce_xipm` **107 ms mean (70–120)** = 98 %
of all GPU kernel time; aperture kernel 0.56 ms; sign-flip/transposes ~1 ms;
H2D 50 MB/call; D2H negligible.

ξ± throughput: 4.25·10⁹ pair-evaluations (265 M pairs × 16 rows) in ~0.11 s
→ **~38 G pair-evaluations/s ≈ 1.8 TB/s at 48 B per evaluation**: the kernel
is memory-bandwidth bound, as modelled in §3 of the guide.

## What this changes in the guide

1. **The July premise "upload dominates, pair kernels are 5–13 % of wall" does
   not hold on the production geometry.** It was measured on the 56-patch
   harness. On Q110 the ξ± kernel is ~75 % of the synchronised call. (The
   36 ms/call figure in the July notes is not reproduced; 149 ms is, with
   the GPU lightly shared.)
2. **Stage 2's keep-if (≥ 3× per-map at nside 512) cannot be met by the row
   space alone** — it can remove at most scatter (8 ms) + most of H2D (25 ms)
   + flip/transposes (~1 ms) of ~157 ms. Stage 2 is still required
   structurally (appended rows; nside-2048 full-sky buffers would be 1.6 GB
   H2D per map-set).
3. **The Stage 6 gate (pair kernels > 50 % of wall) is already met at nside
   512.** Comb tiling (G4) was rejected on a non-representative profile.
4. At nside 512 the treecode itself is a speed lever: 57 % of pairs sit in
   the two largest bins.
5. Disk read (62 ms/set, float64 `.npy`) becomes the next bottleneck as soon
   as the kernel is fixed → Stage 5 matters.

---

# Stages 2–6 results (same day, same hardware)

All numbers: A100 80 GB PCIe (lightly shared), DES Y3 Q110 (917 patches),
4 source bins, float32 maps + float64 accumulators, synchronised wall time
per map-set, `get_full_tomo_shear(flip_g1=True)`.

| step | nside 512 full res (real maps) | nside 2048, 5–175′, k = 2.9 |
|---|---|---|
| 4.20.0 (full-sky input, scatter + call) | 157 ms | impossible (~1.7 TB of pairs) |
| Stage 2 row space (row-space host input) | 114 ms — **bitwise identical** to 4.20.0 | — |
| Stage 3/4 static treecode | 21.9 ms (k = 2.9), 34.3 ms (k = 4) | 411 ms, 592 M pairs, 17.7 GB on device, pair finding 67 s |
| Stage 6.1 tiled ξ± kernel (**bit-identical**, 4.5× / 4.2×) | 110.6 → 24.5 ms (kernel path) | 333.6 → 79.7 ms (kernel path); call 147 ms |
| degrade once per call + cached weight rows | — | call 117 ms (host input), 64.5 ms (device input) |
| Stage 5 ring loader (bitwise ≡ serial loop, 8/8 runs) | **26 ms from disk**, GPU idle 0–5 % | **72–78 ms**, GPU idle 2–4 % |
| treecode + tiled kernel + loader, nside 512 | **9.1 ms (k = 2.9), 11.5 ms (k = 4)** | |

Gates: T1/T5 bitwise vs unmodified 4.20.0 on the GPU (max|Δ| = 0 on 28 real
map-sets); T7 real GPU vs CPU float64 with the treecode on: 5·10⁻¹⁶ (M_ap),
6·10⁻¹⁵ (ξ+), 5·10⁻¹⁵ (ξ−); float32 + float64 acc vs CPU float64: ≤ 1.3·10⁻⁷.

Findings worth remembering
* Stage 2's "≥ 3× at nside 512" keep-if was unreachable (premise wrong, see
  above); kept because it is bitwise-neutral and structurally required.
* First tiled kernel was 3× *slower*: run-time indexed per-thread
  accumulators spill to thread-local memory. Compile-time unrolled
  combinations fixed it (4.5× faster, still bit-identical).
* The bitwise loader check caught a real race (pinned host buffer refilled
  while its async copy was queued); the stream simulator now reproduces it,
  and revealed a second hazard in `PinnedMapPipeline` (upload not waiting for
  queued kernels). Both fixed.
* Default float32 pair search mis-bins ~4 % of pairs at 5′ (nside 2048);
  `pair_search_precision` added, float64 automatically with the treecode.
* The Feb-2025 production pair files are **not** reproduced by a fresh
  `preprocess()` even with unmodified 4.20.0 (they contain ~28 % fewer aperture
  entries per patch — older aperture definition). M_ap from new files will
  differ from old production measurements by up to ~4 % of max|M_ap|.
* 10⁵ map-sets at nside 2048 / θ_min = 5′ / k = 2.9 ≈ 2.2 GPU-hours.

Open (not started): payload packing (§8.2), multi-map kernel (§8.3), block
shapes (§8.4), tiling of the fused-3x2pt / ds / dd kernels, coverage chunking
(§8.5), T10 (ζ level, needs non-Gaussian nside-2048 sims), float16 archives.

---

# Stage 7 results (2026-09-18): tiled + packed kernels for every pair statistic

`stage7_gpu_tiles.py`, A100 (idle), DES Y3 Q110 (917 patches), 4 source + 4
lens bins, float32 maps + float64 accumulators, host input, synchronised
wall time per map-set (median of 8). "per-row" = the 4.21 kernels before
this stage for ξ_g / ξ_t / 3x2pt (one block per bin × combination ×
orientation); "tiled" = one block per bin, every pair once
(`cuda/pair_tiles.cuh`); the fused 3x2pt path now launches the same three
pair tiles after its aperture sections.

| nside 512, 265 M pairs | per-row | tiled | tiled + packed |
|---|---|---|---|
| `get_full_tomo_shear` | 74.1 ms | 13.6 ms | 13.4 ms |
| `get_full_tomo_density` | 31.5 ms | 8.4 ms | 8.3 ms |
| `get_full_tomo_ggl` | 96.4 ms | 23.6 ms | 22.3 ms |
| `get_3x2pt_tomo` | 203 ms | 46.6 ms | 45.1 ms |

| nside 2048, k = 2.9, 592 M pairs, random maps | tiled | tiled + packed |
|---|---|---|
| `get_full_tomo_shear` | 86.7 ms | 87.7 ms |
| `get_full_tomo_density` | 49.2 ms | 50.3 ms |
| `get_full_tomo_ggl` | 129 ms | 132 ms |
| `get_3x2pt_tomo` | 251 ms | 226 ms |

Gates (24–48 patches, `bitwise_check.py` + part A of the script):
* GPU float64 vs CPU float64, every method, packed and unpacked: ≤ 1.4e-14.
* tiled vs per-row on the GPU (float32 maps + float64 acc): **bitwise** for
  M_ap, the auto combinations of ξ±, ξ_g, all of ξ_t and every output of
  `get_3x2pt_tomo` (cross terms included on this data).
* packed vs unpacked: ≤ 1.1e-14 relative (float64 maps).

---

# Stage 8 results (2026-09-18): single-pass 3x2pt tile — kept

`stage8_gpu_3x2pt_single_pass.py`, same setup as stage 7, `get_3x2pt_tomo`
from host arrays, median of 8, third of three consistent runs. "three tiles"
= stage 7 (xi+-, xi_g, xi_t one after the other); "single pass" =
`tile_multi`, one walk over the pairs for all three.

| case | three tiles | single pass | gain |
|---|---|---|---|
| nside 512, 4 source + 4 lens, unpacked | 47.7 ms | 38.7 ms | 19 % |
| same, `gc_auto_correlations_only` | 44.8 | 35.9 | 20 % |
| 4 source + 6 lens, unpacked | 66.6 | 64.2 | 4 % |
| same, gc auto only | 52.5 | 44.4 | 15 % |
| 4 + 4, packed | 46.1 | 39.0 | 15 % |
| 4 + 4, packed, gc auto only | 42.6 | 34.9 | 18 % |
| 4 + 6, packed | 62.4 | 61.6 | 1 % |
| 4 + 6, packed, gc auto only | 50.9 | 44.1 | 13 % |
| nside 2048, k = 2.9, 4 + 4, packed | 289 | 266 | 8 % |

Faster in every case, but far below the ~2x the byte count suggested: only
part of the call is the pair walk (upload, degrade, AoS fill, permuted
gathers), and with 82–120 accumulators per thread the kernel is no longer
purely bandwidth-bound (the gain shrinks as the accumulator count grows:
4 % / 1 % at 120). Used up to 120 accumulators (`_MAX_TILED_3X2PT_ACCUMULATORS`,
the largest measured); beyond that the three tiles run.

A two-pass split (xi+- with xi_t, xi_g separately) was measured in the same
runs: 41.7 / 62.5 ms unpacked, but 48.0 ms (4 + 4 packed) and 298 ms (nside
2048) — *slower* than three tiles on packed pairs. Dropped.

Gates: GPU float64 vs CPU float64 <= 1.4e-14 (packed and unpacked); single
pass vs three tiles: M_ap, M_g, xi+-, xi_g bitwise, xi_t to one rounding of
the map precision (2.5e-16 float64, 1e-7 float32 maps: the tangential shear is
-Re(gamma') of the rotated shear; nvcc merges the two expressions under
fast-math even when written separately — tried).
