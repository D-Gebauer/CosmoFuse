# Optimisation ideas (parked 2026-09-17, reviewed 2026-09-18)

Profile they refer to — A100, DES Y3, 917 patches, 4 source bins, float32 maps
+ float64 accumulators, nside 2048, 5–175′, k = 2.9, device-resident input:
~65–75 ms per map-set = ξ± kernel ~50 ms (592 M pairs, ~120 B moved per pair:
24 B geometry + 2 × 48 B pixel data) + degrade ~10 ms + aperture ~2 ms + rest.
Gains below are estimates unless marked measured.

| # | idea | expected gain | cost / risk | status |
|---|---|---|---|---|
| 1 | **Payload packing + patch-local rows** (24 → 8 B per pair; uint16 angles, uint16 patch-local indices; each patch's rows contiguous → cache-local gathers) | pair memory ÷3; speed: unknown, likely the largest single item | not bit-identical (angle quantisation 2π/65536); extra per-map gather | **done** (5.0, stage 6): memory ÷2.8, speed neutral |
| 2 | Tile the fused-3x2pt ξ± section and the density-shear / density-density kernels like `gpu_tiled_tomo_reduce_xipm` | ~4× on those paths | combinations must be compile-time (run-time indexed accumulators spill → 3× slower); ds with custom `ggl_bin_combinations` needs a fallback | **done** (5.0, stage 7): shared tiles in `pair_tiles.cuh`, 3.8–4.4× on ξ_g / ξ_t / 3x2pt; subsets gathered from the canonical tile |
| 3 | Degrade as one custom kernel writing straight into the kernel-layout (AoS) buffer; sign flip inside the pair kernel; accept AoS row-space input | measured headroom: 0.94 → ~0.2 ms at nside 512 / k = 2.9, i.e. **~17 % of the device-resident call**; ~10 → ~2 ms at nside 2048 | new CUDA kernel + numpy twin in a machine-precision path | **top remaining lever**, not done |
| 4 | Shape noise generated on the device (seeded per realisation) when many maps are noise realisations of few signal maps | removes the per-map upload for those | pipeline-dependent | **rejected** (2026-09-18): CosmoFuse measures, the forward model produces the data |
| 5 | float16 map archives / uploads | halves disk + H2D | needs a parity test on the final data vector | parity **measured** (T11, see below): ζ changes by ≤ 1.2e−4 σ_patch — safe. Writing the archive is the forward model's job; no library change needed |
| 6 | Multi-GPU by patch range (`load_pairs(start_ind, stop_ind)` already slices everything incl. the treecode geometry) | linear | none in the library | **nothing to do in CosmoFuse** — it is a driver-script pattern |
| 7 | Different block sizes / merged blocks for the tiny coarse bins (§8.4) | few % | low | **measured, not worth it**: nsys gives 45 ps/pair for the tiled ξ± kernel at k = 2.9 vs 37 ps/pair at full resolution, so the whole small-block penalty is ~20 % of the pair kernel = ~0.37 ms of a 4.4 ms call |
| 8 | Multi-map kernel (M map-sets per pass) | ≤ ~20 % (geometry is only 24 of ~120 B per pair); M × 60 accumulators cannot stay in registers | high | **rejected**: VRAM is filled with one map's geometry anyway |
| 9 | Per-statistic k (finer levels for ξ−) | recovers ≤ 5 % S/N in some ξ− bins | 4× pair memory, second kernel pass | **rejected**: one global k |
| 10 | Explicit per-bin nside override (bump only the bins with k_eff ≈ k) | targeted accuracy at small memory cost | trivial | offered, not requested; T10 shows k ≈ 3 needs no rescue |
| 11 | Pack pairs already at pair-finding time / in the file (host RAM and disk ÷3) | host memory for k ≳ 5.8 | files no longer exact | after #1 if host RAM becomes the limit — it has not |
| 12 | Spatially sorted pair search instead of O(N²) per patch | preprocessing only (31 s at nside 512, 67 s at 2048) | low value | not done, and not worth it while preprocessing is once per geometry |
| 13 | Single-pass 3x2pt tile (ξ± + ξ_g + ξ_t accumulated in one walk over the pairs; today three passes, 46 ms = 14 + 8 + 24 at nside 512) | ≤ ~1.5× on `get_3x2pt_tomo` (the ds pass already gathers both fields) | ~200 registers/thread for 4 + 4 bins → 1 block/SM, poor latency hiding; may end slower | **done** (stage 8): 8–20 % for 4 + 4 bins, 1–4 % at 120 accumulators; two-pass split rejected (slower on packed pairs) |
| 14 | **Batch the per-combination normalisation** (the wrappers looped over combinations, issuing ~12 tiny cupy kernels each; the CPU could not keep the GPU fed) | — | none: same arithmetic on stacked arrays | **done** (2026-09-18): `xipm_total` 4.40 → 3.03 ms, device-resident call 5.84 → 4.40 ms at nside 512 / k = 2.9 (−25 %); bitwise identical on the A100 over 56 real map-sets |

## Where the time goes now (A100, nside 512, Q110, 917 patches, 7 bins, 4 source bins)

`stage5_profile_2048.py --nside 512 --theta-min 15 --nedges 9 --aperture-nside 0 --pairs ""`,
`get_full_tomo_shear`, median of 20 calls, after #14:

| phase | k = 2.9 (39.0 M pairs) | full resolution (265.4 M pairs) |
|---|---|---|
| whole call, host row-space input | 7.8 ms | 13.6 ms |
| whole call, device-resident input | **4.40 ms** | **11.43 ms** |
| H2D of the map-set (11.8 MB, pageable) | 2.2–3.1 ms | 2.2–3.1 ms |
| degrade (`_expand_rows`, pairs) | 0.94 ms | 0.001 ms |
| aperture pass (total) | 1.10 ms | 1.12 ms |
| ξ± pass (total) | 3.03 ms | 10.13 ms |
| — of which the tiled ξ± kernel (nsys) | 1.77 ms | ~9.9 ms |
| device memory pool | 1.4 GB | 6.7 GB |

Reading, in order of size:

1. **The upload is the largest single item of a host-input call** and it runs
   at ~12 GB/s because the source is pageable. `PinnedMapPipeline` /
   `RowSpaceMapLoader` already fix this by staging through pinned memory and
   overlapping it with the previous map-set — use them for production runs;
   the device-resident column is what they deliver.
2. **The aperture pass is 25 % of the device-resident call at k = 2.9** and is
   untouched by the treecode (0.81 ms of kernel for 5 θ_Q = 550′ discs).
   `aperture_nside` is the knob for it, at the price of a slightly different
   M_ap; nobody has measured how much that changes ζ.
3. **The degrade is 21 %** — idea #3.
4. The pair kernel itself is only 40 % of the call at k = 2.9 and is close to
   bandwidth-bound; at nside 2048 it is still the dominant term.
