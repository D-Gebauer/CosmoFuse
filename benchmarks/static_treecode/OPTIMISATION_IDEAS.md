# Optimisation ideas (parked 2026-09-17)

Profile they refer to — A100, DES Y3, 917 patches, 4 source bins, float32 maps
+ float64 accumulators, nside 2048, 5–175′, k = 2.9, device-resident input:
~65–75 ms per map-set = ξ± kernel ~50 ms (592 M pairs, ~120 B moved per pair:
24 B geometry + 2 × 48 B pixel data) + degrade ~10 ms + aperture ~2 ms + rest.
Gains below are estimates, none is measured.

| # | idea | expected gain | cost / risk | status |
|---|---|---|---|---|
| 1 | **Payload packing + patch-local rows** (24 → 8 B per pair; uint16 angles, uint16 patch-local indices; each patch's rows contiguous → cache-local gathers) | pair memory ÷3; speed: unknown, likely the largest single item | not bit-identical (angle quantisation 2π/65536); extra per-map gather | **done** (4.21, stage 6): memory ÷2.8, speed neutral |
| 2 | Tile the fused-3x2pt ξ± section and the density-shear / density-density kernels like `gpu_tiled_tomo_reduce_xipm` | ~4× on those paths | combinations must be compile-time (run-time indexed accumulators spill → 3× slower); ds with custom `ggl_bin_combinations` needs a fallback | **done** (4.21, stage 7): shared tiles in `pair_tiles.cuh`, 3.8–4.4× on ξ_g / ξ_t / 3x2pt; subsets gathered from the canonical tile |
| 3 | Degrade as one custom kernel writing straight into the kernel-layout (AoS) buffer; sign flip inside the pair kernel; accept AoS row-space input | ~10 → ~2 ms at nside 2048; fewer temporaries | moderate | |
| 4 | Shape noise generated on the device (seeded per realisation) when many maps are noise realisations of few signal maps | removes the per-map upload for those | pipeline-dependent | |
| 5 | float16 map archives / uploads | halves disk + H2D | needs a parity test on the final data vector | |
| 6 | Multi-GPU by patch range (`load_pairs(start_ind, stop_ind)` already slices everything incl. the treecode geometry) | linear | none in the library | |
| 7 | Different block sizes / merged blocks for the tiny coarse bins | few % | low | |
| 8 | Multi-map kernel (M map-sets per pass) | ≤ ~20 % (geometry is only 24 of ~120 B per pair); M × 60 accumulators cannot stay in registers | high | **rejected**: VRAM is filled with one map's geometry anyway |
| 9 | Per-statistic k (finer levels for ξ−) | recovers ≤ 5 % S/N in some ξ− bins | 4× pair memory, second kernel pass | **rejected**: one global k |
| 10 | Explicit per-bin nside override (bump only the bins with k_eff ≈ k) | targeted accuracy at small memory cost | trivial | offered, not requested |
| 11 | Pack pairs already at pair-finding time / in the file (host RAM and disk ÷3) | host memory for k ≳ 5.8 | files no longer exact | after #1 if host RAM becomes the limit |
| 12 | Spatially sorted pair search instead of O(N²) per patch | preprocessing only (67 s today) | low value | |
| 13 | Single-pass 3x2pt tile (ξ± + ξ_g + ξ_t accumulated in one walk over the pairs; today three passes, 46 ms = 14 + 8 + 24 at nside 512) | ≤ ~1.5× on `get_3x2pt_tomo` (the ds pass already gathers both fields) | ~200 registers/thread for 4 + 4 bins → 1 block/SM, poor latency hiding; may end slower | measure before building |
