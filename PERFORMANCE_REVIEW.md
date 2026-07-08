# CosmoFuse Performance Review

*Scope: full source tree at commit `53d34ee` (`src/CosmoFuse/*.py`, `src/CosmoFuse/cuda/*`).*

This review looked for performance opportunities only — no correctness, style, or API findings.
Every item cites the exact code location and was independently re-verified against the code by an
adversarial review pass (see Methodology at the end; none of the findings below were refuted).
Items marked **[measured]** were validated with micro-benchmarks (4-core Linux box, Numba 0.66,
NumPy 2.4; no GPU available, so GPU items are analytical). Impact ratings assume the intended
workload: pair geometry computed once and reused, then the `Correlation` measurement methods
called **per map** over hundreds–thousands of simulation maps, with ~5 tomographic bins (15–25
combinations), 10–30 angular bins, and hundreds–thousands of patches.

---

## Summary

The kernel layer is in good shape: the fused/vectorized tomographic kernels (CPU and GPU), the
compiled-kernel caches, the fused-path input/output buffer reuse, and the int32 index handling are
all the right design. The remaining wins cluster in five places:

1. **Per-map Python-level overhead around the kernels** — redundant sum-of-weights recomputation,
   full-array hashing for cache keys, forced GPU→host synchronizations, and full-map copies that
   happen even when they are no-ops.
2. **CPU kernel loop structure** — the pair stream is re-traversed once per tomographic
   combination, and two sections of the fused CPU kernel parallelize over only ~5 iterations.
3. **The one-time pair-finding path** — the O(N²) kernel is effectively single-threaded despite
   `parallel=True`, with worst-case O(N²) allocation.
4. **GPU data movement** — the stream/pinned-memory infrastructure that exists in `Backend` is
   never used, and several device-side staging copies are re-made per map.
5. **Two architectural opportunities** — a multi-map batch dimension in the kernels (the immutable
   pair geometry, typically more bytes per pair than the map data, is currently re-streamed from
   DRAM for every one of thousands of maps), and compaction of global HEALPix indices to a dense
   used-pixel index space (shrinking uploads and improving gather locality under a mask).

Estimated combined effect (typical tomographic CPU run): **~1.5–3× per-map throughput** from the
non-architectural items; for the one-time pair finding: **~n_cores× on many-core nodes**. GPU
per-map paths gain mainly from items 5, 6, 8, and 15 (sync stalls and redundant transfers dominate
small-kernel launches). The batching item (9) is the largest structural headroom beyond that.

---

## A. Per-map hot path — highest priority

### 1. Sum-of-weights recomputed every map in the clustering and GGL paths — high impact
`src/CosmoFuse/correlations.py:2473-2496` (`_density_density_tomo_vectorized`),
`correlations.py:2656-2670` (`_density_shear_tomo_vectorized`)

`vectorized_shear_shear` caches its sum-of-weights denominators keyed on a weight fingerprint
(`correlations.py:2171-2196`), but the density–density and density–shear paths have **no cache**:
when `sumofweights is None`, every call re-runs, per tomographic combination and orientation, a
fancy-indexed gather of two `ntotpairs`-sized temporaries plus `add.reduceat`
(`w_dev[i][inds[0]] * w_dev[j][inds[1]]`, e.g. `correlations.py:2664-2669`). With 25 lens×source
combinations and ~10⁸–10⁹ total pairs this is tens of GB of memory traffic per map — often
comparable to or larger than the correlation kernel itself — repeated identically for every map
because weights rarely change between maps. (Verification estimated this at plausibly 40–70% of
the per-map runtime of the clustering/GGL paths when callers rely on the default
`sumofweights=None`.)

**Fix:** replicate the `_tomo_sumofweights_cache` fingerprint cache for both paths (key on weight
fingerprint + `prepare_version` + combination selection, e.g. `gc_auto_correlations_only` or the
`ggl_bin_combinations` tuple). Bit-identical results. See also item 3 for making the denominators
free instead.

### 2. Weight fingerprinting hashes the full weight maps on every call — high impact
`src/CosmoFuse/correlations.py:1806-1808` (`_fingerprint_weights`)

For host arrays the cache key is `blake2b(w.tobytes())` over the entire weight stack —
`tobytes()` first makes a full copy, then blake2b scans every byte. **[measured]** For a 5-bin
nside-1024 float64 weight stack (503 MB) this costs **~1.1 s per call** (~0.17 s of it the
`tobytes()` copy). It is paid on every `vectorized_shear_shear` / `compute_shear_shear` call —
*even on cache hits*, since the hash is the lookup key. Two fingerprints are computed per
cross-correlation call.

**Fixes (cheapest first):**
- Drop `.tobytes()`: `hashlib.blake2b(memoryview(w_contiguous))` hashes the buffer directly
  (identical digest, no copy).
- Memoize the digest per array identity: key on
  `(id(w), w.__array_interface__['data'][0], w.shape, w.dtype)` so an unchanged array object hashes
  once per run. Users passing the same weight array every map then pay nothing.
- Optionally allow an explicit user-supplied `weights_token` to bypass hashing entirely
  (the API already supports passing `sumofweights` explicitly, which is the ultimate bypass —
  worth advertising in the docs).

Note the flip side (also observed): for **device** arrays the fingerprint is the data pointer
(`correlations.py:1797-1804`), so users who upload weights freshly each map get cache *misses*
despite identical values. Documenting "upload weights once, reuse the device array" would let
users hit the cache reliably.

### 3. Denominators could be accumulated inside the reduction kernels for free — high impact
`src/CosmoFuse/cuda/tomo_vectorized_xipm.cu:129-140`, `cuda/density_density_tomo_vectorized.cu:79-84`,
`cuda/density_shear_tomo_vectorized.cu:89-111`; CPU mirrors in `backend.py:257-375, 740-813`

The standalone tomographic kernels already **load and multiply the weights** for every pair
(`w_pair = weights[idx_a_bin] * weights[idx_b_bin]`) but only output numerators; the denominator
(sum of the very same `w_pair`) is then computed separately in Python via the gather+reduceat
machinery of item 1 (`_compute_tomo_sumofweights`, `correlations.py:1941-1968`). The fused 3×2pt
kernel already does it the right way — it accumulates `sum_w` alongside the numerators
(`tomo_fused_3x2pt.cu:206-216`) with `block_reduce_sum_triple`.

**Fix:** add a denominator output to the three standalone kernels (one extra accumulator + the
already-existing pair/triple block reduce) and to their Numba CPU mirrors. This eliminates the
entire separate sum-of-weights pass; combined with item 1's cache for the unchanged-weights case,
the denominators become effectively free.

### 4. CPU tomographic kernels re-traverse the pair stream once per combination **[measured]**
`src/CosmoFuse/backend.py:277-310` (`_cpu_density_density_tomo_vectorized_kernel`),
`backend.py:337-375` (density–shear), `backend.py:763-812` (`_cpu_vectorized_tomo_kernel`)

Loop order is `prange(bins) → for comb → for pair`: the pair index/rotation arrays and the map
rows are re-read from memory for each of the 15–25(×2 orientations) combinations. Interchanging to
`prange(bins) → for pair → for comb` with a small per-bin local accumulator array loads each
pair's indices/rotations once and reuses the gathered map rows (all tomo bins of a pixel are
contiguous) across combinations. **[measured]** 1.27× on the density–density kernel (20 M pairs,
15 combinations, 4 threads); the gain grows with pair count per bin (cache pressure) and with
`ncomb`, and applies to all three kernels plus the ξ±/ξ_g/ξ_t sections of the fused CPU kernel.

### 5. Forced GPU→host synchronization per combination in the normalization helpers — high impact
`src/CosmoFuse/correlations.py:1688-1689` (`_normalize_xipm_pairs`),
`correlations.py:1702-1703` (`_normalize_scalar_pairs`)

`np.ndim(self.backend.to_numpy(sumofweights_dev))` downloads the **entire** sum-of-weights array
(a blocking `cupy.asnumpy`, i.e. a device synchronization) merely to check its dimensionality, and
the scalar branch downloads it a second time for the `!= 0` test. These helpers are called in
Python loops over tomographic combinations (`correlations.py:2056-2067`, `2551-2557`, `2682-2683`)
— with 5 tomo bins, on the order of 50–100 forced pipeline stalls per map across the ξ±/ξ_g/ξ_t
paths, which defeats asynchronous kernel execution.

**Fix:** use `sumofweights_dev.ndim` (metadata, no transfer; identical on NumPy) and do the
zero-test on device with the masked-divide pattern that `_safe_div_into_device`
(`correlations.py:3084-3089`) already implements. Also batch the per-`k` loops: the divides across
all combinations can be a single broadcast masked divide over the `(ncomb, nbins_total)` arrays,
turning ~9 kernel launches × ncomb into ~3 launches. Drop the throwaway `backend.zeros`
allocations (`correlations.py:1686-1687, 1701`) that the scalar branch discards.

### 6. Full-map multiply/copy happens even when `flip_g1`/`flip_g2` are off — low impact, trivial fix
`src/CosmoFuse/correlations.py:2006-2009` (`_xipm_tomo_vectorized`),
`correlations.py:2269-2274` (`_compute_tomo_aperture_shear`)

`module.stack((g1_fac * shear_maps_dev[:, 0], g2_fac * shear_maps_dev[:, 1]), axis=1)` runs two
full-map scalar multiplies plus a stack copy **even when both factors are 1** (the default) —
producing a bit-identical copy of the input before the AoS transpose makes yet another copy.
Likewise the aperture loop materializes `g1_fac * shear_maps_arr[i, 0]` per tomo bin per map.

**Fix:** skip the scale when both factors are 1 (pass `shear_maps_dev` straight to the transpose);
when a flip is requested, fold the sign into the transpose copy or flip in-place on the staging
buffer. In the aperture loop, pass the slice directly and apply the sign inside the kernel call
(or negate once on the staged device copy).

### 7. Vectorized tomo paths rebuild their AoS/SoA staging copies every call — low impact
`src/CosmoFuse/correlations.py:1970-1975` (`_transpose_tomo_inputs_aos`),
`correlations.py:2463-2464`, `2590-2593`

`_xipm_tomo_vectorized`, `_density_density_tomo_vectorized`, and `_density_shear_tomo_vectorized`
allocate fresh `ascontiguousarray(transpose(...))` copies of all maps *and weights* on every map.
The fused 3×2pt path already solves this: it keeps persistent staging buffers in `ComputeContext`
(`_get_or_create_fused_input_buffers`, `correlations.py:813-863`) and refills them with `copyto`.
Weights in particular are usually identical across maps, so their staged SoA copies could be built
once per weight fingerprint rather than per map.

**Fix:** extend the fused-path buffer reuse to the three standalone vectorized paths; cache the
weight SoA copies keyed on the weight fingerprint. Output buffers (`out_num`,
`correlations.py:2025-2027`, `2508-2510`, `2618`) can also be cached the way
`_get_or_create_fused_output_buffers` does. Modest per-item cost, but it removes allocator/pool
churn and pairs with item 16 (payload packing).

### 8. `get_full_tomo_shear` uploads the same maps twice per map (GPU) — high impact
`src/CosmoFuse/correlations.py:2306-2335`

The aperture pass (`_compute_tomo_aperture_shear` → per-bin `get_aperture_shear`, three H2D
transfers per bin at `correlations.py:1110-1112`) and the 2PCF pass (`vectorized_shear_shear`,
which re-uploads the full stack at `correlations.py:2166-2169`) each upload the same host arrays.
With 5 bins that is ~16 separate transfers of data that could go up once.

**Fix:** upload `shear_maps`/`w` once at the top of `get_full_tomo_shear` and pass device arrays
down (both callees already accept them via `_is_backend_native_array`). Same pattern applies to
`get_full_tomo_density` and `get_full_tomo_ggl` with `return_N_ap`/`return_M_ap`.

### 9. No multi-map batching: the pair geometry is re-streamed per map — high impact, architectural
`src/CosmoFuse/correlations.py:2146` (and every other measurement entry point)

Every public measurement method accepts exactly one map set per call, and no kernel has a
map/batch dimension. Per pair, the kernels stream ~24–48 bytes of **immutable** geometry
(`ind_i`/`ind_j` int32 + `rot_i`/`rot_j` complex64/128) while the map-side data they gather is
mostly cache-resident — so across thousands of maps, the dominant DRAM stream is the same
geometry read over and over. Adding a small batch dimension (even 4–8 maps: gather from
`map[pix, tomo, map_idx]` inside the existing pair loop, one extra accumulator row per map)
amortizes the geometry stream across the batch, on CPU and GPU alike. It composes with item 4
(amortizing over combinations), and for the common "same weights every map" case the denominators
are shared across the whole batch. This is the largest *structural* headroom in the measurement
path; since it changes the API shape (`shear_maps: (nmaps, nzbins, 2, npix)`), it fits best as a
new `*_batched` entry point.

### 10. Global HEALPix indices force full-sky uploads and wide gathers — medium impact
`src/CosmoFuse/correlations.py:2168` (upload sites); indices created at `correlations.py:189-190`,
`pair_geometry.py:94-97`

Pair and aperture indices store global HEALPix pixel ids, so per-map uploads must be full
`(…, npix)` arrays even when the mask leaves only a fraction of the sky used, and kernel gathers
stride across the full-sky array. A one-time compaction in `prepare()` — remap
`ind_i/ind_j/Q_inds` through `map_inds` to a dense `[0, n_used)` index space and accept/stage
`map[..., map_inds]`-compacted maps — shrinks H2D traffic and the gather working set by the sky
fraction (~2.5× for a 40%-sky survey). Compaction cost is one fancy-index per map, or zero if
users pass already-compact maps (an API option worth exposing anyway).

---

## B. One-time pair precomputation

### 11. The O(N²) pair-finding kernel is effectively single-threaded **[measured]**
`src/CosmoFuse/correlations.py:88-214` (`_compute_pairs_impl`), compiled at `correlations.py:226`

The kernel is compiled with `njit(parallel=True)`, but the dominant double loop uses plain
`range` with a sequentially incremented `out_idx` — Numba cannot parallelize it, so `parallel=True`
only vectorizes the trivial O(N) setup (lines 105–111). The O(N²) work — up to ~10¹⁰ candidate
pairs per patch at high nside — runs on one core, per patch, serially over patches
(`pair_geometry.py:119-124`).

**Fix:** two-pass structure — pass 1 `prange` over rows counting accepted pairs per row (only the
dot product + range test), exclusive prefix sum for per-row write offsets, pass 2 `prange` over
rows writing at precomputed offsets. No atomics, deterministic output order, bit-identical results.
**[measured]** 1.9× on 4 cores (npts=6000; imperfect scaling comes from the triangular row lengths
— pairing row *i* with row *N−2−i* per chunk balances it); on a 32–64-core HPC node this is the
difference between hours and minutes. Parallelizing the outer patch loop (e.g. process patches in
worker processes) stacks on top for many-patch configurations.

### 12. Worst-case O(N²) allocation in the same kernel
`src/CosmoFuse/correlations.py:115-122`

Seven arrays of `npts·(npts−1)/2` elements are allocated regardless of how many pairs survive the
θ-window cut (`bin_indices` even as int64 though `bin_idx < ~30`). At npts ≈ 10⁵ that is hundreds
of GB of virtual allocation for a few percent survival rate. The two-pass fix of item 11 sizes the
output exactly and eliminates this; independently, `bin_indices` fits in int16.

### 13. Per-bin split via `np.where` immediately undone by the caller
`src/CosmoFuse/pair_finder.py:114-122` (split), `src/CosmoFuse/pair_geometry.py:106-113` (re-concatenation)

After stable-sorting pairs by bin, `get_pairs_patch` splits them into `nbins` per-bin arrays with
one full `np.where(bin_indices == b)` scan + fancy-indexed copy per bin — and
`get_pairs_helper` then concatenates them straight back into one flat `(2, total)` array. Since the
array is already sorted by bin, `np.bincount(bin_indices, minlength=nbins)` gives the per-bin
counts and the flat sorted arrays can be used directly: removes `nbins` scans and two full copies
of the pair data per patch. Also minor: the bin-search inside the pair kernel
(`correlations.py:140-143`) is a linear scan over edges; a binary search (or the closed-form
geomspace inverse) is branch-lighter in the innermost loop.

### 14. Cheap hygiene in the same path (negligible–low individually)
- `pair_geometry.py:152-171`: `cos(dec)` computed three times, `sin(dec)` twice, `cos(Δra)` twice
  per patch — hoist into locals (bit-identical).
- `pair_geometry.py:93-98` + `correlations.py:105-108`: pixels go pix → ang → ra/dec → cos/sin
  back to unit vectors; `hp.pix2vec` yields the vectors directly (equal to ~1 ULP).
- `correlations.py:1219-1237` (`prepare()`): `np.empty` instead of `np.zeros` for staging arrays
  that are fully overwritten; preallocate `temp_bins_tot` with its leading slot instead of
  `np.concatenate`; freeing each per-patch list slot as it is copied lowers peak host RSS with
  `release_host_pairs=True`. (One-time cost, verified as minor.)

---

## C. GPU-specific

### 15. Streams and pinned memory exist but are never used — medium impact
`src/CosmoFuse/backend.py:1411-1463` vs. all call sites

`Backend.create_stream` / `use_stream` / `synchronize_stream` / `alloc_pinned` are implemented
(and advertised in docstrings for exactly this purpose) but no production code path calls them:
every per-map upload is a synchronous pageable-memory `cupy.asarray` on the default stream
(`correlations.py:2915-2918`, `2166-2169`, `2585-2588`). For a per-map pipeline this forfeits
transfer/compute overlap — with pinned double-buffering, map *k+1*'s upload can hide entirely
behind map *k*'s kernels.

**Fix:** persistent pinned staging buffer sized to one map set; copy incoming host maps into it and
`to_device(..., stream=transfer_stream)` while the previous map computes; compute stream waits on
the transfer event. This pairs naturally with items 7–8 (upload once, reuse staged buffers).

### 16. Pair-geometry arrays re-streamed from DRAM once per combination×orientation — high impact
`cuda/tomo_vectorized_xipm.cu:102-141` (and the same pattern in the dd/ds/fused kernels)

Each `(bin, comb_ori)` block reads the identical `ind_i/ind_j/rot_i/rot_j` stream (~24–40
bytes/pair). With 30 combination-orientations the pair geometry is fetched up to 30× per map
(L2 absorbs some of it; on large runs it does not fit). Tiling combinations per block — a register
array of accumulators for e.g. 4–8 `comb_ori` per block, loading each pair's geometry once —
divides that traffic by the tile factor. This is the GPU analogue of item 4 and is the main
kernel-level headroom on bandwidth-bound GPUs. (Item 9's map-batching multiplies the benefit.)

### 17. Interleave weights with the map payload — medium impact
`cuda/tomo_vectorized_xipm.cu:122-130` and the staging in `correlations.py:1970-1975`

Per pair member the kernels gather the shear pair `(g1,g2)` and then the weight from a *separate*
array — two random cache lines per pixel instead of one. Since the AoS staging copies are already
rebuilt per map (item 7), packing `(g1, g2, w, pad)` into one 4-wide element (single aligned
`float4`/`double2×2` load) halves the random-access line count. Combine with item 7 so the packing
cost is paid once per weight set, not per map.

### 18. Non-tomographic GPU 2PCF paths materialize per-pair arrays — high impact on that API
`src/CosmoFuse/correlations.py:1356-1378`, `1452-1476`, `1536-1555`, `1638-1663`

`compute_shear_shear`/`compute_density_density`/`compute_density_shear` on CuPy allocate
`ntotpairs`-sized outputs (with a wasted `zeros` memset — every element is overwritten), run the
per-pair ElementwiseKernel, then `add.reduceat` — i.e. a full write+read of pair-sized arrays where
the tomographic kernels reduce in shared memory and write only `nbins_total` values. Routing the
single-pair case through the vectorized kernels with `nzbins=1` (reshape to `(1, …)`) removes the
per-pair materialization entirely; at minimum, use `empty` and reuse a cached scratch buffer.

### 19. Fused 3×2pt kernel launches a dense max-shaped grid where most blocks no-op — low impact
`src/CosmoFuse/backend.py:1211-1214`; kernel guards at `cuda/tomo_fused_3x2pt.cu:99,133,164,228,275`

`blocks = (max_x, max_y, 5)` with `max_x = max(nbins_total, npatches)` and `max_y` the max across
all five section widths. Example (1000 patches, 30 angular bins, 5+5 tomo bins): the aperture
sections need `1000×5` blocks but receive `30000×30`; ~4.4 M of ~4.5 M launched blocks exit at
their guard. Block launch overhead is small individually, but this is pure waste that grows with
problem size.

**Fix:** flatten to a 1D grid with per-section block-count offsets (decode `(z, x, y)` from the
flat index), or launch the five sections as separate kernels on independent streams — which also
removes the false serialization between sections that share no data.

### 20. Aperture statistics on GPU: per-pair materialization + per-bin launches
`src/CosmoFuse/correlations.py:1110-1123`, `1170-1182`, loop at `2268-2276`

`get_aperture_shear`/`get_aperture_density` allocate two `len(Q_inds_flat)`-sized zeroed buffers
(potentially 10⁷–10⁸ elements) per call, then reduceat — and the tomo wrapper calls this once per
bin. The fused kernel's z=0/z=1 sections (`tomo_fused_3x2pt.cu:98-152`) already implement the
block-reduced version; exposing them as a standalone kernel (grid: patches × bins) removes both
the scratch traffic and the per-bin launch loop.
Related (conditional, medium when hit): if device aperture buffers were invalidated and
`prepare()` is not re-run, the fallback branches (`correlations.py:1101-1109`, `1163-1169`)
re-upload the full aperture geometry **every call** without repopulating `compute_context` — add
`self._prepare_aperture_device_buffers()` there instead of building throwaway local copies.

### 21. User-supplied device `sumofweights` round-trips GPU→host→GPU
`src/CosmoFuse/correlations.py:1713`, `1747`, `1813-1825`, `2672`

The `_normalize_*_sumofweights` helpers start with `to_numpy(sumofweights)` (blocking download)
and re-upload — even when the caller passed a device array of already-correct shape, which is the
documented fast path for bypassing the cache. Short-circuit on
`_is_backend_native_array(...)` + shape match: validate from metadata, reshape on device, no
transfer.

---

## D. CPU kernel details

### 22. Fused CPU kernel parallelizes two sections over ~5 iterations — high impact on many-core
`src/CosmoFuse/backend.py:961` (`prange(n_shear)`), `backend.py:978` (`prange(n_density)`),
`backend.py:995/1044` (`prange(2*n_comb)`)

The M_ap/M_g sections of `_cpu_3x2pt_tomo_fused_kernel` parallelize over tomo bins (~5) with all
patches serial inside — on a 32-core node, 27 cores idle through those sections. The ξ sections
parallelize over `2*ncomb` (~30), where the odd auto-bin iterations `continue` immediately
(`backend.py:1001-1002`), wasting ~1/6 of the slots and unbalancing the rest.

**Fix:** `prange` over flattened products — `n_tomo*n_patches` for the aperture sections,
`comb_ori*nbins_total` (matching the CUDA grid) for the ξ sections. Item 4's loop interchange
applies to the ξ sections here too.

### 23. CPU ξ± kernels compute imaginary parts every caller discards — medium impact
`src/CosmoFuse/backend.py:672-684` (cross), `731-737` (auto), `771-812` (vectorized tomo)

The cross kernel maintains 8 accumulators (4 imaginary), the auto kernel 4 (2 imaginary), and the
vectorized tomo kernel uses full complex multiplies and complex128 accumulators — all callers
immediately take `np.real(...)` (`correlations.py:1533-1534`, `2118-2119`). The fused kernel
(`backend.py:1028-1036`) already shows the real-only formulation: roughly halves the FLOPs and
accumulator pressure in the inner loop and makes outputs real (halving their size).

### 24. Scalar density CPU paths stream the pair list twice — medium impact
`src/CosmoFuse/correlations.py:1331-1353`, `1424-1449`

`compute_density_density`/`compute_density_shear` on CPU call their kernel twice (AB then BA with
swapped index arrays), re-streaming `ind_i/ind_j` (and `exp2phi`) and re-gathering map values. A
fused kernel accumulating both orientations in one pass (as `_cpu_xipm_cross_corr_kernel` already
does for four outputs) halves the traffic.

---

## E. Memory & I/O

### 25. Unbounded sum-of-weights cache — medium (memory, long runs)
`src/CosmoFuse/correlations.py:1938`

`_get_xipm_sumofweights` inserts every distinct weight fingerprint into a dict that is only
cleared on re-`prepare()`. With per-map (per-realization) weights this grows by several device
arrays per map across thousands of maps — a slow GPU/host memory leak. Cap it (small LRU; even
size 8 covers realistic reuse).

### 26. HDF5 layout: 8 datasets × n_patches; load does thousands of tiny reads then double-copies
`src/CosmoFuse/io_handler.py:39-50` (save), `52-105` (load), `correlations.py:1211-1237` (`prepare`)

One group per patch with 8 small datasets each ⇒ ~10⁴ h5py accesses for large runs; `load_pairs`
builds per-patch Python lists which `prepare()` immediately re-copies into flat arrays (2× peak
host memory, second full copy pass). Since `prepare()`/`prepare_aperture_flat` already define the
flat layout, storing consolidated flat datasets + offset arrays (with a format-version attribute
and a legacy reader) makes loading a handful of bulk `read_direct` calls straight into the final
arrays. Optional: chunk+compress the pair datasets (`lzf` or `gzip-1` with byte-shuffle) — pair
indices compress well and loads become I/O-bound less often on network filesystems.

### 27. Aperture device buffers upcast float32 geometry to float64
`src/CosmoFuse/correlations.py:789-811`

`Q_cos/Q_sin/Q_val/Q_patch_area` are stored at `rotation_dtype` (float32 by default) but are
widened to `map_dtype` (float64 default) before upload — 2× persistent device memory and transfer
for data whose source precision is float32 anyway. Template the aperture kernels on a second dtype
(promotion to `T` inside the kernel is exact).

### 28. `map_precision` default is float64 for bandwidth-bound kernels — medium impact
`src/CosmoFuse/correlations.py:290`

All pair kernels are gather/bandwidth-bound, so float64 maps cost ~2× the traffic of float32 — and
on consumer GPUs, 1/32–1/64 the double-precision FLOP rate. The infrastructure for float32 already
exists (`map_precision="float32"`); consider (a) defaulting the docs/examples to float32 maps, and
(b) a mixed mode — float32 loads with float64 *accumulators* (template parameter `ACC` in the .cu
kernels; the CPU kernels already accumulate in double) — which preserves reduction accuracy at
half the traffic.

### 29. Small cleanups (low individually)
- `correlation_helpers.py:136-145`: per-triplet means recomputed inside the 35-triplet loop
  (only 5 distinct centers / 15 annuli) — hoist `central.mean(axis=2)` / `annulus.mean(axis=2)`
  out (bit-identical; post-processing stage, not per-map).
- `correlations.py:387-401` (`__getstate__`): pickling ships all per-patch host pair lists plus
  the full `map_mask`; workers then re-`prepare()` from scratch. Drop `map_mask` (rebuildable from
  `map_inds`) and pickle the consolidated flat arrays (protocol-5 buffers) — or just the pair-file
  path.
- `correlations.py:3260-3292` (`get_3x2pt_tomo`): unit-weight arrays re-allocated per call when
  `weights=None`; cache per shape/dtype in `ComputeContext`.
- First-call latency: the ~10 `@njit(parallel=True)` kernels JIT lazily on the first map
  (**[measured by verification]** ~9–20 s first process; ~0.4 s on later runs thanks to
  `cache=True`), and each CUDA template instantiation NVRTC-compiles its full `.cu` on first
  launch. A small `warmup()` API (tiny synthetic inputs after `prepare()`) would move this out of
  the measured path; note also that failed NVRTC compilations are not negatively cached
  (`backend.py:429-440` and siblings), so a permanently failing configuration would retry the
  compile per call — harmless today because the wrappers raise rather than fall back, but worth a
  `kernel_cache[key] = None` sentinel if a fallback is ever added.

---

## Suggested implementation order

| Order | Items | Why |
|-------|-------|-----|
| 1 | 2 (fingerprint memoization), 5 (`.ndim` sync fix), 6 (skip no-op flips) | Few-line changes, immediate per-map wins, zero numerical risk |
| 2 | 1 + 3 (denominators in-kernel + cache) | Biggest non-architectural per-map win; touches kernels + wrappers |
| 3 | 11 + 12 + 13 (parallel pair finding) | Transforms the one-time setup at scale; self-contained |
| 4 | 7 + 8, 20, 21 (staging/upload reuse) | GPU data-movement cleanup |
| 5 | 4 + 22 + 23 + 24 (CPU kernel loop structure) | Measured wins; requires kernel re-validation |
| 6 | 15 (streams/pinned), 16–19 (GPU kernel-level) | Larger engineering effort; profile-guided |
| 7 | 9 + 10 (multi-map batching, index compaction) | Largest structural headroom; API additions, do after the above |
| 8 | 25–29 | Housekeeping |

## What is already good

- Fused 3×2pt kernel design (single launch, in-kernel denominators, block reductions) — the
  standalone paths should converge toward it (items 3, 7, 20).
- Compiled-kernel caches keyed on dtype/bin-count; `cache=True` on all Numba kernels;
  `_get_pairs_numba_kernel` factory cache preventing per-instance recompilation.
- Fused-path input/output device buffer reuse across maps (`_get_or_create_fused_*_buffers`).
- int32 pair indices when `npix < 2³¹`, CSR bin offsets, bin-sorted pair layout.
- ξ± sum-of-weights fingerprint cache (the pattern items 1–2 want to generalize).

---

## Methodology

Full manual read of all source files, plus a multi-agent review: 7 finder passes with distinct
lenses (algorithmic, memory, Numba/CPU kernels, GPU/CUDA, I/O, orchestration, precompute)
produced 64 raw findings, deduplicated to 33; each finding was then checked by two independent
adversarial agents (one attempting to refute the claim against the code, one rating realistic
impact), followed by a completeness pass that contributed 5 further verified findings (items 9,
10, and the first-call-latency notes). None of the 38 verified findings were refuted; two were
downgraded as overstated and are reported here at their verified severity. The three claims
marked **[measured]** were additionally validated with micro-benchmarks in this environment.
