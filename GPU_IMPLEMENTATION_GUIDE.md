# GPU/CUDA Optimization Implementation Guide

This is the exact implementation plan for the CUDA-side findings of
`PERFORMANCE_REVIEW.md` that were deferred because the implementation
environment had no GPU. All Python-side groundwork (device-buffer reuse,
upload-once wrappers, sync-free normalization, sum-of-weights LRU caches,
NVRTC negative caching) is already merged — this guide only touches the
`.cu` kernels, their `_build_cupy_*` wrappers in `src/CosmoFuse/backend.py`,
and the GPU branches of `src/CosmoFuse/correlations.py`.

Anchors refer to the tree as of the commit that adds this guide
(`src/CosmoFuse/backend.py`: builders at lines ~469 (dd), ~547 (ds),
~900 (ξ±), ~1223 (fused); `src/CosmoFuse/correlations.py`:
`_xipm_tomo_vectorized` ~2189, `_density_density_tomo_vectorized` ~2689,
`_density_shear_tomo_vectorized` ~2834, `_compute_3x2pt_tomo_fused` ~3185,
`_compute_tomo_aperture_shear` ~2468, `_prepare_aperture_device_buffers`
~847). If lines have drifted, search for the function names.

Requirements for every step, non-negotiable:

1. **Prove the speedup**: run `benchmarks/bench_gpu_parity.py` before and
   after each step on the same GPU, interleaved (machine state drifts —
   this bit us on CPU), and keep the step only if the affected paths get
   faster.
2. **Prove correctness**: the same script checks every public measurement
   path against the CPU reference (which is validated against treecorr) on
   identical inputs and identical pairs. Scale-relative tolerance:
   `max|Δ|/rms ≤ 1e-8` for float64 maps, `≤ 2e-4` for float32 maps.
   The full pytest suite must stay green (it runs the fake-CuPy launch-path
   tests even without a GPU).

---

## Stage 0 — set up the harness (do this first, before any kernel change)

On the GPU machine:

```bash
pip install -e .[gpu,dev]
python -m pytest tests/ -q                      # must be green
python benchmarks/bench_gpu_parity.py --device 0 --out baseline.json
```

`bench_gpu_parity.py` computes the CPU reference, saves a shared pair file,
runs every measurement path on the GPU with identical inputs, checks parity,
and records steady-state timings. **If parity fails at Stage 0, stop** —
the unmodified GPU path has a pre-existing problem; fix that first.

Also compile-smoke every template eagerly so NVRTC errors surface with a
stack trace instead of a warning + `RuntimeError` later:

```python
# scripts/compile_smoke.py (run after every .cu edit)
import cupy, numpy as np
from CosmoFuse.backend import get_backend
b = get_backend(0)
z = lambda *s: cupy.zeros(s, dtype=cupy.float64)
i32 = lambda n: cupy.zeros(n, dtype=cupy.int32)
c64 = lambda n: cupy.zeros(n, dtype=cupy.complex64)
off = cupy.asarray(np.array([0, 1], dtype=np.int64))
comb = cupy.asarray(np.array([0], dtype=np.int32))
# one launch per kernel family with 1 pair / 1 bin / 1 comb; add every
# (dtype, nbins) combination you use in production
assert b.xipm_tomo_vectorized_kernel(z(2, 1, 2), z(2, 1), i32(1), i32(1),
    c64(1), c64(1), off, comb, comb, z(2, 2, 1), z(2, 1))  # (+out_den after G1)
```

---

## G1 — accumulate denominators inside the three standalone kernels
*(review item 3, GPU half; removes the last per-map gather+reduce work and
the need for the GPU sum-of-weights LRU caches added as item 1)*

The kernels already load and multiply the weights per pair; one extra
accumulator plus the existing `block_reduce_sum_*` helpers makes the
denominators free.

### G1.1 `src/CosmoFuse/cuda/tomo_vectorized_xipm.cu`

Signature: add one output after `out_num`:

```cuda
    T* out_num,              /* numerators [xi+ block | xi- block]         */
    T* out_den,              /* weight sums, one row per comb_ori          */
    const int ncomb,
```

Body: replace the accumulator/reduction section (currently
`T sum_p = (T)0.0; T sum_m = (T)0.0;` … end of kernel) with:

```cuda
    T sum_p = (T)0.0;
    T sum_m = (T)0.0;
    T sum_w = (T)0.0;

    for (long long tid = start + lane; tid < stop; tid += BLOCK_SIZE) {
        /* ... existing loads and rotation math unchanged ... */
        T w_pair = weights[idx_a_bin] * weights[idx_b_bin];
        sum_w += w_pair;
        sum_p += w_pair * (b_R * a_R + b_I * a_I);
        sum_m += w_pair * (b_R * a_R - b_I * a_I);
    }

    block_reduce_sum_triple<T>(sum_p, sum_m, sum_w, &sum_p, &sum_m, &sum_w);

    if (lane == 0) {
        const long long out_p_idx = ((long long)comb_ori) * nbins_total + bin_flat;
        const long long out_m_idx = ((long long)(2 * ncomb + comb_ori)) * nbins_total + bin_flat;
        out_num[out_p_idx] = sum_p;
        out_num[out_m_idx] = sum_m;
        out_den[out_p_idx]  = sum_w;
    }
```

Note the early-`return` for auto-combination B→A blocks stays; those rows
are never read. Because of this, `out_num`/`out_den` **must stay
zero-initialised** (they are: `backend.zeros`) — do not switch them to the
`_get_pair_scratch` empty buffers.

### G1.2 `src/CosmoFuse/cuda/density_density_tomo_vectorized.cu`

Add `T* out_den` after `out_num`. In the loop:

```cuda
    T sum_val = (T)0.0;
    T sum_w   = (T)0.0;
    for (...) {
        /* ... index math unchanged ... */
        const T w_pair = weights[base_a] * weights[base_b];
        sum_w   += w_pair;
        sum_val += w_pair * density[base_a] * density[base_b];
    }
    block_reduce_sum_pair<T>(sum_val, sum_w, &sum_val, &sum_w);
    if (lane == 0) {
        const long long out_idx = ((long long)comb_ori) * nbins_total + bin_flat;
        out_num[out_idx] = sum_val;
        out_den[out_idx] = sum_w;
    }
```

### G1.3 `src/CosmoFuse/cuda/density_shear_tomo_vectorized.cu`

Add `T* out_den` after `out_num`. Accumulate `sum_w += w_ab; ... sum_w += w_ba;`
next to the two existing `sum_val +=` lines (the CPU convention: the ξ_t
denominator is the A→B **plus** B→A weight sum), reduce with
`block_reduce_sum_pair`, write `out_den[comb_idx * nbins_total + bin_flat]`.

### G1.4 Wrappers in `backend.py`

In each of `_build_cupy_tomo_vectorized_kernel`,
`_build_cupy_density_density_tomo_vectorized_kernel`,
`_build_cupy_density_shear_tomo_vectorized_kernel`: add `out_den: Any`
to the inner wrapper signature (after `out_num`) and insert `out_den` into
the launch tuple immediately after `out_num`. Template names and cache keys
are unchanged.

### G1.5 Orchestrator (`correlations.py`)

* `_xipm_tomo_vectorized` (GPU branch, ~line 2230): allocate
  `out_den = self.backend.zeros((2 * nzbin_combs, nbins_total), dtype=map_backend_dtype)`,
  pass it to the kernel, and after a successful launch:

  ```python
  if sumofweights_dev is None:
      sumofweights_dev = module.stack((out_den[0::2], out_den[1::2]), axis=0)
  ```

  Then in `vectorized_shear_shear`, delete the GPU fingerprint/cache block:
  the `elif sumofweights is None:` branch collapses to
  `sumofweights_dev = None` for **both** backends (the CPU branch already
  does this). `_weights_fingerprint_source` plumbing can stay (harmless) or
  be removed together with `_tomo_sumofweights_cache`.

* `_density_density_tomo_vectorized` (GPU branch): allocate/pass `out_den`
  the same way; when `sumofweights is None` use
  `den_ab, den_ba = out_den[0::2], out_den[1::2]` and normalize
  `auto: num_ab[k]/den_ab[k]`, `cross: half*(num_ab[k]/den_ab[k] + num_ba[k]/den_ba[k])`
  via the existing `_normalize_scalar_pairs`. Delete the
  `_compute_dd_sums` closure and its `_sumofweights_cache_get_or_compute`
  call — dead after this change.

* `_density_shear_tomo_vectorized` (GPU branch): `out_den` shape
  `(nzbin_combs, nbins_total)`; `sum_total = out_den` when
  `sumofweights is None`; delete `_compute_ds_sums` + cache call.

### G1.6 Tests to update

`tests/test_backend.py`: `test_cupy_tomo_vectorized_*`,
`test_cupy_density_density_tomo_vectorized_kernel_success_and_source`,
`test_cupy_density_shear_tomo_vectorized_kernel_*` — the fake kernels
receive one extra array in the launch tuple; adjust the recorded-arguments
assertions. `tests/test_correlations_coverage.py`:
`test_vectorized_shear_shear_same_w_reuses_cache` /
`_changed_w_recomputes` assert the fingerprint-cache behaviour that G1.5
removes — replace them with the CPU-style assertions (sum-of-weights
machinery not called; result matches explicit `sumofweights`).

### G1.7 Validation & expected gain

```bash
python scripts/compile_smoke.py
python -m pytest tests/ -q
python benchmarks/bench_gpu_parity.py --device 0 --out g1.json
python benchmarks/bench_gpu_parity.py --compare baseline.json g1.json
```

Expect large wins on `vectorized_density_density` / `vectorized_density_shear`
(the CPU equivalents gained 6.2×/5.7×; on GPU the eliminated work is the
25–50 gather+reduceat passes per map) and the first-call cost of
`vectorized_shear_shear` with fresh weights. Parity: bit-identical
numerators; denominators change summation order (reduceat → block reduce),
so expect ≤1e-12 scale-relative for float64.

---

## G2 — standalone block-reduced aperture kernel, one launch for all bins
*(review item 20, GPU half; also removes the per-bin launch loop of
`_compute_tomo_aperture_shear`/`_density`)*

### G2.1 New file `src/CosmoFuse/cuda/aperture_tomo.cu`

```cuda
/*
 * aperture_tomo.cu -- Block-reduced aperture statistics for all
 * tomographic bins in one launch.  Grid: blockIdx.x = patch,
 * blockIdx.y = tomo bin; threadIdx.x strides the aperture pixels.
 * Replaces the per-pixel ElementwiseKernel + add.reduceat path
 * (two npixels_in_apertures-sized temporaries per call).
 */

__COMMON_CUDA_SOURCE__

template<typename T>
__global__ void gpu_aperture_shear_tomo(
    const T* g1,                 /* base ptr, row b at g1 + b*g_stride  */
    const T* g2,
    const long long g_stride,    /* elements between tomo rows (2*npix for
                                    (nz,2,npix) views, npix for planar)  */
    const T* weights,            /* base ptr, row b at weights + b*w_stride */
    const long long w_stride,
    const unsigned int* q_inds,
    const T* q_cos,
    const T* q_sin,
    const T* q_val,
    const long long* q_offsets,
    const T* q_patch_area,
    T* out_num,                  /* [ntomo x npatches] */
    T* out_den,
    const int npatches,
    const int ntomo)
{
    const int lane = (int)threadIdx.x;
    const int patch = (int)blockIdx.x;
    const int bin = (int)blockIdx.y;
    if (patch >= npatches || bin >= ntomo) return;

    const T* g1b = g1 + (long long)bin * g_stride;
    const T* g2b = g2 + (long long)bin * g_stride;
    const T* wb  = weights + (long long)bin * w_stride;

    const long long start = q_offsets[patch];
    const long long stop  = q_offsets[patch + 1];
    T sum_num = (T)0.0;
    T sum_den = (T)0.0;
    for (long long idx = start + lane; idx < stop; idx += BLOCK_SIZE) {
        const unsigned int pix = q_inds[idx];
        const T wv = wb[pix];
        const T gt = -g1b[pix] * q_cos[idx] - g2b[pix] * q_sin[idx];
        sum_num += wv * gt * q_val[idx];
        sum_den += wv;
    }
    block_reduce_sum_pair(sum_num, sum_den, &sum_num, &sum_den);
    if (lane == 0) {
        const long long o = (long long)bin * npatches + patch;
        out_num[o] = q_patch_area[patch] * sum_num;
        out_den[o] = sum_den;
    }
}

template<typename T>
__global__ void gpu_aperture_density_tomo(
    const T* values,
    const long long v_stride,
    const T* weights,
    const long long w_stride,
    const unsigned int* q_inds,
    const T* q_val,
    const long long* q_offsets,
    const T* q_patch_area,
    T* out_num,
    T* out_den,
    const int npatches,
    const int ntomo)
{
    const int lane = (int)threadIdx.x;
    const int patch = (int)blockIdx.x;
    const int bin = (int)blockIdx.y;
    if (patch >= npatches || bin >= ntomo) return;

    const T* vb = values + (long long)bin * v_stride;
    const T* wb = weights + (long long)bin * w_stride;

    const long long start = q_offsets[patch];
    const long long stop  = q_offsets[patch + 1];
    T sum_num = (T)0.0;
    T sum_den = (T)0.0;
    for (long long idx = start + lane; idx < stop; idx += BLOCK_SIZE) {
        const unsigned int pix = q_inds[idx];
        const T wv = wb[pix];
        sum_num += wv * vb[pix] * q_val[idx];
        sum_den += wv;
    }
    block_reduce_sum_pair(sum_num, sum_den, &sum_num, &sum_den);
    if (lane == 0) {
        const long long o = (long long)bin * npatches + patch;
        out_num[o] = q_patch_area[patch] * sum_num;
        out_den[o] = sum_den;
    }
}
```

Stride contract: pass **views** like `shear_dev[:, 0]` (shape `(nz, npix)`,
row stride `2*npix`) directly — CuPy hands the view's base pointer to the
raw kernel, and the explicit `g_stride`/`w_stride` arguments do the rest.
Compute strides in Python as `arr.strides[0] // arr.itemsize` and require
`arr.strides[1] == arr.itemsize` (fall back to `ascontiguousarray`
otherwise).

### G2.2 Builder in `backend.py`

Clone the `_build_cupy_density_density_tomo_vectorized_kernel` pattern
(closure + `kernel_cache` + `_KERNEL_CACHE_MISS` negative caching):
source `_prepare_cuda_source("aperture_tomo.cu")`, name expressions
`f"gpu_aperture_shear_tomo<{map_c_type}>"` and
`f"gpu_aperture_density_tomo<{map_c_type}>"`; wrapper returns `False`
when no raw compiler / compile failure. Register both on `Backend`
(`aperture_tomo_shear_kernel`, `aperture_tomo_density_kernel`) and wire
them in `get_backend`'s GPU construction. Add
`"cuda/*.cu"` already covers packaging (`pyproject.toml`
`[tool.setuptools.package-data]`) — no change needed there.

### G2.3 Orchestrator

In `_compute_tomo_aperture_shear` (GPU branch), before the per-bin loop:

```python
kernel = getattr(self.backend, "aperture_tomo_shear_kernel", None)
if kernel is not None and self.backend.name == "cupy":
    self._ensure_aperture_pairs(aperture_filter=aperture_filter)
    if self.compute_context.Q_inds_dev is None:
        self._prepare_aperture_device_buffers()
    ctx = self.compute_context
    g1v = shear_dev[:, 0]      # views; strides passed explicitly
    g2v = shear_dev[:, 1]
    out_num = module.empty((nzbins, self.n_patches), dtype=map_backend_dtype)
    out_den = module.empty((nzbins, self.n_patches), dtype=map_backend_dtype)
    launched = kernel(g1v, g2v, w_dev, ctx.Q_inds_dev, ctx.Q_cos_dev,
                      ctx.Q_sin_dev, ctx.Q_val_dev, ctx.Q_offsets_dev,
                      ctx.Q_patch_area_dev, out_num, out_den)
    if launched:
        return self._normalize_by_weights(out_num, out_den)  # already area-scaled
# fallback: existing per-bin ElementwiseKernel loop
```

(the sign flip for `flip_g1`/`flip_g2` stays where it is today — apply it
before slicing, only when a flip is requested). Same shape for
`_compute_tomo_aperture_density`. `get_aperture_shear`/`get_aperture_density`
can route through the same kernel with `ntomo=1` by reshaping the inputs to
`(1, npix)`.

### G2.4 Validation & expected gain

Parity rows `Ma`, `Ma1`, `Mg1`, `f_Ma`, `f_Mg` in the harness. Expect
`get_aperture_*` and the `M_a` part of `get_full_tomo_shear` to drop the
two `len(Q_inds_flat)`-sized temporaries and `nzbins+1` kernel launches +
reduceat per map. Denominator summation order changes → ≤1e-12
scale-relative.

---

## G3 — launch the fused 3×2pt kernel per section instead of a dense max grid
*(review item 19)*

### G3.1 `src/CosmoFuse/cuda/tomo_fused_3x2pt.cu`

Replace the grid-z selector with a launch argument. Change:

```cuda
    const int z = (int)blockIdx.z;               /* correlation type selector  */
```

to

```cuda
    const int z = section;                       /* correlation type selector  */
```

and append `const int section` as the **last** kernel parameter (after
`out_xit_den`).

### G3.2 Wrapper `_cupy_3x2pt_tomo_fused_kernel` (backend.py ~1300)

Replace the single launch (`blocks = (max_x, max_y, 5)` … one
`raw_kernel(...)` call) with:

```python
        section_grids = (
            (npatches, n_shear_bins),        # z=0  M_ap
            (npatches, n_density_bins),      # z=1  M_g
            (nbins_total, 2 * n_ss_comb),    # z=2  xi+/-
            (nbins_total, 2 * n_dd_comb),    # z=3  xi_g
            (nbins_total, n_ds_comb),        # z=4  xi_t
        )
        threads = 256
        base_args = (density_map, shear_map, density_weights, shear_weights,
                     ind_i, ind_j, rot_i, rot_j, pair_offsets,
                     np.int64(nbins_total), np.int32(npatches), np.int32(npix),
                     q_inds, q_cos, q_sin, q_val, q_offsets, q_patch_area,
                     ss_comb_i, ss_comb_j, np.int32(n_ss_comb),
                     dd_comb_i, dd_comb_j, np.int32(n_dd_comb),
                     ds_comb_i, ds_comb_j, np.int32(n_ds_comb),
                     out_ma_num, out_ma_den, out_mg_num, out_mg_den,
                     out_xip_num, out_xim_num, out_xipm_den,
                     out_xig_num, out_xig_den, out_xit_num, out_xit_den)

        # sections write disjoint outputs -> run them concurrently
        current = module.cuda.get_current_stream()
        ready = current.record()                 # inputs staged on current stream
        for z, (gx, gy) in enumerate(section_grids):
            if gx <= 0 or gy <= 0:
                continue
            stream = _section_streams[z]         # cached in the builder closure:
                                                 # [module.cuda.Stream(non_blocking=True) for _ in range(5)]
            stream.wait_event(ready)
            with stream:
                raw_kernel((int(gx), int(gy), 1), (threads,),
                           base_args + (np.int32(z),))
            current.wait_event(stream.record())  # downstream default-stream work waits
        return True
```

Create `_section_streams` lazily inside the builder closure (per device).
This removes every no-op block (~99% of the old grid at realistic sizes)
**and** lets the five sections overlap, which the z-dimension launch never
could.

### G3.3 Tests & validation

Fake-CuPy fused tests (`test_cupy_3x2pt_tomo_fused_kernel_success_and_cache`,
`_complex128_branch`, compile-failure test) now see 5 launches with an extra
trailing `int` — update the fakes to accept/record multiple calls and give
the fake module a `cuda.Stream`/`get_current_stream` stub (a 10-line
namespace with `record`, `wait_event`, context manager). Parity rows
`f_*`: bit-identical (per-block math unchanged). Gain: launch-overhead
plus overlap; measure with `get_3x2pt_tomo` timing and
`nsys profile --stats=true`.

---

## G4 — tile combinations per block in the pair kernels
*(review item 16: stop re-streaming `ind_i/ind_j/rot_i/rot_j` once per
combination×orientation)*

Do this **after** G1 and only if `nsys`/`ncu` shows the ξ± / dd / ds kernels
DRAM-bound on the pair-geometry arrays (metric: `dram__bytes_read` ≫
`pair bytes × 1`). Template change for `tomo_vectorized_xipm.cu` (dd/ds are
analogous):

```cuda
template<typename T, typename C, int TOMO_BINS_T, typename I, int COMB_TILE>
__global__ void gpu_fused_tomo_reduce_xipm(...)
{
    const int lane = (int)threadIdx.x;
    const int tile = (int)blockIdx.y;            /* tile of COMB_TILE comb_oris */
    const long long bin_flat = (long long)blockIdx.x;
    if (bin_flat >= nbins_total) return;

    int  ai[COMB_TILE], bj[COMB_TILE];
    bool active[COMB_TILE];
    #pragma unroll
    for (int t = 0; t < COMB_TILE; ++t) {
        const int comb_ori = tile * COMB_TILE + t;
        active[t] = comb_ori < 2 * ncomb;
        if (active[t]) {
            const int comb_idx = comb_ori >> 1;
            const int i = comb_i[comb_idx];
            const int j = comb_j[comb_idx];
            const bool use_ba = (comb_ori & 1) == 1;
            if (use_ba && i == j) active[t] = false;
            ai[t] = use_ba ? j : i;
            bj[t] = use_ba ? i : j;
        }
    }

    T sum_p[COMB_TILE] = {}, sum_m[COMB_TILE] = {}, sum_w[COMB_TILE] = {};

    const long long start = bin_offsets[bin_flat];
    const long long stop  = bin_offsets[bin_flat + 1];
    for (long long tid = start + lane; tid < stop; tid += BLOCK_SIZE) {
        const long long idx_a = (long long)ind_i[tid];   /* loaded ONCE  */
        const long long idx_b = (long long)ind_j[tid];   /* per tile of  */
        const C exp_a = rot_i[tid];                      /* COMB_TILE    */
        const C exp_b = rot_j[tid];                      /* combinations */
        #pragma unroll
        for (int t = 0; t < COMB_TILE; ++t) {
            if (!active[t]) continue;
            const long long idx_a_bin = idx_a * (long long)TOMO_BINS_T + ai[t];
            const long long idx_b_bin = idx_b * (long long)TOMO_BINS_T + bj[t];
            /* ... existing complex_make / complex_mul math on
                   shear[idx_a_bin*2 ...], weights[...] ... */
            sum_w[t] += w_pair;
            sum_p[t] += w_pair * (b_R * a_R + b_I * a_I);
            sum_m[t] += w_pair * (b_R * a_R - b_I * a_I);
        }
    }

    #pragma unroll
    for (int t = 0; t < COMB_TILE; ++t) {
        block_reduce_sum_triple<T>(sum_p[t], sum_m[t], sum_w[t],
                                   &sum_p[t], &sum_m[t], &sum_w[t]);
        __syncthreads();                          /* reuse shared buffers */
        const int comb_ori = tile * COMB_TILE + t;
        if (lane == 0 && active[t]) {
            const long long op = ((long long)comb_ori) * nbins_total + bin_flat;
            const long long om = ((long long)(2 * ncomb + comb_ori)) * nbins_total + bin_flat;
            out_num[op] = sum_p[t];
            out_num[om] = sum_m[t];
            out_den[op] = sum_w[t];
        }
    }
}
```

Wrapper: grid `((nbins_total, (2*ncomb + COMB_TILE - 1)//COMB_TILE, 1))`;
name expression gains `, {COMB_TILE}` (add the tile size to the cache key).
Start with `COMB_TILE = 4`; sweep 2/4/8 with the harness — register
pressure (3×TILE accumulators × fp64) caps the useful tile. Inactive lanes
must still write zeros? **No** — inactive rows are exactly the auto-B→A
rows plus tail padding; auto rows are never read, and padding rows do not
exist in `out_num`. Guard `comb_ori < 2*ncomb` prevents out-of-range writes.

Parity: bit-identical (same per-pair math, same block reduction per
comb_ori). Expected gain: pair-geometry DRAM traffic ÷ effective tile
factor; at 5 tomo bins the geometry stream drops from 30× to 8×.

---

## G5 — interleave the weight with the map payload
*(review item 17; do after G4, profile-guided)*

Build a packed `(npix, TOMO, 4)` buffer `[g1, g2, w, 0]` on device once per
map (reuse a persistent buffer in `ComputeContext`, filled with three strided
`copyto`s — the pattern of `_get_or_create_fused_input_buffers` /
`_fill_fused_input_buffers` at `correlations.py:1002/1054`):

```python
packed[..., 0] = shear_aos[..., 0]
packed[..., 1] = shear_aos[..., 1]
packed[..., 2] = weights_aos[..., None][..., 0]   # broadcast per-bin weight
```

Kernel side, replace the two separate gathers:

```cuda
    const long long base_a = (idx_a * (long long)TOMO_BINS_T + ai) * 4LL;
    const T ga1 = payload[base_a];
    const T ga2 = payload[base_a + 1];
    const T wa  = payload[base_a + 2];
```

One random cache line per pixel-visit instead of two (for float64 the
32-byte quad still sits in one 128-byte line; for float32 you can
additionally `reinterpret_cast<const float4*>(payload)[idx_a_bin]` — the
CuPy pool guarantees ≥512-byte allocation alignment). Applies to the ξ±,
dd, ds and fused kernels identically. Parity: bit-identical. Keep it only
if `ncu` shows L2/DRAM sector reads drop — on small maps the gather lines
are cache-resident and this is a wash.

---

## G6 — pinned double-buffered uploads
*(review item 15; `Backend.create_stream`/`alloc_pinned` at
`backend.py:1526/1563` exist and are tested but unused)*

Add to `correlations.py` (or a new `pipeline.py`):

```python
class PinnedMapPipeline:
    """Overlap map k+1 H2D transfer with map k compute.

    usage:
        pipe = PinnedMapPipeline(corr, {"shear": (nz, 2, npix), "w": (nz, npix)})
        dev = pipe.stage({"shear": shear_np_0, "w": w_np_0})
        for k in range(nmaps):
            nxt = pipe.stage({"shear": ..., "w": ...}) if k+1 < nmaps else None
            res = corr.get_full_tomo_shear(dev["shear"], dev["w"])   # compute
            dev = pipe.wait(nxt)                                      # swap
    """

    def __init__(self, corr, shapes, dtype=None):
        self.backend = corr.backend
        dtype = np.dtype(dtype or corr.map_dtype)
        self.stream = self.backend.create_stream(non_blocking=True)
        # two pinned host slots + two device slots per named array
        self.host = [{k: self.backend.alloc_pinned(s, dtype) for k, s in shapes.items()}
                     for _ in range(2)]
        self.dev = [{k: self.backend.zeros(s, dtype=dtype) for k, s in shapes.items()}
                    for _ in range(2)]
        self.slot = 0
        self.event = None

    def stage(self, host_arrays):
        """Enqueue async H2D of host_arrays into the back slot; returns nothing."""
        back = 1 - self.slot
        for k, arr in host_arrays.items():
            np.copyto(self.host[back][k], arr)              # host->pinned (CPU)
            with self.stream:
                self.dev[back][k].set(self.host[back][k])   # pinned->device, async
        self.event = self.stream.record() if self.stream is not None else None
        return back

    def wait(self, staged):
        """Make the staged slot current; compute stream waits on the copy."""
        if staged is None:
            return self.dev[self.slot]
        if self.event is not None:
            self.backend.module.cuda.get_current_stream().wait_event(self.event)
        self.slot = staged
        return self.dev[self.slot]
```

No kernel changes. The measurement calls already accept device arrays
(item 8), so users adopt this without API changes; also use it inside any
future `*_batched` wrapper (G9). Validate with the harness plus a
two-map loop timed end-to-end: expect the H2D time (`npix × nz × 3 × 8`
bytes per map ≈ 40 ms at 12 GB/s for nside 1024 / 5 bins) to disappear
from the critical path. Numerically a no-op (`set()` copies bytes).

---

## G7 — stop upcasting the aperture geometry to float64
*(review item 27; touches the fused kernel template, which is why it was
deferred with the CUDA batch)*

1. `_prepare_aperture_device_buffers` (`correlations.py:847`): upload
   `Q_cos/Q_sin/Q_val/Q_patch_area` at `self.rotation_dtype` — delete the
   four `.astype(map_backend_dtype)` conversions (keep `Q_inds` uint32 and
   `Q_offsets` int64 as they are).
2. `tomo_fused_3x2pt.cu` and `aperture_tomo.cu` (G2): add `typename QT`
   to the template; declare `const QT* q_cos, const QT* q_sin,
   const QT* q_val, const QT* q_patch_area`; promote at use:
   `... - g2 * (T)q_sin[idx]`, `out_ma_num[o] = (T)q_patch_area[patch] * sum_num`.
3. ElementwiseKernel builders (`backend.py:157/173`): second type letter —
   `"raw I Q_inds, raw Q Q_cos, raw Q Q_sin, raw Q Q_val, raw T g1, ..."`
   with casts `(T)Q_val[i]` in the body (CuPy resolves multiple type
   placeholders per call).
4. Wrapper name expressions gain the Q c-type
   (`f"...<{map_c_type}, {q_c_type}, ...>"`); add it to the cache keys.
   Determine `q_c_type` from `q_cos.dtype`.
5. CPU is untouched (the Numba kernels already take the float32 arrays).

float32→float64 promotion is exact, so results are **bit-identical**;
persistent aperture device memory and its upload halve. Update the fake
kernel-source assertions in `test_backend.py` for the new template arity.

---

## G8 — mixed precision: float32 payload, float64 accumulators
*(review item 28; biggest bandwidth lever for float32-tolerant users)*

1. `common.cuh`: the reduce helpers are already `template<typename T>` —
   no change; they will be instantiated at the accumulator type.
2. Every pair kernel: `template<typename T, typename ACC, ...>`; declare
   accumulators `ACC sum_p = (ACC)0.0; ...`; accumulate
   `sum_p += (ACC)(w_pair * (...));` reduce with
   `block_reduce_sum_*<ACC>`; outputs become `ACC* out_num, ACC* out_den`.
3. Wrappers: `acc_c_type = "double"`; name expression
   `f"...<{map_c_type}, {acc_c_type}, ...>"`; allocate `out_num/out_den`
   with `np.float64` regardless of map dtype; extend cache keys.
4. `Correlation.__init__`: new kwarg
   `accumulation_precision: str = "same"` (`"same"`|`"float64"`), stored as
   `self.acc_dtype`; the GPU orchestrator branches allocate output buffers
   at `acc_dtype` and the final normalized results cast back to
   `map_dtype`. CPU note: the Numba kernels already accumulate in float64
   when fed float64 accumulator arrays — to mirror exactly, allocate the
   CPU `out_*` arrays at `acc_dtype` too (accumulator variables inherit
   the output dtype via `zero = wa[0] * 0.0` / `np.zeros(..., out.dtype)`).
5. Docs/README: recommend `map_precision="float32",
   accumulation_precision="float64"` for GPU runs — half the gather and
   stream traffic at f64 reduction accuracy.

Validation: run the harness twice — float64 config (tolerance 1e-8,
unchanged) and float32+f64-acc config against the float64 CPU reference
with `--rtol 2e-4`. Expect ~1.6–2× on all bandwidth-bound pair kernels.

---

## G9 / G10 — architectural follow-ups (separate projects)

* **Multi-map batching** (item 9): add `nmaps` as a leading dimension on the
  map/weight arguments of new `vectorized_*_batched` entry points; kernels
  gain an `int NMAPS` template parameter and per-map accumulator arrays
  (structure identical to G4's `COMB_TILE` — batch 4–8 maps). The pair
  geometry is then streamed once per batch instead of once per map, and with
  fixed weights the G1 denominators are shared by the whole batch. Combine
  with G6 to stage batch k+1 while batch k computes.
* **Index compaction** (item 10): in `prepare()`, remap
  `ind_i/ind_j/Q_inds` through `np.searchsorted(self.map_inds, ...)` once
  (`map_inds` is sorted), keep `self._compact_indices = True`, and stage
  maps as `map[..., self.map_inds]` in `_to_backend_array`/the SoA fill.
  Upload volume and gather footprint shrink by the sky fraction. Validate
  bit-identical against the uncompacted path before enabling by default.

---

## Recommended order and stop/go criteria

| Step | Prereq | Keep if |
|------|--------|---------|
| Stage 0 harness | — | parity green on unmodified code |
| G1 denominators | — | dd/ds/ξ± per-map time drops; parity ≤1e-12 |
| G2 aperture kernel | — | `get_aperture_*`, `Ma` rows drop; parity ≤1e-12 |
| G3 per-section launches | — | `get_3x2pt_tomo` drops; `nsys` shows overlap |
| G6 pinned pipeline | item 8 (merged) | end-to-end multi-map loop drops by ~H2D time |
| G4 comb tiling | G1 | `ncu` DRAM reads drop **and** wall time drops |
| G5 payload packing | G4 | same criterion; else revert |
| G7 f32 aperture | G2 | bit-identical, memory halves |
| G8 mixed precision | G1–G5 | f32 path ≥1.5× at ≤2e-4 vs f64 CPU |
| G9/G10 | all above | own design review |

Global pitfalls, learned the hard way on the CPU pass:

* **Never compare timings across machine-state changes** — always interleave
  baseline/candidate runs (`--out` + `--compare` exist for this).
* `name_expressions` strings must match `get_function` byte-for-byte —
  build both from one f-string.
* The `RawKernel` fallback in `_compile_raw_cuda_kernel`
  (`backend.py:59`) cannot compile C++ template expressions; on CuPy
  versions without `RawModule` the wrappers return `False` and the tomo
  orchestrators raise — unchanged behaviour, but remember it when a step
  "mysteriously" refuses to launch.
* Failed compilations are negatively cached (`_KERNEL_CACHE_MISS`); after
  fixing a kernel during development, restart the process (or clear the
  builder's `kernel_cache`) before retesting.
* Keep `out_num`-style buffers zero-initialised wherever a kernel skips
  auto-combination B→A blocks — those rows are never written.
* Every fake-CuPy test in `tests/test_backend.py` that records launch args
  encodes the wrapper↔kernel contract; update them in the same commit as
  the signature change, and mirror every GPU semantic change in the CPU
  kernels (or explicitly document the divergence) so the parity harness
  stays meaningful.
