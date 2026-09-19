# Changelog

## 6.4.0 (2026-09-19)

Weak lensing and galaxy clustering may now carry **different tomographies**.
A full 6x2pt i3PCF -- 4 source bins against 5 lens bins -- did not run before
this release: three of the eight ζ estimators raised.

### Mixed-sample ζ estimators

`zeta_g_plus`, `zeta_g_minus` and `zeta_a_g` cross one sample's aperture with
the other sample's 2PCF, but were built on `_zeta_from_fields`, which asserted
that the annulus held the upper triangle of the *centre's* bins and laid the
output out as `combinations_with_replacement(range(nzbins), 3)`.  With
`n_source != n_lens` that assertion is simply false, and
`calculate_all_zetas(M_a=..., M_g=..., xi_p=..., xi_g=...)` died on it:

    ValueError: annulus_field has incompatible number of tomographic pairs;
    expected 15 for 5 bins, got 10

- **The triangle rule is now applied only where it means something.**
  `_zeta_from_fields` takes `symmetric`: `True` keeps the upper triangle and
  the count check, `False` gives one row per `(z_center, annulus_combination)`
  centre-major, `None` infers from the counts.  `_zeta_from_cross_fields`
  becomes an alias of it -- the two only ever differed in this.
- `zeta_a_plus`, `zeta_a_minus` and `zeta_g_g` are built from a single sample,
  so they pass `symmetric=True` and keep the strict check they always had.
  The three mixed estimators default to inference, and gained an explicit
  `symmetric` argument.
- **`_batched_zetas` no longer requires the centrals to agree on `nzbins`.**
  It read one `nzbins` off the first central and bailed out when another
  disagreed, so M_ap over 4 source bins beside M_g over 5 lens bins fell back
  to the per-estimator path -- which then raised.  Offsets and layouts are
  taken per field; the 4+5 case batches and is bitwise equal to eight separate
  calls.
- `calculate_all_zetas` gained `symmetric`, a per-estimator override
  (`{"zeta_a_g": False}`).  `xi_t_symmetric` is retained as the older spelling
  of the same thing for γ_t, and `symmetric` wins where both name an
  estimator.

### The one ambiguous case, unchanged

With `n_source == n_lens` a cross annulus holds exactly `nz(nz+1)/2`
combinations, so inference cannot distinguish it from the triangle.  That case
keeps the historical reading -- every pre-6.4.0 output is bit-for-bit
unchanged -- and `symmetric=False` is how you ask for the other one.  This is
the trap `_zeta_from_cross_fields` has always documented for γ_t; it now
applies to five estimators and is spelled the same way for all of them.

### Row order

- **`Correlation.zeta_cross_triplets(n_central_bins, n_annulus_combinations)`**
  gives the row order of every cross-sample ζ: `(z_center, annulus_row)`,
  centre-major, where `annulus_row` indexes `tomo_combinations` (ξ±, ξ_g) or
  `ggl_combinations` (ξ_t).  ζ_g_t and ζ_a_t had used this layout since 5.0
  with no public accessor for it.
- `zeta_triplets` is unchanged and now says which estimators it describes.

### What this costs

One assertion.  `zeta_g_plus(M_g, xi_p)` with a non-triangular annulus used to
raise and now returns a cross layout, so a genuinely mis-shaped ξ± is no
longer caught by the pair count -- it cannot be, once the count is legal.
`symmetric=True` reinstates the check for callers who want it, and the
same-sample estimators never lost it.

### Tests

`tests/test_correlation_helpers.py` grew the 4-source/5-lens table: all eight
estimators against the literal definition, 415 triplets per angular bin, the
batched path equal to the single calls, the equal-count case pinned to the old
layout, the override, and `zeta_cross_triplets` gated against `_cross_indices`.

## 6.3.0 (2026-09-19)

An audit pass: everything that could be done without losing performance,
growing memory, or moving a number that was already right.
Every public GPU call is **bitwise identical** to 6.2.0 (measured, A/B) except
`compute_density_density`, which was measuring the wrong estimator.

### Performance

- **The zeta reduction is vectorised over the triplet list.**
  `_zeta_from_fields` and `_zeta_from_cross_fields` walked the tomographic
  triplets in Python, about five array operations per triplet on ~1 MB of
  data.  At the production geometry that made the reduction the largest item
  in a map-set -- roughly nine times the measurement it follows -- and all of
  it host-side launch overhead.
  - All eight estimators now go through one `_zeta_covariance()`.  The triplet
    list becomes a pair of index arrays (`_ZetaIndices`), so a reduction is
    one gather, one product and one mean regardless of the triplet count:
    ~800 array operations become ~80.
  - The index arrays are memoised per binning and **uploaded once per
    device**.  A fresh host index array costs ~0.28 ms per estimator to
    upload -- more than the arithmetic it indexes, and 40 % of what remained
    once the loop was gone.  `getDevice()` keys the cache.
  - **And then all eight share one gather.** Each estimator was a single
    gather, but eight of them is still eight times the fixed array-op cost --
    and at these sizes that cost *is* the runtime (~0.03 ms per cupy call on
    ~1 MB).  Concatenating the centres once and the annuli once makes it one
    set of operations for the whole reduction.  The batch is skipped, and the
    per-estimator path taken, when the inputs do not share a dtype
    (concatenation would silently raise an output's precision) or when the
    annuli disagree on the patch/angular axes.
  - A100, 450 patches, float64, all eight estimators from device-resident
    input: **31.5 -> 0.80 ms (39x)** at 4 tomographic and 10 angular bins.
    The reduction was nine times the measurement; it is now a fifth of it.
    Bit-for-bit identical on numpy; <= 0.3 ULP on cupy.

- **One fused kernel for the safe divide** (`backend.safe_divide`).  `den != 0`
  + `where` + `divide` + `astype` + `*=` is five kernels and two full-size
  temporaries per output array, and `get_3x2pt_tomo` produces six.  Written
  out element by element it is one kernel and no temporary: **0.745 -> 0.068 ms**
  for the six at production shapes.  `_normalize_by_weights` shares it, which
  is three more per `get_full_tomo_shear`.
  - Deliberately the *same expression* rather than a tidier one: the `den == 0`
    branch still multiplies the quotient by zero instead of assigning it, so a
    negative numerator still yields `-0.0` and a non-finite one still yields
    `NaN`.  Verified bit-for-bit over 32 dtype/shape combinations.

- **`get_full_tomo_ggl` shares one row expansion** between its passes, as the
  other `get_full_tomo_*` methods already did -- four expansions per call
  became two when an aperture output is requested.  The scope is taken only
  then, because it is the `aperture_tomo.cu` leaf that fixes the layout at
  `"soa"`; a xi_t-only call keeps its direct AoS write.

- Measured end to end on the A100 (12 patches, 11 M pairs, all values bitwise
  unchanged):

  | call | 6.2.0 | 6.3.0 |
  |---|---|---|
  | `get_3x2pt_tomo` | 4.713 ms | 4.607 ms |
  | `get_full_tomo_shear` | 3.041 ms | 3.000 ms |
  | `get_full_tomo_ggl` (+N_ap, +M_ap) | 3.929 ms | 3.892 ms |
  | `get_3x2pt_tomo`, k = 2 | 1.762 ms | **1.104 ms** |
  | `get_full_tomo_shear`, k = 2 | 1.097 ms | **0.778 ms** |
  | `get_full_tomo_ggl`, k = 2 | 1.455 ms | **1.256 ms** |

  The treecode configurations gain most: the kernels are small there, so the
  host side is the wall clock.

- **The frozen-map memo is bounded by bytes**, not by 16 entries.  Sixteen is
  a harmless count and a ruinous size: one frozen `(4, 2, npix)` float32
  shear map-set is 1.6 GB at nside 2048, so the entry bound alone permitted
  ~25 GB of VRAM for a cache whose stated purpose is a handful of fixed
  weight maps.
- **`release_device_memory()`** (new): drops the pair scratch (32 B/pair), the
  frozen-map memo and the weight-sum caches without touching the prepared
  geometry.  A long-lived `Correlation` can now share a GPU.
- **HDF5 chunks are capped by `flush_every`** (`ZetaWriter`).  A ~1 MiB chunk
  against a 50-map flush window means every flush rewrites a partially filled
  chunk.
- The pair search no longer fills a **write-only `bin_indices` array**: 8 B
  per pair written once and never read (only its `.size`, which is
  `inds_a.size`).

### Fixed -- silent wrong answers

- **`compute_density_density` used a mean of ratios where everything else uses
  a ratio of sums.**  `0.5*(N_ab/D_ab + N_ba/D_ba)` instead of
  `(N_ab+N_ba)/(D_ab+D_ba)`; the two agree only when both orientations carry
  the same weight, i.e. never for a cross pair.  Measured discrepancy against
  `vectorized_density_density` and `get_3x2pt_tomo` on a cross pair:
  **3.7 % relative**.  The auto case is unchanged.  **This changes numbers**
  -- it is the one output in this release that moves.  `tests/test_estimator_parity.py`
  is the gate that was missing: single-map vs vectorised vs fused, per
  statistic, on both backends.
- **`get_3x2pt_tomo(return_device=True)` returned cached buffers** that the
  next call overwrote in place, so `[corr.get_3x2pt_tomo(...) for _ in ...]`
  produced N references to the last realisation with no error.  Each call now
  allocates its outputs; the pool recycles them, so a consume-and-discard loop
  holds no more device memory than the cache did (measured: +0.02 MB per live
  result at production shape).
- **Three caches were keyed on a recycled address.**  `id()` and a CUDA pool
  pointer identify an object only while it is alive: CPython reuses addresses
  and cupy's pool hands a freed pointer straight to the next allocation.  None
  of the three held a reference to what it had keyed.
  - a transient `aperture_filter` -- a lambda or `functools.partial` written
    inline, which is the documented way to pass it -- was silently ignored,
    and the measurement made with the *previous* filter's Q values;
  - `for k: w = cp.asarray(w_host[k]); corr.compute_shear_shear(..., w, w)`
    gave the second map the first one's sum of weights;
  - a frozen host weight map rebuilt per realisation inherited the previous
    one's digest.
  All three now go through `utils.live_object_serial`, which pins nothing and
  cannot be forged by a recycled address.  `tests/test_identity_caches.py`
  recycles the addresses deliberately.
  - **Two more of the same class, not in the audit**, found by sweeping for
    `id()`-keyed cache keys: `_combination_layout` (a module-level cache keyed
    on `(id(comb_i), id(comb_j))`, while an explicit `ggl_bin_combinations`
    selection builds fresh arrays every call and drops them -- a stale hit
    would scatter the results into the *previous* selection's rows), and the
    aperture-cell cache keyed on `id(self.map_inds)`, which `load_pairs()`
    reassigns.  Neither could be made to fire here -- the layout key needs
    *both* ids to recycle at once and CPython's LIFO free lists tend to swap
    rather than match them (0 of 4000 alternating calls) -- but both are
    allocator luck rather than a guarantee, and the fix is a dict lookup of
    the same cost.
- **`MultiDeviceCorrelation.__getattr__` forwarded anything it did not know to
  `parts[0]`**, so `compute_shear_shear`, `phi_center`, `n_patches` and the
  rest silently returned *one device's share* of the patches -- 459 of 917,
  with no error.  It now forwards an explicit allow-list of attributes that
  are the same on every part and raises `NotImplementedError` for the rest.
- **Loading a packed pair file no longer flips `pack_host_pairs`.**  The flag
  is the user's request for how future pair finding should behave; setting it
  from a file made a later `preprocess()` silently measure with uint16
  rotations, which is a different estimator.
- **The CPU accumulators are explicitly float64.**  `zero = x[0] * 0.0` types
  as float64 under numba but as float32 under `NUMBA_DISABLE_JIT=1`, which
  pytest-env forces -- so the suite was measuring a float32 accumulation of
  four kernels that ship accumulating in float64.  The shipped numbers do not
  move; the tests now exercise them.

### Fixed -- failures

- **A packed file written with `aperture_nside` can be read back.**  The
  writer chose its dataset names from one rule and the reader from another,
  which disagreed for exactly that combination (`tc_Q_inds` present,
  `tc_pair_inds` absent), so the load raised `KeyError: 'Q_inds'`.  The flag
  is now written as a file attribute; the inference is kept as a fallback for
  older files and fixed.  This is `pack_host_pairs` + `aperture_nside`, i.e.
  the nside 2048 archive.
- **`preflight_pair_memory` projects the resource it is budgeting against.**
  It charged the 24 B/pair *host* layout against free *device* memory, so runs
  that fit were rejected with `MemoryError` -- exactly the large-geometry case
  `pack_pairs` exists for.  It now uses 8 B/pair when the relevant packing is
  on, and the report and the error name which resource.
- **`save_pairs()` after `release_host_pairs=True` raises** instead of warning
  and returning.  The job used to exit 0 with no file, after hours of pair
  finding, and the warning is invisible under `-W ignore`.
- **A failed `prepare()` refuses to measure.**  It published the device
  buffers before setting `ntotpairs`, so an OOM part-way through left an
  object that `_ensure_prepared()` could not tell from a good one and that
  returned finite numbers from a torn state.
- **`device='auto'` falls back to CPU** when cupy imports but no device is
  usable (driver/runtime mismatch, `CUDA_VISIBLE_DEVICES=""`, an unusable
  card).  It only caught `ImportError`.  `device='gpu'` and an explicit id
  stay strict.
- **Every aperture entry point works on a fresh instance.**  `Q_inds` and its
  siblings are initialised in `__init__`, so `ensure_aperture_pairs` builds
  the geometry on demand instead of raising `AttributeError`; and a
  measurement with no pair geometry at all now states the precondition.
- **`MultiDeviceCorrelation.get_full_tomo_ggl(return_N_ap=True)`** no longer
  raises `IndexError`: `_PATCH_AXIS` declares all three outputs, and a
  mismatch between the declared axes and the returned arity is now an error
  that says so.
- **A transient kernel-compilation failure is no longer cached.**  A failed
  NVRTC compile was cached negatively forever, so one transient hiccup (a
  disk-cache read error) silently pinned the process to the slower fallback
  kernel for its whole lifetime.  Deterministic compile errors are still
  cached; transient ones are retried.  *(Found while running this release's
  tests, not in the audit.)*

### API

- **`Correlation.tomo_combinations()`, `.ggl_combinations()`,
  `.zeta_triplets()`** (new): the row order of every tomographic output, which
  had to be guessed.  Gated against the private builders so they cannot drift.
- **`Correlation.compute_sumofweights()`** (new): the public `sumofweights=`
  argument had no supported way to produce a value.
- **`zeta_g_t` / `zeta_a_t` / `calculate_all_zetas` take `symmetric=`**.  The
  γ_t layout was inferred from the combination *count*, so a GGL subset that
  happens to hold `nz(nz+1)/2` entries was silently treated as the symmetric
  shear ordering -- 10 rows of wrong pairings instead of 40.  The inferred
  default is unchanged.
- **The directional `sumofweights` form is decided from the whole shape.**
  `shape[0] == 2` alone made the per-combination form unusable whenever there
  happened to be exactly two tomographic combinations.
- **`pack_pairs=True` past the accumulator limit** now says where the limit
  is and why there is no fallback, instead of a bare decline.
- The device-buffer property setters **invalidate the caches derived from
  them** (`inds_i_dev`/`inds_j_dev`, the aperture device buffers).
- `radius_filter` is now the single source of the `5 * theta_Q` truncation
  radius rather than an unread attribute beside three literals.

### Numerics

- The zeta reduction is **bit-for-bit identical on numpy** to the per-triplet
  loop.  On cupy it moves by <= 1 ULP (a larger output changes the reduction
  blocking) and gains a property the loop lacked: reducing one map-set at a
  time now gives bitwise the same numbers as a stacked batch, so `ZetaWriter`'s
  device route and a batch reduction agree exactly.
- Integer `central`/`annulus` input to the zeta helpers is no longer truncated
  to integer output.

### Tests

- `tests/test_estimator_parity.py` (new): one estimator per statistic across
  the single-map, vectorised and fused paths, on both backends, plus the
  combination-ordering accessors and the precomputed sum of weights.
- `tests/test_identity_caches.py` (new): the three recycled-address caches,
  with the addresses recycled deliberately.
- **83 lines of test are collected again.**
  `test_xipm_gpu_fallback_path_with_fake_cupy` was nested inside the body of
  the test above it, so pytest never saw it -- and it did not pass once it
  ran (it referenced attributes the test class does not have, and its fake
  cupy was missing half the module surface).  Both fixed.
- **The degrade fixture is built once per configuration**, not once per
  subclass: ~54 s off the suite.
- Two tests that asserted only `len(outputs) == 8` -- which no input can
  falsify -- now assert what the clamp branches actually do.
- **Coverage is no longer forced on every invocation** (`make coverage` and CI
  ask for it), and `make lint` / `make format` have recipes instead of being
  `.PHONY` entries that exit 0 having checked nothing.

### Docs

- `docs/estimator_note.md` said `compute_shear_shear` keeps the historical
  average-of-ratios form.  It does not, and since this release neither does
  `compute_density_density`: every entry point is the ratio of sums.  This was
  the most misleading line in the repo -- a scientific claim, not a cosmetic
  one.
- `README.md`: the ξ± formula now shows the rotation into the pair frame
  (without it ξ− is not rotation-invariant, so the formula was wrong even
  though the code is right); format version 1 files are *not* readable, as
  `MIN_FORMAT_VERSION = 2` has always enforced; `U_crittenden`/`U_schneider`
  are named; and `return_device` is documented, including that it defaults to
  `True`, that a multi-device group ignores it, and that the returned arrays
  are the caller's.
- `warmup()` compiles the pair-search kernel at `pair_search_precision`, which
  is what `preprocess()` uses -- it compiled it at `rotation_precision`, so
  the compile it paid for was never the one that ran.  The docstring now says
  the pair search is Numba on both backends.
- CI: `codecov-action@v4` gets a token and no longer fails the run on an
  upload error unrelated to the code.

### Removed

- `block_reduce_sum` (`cuda/common.cuh`), `_resolve_aperture_filter`, a dead
  `npix` local, and `cross_definition_check()` from
  `benchmarks/static_treecode/benchmark_treecorr.py` -- the last recombined
  CosmoFuse's numerators "TreeCorr-style" to compare against the pre-5.0 cross
  estimator, so since 5.0 it compared the estimator against itself.
- `_get_or_create_fused_post_buffers`, the cache behind the aliasing above.

## 6.2.0 (2026-09-18)

### Performance
- **One block per patch in the aperture kernels, not one per (patch, tomo
  bin).** The aperture pass was the largest non-pair item in
  `get_full_tomo_shear` and the only one the treecode does not shrink, because
  each block re-read the disc geometry -- `q_inds` + `q_cos` + `q_sin` +
  `q_val`, 16 B per disc pixel -- once per tomographic bin while using 12 B of
  map data: 28*nz B/pixel where one block per patch needs 16 + 12*nz (112 vs
  64 at nz = 4). 6.1.0 measured that L2 does not catch those re-reads; this
  ships the restructure. `gpu_aperture_shear_tomo_fused`,
  `gpu_aperture_density_tomo_fused` and `gpu_3x2pt_tomo_aperture_fused` loop
  the bins inside one block, with NZ a template parameter so the per-thread
  accumulators stay in registers, and `block_reduce_sum_pair_into` reduces on
  caller-supplied shared buffers so the bin loop pays for one buffer pair
  rather than one per call site.
  - Measured on the A100 at nside 512, k = 2.9, 450 patches, 4 bins, float32
    maps: the aperture kernel **0.451 -> 0.259 ms** in the probe and
    **431 -> 236 us (1.83x)** inside the real `get_full_tomo_shear` (nsys).
    `get_3x2pt_tomo` -- the 3x2pt production path, where both sections run --
    **4.455 -> 3.962 ms** of wall time.
  - `get_full_tomo_shear` itself gains the same 0.2 ms of *GPU* time
    (1.567 -> 1.367 ms per call) but only 0.02 ms of wall time: at this size
    the isolated call is host-bound, ~1.78 ms of wall against ~1.37 ms of
    kernel time. The saving is real; it shows up as device idle rather than
    wall clock unless the call is overlapped or the kernels are larger
    (nside 2048), which is where the 2.08x of the probe applies.
  - **Bitwise identical**, which is required: `resolution_factor=None` must
    stay bit-for-bit identical to 4.20.0 and this kernel is on that path. Only
    the bin loop moves inside the block -- the thread->pixel mapping,
    `BLOCK_SIZE` and the reduction tree are untouched, so every bin sums the
    same partials in the same order. `tests/test_aperture_fused.py` (new) is
    the gate: `np.array_equal`, never a tolerance, through the real wrappers
    against the numpy twins and, marked `gpu`, against the real kernels
    compiled with the library's own `--use_fast_math` options.
  - The per-(patch, bin) kernels stay as the fallback and are launched
    unchanged beyond `_MAX_FUSED_APERTURE_BINS` = 16 tomographic bins (the
    accumulators would spill) or below 2 blocks/SM of patches (the fused grid
    is `ntomo` times smaller and would underfill the device).
  - SoA stays: AoS wins only at nside-2048-sized discs, so `_expansion_scope`
    and `_pair_row_layout` are unchanged.

## 6.1.0 (2026-09-18)

### Measured (no library change)
- **The aperture pass: both of its levers, quantified.** `gpu_aperture_shear_tomo`
  is the largest non-pair item in `get_full_tomo_shear` (0.450 ms of 1.796 ms at
  nside 512) and the only one the treecode does not shrink. Two benchmarks
  settle what can be done about it; results in
  `benchmarks/static_treecode/APERTURE_RESULTS.md`.
  - `benchmarks/static_treecode/aperture_reuse_probe.py` (new) answers whether
    the kernel really pays for re-reading the disc geometry once per tomographic
    bin. `ncu` cannot say -- performance counters need root on the A100 node
    (`ERR_NVGPUCTRPERM`) -- so it compiles probe kernels and measures the end
    state instead. It does pay: one block per patch instead of one per
    (patch, bin) is **1.75x** at nside 512 (0.451 -> 0.258 ms) and **2.08x** at
    nside 2048, exactly the 112 -> 64 B/pixel traffic ratio, and **bitwise
    identical** to the shipped kernel in every configuration. AoS wins only at
    nside-2048-sized discs, so the scope-chosen layout stays as it is. The
    kernel itself is NOT changed here.
  - `t10_zeta_level.py` gained `--aperture-nside`, so the zeta price of a coarse
    aperture can be measured the way T10 measured the price of a coarse pair
    level. With `resolution_factor=None` only M_ap moves: base 512 -> aperture
    128 costs **0.07 sigma_patch** (aperture 256: 0.04), against the 0.27
    sigma_patch at which treecode k = 2.9 was accepted. That run is 4x more
    aggressive in pixel/theta_Q than pinning the aperture to 512 at base nside
    2048, which is therefore safe by a wide margin -- and worth **20x** on that
    pass (9.10 -> 0.455 ms). Unlike `resolution_factor` the effect is flat in
    theta, so the two knobs are not interchangeable.

### Performance
- **The row expansion writes the kernels' layout, and carries the sign
  flip** (idea #3, the remaining two sub-items). Between the degrade and
  the pair kernels sat three more passes over the map-set: a stack that
  glued the two shear components back together, a SoA -> AoS transpose
  (0.077 ms; the packed `perm` gather adds another 0.105 ms), and --
  whenever `flip_g1`/`flip_g2` was set -- a scale-and-stack copy that
  `get_full_tomo_shear` made *twice*, once per leaf (0.257 ms, **15 %** of
  the call; the production script runs `flip_g1=True`). The fused degrade
  now takes four element strides per buffer instead of one, so it writes
  the interleaved `(nz, 2, n_rows)` or AoS `(n_rows, nz, 2)` layout each
  call wants directly, and applies the sign while it copies the pixel rows
  -- the cell rows inherit it, because level 0 reads the already-signed
  rows. `get_3x2pt_tomo` no longer stages its four inputs through separate
  cached buffers at all (`ComputeContext`'s `fused_*_soa` are gone).
  Measured on the A100 (nside 512, 450 patches, 18.3 M pairs, k = 2.9,
  4 tomographic bins, float32 maps + float64 accumulators):

  | | before | after |
  |---|---|---|
  | `_expand_shear_rows` | 0.361 ms | **0.274 ms** |
  | `vectorized_shear_shear` | 1.439 ms | **1.320 ms** |
  | `vectorized_shear_shear`, `flip_g2` | 1.560 ms | **1.338 ms** |
  | `get_full_tomo_shear`, `flip_g2` | 1.973 ms | **1.796 ms** |
  | `get_3x2pt_tomo` | 4.592 ms | **4.439 ms** |
  | cost of a flip, `get_full_tomo_shear` | 0.257 ms | **0.090 ms** |
  | cost of a flip, `vectorized_shear_shear` | 0.121 ms | **0.018 ms** |

  `get_full_tomo_shear` *without* a flip is unchanged (1.716 -> 1.707 ms,
  within the run-to-run scatter): its aperture pass keeps the SoA layout,
  so its 2PCF leaf still transposes. That is deliberate and measured --
  `aperture_tomo.cu` gathers aperture discs, whose row ids are largely
  contiguous, and AoS costs it **0.450 -> 0.600 ms**, far more than the
  0.077 ms transpose it would save. The layout is therefore chosen by the
  enclosing call (`_expansion_scope(layout=...)`) so that all of its
  leaves agree and one degrade still serves the whole call; the aperture
  kernels take a second element stride so either layout *can* be passed.
  A public AoS *input* layout was rejected: it only helps at full
  resolution, and it would add a second accepted layout to
  `_coerce_map_input_array`, `MapLoader`, `ZetaWriter` and every
  row-space gate. Both changes are pure data movement, and the tests
  require **bitwise** equality with the code they replace: a sign is +-1
  so scaling is exact, the degrade is linear in the values, and a
  transpose moves floats without touching them
  (`tests/test_row_layouts.py`).
- **Fused static-treecode row degrade** (idea #3). Building the virtual
  rows was a chain of sparse matrix products over an
  `(n_active, K * n_lead)` temporary plus two transposes. Two new CUDA
  kernels (`cuda/degrade_rows.cu`) walk each level's CSR children directly
  into an `(n_lead, n_appended)` accumulation-dtype scratch and then
  normalise and scatter into the row buffers: no pair-sized temporary, no
  transpose, no atomics. The number of lanes cooperating on one cell is
  sized from the mean number of children - a treecode level halves nside,
  so a cell has exactly 4 children and a full warp per cell idled 87 % of
  its lanes (2.1x instead of 3.9x). On an A100 (nside 512, 450 patches,
  18.3 M pairs, `resolution_factor=2.9`, 4 tomographic bins, float32 maps
  + float64 accumulators): `_expand_rows` **1.071 -> 0.273 ms (3.9x)**,
  `get_full_tomo_shear` 2.533 -> 1.721 ms (-32 %), `get_3x2pt_tomo`
  6.069 -> 4.584 ms (-24 %). It is the same recursion as the sparse chain
  and differs only in the order of summation inside a cell: the weight
  rows come out bitwise identical at float32 and the value rows agree to
  one ULP (5.96e-8 at float32, **2.0e-16 at float64**). The sparse chain
  stays as the CPU path and the fallback, and `resolution_factor=None`
  reaches neither. Validated against TreeCorr run on the degraded cells
  themselves (`tests/test_degrade_treecorr.py`).

### Added
- **`ZetaWriter`**: streams each map-set's i3PCFs to HDF5 from a background
  thread instead of holding every per-patch array in RAM. zeta averages over
  patches, not over maps, so map-set *k* can be reduced as soon as it is
  measured. At the DES Y3 production geometry that is 9 kB per map-set
  instead of 1.06 MB (90 MB instead of 10.6 GB for 10,000 realisations);
  `reduce="none"` keeps the per-patch arrays, from which zeta, a jackknife
  or a different binning can still be derived. On a GPU the reduction runs
  on the device before the copy, so only the data vector crosses PCIe; with
  a `MultiDeviceCorrelation` (host arrays) it runs on the CPU in the writer
  thread. The file carries `n_flushed` plus `level_table` / `row_pix_hash`
  provenance, and `resume=True` continues after a crash. `swmr=True` is
  available but off by default — it trades a worse crash story (stale write
  lock, needs `h5clear -s`) for a live view of the file.
- The zeta reduction in `correlation_helpers` now dispatches on the array
  module, so it runs on cupy arrays without a host round-trip.

## 6.0.0 (2026-09-18)

Compatibility cleanup: everything that existed only to keep pre-5.0 code,
pickles and files working is gone. No measurement path was rewritten —
except the one estimator inconsistency listed first.

### Removed / breaking

- **`compute_shear_shear` now uses the same cross-bin estimator as
  everything else.** For a cross pair it took the mean of the two
  orientation *ratios*, `(N_ab/W_ab + N_ba/W_ba) / 2`, while every
  tomographic method moved to the ratio of the summed orientations,
  `(N_ab + N_ba) / (W_ab + W_ba)`, in 5.0. It now does the same, so the
  package measures one ξ± estimator. Auto combinations, and cross
  combinations with an explicit `sumofweights`, are unchanged; with
  different per-bin weights the estimate moves (~3 % of max|ξ+| per patch
  on DES Y3, 0.02 σ). Gated by
  `tests/test_correlations_core.py::TestCrossOrientationEstimator`.
- **ξ± no longer follows the rotation precision.** The reduced numerators
  were cast back to float32 whenever `rotation_precision="float32"`; the
  returned estimates now carry the accumulation dtype, like every other
  statistic. Values are unchanged where the cast was lossless.
- Deprecated names removed: `PinnedMapPipeline` (use `MapLoader`),
  `RowSpaceMapLoader` (`MapFileLoader`), `Correlation.precompute()`
  (`preprocess()`), `Q_T` (`Q_crittenden`).
- `select_patch_centers` / `Correlation.from_mask`: the `filter_weighted`
  argument is gone (use `filter_weighting`), as are the values `"raw"`
  (use `"signed"`) and `"pixels"`. A constant `aperture_filter` reproduces
  the plain pixel fraction exactly if you want it.
- `pair_search_precision` accepts only `"float32"` and `"float64"`;
  `"rotation"` and `"auto"` are gone. `"float64"` remains the default.
- Pair files: **format version 1** (one HDF5 group per patch) is no longer
  read; `load_pairs` raises and says to convert with 5.x. Versions 2-4 are
  unaffected.
- `Correlation.__setstate__` no longer migrates pickles written by older
  versions; it only rebuilds what `__getstate__` drops. Re-pickle with 5.x
  first if you hold old pickles. `ComputeContext.ensure_runtime_state()`
  was removed with it.
- The aperture-filter cache key for the default filter is `"Q_crittenden"`
  instead of `"Q_T"` (internal, visible in pickled state).

### Kept deliberately

Full-sky map inputs (they are a convenience, not a compatibility shim),
the per-(bin, row) GPU kernels and `kernel.tiled = False` (they are the
fallback beyond the tile's accumulator budget), the ElementwiseKernel and
CPU fallbacks (they cover missing hardware, not old versions), and the
bit-for-bit guarantee of `resolution_factor=None`.

## 5.1.0

### Added
- **Multi-GPU by patch range.** `Correlation(..., device=[0, 1, ...])` returns a
  `MultiDeviceCorrelation`: one `Correlation` per device over a contiguous
  range of patches, measurement calls run one thread per device and the
  per-patch outputs are concatenated in patch order. Results are bitwise
  identical to a single-device run, and device memory per GPU scales as
  `n_patches / n_devices`. A single device stays the default and is
  unaffected. `load_pairs()` gives each device its own slice of one pair
  file; `save_pairs()` is not supported on a group (write it from a
  single-device instance).

- **`pack_host_pairs=`** (opt-in, default `False`): applies the payload
  packing of `pack_pairs` already at pair-finding time, so the host arrays
  and the pair file hold 8 instead of 24 bytes per pair as well
  (`pair_inds` / `pair_exp2phi` become `None`; the payload is in
  `packed_pairs`). The row blocks store global ids, so a packed file is
  still mask-independent and still slices by patch. Independent of
  `pack_pairs`, and bitwise identical on the device to packing inside
  `prepare()`. The price: the geometry is quantised everywhere, so the file
  is no longer exact and cannot be turned back into one. Such files are
  written as **format version 4** and carry a `packed_pairs` attribute;
  earlier CosmoFuse versions reject them instead of misreading them.

### Changed
- `PinnedMapPipeline` is now **`MapLoader`** and `RowSpaceMapLoader` is now
  **`MapFileLoader`**. The old names still work and warn.

### Performance
- **Pair search: no more sorting.** The search kernel counted accepted pairs
  per row and the result was then put in angular-bin order with an `argsort`
  plus seven fancy-indexed gathers — at nside 2048 that was 97 % of the
  per-patch preprocessing time and ~600 MB of temporaries per patch. The
  kernel now counts per (row, bin) and writes the pairs already grouped by
  bin, in exactly the same order as the stable sort produced. Preprocessing
  is **3.3× faster at nside 512** (16.4 → 5.0 ms/patch) and **6.6× faster at
  nside 2048** (9.1 → 1.4 s/patch, i.e. ~2.3 h → ~21 min for 917 patches).
  Pair output is unchanged bitwise.
- **Batched normalisation of the tomographic wrappers.** `vectorized_shear_shear`,
  `vectorized_density_density`, `vectorized_density_shear` and the device path
  of `get_3x2pt_tomo` normalised one tomographic combination at a time, which
  issued roughly a dozen tiny elementwise kernels per combination; the GPU was
  idle waiting for launches. The whole stack is now normalised in one set of
  operations. Results are bitwise unchanged (verified on an A100 over 56
  DES-Y3-sized map-sets). On an A100 at nside 512, 917 patches, 4 source bins,
  `resolution_factor=2.9`: `vectorized_shear_shear` 4.40 → 3.03 ms and
  `get_full_tomo_shear` 5.84 → 4.40 ms per map-set with device-resident input.

### Testing
- The handful of exactness gates that dominate the suite wall time are marked
  `slow`; `pytest -m "not gpu and not slow"` runs 308 of 317 tests in ~1 min
  instead of ~5.5 min. CI still runs them (it only deselects `gpu`).

## 5.0.0 (2026-09-18)

Major release: static treecode, compact row space, combination-tiled and
packed GPU kernels for every pair statistic, ring-buffer map loader.

**Breaking / result-changing (hence 5.0):**
- cross-bin xi+- is now the ratio of the summed orientations (was the mean of
  the two orientation ratios) — see *Changed*;
- the pair search runs at float64 by default (`pair_search_precision`);
- `select_patch_centers` / `Correlation.from_mask` select patches by
  |filter|-weighted masked fraction by default;
- map arrays of the wrong length raise; pair files with virtual rows use
  format version 3 (unreadable by <= 4.20 by design);
- internal kernel contracts changed (one output row per combination; the
  GPU `kernel_3x2pt_tomo_fused` wrapper became `kernel_3x2pt_tomo_aperture` +
  `kernel_3x2pt_tomo_pairs`).

Unchanged: with explicit patch centres and `pair_search_precision="rotation"`
the auto-combination results of every public method are bit-for-bit those of
4.20.0 (verified on CPU, and on an A100 against 4.20.0 on the DES Y3
production geometry with real maps); existing full-resolution pair files
load as before.

### Added
- **Static treecode** (`Correlation(..., resolution_factor=True)` for the
  default `k = 4`, or `resolution_factor=k`; opt-in, default `None` = full
  resolution). Every
  angular bin is measured on the coarsest HEALPix level whose pixel size is
  `<= theta_lo / k`; coarse cells are built per patch (exact top-hat patch
  window), carry weighted means / weight sums, and sit at the binary-mask
  centroid of their members. Pair geometry shrinks by orders of magnitude
  (nside 2048, 5'–175', 917 DES patches, k = 2.9: 592 M pairs / 17.7 GB instead
  of ~72 G pairs). **It is a different, windowed estimator** — see
  `benchmarks/static_treecode/STAGE1_RESULTS.md` for the measured suppression
  per `k`. No CUDA kernel was changed for it:
  cells are appended as virtual rows behind the pixel rows.
- `aperture_nside=`: aperture statistics on the (weighted-mean) degraded map.
- `Correlation.level_table`: per-bin resolution provenance; written to pair
  files and pickles. `load_pairs` adopts the file's estimator and raises on an
  explicitly conflicting request.
- Preflight memory check before pair finding (`memory_budget_gb=`): raises a
  `MemoryError` naming a `resolution_factor` that fits.
- **Compact row space**: device map buffers and device indices cover only the
  unmasked pixels. Measurement methods accept full-sky *or* row-space maps
  (`Correlation.row_pix`, `n_active`, `to_row_space()`, `row_pix_hash`).
  Masked pixels are never read (they may be NaN). Read-only weight maps are
  gathered / uploaded / degraded once.
- **Combination-tiled GPU pair kernels** for xi+-, xi_g and xi_t (default;
  `kernel.tiled = False` restores the per-row kernels): one pass over the
  pairs for all tomographic combinations, shared tiles in
  `cuda/pair_tiles.cuh`. The fused 3x2pt path (`get_3x2pt_tomo`) runs its
  pair statistics in the same tiled kernels (its own kernel keeps only the
  aperture sections). Auto-only (`gc_auto_correlations_only`) and subset
  (`ggl_bin_combinations`) requests are served by the tiles too.
  `get_3x2pt_tomo` walks the pairs **once** for all three statistics
  (`cuda/tomo_tiled_3x2pt.cu`, up to 120 accumulators per thread; 8–20 %
  faster than three passes for 4 + 4 bins on an A100). xi+-:
  bit-identical auto combinations, 4.2–4.5x faster on an A100.
- **Payload packing** (`pack_pairs=True`, opt-in): 8 instead of 24 bytes per
  pair on the device (uint16 rotation angles + uint16 patch-local row indices,
  per-patch contiguous row blocks; kernels `gpu_tiled_packed_reduce_{xipm,dd,ds}`).
  A100: pair memory 6.6 → 2.3 GB (nside 512 production), 14.5 → 5.2 GB
  (nside 2048, k = 2.9), speed unchanged; GPU vs CPU 5e-15. Not bit-identical
  to unpacked: estimates move by 3e-5 (rms) of their patch scatter, unbiased.
  Serves every tomographic method (`vectorized_*`, `get_full_tomo_*`,
  `get_3x2pt_tomo`) and the aperture statistics on the GPU; the single-map
  `compute_*` methods need `pack_pairs=False`.
- `RowSpaceMapLoader`: reader threads + ring of pinned buffers + upload
  stream; results identical to the serial loop, GPU idle < 5 %.
- `pair_search_precision=`: precision of the pair search, decoupled from the
  stored rotation precision; **default `"float64"`** (no memory cost).
  `"rotation"` restores the historical search at rotation precision (float32
  by default), `"auto"` keeps that at full resolution and uses float64 with
  `resolution_factor`. A float32 search resolves separations only to
  `6e-8 / theta^2` (3 % at 5'); on the DES Y3 nside-512 production geometry it
  mis-bins pairs at the 0.05 sigma (rms) / 0.9 sigma (max) per-patch level
  against the exact result. A warning is logged when the jitter exceeds 1 %
  at `theta_min`.

### Changed
- **Cross-bin xi+- estimator.** For a tomographic combination (i, j) with
  i != j every pair contributes in both orientations (bin i at pixel a and j
  at b, and vice versa). CosmoFuse used to average the two orientation
  *ratios*, `(N_ab/W_ab + N_ba/W_ba) / 2`; it now takes the ratio of the
  summed orientations, `(N_ab + N_ba) / (W_ab + W_ba)` — the standard
  weighted estimator, TreeCorr's definition, and what xi_g and xi_t already
  did. The two coincide for equal per-bin weights; with the real per-bin DES
  Y3 weights they differ by ~3 % of max|xi+| per patch (0.02 sigma). The old
  form also halved a bin's estimate wherever one orientation had zero weight
  (bins with different masks). Auto combinations are unchanged. Applies to
  `vectorized_shear_shear`, `get_full_tomo_shear` and `get_3x2pt_tomo` on
  both backends; an explicit directional `sumofweights` is summed the same
  way. The single-map `compute_shear_shear` keeps its historical form.
- **Pair search precision default** is `"float64"` (see *Added*).
- **Patch selection default** (`select_patch_centers`, `Correlation.from_mask`):
  the filter-support masking check now uses the |filter|-weighted masked
  fraction by default (`filter_weighting="abs"`); `"signed"`/`"raw"` measures
  the deviation of the filter integral instead, `"pixels"` restores the old
  default. Always from the binary mask. New `U_crittenden`, `U_schneider`.
  Explicitly passed patch centres and existing pair files are unaffected.
- Map arrays whose last axis is neither `npix` nor `n_active` now raise a
  `ValueError` (previously undefined behaviour / out-of-bounds gathers on GPU).
- Pair file format: files **with virtual rows** are written as
  `format_version = 3` and store their index datasets under new names
  (`tc_pair_inds`, `tc_Q_inds`). CosmoFuse <= 4.20 accepts any version >= 2
  and would read virtual-row ids as pixel ids; with the new names it fails
  with a `KeyError` instead. Full-resolution files are still written as
  version 2 and remain readable by old versions. Unknown future versions are
  now rejected.

### Fixed
- Cumulative pair offsets were built as int32 and overflowed beyond 2^31
  pairs; now int64.
- `PinnedMapPipeline`: (i) a pinned host slot could be refilled while its
  previous asynchronous copy was still queued, and (ii) the upload stream did
  not wait for queued kernels that still read the device slot it overwrites.
  Both only bite when results stay on the device and the host runs ahead of
  the GPU. Covered by a CUDA stream simulator test (`tests/cuda_stream_sim.py`).

### Performance (A100 80 GB, DES Y3, 917 patches, 4 source bins, float32 maps + float64 accumulators)
| workload | 4.20.0 | 5.0.0 |
|---|---|---|
| nside 512, 15'–176', full resolution, real maps from disk (bit-identical results) | 157 ms / map-set | 26 ms |
| same, `resolution_factor=4` | — | 11.5 ms |
| same, `resolution_factor=2.9` | — | 9.1 ms |
| nside 2048, 5'–175', `resolution_factor=2.9`, `aperture_nside=512` | impossible (~1.7 TB) | 78 ms (17.7 GB) |
| `get_full_tomo_density`, nside 512, 4 lens bins, row-space host input (4.20 column: its per-row kernels) | 31.5 ms | 8.4 ms |
| `get_full_tomo_ggl`, 4 lens x 4 source bins | 96.4 ms | 23.6 ms |
| `get_3x2pt_tomo`, 4 + 4 bins | 203 ms | 38.7 ms |
| `get_3x2pt_tomo`, nside 2048, `resolution_factor=2.9`, packed | impossible | 266 ms |
