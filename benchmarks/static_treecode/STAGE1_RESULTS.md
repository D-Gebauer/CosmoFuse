# Stage 1 results — science gate (2026-09-17)

`stage1_prototype.py` → `results/stage1.npz`, `stage1_analyse.py` →
`results/stage1_analysis.txt`. Standalone brute-force code (independent of the
CosmoFuse measurement path). nside 2048, real DES Y3 mask, 96 Q110 patches
(incl. dec 0°, dec −62°, 25 % and 50 % masked) × 12 Gaussian realisations
from a CCL halofit C_ℓ (sources ⟨z⟩≈0.75, lenses z=0.4, b=1.6), pixel window
applied, lmax = 6143. No nside-2048 sims exist on the cluster (production
maps are nside-512 footprint cut-outs), hence synthetic fields. Window
effects on 2-point functions depend only on ξ(θ), so Gaussian fields suffice;
T10 (ζ level) needs non-Gaussian maps and is deferred to the real path.

Bins: 5–250′, 11 log bins. Every coarse level (1024…64) was measured for
*every* bin, so the result is a (statistic, level, bin) matrix and any k is
just a selection rule.

## 0. Reference validation
Full resolution / theory = 1.00–1.03 for all four statistics below 60′
(larger bins: cosmic variance, flat-sky theory, truncated last bin). Note:
healpy-generated maps need `flip_g2=True` for positive γ_t in CosmoFuse's
convention (`flip_g1` gives identical ξ± but γ_t < 0).

## 1. Suppression vs effective k (= θ_lo / pixel size), noise-free

| k_eff | ξ− | γ_t | ξ+, ξ_g |
|---|---|---|---|
| ≈ 1.45 | −29 % | −12 % | −2.5 % |
| ≈ 2.1 | −18…−20 % | −2…−6 % | ≲ 1 % (noise-limited) |
| ≈ 3.0 | −8…−10 % | −2…−2.6 % | ≲ 1 % |
| ≈ 4.3 | −4.3…−5.3 % | −0.8…−2.4 % | < 0.7 % |
| ≈ 6.1 | −2.0…−2.9 % | −0.3…−0.5 % | < 0.3 % |
| ≈ 8.7 | −0.9…−1.1 % | ≤ 0.3 % | < 0.3 % |

Same at all levels (1024/512/256) → a function of k_eff only. The guide's
estimates (ξ−: 30 / 8 / 2 % at k = 2 / 4 / 8) were pessimistic.

## 2. Decisions for the geometry
* **Centroid vs HEALPix parent centre:** ≤ 0.5 % at k ≥ 2 on levels
  1024/512, 1–3 % on the coarsest levels. Keep the binary-mask centroid.
* **Parallel transport of member shears to the centroid:** ≤ 1·10⁻⁴
  everywhere, including the dec −62° patch. **Not needed** → D is real, all
  entries 1, and can be applied as a chain of child→parent sums.
* High-dec and heavily masked patches show the same ratios within noise.

## 3. T9 data/sim consistency
ratio(n_gal Poisson weights) / ratio(uniform weights) − 1 = 0 ± ~1·10⁻³ for
all statistics at k_eff ≥ 2. Passed.

## 4. Noise: decorrelation without information loss
With realistic shape noise (σ_e = 0.26, 1.5 gal/arcmin²) the per-patch coarse
and fine estimates have **equal variance** (ratio 1.00 ± 0.04 for k_eff ≥ 2)
but are only partially correlated: ρ(ξ+) ≈ 0.65 / 0.83 / 0.88 / 0.93 / 0.96
at k_eff ≈ 2.1 / 3 / 4.3 / 6 / 8.7 (ξ− slightly lower, γ_t higher, ξ_g
≥ 0.97). Interpretation: ~σ = 0.58 p separation scatter moves a large
fraction of fine pairs into the neighbouring bin, so the coarse statistic is a
*differently windowed* estimator with the same noise level, not a noisier copy
of the fine one. S/N per bin therefore changes only through the signal ratio
(table above). Consequence: coarse and full-resolution data vectors must
never be mixed (data vs sims, or covariance vs data) — they differ at the
noise level, not just by the few-% signal suppression.

## 5. Cost per k (median real patch, 24 B/pair; 8 B after §8.2 packing)

| k | nside per bin | pairs/patch | GB per 1000 patches | worst ξ− / γ_t |
|---|---|---|---|---|
| full res | 2048 × 11 | 78.3 M | 1880 | — |
| 2 | 2048,1024²,512²,256²,128²,64² | 0.25 M | 6.1 (2.0) | −20 % / −7 % |
| 2.9 | 2048²,1024²,512²,256²,128²,64 | 0.64 M | 15.5 (5.2) | −10 % / −3 % |
| 4 | 2048³,1024²,512²,256²,128² | 1.44 M | 34.6 (11.5) | −5 % / −2.4 % |
| 5.8 | 2048⁴,1024²,512²,256²,128 | 2.98 M | 71.6 (23.9) | −2.9 % / −0.5 % |
| 8 | 2048⁵,1024²,512²,256² | 5.88 M | 141 (47) | −1.1 % / −0.3 % |

## Gate
A k that fits an 80 GB device exists at every accuracy level down to ~3 %
ξ− suppression (k ≤ 5.8 unpacked, k = 8 with payload packing), and the
measured suppression is below the estimates accepted in guide §2.5 →
**gate for Stage 3 passes on the guide's own criterion.** The choice of the
production k remains the user's (k is a runtime parameter; nothing in the
implementation depends on it).

Recommendation: one global k (no per-statistic levels — ξ+ and ξ− share a
kernel pass and a pair list; separate lists would double the ξ± work for a
few-% S/N gain in ξ−). k = 4 if memory allows (≤ 5 % ξ−), k = 2.9 otherwise.
