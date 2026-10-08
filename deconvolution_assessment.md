# Deconvolution assessment — KEEP / SUPPLEMENT / DROP (Phase 2)

## Question (spec §12)

Should the package/paper attempt deconvolution of the aggregate to recover
\(F_\tau\) or \(g\)?

## Assessment

- **Aggregate-only deconvolution** (recover \(F_\tau\) from \(m\), \(g\) known):
  identifiable in principle (P8a) but severely ill-posed for the smooth
  kernels we use (logistic/probit blur); classical inverse-problem CP
  literature (Neumann 1997) already covers it, and our task file's novelty
  budget is elsewhere. **Decision: SUPPLEMENT.** Implement a minimal,
  clearly-labeled `deconvolve` utility for the known-\(g\) case (useful for
  benchmarking the toy exact cases — step \(g\) where inversion is trivial)
  but do NOT position it as a contribution and do NOT claim recovery-rate
  results.
- **Unknown-\(g\) identification**: impossible from the aggregate alone
  (P8b, blur-kernel exchange). **Decision: DROP** as an inference goal; the
  impossibility result itself is a paper point (motivates unit-level data).
- **Unit-level timing deconvolution** (recover \(F_\tau\) from
  \(\hat\tau_i\) + error law \(F_e\)): tractable and needed for the
  diagnostic's `timing_distribution`. However the methodology is covered by
  the SSRN (2024) econometric result — we reuse, cite, and do not claim it as
  our inferential contribution. **Decision: KEEP, as supporting machinery**
  (the `s2_shape`-corrected \(\hat F_\tau^{\text{dec}}\)), implemented via a
  parametric-error deconvolution with fixed regularization, no asymptotic
  guarantees asserted.

## Bottom line

KEEP deconvolution only inside the diagnostic (unit-level error correction);
SUPPLEMENT aggregate deconvolution as a demo/verification tool; DROP
unknown-\(g\) identification. The paper's novelty stays on the
aggregate-estimand and diagnostic layers.
