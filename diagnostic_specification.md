# False-synchrony diagnostic — specification (Phase 2)

Working name of the output object: `FalseSynchronyResult` (the score field is
`synchrony_score`; the label "False Synchrony Index" is intentionally NOT
finalized per spec §9).

## Goal

Given a panel (unit-level series preferred; aggregate + timing metadata as a
fallback), decide whether an apparent aggregate breakpoint reflects a common
event or aggregation-induced false synchrony, and report the estimand the
aggregate breakpoint actually delivers.

## Inputs

- `aggregate_series`: \(\bar y_w(t)\) on grid \(t_1..t_n\).
- `unit_series` (preferred): matrix \(Y_i(t)\) or equivalent tidy panel.
- `weights` (optional): \(w_i(t)\); absent ⇒ equal weights.
- `window_metadata` (optional): per-unit observation windows \([L_i,U_i]\).
- `unit_cp_options`: per-unit detector settings (default LS single-break).
- `noise_model`: residual model for unit-level fits (default iid Gaussian).
- `regime_calibration`: thresholds derived from inferentially meaningful
  quantities (below), not fixed SD cutoffs.

## Processing

1. Aggregate-break operator fit: \(T_{\mathrm{LS}}\), break-strength
   \(S=(\bar m_R-\bar m_L)/\hat\sigma_{\text{agg}}\).
2. Per-unit break fit: \(\hat\tau_i\), uncertainty \(s_i\) (residual-noise
   delta/subsampling), break-strength \(S_i\).
3. Timing distribution: \(\hat F_\tau\) via weighted kernel over
   \(\hat\tau_i\); **error-corrected** version \(\hat F_\tau^{\text{dec}}\)
   deconvolving the average unit-level error law \(F_e\) (constructed from the
   \(s_i\) distribution — see `resampling_plan.md`).
4. Synchrony score components:
   - `s1_width`: dispersion of \(\hat F_\tau^{\text{dec}}\) relative to the
     aggregate apparent transition width \(w_{\text{agg}}\) (from \(m'\) FWHM).
     Large ratio ⇒ diffuse timing.
   - `s2_shape`: mass concentration of \(\hat F_\tau^{\text{dec}}\)
     (spike/tail structure; guards against CE-SPIKE).
   - `s3_alignment`: consistency of unit break strengths \(S_i\) — faint unit
     breaks under strong aggregate break is the smoking gun.
   - `s4_weighting`: weight-timing correlation (P4) and observation-window
     timing correlation (P5) — flags explainable tilts vs real dispersion.
   - `synchrony_score`: bounded \([0,1]\) composite (calibration constants set
     in Phase 3 simulation; all four components reported individually — no
     opaque scalar).
5. Regime classification (spec §10): posterior-style probabilities over
   {SYNCHRONOUS, CLUSTERED, DIFFUSE_ASYNCHRONOUS, NO_TRANSITION} via
   parametric bootstrap over the timing family (Gaussian vs mixture vs
   uniform-comparison); classification carries uncertainty, never a hard
   label alone.
6. Estimand summary: reported \(T_{\mathrm{LS}}\) is interpreted against
   \(E[\tau]\), median, mode of \(\hat F_\tau^{\text{dec}}\) with the P7
   window-dependence caveat printed whenever window coverage is asymmetric.
7. Event-time alignment diagnostic: realign unit series to \(\hat\tau_i\),
   recompute the aligned aggregate; compare aligned vs unaligned break
   sharpness and timing spread — CE-ALIGN check (`event_aligned_summary`).

## `FalseSynchronyResult` fields (spec §9, verbatim)

`aggregate_breakpoint`, `aggregate_break_strength`, `unit_breakpoints`,
`unit_breakpoint_uncertainty`, `timing_distribution`, `timing_dispersion`,
`cluster_structure`, `synchrony_score`, `regime_probabilities`,
`event_aligned_summary`, `interpretation_warning` — a human-readable string
listing which of the P3/P4/P5/P7 distortions plausibly apply (e.g., "window
asymmetry: LS breakpoint pulled toward better-covered times").

## Non-negotiable requirements

- If `unit_series` is unavailable: the result must still be produced but with
  `interpretation_warning` stating that synchrony cannot be certified from the
  aggregate alone (P6c) and regime probabilities are prior-dominated.
- Every numeric field must carry an uncertainty where identifiable; a point
  estimate alone is a spec violation.
- No field may be silently NaN — a missing-information case writes an explicit
  `not_estimable` sentinel + warning.
