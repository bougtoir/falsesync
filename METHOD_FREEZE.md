# METHOD_FREEZE.md — falsesync diagnostic, final frozen version

Frozen at the final development stage (Phase 4), before manuscript preparation.
After this point scientific behavior may change only to fix verified bugs.

## Final model version
`FalseSynchronyModel` v1 (falsesync/model.py) + `RegimeClassifier`
(falsesync/regimes.py) trained on calibration grid
`simulations/results/calibration.csv` (1071 cells), temperature-scaled
(T=0.63), stored at `simulations/calibration/classifier.joblib`.

## Final feature set (16)
agg_strength, excess_sd_ratio, n_modes, mode_mass2, s1_width,
s3_alignment, s4_weighting, noise_floor_ratio, fwhm_ratio, agg_skew,
log_n, log_T, frac_valid_units, binseg_n_breaks, unit_strength_med
(see falsesync/features.py FEATURE_NAMES).

## Final regime definitions
SYNCHRONOUS | NEAR_SYNCHRONOUS | DIFFUSE_ASYNCHRONOUS | CLUSTERED |
NO_TRANSITION (REGIME_CLASSES). Regime mapping: degenerate/borderline
dispersion -> SYNC; small-dispersion unimodal -> NEAR_SYNC; broad
unimodal -> DIFFUSE; 2+/3-component mixtures -> CLUSTERED; flat series
-> NO_TRANSITION. NEAR_SYNC retained as a class; primary reporting uses
continuous P(sync) and corrected dispersion because the SYNC/NEAR
boundary is statistically hard at tau_sd ~0.05-0.10 window units
(near_sync_sensitivity.csv).

## Final classifier/calibration
sklearn LogisticRegression (multinomial, balanced class weights),
temperature calibration on held-out calibration split. Locked-eval:
accuracy 0.861, balanced 0.893, Brier 0.197, ECE 0.064.

## Final warning rules
- false_common_event_warning: agg_strength >= 3.0 AND
  P(SYNCHRONOUS) < 0.5 (pre-specified, spec §6).
- operator_sensitivity_warning: operator spread > 20% of window.
- trend_confounding_warning: detrending shrinks tau-hat dispersion
  below 0.6x raw AND median unit slope > 0.05.
- indeterminate_observation_process (Phase 4 addition):
  |corr(window_start, tau_hat)| >= 0.5 AND truncated-unit fraction
  >= 0.2 -> synchrony inference marked indeterminate; certification
  suppressed, warning raised (missingness_mitigation_results.csv).

## Final missingness status (spec §5)
DETECTABLE_NOT_CORRECTABLE. Observation-process diagnostic detects
tau-correlated entry (0.5-0.75 across strengths) and suppresses
certification (false-certify 0/89 vs 1/89 baseline in the targeted
subset), but unbiased synchrony inference under informative missingness
is NOT established; undetected weak-informative cells remain a risk.
Late-entry truncation occasionally false-flags (0.4) — conservative.

## Final operator set
ls (least-squares, primary), maxslope, cusum, binseg (3-bkp) for the
operator-sensitivity block.

## Final trend-confounding behavior
Per-unit linear detrend + tau-hat refit; warning when detrended sd <
0.6 * raw sd and median slope > 0.05. Classification is NOT forced —
the flag is surfaced instead.

## Final uncertainty strategy
Unit-level bootstrap break uncertainty (unit_boot=30); noise floor =
median bootstrap sd; corrected dispersion = sqrt(max(sd^2 - floor^2,0));
deconvolved timing density reported but not used for mode counting.
Interval coverage / hierarchical shrinkage not claimed.

## Final API version
falsesync 0.1.0-phase4:
`FalseSynchronyModel(classifier, method='ls', operators=(...),
unit_boot=30, agg_strength_crit=3.0, sync_prob_crit=0.5, seed=0).fit(
t, values, weights=None, windows=None, aggregate=None) ->
FalseSynchronyResult` (+ `indeterminate_observation_process` flag in
components when the observation-process diagnostic fires).
