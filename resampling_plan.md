# Resampling / inference plan (Phase 2)

Purpose: uncertainty for the diagnostic's timing distribution, regime
probabilities, and synchrony score — without claiming asymptotic results we
have not proved.

## Layers

1. **Unit-level break uncertainty** \(s_i\): residual bootstrap of each unit
   series — resample residuals around the two-segment fit, refit
   \(\hat\tau_i^{*}\); \(s_i^2=\mathrm{Var}(\hat\tau_i^{*})\). Also yields
   the unit-level error law \(F_e\) (distribution of \(\hat\tau_i^*-\hat\tau_i\))
   used for deconvolution.
2. **Timing-distribution uncertainty**: cluster bootstrap over units —
   resample units with replacement, recompute
   \(\hat F_\tau^{\text{dec}}\); report band and dispersion CI.
3. **Regime-probability calibration**: parametric bootstrap — simulate panels
   under each candidate regime with parameters matched to the data; measure
   frequency of each regime classification; output probabilities, not a hard
   label.
4. **Aggregate-break strength**: moving-block bootstrap on the aggregate
   series (block length from residual autocorrelation) for \(T_{\mathrm{LS}}\)
   and \(S\) intervals.

## Structure

- All resampling uses `numpy.random.Generator` with a master seed; every
  reported number is reproducible from `simulations/run_phase2_checks.py`.
- No asymptotic coverage claims: bootstrap intervals are presented as
  bootstrap intervals. Any future consistency statement must be proved in
  Phase 3 before appearing in the manuscript.
- Default \(B=500\) for unit level, \(B=1000\) for timing bands; fixed in code,
  not tuned per result.

## Failure handling

- Unit fits that fail on a resample (flat series, edge break) are counted and
  reported as `fit_failure_rate`; a rate > 10% triggers an
  `interpretation_warning` entry.
- Deconvolution stabilization: Tikhonov-type damping on the high-frequency
  tail of \(\hat f\); bandwidth/strength fixed a priori, not tuned to output.
