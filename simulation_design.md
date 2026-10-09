# Phase-3 simulation design (spec §3–§4)

## Grid structure

Simulation cells are defined by `SimConfig` (src/falsesync/simengine.py).
To avoid an exploding full-factorial grid, the design is a **fractional
factorial plus one-factor-at-a-time stress blocks**:

### Regime ↔ timing-family coupling (fixed mapping, §3.C/D)

| regime | f_family used | comment |
|---|---|---|
| synchronous | degenerate | point mass at center |
| near_synchronous | narrow_normal (σ=0.2) | |
| diffuse | broad_normal (σ=0.9), uniform, skewed, spike_diffuse, heavy_tail | 5 variants |
| two_cluster | two_normal_mix (δ=2.5) | separated per P6a |
| three_cluster | three_normal_mix (δ=2.0) | |
| no_transition | degenerate, amplitude 0 | pure noise panel |

### Core grid (calibration)

- N ∈ {25, 60, 120}, T ∈ {60, 120, 240}
- regimes × f-families as above (7 regime–family cells)
- shape ∈ {step, logistic} (probit/linear/asymmetric in eval grid)
- amplitude ∈ {0.5, 1.0, 2.0} (low/medium/high effect size)
- width ∈ {0.2, 0.5}
- noise: iid Gaussian sd 0.15
- weights equal; windows complete
- 3 seeds per cell → roughly 3·3·(7)·2·3·2·3 ≈ 2268 base cells, subsampled
  to the YAML `max_cells` budget via seeded Latin-hypercube-style thinning —
  never per-outcome selection.

### Stress blocks (one dimension varied at a time)

- `stress_noise`: noise_kind ∈ {t3, heterosk, ar1, seasonal}
- `stress_weights`: weight_kind ∈ {static, timing_corr, timevarying}
- `stress_missing`: miss_kind ∈ {mcar, dropout, late_entry, informative}
- `stress_hetero`: amp/width/base/trend/noisevar heterogeneity > 0
- `stress_trend`: trend_kind ∈ {linear, nonlinear, local_drift, mixed}
- `stress_operators`: all 4+ operators on the same cells

### Split (§4)

- **development**: small coarse grid for debugging (discarded).
- **calibration**: the grid above, seeds 1000–1999. Used to fit the
  multinomial-logistic regime classifier + temperature + warning thresholds.
- **evaluation_locked**: disjoint seeds 3000–3999, plus **held-out
  configuration values** (shape=probit/asymmetric, N=40/200, f_params
  off-grid σ values, different mixtures of miss/weights) — committed to git
  BEFORE it is run. Final reported numbers come only from the locked grid.

## Calibration targets (§5–§6)

- Regime probabilities: temperature-scaled multinomial logistic regression
  (5 classes: SYNCHRONOUS, NEAR_SYNCHRONOUS, DIFFUSE_ASYNCHRONOUS,
  CLUSTERED, NO_TRANSITION).
- synchrony_score ≡ P(SYNCHRONOUS) (calibrated), not a heuristic index.
- **False common-event warning** fires iff
  `aggregate_break_strength ≥ 3` AND `P(SYNCHRONOUS) < 0.5`
  (sync_prob_crit). Evaluated on locked grid: sensitivity under
  asynchronous regimes, false-warning rate under true synchrony, and
  behavior under no_transition.
- The 'certifies synchrony when it shouldn't' error (§5 critical error) is
  tracked as `P(SYNCHRONOUS) > 0.5` under non-synchronous truth with strong
  aggregate break.

## Reproducibility

- `configs/*.yaml` hold every grid parameter + master seed; every cell gets
  `seed = master_seed + cell_index`.
- QUICK mode: `max_cells` reduced (≈150 calib, ≈80 eval); FULL mode: full
  grid. Wall time / versions recorded in `reproducibility_manifest.csv`.
