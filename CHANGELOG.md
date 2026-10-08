# Changelog

## v0.1.0 — 2026-10-08

Version corresponding to the manuscript "Aggregation-Induced False Synchrony
in Change-Point Analysis" as submitted to *Technometrics*.

- Unit-level transition model, weighted aggregation, and breakpoint
  operators (least squares, maximum slope, CUSUM, binary segmentation).
- `FalseSynchronyModel` diagnostic: aggregate breakpoint, unit-level break
  estimates with uncertainty, floor-corrected timing dispersion, five-class
  calibrated regime probabilities, operator-spread, trend-confounding and
  observation-process flags, and the false common-event warning.
- Calibrated classifier artifact (`simulations/calibration/classifier.joblib`).
- Simulation configurations: development, calibration, and the locked
  evaluation grid (`simulations/configs/evaluation_locked.yaml`).
- Replication entry points (`replication/run_quick.py`,
  `replication/run_full.py`), data acquisition with checksum verification
  (`fetch_data.py`), and reference result tables and figures.
