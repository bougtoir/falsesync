# Changelog

## v0.1.2 — 2026-10-08

Packaging-only release; code, reference results, and manuscript numbers are
unchanged from v0.1.1.

- Upper bounds on runtime dependencies: `numpy<2.4` (the code uses
  `np.trapz`, removed in NumPy 2.4) and `scikit-learn<1.8` (the classifier
  uses `LogisticRegression(multi_class=...)`, removed in scikit-learn 1.8;
  the bundled `classifier.joblib` was serialized with scikit-learn 1.7.2).
- PyPI metadata: SPDX license expression, keywords, classifiers, and project
  URLs (repository, issues, changelog, Zenodo concept DOI).
- README: note that pixel-level figure comparison in the QUICK replication
  assumes matplotlib 3.10.

## v0.1.1 — 2026-10-08

Version corresponding to the manuscript "Aggregation-Induced False Synchrony
in Change-Point Analysis" as submitted to *Technometrics*. Supersedes
v.0.1.0 (Zenodo 10.5281/zenodo.23233537), whose archive predates the two
reproducibility fixes below.

- `weighting_truncation_results.csv` (`warning`) and
  `missingness_mitigation_results.csv` (`warning_A`) regenerated with the
  frozen model, which raises the false common-event warning when the
  observation process is indeterminate. Only these boolean columns change.
- `replication/run_full.py` requires the `cp` extra (ruptures); the
  reference binary-segmentation results use `ruptures.Binseg`.

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
