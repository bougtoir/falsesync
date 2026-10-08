# falsesync

Diagnostics for aggregation-induced false synchrony in change-point analysis.

## Scientific motivation

Many engineering and observational studies average a panel of unit-level
series (turbines, battery cells, sensors, regions) and then fit a change point
to the aggregate. A sharp aggregate breakpoint is easily read as evidence that
the units changed together. That reading is not justified in general: when
unit transitions occur at heterogeneous times \(\tau_i \sim F_\tau\), the
aggregate mean is a smoothed curve \(m(t) = a + A\,(g * F_\tau)(t)\) that can
still produce a well-defined, strong breakpoint. The fitted aggregate
breakpoint is a functional \(T_M(F_\tau, g, w, \text{window}, \text{noise})\)
that depends on the breakpoint operator \(M\), the observation window, the
weights, the observation process, and the noise, and it need not coincide with
the mean, median, or mode of the timing distribution.

## What falsesync does

- simulates unit-level transition panels with configurable timing laws,
  transition shapes, weights, observation windows, trends, and noise;
- computes weighted aggregates and aggregate breakpoints under several
  operators (least squares, maximum slope, CUSUM, binary segmentation) and
  reports the spread across operators;
- estimates unit-level breakpoints with bootstrap uncertainty and a
  floor-corrected timing dispersion
  \(\mathrm{sd}_{\mathrm{corr}} = \sqrt{\max(\mathrm{sd}^2(\hat\tau_i) - \bar s^2, 0)}\);
- returns calibrated probabilities for five timing regimes
  (`SYNCHRONOUS`, `NEAR_SYNCHRONOUS`, `DIFFUSE_ASYNCHRONOUS`, `CLUSTERED`,
  `NO_TRANSITION`) from a classifier calibrated on a separate simulation grid;
- issues a false common-event warning when the aggregate break is strong but
  the calibrated probability of synchrony is low, plus warnings for trend
  confounding, operator dependence, and observation-process risks.

## What falsesync does not claim

- It is not a causal method and does not identify what caused any transition.
- A low synchrony probability is a warning that the aggregate break should not
  be read as a common event; it is not a test that rejects synchrony.
- The regime probabilities are calibrated only for data-generating conditions
  resembling the calibration grid (see `simulations/configs/`).

## Known limitations

- Informative observation processes (entry or dropout related to transition
  timing) can create apparent synchrony; this failure mode is detectable in
  simulation but not corrected by the package.
- Trend heterogeneity across units can be confounded with timing
  heterogeneity.
- Unit-level break uncertainty inflates apparent timing dispersion; the
  floor correction reduces but does not remove this effect.
- The aggregate breakpoint location is operator-dependent; different
  operators can give materially different locations on the same aggregate.
- Without unit-level data, synchrony cannot be certified from the aggregate
  alone.

## Installation

Python >= 3.10.

```bash
git clone https://github.com/bougtoir/falsesync.git
cd falsesync
pip install .            # or: pip install -e ".[dev]" for tests
```

Optional: `pip install ".[cp]"` adds `ruptures`-backed detectors.

## Minimal example

```bash
python examples/minimal_example.py
```

The example simulates 80 units with diffuse (normal, SD 1.1) transition times,
fits `FalseSynchronyModel` with the shipped calibrated classifier, and prints
the aggregate breakpoint, the corrected timing dispersion, the five regime
probabilities, and the warnings. In outline:

```python
import joblib
import numpy as np
from falsesync import simulation
from falsesync.model import FalseSynchronyModel

rng = np.random.default_rng(1)
t = np.linspace(0, 10, 180)
taus = simulation.sample_taus(80, {"kind": "normal", "mu": 5.0, "sigma": 1.1}, rng)
panel = simulation.simulate_panel(t, taus, shape="logistic", amplitude=1.5,
                                  width=0.3, noise_sd=0.2, rng=rng)
clf = joblib.load("simulations/calibration/classifier.joblib")
res = FalseSynchronyModel(classifier=clf).fit(t, panel.values)
print(res.aggregate_breakpoint, res.timing_dispersion["sd_corrected"])
print(res.regime_probabilities, res.interpretation_warning)
```

## Main workflow

1. `simulations/run_grid.py` simulates a configuration grid
   (`simulations/configs/*.yaml`) and extracts diagnostic features.
2. `simulations/run_calibration.py` fits and temperature-calibrates the
   regime classifier on the calibration grid
   (`simulations/calibration/classifier.joblib`).
3. `simulations/run_evaluation.py` evaluates the frozen classifier on the
   locked evaluation grid.
4. `simulations/run_stress.py`, `run_missingness_mitigation.py`, and
   `run_near_sync.py` run the robustness and stress conditions.
5. `simulations/run_kelmarsh.py` and `simulations/run_nasa_battery.py` run the
   two empirical demonstrations.
6. `simulations/make_phase3_figures.py` builds Figures 1-6.

## Reproducing the Technometrics results

| Mode  | Command                          | Content                                                                 | Runtime (1 CPU core) |
|-------|----------------------------------|-------------------------------------------------------------------------|----------------------|
| QUICK | `python replication/run_quick.py` | tests, minimal example, smoke grid, re-evaluation of the locked grid with the shipped classifier, regeneration of Figures 1-4, comparison with reference tables and figures | < 1 min |
| FULL  | `python replication/run_full.py`  | data download and checksum verification, calibration, locked evaluation, stress tests, both empirical analyses, all figures, comparison with reference tables | several hours |

Both scripts exit non-zero if any regenerated result table differs from the
shipped reference copy. `make all` / `make quick` run the same steps.

**Locked evaluation.** `simulations/configs/evaluation_locked.yaml` (seeds
3000-3999) was fixed and committed before the evaluation was run and was not
edited afterwards; its SHA-256 is recorded in `simulations/configs/SHA256SUMS`.
The calibration grid (`calibration.yaml`) and the development grid
(`development.yaml`) use disjoint seed ranges. `METHOD_FREEZE.md` lists the
frozen thresholds and method choices.

**Traceability.** `manuscript_number_trace.csv` maps every number reported in
the manuscript to the result file and column that produces it;
`figure_manifest.csv` and `table_manifest.csv` map figures and tables to their
generating scripts and inputs.

## Data acquisition

No raw data are redistributed in this repository. `python fetch_data.py`
downloads the two public datasets from their original repositories, verifies
each file against the SHA-256 recorded in `data/acquisition_ledger.csv`, and
extracts the files used by the analyses into `data/raw/` (git-ignored). See
`data/README.md` for sources, licenses, and citations.

## Repository structure

```text
src/falsesync/        package source
tests/                unit tests (theory identities, workflow)
examples/             minimal example
simulations/          simulation, calibration, evaluation, empirical scripts
  configs/            development, calibration, locked evaluation grids
  calibration/        calibrated classifier artifact
  results/            simulated grid outputs
replication/          QUICK and FULL replication entry points
proofs/               proofs of propositions P1-P8
outputs/              reference figures
data/                 acquisition ledger and data documentation
*.csv                 reference result tables
math_specification.md, diagnostic_specification.md, METHOD_FREEZE.md,
counterexamples.md    method documentation
```

## Citation

See `CITATION.cff`. Please cite the software and the accompanying manuscript:
T. Onishi, "Aggregation-Induced False Synchrony in Change-Point Analysis"
(manuscript submitted to *Technometrics*).

## License

MIT (see `LICENSE`). Data obtained through `fetch_data.py` remain under their
original licenses (see `data/README.md`).

## Manuscript status

Version 0.1.0 corresponds to the manuscript as submitted to *Technometrics*.
The manuscript has not been peer reviewed or accepted.
