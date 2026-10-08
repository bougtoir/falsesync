# falsesync reproduction pipeline
#   make all          -> install + data + all simulations, empirical analyses, figures/CSVs
#   make quick        -> QUICK replication (replication/run_quick.py)
PY := python3
PIP := pip
ENV := PYTHONPATH=src

.PHONY: all install data tests sims empirical figures manuscript quick clean

all: install data tests sims empirical figures

install:
	$(PIP) install -e .

data:
	$(PY) fetch_data.py

tests:
	$(ENV) $(PY) -m pytest tests/ -x -q

sims:
	$(ENV) $(PY) simulations/run_grid.py simulations/configs/calibration.yaml simulations/results/calibration.csv
	$(ENV) $(PY) simulations/run_calibration.py
	$(ENV) $(PY) simulations/run_grid.py simulations/configs/evaluation_locked.yaml simulations/results/evaluation_locked.csv
	$(ENV) $(PY) simulations/run_evaluation.py
	$(ENV) $(PY) simulations/run_stress.py
	$(ENV) $(PY) simulations/run_missingness_mitigation.py
	$(ENV) $(PY) simulations/run_near_sync.py

empirical:
	$(ENV) $(PY) simulations/run_kelmarsh.py
	$(ENV) $(PY) simulations/run_nasa_battery.py

figures:
	$(ENV) $(PY) simulations/make_phase3_figures.py

manuscript:
	$(MAKE) -f manuscript/manuscript.mk PY=$(PY)

quick: install
	$(PY) replication/run_quick.py

clean:
	rm -rf falsesync.egg-info src/falsesync.egg-info build dist .pytest_cache
