"""FULL replication: download and verify source data, rerun calibration,
the locked evaluation grid, stress tests, and both empirical analyses, rebuild
all figures, and compare every regenerated result table with the shipped
reference values. Several hours on one CPU core; ~2.5 GB disk for data.

Usage: python replication/run_full.py
"""
import sys
import tempfile
from pathlib import Path

from _common import ROOT, compare, compare_png, run, snapshot

CHECK = ["calibration_results.csv", "evaluation_summary.csv", "regime_confusion_matrix.csv",
         "probability_calibration.csv",
         "operator_sensitivity.csv", "error_floor_results.csv", "trend_heterogeneity_results.csv",
         "weighting_truncation_results.csv", "missingness_mitigation_results.csv",
         "near_sync_sensitivity.csv", "kelmarsh_results.csv", "nasa_battery_results.csv"]
FIGS = [f"outputs/phase3/fig{i}.png" for i in
        ("1_concept", "2_operators", "3_calibration", "4_robustness", "5_kelmarsh", "6_nasa")]
CFG = "simulations/configs"

with tempfile.TemporaryDirectory() as tmp:
    snapshot(CHECK + FIGS, Path(tmp))
    run("fetch_data.py")
    run("-m", "pytest", "tests", "-q")
    run("simulations/run_grid.py", f"{CFG}/calibration.yaml", "simulations/results/calibration.csv")
    run("simulations/run_calibration.py")
    run("simulations/run_grid.py", f"{CFG}/evaluation_locked.yaml",
        "simulations/results/evaluation_locked.csv")
    run("simulations/run_evaluation.py")
    run("simulations/run_stress.py")
    run("simulations/run_missingness_mitigation.py")
    run("simulations/run_near_sync.py")
    run("simulations/run_kelmarsh.py")
    run("simulations/run_nasa_battery.py")
    run("simulations/make_phase3_figures.py")
    bad = compare(CHECK, Path(tmp)) + compare_png(FIGS, Path(tmp))
print("FULL replication:", "PASS" if not bad else f"FAIL ({bad})")
print(f"figures: {ROOT / 'outputs' / 'phase3'}")
sys.exit(1 if bad else 0)
