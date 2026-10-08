"""QUICK replication: tests, minimal example, smoke grid, recomputation of the
locked-evaluation summary from the shipped per-cell results, and regeneration
of Figures 1-4. No data download; well under five minutes on a laptop.

Usage: python replication/run_quick.py
"""
import sys
import tempfile
from pathlib import Path

from _common import compare, compare_png, run, snapshot

CHECK = ["evaluation_summary.csv", "regime_confusion_matrix.csv", "probability_calibration.csv"]
FIGS = [f"outputs/phase3/{f}.png" for f in
        ("fig1_concept", "fig2_operators", "fig3_calibration", "fig4_robustness")]

with tempfile.TemporaryDirectory() as tmp:
    snapshot(CHECK + FIGS, Path(tmp))
    run("-m", "pytest", "tests", "-q")
    run("examples/minimal_example.py")
    run("simulations/run_grid.py", "simulations/configs/development.yaml",
        Path(tmp) / "quick_grid.csv", "--quick")
    run("simulations/run_evaluation.py")
    run("simulations/make_phase3_figures.py")
    bad = compare(CHECK, Path(tmp)) + compare_png(FIGS, Path(tmp))
print("QUICK replication:", "PASS" if not bad else f"FAIL ({bad})")
sys.exit(1 if bad else 0)
