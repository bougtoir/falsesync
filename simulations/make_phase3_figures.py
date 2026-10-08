"""Phase-3 manuscript figures + manifests (spec §21-22).

Usage: PYTHONPATH=src python3 simulations/make_phase3_figures.py
Writes outputs/phase3/fig{1..6}.png, figure_manifest.csv, table_manifest.csv,
reproducibility_manifest.csv.
"""

import hashlib
import shutil
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs/phase3"
OUT.mkdir(parents=True, exist_ok=True)

from falsesync.simengine import SimConfig, simulate
from falsesync.aggregation import weighted_aggregate
from falsesync.changepoints import ls_breakpoint, aggregate_breakpoint, multibreak

GRAY = ["0.15", "0.45", "0.7"]
DPI = 600
REGIME_LABEL = {"synchronous": "Synchronous", "diffuse": "Diffuse asynchronous",
                "two_cluster": "Two clusters"}
CLASS_LABEL = {"SYNCHRONOUS": "Synchronous", "NEAR_SYNCHRONOUS": "Near-synchronous",
               "DIFFUSE_ASYNCHRONOUS": "Diffuse asynchronous",
               "CLUSTERED": "Clustered", "NO_TRANSITION": "No transition"}
MISS_LABEL = {"complete": "Complete", "dropout": "Dropout",
              "informative": "Informative entry/dropout", "late_entry": "Late entry",
              "mcar": "MCAR"}


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=DPI)
    fig.savefig(OUT / f"{stem}.pdf")


def fig1():
    """Conceptual schematic: heterogeneous unit transitions -> aggregate."""
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    rng = np.random.default_rng(1)
    cases = [("synchronous", "degenerate", {}),
             ("diffuse", "broad_normal", {"sigma": 0.4}),
             ("two_cluster", "two_normal_mix", {"delta": 4.0, "sigma": 0.4})]
    for c, (regime, ff, fp) in enumerate(cases):
        cfg = SimConfig(n_units=12, n_t=200, regime=regime, f_family=ff,
                        f_params=fp, shape="step", amplitude=1.0, width=0.08,
                        noise_kind="iid", noise_sd=0.06, seed=c)
        p, truth = simulate(cfg)
        for i in range(12):
            ax[0, c].plot(p.t, p.values[i], color="0.5", lw=0.7)
            if np.isfinite(truth["taus"][i]):
                ax[0, c].axvline(truth["taus"][i], color="0.7", lw=0.4, ls=":")
        ax[0, c].set_title(f"{REGIME_LABEL[regime]}: unit curves")
        ax[1, c].hist(truth["taus"], bins=15, density=True, color="0.5")
        ax2 = ax[1, c].twinx()
        ax2.plot(p.t, truth["aggregate"], color=GRAY[0], lw=1.5)
        r = ls_breakpoint(p.t, np.nan_to_num(truth["aggregate"]))
        ax2.axvline(r.location, color="red", ls="--")
        ax2.set_ylabel("Aggregate", color=GRAY[0])
        ax[1, c].set_xlabel(r"Time $t$ (bars: unit transition times $\tau_i$)")
    fig.tight_layout()
    save(fig, "fig1_concept")


def fig2():
    """Theory: operator-dependent T_M on the same curve."""
    cfg = SimConfig(n_units=40, n_t=200, regime="diffuse", f_family="broad_normal",
                    f_params={"sigma": 0.5}, shape="logistic", amplitude=1.0,
                    width=0.3, noise_kind="iid", noise_sd=0.05, seed=2)
    p, truth = simulate(cfg)
    agg = np.nan_to_num(truth["aggregate"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(p.t, agg, color=GRAY[0], lw=1.5)
    ax[0].hist(truth["taus"], bins=20, density=True, color="0.6",
               alpha=0.5, label=r"$F_\tau$")
    ax[0].set_title(r"Aggregate curve and true $F_\tau$")
    for m in ("ls", "maxslope", "cusum"):
        r = aggregate_breakpoint(p.t, agg, m)
        name = {"ls": "Least squares", "maxslope": "Max slope", "cusum": "CUSUM"}[m]
        ax[1].axvline(r.location, label=f"{name} = {r.location:.2f}", lw=1.2)
    rb = multibreak(p.t, agg, 3)
    ax[1].axvline(rb.location, label=f"Binary segmentation = {rb.location:.2f}", lw=1.2, ls=":")
    ax[1].plot(p.t, agg, color="0.6", lw=0.8)
    ax[1].set_title(r"Operator-dependent breakpoint $T_M$")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig2_operators")


def fig3():
    """Calibration: confusion matrix + reliability."""
    cm = pd.read_csv(ROOT / "regime_confusion_matrix.csv", index_col=0)
    cal = pd.read_csv(ROOT / "probability_calibration.csv")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    im = ax[0].imshow(cm.values, cmap="gray_r")
    ax[0].set_xticks(range(5), [CLASS_LABEL[c] for c in cm.columns], rotation=45,
                     ha="right", fontsize=7)
    ax[0].set_yticks(range(5), [CLASS_LABEL[c] for c in cm.index], fontsize=7)
    ax[0].set_xlabel("Predicted regime")
    ax[0].set_ylabel("True regime")
    for i in range(5):
        for j in range(5):
            ax[0].text(j, i, int(cm.values[i, j]), ha="center", va="center",
                       fontsize=8)
    ax[0].set_title("Locked-evaluation confusion matrix")
    ax[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[1].plot(cal["mean_conf"], cal["empirical_acc"], "o-", color=GRAY[0])
    ax[1].set_xlabel("Mean predicted confidence")
    ax[1].set_ylabel("Empirical accuracy")
    ax[1].set_title("Probability calibration (locked evaluation)")
    fig.tight_layout()
    save(fig, "fig3_calibration")


def fig4():
    """Robustness: warning failure under informative missingness +
    error-floor correction."""
    wt = pd.read_csv(ROOT / "weighting_truncation_results.csv")
    ef = pd.read_csv(ROOT / "error_floor_results.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    piv = wt.pivot_table(index=["regime"], columns="miss_kind",
                         values="p_sync", aggfunc="mean")
    piv = piv.rename(index=REGIME_LABEL, columns=MISS_LABEL)
    piv.columns.name = "Missingness mechanism"
    piv.plot.bar(ax=ax[0], colormap="gray", fontsize=7, rot=0, edgecolor="0.2", linewidth=0.5)
    ax[0].set_xlabel("True regime")
    ax[0].set_ylabel("Mean P(Synchronous)")
    ax[0].set_title("Informative missingness inflates P(Synchronous)")
    ax[0].legend(title="Missingness mechanism", fontsize=7, title_fontsize=7)
    for c, lbl in [("sd_naive", "Naive SD of estimated break times"),
                   ("sd_corrected", "Floor-corrected"),
                   ("sd_deconv", "Deconvolution")]:
        g = ef.groupby("noise_sd")[c].mean()
        ax[1].plot(g.index, g.values, "o-", label=lbl, lw=1.0)
    ax[1].axhline(0, color="k", lw=0.5, ls=":")
    ax[1].set_xlabel("Noise SD (true timing SD = 0)")
    ax[1].set_ylabel("Estimated timing dispersion")
    ax[1].set_title("Error floor: naive SD inflates at low SNR")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig4_robustness")


def fig5():
    """Kelmarsh figure is drawn by run_kelmarsh.py; copy it to its manuscript name."""
    for ext in ("png", "pdf"):
        shutil.copyfile(OUT / f"kelmarsh/kelmarsh_ratio_aggregate.{ext}", OUT / f"fig5_kelmarsh.{ext}")


def fig6():
    """NASA contrast figure is drawn by run_nasa_battery.py."""
    for ext in ("png", "pdf"):
        shutil.copyfile(OUT / f"nasa/nasa_capacity_aggregate.{ext}", OUT / f"fig6_nasa.{ext}")


def manifests():
    figs = [
        ("fig1_concept", "Heterogeneous unit transitions -> apparent aggregate break (3 regimes)", "sim"),
        ("fig2_operators", "Same aggregate curve, operator-dependent breakpoint T_M (P7)", "sim"),
        ("fig3_calibration", "Locked-eval regime confusion + probability calibration", "eval"),
        ("fig4_robustness", "Informative missingness inflates P(Synchronous); naive SD overestimates dispersion", "stress"),
        ("fig5_kelmarsh", "Kelmarsh SCADA daily performance ratio: unit breaks + aggregate + tau density", "kelmarsh"),
        ("fig6_nasa", "NASA battery capacity fade (cycle space) as aligned contrast", "nasa"),
    ]
    scripts = {"sim": "simulations/make_phase3_figures.py", "eval": "simulations/make_phase3_figures.py",
               "stress": "simulations/make_phase3_figures.py", "kelmarsh": "simulations/run_kelmarsh.py",
               "nasa": "simulations/run_nasa_battery.py"}
    rows = []
    for k, (stem, cap, src) in enumerate(figs, 1):
        png, pdf = OUT / f"{stem}.png", OUT / f"{stem}.pdf"
        rows.append({"figure": stem, "caption": cap, "source": src, "script": scripts[src],
                     "manuscript_citation": f"Fig {k}",
                     "png_file": str(png.relative_to(ROOT)), "png_dpi": DPI,
                     "png_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
                     "pdf_file": str(pdf.relative_to(ROOT)),
                     "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()})
    pd.DataFrame(rows).to_csv(ROOT / "figure_manifest.csv", index=False)
    tabs = [
        ("T1_estimands", "Estimands: T_M(F_tau,g,w,window,noise) + regime definitions", "math_specification.md", "Table 1"),
        ("T2_sim_design", "Simulation grid: regimes x families x shapes x stress blocks", "simulation_design.md", "S2"),
        ("T3_locked_eval", "Locked-eval performance: acc/bacc/brier/ECE/warning metrics", "evaluation_summary.csv", "Table 2"),
        ("T4_realdata", "Kelmarsh + NASA diagnostic summaries (diagnostic language only)", "kelmarsh_results.csv/nasa_battery_results.csv", "Table 3"),
        ("TS_operators", "Operator sensitivity per regime", "operator_sensitivity.csv", "S4"),
        ("TS_trend", "Trend confounding raw vs detrended", "trend_heterogeneity_results.csv", "S5"),
        ("TS_error_floor", "Naive vs corrected vs deconvolved dispersion", "error_floor_results.csv", "S6"),
        ("TS_weight_trunc", "Weighting/truncation sensitivity + informative-missingness failure", "weighting_truncation_results.csv", "S7"),
    ]
    pd.DataFrame(tabs, columns=["table", "content", "source_file", "manuscript_citation"]).to_csv(
        ROOT / "table_manifest.csv", index=False)


def main():
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    manifests()
    import platform, scipy, sklearn
    pd.DataFrame([{
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "calibration_cells": 1071, "eval_cells": 581,
        "seeds": "dev=100, calibration=1000-1999, eval_locked=3000-3999, stress=7000+",
        "mode": "FULL (laptop, single-thread, no GPU)",
    }]).to_csv(ROOT / "reproducibility_manifest.csv", index=False)
    print("figures + manifests written")


if __name__ == "__main__":
    main()
