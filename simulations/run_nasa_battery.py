"""NASA PCoE battery analysis (spec §14). See nasa_battery_analysis_plan.md.

Usage: PYTHONPATH=src python3 simulations/run_nasa_battery.py
Writes nasa_battery_results.csv + figures under outputs/phase3/nasa/.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from falsesync.model import FalseSynchronyModel

MAT = ROOT / "data/raw/nasa_battery/mat_08q4"
OUT = ROOT / "outputs/phase3/nasa"
OUT.mkdir(parents=True, exist_ok=True)
CELLS = ["B0005", "B0006", "B0007", "B0018"]


def discharge_capacity(path: Path):
    """(cycle_idx, capacity) for discharge cycles."""
    m = loadmat(path, simplify_cells=True)
    key = [k for k in m if not k.startswith("_")][0]
    cyc = m[key]["cycle"]
    idxs, caps, times = [], [], []
    for j, c in enumerate(cyc):
        if c["type"] != "discharge":
            continue
        cap = c["data"].get("Capacity")
        if cap is None or not np.isfinite(float(np.ravel(cap)[0])):
            continue
        idxs.append(j)
        caps.append(float(np.ravel(cap)[0]))
        times.append(float(np.ravel(c["time"])[0]) if np.ndim(c["time"]) == 0 else np.ravel(c["time"])[0])
    return np.array(idxs), np.array(caps), np.array(times)


def main():
    clf = joblib.load(ROOT / "simulations/calibration/classifier.joblib")
    series = {}
    for cell in CELLS:
        idx, cap, tm = discharge_capacity(MAT / f"{cell}.mat")
        series[cell] = (idx, cap, tm)
        print(f"{cell}: {len(idx)} discharge cycles, cap {cap[0]:.3f}->{cap[-1]:.3f}")

    # cycle-space panel: capacity normalized by each cell's first-cycle cap
    t_max = min(idx.max() for idx, _, _ in series.values())
    t = np.arange(1, t_max + 1, dtype=float)
    values = np.full((len(CELLS), t.size), np.nan)
    times_cal = np.full((len(CELLS), t.size), np.nan)
    for i, cell in enumerate(CELLS):
        idx, cap, tm = series[cell]
        m = idx <= t_max
        values[i, idx[m] - 1] = cap[m] / cap[0]
        times_cal[i, idx[m] - 1] = tm[m]

    model = FalseSynchronyModel(classifier=clf, agg_strength_crit=3.0, sync_prob_crit=0.5)
    res = model.fit(t, values)
    rec = {
        "tag": "nasa_battery_cycle_space",
        "agg_break_cycle": res.aggregate_breakpoint,
        "agg_strength": res.aggregate_break_strength,
        "tau_sd": res.timing_dispersion["sd"],
        "sd_corrected": res.timing_dispersion["sd_corrected"],
        "n_modes": res.cluster_structure["n_components"],
        "synchrony_score": res.synchrony_score,
        "trend_confounded": res.components.get("trend_confounding_warning"),
        "operator_spread": res.components.get("operator_spread"),
        "false_common_event_warning": res.components.get("false_common_event_warning"),
        "warnings": " | ".join(res.interpretation_warning),
    }
    rec.update({f"p_{k}": v for k, v in res.regime_probabilities.items()})
    for i, cell in enumerate(CELLS):
        rec[f"tau_{cell}"] = res.unit_breakpoints[i]
    pd.DataFrame([rec]).to_csv(ROOT / "nasa_battery_results.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from falsesync.aggregation import weighted_aggregate
    agg = weighted_aggregate(values)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for i, cell in enumerate(CELLS):
        idx, cap, tm = series[cell]
        axes[0].plot(idx, cap / cap[0], lw=0.8, ms=2.0, marker=".",
                     label=cell)
    axes[0].set_ylabel("Capacity / initial capacity")
    axes[0].legend()
    finite = np.isfinite(agg)
    axes[1].plot(t[finite], agg[finite], color="black", lw=0.8,
                 ms=2.0, marker=".", ls="", label="Aggregate")
    axes[1].axvline(res.aggregate_breakpoint, color="red", ls="--",
                    label=f"Aggregate break: cycle {res.aggregate_breakpoint:.0f}")
    for i in range(len(CELLS)):
        if np.isfinite(res.unit_breakpoints[i]):
            axes[1].axvline(res.unit_breakpoints[i], color="0.5", ls=":", lw=0.8)
    axes[1].set_xlabel("Cycle index")
    axes[1].set_ylabel("Aggregate")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "nasa_capacity_aggregate.png", dpi=600)
    fig.savefig(OUT / "nasa_capacity_aggregate.pdf")
    print(pd.DataFrame([rec]).to_string())


if __name__ == "__main__":
    main()
