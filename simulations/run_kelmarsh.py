"""Kelmarsh SCADA 2016 analysis (spec §12-13). See kelmarsh_analysis_plan.md.

Usage: PYTHONPATH=src python3 simulations/run_kelmarsh.py
Requires simulations/calibration/classifier.joblib.
Writes kelmarsh_results.csv + figures under outputs/phase3/kelmarsh/.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from falsesync.model import FalseSynchronyModel

RAW = ROOT / "data/raw/kelmarsh/extracted_2016"
OUT = ROOT / "outputs/phase3/kelmarsh"
OUT.mkdir(parents=True, exist_ok=True)

FILES = sorted(RAW.glob("Turbine_Data_Kelmarsh_*_2016-01-03_-*.csv"))
COLS = {"time": "# Date and time", "power": "Power (kW)",
        "pot_pc": "Potential power learned PC (kW)",
        "bearing": "Front bearing temperature (°C)"}


def load_turbine(path: Path) -> pd.DataFrame:
    # 9 leading '# ' comment lines; line 10 is the '# Date and time,...' header
    df = pd.read_csv(path, skiprows=9)
    df.columns = [c.strip().lstrip("# ").strip() for c in df.columns]
    df = df.rename(columns={
        "Date and time": "time", "Power (kW)": "power",
        "Potential power learned PC (kW)": "pot_pc",
        "Front bearing temperature (°C)": "bearing"})
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for c in ("power", "pot_pc", "bearing"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["time", "power", "pot_pc", "bearing"]]


def daily_ratio(df: pd.DataFrame) -> pd.Series:
    ok = (df["pot_pc"] > 100) & (df["power"] >= 0) & np.isfinite(df["power"])
    r = (df.loc[ok, "power"] / df.loc[ok, "pot_pc"]).clip(0, 1.15)
    day = df.loc[ok, "time"].dt.floor("D")
    g = pd.DataFrame({"day": day, "r": r}).groupby("day")["r"]
    s = g.mean()
    cnt = g.count()
    return s[cnt >= 30]  # require >=30 valid 10-min records/day


def daily_bearing(df: pd.DataFrame) -> pd.Series:
    ok = np.isfinite(df["bearing"])
    g = pd.DataFrame({"day": df.loc[ok, "time"].dt.floor("D"),
                      "b": df.loc[ok, "bearing"]}).groupby("day")["b"]
    s = g.mean()
    cnt = g.count()
    return s[cnt >= 30]


def build_panel(series: dict, t0, t1) -> tuple:
    days = np.arange(0, (t1 - t0).days + 1)
    vals = np.full((len(series), days.size), np.nan)
    for i, (u, s) in enumerate(series.items()):
        idx = ((s.index - t0).days).to_numpy()
        m = (idx >= 0) & (idx < days.size)
        vals[i, idx[m]] = s.to_numpy()[m]
    return days.astype(float), vals


def run_diagnostic(t, values, weights=None, windows=None, tag=""):
    m = FalseSynchronyModel(classifier=CLF, agg_strength_crit=3.0, sync_prob_crit=0.5)
    res = m.fit(t, values, weights=weights, windows=windows)
    rec = {
        "tag": tag,
        "agg_break_day": res.aggregate_breakpoint,
        "agg_strength": res.aggregate_break_strength,
        "tau_sd": res.timing_dispersion["sd"],
        "sd_corrected": res.timing_dispersion["sd_corrected"],
        "noise_floor": res.event_aligned_summary.get("est_noise_floor"),
        "n_modes": res.cluster_structure["n_components"],
        "synchrony_score": res.synchrony_score,
        "trend_confounded": res.components.get("trend_confounding_warning"),
        "operator_spread": res.components.get("operator_spread"),
        "false_common_event_warning": res.components.get("false_common_event_warning"),
        "warnings": " | ".join(res.interpretation_warning),
    }
    rec.update({f"p_{k}": v for k, v in res.regime_probabilities.items()})
    for i, tau in enumerate(res.unit_breakpoints):
        rec[f"tau_u{i}"] = tau
    return rec, res


def main():
    global CLF
    CLF = joblib.load(ROOT / "simulations/calibration/classifier.joblib")
    turbines = {}
    for f in FILES:
        unit = f.stem.split("_")[3]  # Kelmarsh_N
        turbines[unit] = load_turbine(f)
        print(f"loaded {unit}: {len(turbines[unit])} rows")

    # primary signal: daily performance ratio
    ratio = {u: daily_ratio(df) for u, df in turbines.items()}
    t0 = min(s.index.min() for s in ratio.values())
    t1 = max(s.index.max() for s in ratio.values())
    t, values = build_panel(ratio, t0, t1)
    units = list(ratio.keys())
    print(f"panel {values.shape}, coverage {np.isfinite(values).mean():.2f}")

    rec, res = run_diagnostic(t, values, tag="primary_ratio")
    # windows = per-unit observed day range
    w = np.array([[t[np.isfinite(v)].min(), t[np.isfinite(v)].max()] if np.isfinite(v).any() else [np.nan, np.nan] for v in values])
    rec2, res2 = run_diagnostic(t, values, windows=w, tag="primary_ratio_windows")

    # negative control 1: time-shift unit 1 by +90d (wrapped)
    v_shift = values.copy()
    v_shift[1] = np.roll(v_shift[1], 90)
    rec_c1, _ = run_diagnostic(t, v_shift, tag="ctrl_timeshift_u2+90d")

    # negative control 2: bearing temperature (unrelated)
    bearing = {u: daily_bearing(df) for u, df in turbines.items()}
    tb, vb = build_panel(bearing, t0, t1)
    rec_c2, _ = run_diagnostic(tb, vb, tag="ctrl_bearing_temp")

    # negative control 3: shuffled unit identity
    v_shuf = values.copy()
    v_shuf = v_shuf[np.random.default_rng(0).permutation(v_shuf.shape[0])]
    rec_c3, _ = run_diagnostic(t, v_shuf, tag="ctrl_shuffled_ids")

    df_out = pd.DataFrame([rec, rec2, rec_c1, rec_c2, rec_c3])
    df_out.to_csv(ROOT / "kelmarsh_results.csv", index=False)

    # figure: unit ratio series + aggregate
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    days_dt = t0 + pd.to_timedelta(t, unit="D")
    for i, u in enumerate(units):
        axes[0].plot(days_dt, values[i], lw=0.8, label=u)
    axes[0].set_ylabel("Power / learned power curve (daily)")
    axes[0].legend(ncol=6, fontsize=7, loc="upper right")
    from falsesync.aggregation import weighted_aggregate
    agg = weighted_aggregate(values)
    axes[1].plot(days_dt, agg, color="black", lw=1.5)
    axes[1].axvline(days_dt[int(np.clip(res.aggregate_breakpoint, 0, len(t) - 1))],
                    color="red", ls="--", label=f"Aggregate break: day {res.aggregate_breakpoint:.0f}")
    for i, u in enumerate(units):
        tau = res.unit_breakpoints[i]
        if np.isfinite(tau):
            axes[1].axvline(days_dt[int(np.clip(tau, 0, len(t) - 1))], color="0.5", ls=":", lw=0.8)
    axes[1].set_ylabel("Aggregate")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "kelmarsh_ratio_aggregate.png", dpi=600)
    fig.savefig(OUT / "kelmarsh_ratio_aggregate.pdf")

    # timing distribution figure
    grid, fdec = res.timing_distribution["grid"], res.timing_distribution["deconvolved"]
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(grid, res.timing_distribution["density"], label="KDE of estimated break times")
    ax2.plot(grid, fdec, label="Deconvolved", ls="--")
    for i, u in enumerate(units):
        if np.isfinite(res.unit_breakpoints[i]):
            ax2.axvline(res.unit_breakpoints[i], color="0.6", lw=0.6)
    ax2.set_xlabel("Day of 2016")
    ax2.set_ylabel("Density")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(OUT / "kelmarsh_timing_distribution.png", dpi=600)
    fig2.savefig(OUT / "kelmarsh_timing_distribution.pdf")

    print(df_out[["tag", "agg_break_day", "agg_strength", "synchrony_score",
                  "false_common_event_warning"]].to_string())


if __name__ == "__main__":
    main()
