"""Phase 4 §3-5: ONE principled informative-missingness mitigation attempt.

Framework (option A+D): explicit observation-process diagnostic.
  statistic: |corr(window_start_i, tau_hat_i)| over units with
             informative-sensitive windows, plus fraction of units with
             truncated windows. Under MCAR/complete, windows are
             independent of tau_i; under tau-correlated entry, corr ~ 1.
  action: when informative observation is detected, the model emits
          INDETERMINATE_OBSERVATION_PROCESS — it does NOT certify
          synchrony (P(sync) still reported for information) and raises
          the false common-event warning regardless of classifier prob.

Evaluation subset (§4): regimes {sync, diffuse, two_cluster,
no_transition} x missingness {complete, mcar, dropout, late_entry,
informative (lead 1.0 / 0.4 strong, kappa varying)} x reps.
Compare: A=Phase-3 original, B=mitigation-enhanced.

Usage: PYTHONPATH=src python3 simulations/run_missingness_mitigation.py [--quick]
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from falsesync.model import FalseSynchronyModel
from falsesync.simengine import SimConfig, simulate

SEED0 = 9000


def obs_process_stat(taus, windows, t):
    """corr(window_start, tau_hat) over truncated units + frac truncated."""
    win = np.asarray(windows)
    if win.ndim != 2 or win.shape[1] != 2:
        return np.nan, 0.0
    lo, hi = np.nanmin(t), np.nanmax(t)
    trunc_start = win[:, 0] > lo + 0.02 * np.ptp(t)
    frac_trunc = float(trunc_start.mean())
    m = trunc_start & np.isfinite(taus)
    if m.sum() < 5:
        return np.nan, frac_trunc
    c = np.corrcoef(win[m, 0], taus[m])[0, 1]
    return float(c), frac_trunc


def detect_informative(taus, windows, t, corr_crit=0.5, frac_crit=0.2):
    c, frac = obs_process_stat(taus, windows, t)
    return bool(np.isfinite(c) and c >= corr_crit and frac >= frac_crit), c, frac


def run_cell(d, seed):
    cfg = SimConfig(n_units=d["n_units"], n_t=d["n_t"], regime=d["regime"],
                    f_family=d["f_family"], f_params=d.get("f_params", {}),
                    shape="step", amplitude=1.0, width=0.25,
                    noise_kind="iid", noise_sd=0.15,
                    miss_kind=d["miss_kind"], miss_params=d.get("miss_params", {}),
                    seed=seed, cell_id="x")
    panel, truth = simulate(cfg)
    m = FalseSynchronyModel(classifier=CLF, unit_boot=10,
                            agg_strength_crit=3.0, sync_prob_crit=0.5)
    # Phase-3 original: pass windows (as Phase 3 did)
    resA = m.fit(panel.t, panel.values, windows=panel.windows)
    # mitigation: observation-process diagnostic
    infor, corr, frac = detect_informative(resA.unit_breakpoints,
                                         panel.windows, panel.t)
    p_sync = resA.regime_probabilities.get("SYNCHRONOUS", np.nan)
    strong = resA.aggregate_break_strength >= 3.0
    certify_A = strong and np.isfinite(p_sync) and p_sync >= 0.5
    # B: indeterminate -> never certifies, always warns when strong
    certify_B = False if infor else certify_A
    warning_B = infor or resA.components.get("false_common_event_warning")
    return {
        **d, "seed": seed, "tau_sd_true": truth["tau_sd"],
        "agg_strength": resA.aggregate_break_strength, "p_sync": p_sync,
        "obs_corr": corr, "obs_frac_trunc": frac,
        "informative_detected": infor,
        "certify_A": bool(certify_A), "certify_B": bool(certify_B),
        "warning_A": bool(resA.components.get("false_common_event_warning")),
        "warning_B": bool(warning_B),
    }


def main(quick=False):
    global CLF
    CLF = joblib.load(ROOT / "simulations/calibration/classifier.joblib")
    n_rep = 4 if quick else 10
    cells = []
    for regime, ff, fp in [
        ("synchronous", "degenerate", {}),
        ("diffuse", "broad_normal", {"sigma": 0.35}),
        ("two_cluster", "two_normal_mix", {"delta": 4.5, "sigma": 0.6}),
        ("no_transition", "degenerate", {}),
    ]:
        for miss_kind, mp in [
            ("complete", {}),
            ("mcar", {"rate": 0.2}),
            ("dropout", {"rate": 0.3}),
            ("late_entry", {"rate": 0.3}),
            ("informative", {"lead": 1.0}),
            ("informative", {"lead": 0.4}),  # stronger tau-correlation
        ]:
            for rep in range(n_rep):
                cells.append({"regime": regime, "f_family": ff, "f_params": fp,
                              "miss_kind": miss_kind,
                              "miss_params": mp, "miss_label": f"{miss_kind}{mp.get('lead','')}",
                              "n_units": 40, "n_t": 120, "rep": rep})
    rows = []
    for i, d in enumerate(cells):
        try:
            rows.append(run_cell(d, SEED0 + i))
        except Exception as e:
            rows.append({**d, "error": repr(e)})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "missingness_mitigation_results.csv", index=False)

    ok = df["error"].isna() if "error" in df else np.ones(len(df), bool)
    df = df[ok].copy()
    is_async = ~df["regime"].isin(["synchronous", "near_synchronous"])
    strong = df["agg_strength"] >= 3
    print("=== per missingness x regime ===")
    print(df.groupby(["miss_label", "regime"])[
        ["p_sync", "certify_A", "certify_B", "warning_B", "informative_detected"]]
        .mean().round(3).to_string())
    fc_A = float((df["certify_A"] & is_async & strong).sum())
    fc_B = float((df["certify_B"] & is_async & strong).sum())
    n_strong_async = int((is_async & strong).sum())
    sync = df["regime"] == "synchronous"
    print(f"\nfalse-certify A: {fc_A}/{n_strong_async}  B: {fc_B}/{n_strong_async}")
    print(f"sync sensitivity loss B: {float((sync & strong & ~df['certify_B'] & df['certify_A']).mean()):.3f}")
    print(f"detected rate by miss: {df.groupby('miss_label')['informative_detected'].mean().round(2).to_dict()}")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
