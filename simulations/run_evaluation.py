"""Locked evaluation: apply frozen classifier to evaluation_locked grid.

Prereq: simulations/results/evaluation_locked.csv (from run_grid.py) and
simulations/calibration/classifier.joblib. Writes evaluation_results.csv,
regime_confusion_matrix.csv.

Warning rule (spec §6): fires iff agg_strength >= 3 AND P(SYNCHRONOUS) < 0.5.
Critical error: P(SYNC) > 0.5 when truth is not synchronous and break is strong.
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from falsesync.features import FEATURE_NAMES
from falsesync.regimes import (
    REGIME_CLASSES, expected_calibration_error, multiclass_brier,
)

AGG_STRENGTH_CRIT = 3.0
SYNC_PROB_CRIT = 0.5


def main():
    df = pd.read_csv(ROOT / "simulations/results/evaluation_locked.csv")
    df = df[df["label"].notna() & (df["label"] != "ERROR")].copy()
    clf = joblib.load(ROOT / "simulations/calibration/classifier.joblib")
    feat_cols = [f"f_{n}" for n in FEATURE_NAMES]
    X = np.nan_to_num(df[feat_cols].to_numpy(dtype=float), nan=0.0)
    P = clf.predict_proba(X)
    y = df["label"].to_numpy()
    y_idx = np.array([REGIME_CLASSES.index(v) for v in y])
    pred_idx = P.argmax(axis=1)
    pred = np.array(REGIME_CLASSES)[pred_idx]
    p_sync = P[:, REGIME_CLASSES.index("SYNCHRONOUS")]

    strong = df["agg_strength"] >= AGG_STRENGTH_CRIT
    warning = strong & (p_sync < SYNC_PROB_CRIT)
    certifies = (p_sync >= SYNC_PROB_CRIT) & strong
    is_async = ~np.isin(y, ["SYNCHRONOUS", "NEAR_SYNCHRONOUS"])
    is_sync_like = np.isin(y, ["SYNCHRONOUS", "NEAR_SYNCHRONOUS"])

    df["p_sync"] = p_sync
    df["pred"] = pred
    df["warning"] = warning
    df["certifies_sync"] = certifies
    df["correct"] = pred_idx == y_idx

    # confusion matrix
    cm = pd.DataFrame(0, index=REGIME_CLASSES, columns=REGIME_CLASSES)
    for t_i, p_i in zip(y_idx, pred_idx):
        cm.iloc[t_i, p_i] += 1
    cm.to_csv(ROOT / "regime_confusion_matrix.csv")

    acc = float(df["correct"].mean())
    bacc = float(np.mean([df.loc[y_idx == k, "correct"].mean()
                          for k in range(len(REGIME_CLASSES)) if (y_idx == k).any()]))
    brier = multiclass_brier(P, y_idx, len(REGIME_CLASSES))
    ece = expected_calibration_error(P, y_idx)

    # warning metrics (§6)
    w_sens = float(warning[is_async & strong].mean()) if (is_async & strong).any() else np.nan
    w_fpr = float(warning[is_sync_like & strong].mean()) if (is_sync_like & strong).any() else np.nan
    w_none = float(warning[y == "NO_TRANSITION"].mean()) if (y == "NO_TRANSITION").any() else np.nan
    certify_err = float((certifies & is_async).mean())
    certify_err_strong = float((certifies & is_async & strong).mean()) if (is_async & strong).any() else np.nan

    # dispersion estimation error: corrected sd = excess_sd_ratio * window span
    span = df["t_span"].fillna(10.0)
    df["tau_sd_hat"] = df["f_excess_sd_ratio"] * span
    disp_err = float(np.mean(np.abs(df["tau_sd_hat"] - df["tau_sd_true"]) / np.maximum(df["tau_sd_true"], 0.1)))

    summary = {
        "n_cells": int(len(df)),
        "accuracy": acc, "balanced_accuracy": bacc,
        "brier": brier, "ece": ece,
        "macro_f1": float(np.mean([
            2 * ((pred_idx == k) & (y_idx == k)).sum() /
            max(((pred_idx == k).sum() + (y_idx == k).sum()), 1)
            for k in range(len(REGIME_CLASSES))])),
        "warning_sensitivity_async": w_sens,
        "warning_false_rate_sync": w_fpr,
        "warning_rate_no_transition": w_none,
        "false_certify_rate_all": certify_err,
        "false_certify_rate_strong_break": certify_err_strong,
        "dispersion_est_rel_error": disp_err,
        "agg_strength_crit": AGG_STRENGTH_CRIT,
        "sync_prob_crit": SYNC_PROB_CRIT,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    df.to_csv(ROOT / "evaluation_results.csv", index=False)
    pd.DataFrame([summary]).to_csv(ROOT / "evaluation_summary.csv", index=False)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()})


if __name__ == "__main__":
    main()
