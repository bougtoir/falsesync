"""Fit the regime classifier on the calibration grid + temperature + thresholds.

Usage: PYTHONPATH=src python3 simulations/run_calibration.py [--quick]
Reads simulations/results/calibration.csv; writes:
  simulations/calibration/classifier.joblib
  calibration_results.csv (per-class calibration-grid performance)
  probability_calibration.csv (reliability bins)
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
    REGIME_CLASSES, RegimeClassifier, expected_calibration_error,
    multiclass_brier,
)

RES = ROOT / "simulations" / "results"
CAL = ROOT / "simulations" / "calibration"
CAL.mkdir(parents=True, exist_ok=True)


def main(quick=False):
    df = pd.read_csv(RES / "calibration.csv")
    df = df[df["label"].notna() & (df["label"] != "ERROR")].copy()
    feat_cols = [f"f_{n}" for n in FEATURE_NAMES]
    X = df[feat_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["label"].to_numpy()
    # hold out 20% for temperature calibration
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(df))
    n_cal = int(0.8 * len(df))
    tr, cal = idx[:n_cal], idx[n_cal:]
    t0 = time.time()
    clf = RegimeClassifier().fit(X[tr], y[tr])
    temp = clf.calibrate(X[cal], y[cal])
    clf.meta = {
        "trained_on": "simulations/results/calibration.csv",
        "n_train": int(len(tr)), "n_temp": int(len(cal)),
        "temperature": temp, "feature_names": FEATURE_NAMES,
        "classes": REGIME_CLASSES,
    }
    joblib.dump(clf, CAL / "classifier.joblib")

    P = clf.predict_proba(X[cal])
    y_idx = np.array([REGIME_CLASSES.index(v) for v in y[cal]])
    pred = P.argmax(axis=1)
    acc = float((pred == y_idx).mean())
    bacc = float(np.mean([(pred[y_idx == k] == y_idx[y_idx == k]).mean()
                          for k in range(len(REGIME_CLASSES)) if (y_idx == k).any()]))
    brier = multiclass_brier(P, y_idx, len(REGIME_CLASSES))
    ece = expected_calibration_error(P, y_idx)

    rows = []
    for k, cls in enumerate(REGIME_CLASSES):
        m = y_idx == k
        if not m.any():
            continue
        rows.append({
            "class": cls, "n": int(m.sum()),
            "recall": float((pred[m] == k).mean()),
            "precision": float((y_idx[pred == k] == k).mean()) if (pred == k).any() else np.nan,
            "mean_pred_prob": float(P[m, k].mean()),
        })
    rows.append({"class": "_ALL", "n": int(len(y_idx)), "recall": acc,
                 "precision": acc, "mean_pred_prob": np.nan})
    out = pd.DataFrame(rows)
    out["balanced_accuracy"] = bacc
    out["brier"] = brier
    out["ece"] = ece
    out["temperature"] = temp
    out.to_csv(ROOT / "calibration_results.csv", index=False)

    # reliability bins
    conf = P.max(axis=1)
    correct = (pred == y_idx).astype(int)
    bins = []
    for b in range(10):
        m = (conf > b / 10) & (conf <= (b + 1) / 10)
        if m.any():
            bins.append({"bin_lo": b / 10, "bin_hi": (b + 1) / 10,
                         "n": int(m.sum()), "mean_conf": float(conf[m].mean()),
                         "empirical_acc": float(correct[m].mean())})
    pd.DataFrame(bins).to_csv(ROOT / "probability_calibration.csv", index=False)
    print(f"acc={acc:.3f} bacc={bacc:.3f} brier={brier:.3f} ece={ece:.3f} T={temp:.2f} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
