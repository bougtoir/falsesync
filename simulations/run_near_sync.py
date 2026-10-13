"""Phase 4 §6: NEAR_SYNC boundary sensitivity — where is synchrony
statistically distinguishable from near-synchrony?

Grid: tau dispersion sigma (near-sync) x SNR x N x T x transition width.
Record: true tau_sd, measured sd/floor, P(SYNC), P(NEAR), separation.

Usage: PYTHONPATH=src python3 simulations/run_near_sync.py [--quick]
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from falsesync.features import extract_features
from falsesync.simengine import SimConfig, simulate

SEED0 = 9500


def run_cell(sigma, noise_sd, n, T, width, rep, seed):
    cfg = SimConfig(n_units=n, n_t=T, regime="near_synchronous",
                    f_family="narrow_normal", f_params={"sigma": sigma},
                    shape="step", amplitude=1.0, width=width,
                    noise_kind="iid", noise_sd=noise_sd, seed=seed, cell_id="x")
    panel, truth = simulate(cfg)
    feats, aux = extract_features(
        panel.t, panel.values, aggregate=truth["aggregate"],
        rng=np.random.default_rng(seed), unit_boot=10)
    P = CLF.predict_proba(np.nan_to_num(feats, nan=0.0))[0]
    p_sync = float(P[0]); p_near = float(P[1])
    return {
        "sigma": sigma, "noise_sd": noise_sd, "n_units": n, "n_t": T,
        "width": width, "rep": rep, "seed": seed,
        "tau_sd_true": truth["tau_sd"],
        "tau_sd_hat": float(np.std(aux["taus"])),
        "sd_corrected": float(np.sqrt(max(np.std(aux["taus"])**2 - (aux["noise_floor"] or 0)**2, 0))),
        "noise_floor": aux["noise_floor"],
        "floor_ratio": aux["noise_floor"] / max(np.std(aux["taus"]), 1e-9),
        "p_sync": p_sync, "p_near": p_near,
        "sep": p_sync - p_near,
        "snr": 1.0 / max(noise_sd, 1e-9),
    }


def main(quick=False):
    global CLF
    CLF = joblib.load(ROOT / "simulations/calibration/classifier.joblib")
    n_rep = 4 if quick else 10
    rows = []
    k = 0
    for sigma in (0.02, 0.05, 0.10, 0.20):
        for noise_sd in (0.10, 0.30):
            for (n, T) in ((40, 120), (100, 240)):
                for width in (0.10, 0.30):
                    for rep in range(n_rep):
                        try:
                            rows.append(run_cell(sigma, noise_sd, n, T, width,
                                                 rep, SEED0 + k))
                        except Exception as e:
                            rows.append({"sigma": sigma, "noise_sd": noise_sd,
                                         "n_units": n, "n_t": T, "width": width,
                                         "rep": rep, "error": repr(e)})
                        k += 1
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "near_sync_sensitivity.csv", index=False)
    ok = df.get("error").isna() if "error" in df else np.ones(len(df), bool)
    df = df[ok]
    g = df.groupby(["sigma", "noise_sd", "n_units", "n_t", "width"])[
        ["tau_sd_true", "sd_corrected", "floor_ratio", "p_sync", "p_near", "sep"]].mean().round(3)
    print(g.to_string())
    print("\nsummary by sigma:")
    print(df.groupby("sigma")[["tau_sd_true", "sd_corrected", "p_sync", "p_near", "sep"]].mean().round(3))


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
