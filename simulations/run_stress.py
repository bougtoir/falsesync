"""Phase-3 stress tests (spec §7 trend, §8 error floor, §10 operators,
§11 weighting/truncation). Produces:
  trend_heterogeneity_results.csv
  error_floor_results.csv
  operator_sensitivity.csv
  weighting_truncation_results.csv

Usage: PYTHONPATH=src python3 simulations/run_stress.py [--quick]
Requires simulations/calibration/classifier.joblib.
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from falsesync.model import FalseSynchronyModel
from falsesync.simengine import SimConfig, simulate

CLF = None
SEED0 = 7000


def model(**kw):
    kw.setdefault("agg_strength_crit", 3.0)
    kw.setdefault("sync_prob_crit", 0.5)
    kw.setdefault("unit_boot", 10)
    return FalseSynchronyModel(classifier=CLF, **kw)


def run(cfg: SimConfig):
    panel, truth = simulate(cfg)
    res = model().fit(panel.t, panel.values, weights=panel.weights,
                      windows=panel.windows, aggregate=truth["aggregate"])
    return panel, truth, res


def base_row(res, truth, cfg, tag):
    p = res.regime_probabilities
    return {
        "tag": tag, "cell": cfg.cell_id,
        "regime": cfg.regime, "tau_sd_true": truth["tau_sd"],
        "agg_break": res.aggregate_breakpoint,
        "agg_strength": res.aggregate_break_strength,
        "p_sync": p.get("SYNCHRONOUS", np.nan),
        "warning": res.components.get("false_common_event_warning"),
        "trend_confounded": res.components.get("trend_confounding_warning"),
        "tau_sd_hat": res.timing_dispersion["sd"],
        "sd_corrected": res.timing_dispersion["sd_corrected"],
        "operator_spread": res.components.get("operator_spread"),
        "operator_consensus": res.components.get("operator_consensus"),
    }


def trend_block(n_rep=12, seed=SEED0):
    """§7: synchronous & near-sync panels with trend heterogeneity.
    Compare raw / detrended preprocessing."""
    rows = []
    k = 0
    for regime in ("synchronous", "near_synchronous"):
        f_family = "degenerate" if regime == "synchronous" else "narrow_normal"
        for trend_kind, trend_scale in [
            ("linear", 0.6), ("nonlinear", 0.5),
            ("local_drift", 0.5), ("mixed", 0.7),
        ]:
            for rep in range(n_rep):
                cfg = SimConfig(n_units=30, n_t=120, regime=regime,
                                f_family=f_family, shape="step", amplitude=1.0,
                                width=0.25, noise_kind="iid", noise_sd=0.15,
                                trend_kind=trend_kind, trend_scale=trend_scale,
                                seed=seed + k, cell_id=f"trend-{k}")
                k += 1
                panel, truth, res = run(cfg)
                r = base_row(res, truth, cfg, "raw")
                r.update({"trend_kind": trend_kind, "trend_scale": trend_scale})
                rows.append(r)
                # detrended preprocessing: remove per-unit linear trend
                v = panel.values.copy()
                for i in range(v.shape[0]):
                    m = np.isfinite(v[i])
                    if m.sum() > 5:
                        c = np.polyfit(panel.t[m], v[i, m], 1)
                        v[i, m] = v[i, m] - np.polyval(c, panel.t[m])
                res_d = model().fit(panel.t, v, weights=panel.weights,
                                    windows=panel.windows)
                rd = base_row(res_d, truth, cfg, "detrended")
                rd.update({"trend_kind": trend_kind, "trend_scale": trend_scale})
                rows.append(rd)
    return pd.DataFrame(rows)


def error_floor_block(n_rep=16, seed=SEED0 + 500):
    """§8: true synchrony at varying SNR; compare naive sd(tau_hat) vs
    corrected (quadratic) vs deconvolution width."""
    rows = []
    k = 0
    for noise_sd in (0.05, 0.15, 0.30, 0.5, 0.8):
        for rep in range(n_rep):
            cfg = SimConfig(n_units=40, n_t=120, regime="synchronous",
                            f_family="degenerate", shape="step",
                            amplitude=1.0, width=0.25, noise_kind="iid",
                            noise_sd=noise_sd, seed=seed + k, cell_id=f"ef-{k}")
            k += 1
            panel, truth, res = run(cfg)
            floor = res.event_aligned_summary.get("est_noise_floor", np.nan)
            naive = res.timing_dispersion["sd"]
            corr = res.timing_dispersion["sd_corrected"]
            grid = res.timing_distribution["grid"]
            dec = res.timing_distribution["deconvolved"]
            # deconvolved sd
            if np.isfinite(dec).all() and dec.sum() > 0:
                w = dec / dec.sum()
                mu = (grid * w).sum()
                dec_sd = np.sqrt(((grid - mu) ** 2 * w).sum())
            else:
                dec_sd = np.nan
            rows.append({
                "tag": "error_floor", "cell": cfg.cell_id,
                "noise_sd": noise_sd, "tau_sd_true": truth["tau_sd"],
                "sd_naive": naive, "sd_corrected": corr,
                "sd_deconv": dec_sd, "est_floor": floor,
                "snr": cfg.amplitude / max(noise_sd * np.sqrt(2), 1e-9),
            })
    return pd.DataFrame(rows)


def operator_block(n_rep=12, seed=SEED0 + 900):
    """§10: per-regime operator sensitivity (fixed in model ops)."""
    rows = []
    k = 0
    for regime, f_family, f_params in [
        ("synchronous", "degenerate", {}),
        ("near_synchronous", "narrow_normal", {"sigma": 0.05}),
        ("diffuse", "broad_normal", {"sigma": 0.35}),
        ("two_cluster", "two_normal_mix", {"delta": 4.5, "sigma": 0.6}),
        ("no_transition", "degenerate", {}),
    ]:
        for rep in range(n_rep):
            cfg = SimConfig(n_units=40, n_t=120, regime=regime,
                            f_family=f_family, f_params=f_params,
                            shape="step", amplitude=1.0, width=0.25,
                            noise_kind="iid", noise_sd=0.15,
                            seed=seed + k, cell_id=f"op-{k}")
            k += 1
            panel, truth, res = run(cfg)
            r = base_row(res, truth, cfg, "operators")
            for op, loc in (res.components.get("operator_locations") or {}).items():
                r[f"op_{op}"] = loc
            rows.append(r)
    return pd.DataFrame(rows)


def weight_trunc_block(n_rep=12, seed=SEED0 + 1400):
    """§11: weighting & truncation sensitivity."""
    rows = []
    k = 0
    for regime, f_family, f_params in [
        ("synchronous", "degenerate", {}),
        ("diffuse", "broad_normal", {"sigma": 0.35}),
        ("two_cluster", "two_normal_mix", {"delta": 4.5, "sigma": 0.6}),
    ]:
        for weight_kind, wparams, miss_kind, mparams in [
            ("equal", {}, "complete", {}),
            ("static", {"shape": 1.0}, "complete", {}),
            ("timing_corr", {"kappa": 1.5}, "complete", {}),
            ("timevarying", {"amp": 0.8}, "complete", {}),
            ("equal", {}, "late_entry", {"frac": 0.35}),
            ("equal", {}, "dropout", {"frac": 0.25}),
            ("equal", {}, "informative", {}),
        ]:
            for rep in range(n_rep):
                cfg = SimConfig(n_units=40, n_t=120, regime=regime,
                                f_family=f_family, f_params=f_params,
                                shape="step", amplitude=1.0, width=0.25,
                                noise_kind="iid", noise_sd=0.15,
                                weight_kind=weight_kind, weight_params=wparams,
                                miss_kind=miss_kind, miss_params=mparams,
                                seed=seed + k, cell_id=f"wt-{k}")
                k += 1
                try:
                    panel, truth, res = run(cfg)
                    r = base_row(res, truth, cfg, "weight_trunc")
                except Exception as e:
                    r = {"tag": "weight_trunc", "cell": cfg.cell_id, "error": repr(e)}
                r.update({"weight_kind": weight_kind, "miss_kind": miss_kind})
                rows.append(r)
    return pd.DataFrame(rows)


def main(quick=False):
    global CLF
    CLF = joblib.load(ROOT / "simulations/calibration/classifier.joblib")
    n = 4 if quick else None
    t0 = time.time()
    trend = trend_block(n_rep=n or 12)
    trend.to_csv(ROOT / "trend_heterogeneity_results.csv", index=False)
    print(f"trend: {len(trend)} rows ({time.time()-t0:.0f}s)")
    ef = error_floor_block(n_rep=n or 16)
    ef.to_csv(ROOT / "error_floor_results.csv", index=False)
    print(f"error_floor: {len(ef)} rows")
    ops = operator_block(n_rep=n or 12)
    ops.to_csv(ROOT / "operator_sensitivity.csv", index=False)
    print(f"operators: {len(ops)} rows")
    wt = weight_trunc_block(n_rep=n or 12)
    wt.to_csv(ROOT / "weighting_truncation_results.csv", index=False)
    print(f"weight_trunc: {len(wt)} rows ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
