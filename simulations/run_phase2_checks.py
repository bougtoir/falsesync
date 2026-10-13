"""Minimum Phase-2 theory checks (spec §20, cases A-H).

Produces phase2_theory_checks.csv and figures under outputs/phase2/.
Run: PYTHONPATH=src python3 simulations/run_phase2_checks.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs" / "phase2"
OUT.mkdir(parents=True, exist_ok=True)

from falsesync.aggregation import population_m, timing_cdf, weighted_aggregate
from falsesync.changepoints import aggregate_breakpoint, ls_breakpoint, maxslope_breakpoint, multibreak
from falsesync.plotting import (
    aggregate_curve_figure,
    aligned_panel_figure,
    operator_comparison_figure,
    timing_distribution_figure,
)
from falsesync.alignment import event_time_realign
from falsesync.simulation import sample_taus, simulate_panel, timing_density

T = np.linspace(0, 10, 2001)
TG = np.linspace(-2, 12, 6001)
RNG = np.random.default_rng(2026)
rows = []


def rec(case, check, value, expected="", status="ok", note=""):
    rows.append(
        {"case": case, "check": check, "value": f"{value:.6g}" if isinstance(value, float) else value,
         "expected": expected, "status": status, "note": note}
    )


def ls_loc(m):
    return ls_breakpoint(T, np.nan_to_num(m, nan=np.nanmean(m[np.isfinite(m)]))).location


# ---- A: step g + degenerate F_tau (point mass) -----------------------------
specA = {"kind": "normal", "mu": 5.0, "sigma": 1e-6}
mA = population_m(T, specA, shape="step", tau_grid=TG)
rec("A", "ls_location", ls_loc(mA), expected=5.0)
rec("A", "maxslope_location", maxslope_breakpoint(T, mA).location, expected=5.0)
aggregate_curve_figure(T, mA, None, OUT / "caseA_step_degenerate.png", "A: step g, degenerate F")

# ---- B: step g + uniform F_tau ---------------------------------------------
specB = {"kind": "uniform", "lo": 3.0, "hi": 7.0}
mB = population_m(T, specB, shape="step", tau_grid=TG)
rec("B", "ls_location", ls_loc(mB), expected=5.0)
rec("B", "m_at_t_is_F", float(np.max(np.abs(mB - timing_cdf(T, specB)))), expected="~0")
rec("B", "maxslope_degenerate", maxslope_breakpoint(T, mB).location, note="uniform density: argmax is flat boundary artifact")
aggregate_curve_figure(T, mB, None, OUT / "caseB_step_uniform.png", "B: step g, uniform F")

# ---- C: step g + normal F_tau ----------------------------------------------
specC = {"kind": "normal", "mu": 5.0, "sigma": 0.8}
mC = population_m(T, specC, shape="step", tau_grid=TG)
rec("C", "ls_location", ls_loc(mC), expected=5.0)
rec("C", "maxslope_location", maxslope_breakpoint(T, mC).location, expected=5.0)
rec("C", "max_slope", float(np.max(np.gradient(mC, T))),
    expected=f"{max(timing_density(T, specC)):.6g}", note="m' = f_tau peak (P1)")
aggregate_curve_figure(T, mC, None, OUT / "caseC_step_normal.png", "C: step g, normal F")

# ---- D: logistic g + normal F_tau ------------------------------------------
specD = {"kind": "normal", "mu": 5.0, "sigma": 0.8}
mD = population_m(T, specD, shape="logistic", width=0.4, tau_grid=TG)
resD = ls_breakpoint(T, mD)
rec("D", "ls_location", resD.location, expected=5.0)
c = resD.index
rec("D", "ls_foc_residual", float(abs(mD[c] - (mD[:c].mean() + mD[c:].mean()) / 2)), expected="~0", note="P7")
rec("D", "point_symmetry_max_dev",
    float(np.max(np.abs(np.interp(5 - np.linspace(0.1, 4, 40), T, mD)
                        + np.interp(5 + np.linspace(0.1, 4, 40), T, mD) - 1.0))),
    expected="~0", note="P2")
aggregate_curve_figure(T, mD, None, OUT / "caseD_logistic_normal.png", "D: logistic g, normal F")

# ---- E: logistic g + two-component mixture F_tau ---------------------------
from scipy.stats import norm as _norm

mode_rows = []
for delta in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
    specE = {"kind": "mixture", "pi": 0.5, "mu1": 5 - delta / 2, "mu2": 5 + delta / 2, "sigma": 1.0}
    mE = population_m(T, specE, shape="logistic", width=0.4, tau_grid=TG)
    f = timing_density(TG, specE)
    modes = int(np.sum((f[1:-1] > f[:-2]) & (f[1:-1] >= f[2:]) & (f[1:-1] > 0.05 * f.max())))
    rec("E", f"bimodal_delta={delta}", modes, expected=">=2 iff delta>2sigma", note="P6a")
    rec("E", f"ls_location_delta={delta}", ls_loc(mE), expected=5.0)
    mode_rows.append(delta)
    if delta in (1.0, 3.0):
        aggregate_curve_figure(T, mE, None, OUT / f"caseE_mixture_delta{delta}.png",
                               f"E: mixture F, delta={delta}")
aggregate_curve_figure(T, mE, None, OUT / "mixture_bimodality.png", "E: mixture F, delta=4")

# pi sweep for P6b numeric table
for pi in [0.1, 0.3, 0.5, 0.7, 0.9]:
    specE2 = {"kind": "mixture", "pi": pi, "mu1": 4.0, "mu2": 6.5, "sigma": 1.0}
    mE2 = population_m(T, specE2, shape="logistic", width=0.4, tau_grid=TG)
    rec("E", f"ls_location_pi={pi}", ls_loc(mE2), expected=f"mean={pi * 4 + (1 - pi) * 6.5:.3g}",
        note="P6b: LS interpolates centers, != mean in asymmetric case")

# ---- F: weighted vs unweighted ----------------------------------------------
nF = 6000
tausF = RNG.normal(5.0, 0.8, nF)
for kappa in [0.0, 0.5, 1.0, 1.5]:
    w = np.exp(-kappa * tausF)
    w /= w.mean()
    panelF = simulate_panel(T, tausF, shape="step", weights=w, noise_sd=0.0, rng=RNG)
    mF = weighted_aggregate(panelF.values, panelF.weights)
    rec("F", f"ls_location_kappa={kappa}", ls_loc(mF), note="P4: weight-tilted estimand")
# exact tilt check: m_w should equal convolution vs tilted F (numeric via importance resample)
w = np.exp(-1.0 * tausF); w /= w.mean()
p = w / w.sum()
idx = RNG.choice(nF, nF, replace=True, p=p)
tilted = np.sort(tausF[idx])
panelF2 = simulate_panel(T, tilted, shape="step", noise_sd=0.0, rng=RNG)
mF2 = weighted_aggregate(panelF2.values)
panelF3 = simulate_panel(T, tausF, shape="step", weights=w, noise_sd=0.0, rng=RNG)
mF3 = weighted_aggregate(panelF3.values, panelF3.weights)
rec("F", "tilted_conv_vs_weighted_maxdev", float(np.nanmax(np.abs(mF2 - mF3))),
    expected="<0.05", note="P4 numerical equivalence")

# ---- G: truncated observation window ----------------------------------------
nG = 6000
tausG = RNG.normal(5.0, 0.7, nG)
win = np.column_stack([tausG + RNG.normal(0, 0.4, nG), np.full(nG, 10.0)])
panelG = simulate_panel(T, tausG, shape="step", windows=win, noise_sd=0.0, rng=RNG)
mG = weighted_aggregate(panelG.values)
fullG = population_m(T, {"kind": "normal", "mu": 5.0, "sigma": 0.7}, shape="step", tau_grid=TG)
rec("G", "ls_location_truncated", ls_loc(mG), note="P5: biased early vs 5.0 (late-entering units arrive post-transition)")
rec("G", "max_dev_vs_F", float(np.nanmax(np.abs(mG - fullG))), expected=">0")
obs_at = (win[:, 0:1] <= T[None, :]) & (win[:, 1:2] >= T[None, :])
obs_count = obs_at.sum(axis=0)
cond_cdf = np.where(
    obs_count > 0,
    ((tausG[:, None] <= T[None, :]) & obs_at).sum(axis=0) / np.maximum(obs_count, 1),
    np.nan,
)
valid_g = np.isfinite(cond_cdf)
rec("G", "conditional_cdf_identity",
    float(np.nanmax(np.abs(mG[valid_g] - cond_cdf[valid_g]))),
    expected="~0", note="P5: m_obs = P(tau<=t | obs at t)")
aggregate_curve_figure(T, fullG, mG, OUT / "caseG_truncation.png", "G: truncated window")

# ---- H: serial noise sanity check -------------------------------------------
tausH = RNG.normal(5.0, 0.8, 400)
panelH = simulate_panel(T, tausH, shape="logistic", width=0.4, noise_sd=0.2, rng=RNG)
# AR(1)-like noise injection
ar = np.zeros_like(panelH.values)
eps = RNG.normal(0, 0.15, panelH.values.shape)
for k in range(1, panelH.values.shape[1]):
    ar[:, k] = 0.7 * ar[:, k - 1] + eps[:, k]
valuesH = np.nan_to_num(panelH.values) + ar
mH = weighted_aggregate(valuesH)
rec("H", "ls_location_ar1_noise", ls_loc(mH), expected="~5.0",
    note="serial noise inflates unit tau dispersion -> CE-UNITERR guard needed")
boot = [ls_loc(weighted_aggregate(valuesH + RNG.normal(0, 0.05, valuesH.shape))) for _ in range(20)]
rec("H", "ls_boot_sd", float(np.std(boot)), note="aggregate break stability under noise jitter")
aggregate_curve_figure(T, mH, None, OUT / "caseH_ar1_noise.png", "H: AR(1) noise")

# ---- CE checks referenced by counterexamples.md -----------------------------
# CE-SPIKE
tausS = sample_taus(8000, {"kind": "spike", "center": 5.0, "eps": 0.3, "lo": 0.0, "hi": 10.0}, RNG)
panelS = simulate_panel(T, tausS, shape="step", noise_sd=0.0, rng=RNG)
mS = weighted_aggregate(panelS.values)
dmS = np.gradient(mS, T)
rec("CE", "spike_max_slope", float(np.nanmax(dmS)), expected="~A*(1-eps)/dt",
    note="CE-SPIKE: large variance but step-sharp aggregate")
# CE-WINDOW
specG = {"kind": "gamma", "shape": 3.0, "scale": 1.0, "shift": 1.0}
tg2 = np.linspace(0, 20, 4001)
for Tend in [8.0, 12.0, 20.0]:
    tw = np.linspace(0, Tend, int(Tend * 200) + 1)
    mw = population_m(tw, specG, shape="logistic", width=0.5, tau_grid=tg2)
    rec("CE", f"window_ls_T={Tend}", ls_breakpoint(tw, mw).location,
        note="CE-WINDOW: same F_tau, different window, different break")
# CE-ALIGN
tausA = np.full(500, 5.0)
panelA = simulate_panel(T, tausA, shape="logistic", width=0.4, noise_sd=0.25, rng=RNG)
taus_hat = np.array([ls_breakpoint(T, np.nan_to_num(row)).location for row in panelA.values])
al = event_time_realign(T, panelA.values, taus_hat)
s_dt = float(np.median(np.diff(T)))
rec("CE", "align_jitter_sd", float(np.std(taus_hat - 5.0)), expected=">0",
    note="CE-ALIGN: estimation noise mimics heterogeneity even at synchronous tau")
s_grid = np.arange(al.shape[1]) * s_dt - (al.shape[1] // 2) * s_dt
aligned_panel_figure(s_grid, al, OUT / "ce_align_panel.png", "CE-ALIGN: synchronous units, noisy tau_hat")

# ---- operator comparison figure ---------------------------------------------
mOC = population_m(T, specG, shape="logistic", width=0.5, tau_grid=tg2)
res_ops = {
    "ls": ls_breakpoint(T, mOC).location,
    "maxslope": maxslope_breakpoint(T, mOC).location,
    "cusum": aggregate_breakpoint(T, mOC, "cusum").location,
    "binseg": multibreak(T, mOC, n_bkps=3).location,
}
operator_comparison_figure(T, mOC, res_ops, OUT / "operator_comparison.png")
for k, v in res_ops.items():
    rec("OPS", f"T_{k}", v, note="operator-dependent estimand (P7)")

# timing distribution figure (case C vs E)
timing_distribution_figure(
    TG, {"normal(5,0.8)": timing_density(TG, specC),
         "mixture delta=4": timing_density(TG, {"kind": "mixture", "pi": 0.5, "mu1": 3, "mu2": 7, "sigma": 1.0})},
    OUT / "timing_distributions.png",
)

with open(ROOT / "phase2_theory_checks.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["case", "check", "value", "expected", "status", "note"])
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} checks -> phase2_theory_checks.csv; figures -> {OUT}")
