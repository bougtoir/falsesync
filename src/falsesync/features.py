"""Feature extraction for regime classification (Phase 3).

Every feature is a rescaled, roughly O(1) quantity so the multinomial
logistic classifier is numerically stable. Features are documented in
diagnostic_specification.md §4 (s1..s4) plus Phase-3 additions.
"""

from __future__ import annotations

import numpy as np

from .aggregation import weighted_aggregate
from .changepoints import aggregate_breakpoint, ls_breakpoint, multibreak
from .inference import deconvolve_timing, unit_break_uncertainties

FEATURE_NAMES = [
    "agg_strength", "excess_sd_ratio", "n_modes", "mode_mass2",
    "s1_width", "s3_alignment", "s4_weighting", "noise_floor_ratio",
    "fwhm_ratio", "agg_skew", "log_n", "log_T", "frac_valid_units",
    "binseg_n_breaks", "unit_strength_med",
]


def _kde(samples, grid, bw):
    u = (grid[:, None] - samples[None, :]) / bw
    return np.exp(-0.5 * u * u).mean(axis=1) / (bw * np.sqrt(2 * np.pi))


def _mode_count(f, thr=0.3, min_sep_frac=0.08):
    """Count modes with prominence threshold and minimum separation.

    A mode must exceed `thr` * max(f) locally and be at least
    `min_sep_frac` * len(f) grid points away from a higher mode
    (greedy, highest peak first).
    """
    if f.size < 3 or f.max() <= 0:
        return 0, []
    cand = [i for i in range(1, len(f) - 1)
            if f[i] >= f[i - 1] and f[i] > f[i + 1] and f[i] > thr * f.max()]
    cand.sort(key=lambda i: -f[i])
    sep = max(1, int(min_sep_frac * len(f)))
    chosen = []
    for i in cand:
        if all(abs(i - j) >= sep for j in chosen):
            chosen.append(i)
    return len(chosen), sorted(chosen)


def extract_features(
    t: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray | None = None,
    windows: np.ndarray | None = None,
    aggregate: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    unit_boot: int = 30,
) -> tuple[np.ndarray, dict]:
    """Feature vector + aux dict used by the classifier and the result object."""
    rng = rng or np.random.default_rng(0)
    t = np.asarray(t, dtype=float)
    n = values.shape[0]
    if aggregate is None:
        aggregate = weighted_aggregate(values, weights)
    valid_u = np.isfinite(values).sum(axis=1) >= 8
    agg_f = np.isfinite(aggregate)
    agg_y = np.nan_to_num(aggregate, nan=np.nanmean(aggregate[agg_f])) if agg_f.any() else np.zeros(t.size)

    # aggregate break
    res_agg = ls_breakpoint(t, agg_y)
    dm = np.gradient(agg_y, t)
    half = np.max(dm) / 2
    idx = np.where(dm >= half)[0]
    w_agg = float(t[idx[-1]] - t[idx[0]]) if idx.size > 1 else np.inf

    # unit breaks
    taus = np.full(n, np.nan)
    strengths = np.full(n, np.nan)
    for i in np.where(valid_u)[0]:
        y = values[i]
        fill = np.nanmean(y[np.isfinite(y)])
        try:
            r = ls_breakpoint(t, np.nan_to_num(y, nan=fill))
            taus[i] = r.location
            strengths[i] = r.strength
        except Exception:
            pass
    tv = taus[np.isfinite(taus)]
    sv = strengths[np.isfinite(taus)]

    s_i = unit_break_uncertainties(t, values[np.isfinite(taus)], tv, rng=rng, n_boot=unit_boot) if tv.size >= 5 else np.array([])
    noise_floor = float(np.nanmedian(s_i)) if s_i.size and np.isfinite(s_i).any() else np.nan

    tau_sd = float(np.std(tv)) if tv.size else np.nan
    excess = np.sqrt(max(tau_sd**2 - (noise_floor if np.isfinite(noise_floor) else 0) ** 2, 0)) if np.isfinite(tau_sd) else np.nan
    scale = np.ptp(t)
    excess_ratio = float(excess / scale) if np.isfinite(excess) else np.nan
    floor_ratio = float(noise_floor / max(tau_sd, 1e-9)) if np.isfinite(noise_floor) and np.isfinite(tau_sd) else np.nan

    grid = np.linspace(t.min(), t.max(), 300)
    bw = max(1.06 * (tau_sd if np.isfinite(tau_sd) else 1.0) * max(len(tv), 2) ** -0.2, 3 * np.median(np.diff(t)))
    f_hat = _kde(tv, grid, bw) if tv.size >= 5 else np.zeros(grid.size)
    f_dec_raw = deconvolve_timing(tv, s_i, grid) if tv.size >= 5 else np.zeros(grid.size)
    # smooth deconvolved density before mode counting (deconvolution is noisy)
    k = max(3, int(0.04 * grid.size) | 1)
    kern = np.exp(-0.5 * (np.arange(k) - k // 2) ** 2 / (k / 4) ** 2)
    kern /= kern.sum()
    f_dec = np.convolve(np.pad(f_dec_raw, k // 2, mode="edge"), kern, mode="valid")
    f_dec = np.clip(f_dec, 0, None)
    # modes counted on the (smoother) KDE; deconvolved density is used for
    # dispersion correction, where its noise matters less
    n_modes, modes = _mode_count(f_hat, thr=0.3, min_sep_frac=0.08)
    mode_mass2 = float(np.sort(f_dec)[::-1][:2].sum() / max(f_dec.sum(), 1e-12)) if f_dec.sum() > 0 else 0.0

    s1 = float(tau_sd / w_agg) if np.isfinite(tau_sd) and np.isfinite(w_agg) and w_agg > 0 else np.nan
    s3 = float(np.mean(np.abs(sv) > 2.0)) if sv.size else np.nan

    s4 = 0.0
    if weights is not None and tv.size >= 5:
        wv = np.asarray(weights)
        if wv.ndim == 2:
            wv = wv.mean(axis=1)
        try:
            s4 = max(s4, abs(float(np.corrcoef(wv[np.isfinite(taus)], tv)[0, 1])))
        except Exception:
            pass
    if windows is not None and tv.size >= 5:
        span = np.asarray(windows)[np.isfinite(taus)]
        span = span[:, 1] - span[:, 0]
        try:
            s4 = max(s4, abs(float(np.corrcoef(span, tv)[0, 1])))
        except Exception:
            pass

    agg_skew = float(((agg_y - agg_y.mean()) ** 3).mean() / max(agg_y.std(), 1e-9) ** 3)
    try:
        nb = len(multibreak(t, agg_y, n_bkps=5).extras.get("all_break_indices", []))
    except Exception:
        nb = 1

    feats = np.array([
        res_agg.strength,
        excess_ratio if np.isfinite(excess_ratio) else 0.0,
        float(n_modes),
        mode_mass2,
        s1 if np.isfinite(s1) else 0.0,
        s3 if np.isfinite(s3) else 0.0,
        float(s4),
        floor_ratio if np.isfinite(floor_ratio) else 1.0,
        float(w_agg / scale) if np.isfinite(w_agg) else 1.0,
        agg_skew,
        float(np.log(max(n, 1))),
        float(np.log(t.size)),
        float(valid_u.mean()),
        float(nb),
        float(np.nanmedian(np.abs(sv))) if sv.size else 0.0,
    ])
    aux = {
        "agg_break": res_agg.location, "agg_strength": res_agg.strength,
        "taus": taus, "unit_strengths": strengths, "s_i": s_i,
        "noise_floor": noise_floor, "tau_sd": tau_sd, "f_dec": (grid, f_dec),
        "f_hat": (grid, f_hat), "w_agg": w_agg, "n_modes": n_modes,
        "aggregate": agg_y,
    }
    return feats, aux
