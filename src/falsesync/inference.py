"""Uncertainty: unit-level bootstrap and error-law deconvolution.

See resampling_plan.md and deconvolution_assessment.md. No asymptotic
coverage claims — intervals are bootstrap intervals.
"""

from __future__ import annotations

import numpy as np


def unit_break_uncertainties(
    t: np.ndarray,
    values: np.ndarray,
    taus: np.ndarray,
    rng: np.random.Generator | None = None,
    n_boot: int = 60,
) -> np.ndarray:
    """Residual bootstrap around each unit's two-segment fit -> s_i (the F_e law)."""
    from .changepoints import ls_breakpoint

    rng = rng or np.random.default_rng(0)
    t = np.asarray(t, dtype=float)
    n = values.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        y = np.asarray(values[i], dtype=float)
        ok = np.isfinite(y)
        if ok.sum() < 8:
            continue
        ti, yi = t[ok], y[ok]
        res = ls_breakpoint(ti, yi)
        c = res.index
        fit = np.concatenate([np.full(c, yi[:c].mean()), np.full(yi.size - c, yi[c:].mean())])
        resid = yi - fit
        resid = resid - resid.mean()
        boots = []
        for _ in range(n_boot):
            yb = fit + rng.choice(resid, resid.size, replace=True)
            try:
                boots.append(ls_breakpoint(ti, yb).location)
            except Exception:
                continue
        if len(boots) >= 10:
            out[i] = float(np.std(boots))
    return out


def deconvolve_timing(
    taus: np.ndarray, s_i: np.ndarray, grid: np.ndarray, damp: float = 0.02
) -> np.ndarray:
    """Recover f_tau from noisy unit break estimates (P8c machinery).

    Gaussian errors with per-unit sd s_i: f_hat_tau = f_tau * phi_bar, where
    phi_bar is the average error kernel. Damped Fourier deconvolution with
    fixed Tikhonov factor `damp` (a priori, not tuned).
    """
    taus = np.asarray(taus, dtype=float)
    s_i = np.asarray(s_i, dtype=float)
    grid = np.asarray(grid, dtype=float)
    dt = grid[1] - grid[0]
    n_g = grid.size
    hist, _ = np.histogram(taus, bins=np.append(grid, grid[-1] + dt), density=True)
    sbar2 = float(np.nanmean(s_i**2)) if np.isfinite(s_i).any() else 0.0
    F_hist = np.fft.rfft(hist)
    freqs = np.fft.rfftfreq(n_g, dt)
    phi_bar = np.exp(-2 * np.pi**2 * freqs**2 * sbar2)  # E over units of error char. fn (approx: mean sd)
    denom = phi_bar**2 + damp
    F_f = F_hist * phi_bar / denom
    f = np.fft.irfft(F_f, n_g).real
    return np.clip(f, 0, None)


def cluster_bootstrap_band(
    taus: np.ndarray,
    grid: np.ndarray,
    bw: float,
    rng: np.random.Generator | None = None,
    n_boot: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster (unit-resample) bootstrap band for the KDE of f_tau."""
    rng = rng or np.random.default_rng(0)
    taus = np.asarray(taus, dtype=float)
    dens = []
    for _ in range(n_boot):
        s = rng.choice(taus, taus.size, replace=True)
        u = (grid[:, None] - s[None, :]) / bw
        dens.append(np.exp(-0.5 * u * u).mean(axis=1) / (bw * np.sqrt(2 * np.pi)))
    d = np.stack(dens)
    return np.quantile(d, 0.025, axis=0), np.quantile(d, 0.975, axis=0)
