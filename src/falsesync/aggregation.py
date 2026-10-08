"""Weighted aggregation and the population convolution m(t)."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .transitions import g


def weighted_aggregate(
    values: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """Row-weighted mean over units, respecting NaN observation windows.

    values: (n_units, n_t); weights: (n_units,) constant over t, or
    (n_units, n_t) for time-varying weights w_i(t).
    Implements m_obs(t) = sum_i w_i 1{obs} Y_i / sum_i w_i 1{obs} (P5).
    """
    v = np.asarray(values, dtype=float)
    n = v.shape[0]
    if weights is None:
        w = np.ones((n, 1))
    else:
        w = np.asarray(weights, dtype=float)
        if w.ndim == 1:
            w = w[:, None]
        elif w.shape != v.shape:
            raise ValueError(
                f"2D weights must match values shape {v.shape}, got {w.shape}"
            )
    obs = np.isfinite(v)
    wv = np.where(obs, v * w, 0.0)
    denom = np.where(obs, w, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, wv.sum(axis=0) / denom, np.nan)


def timing_cdf(tau_grid: np.ndarray, spec: dict) -> np.ndarray:
    """Population F_tau on a grid (used by the P1/P5 identities)."""
    from .simulation import timing_density

    tg = np.asarray(tau_grid, dtype=float)
    f = timing_density(tg, spec)
    c = np.concatenate([[0.0], np.cumsum((f[:-1] + f[1:]) / 2 * np.diff(tg))])
    return np.clip(c / c[-1], 0, 1)


def population_m(
    t: np.ndarray,
    spec: dict,
    shape: str = "logistic",
    amplitude: float = 1.0,
    baseline: float = 0.0,
    width: float = 0.3,
    tau_grid: np.ndarray | None = None,
) -> np.ndarray:
    """Population mean curve a + A * (g * f_tau)(t) via quadrature.

    Closed forms used when available: step g -> F_tau(t) (P1);
    probit + normal -> Phi((t-mu)/sqrt(1+sigma^2)) with unit width (P3a).
    """
    t = np.asarray(t, dtype=float)
    kind = spec.get("kind", "normal")
    if shape == "step" and kind in ("normal", "mixture", "uniform"):
        return baseline + amplitude * np.interp(
            t, np.asarray(tau_grid if tau_grid is not None else t),
            timing_cdf(np.asarray(tau_grid if tau_grid is not None else t), spec),
        )
    if shape == "probit" and kind == "normal" and width == 1.0:
        return baseline + amplitude * norm.cdf(
            (t - spec["mu"]) / np.sqrt(1 + spec["sigma"] ** 2)
        )
    tg = (
        np.asarray(tau_grid, dtype=float)
        if tau_grid is not None
        else np.linspace(t.min() - 8 * width, t.max() + 8 * width, 4001)
    )
    from .simulation import timing_density

    f = timing_density(tg, spec)
    u = (t[:, None] - tg[None, :]) / width
    trapz = getattr(np, "trapezoid", np.trapz)
    conv = trapz(g(u, shape) * f[None, :], tg, axis=1)
    return baseline + amplitude * conv
