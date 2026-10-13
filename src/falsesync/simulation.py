"""Timing simulation and panel generation (model ladder S1-S7)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm

from .transitions import g


@dataclass
class Panel:
    t: np.ndarray
    values: np.ndarray  # (n_units, n_t)
    taus: np.ndarray
    weights: np.ndarray  # (n_units,) applied uniformly over t
    windows: np.ndarray  # (n_units, 2) [L_i, U_i]
    shape: str
    amplitudes: np.ndarray
    meta: dict = field(default_factory=dict)


def sample_taus(n: int, spec: dict, rng: np.random.Generator) -> np.ndarray:
    """Sample transition times from a timing spec.

    kinds: normal {mu,sigma}, mixture {pi,mu1,mu2,sigma1,sigma2},
    gamma {shape,scale,shift}, uniform {lo,hi}, spike {center,eps,lo,hi}.
    """
    kind = spec.get("kind", "normal")
    if kind == "normal":
        return rng.normal(spec["mu"], spec["sigma"], n)
    if kind == "mixture":
        pick = rng.random(n) < spec.get("pi", 0.5)
        out = np.empty(n)
        out[pick] = rng.normal(spec["mu1"], spec.get("sigma1", spec.get("sigma", 0.3)), pick.sum())
        out[~pick] = rng.normal(
            spec["mu2"], spec.get("sigma2", spec.get("sigma", 0.3)), (~pick).sum()
        )
        return out
    if kind == "gamma":
        return spec.get("shift", 0.0) + gamma_dist.rvs(
            spec["shape"], scale=spec["scale"], size=n, random_state=rng
        )
    if kind == "uniform":
        return rng.uniform(spec["lo"], spec["hi"], n)
    if kind == "spike":
        eps = spec.get("eps", 0.2)
        out = np.where(rng.random(n) < 1 - eps, spec["center"], np.nan)
        mask = np.isnan(out)
        out[mask] = rng.uniform(spec["lo"], spec["hi"], mask.sum())
        return out
    raise ValueError(f"unknown tau spec kind {kind!r}")


def simulate_panel(
    t: np.ndarray,
    taus: np.ndarray,
    shape: str = "logistic",
    amplitude: float | np.ndarray = 1.0,
    baseline: float | np.ndarray = 0.0,
    width: float | np.ndarray = 0.3,
    noise_sd: float = 0.0,
    weights: np.ndarray | None = None,
    windows: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> Panel:
    """S2-style homogeneous-shape panel: Y_i(t) = a_i + A_i g((t-tau_i)/h_i) + eps."""
    rng = rng or np.random.default_rng()
    t = np.asarray(t, dtype=float)
    taus = np.asarray(taus, dtype=float)
    n = taus.size
    A = np.broadcast_to(np.asarray(amplitude, dtype=float), (n,))
    a = np.broadcast_to(np.asarray(baseline, dtype=float), (n,))
    h = np.broadcast_to(np.asarray(width, dtype=float), (n,))
    u = (t[None, :] - taus[:, None]) / h[:, None]
    values = a[:, None] + A[:, None] * g(u, shape)
    if noise_sd > 0:
        values = values + rng.normal(0, noise_sd, values.shape)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    win = (
        np.column_stack([np.full(n, t.min()), np.full(n, t.max())])
        if windows is None
        else np.asarray(windows, dtype=float)
    )
    obs = (t[None, :] >= win[:, 0:1]) & (t[None, :] <= win[:, 1:2])
    values = np.where(obs, values, np.nan)
    return Panel(t=t, values=values, taus=taus, weights=w, windows=win, shape=shape,
                 amplitudes=A, meta={"baseline": a, "width": h, "noise_sd": noise_sd})


def timing_density(tau_grid: np.ndarray, spec: dict) -> np.ndarray:
    """Population density f_tau on a grid for closed-form checks."""
    kind = spec.get("kind", "normal")
    tg = np.asarray(tau_grid, dtype=float)
    if kind == "normal":
        return norm.pdf(tg, spec["mu"], spec["sigma"])
    if kind == "mixture":
        pi = spec.get("pi", 0.5)
        s1 = spec.get("sigma1", spec.get("sigma", 0.3))
        s2 = spec.get("sigma2", spec.get("sigma", 0.3))
        return pi * norm.pdf(tg, spec["mu1"], s1) + (1 - pi) * norm.pdf(tg, spec["mu2"], s2)
    if kind == "uniform":
        return np.where((tg >= spec["lo"]) & (tg <= spec["hi"]),
                        1.0 / (spec["hi"] - spec["lo"]), 0.0)
    if kind == "gamma":
        return gamma_dist.pdf(tg - spec.get("shift", 0.0), spec["shape"], scale=spec["scale"])
    raise ValueError(f"no analytic density for kind {kind!r}")
