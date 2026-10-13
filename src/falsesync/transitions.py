"""Unit transition shapes g(u), u = (t - tau)/h, in [0, 1].

All supported shapes satisfy the antisymmetry identity g(u) + g(-u) = 1
required by proposition P2, except `linear` which satisfies it on the
saturated extension used here.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

SHAPES = ("step", "logistic", "probit", "linear")


def g(u: np.ndarray, shape: str = "logistic") -> np.ndarray:
    """Transition function values. `u` is already standardized."""
    u = np.asarray(u, dtype=float)
    if shape == "step":
        out = (u >= 0).astype(float)
        out[u == 0] = 0.5  # antisymmetric convention; measure-zero for integrals
        return out
    if shape == "logistic":
        return 1.0 / (1.0 + np.exp(-u))
    if shape == "probit":
        return norm.cdf(u)
    if shape == "linear":
        return np.clip(0.5 + u / 2.0, 0.0, 1.0)
    raise ValueError(f"unknown transition shape {shape!r}; choose from {SHAPES}")


def g_prime(u: np.ndarray, shape: str = "logistic") -> np.ndarray:
    """Derivative of the transition (density kernel used by P8 ill-posedness).

    For `step` this is a Dirac delta conceptually; returns zeros except at the
    discontinuity, where callers should use the P1 identity directly.
    """
    u = np.asarray(u, dtype=float)
    if shape == "step":
        out = np.full(u.shape, np.inf)
        out[u != 0] = 0.0
        return out
    if shape == "logistic":
        s = g(u, "logistic")
        return s * (1.0 - s)
    if shape == "probit":
        return norm.pdf(u)
    if shape == "linear":
        return np.where(np.abs(u) < 1.0, 0.5, 0.0)
    raise ValueError(f"unknown transition shape {shape!r}")


def is_antisymmetric(shape: str) -> bool:
    """Whether g(u) + g(-u) = 1 holds for this shape (P2 condition)."""
    u = np.linspace(-8, 8, 401)
    return bool(np.allclose(g(u, shape) + g(-u, shape), 1.0, atol=1e-12))
