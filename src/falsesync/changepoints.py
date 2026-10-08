"""Breakpoint operators T_M on a univariate series.

Each operator returns a BreakResult; the `method` tag identifies which
estimand was computed (see breakpoint_operator_comparison.md). ruptures is
used when installed; every method has a graceful fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BreakResult:
    location: float
    index: int
    strength: float  # (mean_R - mean_L) / sigma_hat
    method: str
    extras: dict = field(default_factory=dict)


def ls_breakpoint(t: np.ndarray, y: np.ndarray) -> BreakResult:
    """Single least-squares break in the mean (piecewise-constant model)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    n = y.size
    cum1 = np.concatenate([[0.0], np.cumsum(y)])
    cum2 = np.concatenate([[0.0], np.cumsum(y * y)])

    def sse(c: int) -> float:
        l_sum, l_sq = cum1[c], cum2[c]
        r_sum, r_sq = cum1[n] - cum1[c], cum2[n] - cum2[c]
        l_sse = l_sq - l_sum**2 / max(c, 1)
        r_sse = r_sq - r_sum**2 / max(n - c, 1)
        return l_sse + r_sse

    cs = np.arange(2, n - 2)
    vals = np.array([sse(c) for c in cs])
    c_hat = cs[int(np.argmin(vals))]
    mean_l, mean_r = y[:c_hat].mean(), y[c_hat:].mean()
    sigma = np.sqrt(max(sse(c_hat) / (n - 2), 1e-12))
    return BreakResult(
        location=float(t[c_hat]),
        index=int(c_hat),
        strength=float((mean_r - mean_l) / sigma),
        method="ls",
        extras={"mean_l": float(mean_l), "mean_r": float(mean_r), "sigma": float(sigma)},
    )


def maxslope_breakpoint(t: np.ndarray, y: np.ndarray) -> BreakResult:
    """Breakpoint at argmax of the numerical derivative (T_ms)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    d = np.gradient(y, t)
    i = int(np.argmax(np.abs(d)))
    return BreakResult(
        location=float(t[i]), index=i, strength=float(d[i]), method="maxslope"
    )


def cusum_breakpoint(t: np.ndarray, y: np.ndarray) -> BreakResult:
    """Single CUSUM-type breakpoint (argmax |cumulative deviation|)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    s = np.cumsum(y - y.mean())
    i = int(np.argmax(np.abs(s)))
    res = ls_breakpoint(t, y)
    return BreakResult(t[i], i, res.strength, "cusum", {"ls_location": res.location})


def multibreak(t: np.ndarray, y: np.ndarray, n_bkps: int = 3) -> BreakResult:
    """Penalized multi-break via binary segmentation (LS fallback).

    Uses ruptures.Binseg when installed; otherwise a native binary
    segmentation on the LS objective (same estimand family at k=1).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    try:
        import ruptures as rpt

        bkps = rpt.Binseg(model="l2").fit(y).predict(n_bkps=n_bkps)
        idx = [b for b in bkps if b < y.size]
    except Exception:
        idx = _native_binseg(y, n_bkps)
    main = idx[0] if idx else ls_breakpoint(t, y).index
    res = ls_breakpoint(t, y)
    return BreakResult(
        float(t[main]), main, res.strength, "binseg",
        {"all_break_indices": idx, "all_break_locations": [float(t[i]) for i in idx]},
    )


def _native_binseg(y: np.ndarray, n_bkps: int) -> list[int]:
    """Greedy binary segmentation on the LS objective."""
    n = y.size
    segments = [(0, n)]
    breaks: list[int] = []
    while len(breaks) < n_bkps:
        best_gain, best = 0.0, None
        for lo, hi in segments:
            if hi - lo < 4:
                continue
            seg = y[lo:hi]
            cum1 = np.concatenate([[0.0], np.cumsum(seg)])
            cum2 = np.concatenate([[0.0], np.cumsum(seg * seg)])
            base = cum2[-1] - cum1[-1] ** 2 / seg.size
            for c in range(2, seg.size - 2):
                l = cum2[c] - cum1[c] ** 2 / c
                r = (cum2[-1] - cum2[c]) - (cum1[-1] - cum1[c]) ** 2 / (seg.size - c)
                gain = base - l - r
                if gain > best_gain:
                    best_gain, best = gain, (lo, hi, lo + c)
        if best is None:
            break
        lo, hi, c = best
        breaks.append(c)
        segments.remove((lo, hi))
        segments.extend([(lo, c), (c, hi)])
    return sorted(breaks)


def aggregate_breakpoint(
    t: np.ndarray, y: np.ndarray, method: str = "ls", **kwargs
) -> BreakResult:
    """Dispatch a breakpoint operator. Methods: ls, maxslope, cusum, binseg, wbs."""
    if method == "ls":
        return ls_breakpoint(t, y)
    if method == "maxslope":
        return maxslope_breakpoint(t, y)
    if method == "cusum":
        return cusum_breakpoint(t, y)
    if method in ("binseg", "wbs", "pelt"):
        return multibreak(t, y, **kwargs)
    raise ValueError(f"unknown method {method!r}")
