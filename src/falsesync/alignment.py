"""Event-time realignment (alignment in unit-specific transition time)."""

from __future__ import annotations

import numpy as np


def event_time_realign(
    t: np.ndarray, values: np.ndarray, taus: np.ndarray
) -> np.ndarray:
    """Re-express each unit series in event time s = t - tau_i.

    Returns an (n, 2k+1) array on a common event-time grid centered at 0;
    NaN where a unit's observed window does not cover s.
    """
    t = np.asarray(t, dtype=float)
    taus = np.asarray(taus, dtype=float)
    dt = float(np.median(np.diff(t)))
    half = float(np.minimum(taus - t.min(), t.max() - taus).max())
    k = int(half / dt)
    s_grid = np.arange(-k, k + 1) * dt
    out = np.full((values.shape[0], s_grid.size), np.nan)
    for i in range(values.shape[0]):
        src_t = s_grid + taus[i]
        valid = (src_t >= t.min()) & (src_t <= t.max())
        out[i, valid] = np.interp(src_t[valid], t, values[i])
    return out


def realign_summary(aligned: np.ndarray, s_dt: float) -> dict:
    """Aligned-panel summary for the diagnostic's event_aligned_summary."""
    agg = np.nanmean(aligned, axis=0)
    d = np.gradient(np.nan_to_num(agg, nan=np.nanmean(agg[np.isfinite(agg)])), s_dt)
    return {
        "aligned_max_slope": float(np.nanmax(np.abs(d))),
        "coverage": float(np.isfinite(aligned).mean()),
    }
