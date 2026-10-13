"""False-synchrony diagnostic (see diagnostic_specification.md)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .aggregation import weighted_aggregate
from .changepoints import BreakResult, aggregate_breakpoint
from .inference import deconvolve_timing, unit_break_uncertainties


@dataclass
class FalseSynchronyResult:
    aggregate_breakpoint: float
    aggregate_break_strength: float
    unit_breakpoints: np.ndarray
    unit_breakpoint_uncertainty: np.ndarray
    timing_distribution: dict  # {"grid": t, "density": f_hat, "deconvolved": f_dec}
    timing_dispersion: dict  # {"sd": ..., "iqr": ..., "ci": (lo, hi)}
    cluster_structure: dict  # {"n_components": k, "centers": [...], "weights": [...]}
    synchrony_score: float
    regime_probabilities: dict  # keys: SYNCHRONOUS/CLUSTERED/DIFFUSE_ASYNCHRONOUS/NO_TRANSITION
    event_aligned_summary: dict
    interpretation_warning: list[str]
    method: str = "ls"
    aggregate_break_result: BreakResult | None = None
    components: dict = field(default_factory=dict)


def _kde(samples: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    u = (grid[:, None] - samples[None, :]) / bw
    k = np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)
    return k.mean(axis=1) / bw


def _fit_unit_breaks(
    t: np.ndarray, values: np.ndarray, method: str = "ls"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-unit LS break: (tau_hat, strength, mean_l->r jump)."""
    n = values.shape[0]
    taus = np.full(n, np.nan)
    strengths = np.full(n, np.nan)
    idxs = np.full(n, -1, dtype=int)
    for i in range(n):
        y = values[i]
        if np.isfinite(y).sum() < 8:
            continue
        try:
            res = aggregate_breakpoint(t, np.nan_to_num(y, nan=np.nanmean(y[np.isfinite(y)])), method)
            taus[i] = res.location
            strengths[i] = res.strength
            idxs[i] = res.index
        except Exception:
            continue
    return taus, strengths, idxs


def false_synchrony_diagnostic(
    t: np.ndarray,
    aggregate_series: np.ndarray | None,
    unit_values: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    windows: np.ndarray | None = None,
    method: str = "ls",
    rng: np.random.Generator | None = None,
    n_boot: int = 200,
) -> FalseSynchronyResult:
    """Main diagnostic entry point (spec §9/§10).

    If unit_values is None, returns an aggregate-only result whose
    interpretation_warning states synchrony cannot be certified (P6c).
    """
    rng = rng or np.random.default_rng(0)
    t = np.asarray(t, dtype=float)
    warnings: list[str] = []
    if aggregate_series is None and unit_values is None:
        raise ValueError("need aggregate_series or unit_values")
    if aggregate_series is None:
        aggregate_series = weighted_aggregate(unit_values, weights)
    agg_res = aggregate_breakpoint(t, aggregate_series, method)
    agg_break, agg_strength = agg_res.location, agg_res.strength

    if unit_values is None:
        return FalseSynchronyResult(
            aggregate_breakpoint=agg_break,
            aggregate_break_strength=agg_strength,
            unit_breakpoints=np.array([]),
            unit_breakpoint_uncertainty=np.array([]),
            timing_distribution={"grid": t, "density": np.full(t.size, np.nan)},
            timing_dispersion={"sd": np.nan, "iqr": np.nan, "ci": (np.nan, np.nan)},
            cluster_structure={"n_components": 0, "centers": [], "weights": []},
            synchrony_score=np.nan,
            regime_probabilities={
                "SYNCHRONOUS": np.nan, "CLUSTERED": np.nan,
                "DIFFUSE_ASYNCHRONOUS": np.nan, "NO_TRANSITION": np.nan,
            },
            event_aligned_summary={},
            interpretation_warning=[
                "aggregate-only input: synchrony cannot be certified from the "
                "aggregate curve alone (P6c); unit-level series required.",
                "regime probabilities are not estimable without unit data.",
            ],
            method=method,
            aggregate_break_result=agg_res,
        )

    taus, strengths, _ = _fit_unit_breaks(t, unit_values, method)
    valid = np.isfinite(taus)
    taus_v = taus[valid]
    if taus_v.size == 0:
        warnings.append(
            "no estimable unit breakpoints (insufficient observations or "
            "failed fits): synchrony cannot be certified; unit-level "
            "fields are not estimable."
        )
        return FalseSynchronyResult(
            aggregate_breakpoint=agg_break,
            aggregate_break_strength=agg_strength,
            unit_breakpoints=taus,
            unit_breakpoint_uncertainty=np.full(taus.size, np.nan),
            timing_distribution={"grid": t, "density": np.full(t.size, np.nan)},
            timing_dispersion={"sd": np.nan, "iqr": np.nan, "ci": (np.nan, np.nan)},
            cluster_structure={"n_components": 0, "centers": [], "weights": []},
            synchrony_score=np.nan,
            regime_probabilities={
                "SYNCHRONOUS": np.nan, "CLUSTERED": np.nan,
                "DIFFUSE_ASYNCHRONOUS": np.nan, "NO_TRANSITION": np.nan,
            },
            event_aligned_summary={},
            interpretation_warning=warnings,
            method=method,
            aggregate_break_result=agg_res,
        )
    w_v = weights[valid] if weights is not None else None
    s_i = unit_break_uncertainties(t, unit_values[valid], taus_v, rng=rng, n_boot=60)

    grid = np.linspace(t.min(), t.max(), 400)
    bw = max(1.06 * np.std(taus_v) * taus_v.size ** -0.2, (t[1] - t[0]))
    f_hat = _kde(taus_v, grid, bw)
    f_dec = deconvolve_timing(taus_v, s_i, grid)
    sd = float(np.std(taus_v))
    q = np.quantile(taus_v, [0.25, 0.75])
    iqr = float(q[1] - q[0])

    boot_sd = [
        float(np.std(rng.choice(taus_v, taus_v.size, replace=True)))
        for _ in range(n_boot)
    ]
    sd_ci = tuple(np.quantile(boot_sd, [0.025, 0.975]).tolist())

    # s1: timing spread vs aggregate transition width (FWHM of m')
    dm = np.gradient(np.nan_to_num(aggregate_series, nan=np.nanmean(aggregate_series[np.isfinite(aggregate_series)])), t)
    half = np.max(dm) / 2
    fwhm_idx = np.where(dm >= half)[0]
    w_agg = float(t[fwhm_idx[-1]] - t[fwhm_idx[0]]) if fwhm_idx.size > 1 else np.inf
    s1_width = sd / w_agg if np.isfinite(w_agg) and w_agg > 0 else np.nan

    # s2: mass concentration of deconvolved density (peak share vs uniform)
    f_dec_n = np.clip(f_dec, 0, None)
    f_dec_n = f_dec_n / np.trapz(f_dec_n, grid) if np.trapz(f_dec_n, grid) > 0 else f_dec_n
    peak_share = float(np.max(f_dec_n) * (grid[1] - grid[0]) * 20)
    s2_shape = min(peak_share, 1.0)

    # s3: unit break alignment — fraction of units with a detectable break
    detectable = np.isfinite(strengths) & (np.abs(strengths) > 2.0)
    s3_alignment = float(detectable[valid].mean()) if valid.any() else np.nan

    # s4: weight-timing correlation + window-timing correlation
    s4_parts = []
    if w_v is not None:
        c_w = float(np.corrcoef(w_v, taus_v)[0, 1])
        s4_parts.append(("weight_timing_corr", c_w))
        if abs(c_w) > 0.3:
            warnings.append(
                f"weight-timing correlation {c_w:+.2f}: aggregate breakpoint is "
                "tilted toward heavily weighted units' timing (P4)."
            )
    if windows is not None:
        wv = np.asarray(windows)[valid]
        span = wv[:, 1] - wv[:, 0]
        c_win = float(np.corrcoef(span, taus_v)[0, 1])
        s4_parts.append(("window_timing_corr", c_win))
        if abs(c_win) > 0.3:
            warnings.append(
                f"window-timing correlation {c_win:+.2f}: observed aggregate is "
                "a convolution against a time-varying observed timing law (P5)."
            )
    s4_weighting = float(np.mean([abs(c) for _, c in s4_parts])) if s4_parts else 0.0

    # regime probabilities (simple distance-based classifier; calibrated in Phase 3)
    noise_floor = float(np.median(s_i)) if s_i.size else np.nan
    excess = sd**2 - noise_floor**2 if np.isfinite(noise_floor) else sd**2
    p_sync = float(np.clip(1 - sd / max(2 * noise_floor, 1e-9), 0, 1)) if np.isfinite(noise_floor) else np.nan
    # crude mixture check via dip in kde between two modes
    modes = _local_maxima(f_dec_n)
    p_cluster = float(len(modes) >= 2) * 0.7 + (0.3 if excess > noise_floor**2 else 0.0)
    p_diffuse = float(np.clip(sd / (0.5 * (t.max() - t.min())), 0, 1))
    p_notrans = float(np.clip(1 - abs(agg_strength) / 5, 0, 1)) if np.isfinite(agg_strength) else np.nan
    probs = np.array([max(p_sync, 0.01), p_cluster, p_diffuse * 0.5, p_notrans * 0.2])
    probs = probs / probs.sum()
    regime_probabilities = dict(zip(
        ["SYNCHRONOUS", "CLUSTERED", "DIFFUSE_ASYNCHRONOUS", "NO_TRANSITION"],
        [float(p) for p in probs],
    ))

    components = {
        "s1_width": s1_width,
        "s2_shape": s2_shape,
        "s3_alignment": s3_alignment,
        "s4_weighting": s4_weighting,
    }
    finite = [v for v in components.values() if np.isfinite(v)]
    synchrony_score = float(np.clip(1 - np.mean(finite), 0, 1)) if finite else np.nan

    if excess <= 0:
        warnings.append(
            "unit break-time spread is within the unit-level estimation-noise "
            "floor; observed dispersion may be entirely measurement (CE-UNITERR)."
        )
    warnings.append(
        f"LS breakpoint {agg_break:.3g} is window-dependent (P7): "
        "do not interpret as mean/median/mode of F_tau without the FOC check."
    )

    from .alignment import event_time_realign

    aligned = event_time_realign(t, unit_values[valid], taus_v)
    event_aligned_summary = {
        "n_aligned": int(valid.sum()),
        "aligned_agg_sharpness": float(np.nanmax(np.abs(np.gradient(
            np.nanmean(aligned, axis=0)[np.isfinite(np.nanmean(aligned, axis=0))]
        )))) if valid.any() else np.nan,
        "tau_spread_before": sd,
        "est_noise_floor": noise_floor,
    }

    centers = [float(grid[m]) for m in modes]
    cluster_structure = {
        "n_components": len(modes),
        "centers": centers,
        "weights": [float(f_dec_n[m]) for m in modes],
    }
    return FalseSynchronyResult(
        aggregate_breakpoint=agg_break,
        aggregate_break_strength=agg_strength,
        unit_breakpoints=taus,
        unit_breakpoint_uncertainty=s_i,
        timing_distribution={"grid": grid, "density": f_hat, "deconvolved": f_dec_n},
        timing_dispersion={"sd": sd, "iqr": iqr, "ci": sd_ci},
        cluster_structure=cluster_structure,
        synchrony_score=synchrony_score,
        regime_probabilities=regime_probabilities,
        event_aligned_summary=event_aligned_summary,
        interpretation_warning=warnings,
        method=method,
        aggregate_break_result=agg_res,
        components=components,
    )


def _local_maxima(f: np.ndarray) -> list[int]:
    out = [i for i in range(1, len(f) - 1) if f[i] >= f[i - 1] and f[i] >= f[i + 1]]
    return [i for i in out if f[i] > 0.1 * f.max()] if f.size else []
