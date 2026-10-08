"""FalseSynchronyModel — Phase-3 research-grade workflow (spec §17).

    from falsesync.model import FalseSynchronyModel
    model = FalseSynchronyModel(classifier=clf)
    result = model.fit(t, values, weights=None, windows=None)

`classifier` is a fitted falsesync.regimes.RegimeClassifier. If None, the
result carries uncalibrated heuristic probabilities and a warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .aggregation import weighted_aggregate
from .alignment import event_time_realign, realign_summary
from .changepoints import aggregate_breakpoint, ls_breakpoint, multibreak
from .diagnostics import FalseSynchronyResult
from .features import extract_features
from .regimes import REGIME_CLASSES, RegimeClassifier


@dataclass
class FalseSynchronyModel:
    classifier: RegimeClassifier | None = None
    method: str = "ls"
    operators: tuple = ("ls", "maxslope", "cusum", "binseg")
    unit_boot: int = 30
    agg_strength_crit: float = 3.0  # aggregate break evidence criterion
    sync_prob_crit: float = 0.5     # calibrated synchrony-support threshold
    seed: int = 0

    def _trend_check(self, t, values, taus, rng) -> tuple[bool, dict]:
        """Detrend each unit (per-unit linear fit), refit breaks, compare.

        If unit break dispersion collapses after detrending, part of the
        apparent timing heterogeneity is trend confounding (§7).
        """
        t = np.asarray(t, dtype=float)
        tc = (t - t.mean()) / max(np.ptp(t), 1e-9)
        taus_d = []
        valid = np.isfinite(taus)
        for i in np.where(valid)[0]:
            y = values[i]
            ok = np.isfinite(y)
            if ok.sum() < 8:
                continue
            slope = np.polyfit(tc[ok], y[ok], 1)[0]
            yd = y.copy()
            yd[ok] = y[ok] - slope * tc[ok]
            fill = np.nanmean(yd[ok])
            try:
                r = ls_breakpoint(t, np.nan_to_num(yd, nan=fill))
                taus_d.append((i, r.location, abs(slope)))
            except Exception:
                pass
        if not taus_d:
            return False, {"detrended_sd": np.nan, "raw_sd": np.nan}
        idx = np.array([d[0] for d in taus_d], dtype=int)
        raw_sd = float(np.std(taus[idx]))
        det_sd = float(np.std([d[1] for d in taus_d]))
        med_slope = float(np.median([d[2] for d in taus_d]))
        info = {"raw_sd": raw_sd, "detrended_sd": det_sd, "median_slope": med_slope}
        # confounding: detrending materially shrinks spread AND slopes nonzero
        confounded = (raw_sd > 0 and det_sd < 0.6 * raw_sd and med_slope > 0.05)
        return confounded, info

    def _operator_spread(self, t, agg_y) -> dict:
        locs, strengths = {}, {}
        for m in self.operators:
            try:
                r = aggregate_breakpoint(t, agg_y, m) if m != "binseg" else multibreak(t, agg_y, 3)
                locs[m] = r.location
                strengths[m] = r.strength
            except Exception:
                pass
        vals = np.array(list(locs.values()))
        spread = float(vals.max() - vals.min()) if vals.size > 1 else 0.0
        consensus = float(1 - spread / max(np.ptp(t), 1e-9))
        return {
            "locations": locs, "strengths": strengths,
            "operator_spread": spread,
            "operator_consensus": consensus,
            "operator_sensitivity_warning": spread > 0.2 * np.ptp(t),
        }

    def fit(
        self,
        t: np.ndarray,
        values: np.ndarray,
        weights: np.ndarray | None = None,
        windows: np.ndarray | None = None,
        aggregate: np.ndarray | None = None,
    ) -> FalseSynchronyResult:
        rng = np.random.default_rng(self.seed)
        t = np.asarray(t, dtype=float)
        if aggregate is None:
            aggregate = weighted_aggregate(values, weights)
        if self.method == "ls":
            prim_loc, prim_str = None, None  # taken from aux below
        else:
            prim = (multibreak(t, aggregate, 3) if self.method == "binseg"
                    else aggregate_breakpoint(t, aggregate, self.method))
            prim_loc, prim_str = prim.location, prim.strength
        feats, aux = extract_features(
            t, values, weights=weights, windows=windows,
            aggregate=aggregate, rng=rng, unit_boot=self.unit_boot,
        )
        warnings: list[str] = []
        if self.classifier is not None and self.classifier.fitted:
            probs = self.classifier.predict_proba(feats)[0]
            regime_probabilities = dict(zip(REGIME_CLASSES, probs.tolist()))
        else:
            regime_probabilities = {k: np.nan for k in REGIME_CLASSES}
            warnings.append("no calibrated classifier: regime probabilities not estimable")

        ops = self._operator_spread(t, aux["aggregate"])
        if ops["operator_sensitivity_warning"]:
            warnings.append(
                f"operator spread {ops['operator_spread']:.3g} > 20% of window: "
                "no method-independent aggregate breakpoint (P7); report operator."
            )

        indeterminate = False
        obs_corr = np.nan
        if windows is not None:
            win = np.asarray(windows, dtype=float)
            if win.ndim == 2 and win.shape[1] == 2:
                lo = np.nanmin(t)
                trunc_start = win[:, 0] > lo + 0.02 * max(np.ptp(t), 1e-9)
                m_ok = trunc_start & np.isfinite(aux["taus"])
                if m_ok.sum() >= 5 and trunc_start.mean() >= 0.2:
                    obs_corr = float(np.corrcoef(win[m_ok, 0], aux["taus"][m_ok])[0, 1])
                    indeterminate = obs_corr >= 0.5

        confounded, trend_info = self._trend_check(t, values, aux["taus"], rng)
        if confounded:
            warnings.append(
                "trend confounding: detrending shrinks tau_hat dispersion "
                f"{trend_info['raw_sd']:.3g} -> {trend_info['detrended_sd']:.3g}; "
                "timing inference unreliable until trends modeled (§7)."
            )

        if indeterminate:
            warnings.append(
                "indeterminate observation process: unit entry times correlate "
                f"with estimated break times (corr={obs_corr:.2f}); synchrony "
                "inference is not supported under a possibly informative "
                "observation process (Phase-4 F1 mitigation)."
            )
        agg_loc = aux["agg_break"] if self.method == "ls" else prim_loc
        agg_str = aux["agg_strength"] if self.method == "ls" else prim_str
        p_sync = regime_probabilities.get("SYNCHRONOUS", np.nan)
        warn_fire = (
            np.isfinite(agg_str) and agg_str >= self.agg_strength_crit
            and np.isfinite(p_sync) and p_sync < self.sync_prob_crit
        ) or indeterminate
        if warn_fire:
            warnings.append(
                f"false common-event warning: aggregate strength "
                f"{agg_str:.2f} >= {self.agg_strength_crit} but "
                f"P(SYNCHRONOUS)={p_sync:.2f} < {self.sync_prob_crit}."
            )

        tv = aux["taus"][np.isfinite(aux["taus"])]
        grid, f_dec = aux["f_dec"]
        comps = {
            "s1_width": feats[4], "s2_shape": feats[2],
            "s3_alignment": feats[5], "s4_weighting": feats[6],
            "noise_floor_ratio": feats[7],
        }
        scale = np.ptp(t)
        a_sync = p_sync if np.isfinite(p_sync) else np.nan
        synchrony_score = a_sync  # calibrated: score IS P(synchrony)

        aligned = event_time_realign(t, values[np.isfinite(aux["taus"])], tv) if tv.size else np.zeros((0, 0))
        s_dt = float(np.median(np.diff(t)))
        ea = realign_summary(aligned, s_dt) if aligned.size else {}
        ea.update({"n_aligned": int(tv.size),
                   "tau_spread_before": float(np.std(tv)) if tv.size else np.nan,
                   "est_noise_floor": aux["noise_floor"],
                   "trend_confounded": confounded, "trend_info": trend_info})

        timing_dispersion = {
            "sd": float(np.std(tv)) if tv.size else np.nan,
            "iqr": float(np.quantile(tv, 0.75) - np.quantile(tv, 0.25)) if tv.size else np.nan,
            "sd_corrected": float(np.sqrt(max(np.std(tv) ** 2 - (aux["noise_floor"] or 0) ** 2, 0)))
            if tv.size else np.nan,
        }
        cluster_structure = {
            "n_components": aux["n_modes"],
            "centers": [float(grid[i]) for i in np.argsort(f_dec)[::-1][: aux["n_modes"]]] if f_dec.size else [],
        }
        return FalseSynchronyResult(
            aggregate_breakpoint=agg_loc,
            aggregate_break_strength=agg_str,
            unit_breakpoints=aux["taus"],
            unit_breakpoint_uncertainty=aux["s_i"],
            timing_distribution={"grid": grid, "density": aux["f_hat"][1], "deconvolved": f_dec},
            timing_dispersion=timing_dispersion,
            cluster_structure=cluster_structure,
            synchrony_score=synchrony_score,
            regime_probabilities=regime_probabilities,
            event_aligned_summary=ea,
            interpretation_warning=warnings,
            method=self.method,
            components={
                **comps,
                "operator_locations": ops["locations"],
                "operator_strengths": ops["strengths"],
                "operator_spread": ops["operator_spread"],
                "operator_consensus": ops["operator_consensus"],
                "operator_sensitivity_warning": ops["operator_sensitivity_warning"],
                "trend_confounding_warning": confounded,
                "indeterminate_observation_process": indeterminate,
                "obs_process_corr": obs_corr,
                "false_common_event_warning": warn_fire,
            },
        )
