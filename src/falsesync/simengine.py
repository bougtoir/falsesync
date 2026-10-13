"""Phase-3 config-driven simulation engine (see simulation_design.md).

A SimConfig fully specifies one simulation cell; simulate() returns a Panel
plus ground-truth regime/timing metadata. All randomness flows through one
seeded numpy Generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import t as t_dist

from .aggregation import weighted_aggregate
from .simulation import Panel, sample_taus, simulate_panel
from .transitions import g

REGIMES = (
    "synchronous",
    "near_synchronous",
    "diffuse",
    "two_cluster",
    "three_cluster",
    "no_transition",
)

F_FAMILIES = (
    "degenerate", "narrow_normal", "broad_normal", "uniform", "skewed",
    "two_normal_mix", "three_normal_mix", "spike_diffuse", "heavy_tail",
)

NOISE_KINDS = ("iid", "t3", "heterosk", "ar1", "seasonal")
WEIGHT_KINDS = ("equal", "static", "timing_corr", "timevarying")
MISS_KINDS = ("complete", "mcar", "dropout", "late_entry", "informative")


@dataclass
class SimConfig:
    n_units: int = 100
    n_t: int = 120
    t_span: tuple = (0.0, 10.0)
    regime: str = "diffuse"
    f_family: str = "broad_normal"
    f_params: dict = field(default_factory=dict)  # overrides, e.g. {"sigma":0.8}
    center: float = 5.0  # central transition time
    shape: str = "logistic"  # step/logistic/probit/linear
    amplitude: float = 1.0  # effect size (low 0.5, med 1.0, high 2.0 typical)
    width: float = 0.4  # transition width h
    noise_kind: str = "iid"
    noise_sd: float = 0.15
    noise_params: dict = field(default_factory=dict)
    weight_kind: str = "equal"
    weight_params: dict = field(default_factory=dict)
    miss_kind: str = "complete"
    miss_params: dict = field(default_factory=dict)
    amp_hetero: float = 0.0  # sd of A_i / A
    width_hetero: float = 0.0  # sd of h_i / h
    base_hetero: float = 0.0  # sd of a_i
    trend_kind: str = "none"  # none/linear/nonlinear/local_drift/mixed
    trend_scale: float = 0.0  # sd of trend slope per unit
    noisevar_hetero: float = 0.0
    seed: int = 0
    cell_id: str = ""


def f_tau_spec(cfg: SimConfig) -> dict:
    """Translate (regime, f_family, f_params) into a sample_taus spec."""
    c = cfg.center
    p = dict(cfg.f_params)
    if cfg.regime == "no_transition":
        return {"kind": "degenerate_at", "center": c}  # unused
    fam = cfg.f_family
    if fam == "degenerate" or cfg.regime == "synchronous":
        sd = p.get("sigma", 1e-6)
        return {"kind": "normal", "mu": c, "sigma": sd}
    if fam == "narrow_normal":
        return {"kind": "normal", "mu": c, "sigma": p.get("sigma", 0.2)}
    if fam == "broad_normal":
        return {"kind": "normal", "mu": c, "sigma": p.get("sigma", 0.9)}
    if fam == "uniform":
        lo, hi = p.get("lo", c - 2.0), p.get("hi", c + 2.0)
        return {"kind": "uniform", "lo": lo, "hi": hi}
    if fam == "skewed":
        return {"kind": "gamma", "shape": p.get("shape", 3.0),
                "scale": p.get("scale", 0.5), "shift": c - 1.5}
    if fam == "two_normal_mix":
        d = p.get("delta", 2.5)
        return {"kind": "mixture", "pi": p.get("pi", 0.5), "mu1": c - d / 2,
                "mu2": c + d / 2, "sigma": p.get("sigma", 0.4)}
    if fam == "three_normal_mix":
        d = p.get("delta", 2.0)
        return {"kind": "three_mix", "mu1": c - d, "mu2": c, "mu3": c + d,
                "sigma": p.get("sigma", 0.35),
                "pis": p.get("pis", [0.33, 0.34, 0.33])}
    if fam == "spike_diffuse":
        return {"kind": "spike", "center": c, "eps": p.get("eps", 0.4),
                "lo": c - 2.5, "hi": c + 2.5}
    if fam == "heavy_tail":
        return {"kind": "student_t", "df": p.get("df", 3.0), "mu": c,
                "sigma": p.get("sigma", 0.5)}
    raise ValueError(f"unknown f_family {fam!r}")


def sample_taus_cfg(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    spec = f_tau_spec(cfg)
    kind = spec["kind"]
    if kind == "degenerate_at":
        return np.full(cfg.n_units, cfg.center)
    if kind == "three_mix":
        pis = np.asarray(spec["pis"], dtype=float)
        comp = rng.choice(3, cfg.n_units, p=pis / pis.sum())
        mus = np.array([spec["mu1"], spec["mu2"], spec["mu3"]])
        return mus[comp] + rng.normal(0, spec["sigma"], cfg.n_units)
    if kind == "student_t":
        return spec["mu"] + spec["sigma"] * t_dist.rvs(spec["df"], size=cfg.n_units, random_state=rng)
    return sample_taus(cfg.n_units, spec, rng)


def regime_truth(cfg: SimConfig) -> str:
    if cfg.regime in REGIMES:
        return cfg.regime
    return cfg.regime


def make_weights(cfg: SimConfig, taus: np.ndarray, t: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
    """(n_units, n_t) observation-independent weight matrix."""
    n = taus.size
    kind = cfg.weight_kind
    p = cfg.weight_params
    if kind == "equal":
        return np.ones((n, t.size))
    if kind == "static":
        w = rng.lognormal(0, p.get("sd", 0.8), n)
        return np.tile(w / w.mean(), (t.size, 1)).T
    if kind == "timing_corr":
        kappa = p.get("kappa", 1.0)
        w = np.exp(-kappa * (taus - taus.mean()) / max(taus.std(), 1e-9))
        return np.tile(w / w.mean(), (t.size, 1)).T
    if kind == "timevarying":
        base = rng.lognormal(0, p.get("sd", 0.5), n)
        ramp = np.exp(p.get("slope", 0.3) * (t[None, :] - t.mean()) / np.ptp(t) * taus[:, None])
        w = base[:, None] * ramp
        return w / w.mean(axis=0, keepdims=True)
    raise ValueError(f"unknown weight_kind {kind!r}")


def make_windows(cfg: SimConfig, taus: np.ndarray, t: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
    """(n_units, 2) [L,U] windows; NaN marks unobserved in simulate."""
    n = taus.size
    lo, hi = t.min(), t.max()
    win = np.tile([lo, hi], (n, 1)).astype(float)
    kind = cfg.miss_kind
    p = cfg.miss_params
    if kind == "complete":
        return win
    if kind == "mcar":
        rate = p.get("rate", 0.1)
        # handled as point-wise missing in simulate()
        return win
    if kind == "dropout":
        rate = p.get("rate", 0.3)
        drop = rng.random(n) < rate
        win[drop, 1] = np.sort(t)[rng.integers(len(t) // 3, len(t) - 1, drop.sum())]
        return win
    if kind == "late_entry":
        rate = p.get("rate", 0.3)
        late = rng.random(n) < rate
        win[late, 0] = np.sort(t)[rng.integers(1, len(t) // 2, late.sum())]
        return win
    if kind == "informative":
        kappa = p.get("kappa", 1.0)
        # entry time correlated with tau_i: later-transitioning units enter later
        win[:, 0] = np.clip(taus - p.get("lead", 1.0) + rng.normal(0, 0.3, n), lo, hi)
        return win
    raise ValueError(f"unknown miss_kind {kind!r}")


def add_trend(cfg: SimConfig, values: np.ndarray, t: np.ndarray,
              rng: np.random.Generator) -> np.ndarray:
    """Add b_i(t) heterogeneity to panel values (in-place safe copy)."""
    v = values.copy()
    n = v.shape[0]
    s = cfg.trend_scale
    if cfg.trend_kind == "none" or s == 0:
        return v
    tc = (t - t.mean()) / np.ptp(t)
    if cfg.trend_kind == "linear":
        slopes = rng.normal(0, s, n)
        v += slopes[:, None] * tc[None, :]
    elif cfg.trend_kind == "nonlinear":
        for i in range(n):
            knots = rng.normal(0, s, 3)
            v[i] += knots[0] * tc + knots[1] * tc**2 * 4 + knots[2] * np.sin(2 * np.pi * tc)
    elif cfg.trend_kind == "local_drift":
        for i in range(n):
            onset = rng.uniform(t.min(), t.max())
            drift = np.where(t >= onset, rng.normal(0, s), 0.0)
            v[i] += drift * (t - onset)
    elif cfg.trend_kind == "mixed":
        slopes = rng.normal(0, s, n)
        wobble = rng.normal(0, s * 0.5, n)[:, None] * np.sin(4 * np.pi * tc)[None, :]
        v += slopes[:, None] * tc[None, :] + wobble
    else:
        raise ValueError(f"unknown trend_kind {cfg.trend_kind!r}")
    return v


def add_noise(cfg: SimConfig, values: np.ndarray, t: np.ndarray,
              rng: np.random.Generator) -> np.ndarray:
    n, nt = values.shape
    sd = cfg.noise_sd
    p = cfg.noise_params
    if sd == 0:
        return values
    if cfg.noise_kind == "iid":
        e = rng.normal(0, sd, (n, nt))
    elif cfg.noise_kind == "t3":
        e = sd * t_dist.rvs(3, size=(n, nt), random_state=rng)
    elif cfg.noise_kind == "heterosk":
        e = rng.normal(0, sd * (0.5 + t[None, :] / np.ptp(t)), (n, nt))
    elif cfg.noise_kind == "ar1":
        rho = p.get("rho", 0.7)
        e = np.zeros((n, nt))
        eps = rng.normal(0, sd, (n, nt))
        e[:, 0] = eps[:, 0]
        for k in range(1, nt):
            e[:, k] = rho * e[:, k - 1] + eps[:, k]
    elif cfg.noise_kind == "seasonal":
        e = rng.normal(0, sd, (n, nt)) + sd * 0.8 * np.sin(2 * np.pi * p.get("period", 0.4) * t[None, :])
    else:
        raise ValueError(cfg.noise_kind)
    if cfg.noisevar_hetero > 0:
        e = e * rng.lognormal(0, cfg.noisevar_hetero, n)[:, None]
    return values + e


def simulate(cfg: SimConfig) -> tuple[Panel, dict]:
    """Full cell simulation -> (Panel, truth dict)."""
    rng = np.random.default_rng(cfg.seed)
    t = np.linspace(cfg.t_span[0], cfg.t_span[1], cfg.n_t)
    taus = sample_taus_cfg(cfg, rng)
    n = cfg.n_units
    A = cfg.amplitude * rng.lognormal(0, cfg.amp_hetero, n)
    h = cfg.width * rng.lognormal(0, cfg.width_hetero, n)
    a = rng.normal(0, cfg.base_hetero, n)
    if cfg.regime == "no_transition":
        A = np.zeros(n)
    u = (t[None, :] - taus[:, None]) / h[:, None]
    values = a[:, None] + A[:, None] * g(u, cfg.shape)
    values = add_trend(cfg, values, t, rng)
    values = add_noise(cfg, values, t, rng)
    win = make_windows(cfg, taus, t, rng)
    obs = (t[None, :] >= win[:, 0:1]) & (t[None, :] <= win[:, 1:2])
    values = np.where(obs, values, np.nan)
    if cfg.miss_kind == "mcar":
        drop = rng.random(values.shape) < cfg.miss_params.get("rate", 0.1)
        values = np.where(drop, np.nan, values)
    w = make_weights(cfg, taus, t, rng)
    w_t = np.where(np.isfinite(values), w, np.nan)
    agg = np.nansum(values * w_t, axis=0) / np.maximum(np.nansum(w_t, axis=0), 1e-12)
    panel = Panel(
        t=t, values=values, taus=taus, weights=w.mean(axis=1), windows=win,
        shape=cfg.shape, amplitudes=A,
        meta={"baseline": a, "width": h, "config": cfg},
    )
    truth = {
        "regime": regime_truth(cfg),
        "tau_sd": float(np.std(taus)),
        "tau_iqr": float(np.quantile(taus, 0.75) - np.quantile(taus, 0.25)),
        "taus": taus,
        "aggregate": agg,
        "cell_id": cfg.cell_id,
    }
    return panel, truth
