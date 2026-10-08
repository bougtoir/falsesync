"""Grid runner: enumerate cells from a YAML config, simulate, extract features.

Usage:
  PYTHONPATH=src python3 simulations/run_grid.py <config.yaml> <out_csv> [--quick]

Outputs per-cell CSV rows with config fields, truth, and feature values.
"""

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from falsesync.features import FEATURE_NAMES, extract_features
from falsesync.regimes import regime_label
from falsesync.simengine import SimConfig, simulate


def regime_family_pairs(cfg) -> list:
    out = []
    for regime, fams in cfg["regimes"].items():
        for fam in fams:
            out.append((regime, fam))
    return out


def base_cells(cfg) -> list[SimConfig]:
    """Cartesian product, thinned to max_cells by seeded subsample."""
    pairs = regime_family_pairs(cfg)
    cells = []
    for (regime, fam), n_u, n_t, shape, amp, wdt in itertools.product(
        pairs, cfg["n_units"], cfg["n_t"], cfg["shapes"],
        cfg["amplitudes"], cfg["widths"],
    ):
        for s in range(cfg.get("seeds_per_cell", 1)):
            cells.append(dict(
                n_units=n_u, n_t=n_t, regime=regime, f_family=fam,
                shape=shape, amplitude=amp, width=wdt, rep=s,
            ))
    rng = np.random.default_rng(cfg["master_seed"])
    max_cells = cfg.get("max_cells", len(cells))
    if len(cells) > max_cells:
        keep = rng.choice(len(cells), max_cells, replace=False)
        cells = [cells[i] for i in sorted(keep)]
    return cells


def stress_cells(cfg) -> list[tuple[str, SimConfig]]:
    """One-dim-at-a-time blocks around a moderate base cell."""
    stress = cfg.get("stress", {})
    out = []
    base_regimes = ["synchronous", "diffuse", "two_cluster"]
    fam_map = {"synchronous": "degenerate", "diffuse": "broad_normal",
               "two_cluster": "two_normal_mix"}
    reps = cfg.get("seeds_per_cell", 1)
    for block, variants in stress.items():
        for v in variants:
            for regime in base_regimes:
                for s in range(reps):
                    d = dict(
                        n_units=60, n_t=120, regime=regime,
                        f_family=fam_map[regime], shape="logistic",
                        amplitude=1.0, width=0.4, rep=s,
                        noise_kind=cfg["noise"]["kind"], noise_sd=cfg["noise"]["sd"],
                    )
                    if block == "noise":
                        d["noise_kind"] = v["kind"]
                        d["noise_sd"] = v.get("sd", 0.15)
                        d["noise_params"] = v.get("params", {})
                    elif block == "weights":
                        d["weight_kind"] = v["kind"]
                        d["weight_params"] = v.get("params", {})
                    elif block == "missing":
                        d["miss_kind"] = v["kind"]
                        d["miss_params"] = v.get("params", {})
                    elif block in ("hetero", "trend"):
                        d.update(
                            amp_hetero=v["amp"], width_hetero=v["width"],
                            base_hetero=v["base"], trend_kind=v["trend_kind"],
                            trend_scale=v["trend_scale"], noisevar_hetero=v["noisevar"],
                        )
                    out.append((block, d))
    return out


def cell_to_config(d: dict, cfg: dict, cell_id: str) -> SimConfig:
    fpo = cfg.get("f_params_override", {})
    f_params = fpo.get(d["f_family"], {})
    n = cfg["noise"]
    h = cfg["hetero"]
    return SimConfig(
        n_units=d["n_units"], n_t=d["n_t"], regime=d["regime"],
        f_family=d["f_family"], f_params=f_params, shape=d["shape"],
        amplitude=d["amplitude"], width=d["width"],
        noise_kind=d.get("noise_kind", n["kind"]),
        noise_sd=d.get("noise_sd", n["sd"]),
        noise_params=d.get("noise_params", n.get("params", {})),
        weight_kind=d.get("weight_kind", cfg["weights"]["kind"]),
        weight_params=d.get("weight_params", cfg["weights"].get("params", {})),
        miss_kind=d.get("miss_kind", cfg["missing"]["kind"]),
        miss_params=d.get("miss_params", cfg["missing"].get("params", {})),
        amp_hetero=d.get("amp_hetero", h["amp"]),
        width_hetero=d.get("width_hetero", h["width"]),
        base_hetero=d.get("base_hetero", h["base"]),
        trend_kind=d.get("trend_kind", h["trend_kind"]),
        trend_scale=d.get("trend_scale", h["trend_scale"]),
        noisevar_hetero=d.get("noisevar_hetero", h["noisevar"]),
        seed=d.get("seed", 0), cell_id=cell_id,
    )


def run_grid(config_path: str, out_csv: str, quick: bool = False,
             unit_boot: int = 15) -> pd.DataFrame:
    cfg = yaml.safe_load(open(config_path))
    if quick:
        cfg["max_cells"] = min(cfg.get("max_cells", 200), 120)
        cfg["seeds_per_cell"] = 1
        unit_boot = 10
    cells = base_cells(cfg)
    stress = stress_cells(cfg)
    rng = np.random.default_rng(cfg["master_seed"])
    rows = []
    t0 = time.time()

    def run_one(d, block, idx):
        d = dict(d)
        d["seed"] = int(cfg["master_seed"] + idx * 7919 + d.get("rep", 0) * 131)
        sc = cell_to_config(d, cfg, f"{cfg['name']}-{block}-{idx}")
        panel, truth = simulate(sc)
        feats, aux = extract_features(
            panel.t, panel.values, weights=panel.weights,
            windows=panel.windows, aggregate=truth["aggregate"],
            rng=np.random.default_rng(d["seed"]), unit_boot=unit_boot,
        )
        row = {
            "cell_id": sc.cell_id, "block": block,
            "n_units": sc.n_units, "n_t": sc.n_t, "regime": sc.regime,
            "f_family": sc.f_family, "shape": sc.shape,
            "amplitude": sc.amplitude, "width": sc.width,
            "noise_kind": sc.noise_kind, "noise_sd": sc.noise_sd,
            "weight_kind": sc.weight_kind, "miss_kind": sc.miss_kind,
            "trend_kind": sc.trend_kind, "trend_scale": sc.trend_scale,
            "t_span": sc.t_span[1] - sc.t_span[0],
            "seed": sc.seed,
            "label": regime_label(sc.regime, truth["taus"]),
            "tau_sd_true": truth["tau_sd"],
            "agg_break": aux["agg_break"], "agg_strength": aux["agg_strength"],
            "noise_floor": aux["noise_floor"],
        }
        row.update({f"f_{n}": v for n, v in zip(FEATURE_NAMES, feats)})
        return row

    for i, d in enumerate(cells):
        try:
            rows.append(run_one(d, "base", i))
        except Exception as e:
            rows.append({"cell_id": f"fail-{i}", "block": "base", "label": "ERROR",
                         "error": repr(e)})
        if (i + 1) % 50 == 0:
            print(f"{cfg['name']}: {i+1}/{len(cells)} base cells, {time.time()-t0:.0f}s", flush=True)
    for i, (block, d) in enumerate(stress):
        try:
            rows.append(run_one(d, f"stress_{block}", i))
        except Exception as e:
            rows.append({"cell_id": f"fail-stress-{i}", "block": f"stress_{block}",
                         "label": "ERROR", "error": repr(e)})
    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"wrote {len(df)} rows -> {out_csv} in {time.time()-t0:.0f}s")
    return df


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run_grid(args[0], args[1], quick=quick)
