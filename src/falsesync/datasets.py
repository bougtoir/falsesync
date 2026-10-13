"""Loaders for acquired public engineering datasets.

Raw snapshots live under data/raw/ (see data/acquisition_ledger.csv).
Loaders never mutate raw files; they return tidy DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"


@dataclass
class PanelData:
    """Tidy panel: unit_id, t, value (+ optional weight/channel columns)."""

    frame: pd.DataFrame
    source: str
    notes: str = ""

    def to_unit_matrix(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """(t, values[n_units, n_t], unit_ids) wide form for diagnostics."""
        df = self.frame
        wide = df.pivot_table(index="unit_id", columns="t", values="value", aggfunc="mean")
        t = wide.columns.to_numpy(dtype=float)
        return t, wide.to_numpy(dtype=float), list(wide.index.astype(str))


def load_nasa_battery_cycles(data_dir: Path | None = None) -> PanelData:
    """NASA PCoE Li-ion capacity-vs-cycle per battery unit (discharge capacity)."""
    root = (data_dir or DATA_ROOT) / "nasa_battery"
    rows = []
    for f in sorted(root.rglob("*.csv")):
        df = pd.read_csv(f)
        cap_col = next((c for c in df.columns if c.lower() in ("capacity", "discharge_capacity")), None)
        cyc_col = next((c for c in df.columns if "cycle" in c.lower()), None)
        if cap_col is None:
            continue
        unit = f.parent.name + "/" + f.stem
        t_col = cyc_col or df.columns[0]
        for _, r in df.iterrows():
            rows.append({"unit_id": unit, "t": r[t_col], "value": r[cap_col]})
    return PanelData(pd.DataFrame(rows), "nasa_pcoe_battery")


def load_scada_power(data_dir: Path | None = None, dataset: str = "kelmarsh") -> PanelData:
    """Kelmarsh/Penmanshiel SCADA: power output per turbine over time."""
    root = (data_dir or DATA_ROOT) / dataset
    rows = []
    for f in sorted(root.rglob("*.csv")):
        df = pd.read_csv(f, low_memory=False)
        t_col = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), df.columns[0])
        p_col = next((c for c in df.columns if "power" in c.lower()), None)
        u_col = next((c for c in df.columns if "turbine" in c.lower() or "unit" in c.lower()), None)
        if p_col is None:
            continue
        uvals = df[u_col] if u_col else pd.Series(f.stem, index=df.index)
        tt = pd.to_datetime(df[t_col], errors="coerce")
        t_num = (tt - tt.min()).dt.total_seconds() / 86400.0
        for u, gdf in df.assign(_t=t_num, _u=uvals).groupby("_u"):
            for _, r in gdf.iterrows():
                rows.append({"unit_id": str(u), "t": r["_t"], "value": r[p_col]})
    return PanelData(pd.DataFrame(rows), f"{dataset}_scada")
