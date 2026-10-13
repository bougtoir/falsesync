"""Shared helpers for the replication entry points."""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}


def run(*args):
    print("+", " ".join(str(a) for a in args), flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=ENV, check=True)
    print(f"  ({time.time() - t0:.0f} s)", flush=True)


def snapshot(names, dest):
    dest.mkdir(parents=True, exist_ok=True)
    for n in names:
        shutil.copy2(ROOT / n, dest / Path(n).name)


def _volatile(col):
    return "timestamp" in col or col == "run_utc"


def compare(names, ref_dir, atol=1e-9):
    """Compare regenerated CSVs with the reference copies; return mismatches."""
    bad = []
    for n in names:
        ref = pd.read_csv(ref_dir / Path(n).name)
        new = pd.read_csv(ROOT / n)
        ref = ref.drop(columns=[c for c in ref.columns if _volatile(c)])
        new = new.drop(columns=[c for c in new.columns if _volatile(c)])
        try:
            pd.testing.assert_frame_equal(ref, new, check_exact=False, atol=atol, rtol=1e-7)
            print(f"  match   {n}")
        except AssertionError as e:
            print(f"  DIFFERS {n}: {str(e).splitlines()[0]}")
            bad.append(n)
    return bad


def compare_png(names, ref_dir, max_frac=0.01):
    """Pixel comparison of regenerated figures with reference copies."""
    import numpy as np
    from matplotlib.image import imread
    bad = []
    for n in names:
        a = imread(ref_dir / Path(n).name)
        b = imread(ROOT / n)
        frac = float(np.mean(np.any(a != b, axis=-1))) if a.shape == b.shape else 1.0
        status = "match  " if frac <= max_frac else "DIFFERS"
        print(f"  {status} {n} (differing pixels: {frac:.4%})")
        if frac > max_frac:
            bad.append(n)
    return bad
