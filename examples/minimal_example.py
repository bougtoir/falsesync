"""Minimal falsesync workflow on a simulated asynchronous panel.

Run from the repository root:  python examples/minimal_example.py
"""
from pathlib import Path

import joblib
import numpy as np

from falsesync import simulation
from falsesync.model import FalseSynchronyModel

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(1)

t = np.linspace(0.0, 10.0, 180)
taus = simulation.sample_taus(80, {"kind": "normal", "mu": 5.0, "sigma": 1.1}, rng)
panel = simulation.simulate_panel(t, taus, shape="logistic", amplitude=1.5,
                                  width=0.3, noise_sd=0.2, rng=rng)

clf = joblib.load(ROOT / "simulations" / "calibration" / "classifier.joblib")
res = FalseSynchronyModel(classifier=clf).fit(t, panel.values)

print(f"aggregate breakpoint      : {res.aggregate_breakpoint:.3f} "
      f"(strength {res.aggregate_break_strength:.2f})")
print(f"corrected timing SD       : {res.timing_dispersion.get('sd_corrected', np.nan):.3f}")
print(f"true timing SD (simulated): {np.std(taus):.3f}")
print("regime probabilities      :")
for k, v in res.regime_probabilities.items():
    print(f"  {k:<22s} {v:.3f}")
print("warnings                  :")
for w in res.interpretation_warning:
    print(f"  - {w}")
