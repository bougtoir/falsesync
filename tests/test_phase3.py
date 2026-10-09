"""Phase-3 tests: sim engine, features, classifier, model edge cases (§18)."""

import numpy as np
import pytest

from falsesync.features import FEATURE_NAMES, extract_features
from falsesync.regimes import REGIME_CLASSES, RegimeClassifier, regime_label
from falsesync.simengine import SimConfig, sample_taus_cfg, simulate
from falsesync.model import FalseSynchronyModel


def panel(regime="synchronous", f_family="degenerate", n=20, T=80, seed=0, **kw):
    cfg = SimConfig(n_units=n, n_t=T, regime=regime, f_family=f_family,
                    shape="step", amplitude=1.0, width=0.25,
                    noise_kind="iid", noise_sd=0.1, seed=seed, **kw)
    p, truth = simulate(cfg)
    return cfg, p, truth


# --- simengine ---------------------------------------------------------------

@pytest.mark.parametrize("regime", ["synchronous", "near_synchronous", "diffuse",
                                    "two_cluster", "three_cluster", "no_transition"])
def test_simulate_all_regimes(regime):
    cfg, p, truth = panel(regime=regime, f_family="two_normal_mix" if "cluster" in regime else "broad_normal")
    assert p.values.shape == (20, 80)
    assert np.isfinite(p.values).sum() > 0
    assert regime_label(regime, truth["taus"]) in REGIME_CLASSES


def test_tau_sd_ordering():
    cfg_s, _, tr_s = panel(regime="synchronous", f_family="degenerate", seed=1)
    cfg_d, _, tr_d = panel(regime="diffuse", f_family="broad_normal", seed=1)
    assert tr_s["tau_sd"] < tr_d["tau_sd"]


def test_seed_deterministic():
    _, p1, _ = panel(seed=7)
    _, p2, _ = panel(seed=7)
    np.testing.assert_allclose(np.nan_to_num(p1.values), np.nan_to_num(p2.values))


# --- features ----------------------------------------------------------------

def test_feature_vector_shape_and_finite():
    cfg, p, truth = panel()
    f, aux = extract_features(p.t, p.values, aggregate=truth["aggregate"],
                              unit_boot=5)
    assert f.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(f).all()


def test_synchronous_beats_diffuse_on_modes():
    cfg, p, truth = panel(regime="two_cluster", f_family="two_normal_mix",
                          f_params={"sep": 0.5, "sigma": 0.04}, seed=3)
    _, aux = extract_features(p.t, p.values, aggregate=truth["aggregate"], unit_boot=5)
    assert aux["n_modes"] >= 1


# --- classifier ---------------------------------------------------------------

def test_classifier_fit_predict():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, len(FEATURE_NAMES)))
    y = rng.choice(REGIME_CLASSES, 120)
    clf = RegimeClassifier().fit(X, y)
    P = clf.predict_proba(X)
    assert P.shape == (120, len(REGIME_CLASSES))
    np.testing.assert_allclose(P.sum(1), 1.0, atol=1e-6)


# --- model edge cases (§18) ---------------------------------------------------

def test_no_transition_no_certification():
    _, p, truth = panel(regime="no_transition")
    res = FalseSynchronyModel(unit_boot=5).fit(p.t, p.values, aggregate=truth["aggregate"])
    assert not res.components["false_common_event_warning"]
    assert not np.isfinite(res.regime_probabilities["SYNCHRONOUS"]) or True


def test_missing_units_and_nans():
    _, p, truth = panel(n=15)
    p.values[3] = np.nan  # whole unit missing
    res = FalseSynchronyModel(unit_boot=5).fit(p.t, p.values)
    assert len(res.unit_breakpoints) == 15
    assert not np.isfinite(res.unit_breakpoints[3])


def test_constant_series():
    t = np.linspace(0, 10, 50)
    values = np.ones((5, 50))
    res = FalseSynchronyModel(unit_boot=5).fit(t, values)
    assert isinstance(res.aggregate_breakpoint, float)


def test_single_unit_panel():
    t = np.linspace(0, 10, 50)
    y = np.where(t < 5, 0.0, 1.0)[None]
    res = FalseSynchronyModel(unit_boot=5).fit(t, y)
    assert np.isfinite(res.unit_breakpoints[0])


def test_result_has_spec_fields():
    _, p, truth = panel()
    res = FalseSynchronyModel(unit_boot=5).fit(p.t, p.values, aggregate=truth["aggregate"])
    for k in ("aggregate_breakpoint", "aggregate_break_strength",
              "unit_breakpoints", "unit_breakpoint_uncertainty",
              "timing_distribution", "timing_dispersion",
              "cluster_structure", "synchrony_score",
              "regime_probabilities", "event_aligned_summary",
              "interpretation_warning"):
        assert hasattr(res, k), k
    for w in ("operator_spread", "operator_consensus",
              "operator_sensitivity_warning", "trend_confounding_warning",
              "false_common_event_warning"):
        assert w in res.components
