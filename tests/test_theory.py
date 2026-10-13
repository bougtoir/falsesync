"""Unit tests for the theoretical identities P1-P7.

Every test maps to a row in theory_claims_registry.csv.
"""

import numpy as np
import pytest

from falsesync.aggregation import population_m, timing_cdf, weighted_aggregate
from falsesync.changepoints import ls_breakpoint, maxslope_breakpoint
from falsesync.simulation import sample_taus, simulate_panel, timing_density
from falsesync.transitions import g, is_antisymmetric

T = np.linspace(0, 10, 2001)


def test_antisymmetry_all_shapes():
    for shape in ("step", "logistic", "probit", "linear"):
        assert is_antisymmetric(shape)


def test_p1_step_identity():
    spec = {"kind": "normal", "mu": 5.0, "sigma": 0.8}
    m = population_m(T, spec, shape="step", amplitude=1.0, baseline=0.0, tau_grid=T)
    f_cdf = timing_cdf(T, spec)
    np.testing.assert_allclose(m, f_cdf, atol=2e-3)
    dm = np.gradient(m, T)
    assert T[np.argmax(dm)] == pytest.approx(5.0, abs=0.02)  # T_ms = mode


def test_p2_symmetry_ls_center():
    spec = {"kind": "normal", "mu": 5.0, "sigma": 0.8}
    for shape in ("logistic", "probit"):
        m = population_m(T, spec, shape=shape, width=0.4, tau_grid=T)
        # point symmetry about c=5: m(5+s)+m(5-s)=1 (a=0, A=1)
        s = np.linspace(0.1, 4.0, 40)
        left = np.interp(5 - s, T, m)
        right = np.interp(5 + s, T, m)
        np.testing.assert_allclose(left + right, 1.0, atol=2e-3)
        res = ls_breakpoint(T, m)
        assert res.location == pytest.approx(5.0, abs=0.05)


def test_p3a_probit_normal_closed_form():
    for sigma in (0.3, 0.8, 1.5, 3.0):
        spec = {"kind": "normal", "mu": 5.0, "sigma": sigma}
        m = population_m(T, spec, shape="probit", width=1.0)
        from scipy.stats import norm

        expected = norm.cdf((T - 5.0) / np.sqrt(1 + sigma**2))
        np.testing.assert_allclose(m, expected, atol=1e-10)
        max_slope = np.max(np.gradient(m, T))
        assert max_slope == pytest.approx(1 / np.sqrt(2 * np.pi * (1 + sigma**2)), rel=1e-3)


def test_p3_sharpness_decreases_in_sigma():
    slopes = []
    for sigma in (0.2, 0.5, 1.0, 2.0, 4.0):
        spec = {"kind": "normal", "mu": 5.0, "sigma": sigma}
        m = population_m(T, spec, shape="logistic", width=0.3, tau_grid=T)
        slopes.append(np.max(np.gradient(m, T)))
    assert all(b <= a + 1e-9 for a, b in zip(slopes, slopes[1:]))


def test_p4_weight_tilt():
    rng = np.random.default_rng(7)
    n = 4000
    taus = rng.normal(5.0, 0.8, n)
    kappa = 1.2
    w = np.exp(-kappa * taus)
    w = w / w.mean()
    panel = simulate_panel(T, taus, shape="step", weights=w, noise_sd=0.0, rng=rng)
    m_w = weighted_aggregate(panel.values, panel.weights)
    m_un = weighted_aggregate(panel.values, np.ones(n))
    res_w = ls_breakpoint(T, np.nan_to_num(m_w))
    res_un = ls_breakpoint(T, np.nan_to_num(m_un))
    assert res_w.location < res_un.location - 0.3  # tilt pulls break early


def test_p5_truncation_distorts():
    rng = np.random.default_rng(11)
    n = 4000
    taus = rng.normal(5.0, 0.7, n)
    windows = np.column_stack([taus + rng.normal(0, 0.5, n), np.full(n, 10.0)])
    panel = simulate_panel(T, taus, shape="step", windows=windows, noise_sd=0.0, rng=rng)
    m_obs = weighted_aggregate(panel.values)
    full = population_m(T, {"kind": "normal", "mu": 5.0, "sigma": 0.7}, shape="step", tau_grid=T)
    resid = np.nanmax(np.abs(m_obs - full))
    assert resid > 0.05  # truncation measurably distorts the aggregate


def test_p6_bimodality_threshold():
    from scipy.stats import norm as _n

    grid = np.linspace(0, 10, 4001)
    for delta, expect_bimodal in [(1.0, False), (1.9, False), (2.1, True), (3.0, True)]:
        f = 0.5 * _n.pdf(grid, 5 - delta / 2, 1.0) + 0.5 * _n.pdf(grid, 5 + delta / 2, 1.0)
        modes = np.sum((f[1:-1] > f[:-2]) & (f[1:-1] >= f[2:]) & (f[1:-1] > 0.05 * f.max()))
        assert (modes >= 2) == expect_bimodal


def test_p7_ls_foc():
    spec = {"kind": "gamma", "shape": 3.0, "scale": 1.0, "shift": 1.0}
    tg = np.linspace(0, 20, 8001)
    m = population_m(tg, spec, shape="logistic", width=0.5, tau_grid=tg)
    res = ls_breakpoint(tg, m)
    c = res.index
    ml = m[:c].mean()
    mr = m[c:].mean()
    assert m[c] == pytest.approx((ml + mr) / 2, abs=0.02)


def test_p7_ls_ne_ms_skewed():
    spec = {"kind": "gamma", "shape": 3.0, "scale": 1.0, "shift": 1.0}
    tg = np.linspace(0, 20, 8001)
    m = population_m(tg, spec, shape="logistic", width=0.5, tau_grid=tg)
    ls = ls_breakpoint(tg, m).location
    ms = maxslope_breakpoint(tg, m).location
    assert abs(ls - ms) > 0.2  # different functionals diverge on skewed F


def test_deterministic_rng():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    a = sample_taus(100, {"kind": "normal", "mu": 5, "sigma": 1}, rng1)
    b = sample_taus(100, {"kind": "normal", "mu": 5, "sigma": 1}, rng2)
    np.testing.assert_array_equal(a, b)
