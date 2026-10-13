"""Figures for Phase 2 outputs (matplotlib, non-interactive)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def aggregate_curve_figure(t, m_pop, agg_obs, path, title="Aggregate curve"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, m_pop, label="population m(t)", lw=2)
    if agg_obs is not None:
        ax.plot(t, agg_obs, label="observed aggregate", alpha=0.7)
    ax.set_xlabel("t")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def timing_distribution_figure(grid, densities: dict, path, title="Timing distribution"):
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, d in densities.items():
        ax.plot(grid, d, label=label)
    ax.set_xlabel("tau")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def operator_comparison_figure(t, m, results: dict, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, m, lw=1.5, label="m(t)")
    for name, loc in results.items():
        ax.axvline(loc, ls="--", label=f"{name}: {loc:.2f}")
    ax.set_xlabel("t")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def aligned_panel_figure(s_grid, aligned, path, title="Event-time aligned panel"):
    fig, ax = plt.subplots(figsize=(7, 4))
    for row in aligned:
        ax.plot(s_grid, row, color="0.7", lw=0.5)
    ax.plot(s_grid, np.nanmean(aligned, axis=0), color="C0", lw=2, label="aligned mean")
    ax.set_xlabel("event time s = t - tau_i")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
