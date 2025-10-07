"""Visualisation utilities for the polarized disinformation model."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .analysis import ControlComparison, InfectionComparison

plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 300})


def _ensure_output_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_state_trajectories(
    time: NDArray[np.float64],
    state: Mapping[str, NDArray[np.float64]],
    title: str,
) -> plt.Figure:
    """Plot S, I, R trajectories for both cohorts (Eq. (4a)-(4f))."""

    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
    labels = ["S_G", "I_G", "R_G", "S_O", "I_O", "R_O"]
    for idx, ax in enumerate(axes.flat):
        ax.plot(time, state[labels[idx]], label=labels[idx])
        ax.set_ylabel(labels[idx])
        ax.grid(True, alpha=0.3)
        if idx >= 4:
            ax.set_xlabel("Time")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_time_series(
    time: NDArray[np.float64],
    series: Mapping[str, NDArray[np.float64]],
    title: str,
) -> plt.Figure:
    """Generic helper to plot arbitrary time series bundles on stacked axes."""

    keys = list(series.keys())
    fig, axes = plt.subplots(len(keys), 1, figsize=(8, 2.5 * len(keys)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        ax.plot(time, series[key], label=key)
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_control_comparison(
    comparisons: Sequence[ControlComparison],
) -> plt.Figure:
    """Plot Nash vs Social controls for each group (Eq. (10)-(17))."""

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for idx, comparison in enumerate(comparisons):
        ax_beta = axes[idx, 0]
        ax_mu = axes[idx, 1]
        ax_beta.plot(comparison.time, comparison.beta_nash, label="β_Nash", linestyle="--")
        ax_beta.plot(comparison.time, comparison.beta_social, label="β_Soc")
        ax_beta.set_ylabel(f"β_{comparison.group}")
        ax_beta.grid(True, alpha=0.3)
        ax_mu.plot(comparison.time, comparison.mu_nash, label="μ_Nash", linestyle="--")
        ax_mu.plot(comparison.time, comparison.mu_social, label="μ_Soc")
        ax_mu.set_ylabel(f"μ_{comparison.group}")
        ax_mu.grid(True, alpha=0.3)
        if idx == len(comparisons) - 1:
            ax_beta.set_xlabel("Time")
            ax_mu.set_xlabel("Time")
    axes[0, 0].legend(loc="upper right")
    axes[0, 1].legend(loc="upper right")
    fig.suptitle("Control Comparison: Nash vs Social Optimum")
    fig.tight_layout()
    return fig


def plot_poa_analysis(poa: float) -> plt.Figure:
    """Visualise Price of Anarchy as a single-bar chart."""

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["PoA"], [poa], color="#D95F02")
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1.0)
    ax.set_ylim(0.9, max(1.1, poa * 1.1))
    ax.set_ylabel("PoA")
    ax.set_title("Price of Anarchy")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_infection_comparison(
    comparisons: Sequence[InfectionComparison],
) -> plt.Figure:
    """Plot infection trajectories for Nash vs Social solutions."""

    fig, axes = plt.subplots(len(comparisons), 1, figsize=(8, 4 * len(comparisons)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, comp in zip(axes, comparisons):
        ax.plot(comp.infections_nash, label=f"I_{comp.group} Nash", linestyle="--")
        ax.plot(comp.infections_social, label=f"I_{comp.group} Social")
        ax.set_ylabel(f"I_{comp.group}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time Index")
    axes[0].legend(loc="upper right")
    fig.suptitle("Infection Dynamics Comparison")
    fig.tight_layout()
    return fig


def plot_externalities(
    externalities: Mapping[str, NDArray[np.float64]],
) -> plt.Figure:
    """Plot externality trajectories (Eq. (15))."""

    time = externalities["time"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, externalities["E_G"], label="Group G Externality")
    ax.plot(time, externalities["E_O"], label="Group O Externality")
    ax.set_xlabel("Time")
    ax.set_ylabel("Externality Cost")
    ax.set_title("Total Externality Costs")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_control_gap_heatmap(
    control_gaps: Mapping[str, NDArray[np.float64]],
    metric: str,
    title: str,
) -> plt.Figure:
    """Plot heatmap of control gaps for a chosen metric ('beta' or 'mu')."""

    data = control_gaps[metric]
    time = control_gaps["time"]
    groups = control_gaps["groups"]
    fig, ax = plt.subplots(figsize=(9, 3))
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=[time[0], time[-1], -0.5, len(groups) - 0.5],
        cmap="RdBu",
    )
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_xlabel("Time")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Gap (Nash - Social)")
    fig.tight_layout()
    return fig


def plot_infection_heatmap(
    infection_gaps: Mapping[str, NDArray[np.float64]],
) -> plt.Figure:
    """Heatmap showing infection reduction achieved by social optimum."""

    data = infection_gaps["infection_gap"]
    groups = infection_gaps["groups"]
    time_idx = infection_gaps["time_index"]
    fig, ax = plt.subplots(figsize=(9, 3))
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=[time_idx[0], time_idx[-1], -0.5, len(groups) - 0.5],
        cmap="PuOr",
    )
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_xlabel("Time Index")
    ax.set_title("Infection Gap Heatmap (Nash - Social)")
    fig.colorbar(im, ax=ax, label="ΔI")
    fig.tight_layout()
    return fig


def plot_externalities_heatmap(
    externalities: Mapping[str, NDArray[np.float64]],
) -> plt.Figure:
    """Heatmap for externality costs across groups and time."""

    time = externalities["time"]
    data = np.vstack([externalities["E_G"], externalities["E_O"]])
    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=[time[0], time[-1], -0.5, 1.5],
        cmap="magma",
    )
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["G", "O"])
    ax.set_xlabel("Time")
    ax.set_title("Externality Heatmap")
    fig.colorbar(im, ax=ax, label="Externality")
    fig.tight_layout()
    return fig


def plot_cost_summary(cost_nash: float, cost_social: float) -> plt.Figure:
    """Bar chart comparing Nash and Social costs."""

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Nash", "Social"], [cost_nash, cost_social], color=["#1b9e77", "#d95f02"])
    ax.set_ylabel("Total Cost")
    ax.set_title("Cost Comparison")
    for idx, value in enumerate([cost_nash, cost_social]):
        ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_risk_profiles(
    time: NDArray[np.float64],
    risk_inter: NDArray[np.float64],
    risk_media: NDArray[np.float64],
) -> plt.Figure:
    """Plot risk trajectories perceived by the planner."""

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    labels = ["Interpersonal Risk G", "Interpersonal Risk O", "Media Risk G", "Media Risk O"]
    data = [risk_inter[0], risk_inter[1], risk_media[0], risk_media[1]]
    for ax, label, series in zip(axes.flat, labels, data):
        ax.plot(time, series)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("Time")
    axes[1, 1].set_xlabel("Time")
    fig.suptitle("Risk Environment Trajectories")
    fig.tight_layout()
    return fig


def plot_statistics_table(statistics: Mapping[str, float]) -> plt.Figure:
    """Render key statistics as a table within a figure."""

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    rows = sorted(statistics.items())
    table_data = [[key, f"{value:.4f}"] for key, value in rows]
    table = ax.table(cellText=table_data, colLabels=["Metric", "Value"], loc="center")
    table.scale(1, 1.4)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax.set_title("Statistical Summary")
    return fig


def create_dashboard(
    time: NDArray[np.float64],
    nash_state: Mapping[str, NDArray[np.float64]],
    social_state: Mapping[str, NDArray[np.float64]],
    control_comparisons: Sequence[ControlComparison],
    infection_comparisons: Sequence[InfectionComparison],
    externalities: Mapping[str, NDArray[np.float64]],
    control_gaps: Mapping[str, NDArray[np.float64]],
    infection_gaps: Mapping[str, NDArray[np.float64]],
    statistics: Mapping[str, float],
    poa: float,
    cost_nash: float,
    cost_social: float,
    risk_inter: NDArray[np.float64],
    risk_media: NDArray[np.float64],
) -> list[plt.Figure]:
    """Create comprehensive dashboard including advanced technical plots."""

    figures = [
        plot_state_trajectories(time, nash_state, title="Nash State Trajectories"),
        plot_state_trajectories(time, social_state, title="Social Optimum State Trajectories"),
        plot_control_comparison(control_comparisons),
        plot_infection_comparison(infection_comparisons),
        plot_externalities(externalities),
        plot_poa_analysis(poa),
        plot_control_gap_heatmap(control_gaps, metric="beta", title="β Gap Heatmap (Nash - Social)"),
        plot_control_gap_heatmap(control_gaps, metric="mu", title="μ Gap Heatmap (Nash - Social)"),
        plot_infection_heatmap(infection_gaps),
        plot_externalities_heatmap(externalities),
        plot_cost_summary(cost_nash, cost_social),
        plot_risk_profiles(time, risk_inter, risk_media),
        plot_statistics_table(statistics),
    ]
    return figures


def save_figures(figures: Iterable[plt.Figure], output_dir: Path, stem: str) -> None:
    """Persist figures as PDF files with publication-friendly DPI."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, fig in enumerate(figures, start=1):
        path = output_dir / f"{stem}_{idx:02d}.pdf"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
