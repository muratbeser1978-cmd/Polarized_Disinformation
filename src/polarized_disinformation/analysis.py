"""Analytical utilities for comparing Nash and social planner outcomes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .social_optimum import SocialOptimumResult
from .nash_equilibrium import compute_nash_equilibrium
from .parameters import ModelParameters


@dataclass(slots=True)
class ControlComparison:
    """Container summarising control trajectories for Nash vs Social optima."""

    beta_nash: NDArray[np.float64]
    beta_social: NDArray[np.float64]
    mu_nash: NDArray[np.float64]
    mu_social: NDArray[np.float64]
    time: NDArray[np.float64]
    group: str


@dataclass(slots=True)
class InfectionComparison:
    """Container summarising infection trajectories for Nash vs Social optima."""

    infections_nash: NDArray[np.float64]
    infections_social: NDArray[np.float64]
    peak_time_nash: float
    peak_time_social: float
    group: str


def compute_control_gap_matrix(
    comparisons: Sequence[ControlComparison],
) -> Mapping[str, NDArray[np.float64]]:
    """Return control gaps (β/μ Nash minus Social) for heatmap visualisation."""

    if not comparisons:
        raise ValueError("comparisons must not be empty")
    beta_gap = np.vstack([comp.beta_nash - comp.beta_social for comp in comparisons])
    mu_gap = np.vstack([comp.mu_nash - comp.mu_social for comp in comparisons])
    groups = np.array([comp.group for comp in comparisons])
    time = comparisons[0].time
    return {
        "beta": beta_gap,
        "mu": mu_gap,
        "groups": groups,
        "time": time,
    }


def compute_infection_gap_matrix(
    comparisons: Sequence[InfectionComparison],
) -> Mapping[str, NDArray[np.float64]]:
    """Return infection difference matrix (Nash - Social)."""

    if not comparisons:
        raise ValueError("comparisons must not be empty")
    infections = np.vstack([comp.infections_nash - comp.infections_social for comp in comparisons])
    groups = np.array([comp.group for comp in comparisons])
    time_index = np.arange(comparisons[0].infections_nash.size, dtype=np.float64)
    return {
        "infection_gap": infections,
        "groups": groups,
        "time_index": time_index,
    }


def compute_summary_statistics(
    control_comparisons: Sequence[ControlComparison],
    infection_comparisons: Sequence[InfectionComparison],
    cost_nash: float,
    cost_social: float,
    poa: float,
) -> Mapping[str, float]:
    """Compute numerical summary metrics for reporting and plots."""

    stats: dict[str, float] = {
        "poa": float(poa),
        "cost_nash": float(cost_nash),
        "cost_social": float(cost_social),
    }
    beta_diffs = []
    mu_diffs = []
    peak_shift = []
    for comp in control_comparisons:
        beta_diffs.append(np.mean(comp.beta_nash - comp.beta_social))
        mu_diffs.append(np.mean(comp.mu_nash - comp.mu_social))
    for comp in infection_comparisons:
        peak_shift.append(comp.peak_time_nash - comp.peak_time_social)
    if beta_diffs:
        stats["avg_beta_gap"] = float(np.mean(beta_diffs))
        stats["max_beta_gap"] = float(np.max(beta_diffs))
    if mu_diffs:
        stats["avg_mu_gap"] = float(np.mean(mu_diffs))
        stats["max_mu_gap"] = float(np.max(mu_diffs))
    if peak_shift:
        stats["avg_peak_shift"] = float(np.mean(peak_shift))
    if cost_social > 0.0:
        stats["cost_ratio"] = float(cost_nash / cost_social)
    return stats


def compute_price_of_anarchy(
    params: ModelParameters,
    nash_cost: float,
    social_cost: float,
) -> float:
    """
    Compute Price of Anarchy using Eq. (PoA_Polarized).

    Parameters
    ----------
    params : ModelParameters
        Structural parameters (unused, kept for API symmetry).
    nash_cost : float
        Total social cost evaluated at Nash controls.
    social_cost : float
        Total social cost evaluated at planner-optimal controls.

    Returns
    -------
    float
        PoA value (≥ 1).
    """

    if social_cost <= 0.0:
        raise ValueError("Social cost must be positive.")
    ratio = max(1.0, nash_cost / social_cost)
    return ratio


def compare_control_strategies(
    time: NDArray[np.float64],
    nash_controls: Mapping[str, NDArray[np.float64]],
    social_controls: Mapping[str, NDArray[np.float64]],
) -> list[ControlComparison]:
    """
    Compare Nash and social optimal controls for each group (Eq. 16-17 vs Nash rules).

    Returns a list of `ControlComparison` objects for convenient plotting/reporting.
    """

    comparisons: list[ControlComparison] = []
    for group in ("G", "O"):
        comparisons.append(
            ControlComparison(
                beta_nash=nash_controls[f"beta_{group}"] - 0.0,
                beta_social=social_controls[f"beta_{group}"] - 0.0,
                mu_nash=nash_controls[f"mu_{group}"] - 0.0,
                mu_social=social_controls[f"mu_{group}"] - 0.0,
                time=time,
                group=group,
            )
        )
    return comparisons


def compare_infection_dynamics(
    time: NDArray[np.float64],
    nash_state: Mapping[str, NDArray[np.float64]],
    social_state: Mapping[str, NDArray[np.float64]],
) -> list[InfectionComparison]:
    """
    Compare infection trajectories and peak timings between Nash and Social solutions.
    (Dynamic system derived from Eq. (4a)-(4f)).
    """

    comparisons: list[InfectionComparison] = []
    for group in ("G", "O"):
        infections_nash = nash_state[f"I_{group}"]
        infections_social = social_state[f"I_{group}"]
        peak_time_nash = float(time[np.argmax(infections_nash)])
        peak_time_social = float(time[np.argmax(infections_social)])
        comparisons.append(
            InfectionComparison(
                infections_nash=infections_nash,
                infections_social=infections_social,
                peak_time_nash=peak_time_nash,
                peak_time_social=peak_time_social,
                group=group,
            )
        )
    return comparisons


def analyze_externalities(
    social_result: SocialOptimumResult,
) -> Mapping[str, NDArray[np.float64]]:
    """
    Extract externality breakdown trajectories from social planner solution (Eq. 15).
    """

    return {
        "E_G": social_result.externality["E_G"].copy(),
        "E_O": social_result.externality["E_O"].copy(),
        "time": social_result.time.copy(),
    }


def generate_summary_report(
    params: ModelParameters,
    poa: float,
    control_comparisons: Sequence[ControlComparison],
    infection_comparisons: Sequence[InfectionComparison],
    statistics: Mapping[str, float],
) -> str:
    """
    Generate text report summarising PoA, control tightening, peaks, and statistics.
    """

    lines = [
        "Polarized Strategic Disinformation Diffusion Model",
        "================================================================",
        f"Horizon: {params.horizon:.2f} | Discount rate: {params.discount_rate:.3f}",
        f"Price of Anarchy (PoA): {poa:.3f}",
        "",
        "Control Tightening (β_Nash - β_Soc, μ_Nash - μ_Soc):",
    ]
    for comparison in control_comparisons:
        beta_gap = np.mean(comparison.beta_nash - comparison.beta_social)
        mu_gap = np.mean(comparison.mu_nash - comparison.mu_social)
        lines.append(
            f"  Group {comparison.group}: Δβ̄={beta_gap:.4f}, Δμ̄={mu_gap:.4f}"
        )

    lines.append("")
    lines.append("Infection Peaks (time of max I_k):")
    for comp in infection_comparisons:
        lines.append(
            f"  Group {comp.group}: Nash t*={comp.peak_time_nash:.2f}, "
            f"Social t*={comp.peak_time_social:.2f}"
        )

    lines.append("")
    lines.append("Summary Statistics:")
    for key, value in statistics.items():
        lines.append(f"  {key}: {value:.4f}")

    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "  Higher PoA indicates stronger externality-driven inefficiencies. "
        "Negative Δβ̄/Δμ̄ confirm tighter social controls, while shifted infection "
        "peaks reflect aggregate welfare gains."  # Denklem (12)-(17)
    )
    return "\n".join(lines)
