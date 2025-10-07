"""Entry point for running Nash vs Social optimum analysis and visualisation."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from polarized_disinformation import (
    ModelParameters,
    SocialOptimumResult,
    analyze_externalities,
    compare_control_strategies,
    compare_infection_dynamics,
    compare_nash_vs_social,
    compute_nash_equilibrium,
    compute_price_of_anarchy,
    compute_total_social_cost,
    default_matrices,
    default_parameters,
    generate_summary_report,
    solve_social_optimum,
)
from polarized_disinformation.analysis import (
    compute_control_gap_matrix,
    compute_infection_gap_matrix,
    compute_summary_statistics,
)
from polarized_disinformation.visualization import create_dashboard, save_figures

CONFIG_PATH = Path("configs/default_params.yaml")
OUTPUT_DIR = Path("outputs")


def load_parameters(path: Path) -> ModelParameters:
    """Load model parameters from YAML, fallback to defaults if missing."""

    if not path.exists():
        return default_parameters()
    data: Mapping[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    from polarized_disinformation.parameters import load_parameters as load_fn

    return load_fn(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to parameter configuration file.",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=200,
        help="Number of discretisation points over the planning horizon.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip saving figures (useful for quick testing).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for reports and figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_parameters(args.config)
    matrices = default_matrices()
    time_grid = np.linspace(0.0, params.horizon, args.time_steps, dtype=np.float64)

    def media_schedule(_: float) -> np.ndarray:
        return np.array([0.4, 0.4], dtype=np.float64)

    nash_result = compute_nash_equilibrium(
        params=params,
        matrices=matrices,
        media_intensity=media_schedule,
        T=params.horizon,
        time_grid=time_grid,
        max_iterations=20,
        tolerance=1e-4,
        verbose=False,
    )
    social_result: SocialOptimumResult = solve_social_optimum(
        params=params,
        matrices=matrices,
        media_intensity=media_schedule,
        time_grid=time_grid,
        max_iterations=30,
        tolerance=1e-4,
        relaxation=0.6,
        verbose=False,
    )

    cost_nash = compute_total_social_cost(
        params,
        nash_result["state_nash"],
        nash_result["controls_nash"],
        nash_result["time"],
    )
    cost_social = social_result.total_cost
    poa = compute_price_of_anarchy(params, cost_nash, cost_social)

    control_comparisons = compare_control_strategies(time_grid, nash_result["controls_nash"], social_result.controls)
    infection_comparisons = compare_infection_dynamics(time_grid, nash_result["state_nash"], social_result.state)
    control_gaps = compute_control_gap_matrix(control_comparisons)
    infection_gaps = compute_infection_gap_matrix(infection_comparisons)
    statistics = compute_summary_statistics(control_comparisons, infection_comparisons, cost_nash, cost_social, poa)
    externalities = analyze_externalities(social_result)
    comparison_metrics = compare_nash_vs_social(params, nash_result, social_result)

    report = generate_summary_report(
        params,
        poa,
        control_comparisons,
        infection_comparisons,
        statistics,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary_report.txt").write_text(report, encoding="utf-8")

    if not args.no_figures:
        figures = create_dashboard(
            time_grid,
            nash_result["state_nash"],
            social_result.state,
            control_comparisons,
            infection_comparisons,
            externalities,
            control_gaps,
            infection_gaps,
            statistics,
            poa,
            cost_nash,
            cost_social,
            social_result.risk_inter,
            social_result.risk_media,
        )
        save_figures(figures, args.output, stem="dashboard")

    print(report)
    print(
        f"Costs | Nash: {comparison_metrics['cost_nash']:.3f} | "
        f"Social: {comparison_metrics['cost_social']:.3f} | PoA: {comparison_metrics['poa']:.3f}"
    )


if __name__ == "__main__":
    main()
