"""Core modules for the polarized strategic disinformation diffusion model."""

from .parameters import ModelParameters, GroupParameters, default_parameters
from .matrices import InteractionMatrices, default_matrices
from .hjb_solver import (
    HJBSolution,
    HJBSolver,
    GroupHJBSolution,
    compute_best_response,
    compute_interpersonal_risk_environment,
    compute_media_risk_environment,
)
from .social_optimum import (
    CostateTrajectories,
    SocialOptimumResult,
    adjoint_system_ode,
    compare_nash_vs_social,
    compute_externality_cost,
    compute_hamiltonian,
    compute_optimal_beta_social,
    compute_optimal_mu_social,
    compute_total_social_cost,
    solve_social_optimum,
)
from .nash_equilibrium import compute_nash_equilibrium
from .analysis import (
    analyze_externalities,
    compare_control_strategies,
    compare_infection_dynamics,
    compute_price_of_anarchy,
    generate_summary_report,
)

__all__ = [
    "ModelParameters",
    "GroupParameters",
    "InteractionMatrices",
    "default_parameters",
    "default_matrices",
    "HJBSolver",
    "HJBSolution",
    "GroupHJBSolution",
    "compute_best_response",
    "compute_interpersonal_risk_environment",
    "compute_media_risk_environment",
    "SocialOptimumResult",
    "CostateTrajectories",
    "compute_total_social_cost",
    "compute_hamiltonian",
    "compute_externality_cost",
    "adjoint_system_ode",
    "compute_optimal_beta_social",
    "compute_optimal_mu_social",
    "solve_social_optimum",
    "compare_nash_vs_social",
    "analyze_externalities",
    "compare_control_strategies",
    "compare_infection_dynamics",
    "compute_price_of_anarchy",
    "generate_summary_report",
    "compute_nash_equilibrium",
]
