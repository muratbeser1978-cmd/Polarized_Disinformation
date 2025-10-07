"""Pontryagin-based social planner solver for the polarized diffusion model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, MutableMapping, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .hjb_solver import (
    compute_interpersonal_risk_environment,
    compute_media_risk_environment,
)
from .matrices import InteractionMatrices
from .parameters import GroupParameters, ModelParameters
from .sirs_dynamics import (
    SIRSSimulationResult,
    default_intervention_schedule,
    simulate_sirs,
)

GroupName = str


@dataclass(slots=True)
class CostateTrajectories:
    """Stores costate trajectories for both groups along the planning horizon."""

    t: NDArray[np.float64]
    lambda_S_G: NDArray[np.float64]
    lambda_I_G: NDArray[np.float64]
    lambda_R_G: NDArray[np.float64]
    lambda_S_O: NDArray[np.float64]
    lambda_I_O: NDArray[np.float64]
    lambda_R_O: NDArray[np.float64]


@dataclass(slots=True)
class SocialOptimumResult:
    """Aggregated outputs of the social planner problem."""

    time: NDArray[np.float64]
    controls: Mapping[str, NDArray[np.float64]]
    state: Mapping[str, NDArray[np.float64]]
    costate: CostateTrajectories
    convergence: Mapping[str, object]
    total_cost: float
    risk_inter: NDArray[np.float64]
    risk_media: NDArray[np.float64]
    externality: Mapping[str, NDArray[np.float64]]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _group_params(params: ModelParameters, group: GroupName) -> GroupParameters:
    return params.government if group == "G" else params.opposition


def _project(value: float, lower: float, upper: float) -> float:
    return float(min(upper, max(lower, value)))


def _effort_cost(beta: NDArray[np.float64], mu: NDArray[np.float64], params: GroupParameters) -> NDArray[np.float64]:
    return (
        params.w_beta * (params.beta_0 / beta - 1.0)
        + params.w_mu * (params.mu_0 / mu - 1.0)
    )


def _ensure_matrices(
    matrices: InteractionMatrices | Mapping[str, NDArray[np.float64]]
) -> InteractionMatrices:
    if isinstance(matrices, InteractionMatrices):
        return matrices
    required = {"omega", "alpha", "pi", "kappa"}
    missing = required - set(matrices)
    if missing:
        raise ValueError(f"Matrix dictionary missing keys: {sorted(missing)}")
    return InteractionMatrices(
        omega=np.asarray(matrices["omega"], dtype=np.float64),
        a=np.asarray(matrices["alpha"], dtype=np.float64),
        pi=np.asarray(matrices["pi"], dtype=np.float64),
        k=np.asarray(matrices["kappa"], dtype=np.float64),
    )


def _initial_control_guess(
    params: ModelParameters,
    time_grid: NDArray[np.float64],
    initial_controls: Optional[Mapping[str, NDArray[np.float64]]],
) -> MutableMapping[str, NDArray[np.float64]]:
    defaults = {
        "beta_G": params.government.beta_0,
        "mu_G": params.government.mu_0,
        "beta_O": params.opposition.beta_0,
        "mu_O": params.opposition.mu_0,
    }
    control_paths: MutableMapping[str, NDArray[np.float64]] = {}
    for key, default_value in defaults.items():
        if initial_controls is not None and key in initial_controls:
            arr = np.asarray(initial_controls[key], dtype=np.float64)
            if arr.shape != time_grid.shape:
                raise ValueError(
                    f"Initial control '{key}' must have shape {time_grid.shape}."
                )
        else:
            arr = np.full(time_grid.shape, default_value, dtype=np.float64)

        if key.endswith("G"):
            bounds = (
                params.government.beta_min,
                params.government.beta_0,
            ) if key.startswith("beta") else (
                params.government.mu_min,
                params.government.mu_0,
            )
        else:
            bounds = (
                params.opposition.beta_min,
                params.opposition.beta_0,
            ) if key.startswith("beta") else (
                params.opposition.mu_min,
                params.opposition.mu_0,
            )
        control_paths[key] = np.clip(arr, bounds[0], bounds[1])
    return control_paths


def _simulate_forward(
    params: ModelParameters,
    matrices: InteractionMatrices,
    controls: Mapping[str, NDArray[np.float64]],
    media_intensity: Callable[[float], Iterable[float]],
    time_grid: NDArray[np.float64],
    intervention_schedule: Callable[[float], Iterable[float]],
) -> SIRSSimulationResult:
    def control_policy(t: float, _: NDArray[np.float64]) -> NDArray[np.float64]:
        beta_g = float(np.interp(t, time_grid, controls["beta_G"]))
        mu_g = float(np.interp(t, time_grid, controls["mu_G"]))
        beta_o = float(np.interp(t, time_grid, controls["beta_O"]))
        mu_o = float(np.interp(t, time_grid, controls["mu_O"]))
        return np.array([beta_g, mu_g, beta_o, mu_o], dtype=np.float64)

    def media_schedule(t: float) -> NDArray[np.float64]:
        values = np.asarray(media_intensity(t), dtype=np.float64)
        if values.shape != (2,):
            raise ValueError("media_intensity must return an iterable of length 2.")
        return values

    def delta_schedule(t: float) -> NDArray[np.float64]:
        values = np.asarray(intervention_schedule(t), dtype=np.float64)
        if values.shape != (2,):
            raise ValueError("intervention_schedule must return an iterable of length 2.")
        return values

    x0 = _default_initial_state(params)

    return simulate_sirs(
        params=params,
        matrices=matrices,
        x0=x0,
        control_policy=control_policy,
        media_schedule=media_schedule,
        intervention_schedule=delta_schedule,
        t_span=(0.0, params.horizon),
        time_grid=time_grid,
    )


def _default_initial_state(params: ModelParameters) -> NDArray[np.float64]:
    fraction = 1e-4
    i_g = min(fraction, 0.1 * params.government.population_share)
    i_o = min(2 * fraction, 0.1 * params.opposition.population_share)
    r_g = r_o = 0.0
    s_g = params.government.population_share - i_g
    s_o = params.opposition.population_share - i_o
    return np.array([s_g, i_g, r_g, s_o, i_o, r_o], dtype=np.float64)


def _extract_state_dictionary(result: SIRSSimulationResult) -> Mapping[str, NDArray[np.float64]]:
    return {
        "S_G": result.x[:, 0].copy(),
        "I_G": result.x[:, 1].copy(),
        "R_G": result.x[:, 2].copy(),
        "S_O": result.x[:, 3].copy(),
        "I_O": result.x[:, 4].copy(),
        "R_O": result.x[:, 5].copy(),
    }


def _control_sup_norm(
    current: Mapping[str, NDArray[np.float64]],
    candidate: Mapping[str, NDArray[np.float64]],
) -> float:
    diffs = [np.max(np.abs(candidate[key] - current[key])) for key in candidate]
    return float(np.max(diffs))


def _compute_risk_paths(
    params: ModelParameters,
    matrices: InteractionMatrices,
    state: Mapping[str, NDArray[np.float64]],
    media_path: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n = media_path.shape[0]
    risk_inter = np.empty((2, n), dtype=np.float64)
    risk_media = np.empty((2, n), dtype=np.float64)
    N_G = params.government.population_share
    N_O = params.opposition.population_share
    for idx in range(n):
        I_G = float(state["I_G"][idx])
        I_O = float(state["I_O"][idx])
        M_G, M_O = media_path[idx]
        risk_inter[0, idx] = compute_interpersonal_risk_environment(
            I_G,
            I_O,
            N_G,
            N_O,
            matrices.omega,
            matrices.a,
            "G",
        )
        risk_inter[1, idx] = compute_interpersonal_risk_environment(
            I_G,
            I_O,
            N_G,
            N_O,
            matrices.omega,
            matrices.a,
            "O",
        )
        risk_media[0, idx] = compute_media_risk_environment(
            M_G,
            M_O,
            matrices.pi,
            matrices.k,
            "G",
        )
        risk_media[1, idx] = compute_media_risk_environment(
            M_G,
            M_O,
            matrices.pi,
            matrices.k,
            "O",
        )
    return risk_inter, risk_media


def _sample_schedule(
    time_grid: NDArray[np.float64],
    schedule: Callable[[float], Iterable[float]],
) -> NDArray[np.float64]:
    path = np.empty((time_grid.size, 2), dtype=np.float64)
    for idx, t in enumerate(time_grid):
        values = np.asarray(schedule(float(t)), dtype=np.float64)
        if values.shape != (2,):
            raise ValueError("schedule must return an iterable of length 2.")
        path[idx] = values
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_total_social_cost(
    params: ModelParameters,
    state: Mapping[str, NDArray[np.float64]],
    controls: Mapping[str, NDArray[np.float64]],
    time: NDArray[np.float64],
) -> float:
    """Compute the aggregate social planner objective (Eq. (12))."""

    time = np.asarray(time, dtype=np.float64)
    if time.ndim != 1:
        raise ValueError("time must be one-dimensional.")
    integrand = np.zeros(time.shape, dtype=np.float64)
    for group, group_params in (("G", params.government), ("O", params.opposition)):
        beta = controls[f"beta_{group}"]
        mu = controls[f"mu_{group}"]
        effort = group_params.population_share * _effort_cost(beta, mu, group_params)
        infections = group_params.r_I * state[f"I_{group}"]
        integrand += effort + infections
    return float(np.trapezoid(integrand, time))


def compute_hamiltonian(
    params: ModelParameters,
    matrices: InteractionMatrices,
    state_vector: NDArray[np.float64],
    control_vector: NDArray[np.float64],
    costate_vector: NDArray[np.float64],
    delta_vector: NDArray[np.float64],
    risk_inter: NDArray[np.float64],
    risk_media: NDArray[np.float64],
) -> float:
    """Evaluate the Hamiltonian for a single state (Eq. (13))."""

    S_G, I_G, R_G, S_O, I_O, R_O = state_vector
    beta_G, mu_G, beta_O, mu_O = control_vector
    lambda_S_G, lambda_I_G, lambda_R_G, lambda_S_O, lambda_I_O, lambda_R_O = costate_vector
    delta_G, delta_O = delta_vector

    populations = np.array(
        [params.government.population_share, params.opposition.population_share],
        dtype=np.float64,
    )
    betas = np.array([beta_G, beta_O], dtype=np.float64)
    mus = np.array([mu_G, mu_O], dtype=np.float64)
    suscept = np.array([S_G, S_O], dtype=np.float64)
    infected = np.array([I_G, I_O], dtype=np.float64)
    recovered = np.array([R_G, R_O], dtype=np.float64)
    lambdas_S = np.array([lambda_S_G, lambda_S_O], dtype=np.float64)
    lambdas_I = np.array([lambda_I_G, lambda_I_O], dtype=np.float64)
    lambdas_R = np.array([lambda_R_G, lambda_R_O], dtype=np.float64)
    deltas = np.array([delta_G, delta_O], dtype=np.float64)

    effort_cost = (
        populations
        * np.array(
            [
                _effort_cost(np.array([beta_G]), np.array([mu_G]), params.government)[0],
                _effort_cost(np.array([beta_O]), np.array([mu_O]), params.opposition)[0],
            ]
        )
    )
    infection_cost = np.array([
        params.government.r_I * I_G,
        params.opposition.r_I * I_O,
    ])
    lambda_diff = lambdas_I - lambdas_S
    Lambda = betas * risk_inter + mus * risk_media

    term_transition = (
        Lambda * suscept * (lambdas_I - lambdas_S)
        + suscept * deltas * (lambdas_R - lambdas_S)
        + infected
        * np.array([
            params.government.gamma * (lambda_R_G - lambda_I_G),
            params.opposition.gamma * (lambda_R_O - lambda_I_O),
        ])
        + recovered
        * np.array([
            params.government.rho * (lambda_S_G - lambda_R_G),
            params.opposition.rho * (lambda_S_O - lambda_R_O),
        ])
    )

    total = np.sum(effort_cost + infection_cost + term_transition)
    return float(total)


def compute_externality_cost(
    params: ModelParameters,
    matrices: InteractionMatrices,
    susceptible: NDArray[np.float64],
    lambda_diff: NDArray[np.float64],
    beta: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Compute the total externality cost ``\mathcal{E}_k`` (Eq. (15))."""

    if susceptible.shape != (2,) or lambda_diff.shape != (2,) or beta.shape != (2,):
        raise ValueError("Inputs must be length-2 arrays for both groups.")
    weights = matrices.omega * matrices.a
    populations = np.array(
        [params.government.population_share, params.opposition.population_share],
        dtype=np.float64,
    )
    externality = np.empty(2, dtype=np.float64)
    for k in range(2):
        contribution = 0.0
        for j in range(2):
            contribution += (
                susceptible[j]
                * lambda_diff[j]
                * beta[j]
                * weights[j, k]
            )
        externality[k] = contribution / populations[k]
    return externality


def adjoint_system_ode(
    t: float,
    lam: NDArray[np.float64],
    *,
    time_grid: NDArray[np.float64],
    params: ModelParameters,
    controls: Mapping[str, NDArray[np.float64]],
    risk_inter: NDArray[np.float64],
    risk_media: NDArray[np.float64],
    susceptibles: NDArray[np.float64],
    delta_path: NDArray[np.float64],
    matrices: InteractionMatrices,
) -> NDArray[np.float64]:
    """Evaluate costate ODE right-hand side (Eq. (14a)-(14c))."""

    lambda_S_G, lambda_I_G, lambda_R_G, lambda_S_O, lambda_I_O, lambda_R_O = lam
    s_g = float(np.interp(t, time_grid, susceptibles[0]))
    s_o = float(np.interp(t, time_grid, susceptibles[1]))
    beta_g = float(np.interp(t, time_grid, controls["beta_G"]))
    beta_o = float(np.interp(t, time_grid, controls["beta_O"]))
    mu_g = float(np.interp(t, time_grid, controls["mu_G"]))
    mu_o = float(np.interp(t, time_grid, controls["mu_O"]))
    risk_inter_g = float(np.interp(t, time_grid, risk_inter[0]))
    risk_inter_o = float(np.interp(t, time_grid, risk_inter[1]))
    risk_media_g = float(np.interp(t, time_grid, risk_media[0]))
    risk_media_o = float(np.interp(t, time_grid, risk_media[1]))
    delta_g = float(np.interp(t, time_grid, delta_path[:, 0]))
    delta_o = float(np.interp(t, time_grid, delta_path[:, 1]))

    lambda_diff = np.array(
        [lambda_I_G - lambda_S_G, lambda_I_O - lambda_S_O],
        dtype=np.float64,
    )
    externality = compute_externality_cost(
        params,
        matrices,
        susceptible=np.array([s_g, s_o], dtype=np.float64),
        lambda_diff=lambda_diff,
        beta=np.array([beta_g, beta_o], dtype=np.float64),
    )

    Lambda = np.array(
        [beta_g * risk_inter_g + mu_g * risk_media_g,
         beta_o * risk_inter_o + mu_o * risk_media_o],
        dtype=np.float64,
    )

    gov = params.government
    opp = params.opposition

    d_lambda_S_G = Lambda[0] * (lambda_S_G - lambda_I_G) + delta_g * (lambda_S_G - lambda_R_G)
    d_lambda_S_O = Lambda[1] * (lambda_S_O - lambda_I_O) + delta_o * (lambda_S_O - lambda_R_O)

    d_lambda_R_G = gov.rho * (lambda_R_G - lambda_S_G)
    d_lambda_R_O = opp.rho * (lambda_R_O - lambda_S_O)

    d_lambda_I_G = -gov.r_I + gov.gamma * (lambda_I_G - lambda_R_G) - externality[0]
    d_lambda_I_O = -opp.r_I + opp.gamma * (lambda_I_O - lambda_R_O) - externality[1]

    return np.array([
        d_lambda_S_G,
        d_lambda_I_G,
        d_lambda_R_G,
        d_lambda_S_O,
        d_lambda_I_O,
        d_lambda_R_O,
    ], dtype=np.float64)


def compute_optimal_beta_social(
    params: ModelParameters,
    group: GroupName,
    risk_inter: NDArray[np.float64],
    susceptible: NDArray[np.float64],
    lambda_diff: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute optimal interpersonal controls ``β_k^{Soc}`` (Eq. (16))."""

    group_params = _group_params(params, group)
    N_k = group_params.population_share
    msc = risk_inter * susceptible * lambda_diff
    beta_star = np.empty_like(msc)
    for idx, value in enumerate(msc):
        if value <= 0.0:
            beta_star[idx] = group_params.beta_0
        else:
            candidate = np.sqrt((N_k * group_params.w_beta * group_params.beta_0) / max(value, 1e-12))
            beta_star[idx] = _project(candidate, group_params.beta_min, group_params.beta_0)
    return beta_star


def compute_optimal_mu_social(
    params: ModelParameters,
    group: GroupName,
    risk_media: NDArray[np.float64],
    susceptible: NDArray[np.float64],
    lambda_diff: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute optimal media controls ``μ_k^{Soc}`` (Eq. (17))."""

    group_params = _group_params(params, group)
    N_k = group_params.population_share
    msc = risk_media * susceptible * lambda_diff
    mu_star = np.empty_like(msc)
    for idx, value in enumerate(msc):
        if value <= 0.0:
            mu_star[idx] = group_params.mu_0
        else:
            candidate = np.sqrt((N_k * group_params.w_mu * group_params.mu_0) / max(value, 1e-12))
            mu_star[idx] = _project(candidate, group_params.mu_min, group_params.mu_0)
    return mu_star


def solve_social_optimum(
    params: ModelParameters,
    matrices: InteractionMatrices | Mapping[str, NDArray[np.float64]],
    media_intensity: Callable[[float], Iterable[float]],
    time_grid: NDArray[np.float64],
    initial_controls: Optional[Mapping[str, NDArray[np.float64]]] = None,
    intervention_schedule: Optional[Callable[[float], Iterable[float]]] = None,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    relaxation: float = 0.5,
    verbose: bool = True,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> SocialOptimumResult:
    """Solve the planner's PMP problem via forward-backward iteration."""

    time_grid = np.asarray(time_grid, dtype=np.float64)
    if time_grid.ndim != 1 or time_grid.size < 2:
        raise ValueError("time_grid must be a one-dimensional array with length ≥ 2.")
    if np.any(np.diff(time_grid) <= 0.0):
        raise ValueError("time_grid must be strictly increasing.")
    if not np.isclose(time_grid[0], 0.0, atol=1e-9):
        raise ValueError("time_grid must start at 0.")
    if not np.isclose(time_grid[-1], params.horizon, atol=1e-6):
        raise ValueError("time_grid must reach params.horizon.")
    if not (0.0 < relaxation <= 1.0):
        raise ValueError("relaxation must lie in (0, 1].")

    interaction_matrices = _ensure_matrices(matrices)
    controls = _initial_control_guess(params, time_grid, initial_controls)
    intervention_schedule = intervention_schedule or default_intervention_schedule(params)
    delta_path = _sample_schedule(time_grid, intervention_schedule)

    error_history: list[float] = []
    converged = False

    for iteration in range(1, max_iterations + 1):
        forward_result = _simulate_forward(
            params,
            interaction_matrices,
            controls,
            media_intensity,
            time_grid,
            intervention_schedule,
        )
        state_dict = _extract_state_dictionary(forward_result)
        media_path = _sample_schedule(time_grid, media_intensity)
        risk_inter, risk_media = _compute_risk_paths(
            params,
            interaction_matrices,
            state_dict,
            media_path,
        )
        susceptibles = np.vstack((state_dict["S_G"], state_dict["S_O"]))

        costate = _solve_costate_system(
            params,
            interaction_matrices,
            controls,
            risk_inter,
            risk_media,
            susceptibles,
            delta_path,
            time_grid,
            rtol,
            atol,
        )

        lambda_diff = np.vstack(
            (
                costate.lambda_I_G - costate.lambda_S_G,
                costate.lambda_I_O - costate.lambda_S_O,
            )
        )
        new_controls = {
            "beta_G": compute_optimal_beta_social(
                params,
                "G",
                risk_inter[0],
                susceptibles[0],
                lambda_diff[0],
            ),
            "mu_G": compute_optimal_mu_social(
                params,
                "G",
                risk_media[0],
                susceptibles[0],
                lambda_diff[0],
            ),
            "beta_O": compute_optimal_beta_social(
                params,
                "O",
                risk_inter[1],
                susceptibles[1],
                lambda_diff[1],
            ),
            "mu_O": compute_optimal_mu_social(
                params,
                "O",
                risk_media[1],
                susceptibles[1],
                lambda_diff[1],
            ),
        }

        diff = _control_sup_norm(controls, new_controls)
        error_history.append(diff)
        if verbose:
            print(f"Social iteration {iteration}: control sup norm = {diff:.3e}")

        if diff < tolerance:
            controls = new_controls
            converged = True
            state_dict_final = state_dict
            costate_final = costate
            risk_inter_final = risk_inter
            risk_media_final = risk_media
            break

        for key in controls:
            controls[key] = relaxation * new_controls[key] + (1.0 - relaxation) * controls[key]

        state_dict_final = state_dict
        costate_final = costate
        risk_inter_final = risk_inter
        risk_media_final = risk_media

    if not converged:
        raise RuntimeError("Social optimum solver failed to converge within max_iterations.")

    total_cost = compute_total_social_cost(params, state_dict_final, controls, time_grid)
    externality_path = _externality_trajectory(
        params,
        interaction_matrices,
        controls,
        susceptibles,
        costate_final,
    )

    convergence = {
        "iterations": len(error_history),
        "converged": True,
        "final_error": error_history[-1] if error_history else 0.0,
        "error_history": error_history,
    }

    return SocialOptimumResult(
        time=time_grid.copy(),
        controls={key: value.copy() for key, value in controls.items()},
        state=state_dict_final,
        costate=costate_final,
        convergence=convergence,
        total_cost=total_cost,
        risk_inter=risk_inter_final,
        risk_media=risk_media_final,
        externality=externality_path,
    )


def compare_nash_vs_social(
    params: ModelParameters,
    nash_result: Mapping[str, object],
    social_result: SocialOptimumResult,
) -> Mapping[str, float]:
    """Compare Nash and social outcomes computing PoA (must satisfy PoA ≥ 1)."""

    time_nash = np.asarray(nash_result["time"], dtype=np.float64)
    time_soc = social_result.time
    if time_nash.shape != time_soc.shape or not np.allclose(time_nash, time_soc):
        raise ValueError("Nash and social solutions must share the same time grid.")

    cost_nash = compute_total_social_cost(
        params,
        nash_result["state_nash"],
        nash_result["controls_nash"],
        time_nash,
    )
    cost_social = social_result.total_cost
    if cost_social <= 0.0:
        raise ValueError("Social optimum cost must be positive.")
    poa = max(1.0, cost_nash / cost_social)
    return {
        "cost_nash": cost_nash,
        "cost_social": cost_social,
        "poa": poa,
    }


# ---------------------------------------------------------------------------
# Internal helpers for adjoint integration
# ---------------------------------------------------------------------------

def _solve_costate_system(
    params: ModelParameters,
    matrices: InteractionMatrices,
    controls: Mapping[str, NDArray[np.float64]],
    risk_inter: NDArray[np.float64],
    risk_media: NDArray[np.float64],
    susceptibles: NDArray[np.float64],
    delta_path: NDArray[np.float64],
    time_grid: NDArray[np.float64],
    rtol: float,
    atol: float,
) -> CostateTrajectories:
    y_terminal = np.zeros(6, dtype=np.float64)
    solution = solve_ivp(
        lambda t, y: adjoint_system_ode(
            t,
            y,
            time_grid=time_grid,
            params=params,
            controls=controls,
            risk_inter=risk_inter,
            risk_media=risk_media,
            susceptibles=susceptibles,
            delta_path=delta_path,
            matrices=matrices,
        ),
        t_span=(time_grid[-1], time_grid[0]),
        y0=y_terminal,
        t_eval=time_grid[::-1],
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"Costate integration failed: {solution.message}")

    values = solution.y.T[::-1]
    t = solution.t[::-1]
    return CostateTrajectories(
        t=t,
        lambda_S_G=values[:, 0],
        lambda_I_G=values[:, 1],
        lambda_R_G=values[:, 2],
        lambda_S_O=values[:, 3],
        lambda_I_O=values[:, 4],
        lambda_R_O=values[:, 5],
    )


def _externality_trajectory(
    params: ModelParameters,
    matrices: InteractionMatrices,
    controls: Mapping[str, NDArray[np.float64]],
    susceptibles: NDArray[np.float64],
    costate: CostateTrajectories,
) -> Mapping[str, NDArray[np.float64]]:
    lambda_diff = np.vstack(
        (
            costate.lambda_I_G - costate.lambda_S_G,
            costate.lambda_I_O - costate.lambda_S_O,
        )
    )
    beta = np.vstack((controls["beta_G"], controls["beta_O"]))
    horizon_len = beta.shape[1]
    externality = np.empty((2, horizon_len), dtype=np.float64)
    for idx in range(horizon_len):
        externality[:, idx] = compute_externality_cost(
            params,
            matrices,
            susceptible=np.array(
                [susceptibles[0, idx], susceptibles[1, idx]],
                dtype=np.float64,
            ),
            lambda_diff=lambda_diff[:, idx],
            beta=np.array([beta[0, idx], beta[1, idx]], dtype=np.float64),
        )
    return {
        "E_G": externality[0],
        "E_O": externality[1],
    }
