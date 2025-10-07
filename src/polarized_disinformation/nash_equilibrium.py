"""Nash equilibrium computation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, MutableMapping

import numpy as np
from numpy.typing import NDArray

from .hjb_solver import HJBSolution, HJBSolver
from .matrices import InteractionMatrices
from .parameters import ModelParameters
from .sirs_dynamics import SIRSSimulationResult, simulate_sirs

ControlVector = NDArray[np.float64]


@dataclass(slots=True)
class FixedPointResult:
    """Convergence metadata for classical fixed-point iteration."""

    control: ControlVector
    iterations: int
    converged: bool


BestResponse = Callable[[ControlVector], ControlVector]


def nash_fixed_point(
    initial: ControlVector,
    best_response: BestResponse,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> FixedPointResult:
    """Perform synchronous fixed-point iteration (auxiliary utility)."""

    current = np.array(initial, dtype=np.float64)
    for it in range(1, max_iter + 1):
        next_control = np.asarray(best_response(current), dtype=np.float64)
        if next_control.shape != current.shape:
            raise ValueError("Best response must return control vector of identical shape.")
        diff = np.linalg.norm(next_control - current, ord=np.inf)
        current = next_control
        if diff < tol:
            return FixedPointResult(control=current, iterations=it, converged=True)
    return FixedPointResult(control=current, iterations=max_iter, converged=False)


def compute_nash_equilibrium(
    params: ModelParameters,
    matrices: InteractionMatrices | Mapping[str, NDArray[np.float64]],
    media_intensity: Callable[[float], Iterable[float]],
    T: float,
    time_grid: NDArray[np.float64],
    initial_controls: Mapping[str, NDArray[np.float64]] | None = None,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    verbose: bool = True,
) -> Mapping[str, object]:
    """
    Compute the polarized mean-field Nash equilibrium via forward-backward iteration.

    Implements the FBODE loop described in the specification:

    1. Forward SIRS pass with the current strategy profile.
    2. Risk environment construction from the forward trajectory.
    3. Backward HJB sweep to obtain value functions.
    4. Point-wise best response update for controls.

    Iterates until the sup norm between successive control profiles falls below
    ``tolerance`` or until ``max_iterations`` is reached.
    """

    time_grid = np.asarray(time_grid, dtype=np.float64)
    if time_grid.ndim != 1 or time_grid.size < 2:
        raise ValueError("time_grid must be a one-dimensional array with length ≥ 2.")
    if np.any(np.diff(time_grid) <= 0.0):
        raise ValueError("time_grid must be strictly increasing.")
    if not np.isclose(time_grid[0], 0.0, atol=1e-9):
        raise ValueError("time_grid must start at t = 0.")
    if not np.isclose(time_grid[-1], T, atol=1e-6):
        raise ValueError("time_grid must end at T.")
    if not np.isclose(T, params.horizon, atol=1e-6):
        raise ValueError("T must match params.horizon for backward integration.")

    interaction_matrices = _ensure_matrices(matrices)
    control_paths = _initial_control_guess(params, time_grid, initial_controls)

    error_history: list[float] = []
    converged = False

    for iteration in range(max_iterations):
        forward_result = _simulate_forward(
            params,
            interaction_matrices,
            control_paths,
            media_intensity,
            T,
            time_grid,
        )
        infected_path = forward_result.x[:, [1, 4]]
        media_path = _sample_media(time_grid, media_intensity)

        hjb_solver = HJBSolver(
            params=params,
            matrices=interaction_matrices,
            time_grid=time_grid,
            infected_path=infected_path,
            media_intensity=media_path,
        )
        hjb_solution = hjb_solver.solve()

        new_control_paths = _extract_control_paths(hjb_solution)
        diff = _control_sup_norm(control_paths, new_control_paths)
        error_history.append(diff)
        if verbose:
            print(f"Iteration {iteration + 1}: control sup norm = {diff:.3e}")

        if diff < tolerance:
            control_paths = new_control_paths
            converged = True
            break

        control_paths = new_control_paths

    if not converged:
        raise RuntimeError(
            "Nash equilibrium iteration failed to converge within max_iterations."
        )

    forward_result = _simulate_forward(
        params,
        interaction_matrices,
        control_paths,
        media_intensity,
        T,
        time_grid,
    )
    infected_path = forward_result.x[:, [1, 4]]
    media_path = _sample_media(time_grid, media_intensity)
    hjb_solver = HJBSolver(
        params=params,
        matrices=interaction_matrices,
        time_grid=time_grid,
        infected_path=infected_path,
        media_intensity=media_path,
    )
    hjb_solution = hjb_solver.solve()

    result = {
        "controls_nash": {
            "beta_G": control_paths["beta_G"].copy(),
            "mu_G": control_paths["mu_G"].copy(),
            "beta_O": control_paths["beta_O"].copy(),
            "mu_O": control_paths["mu_O"].copy(),
        },
        "state_nash": _state_dictionary(forward_result),
        "value_functions": _value_dictionary(hjb_solution),
        "convergence": {
            "iterations": len(error_history),
            "converged": True,
            "final_error": error_history[-1] if error_history else 0.0,
            "error_history": error_history,
        },
        "time": time_grid.copy(),
    }
    if verbose:
        print(
            f"Nash equilibrium converged in {len(error_history)} iterations "
            f"with final error {error_history[-1] if error_history else 0.0:.3e}."
        )
    return result


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
    initial_controls: Mapping[str, NDArray[np.float64]] | None,
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

        if "beta" in key:
            bounds = (
                params.government.beta_min, params.government.beta_0
            ) if key.endswith("G") else (
                params.opposition.beta_min, params.opposition.beta_0
            )
        else:
            bounds = (
                params.government.mu_min, params.government.mu_0
            ) if key.endswith("G") else (
                params.opposition.mu_min, params.opposition.mu_0
            )
        control_paths[key] = np.clip(arr, bounds[0], bounds[1])
    return control_paths


def _simulate_forward(
    params: ModelParameters,
    matrices: InteractionMatrices,
    controls: Mapping[str, NDArray[np.float64]],
    media_intensity: Callable[[float], Iterable[float]],
    T: float,
    time_grid: NDArray[np.float64],
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

    x0 = _default_initial_state(params)

    return simulate_sirs(
        params=params,
        matrices=matrices,
        x0=x0,
        control_policy=control_policy,
        media_schedule=media_schedule,
        t_span=(0.0, T),
        time_grid=time_grid,
    )


def _sample_media(
    time_grid: NDArray[np.float64],
    media_intensity: Callable[[float], Iterable[float]],
) -> NDArray[np.float64]:
    path = np.empty((time_grid.size, 2), dtype=np.float64)
    for idx, t in enumerate(time_grid):
        values = np.asarray(media_intensity(float(t)), dtype=np.float64)
        if values.shape != (2,):
            raise ValueError("media_intensity must return an iterable of length 2.")
        path[idx] = values
    return path


def _extract_control_paths(hjb_solution: HJBSolution) -> MutableMapping[str, NDArray[np.float64]]:
    beta_g = hjb_solution.groups["G"].beta
    mu_g = hjb_solution.groups["G"].mu
    beta_o = hjb_solution.groups["O"].beta
    mu_o = hjb_solution.groups["O"].mu
    return {
        "beta_G": beta_g.copy(),
        "mu_G": mu_g.copy(),
        "beta_O": beta_o.copy(),
        "mu_O": mu_o.copy(),
    }


def _control_sup_norm(
    current: Mapping[str, NDArray[np.float64]],
    candidate: Mapping[str, NDArray[np.float64]],
) -> float:
    diffs = []
    for key in ("beta_G", "mu_G", "beta_O", "mu_O"):
        diffs.append(np.max(np.abs(candidate[key] - current[key])))
    return float(np.max(diffs))


def _default_initial_state(params: ModelParameters) -> NDArray[np.float64]:
    fraction = 1e-4
    i_g = min(fraction, 0.1 * params.government.population_share)
    i_o = min(2 * fraction, 0.1 * params.opposition.population_share)
    r_g = r_o = 0.0
    s_g = params.government.population_share - i_g
    s_o = params.opposition.population_share - i_o
    return np.array([s_g, i_g, r_g, s_o, i_o, r_o], dtype=np.float64)


def _state_dictionary(result: SIRSSimulationResult) -> Mapping[str, NDArray[np.float64]]:
    return {
        "S_G": result.x[:, 0].copy(),
        "I_G": result.x[:, 1].copy(),
        "R_G": result.x[:, 2].copy(),
        "S_O": result.x[:, 3].copy(),
        "I_O": result.x[:, 4].copy(),
        "R_O": result.x[:, 5].copy(),
    }


def _value_dictionary(hjb_solution: HJBSolution) -> Mapping[str, NDArray[np.float64]]:
    group_g = hjb_solution.groups["G"]
    group_o = hjb_solution.groups["O"]
    return {
        "u_S_G": group_g.u_s.copy(),
        "u_I_G": group_g.u_i.copy(),
        "u_R_G": group_g.u_r.copy(),
        "u_S_O": group_o.u_s.copy(),
        "u_I_O": group_o.u_i.copy(),
        "u_R_O": group_o.u_r.copy(),
    }
