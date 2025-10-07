r"""Hamilton-Jacobi-Bellman solver for the polarized Nash equilibrium."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .matrices import InteractionMatrices
from .parameters import GroupParameters, ModelParameters

GroupName = str


def _group_index(group: GroupName) -> int:
    if group.upper() == "G":
        return 0
    if group.upper() == "O":
        return 1
    raise ValueError("Group identifier must be 'G' or 'O'.")


def compute_interpersonal_risk_environment(
    I_G: float,
    I_O: float,
    N_G: float,
    N_O: float,
    omega: NDArray[np.float64],
    alpha: NDArray[np.float64],
    group: GroupName,
) -> float:
    r"""
    Compute interpersonal risk environment ``\mathcal{R}_k^{Inter}`` (Denklem 6).

    Parameters
    ----------
    I_G, I_O : float
        Group-level infection shares ``I_j(t)``.
    N_G, N_O : float
        Population shares ``N_j``.
    omega : ndarray shape (2, 2)
        Social separation matrix ``Ω``.
    alpha : ndarray shape (2, 2)
        Psychological bias matrix ``A``.
    group : {"G", "O"}
        Target group identifier ``k``.

    Returns
    -------
    float
        Interpersonal risk environment ``\mathcal{R}_k^{Inter}(t)``.
    """

    idx = _group_index(group)
    infected_ratios = np.array([I_G / N_G, I_O / N_O], dtype=np.float64)
    weights = omega[idx, :] * alpha[idx, :]
    risk = float(np.dot(weights, infected_ratios))
    return risk


def compute_media_risk_environment(
    M_G: float,
    M_O: float,
    pi: NDArray[np.float64],
    kappa: NDArray[np.float64],
    group: GroupName,
) -> float:
    r"""
    Compute media risk environment ``\mathcal{R}_k^{Media}`` (Denklem 7).

    Parameters
    ----------
    M_G, M_O : float
        Media misinformation intensities ``M_j(t)``.
    pi : ndarray shape (2, 2)
        Media preference matrix ``Π``.
    kappa : ndarray shape (2, 2)
        Media trust matrix ``K``.
    group : {"G", "O"}
        Target group identifier ``k``.

    Returns
    -------
    float
        Media risk environment ``\mathcal{R}_k^{Media}(t)``.
    """

    idx = _group_index(group)
    media_levels = np.array([M_G, M_O], dtype=np.float64)
    weights = pi[idx, :] * kappa[idx, :]
    risk = float(np.dot(weights, media_levels))
    return risk


def _project(value: float, lower: float, upper: float) -> float:
    return float(min(upper, max(lower, value)))


def _effort_cost(beta: float, mu: float, params: GroupParameters) -> float:
    return (
        params.w_beta * (params.beta_0 / beta - 1.0)
        + params.w_mu * (params.mu_0 / mu - 1.0)
    )


@dataclass(slots=True)
class BestResponse:
    """Stores optimal controls and implied exposure."""

    beta: float
    mu: float
    lambda_value: float
    effort_cost: float


def compute_best_response(
    params: GroupParameters,
    risk_inter: float,
    risk_media: float,
    delta_u_si: float,
) -> BestResponse:
    r"""Compute optimal controls using Proposition (\ref{prop:best_response_polarized})."""

    perceived_inter = risk_inter * delta_u_si
    perceived_media = risk_media * delta_u_si

    if perceived_inter <= 0.0:
        beta_star = params.beta_0
    else:
        candidate = np.sqrt((params.w_beta * params.beta_0) / max(perceived_inter, 1e-12))
        beta_star = _project(candidate, params.beta_min, params.beta_0)

    if perceived_media <= 0.0:
        mu_star = params.mu_0
    else:
        candidate = np.sqrt((params.w_mu * params.mu_0) / max(perceived_media, 1e-12))
        mu_star = _project(candidate, params.mu_min, params.mu_0)

    lambda_val = beta_star * risk_inter + mu_star * risk_media
    effort = _effort_cost(beta_star, mu_star, params)
    return BestResponse(beta=beta_star, mu=mu_star, lambda_value=lambda_val, effort_cost=effort)


@dataclass(slots=True)
class GroupHJBSolution:
    """Backward sweep results for a single cohort."""

    t: NDArray[np.float64]
    u_s: NDArray[np.float64]
    u_i: NDArray[np.float64]
    u_r: NDArray[np.float64]
    beta: NDArray[np.float64]
    mu: NDArray[np.float64]


@dataclass(slots=True)
class HJBSolution:
    """Aggregated value functions and controls for both groups."""

    groups: Dict[GroupName, GroupHJBSolution]


class HJBSolver:
    """
    Solve the polarized HJB system (Denklem (\ref{eq:HJB_Nash_Polarized})) for both groups.

    Parameters
    ----------
    params : ModelParameters
        Structural parameters including group-specific costs and bounds.
    matrices : InteractionMatrices
        Social and media matrices used for risk environment computations.
    time_grid : Iterable[float]
        Monotone grid covering ``[0, T]`` with ``T = params.horizon``.
    infected_path : ndarray shape (n, 2)
        Trajectory of ``(I_G(t), I_O(t))`` aligned with ``time_grid``.
    media_intensity : ndarray shape (n, 2)
        Trajectory of media intensities ``(M_G(t), M_O(t))``.
    delta_path : ndarray shape (n, 2), optional
        Exogenous intervention intensities ``δ_k(t)`` per group. Defaults to
        parameter constants when omitted.
    rtol, atol : float
        Integration tolerances for ``solve_ivp``.
    """

    def __init__(
        self,
        params: ModelParameters,
        matrices: InteractionMatrices,
        time_grid: Iterable[float],
        infected_path: NDArray[np.float64],
        media_intensity: NDArray[np.float64],
        delta_path: Optional[NDArray[np.float64]] = None,
        rtol: float = 1e-6,
        atol: float = 1e-9,
    ) -> None:
        self.params = params
        self.matrices = matrices
        self.time_grid = np.asarray(list(time_grid), dtype=np.float64)
        if self.time_grid.ndim != 1 or self.time_grid.size < 2:
            raise ValueError("time_grid must be a one-dimensional iterable with length ≥ 2.")
        if not np.all(np.diff(self.time_grid) > 0):
            raise ValueError("time_grid must be strictly increasing.")
        if not np.isclose(self.time_grid[0], 0.0, atol=1e-9):
            raise ValueError("time_grid must start at 0.")
        if not np.isclose(self.time_grid[-1], params.horizon, atol=1e-6):
            raise ValueError("time_grid must end at params.horizon.")

        self.infected_path = np.asarray(infected_path, dtype=np.float64)
        self.media_intensity = np.asarray(media_intensity, dtype=np.float64)
        if self.infected_path.shape != (self.time_grid.size, 2):
            raise ValueError("infected_path must have shape (n_time, 2).")
        if self.media_intensity.shape != (self.time_grid.size, 2):
            raise ValueError("media_intensity must have shape (n_time, 2).")

        if delta_path is None:
            self.delta_path = np.column_stack(
                (
                    np.full(self.time_grid.size, params.government.delta, dtype=np.float64),
                    np.full(self.time_grid.size, params.opposition.delta, dtype=np.float64),
                )
            )
        else:
            arr = np.asarray(delta_path, dtype=np.float64)
            if arr.shape != (self.time_grid.size, 2):
                raise ValueError("delta_path must have shape (n_time, 2).")
            self.delta_path = arr

        self.rtol = rtol
        self.atol = atol

        self._risk_inter = self._compute_risk_inter()
        self._risk_media = self._compute_risk_media()

    def _compute_risk_inter(self) -> NDArray[np.float64]:
        values = np.empty((2, self.time_grid.size), dtype=np.float64)
        for idx, group in enumerate(("G", "O")):
            for t_idx, (I_G, I_O) in enumerate(self.infected_path):
                values[idx, t_idx] = compute_interpersonal_risk_environment(
                    I_G,
                    I_O,
                    self.params.government.population_share,
                    self.params.opposition.population_share,
                    self.matrices.omega,
                    self.matrices.a,
                    group,
                )
        return values

    def _compute_risk_media(self) -> NDArray[np.float64]:
        values = np.empty((2, self.time_grid.size), dtype=np.float64)
        for idx, group in enumerate(("G", "O")):
            for t_idx, (M_G, M_O) in enumerate(self.media_intensity):
                values[idx, t_idx] = compute_media_risk_environment(
                    M_G,
                    M_O,
                    self.matrices.pi,
                    self.matrices.k,
                    group,
                )
        return values

    def solve(self) -> HJBSolution:
        solutions: Dict[GroupName, GroupHJBSolution] = {}
        for group in ("G", "O"):
            solutions[group] = self._solve_group(group)
        return HJBSolution(groups=solutions)

    def _solve_group(self, group: GroupName) -> GroupHJBSolution:
        idx = _group_index(group)
        group_params = self.params.government if idx == 0 else self.params.opposition
        risk_inter = self._risk_inter[idx, :]
        risk_media = self._risk_media[idx, :]
        delta_path = self.delta_path[:, idx]

        def interp(values: NDArray[np.float64], t: float) -> float:
            return float(np.interp(t, self.time_grid, values))

        def rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            u_s, u_i, u_r = y
            delta_u = u_i - u_s
            r_inter = interp(risk_inter, t)
            r_media = interp(risk_media, t)
            delta_t = interp(delta_path, t)
            best = compute_best_response(group_params, r_inter, r_media, delta_u)
            min_value = best.effort_cost + best.lambda_value * delta_u
            du_s = -(min_value + delta_t * (u_r - u_s))  # Denklem (4a)
            du_i = -(group_params.r_I + group_params.gamma * (u_r - u_i))  # Denklem (4b)
            du_r = -(group_params.rho * (u_s - u_r))  # Denklem (4c)
            return np.array([du_s, du_i, du_r], dtype=np.float64)

        y_terminal = np.zeros(3, dtype=np.float64)
        t_eval = self.time_grid[::-1]
        solution = solve_ivp(
            rhs,
            t_span=(self.time_grid[-1], self.time_grid[0]),
            y0=y_terminal,
            t_eval=t_eval,
            method="RK45",
            rtol=self.rtol,
            atol=self.atol,
        )
        if not solution.success:
            raise RuntimeError(f"HJB solver failed for group {group}: {solution.message}")

        values_desc = solution.y.T
        t_asc = solution.t[::-1]
        values = values_desc[::-1, :]

        beta = np.empty_like(t_asc)
        mu = np.empty_like(t_asc)
        for idx_t, (time, state_vals) in enumerate(zip(t_asc, values)):
            u_s, u_i, _ = state_vals
            delta_u = u_i - u_s
            r_inter = interp(risk_inter, time)
            r_media = interp(risk_media, time)
            best = compute_best_response(group_params, r_inter, r_media, delta_u)
            beta[idx_t] = best.beta
            mu[idx_t] = best.mu

        return GroupHJBSolution(
            t=t_asc,
            u_s=values[:, 0],
            u_i=values[:, 1],
            u_r=values[:, 2],
            beta=beta,
            mu=mu,
        )
