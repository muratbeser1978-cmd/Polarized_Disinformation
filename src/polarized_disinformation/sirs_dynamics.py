"""Forward SIRS dynamics incorporating interpersonal and media transmission."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .matrices import InteractionMatrices
from .parameters import ModelParameters

StateVector = NDArray[np.float64]
ControlVector = NDArray[np.float64]


class ControlPolicy(Protocol):
    """Callable interface delivering controls ``[β_G, μ_G, β_O, μ_O]``."""

    def __call__(self, t: float, state: StateVector) -> ControlVector:
        """Return admissible control vector at time ``t``."""


class MediaSchedule(Protocol):
    """Callable returning media intensities ``[M_G(t), M_O(t)]``."""

    def __call__(self, t: float) -> NDArray[np.float64]:
        """Return non-negative media intensities for both outlets."""


class InterventionSchedule(Protocol):
    """Callable returning pre-bunking intensities ``[δ_G(t), δ_O(t)]``."""

    def __call__(self, t: float) -> NDArray[np.float64]:
        """Return intervention rates applied to susceptible cohorts."""


@dataclass(slots=True)
class SIRSSimulationResult:
    """Holds dense time grid and state trajectories from the forward solver."""

    t: NDArray[np.float64]
    x: NDArray[np.float64]

    def conservation_error(self, params: ModelParameters) -> float:
        """Return maximum deviation from population conservation."""

        gov_total = self.x[:, 0] + self.x[:, 1] + self.x[:, 2]
        opp_total = self.x[:, 3] + self.x[:, 4] + self.x[:, 5]
        err_g = np.max(np.abs(gov_total - params.government.population_share))
        err_o = np.max(np.abs(opp_total - params.opposition.population_share))
        return max(err_g, err_o)


def _control_bounds(params: ModelParameters) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    beta_bounds = (
        (params.government.beta_min, params.government.beta_0),
        (params.opposition.beta_min, params.opposition.beta_0),
    )
    mu_bounds = (
        (params.government.mu_min, params.government.mu_0),
        (params.opposition.mu_min, params.opposition.mu_0),
    )
    return beta_bounds, mu_bounds


def _clip_controls(control: ControlVector, params: ModelParameters) -> ControlVector:
    beta_bounds, mu_bounds = _control_bounds(params)
    beta_g = np.clip(control[0], *beta_bounds[0])
    mu_g = np.clip(control[1], *mu_bounds[0])
    beta_o = np.clip(control[2], *beta_bounds[1])
    mu_o = np.clip(control[3], *mu_bounds[1])
    return np.array([beta_g, mu_g, beta_o, mu_o], dtype=np.float64)


def default_control_policy(params: ModelParameters) -> ControlPolicy:
    """Return a policy holding controls at their default (no-effort) levels."""

    def policy(_: float, __: StateVector) -> ControlVector:
        return np.array(
            [
                params.government.beta_0,
                params.government.mu_0,
                params.opposition.beta_0,
                params.opposition.mu_0,
            ],
            dtype=np.float64,
        )

    return policy


def default_media_schedule(_: float) -> NDArray[np.float64]:
    """Zero media intensity baseline (can be overridden per scenario)."""

    return np.zeros(2, dtype=np.float64)


def default_intervention_schedule(params: ModelParameters) -> InterventionSchedule:
    """Create a schedule that applies constant δ_k values from parameters."""

    def schedule(_: float) -> NDArray[np.float64]:
        return np.array(
            [params.government.delta, params.opposition.delta],
            dtype=np.float64,
        )

    return schedule


def infection_force(
    t: float,
    state: StateVector,
    params: ModelParameters,
    matrices: InteractionMatrices,
    controls: ControlVector,
    media_schedule: MediaSchedule,
) -> NDArray[np.float64]:
    """
    Compute total infection forces ``Λ_k(t)`` (Denklem (2)).

    The interpersonal component implements ``Λ_k^{Inter}`` (Denklem (\ref{eq:lambda_inter}))
    while the media component follows ``Λ_k^{Media}`` (Denklem (\ref{eq:lambda_media})).
    """

    beta_g, mu_g, beta_o, mu_o = controls
    beta = np.array([beta_g, beta_o], dtype=np.float64)
    mu = np.array([mu_g, mu_o], dtype=np.float64)

    i_g, i_o = state[1], state[4]
    populations = np.array(
        [params.government.population_share, params.opposition.population_share],
        dtype=np.float64,
    )
    infected_ratio = np.array([i_g, i_o], dtype=np.float64) / populations

    interpersonal_weights = matrices.omega * matrices.a
    interpersonal_exposure = interpersonal_weights @ infected_ratio
    lambda_inter = beta * interpersonal_exposure

    media_levels = np.asarray(media_schedule(t), dtype=np.float64)
    if media_levels.shape != (2,):
        raise ValueError("Media schedule must return a length-2 vector.")
    media_weights = matrices.pi * matrices.k
    media_exposure = media_weights @ media_levels
    lambda_media = mu * media_exposure

    return lambda_inter + lambda_media


def sirs_rhs(
    t: float,
    state: StateVector,
    params: ModelParameters,
    matrices: InteractionMatrices,
    control_policy: ControlPolicy,
    media_schedule: MediaSchedule,
    intervention_schedule: InterventionSchedule,
) -> StateVector:
    """Evaluate the coupled SIRS right-hand side (Denklem (4a)-(4f))."""

    controls = _clip_controls(control_policy(t, state), params)
    lambda_k = infection_force(t, state, params, matrices, controls, media_schedule)
    delta_vals = np.asarray(intervention_schedule(t), dtype=np.float64)
    if delta_vals.shape != (2,):
        raise ValueError("Intervention schedule must return a length-2 vector.")

    s_g, i_g, r_g, s_o, i_o, r_o = state
    lambda_g, lambda_o = lambda_k
    delta_g, delta_o = delta_vals

    gov = params.government
    opp = params.opposition

    ds_g = -lambda_g * s_g + gov.rho * r_g - delta_g * s_g  # Denklem (4a)
    di_g = lambda_g * s_g - gov.gamma * i_g  # Denklem (4b)
    dr_g = gov.gamma * i_g - gov.rho * r_g + delta_g * s_g  # Denklem (4c)

    ds_o = -lambda_o * s_o + opp.rho * r_o - delta_o * s_o  # Denklem (4d)
    di_o = lambda_o * s_o - opp.gamma * i_o  # Denklem (4e)
    dr_o = opp.gamma * i_o - opp.rho * r_o + delta_o * s_o  # Denklem (4f)

    return np.array([ds_g, di_g, dr_g, ds_o, di_o, dr_o], dtype=np.float64)


def initial_state_from_infections(
    params: ModelParameters,
    initial_infected: Tuple[float, float],
    recovered_share: Tuple[float, float] | None = None,
) -> StateVector:
    """Construct an initial state satisfying group-level conservation."""

    rec = recovered_share or (0.0, 0.0)
    i_g, i_o = initial_infected
    r_g, r_o = rec
    s_g = params.government.population_share - i_g - r_g
    s_o = params.opposition.population_share - i_o - r_o
    if min(s_g, s_o) < -1e-12:
        raise ValueError("Initial S, I, R must not exceed group population shares.")
    return np.array([s_g, i_g, r_g, s_o, i_o, r_o], dtype=np.float64)


def simulate_sirs(
    params: ModelParameters,
    matrices: InteractionMatrices,
    x0: StateVector,
    control_policy: ControlPolicy | None = None,
    media_schedule: MediaSchedule | None = None,
    intervention_schedule: InterventionSchedule | None = None,
    t_span: Tuple[float, float] | None = None,
    time_grid: NDArray[np.float64] | None = None,
    num_points: int = 200,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> SIRSSimulationResult:
    """Integrate the SIRS system using ``solve_ivp`` with adaptive stepping."""

    if control_policy is None:
        control_policy = default_control_policy(params)
    if media_schedule is None:
        media_schedule = default_media_schedule
    if intervention_schedule is None:
        intervention_schedule = default_intervention_schedule(params)
    if t_span is None:
        t_span = (0.0, params.horizon)

    if time_grid is not None:
        t_eval = np.asarray(time_grid, dtype=np.float64)
        if t_eval.ndim != 1 or t_eval.size < 2:
            raise ValueError("time_grid must be a one-dimensional array with length ≥ 2.")
        if not np.isclose(t_eval[0], t_span[0], atol=1e-9):
            raise ValueError("time_grid must start at the beginning of t_span.")
        if not np.isclose(t_eval[-1], t_span[1], atol=1e-6):
            raise ValueError("time_grid must end at the end of t_span.")
        if np.any(np.diff(t_eval) <= 0.0):
            raise ValueError("time_grid must be strictly increasing.")
    else:
        t_eval = np.linspace(t_span[0], t_span[1], num_points, dtype=np.float64)

    def rhs(t: float, state: StateVector) -> StateVector:
        return sirs_rhs(
            t,
            state,
            params,
            matrices,
            control_policy,
            media_schedule,
            intervention_schedule,
        )

    solution = solve_ivp(
        rhs,
        t_span=t_span,
        y0=x0,
        t_eval=t_eval,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"SIRS simulation failed: {solution.message}")

    trajectory = solution.y.T
    result = SIRSSimulationResult(t=solution.t, x=trajectory)
    if result.conservation_error(params) > 1e-6:
        raise RuntimeError("Population conservation violated beyond tolerance.")
    if np.any(result.x < -1e-9):
        raise RuntimeError("State trajectory produced negative population shares.")
    return result
