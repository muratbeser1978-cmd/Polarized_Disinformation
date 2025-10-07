"""Policy intervention scenario with mid-horizon countermeasures."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from polarized_disinformation.parameters import default_parameters
from polarized_disinformation.matrices import default_matrices
from polarized_disinformation.sirs_dynamics import (
    StateVector,
    initial_state_from_infections,
    simulate_sirs,
)


def control_policy(t: float, _: StateVector) -> np.ndarray:
    """Reduce controls after the intervention time."""

    if t < 25.0:
        return np.array([0.6, 0.5, 0.6, 0.5], dtype=np.float64)
    return np.array([0.3, 0.2, 0.3, 0.2], dtype=np.float64)


def intervention_schedule(t: float) -> np.ndarray:
    """Activate pre-bunking after the intervention threshold."""

    if t < 25.0:
        return np.zeros(2, dtype=np.float64)
    return np.array([0.05, 0.05], dtype=np.float64)


def main() -> None:
    params = default_parameters()
    matrices = default_matrices()
    x0 = initial_state_from_infections(params, initial_infected=(0.015, 0.015))
    media_schedule = lambda t: np.array([0.6, 0.4], dtype=np.float64)
    simulation = simulate_sirs(
        params,
        matrices,
        x0,
        control_policy=control_policy,
        media_schedule=media_schedule,
        intervention_schedule=intervention_schedule,
        t_span=(0.0, 50.0),
    )
    # Create a simple plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(simulation.t, simulation.x[:, 0], label='S_G')
    ax.plot(simulation.t, simulation.x[:, 1], label='I_G')
    ax.plot(simulation.t, simulation.x[:, 2], label='R_G')
    ax.plot(simulation.t, simulation.x[:, 3], label='S_O')
    ax.plot(simulation.t, simulation.x[:, 4], label='I_O')
    ax.plot(simulation.t, simulation.x[:, 5], label='R_O')
    ax.set_xlabel('Time')
    ax.set_ylabel('Population Share')
    ax.set_title('Policy Intervention SIRS Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "policy_intervention_sirs.png", dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_dir / 'policy_intervention_sirs.png'}")


if __name__ == "__main__":
    main()
