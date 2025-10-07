"""High polarization scenario highlighting echo chambers."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from polarized_disinformation.parameters import default_parameters
from polarized_disinformation.matrices import InteractionMatrices, default_matrices
from polarized_disinformation.sirs_dynamics import (
    initial_state_from_infections,
    simulate_sirs,
)
import matplotlib.pyplot as plt


def high_polarization_matrices() -> InteractionMatrices:
    base = default_matrices()
    omega = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=np.float64)
    pi = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=np.float64)
    return InteractionMatrices(omega=omega, a=base.a, pi=pi, k=base.k)


def main() -> None:
    params = default_parameters()
    matrices = high_polarization_matrices()
    x0 = initial_state_from_infections(params, initial_infected=(0.02, 0.03))
    media_schedule = lambda t: np.array([0.8, 0.2], dtype=np.float64)
    simulation = simulate_sirs(
        params,
        matrices,
        x0,
        media_schedule=media_schedule,
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
    ax.set_title('High Polarization SIRS Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "high_polarization_sirs.png", dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_dir / 'high_polarization_sirs.png'}")


if __name__ == "__main__":
    main()
