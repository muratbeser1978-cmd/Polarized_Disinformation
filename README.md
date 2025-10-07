# Polarized Strategic Disinformation Diffusion Model

This repository implements a polarized SIRS-based diffusion model that captures strategic behaviour of two antagonistic groups (pro-government `G` and opposition `O`). It couples forward epidemiological dynamics with backward optimal-control solvers to study both decentralized (Nash) and centralized (social planner) responses to disinformation risk.

## Key Features

- **Modular architecture**: parameters, matrices, SIRS dynamics, Nash solver, social planner (PMP), analysis, and visualization modules under `polarized_disinformation/src/polarized_disinformation/`.
- **Mean-field Nash equilibrium**: forward–backward solver couples the SIRS system with HJB equations to find consistent controls.
- **Social optimum**: Pontryagin Maximum Principle implementation internalises group-internal and cross-group externalities.
- **Price of Anarchy analysis**: comparative utilities, control gaps, infection peak shifts, heatmaps, and summary statistics.
- **Extensive visualizations**: time series, control vs. risk panels, heatmaps, externality surfaces, and summary tables saved as PDF.
- **Comprehensive test suite**: conservation laws, matrix validation, numerical convergence, economic consistency, edge cases, and integration pipelines.

## Installation

```bash
# create and activate a virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# install the package (editable mode during development)
pip install -e .

# optional: install test extras
pip install -e .[tests]
```

## Running the Pipeline

```bash
# run the end-to-end analysis; results stored under outputs/
polarized-disinformation --output outputs/

# options
polarized-disinformation --config configs/default_params.yaml \
                         --time-steps 150 \
                         --no-figures
```

The command produces:

- `outputs/dashboard_*.pdf`: collection of technical plots (time series, heatmaps, cost/risk panels, statistics table).
- `outputs/summary_report.txt`: textual summary including Price of Anarchy, control gaps, infection peak shifts, and core statistics.

## Testing

```bash
pytest          # run all tests
pytest tests/   # explicit tests directory
pytest --cov=polarized_disinformation --cov-report=html
```

## Project Layout

```
polarized_disinformation/
├── src/polarized_disinformation/
│   ├── analysis.py
│   ├── matrices.py
│   ├── nash_equilibrium.py
│   ├── parameters.py
│   ├── sirs_dynamics.py
│   ├── social_optimum.py
│   └── visualization.py
├── tests/
├── configs/default_params.yaml
├── main.py
├── pyproject.toml
└── setup.py
```

## License

MIT License (see `LICENSE` if provided).

## Acknowledgements

Model specification (Teorem, Önerme, denklemler) has been faithfully implemented in the codebase.
