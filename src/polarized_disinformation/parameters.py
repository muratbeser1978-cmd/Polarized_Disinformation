"""Parameter definitions and validation utilities for the diffusion model."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping
import json

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore


class ParameterValidationError(ValueError):
    """Raised when model parameters violate feasibility constraints."""


@dataclass(slots=True)
class GroupParameters:
    """
    Cohort-specific parameters for the polarized SIRS diffusion model.

    Parameters
    ----------
    population_share : float
        Normalised population share ``N_k`` such that ``N_G + N_O = 1``.
    gamma : float
        Recovery rate ``γ_k`` (Denklem (4b)/(4e)).
    rho : float
        Immunity loss rate ``ρ_k`` (Denklem (4a)/(4d)).
    delta : float
        Baseline pre-bunking intensity ``δ_k(t)`` (Denklem (4a)/(4d)).
    beta_min : float
        Lowest feasible interpersonal interaction rate ``β_{min,k}``.
    beta_0 : float
        Default interaction rate ``β_{0,k}`` (upper bound, no effort).
    mu_min : float
        Lowest feasible media consumption rate ``μ_{min,k}``.
    mu_0 : float
        Default media consumption rate ``μ_{0,k}`` (upper bound, no effort).
    w_beta : float
        Effort cost weight for reducing ``β_k`` (Denklem (C_e)).
    w_mu : float
        Effort cost weight for reducing ``μ_k`` (Denklem (C_e)).
    r_I : float
        Instantaneous infection cost ``r_{I,k}`` (Denklem ``C_I``).
    """

    population_share: float
    gamma: float
    rho: float
    delta: float
    beta_min: float
    beta_0: float
    mu_min: float
    mu_0: float
    w_beta: float
    w_mu: float
    r_I: float

    def __post_init__(self) -> None:
        if not (0.0 < self.population_share <= 1.0):
            raise ParameterValidationError("Population shares must lie in (0, 1].")
        if self.gamma <= 0.0:
            raise ParameterValidationError("Recovery rates γ_k must be positive.")
        if self.rho < 0.0:
            raise ParameterValidationError("Immunity loss rates ρ_k must be non-negative.")
        if self.delta < 0.0:
            raise ParameterValidationError("Intervention intensities δ_k must be non-negative.")
        if not (0.0 < self.beta_min < self.beta_0):
            raise ParameterValidationError("β_min must satisfy 0 < β_min < β_0.")
        if not (0.0 < self.mu_min < self.mu_0):
            raise ParameterValidationError("μ_min must satisfy 0 < μ_min < μ_0.")
        for name, value in {
            "w_beta": self.w_beta,
            "w_mu": self.w_mu,
            "r_I": self.r_I,
        }.items():
            if value < 0.0:
                raise ParameterValidationError(f"Cost parameter {name} must be non-negative.")


@dataclass(slots=True)
class ModelParameters:
    """Container aggregating both cohorts' parameters and global settings."""

    government: GroupParameters
    opposition: GroupParameters
    discount_rate: float
    horizon: float
    seed: int = field(default=0)

    def __post_init__(self) -> None:
        pop_total = self.government.population_share + self.opposition.population_share
        if abs(pop_total - 1.0) > 1e-8:
            raise ParameterValidationError("Population shares must sum to 1.")
        if self.discount_rate <= 0.0:
            raise ParameterValidationError("Discount rate must be strictly positive.")
        if self.horizon <= 0.0:
            raise ParameterValidationError("Time horizon must be strictly positive.")
        if self.seed < 0:
            raise ParameterValidationError("Random seed must be non-negative.")

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view of the parameters."""

        def encode_group(group: GroupParameters) -> Dict[str, Any]:
            return {
                "population_share": group.population_share,
                "gamma": group.gamma,
                "rho": group.rho,
                "delta": group.delta,
                "beta_min": group.beta_min,
                "beta_0": group.beta_0,
                "mu_min": group.mu_min,
                "mu_0": group.mu_0,
                "w_beta": group.w_beta,
                "w_mu": group.w_mu,
                "r_I": group.r_I,
            }

        return {
            "government": encode_group(self.government),
            "opposition": encode_group(self.opposition),
            "discount_rate": self.discount_rate,
            "horizon": self.horizon,
            "seed": self.seed,
        }

    def save(self, path: Path) -> None:
        """Persist parameters to JSON or YAML, inferred from file suffix."""

        payload = self.to_dict()
        suffix = path.suffix.lower()
        if suffix == ".json":
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        elif suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is required to write YAML files.")
            path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        else:
            raise ValueError("Unsupported parameter file extension.")


def _build_group(group_cfg: Mapping[str, Any]) -> GroupParameters:
    """Create a ``GroupParameters`` instance from mapping input."""

    required_keys = {
        "population_share",
        "gamma",
        "rho",
        "delta",
        "beta_min",
        "beta_0",
        "mu_min",
        "mu_0",
        "w_beta",
        "w_mu",
        "r_I",
    }
    missing = required_keys - set(group_cfg)
    if missing:
        raise ParameterValidationError(f"Missing group parameters: {sorted(missing)}")

    return GroupParameters(
        population_share=float(group_cfg["population_share"]),
        gamma=float(group_cfg["gamma"]),
        rho=float(group_cfg["rho"]),
        delta=float(group_cfg["delta"]),
        beta_min=float(group_cfg["beta_min"]),
        beta_0=float(group_cfg["beta_0"]),
        mu_min=float(group_cfg["mu_min"]),
        mu_0=float(group_cfg["mu_0"]),
        w_beta=float(group_cfg["w_beta"]),
        w_mu=float(group_cfg["w_mu"]),
        r_I=float(group_cfg["r_I"]),
    )


def load_parameters(path: Path) -> ModelParameters:
    """Load parameters from JSON or YAML configuration files."""

    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML files.")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("Unsupported parameter file extension.")

    if not isinstance(data, Mapping):
        raise ParameterValidationError("Root of parameter file must be a mapping.")

    return ModelParameters(
        government=_build_group(data["government"]),
        opposition=_build_group(data["opposition"]),
        discount_rate=float(data["discount_rate"]),
        horizon=float(data["horizon"]),
        seed=int(data.get("seed", 0)),
    )


def default_parameters() -> ModelParameters:
    """Return a baseline parameter set for moderate polarization scenarios."""

    government = GroupParameters(
        population_share=0.5,
        gamma=0.2,
        rho=0.05,
        delta=0.0,
        beta_min=0.2,
        beta_0=0.6,
        mu_min=0.1,
        mu_0=0.5,
        w_beta=1.0,
        w_mu=1.2,
        r_I=5.0,
    )
    opposition = GroupParameters(
        population_share=0.5,
        gamma=0.2,
        rho=0.05,
        delta=0.0,
        beta_min=0.2,
        beta_0=0.6,
        mu_min=0.1,
        mu_0=0.5,
        w_beta=1.1,
        w_mu=1.3,
        r_I=5.5,
    )
    return ModelParameters(
        government=government,
        opposition=opposition,
        discount_rate=0.03,
        horizon=100.0,
        seed=42,
    )
