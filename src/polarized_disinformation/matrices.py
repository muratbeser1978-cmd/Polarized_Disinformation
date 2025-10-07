"""Matrix utilities for the polarized disinformation model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from .parameters import ParameterValidationError

Matrix2x2 = NDArray[np.float64]


@dataclass(slots=True)
class InteractionMatrices:
    """
    Aggregates structural 2x2 matrices used across the model.

    Attributes
    ----------
    omega : Matrix2x2
        Social separation matrix Ω with stochastic rows.
    a : Matrix2x2
        Psychological bias matrix A, entries in [0, 1].
    pi : Matrix2x2
        Media preference matrix Π with stochastic rows.
    k : Matrix2x2
        Media trust matrix K, entries in [0, 1].
    """

    omega: Matrix2x2
    a: Matrix2x2
    pi: Matrix2x2
    k: Matrix2x2

    def __post_init__(self) -> None:
        self.omega = validate_stochastic(self.omega, name="Ω")
        self.pi = validate_stochastic(self.pi, name="Π")
        self.a = validate_unit_interval(self.a, name="A")
        self.k = validate_unit_interval(self.k, name="K")


def ensure_shape(matrix: NDArray[np.float64], name: str) -> Matrix2x2:
    """Ensure the matrix is 2×2."""

    if matrix.shape != (2, 2):
        raise ParameterValidationError(f"Matrix {name} must be 2x2.")
    return matrix.astype(np.float64, copy=False)


def validate_stochastic(matrix: NDArray[np.float64], name: str) -> Matrix2x2:
    """Validate 2×2 row-stochastic matrix within numerical tolerance."""

    matrix = ensure_shape(matrix, name)
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-10):
        raise ParameterValidationError(f"Rows of {name} must sum to 1 (±1e-10).")
    if np.any(matrix < 0.0):
        raise ParameterValidationError(f"Matrix {name} cannot have negative entries.")
    return matrix


def validate_unit_interval(matrix: NDArray[np.float64], name: str) -> Matrix2x2:
    """Validate a 2×2 matrix with entries restricted to [0, 1]."""

    matrix = ensure_shape(matrix, name)
    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ParameterValidationError(f"Entries of {name} must be in [0, 1].")
    return matrix


def default_matrices() -> InteractionMatrices:
    """Provide a baseline matrix configuration consistent with the baseline scenario."""

    omega = np.array([[0.7, 0.3], [0.3, 0.7]], dtype=np.float64)
    a = np.array([[1.0, 0.2], [0.2, 1.0]], dtype=np.float64)
    pi = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    k = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=np.float64)
    return InteractionMatrices(omega=omega, a=a, pi=pi, k=k)


def apply_media_shift(matrix: Matrix2x2, delta: Tuple[float, float]) -> Matrix2x2:
    """
    Adjust media preference matrix rows with a small shift while renormalising.

    Parameters
    ----------
    matrix : Matrix2x2
        Baseline 2×2 matrix to perturb.
    delta : tuple[float, float]
        Row-wise shift applied to the first column; the second column adjusts to preserve
        the stochastic constraint.
    """

    perturbed = ensure_shape(matrix.copy(), name="Π")
    for row, shift in enumerate(delta):
        perturbed[row, 0] = np.clip(perturbed[row, 0] + shift, 0.0, 1.0)
        perturbed[row, 1] = 1.0 - perturbed[row, 0]
    return validate_stochastic(perturbed, name="Π")
