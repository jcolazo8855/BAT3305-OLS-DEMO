from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Coefficients are expressed in powers of z=x/3 so degrees remain comparable
# and numerically stable from the linear through quintic populations.
POPULATION_COEFFICIENTS = np.array([100.0, 28.0, -35.0, 25.0, 18.0, -15.0], dtype=float)


@dataclass(frozen=True)
class PolynomialData:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    population_degree: int
    x_scale: float


@dataclass(frozen=True)
class PolynomialModel:
    degree: int
    coefficients: np.ndarray
    x_scale: float


def polynomial_design(x: np.ndarray, degree: int, x_scale: float = 3.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    z = x / x_scale
    return np.vander(z, N=degree + 1, increasing=True)


def population_mean(x: np.ndarray, degree: int, x_scale: float = 3.0) -> np.ndarray:
    design = polynomial_design(x, degree, x_scale)
    return design @ POPULATION_COEFFICIENTS[: degree + 1]


def simulate_polynomial_data(
    *,
    seed: int,
    total_sample_size: int,
    train_percent: int,
    population_degree: int,
    noise_sd: float,
    x_scale: float = 3.0,
) -> PolynomialData:
    """Generate one-x/one-y polynomial data with a fixed held-out test sample."""
    if population_degree not in range(1, 6):
        raise ValueError("Population degree must be between 1 and 5.")
    rng = np.random.default_rng(seed)
    x_all = rng.uniform(-x_scale, x_scale, total_sample_size)
    y_all = population_mean(x_all, population_degree, x_scale) + rng.normal(
        0.0, noise_sd, total_sample_size
    )
    # The population is designed to remain positive. This guard prevents an
    # exceptionally large random error from making RMSLE or APE undefined.
    y_all = np.maximum(y_all, 1.0)

    shuffled = rng.permutation(total_sample_size)
    n_train = int(round(total_sample_size * train_percent / 100.0))
    n_train = min(max(n_train, 10), total_sample_size - 5)
    train_indices = shuffled[:n_train]
    test_indices = shuffled[n_train:]
    return PolynomialData(
        x_train=x_all[train_indices],
        x_test=x_all[test_indices],
        y_train=y_all[train_indices],
        y_test=y_all[test_indices],
        population_degree=population_degree,
        x_scale=x_scale,
    )


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int, x_scale: float = 3.0) -> PolynomialModel:
    if degree not in range(1, 6):
        raise ValueError("Model degree must be between 1 and 5.")
    design = polynomial_design(x, degree, x_scale)
    coefficients = np.linalg.lstsq(design, np.asarray(y, dtype=float), rcond=None)[0]
    return PolynomialModel(degree=degree, coefficients=coefficients, x_scale=x_scale)


def predict_polynomial(model: PolynomialModel, x: np.ndarray) -> np.ndarray:
    return polynomial_design(x, model.degree, model.x_scale) @ model.coefficients
