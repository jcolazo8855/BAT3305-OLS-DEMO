from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FEATURE_NAMES = (
    "Advertising",
    "Price index",
    "Promotion",
    "Website visits",
    "Competitor price",
    "Seasonality",
)

BASE_TRUE_COEFFICIENTS = np.array([0.45, -0.38, 0.31, 0.24, -0.19, 0.14], dtype=float)


@dataclass(frozen=True)
class SimulationData:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    true_coefficients: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class PreparedData:
    x_train_standardized: np.ndarray
    x_test_standardized: np.ndarray
    log_y_train: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    log_y_mean: float


@dataclass(frozen=True)
class RegularizedModel:
    kind: str
    alpha: float
    intercept: float
    coefficients: np.ndarray
    iterations: int


def simulate_business_data(
    *,
    seed: int,
    total_sample_size: int,
    train_percent: int,
    relevant_predictors: int,
    correlation: float,
    noise_sd: float,
) -> SimulationData:
    """Create a positive sales-like outcome with correlated numeric predictors."""
    rng = np.random.default_rng(seed)
    predictor_count = len(FEATURE_NAMES)
    indices = np.arange(predictor_count)
    covariance = correlation ** np.abs(indices[:, None] - indices[None, :])
    x_all = rng.multivariate_normal(
        mean=np.zeros(predictor_count),
        cov=covariance,
        size=total_sample_size,
    )

    true_coefficients = BASE_TRUE_COEFFICIENTS.copy()
    true_coefficients[relevant_predictors:] = 0.0
    log_sales = 4.5 + x_all @ true_coefficients + rng.normal(0.0, noise_sd, total_sample_size)
    # A positive target makes RMSLE and APE well-defined. The lower guard is only
    # a numerical safety net for extreme student-selected settings.
    y_all = np.expm1(np.maximum(log_sales, 0.1))

    shuffled = rng.permutation(total_sample_size)
    n_train = int(round(total_sample_size * train_percent / 100.0))
    n_train = min(max(n_train, predictor_count + 2), total_sample_size - 5)
    train_indices = shuffled[:n_train]
    test_indices = shuffled[n_train:]

    return SimulationData(
        x_train=x_all[train_indices],
        x_test=x_all[test_indices],
        y_train=y_all[train_indices],
        y_test=y_all[test_indices],
        true_coefficients=true_coefficients,
        feature_names=FEATURE_NAMES,
    )


def prepare_data(data: SimulationData) -> PreparedData:
    x_mean = np.mean(data.x_train, axis=0)
    x_scale = np.std(data.x_train, axis=0, ddof=0)
    x_scale = np.where(x_scale < 1e-12, 1.0, x_scale)
    x_train_standardized = (data.x_train - x_mean) / x_scale
    x_test_standardized = (data.x_test - x_mean) / x_scale
    log_y_train = np.log1p(data.y_train)
    log_y_mean = float(np.mean(log_y_train))
    return PreparedData(
        x_train_standardized=x_train_standardized,
        x_test_standardized=x_test_standardized,
        log_y_train=log_y_train,
        x_mean=x_mean,
        x_scale=x_scale,
        log_y_mean=log_y_mean,
    )


def soft_threshold(value: float, threshold: float) -> float:
    return float(np.sign(value) * max(abs(value) - threshold, 0.0))


def fit_regularized(
    prepared: PreparedData,
    *,
    kind: str,
    alpha: float,
    max_iter: int = 10_000,
    tolerance: float = 1e-9,
) -> RegularizedModel:
    """Fit OLS, Ridge, or LASSO to standardized X and centered log1p(y)."""
    x = prepared.x_train_standardized
    centered_y = prepared.log_y_train - prepared.log_y_mean
    n, p = x.shape

    if kind == "OLS":
        coefficients = np.linalg.lstsq(x, centered_y, rcond=None)[0]
        iterations = 1
    elif kind == "Ridge":
        gram = (x.T @ x) / n + alpha * np.eye(p)
        coefficients = np.linalg.solve(gram, (x.T @ centered_y) / n)
        iterations = 1
    elif kind == "LASSO":
        coefficients = np.zeros(p, dtype=float)
        column_norms = np.mean(x**2, axis=0)
        for iteration in range(1, max_iter + 1):
            previous = coefficients.copy()
            for j in range(p):
                partial_residual = centered_y - x @ coefficients + x[:, j] * coefficients[j]
                score = float(np.mean(x[:, j] * partial_residual))
                coefficients[j] = soft_threshold(score, alpha) / max(column_norms[j], 1e-12)
            if float(np.max(np.abs(coefficients - previous))) < tolerance:
                break
        iterations = iteration
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    return RegularizedModel(
        kind=kind,
        alpha=float(alpha),
        intercept=prepared.log_y_mean,
        coefficients=np.asarray(coefficients, dtype=float),
        iterations=iterations,
    )


def predict_positive(model: RegularizedModel, x_standardized: np.ndarray) -> np.ndarray:
    predicted_log = model.intercept + x_standardized @ model.coefficients
    return np.expm1(np.maximum(predicted_log, 0.0))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    residuals = y_true - y_pred
    denominator = np.maximum(np.abs(y_true), 1e-12)
    ape = np.abs(residuals) / denominator * 100.0
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float("nan") if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
    return {
        "RMSE": float(np.sqrt(np.mean(residuals**2))),
        "RMSLE": float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))),
        "Mean APE (%)": float(np.mean(ape)),
        "Median APE (%)": float(np.median(ape)),
        "R²": r2,
    }


def comparison_table(
    data: SimulationData,
    prepared: PreparedData,
    regularized_model: RegularizedModel,
) -> pd.DataFrame:
    ols_model = fit_regularized(prepared, kind="OLS", alpha=0.0)
    rows: list[dict[str, float | str]] = []
    for model in (ols_model, regularized_model):
        train_metrics = regression_metrics(
            data.y_train,
            predict_positive(model, prepared.x_train_standardized),
        )
        test_metrics = regression_metrics(
            data.y_test,
            predict_positive(model, prepared.x_test_standardized),
        )
        rows.append(
            {
                "Model": model.kind,
                "α": model.alpha,
                "Train RMSLE": train_metrics["RMSLE"],
                "Test RMSLE": test_metrics["RMSLE"],
                "Test mean APE (%)": test_metrics["Mean APE (%)"],
                "Test median APE (%)": test_metrics["Median APE (%)"],
                "Test RMSE": test_metrics["RMSE"],
                "Test R²": test_metrics["R²"],
            }
        )
    return pd.DataFrame(rows)


def coefficient_path(
    prepared: PreparedData,
    *,
    kind: str,
    log10_alphas: np.ndarray,
) -> np.ndarray:
    return np.vstack(
        [
            fit_regularized(prepared, kind=kind, alpha=10.0 ** float(log_alpha)).coefficients
            for log_alpha in log10_alphas
        ]
    )
