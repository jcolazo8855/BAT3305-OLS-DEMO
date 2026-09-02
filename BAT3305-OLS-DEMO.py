from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from polynomial import (
    POPULATION_COEFFICIENTS,
    fit_polynomial,
    population_mean,
    predict_polynomial,
    simulate_polynomial_data,
)
from regularization import (
    coefficient_path,
    comparison_table,
    fit_regularized,
    predict_positive,
    prepare_data,
    regression_metrics,
    simulate_business_data,
)


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="BAT 3305 - Colazo | Regression Learning Lab",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# Styling
# ============================================================
st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at top left, #f8fbff 0%, #f6f8fc 40%, #ffffff 100%);
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.5rem;
            max-width: 1400px;
        }
        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 26px;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 64, 175, 0.92));
            color: white;
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.18);
            margin-bottom: 1rem;
            border: 1px solid rgba(255,255,255,0.10);
        }
        .hero h1 {
            margin: 0;
            font-size: 2.6rem;
            font-weight: 900;
            letter-spacing: -0.03em;
        }
        .hero h3 {
            margin: 0.4rem 0 0 0;
            font-size: 1.18rem;
            font-weight: 600;
            color: rgba(255,255,255,0.95);
        }
        .hero p {
            margin: 0.65rem 0 0 0;
            font-size: 1rem;
            color: rgba(255,255,255,0.84);
            max-width: 1000px;
            line-height: 1.45;
        }
        .metric-card {
            background: white;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 20px;
            padding: 0.95rem 1rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }
        .metric-label {
            font-size: 0.86rem;
            color: #475569;
            margin-bottom: 0.12rem;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: 800;
            color: #0f172a;
        }
        .panel {
            background: white;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 22px;
            padding: 1rem 1rem 0.7rem 1rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }
        .small-note {
            color: #64748b;
            font-size: 0.92rem;
        }
        .callout {
            background: linear-gradient(135deg, rgba(219,234,254,0.8), rgba(224,231,255,0.8));
            border-left: 5px solid #2563eb;
            border-radius: 16px;
            padding: 0.9rem 1rem;
            margin: 0.5rem 0 0.9rem 0;
            color: #1e293b;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 14px 14px 0 0;
            padding: 0.55rem 0.9rem;
            font-weight: 600;
        }
        .teacher-box {
            background: #f8fafc;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 16px;
            padding: 0.85rem 0.95rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Data structures and utilities
# ============================================================
@dataclass
class FitResults:
    slope: float
    intercept: float
    r2: float
    mse: float
    rmse: float
    mae: float
    y_hat: np.ndarray
    residuals: np.ndarray
    sse: float
    leverage: np.ndarray
    cooks_distance: np.ndarray
    ci_mean_low: np.ndarray
    ci_mean_high: np.ndarray
    pi_low: np.ndarray
    pi_high: np.ndarray


@dataclass
class TestResults:
    mse: float
    rmse: float
    mae: float
    r2: float


def safe_float(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"
    return f"{value:.{digits}f}"


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.isclose(np.var(y_true), 0.0):
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if np.isclose(ss_tot, 0.0):
        return float("nan")
    return 1.0 - ss_res / ss_tot


def generate_y(
    x: np.ndarray,
    dgp: str,
    intercept: float,
    slope: float,
    noise_sd: float,
    rng: np.random.Generator,
    quad_strength: float,
    heteroskedastic: bool,
    relationship_scale: float | None = None,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    observed_scale = float(np.std(x))
    scale = float(relationship_scale) if relationship_scale is not None else (observed_scale if observed_scale > 1e-9 else 1.0)

    if dgp == "Linear":
        mean = intercept + slope * x
    elif dgp == "Quadratic (misspecified for OLS)":
        mean = intercept + slope * x + quad_strength * (x / scale) ** 2
    elif dgp == "Exponential (misspecified for OLS)":
        curvature = quad_strength / 8.0
        if curvature <= 1e-9:
            mean = intercept + slope * x
        else:
            mean = intercept + slope * scale * np.expm1(curvature * x / scale) / curvature
    elif dgp == "Piecewise / drift":
        mean = intercept + slope * x
        mean = np.where(x < 0, mean + 0.75 * quad_strength * (x / scale), mean - 0.75 * quad_strength * (x / scale))
    else:
        raise ValueError(f"Unknown data-generating process: {dgp}")

    if heteroskedastic:
        sigma = noise_sd * (0.55 + 0.90 * np.abs(x) / max(np.max(np.abs(x)), 1.0))
    else:
        sigma = np.full_like(x, noise_sd, dtype=float)

    return mean + rng.normal(0.0, sigma, size=len(x))


def fit_ols(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> FitResults:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    n = len(x)

    if n == 0:
        empty = np.array([], dtype=float)
        nan_grid = np.full_like(x_grid, np.nan, dtype=float)
        return FitResults(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, empty, empty, np.nan, empty, empty, nan_grid, nan_grid, nan_grid, nan_grid)

    if n == 1 or np.allclose(x, x[0]):
        intercept = float(np.mean(y))
        slope = 0.0
    else:
        slope, intercept = np.polyfit(x, y, 1)

    y_hat = intercept + slope * x
    residuals = y - y_hat
    sse = float(np.sum(residuals**2))
    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    r2 = compute_r2(y, y_hat)

    x_bar = float(np.mean(x))
    sxx = float(np.sum((x - x_bar) ** 2))

    if n >= 2 and sxx > 1e-12:
        leverage = 1.0 / n + ((x - x_bar) ** 2) / sxx
    else:
        leverage = np.full(n, 1.0 / max(n, 1), dtype=float)

    if n > 2 and sxx > 1e-12:
        mse_resid = sse / (n - 2)
        denom = (1.0 - leverage) ** 2
        denom = np.where(np.isclose(denom, 0.0), np.nan, denom)
        cooks_distance = (residuals**2 / (2.0 * mse_resid)) * (leverage / denom)

        se_mean = np.sqrt(mse_resid * (1.0 / n + ((x_grid - x_bar) ** 2) / sxx))
        se_pred = np.sqrt(mse_resid * (1.0 + 1.0 / n + ((x_grid - x_bar) ** 2) / sxx))
        y_grid = intercept + slope * x_grid
        ci_mean_low = y_grid - 1.96 * se_mean
        ci_mean_high = y_grid + 1.96 * se_mean
        pi_low = y_grid - 1.96 * se_pred
        pi_high = y_grid + 1.96 * se_pred
    else:
        cooks_distance = np.full(n, np.nan, dtype=float)
        y_grid = intercept + slope * x_grid
        ci_mean_low = np.full_like(x_grid, np.nan, dtype=float)
        ci_mean_high = np.full_like(x_grid, np.nan, dtype=float)
        pi_low = np.full_like(x_grid, np.nan, dtype=float)
        pi_high = np.full_like(x_grid, np.nan, dtype=float)

    return FitResults(
        slope=float(slope),
        intercept=float(intercept),
        r2=float(r2),
        mse=mse,
        rmse=rmse,
        mae=mae,
        y_hat=y_hat,
        residuals=residuals,
        sse=sse,
        leverage=leverage,
        cooks_distance=cooks_distance,
        ci_mean_low=ci_mean_low,
        ci_mean_high=ci_mean_high,
        pi_low=pi_low,
        pi_high=pi_high,
    )


def evaluate_on_test(x_test: np.ndarray, y_test: np.ndarray, slope: float, intercept: float) -> TestResults:
    y_pred = intercept + slope * x_test
    residuals = y_test - y_pred
    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    r2 = compute_r2(y_test, y_pred)
    return TestResults(mse=mse, rmse=rmse, mae=mae, r2=r2)


def build_hover_text(index: np.ndarray, leverage: np.ndarray, cooks: np.ndarray) -> List[str]:
    text = []
    for i, lev, cook in zip(index, leverage, cooks):
        cook_str = safe_float(float(cook)) if not np.isnan(cook) else "—"
        text.append(f"Obs {int(i)}<br>Leverage: {safe_float(float(lev))}<br>Cook's D: {cook_str}")
    return text


# ============================================================
# Session state
# ============================================================
def ensure_state() -> None:
    defaults: Dict[str, object] = {
        "x": [],
        "y": [],
        "train_pool_x": [],
        "train_pool_y": [],
        "test_x": [],
        "test_y": [],
        "relationship_scale": 1.0,
        "dataset_config": None,
        "seed": 3305,
        "history_n": [],
        "history_slope": [],
        "history_intercept": [],
        "history_r2": [],
        "history_train_rmse": [],
        "history_test_rmse": [],
        "history_test_r2": [],
        "history_mae": [],
        "history_note": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, list) else value


ensure_state()


def render_regularization_tab(kind: str, key_prefix: str, default_log_alpha: float) -> None:
    """Render a self-contained Ridge or LASSO learning simulation."""
    model_name = "Ridge" if kind == "Ridge" else "LASSO"
    penalty_name = "squared-coefficient" if kind == "Ridge" else "absolute-coefficient"
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader(f"{model_name} regression tuning lab")
    if kind == "Ridge":
        st.markdown(
            "Ridge shrinks correlated predictors together. Tune **α** and watch coefficient magnitudes, "
            "test error, and proportional-error metrics change relative to unpenalized OLS."
        )
    else:
        st.markdown(
            "LASSO can shrink weak predictors exactly to zero. Tune **α** and study the tradeoff between "
            "feature selection, prediction error, and model simplicity."
        )
    st.markdown("</div>", unsafe_allow_html=True)

    control_columns = st.columns(3)
    with control_columns[0]:
        log_alpha = st.slider(
            "log₁₀ penalty (α)",
            -4.0,
            1.5,
            default_log_alpha,
            0.1,
            key=f"{key_prefix}_log_alpha",
            help="The fitted penalty is 10 raised to this value. Move right for stronger shrinkage.",
        )
        alpha = 10.0**log_alpha
        total_n = st.slider(
            "Simulated sample size",
            80,
            600,
            240,
            20,
            key=f"{key_prefix}_sample_size",
        )
    with control_columns[1]:
        train_share = st.slider(
            "Training share (%)",
            60,
            90,
            75,
            5,
            key=f"{key_prefix}_train_share",
        )
        relevant_predictors = st.slider(
            "Predictors with a true signal",
            1,
            6,
            3,
            1,
            key=f"{key_prefix}_relevant",
            help="The remaining simulated predictors are noise variables with a true coefficient of zero.",
        )
    with control_columns[2]:
        correlation = st.slider(
            "Predictor correlation",
            0.0,
            0.9,
            0.65,
            0.05,
            key=f"{key_prefix}_correlation",
        )
        outcome_noise = st.slider(
            "Outcome noise",
            0.05,
            0.80,
            0.25,
            0.05,
            key=f"{key_prefix}_noise",
        )
    simulation_seed = int(
        st.number_input(
            "Simulation seed",
            min_value=1,
            max_value=999999,
            value=441 if kind == "Ridge" else 442,
            step=1,
            key=f"{key_prefix}_seed",
        )
    )

    data = simulate_business_data(
        seed=simulation_seed,
        total_sample_size=total_n,
        train_percent=train_share,
        relevant_predictors=relevant_predictors,
        correlation=correlation,
        noise_sd=outcome_noise,
    )
    prepared = prepare_data(data)
    selected_model = fit_regularized(prepared, kind=kind, alpha=alpha)
    ols_model = fit_regularized(prepared, kind="OLS", alpha=0.0)
    selected_test_predictions = predict_positive(selected_model, prepared.x_test_standardized)
    selected_test_metrics = regression_metrics(data.y_test, selected_test_predictions)
    comparison = comparison_table(data, prepared, selected_model).round(4)

    coefficient_norm = float(np.linalg.norm(selected_model.coefficients))
    ols_norm = float(np.linalg.norm(ols_model.coefficients))
    zero_count = int(np.sum(np.abs(selected_model.coefficients) < 1e-6))
    summary_columns = st.columns(5)
    summary_items = [
        ("Selected α", f"{alpha:.4g}"),
        ("Test RMSLE", f"{selected_test_metrics['RMSLE']:.4f}"),
        ("Test mean APE", f"{selected_test_metrics['Mean APE (%)']:.2f}%"),
        ("Test R²", f"{selected_test_metrics['R²']:.3f}"),
        (
            "Exact zero coefficients" if kind == "LASSO" else "Coefficient norm",
            str(zero_count) if kind == "LASSO" else f"{coefficient_norm:.3f}",
        ),
    ]
    for column, (label, value) in zip(summary_columns, summary_items):
        column.metric(label, value)

    st.caption(
        f"Positive simulated outcome · {len(data.y_train)} training and {len(data.y_test)} test observations · "
        f"six standardized predictors · {penalty_name} penalty."
    )

    top_left, top_right = st.columns([1.05, 0.95])
    with top_left:
        st.markdown("#### Model comparison")
        st.dataframe(comparison, hide_index=True, width="stretch")
        st.caption(
            "RMSLE compares log(1 + actual) with log(1 + predicted), so proportional misses matter more "
            "than the largest raw-value misses. APE is the absolute percentage error for each observation; "
            "its mean can be sensitive when actual values are small."
        )

    with top_right:
        st.markdown("#### What the current penalty did")
        if kind == "Ridge":
            shrinkage = 0.0 if ols_norm <= 1e-12 else 100.0 * (1.0 - coefficient_norm / ols_norm)
            st.info(
                f"The combined coefficient magnitude is **{shrinkage:.1f}% smaller** than OLS. "
                "Ridge usually retains every predictor, even when its true signal is zero."
            )
        else:
            st.info(
                f"LASSO set **{zero_count} of 6 coefficients** to zero at this α and converged in "
                f"{selected_model.iterations} coordinate-descent iterations."
            )
        st.latex(
            r"\operatorname{RMSLE}=\sqrt{\frac{1}{n}\sum_i[\log(1+\hat y_i)-\log(1+y_i)]^2}"
        )
        st.latex(r"\operatorname{APE}_i=\left|\frac{y_i-\hat y_i}{y_i}\right|\times100\%")

    standardized_truth = data.true_coefficients * prepared.x_scale
    coefficient_frame = pd.DataFrame(
        {
            "Feature": data.feature_names,
            "True signal": standardized_truth,
            "OLS": ols_model.coefficients,
            model_name: selected_model.coefficients,
        }
    )
    coefficient_figure = go.Figure()
    for column_name, color in (("True signal", "#0f172a"), ("OLS", "#f59e0b"), (model_name, "#2563eb")):
        coefficient_figure.add_trace(
            go.Bar(
                x=coefficient_frame["Feature"],
                y=coefficient_frame[column_name],
                name=column_name,
                marker_color=color,
            )
        )
    coefficient_figure.update_layout(
        title="Standardized coefficients: truth, OLS, and the tuned model",
        barmode="group",
        template="plotly_white",
        height=430,
        margin=dict(l=10, r=10, t=55, b=80),
        yaxis_title="Coefficient on log(1 + outcome)",
    )

    log_alpha_grid = np.linspace(-4.0, 1.5, 46)
    path = coefficient_path(prepared, kind=kind, log10_alphas=log_alpha_grid)
    path_figure = go.Figure()
    for feature_index, feature_name in enumerate(data.feature_names):
        path_figure.add_trace(
            go.Scatter(
                x=log_alpha_grid,
                y=path[:, feature_index],
                mode="lines",
                name=feature_name,
                hovertemplate=f"{feature_name}<br>log₁₀ α=%{{x:.1f}}<br>coefficient=%{{y:.3f}}<extra></extra>",
            )
        )
    path_figure.add_vline(x=log_alpha, line_dash="dash", line_color="#0f172a", annotation_text="Selected α")
    path_figure.update_layout(
        title=f"{model_name} coefficient path",
        template="plotly_white",
        height=430,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis_title="log₁₀ penalty (α)",
        yaxis_title="Standardized coefficient",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(coefficient_figure, width="stretch")
    with chart_right:
        st.plotly_chart(path_figure, width="stretch")

    test_ape = np.abs(data.y_test - selected_test_predictions) / np.maximum(data.y_test, 1e-12) * 100.0
    prediction_figure = go.Figure()
    prediction_figure.add_trace(
        go.Scatter(
            x=data.y_test,
            y=selected_test_predictions,
            mode="markers",
            name=f"{model_name} predictions",
            marker=dict(
                size=9,
                color=test_ape,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="APE %"),
                line=dict(width=0.5, color="rgba(15,23,42,0.35)"),
            ),
            customdata=test_ape,
            hovertemplate="Actual=%{x:.2f}<br>Predicted=%{y:.2f}<br>APE=%{customdata:.2f}%<extra></extra>",
        )
    )
    lower = float(min(np.min(data.y_test), np.min(selected_test_predictions)))
    upper = float(max(np.max(data.y_test), np.max(selected_test_predictions)))
    prediction_figure.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=[lower, upper],
            mode="lines",
            name="Perfect prediction",
            line=dict(dash="dash", color="#0f172a"),
        )
    )
    prediction_figure.update_layout(
        title=f"{model_name} test predictions colored by APE",
        template="plotly_white",
        height=430,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis_title="Actual positive outcome",
        yaxis_title="Predicted positive outcome",
    )

    ape_figure = go.Figure(
        go.Histogram(
            x=test_ape,
            nbinsx=24,
            marker_color="#2563eb",
            hovertemplate="APE bin=%{x:.1f}%<br>Observations=%{y}<extra></extra>",
        )
    )
    ape_figure.add_vline(
        x=float(np.mean(test_ape)),
        line_dash="dash",
        line_color="#dc2626",
        annotation_text="Mean APE",
    )
    ape_figure.update_layout(
        title="Distribution of observation-level APE",
        template="plotly_white",
        height=430,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis_title="Absolute percentage error (%)",
        yaxis_title="Test observations",
    )
    result_left, result_right = st.columns(2)
    with result_left:
        st.plotly_chart(prediction_figure, width="stretch")
    with result_right:
        st.plotly_chart(ape_figure, width="stretch")

    with st.expander("Student exploration prompts"):
        if kind == "Ridge":
            st.markdown(
                """
                1. Raise predictor correlation. Does Ridge become more competitive with OLS on the test sample?
                2. Increase α until test RMSLE begins to worsen. What happened to the coefficient path?
                3. Reduce the sample size and change the seed. Which method is more stable across simulated samples?
                4. Explain why Ridge shrinks noise predictors but rarely removes them exactly.
                """
            )
        else:
            st.markdown(
                """
                1. Increase α until LASSO finds the correct number of relevant predictors. Did test RMSLE improve?
                2. Raise predictor correlation. Does LASSO consistently choose the same member of a correlated group?
                3. Increase outcome noise. How does the selected set of nonzero coefficients change?
                4. Find an α that makes the model simpler without materially worsening test RMSLE or mean APE.
                """
            )


def render_polynomial_tab() -> None:
    """Render a one-predictor/one-outcome polynomial specification lab."""
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Bivariate polynomial regression lab")
    st.markdown(
        "Choose the degree of the **population relationship** and the degree of the **fitted model** "
        "independently. Degree 1 is linear; degrees 2–5 add progressively higher powers of the same predictor."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    controls = st.columns(3)
    with controls[0]:
        population_degree = st.slider(
            "Population polynomial degree (n)",
            1,
            5,
            3,
            1,
            key="poly_population_degree",
            help="The highest power of x used to generate the population mean. n=1 is linear.",
        )
        model_degree = st.slider(
            "Fitted model polynomial degree (n)",
            1,
            5,
            2,
            1,
            key="poly_model_degree",
            help="The highest power of x included in the estimated regression model.",
        )
    with controls[1]:
        total_n = st.slider(
            "Polynomial sample size",
            40,
            500,
            180,
            10,
            key="poly_sample_size",
        )
        train_share = st.slider(
            "Polynomial training share (%)",
            60,
            90,
            75,
            5,
            key="poly_train_share",
        )
    with controls[2]:
        noise_sd = st.slider(
            "Polynomial outcome noise",
            0.0,
            20.0,
            8.0,
            0.5,
            key="poly_noise",
        )
        seed = int(
            st.number_input(
                "Polynomial simulation seed",
                min_value=1,
                max_value=999999,
                value=3305,
                step=1,
                key="poly_seed",
            )
        )

    data = simulate_polynomial_data(
        seed=seed,
        total_sample_size=total_n,
        train_percent=train_share,
        population_degree=population_degree,
        noise_sd=noise_sd,
    )
    model = fit_polynomial(data.x_train, data.y_train, model_degree, data.x_scale)
    train_predictions = np.maximum(predict_polynomial(model, data.x_train), 0.0)
    test_predictions = np.maximum(predict_polynomial(model, data.x_test), 0.0)
    train_metrics = regression_metrics(data.y_train, train_predictions)
    test_metrics = regression_metrics(data.y_test, test_predictions)

    if model_degree < population_degree:
        specification_label = "Under-specified"
        specification_message = (
            "The fitted model cannot represent every term in the population relationship. More data can reduce "
            "sampling noise, but it cannot supply the omitted powers."
        )
    elif model_degree == population_degree:
        specification_label = "Correct degree"
        specification_message = (
            "The fitted model contains the population's highest power. Sampling noise can still move the estimated "
            "curve away from the population curve."
        )
    else:
        specification_label = "Over-specified"
        specification_message = (
            "The fitted model includes powers whose population coefficients are zero. This adds flexibility and can "
            "increase variance, especially in small samples."
        )

    metric_columns = st.columns(6)
    metric_items = [
        ("Population degree", str(population_degree)),
        ("Model degree", str(model_degree)),
        ("Test RMSLE", f"{test_metrics['RMSLE']:.4f}"),
        ("Test mean APE", f"{test_metrics['Mean APE (%)']:.2f}%"),
        ("Test RMSE", f"{test_metrics['RMSE']:.2f}"),
        ("Test R²", f"{test_metrics['R²']:.3f}"),
    ]
    for column, (label, value) in zip(metric_columns, metric_items):
        column.metric(label, value)

    st.info(f"**{specification_label}:** {specification_message}")

    formula_terms = [f"{POPULATION_COEFFICIENTS[0]:.0f}"]
    for power in range(1, population_degree + 1):
        coefficient = POPULATION_COEFFICIENTS[power]
        power_text = "z" if power == 1 else rf"z^{{{power}}}"
        formula_terms.append(f"{coefficient:+.0f}{power_text}")
    st.markdown("**Selected population**")
    st.latex(r"E[Y\mid x]=" + "".join(formula_terms) + r",\qquad z=x/3")
    st.caption(
        f"The observed outcome adds normally distributed noise with SD {noise_sd:.1f}. "
        "The simulated population is kept positive so RMSLE and APE remain defined."
    )

    x_grid = np.linspace(-data.x_scale, data.x_scale, 320)
    true_grid = population_mean(x_grid, population_degree, data.x_scale)
    fitted_grid = np.maximum(predict_polynomial(model, x_grid), 0.0)
    curve_figure = go.Figure()
    curve_figure.add_trace(
        go.Scatter(
            x=data.x_train,
            y=data.y_train,
            mode="markers",
            name="Training observations",
            marker=dict(size=8, opacity=0.70, color="#2563eb"),
        )
    )
    curve_figure.add_trace(
        go.Scatter(
            x=data.x_test,
            y=data.y_test,
            mode="markers",
            name="Test observations",
            marker=dict(size=8, opacity=0.70, color="#f59e0b", symbol="diamond"),
        )
    )
    curve_figure.add_trace(
        go.Scatter(
            x=x_grid,
            y=true_grid,
            mode="lines",
            name=f"Population degree {population_degree}",
            line=dict(width=4, dash="dash", color="#0f172a"),
        )
    )
    curve_figure.add_trace(
        go.Scatter(
            x=x_grid,
            y=fitted_grid,
            mode="lines",
            name=f"Fitted degree {model_degree}",
            line=dict(width=4, color="#dc2626"),
        )
    )
    curve_figure.update_layout(
        title="Population curve and fitted polynomial",
        template="plotly_white",
        height=500,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis_title="Predictor x",
        yaxis_title="Positive outcome y",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    coefficient_rows = []
    for power in range(0, 6):
        coefficient_rows.append(
            {
                "Term": "Intercept" if power == 0 else ("z" if power == 1 else f"z^{power}"),
                "Population coefficient": (
                    POPULATION_COEFFICIENTS[power] if power <= population_degree else 0.0
                ),
                "Estimated coefficient": model.coefficients[power] if power <= model_degree else np.nan,
                "Included in fitted model": power <= model_degree,
            }
        )
    coefficient_table = pd.DataFrame(coefficient_rows).round(3)

    curve_left, curve_right = st.columns([1.25, 0.75])
    with curve_left:
        st.plotly_chart(curve_figure, width="stretch")
    with curve_right:
        st.markdown("#### Population and estimated terms")
        st.dataframe(coefficient_table, hide_index=True, width="stretch")
        st.caption(
            "Coefficients use z=x/3. A zero population coefficient means that power is absent from the true relationship; "
            "a blank estimate means the fitted model did not include that power."
        )

    degree_rows = []
    for candidate_degree in range(1, 6):
        candidate_model = fit_polynomial(
            data.x_train,
            data.y_train,
            candidate_degree,
            data.x_scale,
        )
        candidate_train = np.maximum(predict_polynomial(candidate_model, data.x_train), 0.0)
        candidate_test = np.maximum(predict_polynomial(candidate_model, data.x_test), 0.0)
        candidate_train_metrics = regression_metrics(data.y_train, candidate_train)
        candidate_test_metrics = regression_metrics(data.y_test, candidate_test)
        degree_rows.append(
            {
                "Model degree": candidate_degree,
                "Selected": "✓" if candidate_degree == model_degree else "",
                "Train RMSLE": candidate_train_metrics["RMSLE"],
                "Test RMSLE": candidate_test_metrics["RMSLE"],
                "Test mean APE (%)": candidate_test_metrics["Mean APE (%)"],
                "Test RMSE": candidate_test_metrics["RMSE"],
                "Test R²": candidate_test_metrics["R²"],
            }
        )
    degree_table = pd.DataFrame(degree_rows).round(4)

    rmsle_matrix = np.empty((5, 5), dtype=float)
    for population_index, candidate_population_degree in enumerate(range(1, 6)):
        candidate_data = simulate_polynomial_data(
            seed=seed,
            total_sample_size=total_n,
            train_percent=train_share,
            population_degree=candidate_population_degree,
            noise_sd=noise_sd,
        )
        for model_index, candidate_model_degree in enumerate(range(1, 6)):
            candidate_model = fit_polynomial(
                candidate_data.x_train,
                candidate_data.y_train,
                candidate_model_degree,
                candidate_data.x_scale,
            )
            candidate_predictions = np.maximum(
                predict_polynomial(candidate_model, candidate_data.x_test),
                0.0,
            )
            rmsle_matrix[population_index, model_index] = regression_metrics(
                candidate_data.y_test,
                candidate_predictions,
            )["RMSLE"]

    heatmap = go.Figure(
        go.Heatmap(
            z=rmsle_matrix,
            x=[1, 2, 3, 4, 5],
            y=[1, 2, 3, 4, 5],
            colorscale="Blues",
            reversescale=True,
            text=np.round(rmsle_matrix, 3),
            texttemplate="%{text:.3f}",
            colorbar=dict(title="Test RMSLE"),
            hovertemplate="Population degree=%{y}<br>Model degree=%{x}<br>Test RMSLE=%{z:.4f}<extra></extra>",
        )
    )
    heatmap.add_shape(
        type="rect",
        x0=model_degree - 0.48,
        x1=model_degree + 0.48,
        y0=population_degree - 0.48,
        y1=population_degree + 0.48,
        line=dict(color="#dc2626", width=4),
    )
    heatmap.update_layout(
        title="Population degree × fitted-model degree",
        template="plotly_white",
        height=430,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis=dict(title="Fitted model degree", tickmode="array", tickvals=[1, 2, 3, 4, 5]),
        yaxis=dict(title="Population degree", tickmode="array", tickvals=[1, 2, 3, 4, 5]),
    )

    comparison_left, comparison_right = st.columns([1.0, 1.0])
    with comparison_left:
        st.markdown("#### Fit every model degree to the selected population")
        st.dataframe(degree_table, hide_index=True, width="stretch")
        st.caption(
            "Training error generally falls as degree increases. Use the held-out test metrics to judge whether the extra "
            "powers generalize."
        )
    with comparison_right:
        st.plotly_chart(heatmap, width="stretch")
        st.caption("The red outline marks the currently selected population/model combination.")

    with st.expander("Student exploration prompts"):
        st.markdown(
            """
            1. Hold the population at degree 1 and raise the fitted-model degree. What happens to train versus test RMSLE?
            2. Set the population to degree 5 and the model to degree 1. Can a larger sample repair the missing curvature?
            3. Match both degrees, then increase noise. Does the correct model always have the lowest test error in one sample?
            4. Change only the seed. Which conclusions are stable, and which reflect sampling variation?
            5. Find a case where a simpler model has nearly the same test RMSLE and APE as the true-degree model.
            """
        )


# ============================================================
# Simulation functions
# ============================================================
def refresh_history(
    x_grid: np.ndarray,
    note: str,
) -> FitResults:
    x = np.array(st.session_state.x, dtype=float)
    y = np.array(st.session_state.y, dtype=float)
    fit = fit_ols(x, y, x_grid)

    x_test = np.array(st.session_state.test_x, dtype=float)
    y_test = np.array(st.session_state.test_y, dtype=float)
    test = evaluate_on_test(x_test, y_test, fit.slope, fit.intercept)

    current_n = len(x)
    if not st.session_state.history_n or st.session_state.history_n[-1] != current_n or note == "outlier":
        st.session_state.history_n.append(current_n)
        st.session_state.history_slope.append(fit.slope)
        st.session_state.history_intercept.append(fit.intercept)
        st.session_state.history_r2.append(fit.r2)
        st.session_state.history_train_rmse.append(fit.rmse)
        st.session_state.history_test_rmse.append(test.rmse)
        st.session_state.history_test_r2.append(test.r2)
        st.session_state.history_mae.append(fit.mae)
        st.session_state.history_note.append(note)
    return fit


def reset_simulation(
    total_sample_size: int,
    train_percent: int,
    n_initial: int,
    x_min: float,
    x_max: float,
    dgp: str,
    intercept: float,
    slope: float,
    noise_sd: float,
    quad_strength: float,
    heteroskedastic: bool,
    x_grid: np.ndarray,
    dataset_config: Tuple[object, ...],
) -> FitResults:
    rng = np.random.default_rng(int(st.session_state.seed))
    all_x = rng.uniform(x_min, x_max, size=total_sample_size)
    relationship_scale = float(np.std(all_x)) if np.std(all_x) > 1e-9 else 1.0
    all_y = generate_y(
        x=all_x,
        dgp=dgp,
        intercept=intercept,
        slope=slope,
        noise_sd=noise_sd,
        rng=rng,
        quad_strength=quad_strength,
        heteroskedastic=heteroskedastic,
        relationship_scale=relationship_scale,
    )

    n_train = int(round(total_sample_size * train_percent / 100.0))
    n_train = min(max(n_train, 3), total_sample_size - 2)
    shuffled_indices = rng.permutation(total_sample_size)
    train_indices = shuffled_indices[:n_train]
    test_indices = shuffled_indices[n_train:]

    train_x = all_x[train_indices]
    train_y = all_y[train_indices]
    test_x = all_x[test_indices]
    test_y = all_y[test_indices]
    active_n = min(max(n_initial, 3), n_train)

    st.session_state.train_pool_x = list(map(float, train_x))
    st.session_state.train_pool_y = list(map(float, train_y))
    st.session_state.test_x = list(map(float, test_x))
    st.session_state.test_y = list(map(float, test_y))
    st.session_state.x = list(map(float, train_x[:active_n]))
    st.session_state.y = list(map(float, train_y[:active_n]))
    st.session_state.relationship_scale = relationship_scale
    st.session_state.dataset_config = dataset_config
    st.session_state.history_n = []
    st.session_state.history_slope = []
    st.session_state.history_intercept = []
    st.session_state.history_r2 = []
    st.session_state.history_train_rmse = []
    st.session_state.history_test_rmse = []
    st.session_state.history_test_r2 = []
    st.session_state.history_mae = []
    st.session_state.history_note = []
    return refresh_history(x_grid, note="reset")


def add_points(
    k: int,
    x_grid: np.ndarray,
    note: str,
) -> FitResults:
    start = len(st.session_state.x)
    stop = min(start + k, len(st.session_state.train_pool_x))
    if stop > start:
        st.session_state.x.extend(st.session_state.train_pool_x[start:stop])
        st.session_state.y.extend(st.session_state.train_pool_y[start:stop])
    return refresh_history(x_grid, note=note)


def inject_outlier(
    noise_sd: float,
    x_grid: np.ndarray,
    strength: float,
) -> FitResults:
    rng = np.random.default_rng(int(st.session_state.seed) + 777_777 + len(st.session_state.x))
    direction = float(rng.choice([-1.0, 1.0]))
    next_index = len(st.session_state.x)
    if next_index < len(st.session_state.train_pool_x):
        ox = float(st.session_state.train_pool_x[next_index])
        baseline = float(st.session_state.train_pool_y[next_index])
        oy = float(baseline + direction * strength * noise_sd * (1.0 + rng.random()))
        st.session_state.x.append(ox)
        st.session_state.y.append(oy)
    else:
        target_index = int(rng.integers(0, len(st.session_state.y)))
        st.session_state.y[target_index] = float(
            st.session_state.y[target_index] + direction * strength * noise_sd * (1.0 + rng.random())
        )
    return refresh_history(x_grid, note="outlier")


# ============================================================
# Header
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>BAT 3305 - Colazo</h1>
        <h3>Interactive OLS, Ridge, LASSO, and Polynomial Regression Studio</h3>
        <p>
            Explore how ordinary least squares becomes more stable as it ingests more data.
            This lab lets students observe coefficient convergence, uncertainty reduction,
            outlier sensitivity, train-versus-test behavior, and what happens when the
            underlying data-generating process is not truly linear. Dedicated Ridge and LASSO
            labs add correlated predictors, penalty tuning, feature selection, RMSLE, and APE.
            The polynomial lab separates the population degree from the fitted-model degree.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar controls
# ============================================================
with st.sidebar:
    st.markdown("## Controls")
    st.markdown("Use the controls below to generate a new world, then add data and watch OLS react.")

    dgp = st.selectbox(
        "Underlying data-generating process",
        ["Linear", "Quadratic (misspecified for OLS)", "Exponential (misspecified for OLS)", "Piecewise / drift"],
        index=0,
        help="OLS still fits a straight line, even when the true relationship is not linear.",
    )
    true_slope = st.slider("True slope", -4.0, 4.0, 1.5, 0.1)
    true_intercept = st.slider("True intercept", -12.0, 12.0, 2.0, 0.1)
    noise_sd = st.slider("Noise standard deviation", 0.1, 10.0, 2.5, 0.1)
    quad_strength = st.slider(
        "Nonlinearity / drift strength",
        0.0,
        8.0,
        2.5,
        0.1,
        help="Controls curvature in the quadratic and exponential worlds and the slope change in the piecewise world.",
    )
    heteroskedastic = st.toggle(
        "Heteroskedastic errors",
        value=False,
        help="When on, noise increases as x moves away from zero.",
    )

    st.markdown("---")
    x_min, x_max = st.slider("x-range", -25.0, 25.0, (-10.0, 10.0), 0.5)
    total_sample_size = st.slider(
        "Total sample size",
        20,
        1000,
        120,
        10,
        help="The complete dataset generated before it is randomly divided into training and test samples.",
    )
    train_percent = st.slider(
        "Train/test split (% training)",
        50,
        90,
        80,
        5,
        help="The percentage of the total sample assigned to training. The remaining observations form the test holdout.",
    )
    n_initial = st.slider("Initial sample size", 3, 80, 6, 1)
    add_k = st.slider("Points added per click", 1, 200, 10, 1)
    outlier_strength = st.slider("Outlier strength", 2.0, 12.0, 5.5, 0.5)
    st.session_state.seed = int(st.number_input("Random seed", min_value=1, max_value=999999, value=int(st.session_state.seed), step=1))

    x_grid = np.linspace(x_min, x_max, 260)
    planned_train_size = int(round(total_sample_size * train_percent / 100.0))
    planned_train_size = min(max(planned_train_size, 3), total_sample_size - 2)
    planned_test_size = total_sample_size - planned_train_size
    st.caption(
        f"Split: {planned_train_size} training observations and {planned_test_size} test observations. "
        "Add controls reveal more of the fixed training sample."
    )

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        reset_clicked = st.button("Reset", key="reset_simulation", width="stretch")
    with col_b:
        add_clicked = st.button(f"Add {add_k}", key="add_batch", width="stretch")

    col_c, col_d = st.columns(2)
    with col_c:
        add_one_clicked = st.button("Add 1", key="add_single", width="stretch")
    with col_d:
        outlier_clicked = st.button("Outlier", key="inject_outlier", width="stretch")

    auto_steps = st.slider("Auto-grow steps", 2, 40, 8, 1)
    auto_clicked = st.button("Run auto-growth", key="run_auto_growth", width="stretch")

    st.caption("Changing a data or split control automatically generates the corresponding fixed dataset.")


# ============================================================
# Apply controls
# ============================================================
dataset_config = (
    int(total_sample_size),
    int(train_percent),
    int(n_initial),
    float(x_min),
    float(x_max),
    dgp,
    float(true_intercept),
    float(true_slope),
    float(noise_sd),
    float(quad_strength),
    bool(heteroskedastic),
    int(st.session_state.seed),
)
needs_reset = not st.session_state.x or st.session_state.dataset_config != dataset_config

if needs_reset or reset_clicked:
    fit = reset_simulation(
        total_sample_size=total_sample_size,
        train_percent=train_percent,
        n_initial=n_initial,
        x_min=x_min,
        x_max=x_max,
        dgp=dgp,
        intercept=true_intercept,
        slope=true_slope,
        noise_sd=noise_sd,
        quad_strength=quad_strength,
        heteroskedastic=heteroskedastic,
        x_grid=x_grid,
        dataset_config=dataset_config,
    )
elif add_clicked:
    fit = add_points(
        k=add_k,
        x_grid=x_grid,
        note=f"add_{add_k}",
    )
elif add_one_clicked:
    fit = add_points(
        k=1,
        x_grid=x_grid,
        note="add_1",
    )
elif outlier_clicked:
    fit = inject_outlier(
        noise_sd=noise_sd,
        x_grid=x_grid,
        strength=outlier_strength,
    )
elif auto_clicked:
    for _ in range(auto_steps):
        prior_n = len(st.session_state.x)
        fit = add_points(
            k=add_k,
            x_grid=x_grid,
            note=f"auto_{add_k}",
        )
        if len(st.session_state.x) == prior_n:
            break
else:
    fit = refresh_history(x_grid=x_grid, note="view")

x = np.array(st.session_state.x, dtype=float)
y = np.array(st.session_state.y, dtype=float)
fit = fit_ols(x, y, x_grid)

true_line = generate_y(
    x=x_grid,
    dgp=dgp,
    intercept=true_intercept,
    slope=true_slope,
    noise_sd=0.0,
    rng=np.random.default_rng(1234),
    quad_strength=quad_strength,
    heteroskedastic=False,
    relationship_scale=float(st.session_state.relationship_scale),
)
est_line = fit.intercept + fit.slope * x_grid

# Fixed held-out test set for the current dataset
x_test = np.array(st.session_state.test_x, dtype=float)
y_test = np.array(st.session_state.test_y, dtype=float)
test_results = evaluate_on_test(x_test, y_test, fit.slope, fit.intercept)

history_df = pd.DataFrame(
    {
        "n": st.session_state.history_n,
        "slope_hat": st.session_state.history_slope,
        "intercept_hat": st.session_state.history_intercept,
        "r2": st.session_state.history_r2,
        "train_rmse": st.session_state.history_train_rmse,
        "test_rmse": st.session_state.history_test_rmse,
        "test_r2": st.session_state.history_test_r2,
        "mae": st.session_state.history_mae,
        "note": st.session_state.history_note,
    }
)

# ============================================================
# Metrics row
# ============================================================
metric_cols = st.columns(6)
metric_items = [
    ("Training used", f"{len(x)} / {len(st.session_state.train_pool_x)}"),
    ("Slope estimate", safe_float(fit.slope)),
    ("Intercept estimate", safe_float(fit.intercept)),
    ("Train RMSE", safe_float(fit.rmse)),
    ("Test RMSE", safe_float(test_results.rmse)),
    ("R²", safe_float(fit.r2)),
]
for col, (label, value) in zip(metric_cols, metric_items):
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    f"Total sample: {total_sample_size} · Training allocation: {len(st.session_state.train_pool_x)} "
    f"· Test holdout: {len(x_test)} · OLS currently uses {len(x)} revealed training observations."
)

st.markdown(
    f"""
    <div class="callout">
        <strong>Current world:</strong> {dgp} | y is generated around a baseline with intercept {true_intercept:.1f}, slope {true_slope:.1f}, noise SD {noise_sd:.1f}
        {'with heteroskedasticity' if heteroskedastic else 'with constant error variance'}.
        The app always fits a <strong>linear OLS model</strong>, so students can see when more data helps and when misspecification still matters.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "Playground",
        "Convergence & Generalization",
        "Diagnostics",
        "Ridge Regression",
        "LASSO Regression",
        "Polynomial Regression",
        "Student Challenge",
    ]
)

with tab1:
    plot_area = st.container()

    with plot_area:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Observed data",
                marker=dict(size=9, opacity=0.88, line=dict(width=1, color="rgba(15,23,42,0.35)")),
                text=build_hover_text(np.arange(1, len(x) + 1), fit.leverage, fit.cooks_distance),
                hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>%{text}<extra></extra>",
            )
        )

        if len(x) > 2 and not np.all(np.isnan(fit.pi_low)):
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x_grid, x_grid[::-1]]),
                    y=np.concatenate([fit.pi_high, fit.pi_low[::-1]]),
                    fill="toself",
                    fillcolor="rgba(99, 102, 241, 0.10)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    name="95% prediction band",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x_grid, x_grid[::-1]]),
                    y=np.concatenate([fit.ci_mean_high, fit.ci_mean_low[::-1]]),
                    fill="toself",
                    fillcolor="rgba(37, 99, 235, 0.16)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    name="95% mean CI",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=true_line,
                mode="lines",
                name="True relationship",
                line=dict(width=3, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=est_line,
                mode="lines",
                name="OLS fit",
                line=dict(width=4),
            )
        )
        fig.update_layout(
            title="Observed data, true relationship, and estimated OLS line",
            template="plotly_white",
            height=560,
            margin=dict(l=10, r=10, t=55, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="x",
            yaxis_title="y",
        )
        st.plotly_chart(fig, width="stretch")

with tab2:
    c1, c2 = st.columns(2)

    with c1:
        fig_coef = go.Figure()
        fig_coef.add_trace(
            go.Scatter(
                x=history_df["n"],
                y=history_df["slope_hat"],
                mode="lines+markers",
                name="Estimated slope",
                hovertemplate="n=%{x}<br>slopê=%{y:.3f}<extra></extra>",
            )
        )
        fig_coef.add_hline(y=true_slope, line_dash="dash", annotation_text="True slope", annotation_position="top left")
        fig_coef.update_layout(
            title="Coefficient convergence as the sample grows",
            template="plotly_white",
            height=360,
            margin=dict(l=10, r=10, t=55, b=10),
            xaxis_title="Sample size (n)",
            yaxis_title="Slope estimate",
        )
        st.plotly_chart(fig_coef, width="stretch")

    with c2:
        fig_fit = go.Figure()
        fig_fit.add_trace(
            go.Scatter(
                x=history_df["n"],
                y=history_df["train_rmse"],
                mode="lines+markers",
                name="Train RMSE",
                hovertemplate="n=%{x}<br>Train RMSE=%{y:.3f}<extra></extra>",
            )
        )
        fig_fit.add_trace(
            go.Scatter(
                x=history_df["n"],
                y=history_df["test_rmse"],
                mode="lines+markers",
                name="Test RMSE",
                hovertemplate="n=%{x}<br>Test RMSE=%{y:.3f}<extra></extra>",
            )
        )
        fig_fit.update_layout(
            title="Training versus test error over time",
            template="plotly_white",
            height=360,
            margin=dict(l=10, r=10, t=55, b=10),
            xaxis_title="Sample size (n)",
            yaxis_title="RMSE",
        )
        st.plotly_chart(fig_fit, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        fig_r2 = go.Figure()
        fig_r2.add_trace(
            go.Scatter(
                x=history_df["n"],
                y=history_df["r2"],
                mode="lines+markers",
                name="Train R²",
            )
        )
        fig_r2.add_trace(
            go.Scatter(
                x=history_df["n"],
                y=history_df["test_r2"],
                mode="lines+markers",
                name="Test R²",
            )
        )
        fig_r2.update_layout(
            title="Explained variation over time",
            template="plotly_white",
            height=330,
            margin=dict(l=10, r=10, t=55, b=10),
            xaxis_title="Sample size (n)",
            yaxis_title="R²",
        )
        st.plotly_chart(fig_r2, width="stretch")

    with c4:
        st.markdown('<div class="teacher-box">', unsafe_allow_html=True)
        st.subheader("Interpretation guide")
        st.markdown(
            """
            - In a **linear world**, the slope estimate should wander early and then settle near the true slope.  
            - **Train and test RMSE often converge** as more data reduces estimator variance.  
            - In a **misspecified world**, the line can stabilize even though it is still wrong.  
            - If error variance is not constant, the fit may look reasonable while residual patterns still reveal trouble.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    d1, d2 = st.columns(2)

    with d1:
        residual_fig = go.Figure()
        residual_fig.add_trace(
            go.Scatter(
                x=fit.y_hat,
                y=fit.residuals,
                mode="markers",
                name="Residuals",
                marker=dict(size=9, line=dict(width=1, color="rgba(15,23,42,0.35)")),
                hovertemplate="ŷ=%{x:.2f}<br>Residual=%{y:.2f}<extra></extra>",
            )
        )
        residual_fig.add_hline(y=0, line_dash="dash")
        residual_fig.update_layout(
            title="Residuals versus fitted values",
            template="plotly_white",
            height=360,
            margin=dict(l=10, r=10, t=55, b=10),
            xaxis_title="Predicted value (ŷ)",
            yaxis_title="Residual",
        )
        st.plotly_chart(residual_fig, width="stretch")

    with d2:
        influence_fig = go.Figure()
        influence_fig.add_trace(
            go.Scatter(
                x=fit.leverage,
                y=np.abs(fit.residuals),
                mode="markers+text",
                text=[str(i) for i in range(1, len(x) + 1)],
                textposition="top center",
                name="Influence map",
                hovertemplate="Leverage=%{x:.3f}<br>|Residual|=%{y:.3f}<extra></extra>",
            )
        )
        influence_fig.update_layout(
            title="Influence map: leverage versus absolute residual",
            template="plotly_white",
            height=360,
            margin=dict(l=10, r=10, t=55, b=10),
            xaxis_title="Leverage",
            yaxis_title="|Residual|",
        )
        st.plotly_chart(influence_fig, width="stretch")

    st.markdown("### Current observations and diagnostic measures")
    diag_df = pd.DataFrame(
        {
            "obs": np.arange(1, len(x) + 1),
            "x": x,
            "y": y,
            "y_hat": fit.y_hat,
            "residual": fit.residuals,
            "abs_residual": np.abs(fit.residuals),
            "leverage": fit.leverage,
            "cooks_d": fit.cooks_distance,
        }
    ).round(4)
    st.dataframe(diag_df, width="stretch", height=320)

with tab4:
    render_regularization_tab("Ridge", "ridge", default_log_alpha=-1.0)

with tab5:
    render_regularization_tab("LASSO", "lasso", default_log_alpha=-1.4)

with tab6:
    render_polynomial_tab()

with tab7:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Can students beat OLS?")
    st.markdown(
        "Use the sliders to create a human-chosen line. Reveal the true relationship when you are ready to compare it with the OLS solution."
    )
    show_true_relationship = st.toggle(
        "Show true relationship",
        value=False,
        key="show_true_relationship",
        help="Reveal the true formula, the OLS benchmark line, and error metrics for your line and the OLS-fitted line.",
    )

    g1, g2 = st.columns([1.25, 1.0])
    with g1:
        guess_slope = st.slider("Your slope guess", -6.0, 6.0, float(np.clip(round(fit.slope, 2), -6.0, 6.0)), 0.05, key="guess_slope")
        guess_intercept = st.slider("Your intercept guess", -15.0, 15.0, float(np.clip(round(fit.intercept, 2), -15.0, 15.0)), 0.05, key="guess_intercept")

        guess_line = guess_intercept + guess_slope * x_grid
        guess_y_hat = guess_intercept + guess_slope * x
        guess_residuals = y - guess_y_hat
        guess_sse = float(np.sum(guess_residuals**2))
        guess_rmse = float(np.sqrt(np.mean(guess_residuals**2)))
        guess_test_results = evaluate_on_test(x_test, y_test, guess_slope, guess_intercept)
        guess_truth_rmse = float(np.sqrt(np.mean((guess_line - true_line) ** 2)))
        ols_truth_rmse = float(np.sqrt(np.mean((est_line - true_line) ** 2)))
        gap = guess_sse - fit.sse

        relationship_scale = float(st.session_state.relationship_scale)
        if dgp == "Linear":
            true_formula = rf"E[y \mid x] = {true_intercept:.2f} {true_slope:+.2f}x"
        elif dgp == "Quadratic (misspecified for OLS)":
            effective_quadratic = quad_strength / relationship_scale**2
            true_formula = rf"E[y \mid x] = {true_intercept:.2f} {true_slope:+.2f}x {effective_quadratic:+.3f}x^2"
        elif dgp == "Exponential (misspecified for OLS)":
            curvature = quad_strength / 8.0
            if curvature <= 1e-9:
                true_formula = rf"E[y \mid x] = {true_intercept:.2f} {true_slope:+.2f}x"
            else:
                exponential_rate = curvature / relationship_scale
                exponential_multiplier = true_slope / exponential_rate
                true_formula = (
                    rf"E[y \mid x] = {true_intercept:.2f} "
                    rf"{exponential_multiplier:+.2f}\left(e^{{{exponential_rate:.3f}x}} - 1\right)"
                )
        else:  # Piecewise / drift
            left_slope = true_slope + 0.75 * quad_strength / relationship_scale
            right_slope = true_slope - 0.75 * quad_strength / relationship_scale
            true_formula = (
                rf"E[y \mid x] = \begin{{cases}}"
                rf"{true_intercept:.2f} {left_slope:+.2f}x, & x < 0 \\ "
                rf"{true_intercept:.2f} {right_slope:+.2f}x, & x \ge 0"
                rf"\end{{cases}}"
            )

        error_metrics = pd.DataFrame(
            [
                {
                    "Line": "Student-fitted",
                    "Train SSE": guess_sse,
                    "Train RMSE": guess_rmse,
                    "Test RMSE": guess_test_results.rmse,
                    "Test MAE": guess_test_results.mae,
                    "Test R²": guess_test_results.r2,
                    "True-relationship RMSE": guess_truth_rmse,
                },
                {
                    "Line": "OLS-fitted",
                    "Train SSE": fit.sse,
                    "Train RMSE": fit.rmse,
                    "Test RMSE": test_results.rmse,
                    "Test MAE": test_results.mae,
                    "Test R²": test_results.r2,
                    "True-relationship RMSE": ols_truth_rmse,
                },
            ]
        ).round(3)

        guess_fig = go.Figure()
        guess_fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Observed data",
                marker=dict(size=9, line=dict(width=1, color="rgba(15,23,42,0.35)")),
            )
        )
        guess_fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=guess_line,
                mode="lines",
                name="Your line",
                line=dict(width=3, dash="dash"),
            )
        )
        if show_true_relationship:
            guess_fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=est_line,
                    mode="lines",
                    name="OLS line",
                    line=dict(width=4),
                )
            )
        guess_fig.update_layout(
            title=(
                "Your line and the OLS solution"
                if show_true_relationship
                else "Your fitted line"
            ),
            template="plotly_white",
            height=430,
            margin=dict(l=10, r=10, t=55, b=10),
            xaxis_title="x",
            yaxis_title="y",
        )
        st.plotly_chart(guess_fig, width="stretch")

    with g2:
        if show_true_relationship:
            better_text = "Perfect match with OLS" if abs(gap) < 1e-9 else ("Above OLS" if gap > 0 else "Below OLS")
            st.markdown("**True relationship**")
            st.latex(true_formula)
            if heteroskedastic:
                st.caption("Observed y also includes normally distributed error whose standard deviation increases with |x|.")
            else:
                st.caption(f"Observed y also includes normal error with standard deviation {noise_sd:.2f}.")

            st.markdown("**Error comparison**")
            st.caption(
                "Training metrics use the revealed training observations; test metrics use the fixed held-out sample. "
                "True-relationship RMSE compares each fitted line with the noise-free relationship across the plotted x-range."
            )
            st.dataframe(error_metrics, hide_index=True, width="stretch")
            st.markdown(
                f"""
                <div class="teacher-box">
                <strong>Training SSE gap:</strong> {gap:.3f} ({better_text})
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Fit your line first, then turn on **Show true relationship** to reveal the formula and comparison metrics.")
        st.markdown(
            """
            **Discussion prompts**
            - Why does OLS usually beat eyeballing?  
            - With very few points, why can many lines look plausible?  
            - How does the best line become easier to spot as n grows?  
            - What does this reveal about optimization versus visual intuition?
            """
        )
    st.markdown("</div>", unsafe_allow_html=True)
