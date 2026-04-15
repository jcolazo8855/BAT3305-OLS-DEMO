from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="BAT 3305 - Colazo | OLS Learning Lab",
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
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    scale = np.std(x) if np.std(x) > 1e-9 else 1.0

    if dgp == "Linear":
        mean = intercept + slope * x
    elif dgp == "Quadratic (misspecified for OLS)":
        mean = intercept + slope * x + quad_strength * (x / scale) ** 2
    else:  # Piecewise / drift
        mean = intercept + slope * x
        mean = np.where(x < 0, mean + 0.75 * quad_strength * (x / scale), mean - 0.75 * quad_strength * (x / scale))

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


# ============================================================
# Simulation functions
# ============================================================
def refresh_history(
    x_grid: np.ndarray,
    dgp: str,
    intercept: float,
    slope: float,
    noise_sd: float,
    quad_strength: float,
    heteroskedastic: bool,
    note: str,
) -> FitResults:
    x = np.array(st.session_state.x, dtype=float)
    y = np.array(st.session_state.y, dtype=float)
    fit = fit_ols(x, y, x_grid)

    rng_test = np.random.default_rng(999_991)
    x_test = np.linspace(np.min(x_grid), np.max(x_grid), 220)
    y_test = generate_y(
        x=x_test,
        dgp=dgp,
        intercept=intercept,
        slope=slope,
        noise_sd=noise_sd,
        rng=rng_test,
        quad_strength=quad_strength,
        heteroskedastic=heteroskedastic,
    )
    test = evaluate_on_test(x_test, y_test, fit.slope, fit.intercept)

    current_n = len(x)
    if not st.session_state.history_n or st.session_state.history_n[-1] != current_n:
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
) -> FitResults:
    rng = np.random.default_rng(int(st.session_state.seed))
    x = rng.uniform(x_min, x_max, size=n_initial)
    y = generate_y(
        x=x,
        dgp=dgp,
        intercept=intercept,
        slope=slope,
        noise_sd=noise_sd,
        rng=rng,
        quad_strength=quad_strength,
        heteroskedastic=heteroskedastic,
    )
    st.session_state.x = list(map(float, x))
    st.session_state.y = list(map(float, y))
    st.session_state.history_n = []
    st.session_state.history_slope = []
    st.session_state.history_intercept = []
    st.session_state.history_r2 = []
    st.session_state.history_train_rmse = []
    st.session_state.history_test_rmse = []
    st.session_state.history_test_r2 = []
    st.session_state.history_mae = []
    st.session_state.history_note = []
    return refresh_history(x_grid, dgp, intercept, slope, noise_sd, quad_strength, heteroskedastic, note="reset")


def add_points(
    k: int,
    x_min: float,
    x_max: float,
    dgp: str,
    intercept: float,
    slope: float,
    noise_sd: float,
    quad_strength: float,
    heteroskedastic: bool,
    x_grid: np.ndarray,
    note: str,
) -> FitResults:
    rng = np.random.default_rng(int(st.session_state.seed) + 10_007 * len(st.session_state.x) + k)
    new_x = rng.uniform(x_min, x_max, size=k)
    new_y = generate_y(
        x=new_x,
        dgp=dgp,
        intercept=intercept,
        slope=slope,
        noise_sd=noise_sd,
        rng=rng,
        quad_strength=quad_strength,
        heteroskedastic=heteroskedastic,
    )
    st.session_state.x.extend(map(float, new_x))
    st.session_state.y.extend(map(float, new_y))
    return refresh_history(x_grid, dgp, intercept, slope, noise_sd, quad_strength, heteroskedastic, note=note)


def inject_outlier(
    x_min: float,
    x_max: float,
    intercept: float,
    slope: float,
    noise_sd: float,
    x_grid: np.ndarray,
    dgp: str,
    quad_strength: float,
    heteroskedastic: bool,
    strength: float,
) -> FitResults:
    rng = np.random.default_rng(int(st.session_state.seed) + 777_777 + len(st.session_state.x))
    direction = float(rng.choice([-1.0, 1.0]))
    if rng.random() < 0.5:
        ox = float(x_max if rng.random() < 0.5 else x_min)
    else:
        ox = float(rng.uniform(x_min, x_max))
    baseline = intercept + slope * ox
    oy = float(baseline + direction * strength * noise_sd * (1.0 + rng.random()))
    st.session_state.x.append(ox)
    st.session_state.y.append(oy)
    return refresh_history(x_grid, dgp, intercept, slope, noise_sd, quad_strength, heteroskedastic, note="outlier")


# ============================================================
# Header
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>BAT 3305 - Colazo</h1>
        <h3>Interactive OLS Regression Studio</h3>
        <p>
            Explore how ordinary least squares becomes more stable as it ingests more data.
            This lab lets students observe coefficient convergence, uncertainty reduction,
            outlier sensitivity, train-versus-test behavior, and what happens when the
            underlying data-generating process is not truly linear.
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
        ["Linear", "Quadratic (misspecified for OLS)", "Piecewise / drift"],
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
        help="Only matters for the quadratic and piecewise data-generating processes.",
    )
    heteroskedastic = st.toggle(
        "Heteroskedastic errors",
        value=False,
        help="When on, noise increases as x moves away from zero.",
    )

    st.markdown("---")
    x_min, x_max = st.slider("x-range", -25.0, 25.0, (-10.0, 10.0), 0.5)
    n_initial = st.slider("Initial sample size", 3, 80, 6, 1)
    add_k = st.slider("Points added per click", 1, 200, 10, 1)
    outlier_strength = st.slider("Outlier strength", 2.0, 12.0, 5.5, 0.5)
    st.session_state.seed = int(st.number_input("Random seed", min_value=1, max_value=999999, value=int(st.session_state.seed), step=1))

    x_grid = np.linspace(x_min, x_max, 260)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        reset_clicked = st.button("Reset", use_container_width=True)
    with col_b:
        add_clicked = st.button(f"Add {add_k}", use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        add_one_clicked = st.button("Add 1", use_container_width=True)
    with col_d:
        outlier_clicked = st.button("Outlier", use_container_width=True)

    auto_steps = st.slider("Auto-grow steps", 2, 40, 8, 1)
    auto_clicked = st.button("Run auto-growth", use_container_width=True)

    st.caption("Tip: after changing the true process, press Reset so the sample is regenerated from the new world.")


# ============================================================
# Apply controls
# ============================================================
if not st.session_state.x:
    fit = reset_simulation(
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
    )
else:
    fit = refresh_history(
        x_grid=x_grid,
        dgp=dgp,
        intercept=true_intercept,
        slope=true_slope,
        noise_sd=noise_sd,
        quad_strength=quad_strength,
        heteroskedastic=heteroskedastic,
        note="view",
    )

if reset_clicked:
    fit = reset_simulation(
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
    )
elif add_clicked:
    fit = add_points(
        k=add_k,
        x_min=x_min,
        x_max=x_max,
        dgp=dgp,
        intercept=true_intercept,
        slope=true_slope,
        noise_sd=noise_sd,
        quad_strength=quad_strength,
        heteroskedastic=heteroskedastic,
        x_grid=x_grid,
        note=f"add_{add_k}",
    )
elif add_one_clicked:
    fit = add_points(
        k=1,
        x_min=x_min,
        x_max=x_max,
        dgp=dgp,
        intercept=true_intercept,
        slope=true_slope,
        noise_sd=noise_sd,
        quad_strength=quad_strength,
        heteroskedastic=heteroskedastic,
        x_grid=x_grid,
        note="add_1",
    )
elif outlier_clicked:
    fit = inject_outlier(
        x_min=x_min,
        x_max=x_max,
        intercept=true_intercept,
        slope=true_slope,
        noise_sd=noise_sd,
        x_grid=x_grid,
        dgp=dgp,
        quad_strength=quad_strength,
        heteroskedastic=heteroskedastic,
        strength=outlier_strength,
    )
elif auto_clicked:
    for _ in range(auto_steps):
        fit = add_points(
            k=add_k,
            x_min=x_min,
            x_max=x_max,
            dgp=dgp,
            intercept=true_intercept,
            slope=true_slope,
            noise_sd=noise_sd,
            quad_strength=quad_strength,
            heteroskedastic=heteroskedastic,
            x_grid=x_grid,
            note=f"auto_{add_k}",
        )

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
)
est_line = fit.intercept + fit.slope * x_grid

# Test set for current snapshot
rng_test = np.random.default_rng(999_991)
x_test = np.linspace(x_min, x_max, 220)
y_test = generate_y(
    x=x_test,
    dgp=dgp,
    intercept=true_intercept,
    slope=true_slope,
    noise_sd=noise_sd,
    rng=rng_test,
    quad_strength=quad_strength,
    heteroskedastic=heteroskedastic,
)
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
    ("Sample size", f"{len(x)}"),
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
tab1, tab2, tab3, tab4 = st.tabs(["Playground", "Convergence & Generalization", "Diagnostics", "Student Challenge"])

with tab1:
    left, right = st.columns((1.65, 1.0))

    with left:
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
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("What students should notice")
        st.markdown(
            """
            1. **Early estimates jump around** because small samples are noisy.  
            2. **More data stabilizes the slope and intercept** when the true relationship is linear.  
            3. **Confidence and prediction bands narrow** as uncertainty falls.  
            4. **Outliers can still bend OLS sharply** because squared errors give them extra weight.  
            5. **More data does not cure misspecification** when the true pattern is not linear.
            """
        )
        st.markdown(
            f"""
            <div class="small-note">
            Estimated model right now: ŷ = {fit.intercept:.3f} + {fit.slope:.3f}x
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Quick teaching prompts")
        st.markdown(
            """
            - Ask students to predict whether the next 20 points will change the slope a lot or a little.  
            - Increase the noise and ask why convergence gets slower.  
            - Switch to a quadratic world and ask whether OLS is improving or merely settling on the best straight-line compromise.  
            - Inject an outlier and ask which coefficient will be hit harder: slope or intercept.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

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
        st.plotly_chart(fig_coef, use_container_width=True)

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
        st.plotly_chart(fig_fit, use_container_width=True)

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
        st.plotly_chart(fig_r2, use_container_width=True)

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
        st.plotly_chart(residual_fig, use_container_width=True)

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
        st.plotly_chart(influence_fig, use_container_width=True)

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
    st.dataframe(diag_df, use_container_width=True, height=320)

with tab4:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Can students beat OLS?")
    st.markdown(
        "Use the sliders to create a human-chosen line. Compare its SSE with the OLS solution. This makes the optimization target concrete."
    )

    g1, g2 = st.columns([1.25, 1.0])
    with g1:
        guess_slope = st.slider("Your slope guess", -6.0, 6.0, float(np.clip(round(fit.slope, 2), -6.0, 6.0)), 0.05, key="guess_slope")
        guess_intercept = st.slider("Your intercept guess", -15.0, 15.0, float(np.clip(round(fit.intercept, 2), -15.0, 15.0)), 0.05, key="guess_intercept")

        guess_line = guess_intercept + guess_slope * x_grid
        guess_y_hat = guess_intercept + guess_slope * x
        guess_sse = float(np.sum((y - guess_y_hat) ** 2))
        gap = guess_sse - fit.sse

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
                y=est_line,
                mode="lines",
                name="OLS line",
                line=dict(width=4),
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
        guess_fig.update_layout(
            title="Your line versus the OLS solution",
            template="plotly_white",
            height=430,
            margin=dict(l=10, r=10, t=55, b=10),
            xaxis_title="x",
            yaxis_title="y",
        )
        st.plotly_chart(guess_fig, use_container_width=True)

    with g2:
        better_text = "Perfect match with OLS" if abs(gap) < 1e-9 else ("Above OLS" if gap > 0 else "Below OLS")
        st.markdown(
            f"""
            <div class="teacher-box">
            <strong>Your SSE:</strong> {guess_sse:.3f}<br>
            <strong>OLS SSE:</strong> {fit.sse:.3f}<br>
            <strong>Gap:</strong> {gap:.3f} ({better_text})
            </div>
            """,
            unsafe_allow_html=True,
        )
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


# ============================================================
# Footer notes
# ============================================================
with st.expander("Instructor notes and suggested classroom flow"):
    st.markdown(
        """
        **Recommended sequence for a live class demo**

        1. Start in a linear world with 5–6 points and ask students to estimate the slope by eye.  
        2. Add 1 point at a time so they can feel the instability of tiny samples.  
        3. Switch to larger batches and discuss why the estimate settles down.  
        4. Turn up noise and compare how much slower convergence becomes.  
        5. Inject an outlier and ask whether OLS is robust.  
        6. Switch to a quadratic world and ask whether more data is helping OLS become “right,” or just helping it find the best straight-line approximation.  
        7. End with the Student Challenge tab to make the sum-of-squared-errors objective tangible.

        **Core learning objective**

        OLS improves with more data when the underlying relationship is stable and approximately linear because the parameter estimates become less variable.
        But more data does **not** fix outliers, heteroskedasticity, or model misspecification by itself.
        """
    )
