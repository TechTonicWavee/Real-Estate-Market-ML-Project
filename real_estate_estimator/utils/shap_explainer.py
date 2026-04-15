"""
shap_explainer.py — SHAP-based model explainability.

Produces a top-N bar chart (Plotly) showing which features pushed the
predicted price UP (green) or DOWN (red), with dollar-amount labels.
"""

import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go


def get_shap_explanation(model, input_df: pd.DataFrame,
                         feature_names: list, top_n: int = 10,
                         predicted_log_price: float = None) -> pd.DataFrame:
    """
    Compute SHAP values and convert them from log-scale to approximate
    dollar impact so the chart is interpretable by end users.

    Parameters
    ----------
    model : XGBRegressor
    input_df : pd.DataFrame   (single row, aligned to training features)
    feature_names : list[str]
    top_n : int                number of features to show
    predicted_log_price : float  log1p predicted price (used for $ conversion)

    Returns
    -------
    shap_df : pd.DataFrame with columns [feature, shap_value, abs_shap]
              where shap_value is in approximate dollars.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    # Convert SHAP values from log-scale to approximate dollar impact
    # Using the derivative approach: dollar ≈ shap_log * exp(predicted_log_price)
    if predicted_log_price is not None:
        scale_factor = np.expm1(predicted_log_price)
    else:
        scale_factor = 1.0

    raw = shap_values[0] if hasattr(shap_values, "__len__") and len(shap_values.shape) > 1 else shap_values[0]

    dollar_shap = raw * scale_factor

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": dollar_shap,
        "abs_shap": np.abs(dollar_shap),
    }).sort_values("abs_shap", ascending=False).head(top_n)

    return shap_df


def plot_shap_bar(shap_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart — green for value-adding factors,
    red for value-reducing factors, with dollar labels.
    """
    colors = ["#10b981" if v > 0 else "#ef4444" for v in shap_df["shap_value"]]

    fig = go.Figure(
        go.Bar(
            x=shap_df["shap_value"],
            y=shap_df["feature"],
            orientation="h",
            marker_color=colors,
            text=[
                f"+${v:,.0f}" if v > 0 else f"-${abs(v):,.0f}"
                for v in shap_df["shap_value"]
            ],
            textposition="outside",
            textfont=dict(size=12, family="Inter, sans-serif"),
        )
    )

    fig.update_layout(
        title=dict(
            text="Why This Price? — Top Contributing Factors",
            font=dict(size=16, family="Inter, sans-serif", color="#1a1a2e"),
        ),
        xaxis_title="Impact on Price ($)",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=13, color="#374151"),
        height=420,
        margin=dict(l=20, r=100, t=60, b=30),
        xaxis=dict(gridcolor="#e5e7eb", zerolinecolor="#9ca3af"),
    )
    return fig
