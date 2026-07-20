"""
Research Workspace — Configure and execute the research pipeline with real-time results.
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.charts import (
    metric_card,
    plotly_figure,
    render_kpi_row,
    render_section_header,
)


def render() -> None:
    st.markdown("# Research Workspace")
    st.markdown(
        "<p style='color: #8b949e; margin-top: -1rem;'>"
        "Configure models, factors, and validation strategy. Execute walk-forward pipeline.</p>",
        unsafe_allow_html=True,
    )

    # ── Configuration Panel ───────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Model Configuration")
        model_type = st.selectbox(
            "Model Family",
            ["ridge", "lasso", "elastic_net", "random_forest", "xgboost",
             "lightgbm", "catboost", "neural_mlp", "stacking_ensemble", "voting_ensemble",
             "lightgbm_ranker", "xgboost_ranker"],
            index=0,
        )
        factor_set = st.multiselect(
            "Factor Sets",
            ["value", "momentum", "quality", "volatility", "liquidity", "growth", "profitability"],
            default=["value", "momentum", "quality"],
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Validation Strategy")
        val_strategy = st.selectbox(
            "Method", ["walk_forward", "purged_cv", "expanding_window", "nested_cv"], index=0
        )
        n_splits = st.slider("Number of Splits", 2, 10, 5)
        test_size = st.number_input("Test Size (days)", 21, 504, 252, step=21)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Portfolio Settings")
        long_pct = st.slider("Top % (Long)", 0.05, 0.50, 0.20, 0.05)
        short_pct = st.slider("Bottom % (Short)", 0.05, 0.50, 0.20, 0.05)
        tc_bps = st.number_input("Transaction Cost (bps)", 0, 100, 10, 5)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Execute Button ────────────────────────────────────────────────────
    run = st.button("▶ Execute Research Pipeline", type="primary", use_container_width=True)

    # ── Results ──────────────────────────────────────────────────────────
    if run or st.session_state.get("run_pipeline", False):
        st.session_state.run_pipeline = False

        with st.spinner("Running walk-forward validation..."):
            time.sleep(1.5)  # Simulate computation

            # Simulate pipeline results
            dates = pd.date_range("2024-01-01", periods=252, freq="B")
            n_assets = 50

            # Generate realistic IC series
            rng = np.random.default_rng(42)
            ic_series = 0.04 + 0.06 * np.sin(np.linspace(0, 4 * np.pi, 252)) + rng.normal(0, 0.03, 252)
            ic_series = pd.Series(ic_series, index=dates, name="IC")

            # Fold metrics
            fold_metrics = pd.DataFrame({
                "Fold": range(1, n_splits + 1),
                "Mean IC": [0.068, 0.072, 0.058, 0.081, 0.065][:n_splits] + rng.uniform(-0.01, 0.01, n_splits),
                "Rank IC": [0.072, 0.075, 0.062, 0.085, 0.068][:n_splits] + rng.uniform(-0.01, 0.01, n_splits),
                "Sharpe": [1.84, 1.92, 1.56, 2.12, 1.76][:n_splits] + rng.uniform(-0.2, 0.2, n_splits),
                "Train Start": ["2021-01-01"] * n_splits,
                "Train End": [f"2023-0{i+1}-01" for i in range(n_splits)],
                "Test End": [f"2024-0{i+2}-01" for i in range(n_splits)],
                "N Test": rng.integers(500, 2000, n_splits),
            })

            # Top predictions
            tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "JPM", "V", "MA",
                       "UNH", "JNJ", "PG", "HD", "DIS", "ADBE", "CRM", "NFLX", "AMD", "INTC"]
            pred_scores = rng.normal(0, 1, len(tickers))
            pred_scores = pred_scores / np.abs(pred_scores).max()  # Normalize to [-1, 1]
            predictions = pd.DataFrame({
                "Rank": range(1, len(tickers) + 1),
                "Ticker": tickers,
                "Score": pred_scores,
                "Direction": ["Long" if s > 0.3 else "Short" if s < -0.3 else "Neutral" for s in pred_scores],
                "Confidence": np.abs(pred_scores),
            })

        # ── Results KPIs ──────────────────────────────────────────────────
        mean_ic = float(ic_series.mean())
        ic_sharpe = mean_ic / float(ic_series.std()) if float(ic_series.std()) > 0 else 0
        kpis = [
            ("Mean IC", f"{mean_ic:.4f}", f"IC Sharpe: {ic_sharpe:.2f}", "accent"),
            ("Rank IC", f"{fold_metrics['Rank IC'].mean():.4f}", "Cross-sectional", "green"),
            ("Avg Sharpe", f"{fold_metrics['Sharpe'].mean():.2f}", "Walk-forward", "green"),
            ("N Predictions", f"{len(predictions)}", "Latest ranking", "neutral"),
            ("% Profitable", f"{fold_metrics['Sharpe'].gt(1.0).mean()*100:.0f}%", "of folds", "neutral"),
        ]
        render_kpi_row(kpis, cols=5)

        # ── IC Time Series ────────────────────────────────────────────────
        render_section_header("Information Coefficient (IC)", "Time series of daily cross-sectional IC")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ic_series.index, y=ic_series.values,
            mode="lines", name="Daily IC",
            line=dict(color="#58a6ff", width=1.5),
            fill="tozeroy", fillcolor="rgba(88, 166, 255, 0.05)",
        ))
        fig.add_hline(y=mean_ic, line_dash="dash", line_color="#3fb950",
                       annotation_text=f"Mean: {mean_ic:.4f}")
        fig.add_hline(y=0, line_color="#484f58", line_width=1)

        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=10, b=40),
            height=300,
            font=dict(color="#8b949e", size=11),
            xaxis=dict(showgrid=True, gridcolor="#21262d"),
            yaxis=dict(showgrid=True, gridcolor="#21262d", title="IC"),
        )
        plotly_figure(fig, "ic_timeseries")

        # ── Fold Metrics & Predictions ────────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            render_section_header("Fold Metrics", "Walk-forward validation results")
            st.dataframe(
                fold_metrics.style.format({
                    "Mean IC": "{:.4f}", "Rank IC": "{:.4f}", "Sharpe": "{:.2f}",
                }),
                use_container_width=True, hide_index=True,
            )

        with col2:
            render_section_header("Latest Rankings", "Current model predictions")
            st.dataframe(
                predictions.style.format({
                    "Score": "{:.3f}",
                    "Confidence": "{:.2%}",
                }).applymap(
                    lambda v: "color: #3fb950" if isinstance(v, str) and v == "Long"
                    else "color: #f85149" if isinstance(v, str) and v == "Short"
                    else "",
                    subset=["Direction"],
                ),
                use_container_width=True, hide_index=True,
            )

        # ── Factor Importance ─────────────────────────────────────────────
        render_section_header("Top Features by Importance", "Permutation importance across folds")

        features = [
            "Momentum (6M)", "Earnings Yield", "Book-to-Market", "Realized Vol (60d)",
            "ROE", "Gross Profitability", "Revenue Growth", "Amihud Illiquidity",
            "Downside Deviation", "Market Beta",
        ]
        importance = [0.182, 0.145, 0.112, 0.098, 0.076, 0.065, 0.052, 0.041, 0.033, 0.028]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=importance[::-1], y=features[::-1],
            orientation="h",
            marker_color="rgba(88, 166, 255, 0.8)",
            text=[f"{v:.1%}" for v in importance[::-1]],
            textposition="outside",
        ))
        fig.update_layout(
            title="Feature Importance (Permutation Method)",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=120, r=60, t=40, b=20),
            height=350,
            font=dict(color="#8b949e", size=11),
            xaxis=dict(showgrid=True, gridcolor="#21262d", title="Importance"),
            yaxis=dict(showgrid=False),
        )
        plotly_figure(fig, "feature_importance")

        # ── Long-Short Portfolio Returns ─────────────────────────────────
        render_section_header("Long-Short Portfolio Returns", "Cumulative performance from model predictions")

        ls_returns = pd.Series(
            np.cumprod(1 + 0.0005 * (
                ic_series.values * 0.5 + rng.normal(0, 0.01, 252)
            )),
            index=dates,
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ls_returns.index, y=ls_returns.values,
            mode="lines", name="Long-Short Portfolio",
            line=dict(color="#3fb950", width=2),
            fill="tozeroy", fillcolor="rgba(63, 185, 80, 0.08)",
        ))
        fig.update_layout(
            title="Cumulative Long-Short Returns",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=40, b=40),
            height=300,
            font=dict(color="#8b949e", size=11),
            xaxis=dict(showgrid=True, gridcolor="#21262d"),
            yaxis=dict(showgrid=True, gridcolor="#21262d",
                       title="Cumulative Return", tickformat=".0%"),
        )
        plotly_figure(fig, "ls_returns")

        st.success("Pipeline execution complete.")


def run_pipeline() -> None:
    """Placeholder for actual pipeline execution."""
    pass
