"""
Executive Dashboard — High-level KPIs, market overview, and model performance summary.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.charts import (
    metric_card,
    render_kpi_row,
    render_section_header,
    plotly_figure,
)


def render() -> None:
    st.markdown("# Executive Dashboard")
    st.markdown(
        "<p style='color: #8b949e; margin-top: -1rem;'>"
        "Portfolio performance • Risk metrics • Market overview • Model health</p>",
        unsafe_allow_html=True,
    )

    # ── Top KPI Row ───────────────────────────────────────────────────────
    kpis = [
        ("Total AUM", "$12.4B", "+2.3%", "accent"),
        ("Sharpe Ratio", "1.84", "+0.12 (1M)", "green"),
        ("YTD Return", "+14.7%", "+3.2% vs SPY", "green"),
        ("Active Risk", "3.2%", "0.8% below limit", "neutral"),
        ("Max Drawdown", "-8.3%", "recovered in 45d", "orange"),
        ("IC (12M)", "0.072", "z-score: 2.31", "accent"),
    ]
    render_kpi_row(kpis, cols=6)

    # ── Market Overview ───────────────────────────────────────────────────
    render_section_header("Market Overview", "Major indices and macro context")

    col1, col2 = st.columns([2, 1])

    with col1:
        dates = pd.date_range(end=datetime.today(), periods=252, freq="B")
        spy = 100 + np.cumsum(np.random.default_rng(42).normal(0.0005, 0.01, 252))
        qqq = 100 + np.cumsum(np.random.default_rng(43).normal(0.0006, 0.012, 252))
        iwm = 100 + np.cumsum(np.random.default_rng(44).normal(0.0003, 0.015, 252))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=spy, mode="lines", name="S&P 500 (SPY)",
                                  line=dict(color="#58a6ff", width=2)))
        fig.add_trace(go.Scatter(x=dates, y=qqq, mode="lines", name="Nasdaq (QQQ)",
                                  line=dict(color="#3fb950", width=2)))
        fig.add_trace(go.Scatter(x=dates, y=iwm, mode="lines", name="Russell 2000 (IWM)",
                                  line=dict(color="#d29922", width=2)))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=20, t=20, b=40),
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(color="#8b949e", size=11),
            xaxis=dict(showgrid=True, gridcolor="#21262d", title=""),
            yaxis=dict(showgrid=True, gridcolor="#21262d", title="Cumulative Return (%)"),
        )
        plotly_figure(fig, "market_overview")

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### VIX Term Structure")
        tenors = ["1M", "2M", "3M", "6M", "1Y"]
        vix_fwd = [14.2, 15.8, 16.5, 17.2, 17.8]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tenors, y=vix_fwd, mode="lines+markers",
                                  line=dict(color="#f85149", width=2),
                                  marker=dict(size=8)))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=10, b=20),
            height=200, showlegend=False,
            font=dict(color="#8b949e", size=10),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#21262d"),
        )
        plotly_figure(fig, "vix_term")

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Factor Performance ────────────────────────────────────────────────
    render_section_header("Factor Performance (12M Rolling)", "Risk-adjusted returns by factor")

    factors_data = {
        "Factor": ["Value", "Momentum", "Quality", "Volatility", "Growth", "Size"],
        "Return (ann.)": [8.2, 12.4, 6.8, 5.1, 9.3, 3.7],
        "Volatility (ann.)": [12.1, 15.3, 9.2, 7.8, 14.5, 11.2],
        "Sharpe": [0.68, 0.81, 0.74, 0.65, 0.64, 0.33],
        "IC": [0.042, 0.067, 0.038, 0.051, 0.029, 0.018],
    }
    df_factors = pd.DataFrame(factors_data)

    col1, col2 = st.columns([3, 2])

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_factors["Factor"],
            y=df_factors["Sharpe"],
            marker_color=["#58a6ff", "#3fb950", "#bc8cff", "#d29922", "#f85149", "#8b949e"],
            text=[f"{s:.2f}" for s in df_factors["Sharpe"]],
            textposition="outside",
        ))
        fig.update_layout(
            title="Factor Sharpe Ratios",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=40, b=40),
            height=300,
            font=dict(color="#8b949e", size=11),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#21262d", title="Sharpe Ratio"),
        )
        plotly_figure(fig, "factor_sharpe")

    with col2:
        st.dataframe(
            df_factors.style.format({
                "Return (ann.)": "{:.1f}%",
                "Volatility (ann.)": "{:.1f}%",
                "Sharpe": "{:.2f}",
                "IC": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # ── Recent Activity ──────────────────────────────────────────────────
    render_section_header("Recent Activity", "Latest research pipeline runs")

    activity = pd.DataFrame({
        "Timestamp": [
            (datetime.today() - timedelta(hours=n)).strftime("%Y-%m-%d %H:%M")
            for n in [1, 3, 6, 12, 24]
        ],
        "Model": ["LGBM Ranker", "XGBoost", "Stacking Ensemble", "Ridge", "CatBoost"],
        "Universe": ["S&P 500", "S&P 500", "Russell 1000", "S&P 500", "Russell 1000"],
        "IC": [0.071, 0.065, 0.078, 0.042, 0.069],
        "Sharpe": [1.92, 1.78, 2.04, 1.21, 1.85],
        "Status": ["✓ Complete", "✓ Complete", "✓ Complete", "✓ Complete", "⚠ Warning"],
    })
    st.dataframe(activity, use_container_width=True, hide_index=True)

    # ── Quick Links ───────────────────────────────────────────────────────
    st.markdown("---")
    cols = st.columns(3)
    with cols[0]:
        st.markdown(
            '<div class="card card-accent" style="text-align: center; padding: 1rem;">'
            '<div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>'
            '<div style="font-weight: 600;">Run New Pipeline</div>'
            '<div style="color: #8b949e; font-size: 0.85rem;">Configure and execute research</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            '<div class="card card-green" style="text-align: center; padding: 1rem;">'
            '<div style="font-size: 2rem; margin-bottom: 0.5rem;">📈</div>'
            '<div style="font-weight: 600;">Portfolio Optimizer</div>'
            '<div style="color: #8b949e; font-size: 0.85rem;">MVO, Risk Parity, Black-Litterman</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            '<div class="card card-orange" style="text-align: center; padding: 1rem;">'
            '<div style="font-size: 2rem; margin-bottom: 0.5rem;">🔬</div>'
            '<div style="font-weight: 600;">Risk Analytics</div>'
            '<div style="color: #8b949e; font-size: 0.85rem;">VaR, CVaR, factor decomposition</div>'
            '</div>',
            unsafe_allow_html=True,
        )
