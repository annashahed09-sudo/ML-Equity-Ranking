"""
Backtest Explorer — Walk-forward simulation results, performance attribution, trade analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from dashboard.components.charts import (
    plotly_figure,
    render_kpi_row,
    render_section_header,
)


def render() -> None:
    st.markdown("# Backtest Explorer")
    st.markdown(
        "<p style='color: #8b949e; margin-top: -1rem;'>"
        "Walk-forward • Performance attribution • Trade analysis • Benchmark comparison</p>",
        unsafe_allow_html=True,
    )

    rng = np.random.default_rng(42)
    n_days = 756
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")

    strategy_returns = 0.0006 + 0.006 * rng.standard_t(5, n_days)
    benchmark_returns = 0.0003 + 0.01 * rng.normal(0, 1, n_days)
    strategy_cum = (1 + strategy_returns).cumprod()
    benchmark_cum = (1 + benchmark_returns).cumprod()

    # Strategy config
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Strategy Return", f"{(strategy_cum[-1] - 1) * 100:.1f}%", "+2.3% vs SPY")
    with col2:
        strategy_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        st.metric("Sharpe Ratio", f"{strategy_sharpe:.2f}", "Walk-forward avg")
    with col3:
        strategy_mdd = (strategy_cum / strategy_cum.cummax() - 1).min()
        st.metric("Max Drawdown", f"{strategy_mdd * 100:.1f}%", "Recovered in 45d")
    with col4:
        te = (strategy_returns - benchmark_returns).std() * np.sqrt(252)
        st.metric("Tracking Error", f"{te * 100:.2f}%", "Annualized")

    # Cumulative returns
    render_section_header("Cumulative Performance", "Strategy vs benchmark")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=(strategy_cum - 1) * 100,
        mode="lines", name="Strategy",
        line=dict(color="#3fb950", width=2),
        fill="tozeroy", fillcolor="rgba(63, 185, 80, 0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=(benchmark_cum - 1) * 100,
        mode="lines", name="S&P 500 (SPY)",
        line=dict(color="#8b949e", width=1.5, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=(strategy_cum - benchmark_cum + 1) * 100 - 100,
        mode="lines", name="Active Return",
        line=dict(color="#58a6ff", width=1.5),
        fill="tozeroy", fillcolor="rgba(88, 166, 255, 0.08)",
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=10, b=40),
        height=400,
        font=dict(color="#8b949e", size=11),
        xaxis=dict(showgrid=True, gridcolor="#21262d"),
        yaxis=dict(showgrid=True, gridcolor="#21262d", title="Cumulative Return (%)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    plotly_figure(fig, "cum_perf")

    # Rolling metrics
    render_section_header("Rolling Performance Metrics", "252-day rolling Sharpe, volatility, and alpha")

    window = 252
    roll_sharpe = strategy_returns.rolling(window).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    )
    roll_vol = strategy_returns.rolling(window).std() * np.sqrt(252) * 100
    roll_alpha = strategy_returns.rolling(window).mean() - benchmark_returns.rolling(window).mean()
    roll_alpha = roll_alpha * 252 * 100

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)

    for i, (data, name, color, title) in enumerate([
        (roll_sharpe, "Rolling Sharpe", "#58a6ff", "Rolling 252d Sharpe"),
        (roll_vol, "Rolling Vol", "#f85149", "Rolling 252d Volatility (%)"),
        (roll_alpha, "Rolling Alpha", "#3fb950", "Rolling 252d Alpha (%)"),
    ]):
        fig.add_trace(go.Scatter(
            x=dates, y=data.values, mode="lines",
            name=name, line=dict(color=color, width=1.5),
        ), row=i + 1, col=1)
        fig.add_hline(y=0, line_color="#484f58", line_width=1, row=i + 1, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=10, b=40),
        height=500,
        font=dict(color="#8b949e", size=10),
        showlegend=False,
        xaxis3=dict(showgrid=True, gridcolor="#21262d"),
    )
    plotly_figure(fig, "rolling_metrics")

    # Performance attribution
    render_section_header("Performance Attribution", "Brinson-style factor attribution")

    attr_data = pd.DataFrame({
        "Factor": ["Value", "Momentum", "Quality", "Size", "Volatility",
                    "Sector Allocation", "Security Selection", "Timing", "Trading Costs"],
        "Allocation Effect": [0.82, 1.21, 0.45, -0.12, 0.31, 0.64, 0.0, 0.0, 0.0],
        "Selection Effect": [0.34, 0.56, 0.28, 0.08, -0.15, 0.0, 1.42, 0.0, 0.0],
        "Interaction": [0.12, 0.08, -0.05, 0.03, 0.02, 0.0, 0.0, 0.0, 0.0],
        "Total": [1.28, 1.85, 0.68, -0.01, 0.18, 0.64, 1.42, -0.08, -0.45],
    })

    col1, col2 = st.columns([1.5, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=attr_data["Factor"], y=attr_data["Total"],
            marker_color=["#58a6ff" if v > 0 else "#f85149" for v in attr_data["Total"]],
            text=[f"{v:.2f}%" for v in attr_data["Total"]],
            textposition="outside",
        ))
        fig.update_layout(
            title="Total Attribution Effect (%)",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=40, b=80),
            height=350,
            font=dict(color="#8b949e", size=10),
            xaxis=dict(showgrid=False, tickangle=-45),
            yaxis=dict(showgrid=True, gridcolor="#21262d", title="Attribution (%)"),
        )
        plotly_figure(fig, "attribution")

    with col2:
        st.dataframe(
            attr_data.style.format({
                "Allocation Effect": "{:.2f}%",
                "Selection Effect": "{:.2f}%",
                "Interaction": "{:.2f}%",
                "Total": "{:.2f}%",
            }).map(
                lambda v: "color: #3fb950" if isinstance(v, (int, float)) and v > 0
                else "color: #f85149" if isinstance(v, (int, float)) and v < 0
                else "",
                subset=["Total"],
            ),
            width='stretch', hide_index=True,
        )

    # Monthly returns heatmap
    render_section_header("Monthly Returns Heatmap", "Strategy monthly performance")

    years_months = pd.date_range("2022-01-01", periods=24, freq="ME")
    monthly_returns = np.random.default_rng(42).normal(0.008, 0.025, 24)
    monthly_matrix = monthly_returns.reshape(2, 12)

    fig = go.Figure(data=go.Heatmap(
        z=monthly_matrix * 100,
        x=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        y=["2023", "2022"],
        colorscale="RdYlBu_r",
        zmid=0,
        text=np.round(monthly_matrix * 100, 1),
        texttemplate="%{text}%",
        hovertemplate="%{y} %{x}: %{text}%<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=10, b=40),
        height=250,
        font=dict(color="#8b949e", size=11),
        xaxis=dict(side="bottom"),
    )
    plotly_figure(fig, "monthly_heatmap")

    # Trade statistics
    render_section_header("Trade Statistics", "Execution and turnover analysis")

    trades = pd.DataFrame({
        "Metric": ["Total Trades", "Winners", "Losers", "Win Rate", "Avg Win",
                    "Avg Loss", "Profit Factor", "Avg Holding Period",
                    "Avg Turnover/Period", "Round-trip Cost (bps)"],
        "Value": [1247, 718, 529, "57.6%", "1.24%", "-0.87%",
                    "2.42", "18.4 days", "12.3%", "9.5"],
    })
    st.dataframe(trades, width='stretch', hide_index=True)
