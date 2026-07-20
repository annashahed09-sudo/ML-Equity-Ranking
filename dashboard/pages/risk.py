"""
Risk Analytics — VaR, CVaR, factor decomposition, tracking error, drawdown waterfall, sensitivity.
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
    st.markdown("# Risk Analytics")
    st.markdown(
        "<p style='color: #8b949e; margin-top: -1rem;'>"
        "VaR/CVaR • Factor risk decomposition • Drawdown • Beta exposure • Sensitivity</p>",
        unsafe_allow_html=True,
    )

    rng = np.random.default_rng(42)
    n_days = 756
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")
    portfolio_returns = rng.normal(0.0004, 0.008, n_days)  # 0.04% daily, 0.8% vol
    benchmark_returns = rng.normal(0.0003, 0.01, n_days)

    # ── VaR/CVaR Analysis ────────────────────────────────────────────────
    render_section_header("Value at Risk & Expected Shortfall", "Multiple confidence levels and methods")

    confidence_levels = [0.95, 0.97, 0.99, 0.995]
    var_historical = [np.percentile(portfolio_returns, (1 - c) * 100) for c in confidence_levels]
    var_parametric = [
        portfolio_returns.mean() + float(np.percentile(np.random.normal(0, 1, 100000), (1 - c) * 100)) * portfolio_returns.std()
        for c in confidence_levels
    ]
    cvar_values = []
    for c, vh in zip(confidence_levels, var_historical):
        tail = portfolio_returns[portfolio_returns <= vh]
        cvar_values.append(tail.mean() if len(tail) > 0 else vh)

    col1, col2 = st.columns([2, 1.5])

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"{c*100:.0f}%" for c in confidence_levels],
            y=[abs(v) * 10000 for v in var_historical],
            name="Historical VaR",
            marker_color="#58a6ff",
        ))
        fig.add_trace(go.Bar(
            x=[f"{c*100:.0f}%" for c in confidence_levels],
            y=[abs(v) * 10000 for v in cvar_values],
            name="CVaR (Expected Shortfall)",
            marker_color="#f85149",
        ))
        fig.update_layout(
            barmode="group",
            title="VaR & CVaR by Confidence Level",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=40, b=40),
            height=350,
            font=dict(color="#8b949e", size=11),
            xaxis=dict(showgrid=False, title="Confidence Level"),
            yaxis=dict(showgrid=True, gridcolor="#21262d", title="Loss (bps)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        plotly_figure(fig, "var_cvar")

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Risk Summary")
        var_95 = abs(np.percentile(portfolio_returns, 5)) * 100
        cvar_95 = abs(cvar_values[0]) * 100
        vol_ann = portfolio_returns.std() * np.sqrt(252) * 100
        st.markdown(f"- **95% VaR (1d):** {var_95:.2f}%")
        st.markdown(f"- **95% CVaR (1d):** {cvar_95:.2f}%")
        st.markdown(f"- **Annualized Vol:** {vol_ann:.2f}%")
        st.markdown(f"- **Skewness:** {portfolio_returns.skew():.2f}")
        st.markdown(f"- **Excess Kurtosis:** {portfolio_returns.kurtosis():.2f}")
        st.markdown("---")
        st.markdown("**VaR Methodology**")
        st.markdown(
            "<span style='color: #8b949e; font-size: 0.85rem;'>"
            "Historical VaR uses empirical quantile. Parametric VaR assumes "
            "normal distribution. CVaR (Expected Shortfall) averages losses "
            "beyond VaR threshold.</span>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Drawdown Analysis ─────────────────────────────────────────────────
    render_section_header("Drawdown Analysis", "Maximum drawdown, duration, and recovery")

    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative / running_max - 1) * 100

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=(cumulative - 1) * 100,
            mode="lines", name="Cumulative Return",
            line=dict(color="#3fb950", width=2),
            fill="tozeroy", fillcolor="rgba(63, 185, 80, 0.08)",
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=running_max * 100 - 100,
            mode="lines", name="Peak",
            line=dict(color="#8b949e", width=1, dash="dash"),
        ))
        fig.update_layout(
            title="Cumulative Return & Peak",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=40, b=40),
            height=300,
            font=dict(color="#8b949e", size=10),
            xaxis=dict(showgrid=True, gridcolor="#21262d"),
            yaxis=dict(showgrid=True, gridcolor="#21262d", title="Return (%)"),
        )
        plotly_figure(fig, "cumulative")

    with col2:
        fig = go.Figure()
        colors = ["#f85149" if d < 0 else "rgba(0,0,0,0)" for d in drawdown]
        fig.add_trace(go.Scatter(
            x=dates, y=drawdown,
            mode="lines", name="Drawdown",
            line=dict(color="#f85149", width=1.5),
            fill="tozeroy", fillcolor="rgba(248, 81, 73, 0.15)",
        ))
        fig.update_layout(
            title=f"Max Drawdown: {drawdown.min():.1f}%",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=40, b=40),
            height=300,
            font=dict(color="#8b949e", size=10),
            xaxis=dict(showgrid=True, gridcolor="#21262d"),
            yaxis=dict(showgrid=True, gridcolor="#21262d", title="Drawdown (%)"),
        )
        plotly_figure(fig, "drawdown")

    # ── Factor Risk Decomposition ─────────────────────────────────────────
    render_section_header("Factor Risk Decomposition", "Systematic vs idiosyncratic risk breakdown")

    factors = ["Market Beta", "Value", "Momentum", "Quality", "Size", "Volatility",
               "Liquidity", "Growth", "Interest Rate", "Credit Spread", "FX", "Commodity"]
    factor_contrib = np.array([25, 15, 12, 8, 7, 6, 5, 4, 3, 2, 2, 1])
    idiosyncratic = 10

    fig = go.Figure(data=[go.Pie(
        labels=factors + ["Idiosyncratic"],
        values=list(factor_contrib) + [idiosyncratic],
        hole=0.5,
        marker=dict(
            colors=["#58a6ff", "#3fb950", "#bc8cff", "#d29922", "#ff7b72",
                    "#79c0ff", "#e3b341", "#a5d6ff", "#db6d28", "#56d364",
                    "#8b949e", "#c9d1d9", "#484f58"]
        ),
        textinfo="label+percent",
        textposition="outside",
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    )])
    fig.update_layout(
        title="Portfolio Risk Decomposition",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=40, b=20),
        height=450,
        font=dict(color="#8b949e", size=10),
        showlegend=False,
    )
    plotly_figure(fig, "risk_decomp")

    # ── Risk Metrics ─────────────────────────────────────────────────────
    render_section_header("Risk Metrics Summary", "Key risk statistics")

    metrics_data = pd.DataFrame({
        "Metric": [
            "95% VaR (1d)", "95% CVaR (1d)", "99% VaR (1d)",
            "Annualized Volatility", "Tracking Error (ann.)",
            "Information Ratio", "Beta (vs SPY)", "Alpha (ann.)",
            "Max Drawdown", "Drawdown Duration (max)",
            "Skewness", "Excess Kurtosis",
        ],
        "Portfolio": [
            f"{abs(var_historical[0])*100:.2f}%",
            f"{abs(cvar_values[0])*100:.2f}%",
            f"{abs(var_historical[2])*100:.2f}%",
            f"{portfolio_returns.std()*np.sqrt(252)*100:.2f}%",
            f"{np.std(portfolio_returns - benchmark_returns)*np.sqrt(252)*100:.2f}%",
            f"{(portfolio_returns.mean()-portfolio_returns.std()/np.sqrt(252)*0)/portfolio_returns.std()*np.sqrt(252):.2f}",
            f"{np.corrcoef(portfolio_returns, benchmark_returns)[0,1]:.2f}",
            f"{portfolio_returns.mean()*252*100:.2f}%",
            f"{drawdown.min():.1f}%",
            f"{np.argmax(drawdown.cumsum() == drawdown.cumsum().max()) if drawdown.cumsum().max() > 0 else 0:d} days",
            f"{portfolio_returns.skew():.2f}",
            f"{portfolio_returns.kurtosis():.2f}",
        ],
        "Benchmark": [
            f"{abs(np.percentile(benchmark_returns, 5))*100:.2f}%",
            f"{abs(benchmark_returns[benchmark_returns <= np.percentile(benchmark_returns, 5)].mean())*100:.2f}%",
            f"{abs(np.percentile(benchmark_returns, 1))*100:.2f}%",
            f"{benchmark_returns.std()*np.sqrt(252)*100:.2f}%",
            "—",
            "—",
            "1.00",
            "0.00%",
            f"{(1+benchmark_returns).cumprod().div((1+benchmark_returns).cumprod().cummax()).min()*100-100:.1f}%",
            "—",
            f"{benchmark_returns.skew():.2f}",
            f"{benchmark_returns.kurtosis():.2f}",
        ],
    })
    st.dataframe(metrics_data, use_container_width=True, hide_index=True)

    # ── Rolling Risk Metrics ──────────────────────────────────────────────
    render_section_header("Rolling Risk Metrics", "60-day rolling VaR, volatility, and beta")

    window = 60
    roll_var = portfolio_returns.rolling(window).quantile(0.05)
    roll_vol = portfolio_returns.rolling(window).std() * np.sqrt(252) * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=dates, y=abs(roll_var) * 100,
        mode="lines", name="95% VaR (1d, %)",
        line=dict(color="#f85149", width=1.5),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=dates, y=roll_vol,
        mode="lines", name="Ann. Volatility (%)",
        line=dict(color="#58a6ff", width=1.5),
    ), secondary_y=True)
    fig.update_layout(
        title=f"Rolling {window}-day Risk Metrics",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=40, b=40),
        height=350,
        font=dict(color="#8b949e", size=11),
        xaxis=dict(showgrid=True, gridcolor="#21262d"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="VaR (%)", gridcolor="#21262d", secondary_y=False)
    fig.update_yaxes(title_text="Volatility (%)", gridcolor="#21262d", secondary_y=True)
    plotly_figure(fig, "rolling_risk")
