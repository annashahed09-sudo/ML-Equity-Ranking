"""
Factor Explorer — Factor performance, exposures, correlations, and time-series analysis.
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
    st.markdown("# Factor Explorer")
    st.markdown(
        "<p style='color: #8b949e; margin-top: -1rem;'>"
        "Factor returns • Correlations • Rolling performance • Exposure analysis</p>",
        unsafe_allow_html=True,
    )

    rng = np.random.default_rng(42)
    n_days = 504
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")

    # ── Factor Selection ─────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        families = st.multiselect(
            "Factor Families",
            ["Value", "Momentum", "Quality", "Volatility", "Liquidity", "Growth", "Profitability"],
            default=["Value", "Momentum", "Quality", "Volatility"],
        )
    with col2:
        window = st.selectbox("Rolling Window", [21, 63, 126, 252], index=3)
    with col3:
        normalize = st.checkbox("Z-score normalize", value=True)

    # ── Simulated Factor Returns ──────────────────────────────────────────
    factor_map = {
        "Value": {"color": "#58a6ff", "ret": 0.0004, "vol": 0.008},
        "Momentum": {"color": "#3fb950", "ret": 0.0006, "vol": 0.010},
        "Quality": {"color": "#bc8cff", "ret": 0.0003, "vol": 0.006},
        "Volatility": {"color": "#d29922", "ret": 0.0002, "vol": 0.005},
        "Liquidity": {"color": "#f85149", "ret": 0.0001, "vol": 0.007},
        "Growth": {"color": "#79c0ff", "ret": 0.0005, "vol": 0.009},
        "Profitability": {"color": "#ff7b72", "ret": 0.00035, "vol": 0.0065},
    }

    factor_returns = {}
    factor_cum = {}
    for fam in families:
        if fam in factor_map:
            info = factor_map[fam]
            returns = info["ret"] + info["vol"] * rng.standard_t(4, n_days)
            factor_returns[fam] = pd.Series(returns, index=dates)
            factor_cum[fam] = (1 + returns).cumprod()

    if not families:
        st.info("Select at least one factor family above.")
        return

    # ── KPI Row ──────────────────────────────────────────────────────────
    kpis = []
    for fam in families:
        if fam in factor_returns:
            r = factor_returns[fam]
            ann_ret = r.mean() * 252 * 100
            ann_vol = r.std() * np.sqrt(252) * 100
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            kpis.append((f"{fam} Sharpe", f"{sharpe:.2f}", f"R:{ann_ret:.1f}% V:{ann_vol:.1f}%", "accent"))
    if kpis:
        render_kpi_row(kpis[:6], cols=min(6, len(kpis)))

    # ── Cumulative Returns ────────────────────────────────────────────────
    render_section_header("Cumulative Factor Returns", "Equal-weighted factor portfolio performance")

    fig = go.Figure()
    for fam in families:
        if fam in factor_cum:
            fig.add_trace(go.Scatter(
                x=dates, y=factor_cum[fam].values,
                mode="lines", name=fam,
                line=dict(color=factor_map[fam]["color"], width=2),
            ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=10, b=40),
        height=350,
        font=dict(color="#8b949e", size=11),
        xaxis=dict(showgrid=True, gridcolor="#21262d"),
        yaxis=dict(showgrid=True, gridcolor="#21262d", title="Cumulative Return", tickformat=".2f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    plotly_figure(fig, "factor_cumulative")

    # ── Correlation Matrix ───────────────────────────────────────────────
    render_section_header("Factor Correlation Matrix", "Long-term pairwise correlations")

    factor_df = pd.DataFrame(factor_returns)
    corr = factor_df.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=60, r=20, t=10, b=60),
        height=400,
        font=dict(color="#8b949e", size=11),
        xaxis=dict(side="bottom", tickangle=-45),
        yaxis=dict(tickangle=-45),
    )
    plotly_figure(fig, "factor_corr")

    # ── Rolling Factor Metrics ───────────────────────────────────────────
    render_section_header("Rolling Factor Metrics", f"{window}-day rolling statistics")

    col1, col2 = st.columns(2)

    with col1:
        # Rolling Sharpe
        fig = go.Figure()
        for fam in families:
            if fam in factor_returns:
                roll_sharpe = factor_returns[fam].rolling(window).mean() / \
                              factor_returns[fam].rolling(window).std().clip(lower=1e-10)
                roll_sharpe = roll_sharpe * np.sqrt(252)
                fig.add_trace(go.Scatter(
                    x=dates, y=roll_sharpe.values,
                    mode="lines", name=fam,
                    line=dict(color=factor_map[fam]["color"], width=1.5),
                ))
        fig.add_hline(y=0, line_color="#484f58", line_width=1)
        fig.update_layout(
            title=f"Rolling {window}d Sharpe Ratio",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=40, b=40),
            height=300,
            font=dict(color="#8b949e", size=10),
            xaxis=dict(showgrid=True, gridcolor="#21262d"),
            yaxis=dict(showgrid=True, gridcolor="#21262d"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        plotly_figure(fig, "rolling_sharpe")

    with col2:
        # Rolling Volatility
        fig = go.Figure()
        for fam in families:
            if fam in factor_returns:
                roll_vol = factor_returns[fam].rolling(window).std() * np.sqrt(252) * 100
                fig.add_trace(go.Scatter(
                    x=dates, y=roll_vol.values,
                    mode="lines", name=fam,
                    line=dict(color=factor_map[fam]["color"], width=1.5),
                ))
        fig.update_layout(
            title=f"Rolling {window}d Volatility (ann. %)",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=40, b=40),
            height=300,
            font=dict(color="#8b949e", size=10),
            xaxis=dict(showgrid=True, gridcolor="#21262d"),
            yaxis=dict(showgrid=True, gridcolor="#21262d", title="Volatility (%)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        plotly_figure(fig, "rolling_vol")

    # ── Factor Return Distribution ────────────────────────────────────────
    render_section_header("Return Distributions", "Factor return histogram comparison")

    fig = go.Figure()
    for fam in families:
        if fam in factor_returns:
            fig.add_trace(go.Histogram(
                x=factor_returns[fam].values * 100,
                name=fam,
                opacity=0.6,
                nbinsx=60,
                marker_color=factor_map[fam]["color"],
            ))
    fig.update_layout(
        barmode="overlay",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=10, b=40),
        height=350,
        font=dict(color="#8b949e", size=11),
        xaxis=dict(showgrid=True, gridcolor="#21262d", title="Daily Return (%)"),
        yaxis=dict(showgrid=True, gridcolor="#21262d", title="Frequency"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    plotly_figure(fig, "factor_dist")

    # ── Factor Summary Table ─────────────────────────────────────────────
    render_section_header("Factor Summary Statistics", "Full statistical breakdown")

    summary_data = []
    for fam in families:
        if fam in factor_returns:
            r = factor_returns[fam]
            summary_data.append({
                "Factor": fam,
                "Mean Ret (bps/d)": f"{r.mean() * 10000:.1f}",
                "Vol (ann%)": f"{r.std() * np.sqrt(252) * 100:.1f}%",
                "Sharpe": f"{r.mean() * 252 / (r.std() * np.sqrt(252) + 1e-10):.2f}",
                "Max DD": f"{(1 + r).cumprod().div((1 + r).cumprod().cummax()).min() * 100 - 100:.1f}%",
                "Skew": f"{r.skew():.2f}",
                "Kurtosis": f"{r.kurtosis():.2f}",
                "Hit Rate": f"{(r > 0).mean() * 100:.1f}%",
            })
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
