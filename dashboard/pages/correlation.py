"""
Correlation Matrix — Asset, factor, and sector correlation analysis with clustering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.charts import (
    plotly_figure,
    render_kpi_row,
    render_section_header,
)


def render() -> None:
    st.markdown("# Correlation Matrix")
    st.markdown(
        "<p style='color: #8b949e; margin-top: -1rem;'>"
        "Asset correlations • Factor correlations • Rolling correlation • Cluster analysis</p>",
        unsafe_allow_html=True,
    )

    rng = np.random.default_rng(42)
    n_assets = 20
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
               "ADBE", "CRM", "NFLX", "AMD", "INTC", "QCOM", "TXN", "IBM",
               "ORCL", "NOW", "SAP", "CSCO"]

    # ── Configuration ────────────────────────────────────────────────────
    render_section_header("Configuration", "Select correlation parameters")

    col1, col2, col3 = st.columns(3)
    with col1:
        window = st.selectbox("Lookback Window", [30, 60, 126, 252], index=3)
    with col2:
        method = st.selectbox("Correlation Method", ["pearson", "spearman"], index=0)
    with col3:
        cluster = st.checkbox("Apply Hierarchical Clustering", value=True)

    # ── Asset Correlation Matrix ──────────────────────────────────────────
    render_section_header("Asset Correlation Matrix", f"{window}-day {method} correlations")

    # Generate realistic correlation matrix with sector structure
    n_sectors = 5
    sector_assign = np.repeat(range(n_sectors), n_assets // n_sectors + 1)[:n_assets]
    rng.shuffle(sector_assign)

    base_corr = np.zeros((n_assets, n_assets))
    for i in range(n_assets):
        for j in range(n_assets):
            if i == j:
                base_corr[i, j] = 1.0
            elif sector_assign[i] == sector_assign[j]:
                base_corr[i, j] = rng.uniform(0.4, 0.8)
            else:
                base_corr[i, j] = rng.uniform(0.1, 0.5)
    base_corr = (base_corr + base_corr.T) / 2
    np.fill_diagonal(base_corr, 1.0)

    # Add noise for realism
    noise = rng.normal(0, 0.05, (n_assets, n_assets))
    noise = (noise + noise.T) / 2
    corr_matrix = np.clip(base_corr + noise, -1, 1)
    np.fill_diagonal(corr_matrix, 1.0)

    # Create sector labels for display
    sector_names = ["Technology", "Finance", "Healthcare", "Consumer", "Energy"]
    ticker_labels = [f"{t} ({sector_names[sector_assign[i]][:3]})" for i, t in enumerate(tickers[:n_assets])]

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=ticker_labels,
        y=ticker_labels,
        colorscale="RdBu",
        zmin=-1, zmax=1,
        text=np.round(corr_matrix, 2),
        texttemplate="%{text:.2f}",
        hovertemplate="%{x}<br>%{y}<br>Correlation: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=120, r=20, t=10, b=120),
        height=600,
        font=dict(color="#8b949e", size=8),
        xaxis=dict(side="bottom", tickangle=-90),
        yaxis=dict(tickangle=0),
    )
    plotly_figure(fig, "asset_corr")

    # ── Correlation Statistics ────────────────────────────────────────────
    render_section_header("Correlation Summary", "Distribution and key statistics")

    col1, col2 = st.columns(2)

    with col1:
        # Extract upper triangle
        upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=upper_tri,
            nbinsx=30,
            marker_color="#58a6ff",
            opacity=0.8,
        ))
        fig.update_layout(
            title="Pairwise Correlation Distribution",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=40, b=40),
            height=300,
            font=dict(color="#8b949e", size=11),
            xaxis=dict(showgrid=True, gridcolor="#21262d", title="Correlation",
                       range=[-1, 1]),
            yaxis=dict(showgrid=True, gridcolor="#21262d", title="Count"),
        )
        plotly_figure(fig, "corr_dist")

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Statistics")
        st.markdown(f"- **Mean Correlation:** {upper_tri.mean():.3f}")
        st.markdown(f"- **Median Correlation:** {np.median(upper_tri):.3f}")
        st.markdown(f"- **Std Dev:** {upper_tri.std():.3f}")
        st.markdown(f"- **% > 0.7:** {(upper_tri > 0.7).mean() * 100:.1f}%")
        st.markdown(f"- **% < 0.0:** {(upper_tri < 0.0).mean() * 100:.1f}%")
        st.markdown(f"- **Condition Number:** {np.linalg.cond(corr_matrix):.1f}")
        st.markdown(f"- **Determinant:** {np.linalg.det(corr_matrix):.6f}")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Rolling Correlation ──────────────────────────────────────────────
    render_section_header("Rolling Correlation", "Average pairwise correlation over time")

    n_days = 504
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")

    # Simulate rolling average correlation
    rolling_corr = 0.45 + 0.15 * np.sin(np.linspace(0, 4 * np.pi, n_days)) + rng.normal(0, 0.02, n_days)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=rolling_corr,
        mode="lines", name="Avg Pairwise Correlation",
        line=dict(color="#58a6ff", width=2),
        fill="tozeroy", fillcolor="rgba(88, 166, 255, 0.08)",
    ))
    fig.add_hline(y=rolling_corr.mean(), line_dash="dash", line_color="#8b949e",
                   annotation_text=f"Mean: {rolling_corr.mean():.3f}")
    fig.update_layout(
        title="Rolling Average Pairwise Correlation",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=40, b=40),
        height=350,
        font=dict(color="#8b949e", size=11),
        xaxis=dict(showgrid=True, gridcolor="#21262d"),
        yaxis=dict(showgrid=True, gridcolor="#21262d", title="Avg Correlation",
                   range=[0, 1]),
    )
    plotly_figure(fig, "rolling_corr")

    # ── Factor Correlation ────────────────────────────────────────────────
    render_section_header("Factor Correlation", "Cross-factor correlation matrix")

    factors = ["Market Beta", "Value", "Momentum", "Quality", "Size", "Volatility",
               "Liquidity", "Growth", "Profitability"]
    n_factors = len(factors)

    factor_corr = np.zeros((n_factors, n_factors))
    for i in range(n_factors):
        for j in range(n_factors):
            if i == j:
                factor_corr[i, j] = 1.0
            else:
                factor_corr[i, j] = rng.uniform(-0.3, 0.5)
    factor_corr = (factor_corr + factor_corr.T) / 2
    np.fill_diagonal(factor_corr, 1.0)

    fig = go.Figure(data=go.Heatmap(
        z=factor_corr,
        x=factors, y=factors,
        colorscale="RdBu", zmin=-1, zmax=1,
        text=np.round(factor_corr, 2),
        texttemplate="%{text:.2f}",
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=100, r=20, t=10, b=100),
        height=500,
        font=dict(color="#8b949e", size=10),
        xaxis=dict(side="bottom", tickangle=-45),
        yaxis=dict(tickangle=-45),
    )
    plotly_figure(fig, "factor_corr_matrix")
