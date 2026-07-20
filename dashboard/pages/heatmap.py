"""
Market Heatmap — Real-time sector and industry group performance visualization.
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
    st.markdown("# Market Heatmap")
    st.markdown(
        "<p style='color: #8b949e; margin-top: -1rem;'>"
        "Sector performance • Industry groups • Factor exposure heatmap • Market map</p>",
        unsafe_allow_html=True,
    )

    rng = np.random.default_rng(42)

    # ── Sector Performance ────────────────────────────────────────────────
    render_section_header("Sector Performance", "Daily and YTD returns by sector")

    sectors = ["Technology", "Healthcare", "Financials", "Energy", "Consumer Disc.",
               "Consumer Staples", "Industrials", "Materials", "Utilities", "Real Estate",
               "Communication"]

    daily_ret = rng.normal(0.001, 0.012, len(sectors))
    ytd_ret = rng.uniform(-5, 20, len(sectors))

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        colors = ["#3fb950" if r > 0 else "#f85149" for r in daily_ret]
        fig.add_trace(go.Bar(
            x=sectors, y=daily_ret * 100,
            marker_color=colors,
            text=[f"{r*100:.2f}%" for r in daily_ret],
            textposition="outside",
        ))
        fig.update_layout(
            title="Daily Sector Returns",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=40, b=80),
            height=350,
            font=dict(color="#8b949e", size=10),
            xaxis=dict(showgrid=False, tickangle=-45),
            yaxis=dict(showgrid=True, gridcolor="#21262d", title="Return (%)"),
        )
        plotly_figure(fig, "sector_daily")

    with col2:
        fig = go.Figure()
        colors = ["#3fb950" if r > 0 else "#f85149" for r in ytd_ret]
        fig.add_trace(go.Bar(
            x=sectors, y=ytd_ret,
            marker_color=colors,
            text=[f"{r:.1f}%" for r in ytd_ret],
            textposition="outside",
        ))
        fig.update_layout(
            title="Year-to-Date Sector Returns",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=40, b=80),
            height=350,
            font=dict(color="#8b949e", size=10),
            xaxis=dict(showgrid=False, tickangle=-45),
            yaxis=dict(showgrid=True, gridcolor="#21262d", title="Return (%)"),
        )
        plotly_figure(fig, "sector_ytd")

    # ── Market Map ────────────────────────────────────────────────────────
    render_section_header("Market Map", "Treemap of sector and industry performance")

    # Build hierarchical market data
    market_data = []
    for sector in sectors:
        n_industries = rng.integers(3, 6)
        for j in range(n_industries):
            industry = f"{sector[:4]}-{j+1}"
            n_tickers = rng.integers(5, 15)
            for k in range(n_tickers // 3):
                ticker = f"{industry[:2]}{k+1:02d}"
                market_data.append({
                    "Sector": sector,
                    "Industry": industry,
                    "Ticker": ticker,
                    "Return": rng.normal(0.001, 0.02),
                    "Volume": rng.integers(1e6, 1e9),
                    "MarketCap": rng.uniform(1e9, 3e12),
                })

    market_df = pd.DataFrame(market_data)

    fig = px.treemap(
        market_df,
        path=["Sector", "Industry", "Ticker"],
        values="MarketCap",
        color="Return",
        color_continuous_scale="RdYlBu_r",
        color_continuous_midpoint=0,
        hover_data={"Return": ":.2%", "Volume": ":,.0f", "MarketCap": ":,.0f"},
    )
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=10, b=10),
        height=600,
        font=dict(color="#8b949e", size=9),
    )
    fig.update_traces(
        textfont=dict(color="white", size=9),
        hovertemplate="<b>%{label}</b><br>Return: %{customdata[0]:.2%}<br>Volume: %{customdata[1]:,.0f}<br>Mkt Cap: %{customdata[2]:$,.0f}<extra></extra>",
    )
    plotly_figure(fig, "market_map")

    # ── Factor Exposure by Sector ─────────────────────────────────────────
    render_section_header("Factor Exposure by Sector", "Heatmap of factor loadings across sectors")

    factors = ["Market Beta", "Value", "Momentum", "Quality", "Size", "Volatility", "Growth"]
    exposure_matrix = rng.normal(0, 0.3, (len(sectors), len(factors)))

    fig = go.Figure(data=go.Heatmap(
        z=exposure_matrix,
        x=factors,
        y=sectors,
        colorscale="RdBu",
        zmin=-1, zmax=1,
        text=np.round(exposure_matrix, 2),
        texttemplate="%{text:.2f}",
    ))
    fig.update_layout(
        title="Factor Exposures by Sector",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=80, r=20, t=40, b=40),
        height=400,
        font=dict(color="#8b949e", size=10),
        xaxis=dict(side="bottom"),
        yaxis=dict(tickangle=0),
    )
    plotly_figure(fig, "factor_heatmap")

    # ── Top/Bottom Performers ─────────────────────────────────────────────
    render_section_header("Top & Bottom Performers", "Best and worst performing stocks today")

    all_tickers = market_df.groupby("Ticker").agg(
        Return=("Return", "mean"),
        Sector=("Sector", "first"),
    ).reset_index()

    top5 = all_tickers.nlargest(5, "Return")
    bottom5 = all_tickers.nsmallest(5, "Return")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Top 5 Performers")
        st.dataframe(
            top5.style.format({"Return": "{:.2%}"})
            .applymap(lambda _: "color: #3fb950", subset=["Return"]),
            use_container_width=True, hide_index=True,
        )
    with col2:
        st.markdown("#### Bottom 5 Performers")
        st.dataframe(
            bottom5.style.format({"Return": "{:.2%}"})
            .applymap(lambda _: "color: #f85149", subset=["Return"]),
            use_container_width=True, hide_index=True,
        )
