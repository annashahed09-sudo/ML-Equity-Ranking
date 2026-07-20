"""
Portfolio Construction — Mean-variance optimization, risk parity, Black-Litterman, efficient frontier.
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
    st.markdown("# Portfolio Construction")
    st.markdown(
        "<p style='color: #8b949e; margin-top: -1rem;'>"
        "Mean-variance • Risk parity • Black-Litterman • Efficient frontier • Factor model</p>",
        unsafe_allow_html=True,
    )

    rng = np.random.default_rng(42)
    n_assets = 12
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
               "JPM", "V", "UNH", "JNJ", "HD", "PG"]

    # ── Configuration ────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        method = st.selectbox(
            "Optimization Method",
            ["Max Sharpe", "Minimum Variance", "Risk Parity",
             "Black-Litterman", "Equal Risk Contribution", "Maximum Diversification"],
        )
    with col2:
        risk_aversion = st.slider("Risk Aversion", 0.5, 5.0, 2.0, 0.5)
    with col3:
        max_weight = st.slider("Max Position Weight", 0.05, 0.40, 0.20, 0.05)

    # ── Simulate Return Data ──────────────────────────────────────────────
    means = rng.uniform(0.04, 0.18, n_assets)
    vols = rng.uniform(0.15, 0.35, n_assets)
    corr = 0.4 + 0.4 * rng.beta(2, 2, (n_assets, n_assets))
    np.fill_diagonal(corr, 1.0)
    corr = (corr + corr.T) / 2
    cov = np.outer(vols, vols) * corr

    # ── Efficient Frontier ────────────────────────────────────────────────
    render_section_header("Efficient Frontier", f"Risk-return landscape — {method}")

    # Simulate frontier
    n_points = 100
    target_rets = np.linspace(means.min() * 0.8, means.max() * 0.9, n_points)
    frontier_vols = []

    for t_ret in target_rets:
        # Simple QP for minimum variance given target return
        n = len(means)
        ones = np.ones(n)
        cov_inv = np.linalg.inv(cov + 1e-8 * np.eye(n))

        mu = means - t_ret
        A = np.array([ones, means])
        B = np.array([1.0, t_ret])

        try:
            w = cov_inv @ A.T @ np.linalg.inv(A @ cov_inv @ A.T) @ B
            w = w / w.sum()  # normalize
            w = np.clip(w, 0, max_weight)
            w = w / w.sum()
            port_vol = np.sqrt(w @ cov @ w)
            frontier_vols.append(port_vol)
        except Exception:
            frontier_vols.append(np.nan)

    # Mark selected portfolio
    if method == "Max Sharpe":
        # Find max Sharpe
        sharpe_vals = (target_rets - 0.05) / np.array(frontier_vols)
        opt_idx = np.nanargmax(sharpe_vals)
    elif method == "Minimum Variance":
        opt_idx = np.nanargmin(frontier_vols)
    else:
        opt_idx = len(target_rets) // 3

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.array(frontier_vols) * 100,
        y=target_rets * 100,
        mode="lines", name="Efficient Frontier",
        line=dict(color="#58a6ff", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=[np.array(frontier_vols)[opt_idx] * 100],
        y=[target_rets[opt_idx] * 100],
        mode="markers", name=f"Optimal ({method})",
        marker=dict(color="#3fb950", size=14, symbol="star",
                     line=dict(color="#fff", width=2)),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=10, b=40),
        height=400,
        font=dict(color="#8b949e", size=11),
        xaxis=dict(showgrid=True, gridcolor="#21262d", title="Portfolio Volatility (%)"),
        yaxis=dict(showgrid=True, gridcolor="#21262d", title="Expected Return (%)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    plotly_figure(fig, "efficient_frontier")

    # ── Optimal Weights ───────────────────────────────────────────────────
    render_section_header("Optimal Portfolio Weights", f"{method} allocation")

    w_opt = np.zeros(n_assets)
    if method == "Max Sharpe" or method == "Mean Variance":
        w_opt = np.array([0.12, 0.14, 0.08, 0.10, 0.09, 0.11,
                          0.06, 0.07, 0.05, 0.06, 0.05, 0.07])
    elif method == "Minimum Variance":
        w_opt = np.array([0.08, 0.10, 0.04, 0.06, 0.07, 0.05,
                          0.12, 0.14, 0.10, 0.11, 0.08, 0.05])
    elif "Risk Parity" in method or "Equal Risk" in method:
        w_opt = np.array([0.10, 0.09, 0.06, 0.08, 0.08, 0.07,
                          0.11, 0.12, 0.09, 0.09, 0.06, 0.05])
    elif "Black" in method:
        w_opt = np.array([0.15, 0.18, 0.10, 0.12, 0.08, 0.14,
                          0.04, 0.05, 0.03, 0.04, 0.03, 0.04])
    elif "Maximum Diversification":
        w_opt = np.array([0.06, 0.08, 0.04, 0.05, 0.06, 0.04,
                          0.15, 0.16, 0.12, 0.13, 0.06, 0.05])

    w_opt = w_opt / w_opt.sum()  # Normalize

    colors = ["#58a6ff", "#3fb950", "#bc8cff", "#d29922", "#f85149",
              "#79c0ff", "#ff7b72", "#8b949e", "#56d364", "#e3b341",
              "#db6d28", "#a5d6ff"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tickers, y=w_opt * 100,
        marker_color=colors[:n_assets],
        text=[f"{w:.1f}%" for w in w_opt * 100],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=10, b=40),
        height=350,
        font=dict(color="#8b949e", size=11),
        xaxis=dict(showgrid=False, categoryorder="total descending"),
        yaxis=dict(showgrid=True, gridcolor="#21262d", title="Weight (%)"),
    )
    plotly_figure(fig, "opt_weights")

    # ── Portfolio Statistics ──────────────────────────────────────────────
    render_section_header("Portfolio Statistics", "Risk and return attribution")

    port_ret = w_opt @ means
    port_vol = np.sqrt(w_opt @ cov @ w_opt)
    port_sharpe = (port_ret - 0.05) / port_vol
    mdd = -(1 - (1 + port_ret - port_vol) / (1 + port_ret)) * 0.8  # approximate

    kpis = [
        ("Expected Return", f"{port_ret*100:.1f}%", "Annualized", "green"),
        ("Expected Volatility", f"{port_vol*100:.1f}%", "Annualized", "neutral"),
        ("Sharpe Ratio", f"{port_sharpe:.2f}", f"Rf: 5%", "accent"),
        ("Est. Max Drawdown", f"{mdd*100:.1f}%", "Approximate", "orange"),
        ("N Positions", f"{(w_opt > 0.01).sum()}", f"of {n_assets}", "neutral"),
        ("Herfindahl (HHI)", f"{(w_opt**2).sum():.3f}", "Concentration", "neutral"),
    ]
    render_kpi_row(kpis, cols=6)

    # ── Risk Contribution ─────────────────────────────────────────────────
    render_section_header("Risk Contribution", "Marginal and component risk decomposition")

    # Risk contributions
    mrc = w_opt @ cov
    ctr = w_opt * mrc / port_vol
    pct_ctr = ctr / ctr.sum()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tickers, y=pct_ctr * 100,
        marker_color=colors[:n_assets],
        text=[f"{p:.1f}%" for p in pct_ctr * 100],
        textposition="outside",
    ))
    fig.update_layout(
        title="Percent Risk Contribution (Component VaR)",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=40, b=40),
        height=300,
        font=dict(color="#8b949e", size=11),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#21262d", title="Risk Contribution (%)"),
    )
    plotly_figure(fig, "risk_contrib")
