"""
QERP Landing Page — Futuristic glassmorphism hero with feature showcase and quick actions.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.charts import (
    plotly_figure,
    render_section_header,
)


def render() -> None:
    rng = np.random.default_rng(42)

    # ── Hero Section ──────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-section">
            <div style="display: inline-block; padding: 0.3rem 1rem;
                        background: rgba(0, 212, 255, 0.06);
                        border: 1px solid rgba(0, 212, 255, 0.12);
                        border-radius: 100px; font-size: 0.7rem;
                        color: #00d4ff; text-transform: uppercase;
                        letter-spacing: 0.12em; font-weight: 600;
                        margin-bottom: 1.5rem;">
                v2.0.0 — Institutional Grade
            </div>
            <div class="hero-title">
                <span>Quantitative Equity<br></span>
                <span class="gradient-text">Research Platform</span>
            </div>
            <div class="hero-subtitle">
                Cross-sectional equity ranking, factor-driven analysis, portfolio optimization,
                and risk management powered by institutional-grade machine learning.
            </div>
            <div style="display: flex; gap: 0.75rem; justify-content: center; flex-wrap: wrap;">
                <button onclick="alert('Navigate to Research Workspace')"
                        style="padding: 0.65rem 1.5rem; background: linear-gradient(135deg, #00d4ff, #7c3aed);
                               border: none; border-radius: 8px; color: white; font-weight: 600;
                               font-size: 0.9rem; cursor: pointer;
                               box-shadow: 0 4px 20px rgba(0, 212, 255, 0.25);
                               transition: all 0.2s ease;">
                    ▶ Launch Pipeline
                </button>
                <button onclick="alert('Navigate to Executive Dashboard')"
                        style="padding: 0.65rem 1.5rem; background: rgba(56, 189, 248, 0.06);
                               border: 1px solid rgba(56, 189, 248, 0.15); border-radius: 8px;
                               color: #94a3b8; font-weight: 500; font-size: 0.9rem; cursor: pointer;
                               transition: all 0.2s ease;">
                    View Dashboard
                </button>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI Row ──────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    metrics = [
        ("Total AUM", "$12.4B", "+2.3% (1M)", "accent"),
        ("Sharpe Ratio", "1.84", "+0.12 vs benchmark", "green"),
        ("Models Deployed", "12", "6 families", "green"),
        ("Avg Information Coefficient", "0.068", "z-score: 2.31", "violet"),
        ("Coverage", "S&P 500", "500 tickers", "amber"),
    ]

    for i, (label, value, delta, color) in enumerate(metrics):
        col_class = {
            "accent": "glass-accent",
            "green": "glass-green",
            "violet": "glass-violet",
            "amber": "glass-amber",
            "rose": "glass-rose",
        }.get(color, "")

        with [col1, col2, col3, col4, col5][i]:
            st.markdown(
                f"""
                <div class="glass {col_class}">
                    <div style="color: #475569; font-size: 0.65rem; text-transform: uppercase;
                                letter-spacing: 0.08em; font-weight: 600; margin-bottom: 0.25rem;">
                        {label}
                    </div>
                    <div style="color: #f1f5f9; font-size: 1.5rem; font-weight: 700;
                                letter-spacing: -0.02em;">
                        {value}
                    </div>
                    <div style="color: #64748b; font-size: 0.7rem; margin-top: 0.1rem;">
                        {delta}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Platform Tabs ────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["Active Models", "Market Overview", "Recent Activity", "Quick Actions"])

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            render_section_header("Model Performance", "Live IC and Sharpe by model type")

            models_data = pd.DataFrame({
                "Model": ["LGBM Ranker", "XGBoost", "Stacking Ensemble",
                          "Ridge", "Neural MLP", "Random Forest"],
                "IC": [0.078, 0.072, 0.068, 0.042, 0.058, 0.062],
                "Sharpe": [2.12, 1.98, 1.88, 1.21, 1.52, 1.65],
                "Status": ["Active", "Active", "Active", "Active", "Beta", "Active"],
            })

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=models_data["Model"],
                y=models_data["IC"],
                marker_color=["#00d4ff", "#7c3aed", "#10b981", "#38bdf8", "#f59e0b", "#f43f5e"],
                text=[f"{v:.4f}" for v in models_data["IC"]],
                textposition="outside",
                textfont=dict(size=10),
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=40, r=20, t=10, b=60),
                height=280,
                font=dict(color="#94a3b8", size=10, family="Inter"),
                xaxis=dict(showgrid=False, tickangle=-30),
                yaxis=dict(showgrid=True, gridcolor="rgba(56,189,248,0.06)",
                          title="Information Coefficient"),
            )
            plotly_figure(fig, "home_model_perf")

        with col2:
            st.markdown(
                """
                <div class="glass" style="height: 100%;">
                    <div style="color: #475569; font-size: 0.65rem; text-transform: uppercase;
                                letter-spacing: 0.08em; font-weight: 600; margin-bottom: 0.75rem;">
                        System Overview
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 0.6rem;">
                """,
                unsafe_allow_html=True,
            )

            system_items = [
                ("Models", "12 active", "active"),
                ("Data Sources", "4 connected", "active"),
                ("Pipeline", "Walk-forward", "active"),
                ("Last Run", "2h ago", "active"),
                ("Factors", "25 signals", "active"),
                ("Coverage", "500 stocks", "active"),
            ]

            for label, value, status in system_items:
                dot_class = f"status-dot {status}"
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; align-items: center;
                                padding: 0.4rem 0; border-bottom: 1px solid rgba(56,189,248,0.04);">
                        <span style="color: #94a3b8; font-size: 0.8rem;">{label}</span>
                        <span style="display: flex; align-items: center; gap: 0.4rem;">
                            <span style="color: #f1f5f9; font-size: 0.8rem; font-weight: 500;">{value}</span>
                            <span class="{dot_class}"></span>
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div></div>", unsafe_allow_html=True)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            render_section_header("Factor Performance", "12-month rolling Sharpe by factor")

            factors = ["Value", "Momentum", "Quality", "Volatility", "Growth", "Size"]
            factor_sharpe = [0.68, 0.81, 0.74, 0.65, 0.64, 0.33]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=factors,
                y=factor_sharpe,
                marker_color=["#00d4ff", "#7c3aed", "#10b981", "#38bdf8", "#f59e0b", "#f43f5e"],
                text=[f"{s:.2f}" for s in factor_sharpe],
                textposition="outside",
                textfont=dict(size=11),
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=40, r=20, t=10, b=40),
                height=280,
                font=dict(color="#94a3b8", size=10, family="Inter"),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(56,189,248,0.06)",
                          title="Sharpe Ratio"),
            )
            plotly_figure(fig, "home_factor_sharpe")

        with col2:
            render_section_header("Risk Metrics", "Current portfolio risk decomposition")

            risk_labels = ["Market", "Factor", "Idiosyncratic", "Liquidity", "FX"]
            risk_values = [35, 28, 18, 12, 7]

            fig = go.Figure(data=[go.Pie(
                labels=risk_labels,
                values=risk_values,
                hole=0.55,
                marker=dict(colors=["#00d4ff", "#7c3aed", "#10b981", "#38bdf8", "#f59e0b"]),
                textinfo="label+percent",
                textposition="outside",
                textfont=dict(size=10),
                hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
            )])
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=10, b=20),
                height=280,
                font=dict(color="#94a3b8", size=10, family="Inter"),
                showlegend=False,
            )
            plotly_figure(fig, "home_risk_decomp")

    with tab3:
        render_section_header("Recent Pipeline Runs", "Last 5 walk-forward experiments")

        activity_data = pd.DataFrame({
            "Run ID": ["EXP-2025001", "EXP-2025002", "EXP-2025003", "EXP-2025004", "EXP-2025005"],
            "Model": ["LGBM Ranker", "XGBoost", "Stacking Ensemble", "Ridge", "Neural MLP"],
            "IC": [0.0782, 0.0715, 0.0681, 0.0423, 0.0584],
            "Sharpe": [2.12, 1.98, 1.88, 1.21, 1.52],
            "Status": ["Completed", "Completed", "Completed", "Completed", "Running"],
        })

        def color_status(val):
            if val == "Completed":
                return "color: #10b981"
            elif val == "Running":
                return "color: #00d4ff"
            elif val == "Failed":
                return "color: #f43f5e"
            return ""

        st.dataframe(
            activity_data.style.format({
                "IC": "{:.4f}",
                "Sharpe": "{:.2f}",
            }).map(color_status, subset=["Status"]),
            width='stretch',
            hide_index=True,
        )

    with tab4:
        cols = st.columns(3)
        actions = [
            ("▶ Run Pipeline", "Configure and execute walk-forward research", "accent"),
            ("📊 View Dashboard", "Executive overview and KPIs", "green"),
            ("📈 Factor Explorer", "Analyze factor returns and exposures", "violet"),
            ("💼 Portfolio Builder", "Optimize with MVO, Risk Parity, BL", "amber"),
            ("⚠️ Risk Analytics", "VaR, CVaR, drawdown analysis", "rose"),
            ("📰 News Intelligence", "Sentiment analysis and event detection", "accent"),
        ]

        for i, (title, desc, color) in enumerate(actions):
            col = cols[i % 3]
            with col:
                st.markdown(
                    f"""
                    <div class="feature-card" onclick="alert('Navigate to page')"
                         style="cursor: pointer; text-align: left; padding: 1.25rem;">
                        <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.3rem;
                                    color: #f1f5f9;">
                            {title}
                        </div>
                        <div style="font-size: 0.8rem; color: #64748b; line-height: 1.5;">
                            {desc}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ── Features Grid ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; margin: 1.5rem 0 0.5rem;">
            <span style="font-size: 1.1rem; font-weight: 600; color: #f1f5f9;">
                Core Capabilities
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    features = [
        ("📈", "Factor Engine", "7 academically-grounded factor families with 25+ signals"),
        ("🧠", "ML Models", "12 model types with unified factory interface"),
        ("📊", "Portfolio Optimization", "5 institutional optimizers: MVO, RP, BL, Min Var, Factor"),
        ("⚠️", "Risk Analytics", "VaR, CVaR, factor decomposition, shrinkage covariance"),
        ("✅", "Walk-Forward Validation", "Purging and embargo windows (Lopez de Prado)"),
        ("📰", "News Intelligence", "FinBERT sentiment, entity extraction, event detection"),
        ("💡", "Explainability", "SHAP, permutation importance, PDP, ICE plots"),
        ("🔬", "Research Pipeline", "End-to-end orchestration from data to backtest"),
    ]

    cols = st.columns(4)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 4]:
            st.markdown(
                f"""
                <div class="feature-card" style="text-align: center; padding: 1.25rem;">
                    <div style="font-size: 1.75rem; margin-bottom: 0.5rem;">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
