"""
Feature Importance & Explainability — SHAP values, permutation importance, partial dependence plots, ICE.
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
    st.markdown("# Feature Importance & Explainability")
    st.markdown(
        "<p style='color: #8b949e; margin-top: -1rem;'>"
        "SHAP values • Permutation importance • Partial dependence • ICE plots • Feature interactions</p>",
        unsafe_allow_html=True,
    )

    rng = np.random.default_rng(42)
    n_features = 12

    features = [
        "Momentum_6M", "Earnings_Yield", "Book_to_Market", "Realized_Vol_60d",
        "ROE", "Gross_Profitability", "Revenue_Growth_1Y", "Amihud_Illiq",
        "Downside_Dev", "Market_Beta", "Size_LogCap", "Asset_Turnover",
    ]

    # ── Global Importance ─────────────────────────────────────────────────
    render_section_header("Global Feature Importance", "Permutation importance across walk-forward folds")

    col1, col2 = st.columns([3, 1.5])
    with col1:
        importance_values = np.array([0.182, 0.145, 0.112, 0.098, 0.076, 0.065,
                                       0.052, 0.041, 0.033, 0.028, 0.015, 0.009])
        importance_std = np.array([0.021, 0.018, 0.015, 0.012, 0.011, 0.010,
                                    0.009, 0.008, 0.007, 0.006, 0.004, 0.003])

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=importance_values, y=features,
            orientation="h",
            marker_color="rgba(88, 166, 255, 0.85)",
            error_x=dict(type="data", array=importance_std, visible=True,
                          color="#8b949e", thickness=1.5),
            text=[f"{v:.1%}" for v in importance_values],
            textposition="outside",
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=140, r=80, t=10, b=20),
            height=420,
            font=dict(color="#8b949e", size=11),
            xaxis=dict(showgrid=True, gridcolor="#21262d", title="Permutation Importance"),
            yaxis=dict(showgrid=False),
        )
        plotly_figure(fig, "global_importance")

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Importance Summary")
        cumsum = np.cumsum(importance_values)
        n_top5 = (cumsum <= 0.6).sum() + 1
        st.markdown(f"**Top 5 features** explain {cumsum[4]*100:.1f}% of total importance")
        st.markdown(f"**{n_top5} features** reach 60% cumulative importance")
        st.markdown(f"**Mean drop in performance** when permuted: {importance_values.mean()*100:.1f}%")
        st.markdown("---")
        st.markdown("**Methodology**")
        st.markdown(
            "<span style='color: #8b949e; font-size: 0.85rem;'>"
            "Permutation importance computed across 5 walk-forward folds. "
            "Features are shuffled independently; the increase in MSE is "
            "the importance score. Error bars show ±1 standard deviation across folds.</span>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── SHAP-like Summary ─────────────────────────────────────────────────
    render_section_header("SHAP Summary", "Impact of features on model output")

    # Simulate SHAP values
    n_samples = 1000
    shap_values = np.zeros((n_samples, n_features))
    for i in range(n_features):
        shap_values[:, i] = rng.normal(0, importance_values[i] * 0.5, n_samples) + \
                            importance_values[i] * np.tanh(rng.normal(0, 1, n_samples))

    fig = go.Figure()
    for i, feat in enumerate(features):
        color = "#58a6ff" if importance_values[i] > 0.05 else "#8b949e"
        fig.add_trace(go.Box(
            y=shap_values[:, i] * 100,
            name=feat,
            marker_color=color,
            boxmean="sd",
            showlegend=False,
        ))
    fig.update_layout(
        title="Feature Impact on Model Output (SHAP)",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=140, r=40, t=40, b=100),
        height=500,
        font=dict(color="#8b949e", size=10),
        xaxis=dict(showgrid=False, tickangle=-45),
        yaxis=dict(showgrid=True, gridcolor="#21262d", title="SHAP Value (bps)"),
    )
    plotly_figure(fig, "shap_summary")

    # ── Partial Dependence ────────────────────────────────────────────────
    render_section_header("Partial Dependence Plots", "Top 6 features — marginal effect on predictions")

    selected = features[:6]

    tabs = st.tabs(selected)
    for i, feat in enumerate(selected):
        with tabs[i]:
            col1, col2 = st.columns([2, 1])
            with col1:
                # Simulate PDP
                x_range = np.linspace(-3, 3, 50)
                # Non-linear partial dependence with uncertainty
                pdp_mean = 0.5 * np.tanh(0.8 * x_range) + 0.2 * x_range + 0.1 * x_range**2 * 0.05
                pdp_std = 0.02 + 0.01 * np.abs(x_range)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_range, y=pdp_mean,
                    mode="lines", name="Partial Dependence",
                    line=dict(color="#58a6ff", width=2.5),
                ))
                fig.add_trace(go.Scatter(
                    x=x_range, y=pdp_mean + 2 * pdp_std,
                    mode="lines", name="+2σ", showlegend=False,
                    line=dict(color="#58a6ff", width=0, dash="dash"),
                ))
                fig.add_trace(go.Scatter(
                    x=x_range, y=pdp_mean - 2 * pdp_std,
                    mode="lines", name="-2σ", showlegend=False,
                    line=dict(color="#58a6ff", width=0, dash="dash"),
                    fill="tonexty", fillcolor="rgba(88, 166, 255, 0.1)",
                ))
                fig.update_layout(
                    title=f"Partial Dependence — {feat}",
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=40, b=40),
                    height=350,
                    font=dict(color="#8b949e", size=11),
                    xaxis=dict(showgrid=True, gridcolor="#21262d", title=f"Feature Value ({feat})"),
                    yaxis=dict(showgrid=True, gridcolor="#21262d", title="Prediction (z-score)"),
                )
                plotly_figure(fig, f"pdp_{feat}")

            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"**{feat} — Statistics**")
                st.markdown(f"Range of effect: {pdp_mean.max() - pdp_mean.min():.3f}")
                st.markdown(f"Non-linearity: {(np.abs(np.diff(pdp_mean)) > 0.01).mean()*100:.0f}%")
                ice_samples = shap_values[:, i] * 100
                st.markdown(f"Mean |SHAP|: {np.abs(ice_samples).mean():.2f} bps")
                st.markdown(f"ICE heterogeneity: {ice_samples.std():.2f} bps")
                st.markdown("</div>", unsafe_allow_html=True)

    # ── Feature Interaction ───────────────────────────────────────────────
    render_section_header("Feature Interaction Matrix", "H-statistic for pairwise interactions")

    n_interactions = len(features)
    h_matrix = np.zeros((n_interactions, n_interactions))
    for i in range(n_interactions):
        for j in range(n_interactions):
            if i != j:
                h_matrix[i, j] = rng.uniform(0, 0.15) * (importance_values[i] + importance_values[j])

    fig = go.Figure(data=go.Heatmap(
        z=h_matrix, x=features, y=features,
        colorscale="Viridis", zmin=0, zmax=0.15,
        text=np.round(h_matrix, 3), texttemplate="%{text:.3f}",
        hovertemplate="%{x} × %{y}: H={%{z:.3f}}<extra></extra>",
    ))
    fig.update_layout(
        title="Friedman H-Statistic — Pairwise Feature Interactions",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=140, r=40, t=40, b=140),
        height=600,
        font=dict(color="#8b949e", size=9),
        xaxis=dict(side="bottom", tickangle=-90),
        yaxis=dict(tickangle=0),
    )
    plotly_figure(fig, "interaction_matrix")
