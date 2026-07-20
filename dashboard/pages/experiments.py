"""
Experiment Tracking — Run history, parameter comparison, metrics dashboard.
"""

from __future__ import annotations

from datetime import datetime, timedelta

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
    st.markdown("# Experiment Tracking")
    st.markdown(
        "<p style='color: #8b949e; margin-top: -1rem;'>"
        "Run history • Parameter comparison • Results dashboard • Configuration management</p>",
        unsafe_allow_html=True,
    )

    rng = np.random.default_rng(42)
    n_experiments = 25

    # ── Experiment History ────────────────────────────────────────────────
    render_section_header("Experiment Runs", "Recent experiment execution history")

    models = ["LGBM Ranker", "XGBoost", "Stacking Ensemble", "Ridge",
              "Random Forest", "Neural MLP", "CatBoost", "Elastic Net"]
    factor_sets = ["Value+Momentum+Quality", "All Factors", "Value+Momentum",
                    "Fundamentals", "Technical", "Sentiment+Fundamentals"]
    status = ["Completed", "Completed", "Completed", "Failed", "Running"]
    status_weights = [0.75, 0.15, 0.05, 0.03, 0.02]

    experiments = []
    for i in range(n_experiments):
        hours_ago = rng.integers(1, 720)
        experiments.append({
            "Run ID": f"EXP-{2025001 + i}",
            "Timestamp": (datetime.today() - timedelta(hours=int(hours_ago))).strftime("%Y-%m-%d %H:%M"),
            "Model": rng.choice(models),
            "Factor Set": rng.choice(factor_sets),
            "Validation": rng.choice(["Walk Forward", "Purged CV", "Expanding Window"]),
            "IC (Mean)": f"{rng.uniform(0.03, 0.085):.4f}",
            "Sharpe": f"{rng.uniform(1.0, 2.2):.2f}",
            "Duration": f"{rng.uniform(5, 180):.0f}s",
            "Status": rng.choice(status, p=status_weights),
        })

    experiments_df = pd.DataFrame(experiments)

    # Color status column
    def color_status(val):
        if val == "Completed":
            return "color: #3fb950"
        elif val == "Running":
            return "color: #58a6ff"
        elif val == "Failed":
            return "color: #f85149"
        return ""

    st.dataframe(
        experiments_df.style.applymap(color_status, subset=["Status"]),
        use_container_width=True, hide_index=True,
    )

    # ── Recent Results Summary ────────────────────────────────────────────
    render_section_header("Recent Results", "Last 5 completed experiments")

    recent = experiments_df[experiments_df["Status"] == "Completed"].head(5)
    if len(recent) > 0:
        best_ic = recent["IC (Mean)"].str.replace(",", ".").astype(float).max()
        best_sharpe = recent["Sharpe"].astype(float).max()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Completed Runs", f"{len(recent)}")
        col2.metric("Best IC", f"{best_ic:.4f}")
        col3.metric("Best Sharpe", f"{best_sharpe:.2f}")
        col4.metric("Avg Duration", f"{recent['Duration'].str.replace('s', '').astype(float).mean():.0f}s")

    # ── Performance Distribution ──────────────────────────────────────────
    render_section_header("Performance Distribution", "IC and Sharpe across all experiments")

    ic_values = np.array([float(e["IC (Mean)"]) for e in experiments])
    sharpe_values = np.array([float(e["Sharpe"]) for e in experiments])
    model_names = [e["Model"] for e in experiments]

    # Create a mapping for colors
    unique_models = list(set(model_names))
    model_colors = {m: c for m, c in zip(unique_models, [
        "#58a6ff", "#3fb950", "#bc8cff", "#d29922", "#f85149",
        "#79c0ff", "#e3b341", "#ff7b72"
    ])}

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        for model in unique_models:
            mask = np.array(model_names) == model
            fig.add_trace(go.Violin(
                y=ic_values[mask],
                x=[model] * mask.sum(),
                name=model,
                box_visible=True,
                meanline_visible=True,
                line_color=model_colors[model],
                fillcolor=model_colors[model],
                opacity=0.6,
            ))
        fig.update_layout(
            title="IC Distribution by Model",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=40, b=100),
            height=350,
            font=dict(color="#8b949e", size=10),
            xaxis=dict(showgrid=False, tickangle=-45),
            yaxis=dict(showgrid=True, gridcolor="#21262d", title="IC"),
        )
        plotly_figure(fig, "ic_violin")

    with col2:
        fig = go.Figure()
        for model in unique_models:
            mask = np.array(model_names) == model
            fig.add_trace(go.Violin(
                y=sharpe_values[mask],
                x=[model] * mask.sum(),
                name=model,
                box_visible=True,
                meanline_visible=True,
                line_color=model_colors[model],
                fillcolor=model_colors[model],
                opacity=0.6,
            ))
        fig.update_layout(
            title="Sharpe Distribution by Model",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=40, b=100),
            height=350,
            font=dict(color="#8b949e", size=10),
            xaxis=dict(showgrid=False, tickangle=-45),
            yaxis=dict(showgrid=True, gridcolor="#21262d", title="Sharpe"),
        )
        plotly_figure(fig, "sharpe_violin")

    # ── Experiment Comparison ─────────────────────────────────────────────
    render_section_header("Experiment Comparison", "Select and compare runs")

    selected = st.multiselect(
        "Select Experiments to Compare",
        [e["Run ID"] for e in experiments],
        default=[experiments[0]["Run ID"], experiments[1]["Run ID"]],
    )

    if selected:
        compare_df = experiments_df[experiments_df["Run ID"].isin(selected)]
        st.dataframe(compare_df, use_container_width=True, hide_index=True)

        # Radar chart for comparison
        if len(selected) >= 2:
            metrics = ["IC (Mean)", "Sharpe", "IC Sharpe"]
            fig = go.Figure()
            for _, exp in compare_df.iterrows():
                values = [
                    float(exp["IC (Mean)"]),
                    float(exp["Sharpe"]) / 3,  # Normalize
                    float(exp["IC (Mean)"]) * 10,  # Proxy
                ]
                fig.add_trace(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=metrics + [metrics[0]],
                    name=exp["Run ID"],
                    fill="toself",
                    opacity=0.6,
                ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, gridcolor="#30363d"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                title="Experiment Comparison",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8b949e", size=11),
                height=400,
            )
            plotly_figure(fig, "exp_radar")
