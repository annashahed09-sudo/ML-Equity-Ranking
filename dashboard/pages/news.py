"""
News Intelligence — Financial news ingestion, sentiment analysis, event detection, entity extraction.
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
    st.markdown("# News Intelligence")
    st.markdown(
        "<p style='color: #8b949e; margin-top: -1rem;'>"
        "Sentiment analysis • Event detection • Entity extraction • News timeline</p>",
        unsafe_allow_html=True,
    )

    rng = np.random.default_rng(42)
    n_days = 90
    dates = pd.date_range("2024-10-01", periods=n_days, freq="B")

    # ── Sentiment Overview ────────────────────────────────────────────────
    render_section_header("Market Sentiment", "Aggregate news sentiment across sectors")

    sectors = ["Technology", "Healthcare", "Finance", "Energy", "Consumer",
               "Industrial", "Materials", "Utilities", "Real Estate", "Communication"]

    sector_sentiment = rng.uniform(-0.3, 0.5, len(sectors))
    sector_volume = rng.integers(50, 500, len(sectors))

    col1, col2 = st.columns([2, 1.5])

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sectors, y=sector_sentiment,
            marker_color=["#3fb950" if s > 0.1 else "#f85149" if s < -0.1 else "#d29922"
                          for s in sector_sentiment],
            text=[f"{s:.1%}" for s in sector_sentiment],
            textposition="outside",
        ))
        fig.update_layout(
            title="Aggregate Sentiment by Sector",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=40, b=80),
            height=350,
            font=dict(color="#8b949e", size=10),
            xaxis=dict(showgrid=False, tickangle=-45),
            yaxis=dict(showgrid=True, gridcolor="#21262d", title="Sentiment Score",
                       tickformat=".0%"),
        )
        plotly_figure(fig, "sector_sentiment")

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Sentiment Summary")
        avg_sent = np.mean(sector_sentiment)
        pct_pos = np.mean(sector_sentiment > 0.1) * 100
        pct_neg = np.mean(sector_sentiment < -0.1) * 100
        st.markdown(f"- **Average Sentiment:** {avg_sent:.1%}")
        st.markdown(f"- **Positive Sectors:** {pct_pos:.0f}%")
        st.markdown(f"- **Negative Sectors:** {pct_neg:.0f}%")
        st.markdown(f"- **Articles Analyzed:** {sector_volume.sum():,}")
        st.markdown(f"- **Confidence Threshold:** >60%")
        st.markdown("---")
        st.markdown("**Model: FinBERT**")
        st.markdown(
            "<span style='color: #8b949e; font-size: 0.85rem;'>"
            "Fine-tuned BERT for financial sentiment. Lexicon fallback for "
            "low-confidence predictions. Rolling 5-day weighted average.</span>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Sentiment Time Series ─────────────────────────────────────────────
    render_section_header("Sentiment Time Series", "Rolling 5-day average sentiment")

    # Simulate sentiment over time
    tech_sent = 0.3 + 0.15 * np.sin(np.linspace(0, 3 * np.pi, n_days)) + rng.normal(0, 0.1, n_days)
    fin_sent = 0.1 + 0.1 * np.cos(np.linspace(0, 2 * np.pi, n_days)) + rng.normal(0, 0.08, n_days)
    health_sent = 0.2 + 0.08 * np.sin(np.linspace(0, 2.5 * np.pi, n_days)) + rng.normal(0, 0.06, n_days)

    fig = go.Figure()
    for name, data, color in [
        ("Technology", tech_sent, "#58a6ff"),
        ("Finance", fin_sent, "#3fb950"),
        ("Healthcare", health_sent, "#bc8cff"),
    ]:
        fig.add_trace(go.Scatter(
            x=dates, y=data, mode="lines", name=name,
            line=dict(color=color, width=2),
        ))
    fig.add_hline(y=0, line_color="#484f58", line_width=1)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=10, b=40),
        height=300,
        font=dict(color="#8b949e", size=11),
        xaxis=dict(showgrid=True, gridcolor="#21262d"),
        yaxis=dict(showgrid=True, gridcolor="#21262d", title="Sentiment Score",
                   tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    plotly_figure(fig, "sentiment_ts")

    # ── Event Detection ──────────────────────────────────────────────────
    render_section_header("Detected Events", "Market-moving events from recent news")

    events = pd.DataFrame({
        "Date": ["2025-01-08", "2025-01-07", "2025-01-06", "2025-01-03", "2025-01-02"],
        "Ticker": ["NVDA", "JPM", "TSLA", "AAPL", "XOM"],
        "Event Type": ["Product Launch", "Earnings", "Regulatory", "Analyst Upgrade", "M&A Speculation"],
        "Sentiment": [0.82, 0.71, -0.45, 0.63, 0.38],
        "Confidence": [0.91, 0.87, 0.78, 0.85, 0.72],
        "Impact Score": [8.5, 7.2, 6.8, 6.1, 5.4],
        "Source": ["Reuters", "Bloomberg", "WSJ", "CNBC", "FT"],
    })

    st.dataframe(
        events.style.format({
            "Sentiment": "{:.0%}",
            "Confidence": "{:.0%}",
            "Impact Score": "{:.1f}/10",
        }).applymap(
            lambda v: "color: #3fb950" if isinstance(v, (int, float)) and v > 0.5
            else "color: #f85149" if isinstance(v, (int, float)) and v < 0
            else "",
            subset=["Sentiment"],
        ),
        use_container_width=True, hide_index=True,
    )

    # ── Entity Extraction ────────────────────────────────────────────────
    render_section_header("Entity Extraction", "Named entities from financial news")

    entities = pd.DataFrame({
        "Entity": ["NVIDIA Corporation", "Federal Reserve", "JPMorgan Chase",
                    "Tesla Inc", "Apple Inc", "Exxon Mobil"],
        "Type": ["ORG", "GOV", "ORG", "ORG", "ORG", "ORG"],
        "Mentions": [342, 287, 198, 165, 142, 98],
        "Avg Sentiment": [0.72, -0.18, 0.45, -0.32, 0.38, 0.12],
        "Sectors": ["Technology", "Macro/Policy", "Finance",
                     "Consumer", "Technology", "Energy"],
    })
    st.dataframe(entities, use_container_width=True, hide_index=True)

    # ── News Volume ──────────────────────────────────────────────────────
    render_section_header("News Volume", "Article count by category over time")

    article_counts = rng.integers(30, 100, (7, n_days // 15 + 1))
    categories = ["Earnings", "M&A", "Macro", "Regulatory", "CEO Changes",
                   "Product", "Litigation"]

    fig = go.Figure()
    for i, cat in enumerate(categories):
        fig.add_trace(go.Bar(
            name=cat,
            x=[f"W{k+1}" for k in range(article_counts.shape[1])],
            y=article_counts[i],
        ))
    fig.update_layout(
        barmode="stack",
        title="Weekly Article Volume by Category",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=40, b=40),
        height=350,
        font=dict(color="#8b949e", size=10),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#21262d", title="Article Count"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    plotly_figure(fig, "news_volume")
