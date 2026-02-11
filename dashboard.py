"""Interactive Streamlit dashboard for ML Equity Intelligence."""

from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.express as px

from src.market_intelligence import MarketIntelligenceService

st.set_page_config(page_title="ML Equity Intelligence", page_icon="📈", layout="wide")

st.title("📈 ML Equity Intelligence Dashboard")
st.caption("Cross-sectional ranking + review sentiment intelligence")

service = MarketIntelligenceService()

with st.sidebar:
    st.header("Configuration")
    tickers_str = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,NVDA,AMZN,META")
    start_date = st.date_input("Start date", value=pd.Timestamp("2021-01-01"))
    end_date = st.date_input("End date", value=pd.Timestamp("2024-12-31"))
    model_type = st.selectbox(
        "Model",
        ["advanced_ensemble", "quantum_inspired", "ridge", "random_forest", "histgb", "mlp"],
        index=0,
    )
    run_btn = st.button("Run analysis", type="primary")

st.markdown(
    """
    <style>
    .metric-card {background-color: #111827; padding: 14px; border-radius: 12px; color: white;}
    </style>
    """,
    unsafe_allow_html=True,
)

if run_btn:
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    if len(tickers) < 2:
        st.error("Please provide at least 2 tickers.")
        st.stop()

    with st.spinner("Loading data and running model..."):
        try:
            feat = service.prepare_data(tickers, str(start_date), str(end_date))
            ranking = service.rank_tickers(
                feat,
                model_type=model_type,
                model_kwargs={"prefer_gpu": True, "prefer_numba": True},
            )
            report = service.build_market_report(ranking)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Tickers ranked", len(report))
    c2.metric("Top ticker", report.iloc[0]["ticker"])
    c3.metric("Top score", f"{report.iloc[0]['model_score']:.3f}")

    st.subheader("Latest ranking")
    st.dataframe(report, use_container_width=True)

    st.subheader("Score distribution")
    fig = px.bar(report, x="ticker", y="model_score", color="expected_direction", text="rank")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sentiment overview")
    sent_counts = report["review_sentiment"].value_counts().reset_index()
    sent_counts.columns = ["sentiment", "count"]
    fig2 = px.pie(sent_counts, values="count", names="sentiment", hole=0.5)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Set parameters and click **Run analysis**.")
