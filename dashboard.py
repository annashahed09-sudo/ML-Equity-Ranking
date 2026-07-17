"""Interactive secured Streamlit dashboard for ML Equity Intelligence."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.market_intelligence import MarketIntelligenceService
from src.reporting import generate_pdf_report
from src.security import get_security_settings, token_is_valid
from src.sp500 import run_sp500_simulation

st.set_page_config(page_title="ML Equity Intelligence", page_icon="📈", layout="wide")

settings = get_security_settings()

st.title("📈 ML Equity Intelligence Portal")
st.caption("Secured cross-sectional ranking, S&P 500 simulation, and review sentiment intelligence")

if settings.using_default_dashboard_password:
    st.warning(
        "Using the default development password. Set ML_EQUITY_DASHBOARD_PASSWORD before deployment."
    )

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.subheader("Secure portal login")
    password = st.text_input("Portal password", type="password")
    if st.button("Unlock portal", type="primary"):
        if token_is_valid(password, settings.dashboard_password):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid portal password.")
    st.stop()

service = MarketIntelligenceService()

with st.sidebar:
    st.header("Configuration")
    mode = st.radio("Workflow", ["Custom ticker ranking", "S&P 500 simulation"], index=1)
    model_type = st.selectbox(
        "Model",
        ["advanced_ensemble", "quantum_inspired", "ridge", "random_forest", "histgb", "mlp"],
        index=0,
    )
    start_date = st.date_input("Start date", value=pd.Timestamp("2021-01-01"))
    end_date = st.date_input("End date", value=pd.Timestamp("2024-12-31"))

    if mode == "Custom ticker ranking":
        tickers_str = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,NVDA,AMZN,META")
    else:
        sp500_limit = st.slider("S&P 500 universe size", 5, 100, 25, 5)
        use_yahoo_screener = st.checkbox("Use Yahoo Finance screener universe", value=True)
        include_news = st.checkbox("Include NYT/Economist geopolitics evidence", value=True)

    run_btn = st.button("Run secured analysis", type="primary")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    div[data-testid="metric-container"] {background: #0f172a; border: 1px solid #1e293b; padding: 1rem; border-radius: 0.9rem;}
    div[data-testid="metric-container"] label, div[data-testid="metric-container"] div {color: #f8fafc;}
    </style>
    """,
    unsafe_allow_html=True,
)

if run_btn:
    with st.spinner("Loading data, training model, and running simulation..."):
        try:
            if mode == "Custom ticker ranking":
                tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
                if len(tickers) < 2:
                    st.error("Please provide at least 2 tickers.")
                    st.stop()
                feat = service.prepare_data(tickers, str(start_date), str(end_date))
                ranking = service.rank_tickers(
                    feat,
                    model_type=model_type,
                    model_kwargs={"prefer_gpu": True, "prefer_numba": True},
                )
                report = service.build_market_report(ranking)
                portfolio_summary = None
                fold_metrics = None
                current_simulation = None
            else:
                simulation = run_sp500_simulation(
                    start_date=str(start_date),
                    end_date=str(end_date),
                    model_type=model_type,
                    limit=sp500_limit,
                    use_yahoo_screener=use_yahoo_screener,
                    include_news=include_news,
                )
                ranking = simulation.ranking
                report = simulation.report
                portfolio_summary = simulation.portfolio_summary
                fold_metrics = simulation.fold_metrics
                current_simulation = simulation
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickers ranked", len(report))
    c2.metric("Top ticker", report.iloc[0]["ticker"])
    c3.metric("Top score", f"{report.iloc[0]['model_score']:.3f}")
    if portfolio_summary:
        c4.metric("Net Sharpe", f"{portfolio_summary['sharpe_net']:.3f}")
    else:
        c4.metric("Workflow", "Ranking")

    tab_rank, tab_charts, tab_backtest = st.tabs(
        ["Ranking report", "Charts", "Backtest / simulation"]
    )

    with tab_rank:
        st.subheader("Latest ranking and market-intelligence report")
        st.dataframe(report, use_container_width=True)

    with tab_charts:
        st.subheader("Model score distribution")
        fig = px.bar(report, x="ticker", y="model_score", color="expected_direction", text="rank")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Sentiment overview")
        sent_counts = report["review_sentiment"].value_counts().reset_index()
        sent_counts.columns = ["sentiment", "count"]
        fig2 = px.pie(sent_counts, values="count", names="sentiment", hole=0.5)
        st.plotly_chart(fig2, use_container_width=True)

    with tab_backtest:
        if portfolio_summary and fold_metrics is not None:
            st.subheader("Walk-forward fold metrics")
            st.dataframe(fold_metrics, use_container_width=True)
            st.subheader("Portfolio summary")
            st.json(portfolio_summary)
            if current_simulation is not None:
                report_path = Path(tempfile.gettempdir()) / "ml_equity_sp500_report.pdf"
                generate_pdf_report(
                    current_simulation, report_path, current_simulation.news_evidence or []
                )
                st.download_button(
                    "Download PDF report",
                    data=report_path.read_bytes(),
                    file_name="ml_equity_sp500_report.pdf",
                    mime="application/pdf",
                )
                if current_simulation.evidence_narrative:
                    st.subheader("News and geopolitics evidence")
                    st.text(current_simulation.evidence_narrative)
        else:
            st.info("Backtest summary is available in S&P 500 simulation mode.")
else:
    st.info("Choose a workflow and click **Run secured analysis**.")
