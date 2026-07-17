"""Interactive secured Streamlit dashboard for ML Equity Intelligence.

Dark "IBM Plex" financial-analysis theme with cross-sectional rankings,
walk-forward diagnostics, and long/short portfolio performance.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard_theme import (
    THEME_CSS,
    compute_portfolio_kpis,
    cumulative_pnl_figure,
    drawdown_figure,
    ic_by_fold_figure,
    long_short_buckets,
    score_figure,
    sentiment_figure,
    turnover_figure,
)
from src.market_intelligence import MarketIntelligenceService
from src.reporting import generate_pdf_report
from src.sp500 import run_sp500_simulation

st.set_page_config(page_title="ML Equity Intelligence", page_icon="📈", layout="wide")
st.markdown(THEME_CSS, unsafe_allow_html=True)

PRIVACY_NOTICE = (
    "**Privacy — we collect no data.** This app runs entirely on your machine. "
    "There are no accounts, no logins, and no tracking or analytics. Market data "
    "is fetched from public sources (yfinance) only when you run an analysis, and "
    "nothing you enter is stored, transmitted to us, or shared with any third party. "
    "Streamlit usage statistics are disabled."
)


def section(num: str, title: str, sub: str = "") -> None:
    """Render a numbered section header."""
    st.markdown(
        f'<div class="mlq-section"><span class="num">{num}</span>'
        f'<span class="title">{title}</span>'
        f'<span class="sub">{sub}</span></div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "", tone: str = "neutral") -> str:
    """Return HTML for a single KPI card."""
    return (
        f'<div class="mlq-kpi"><div class="k-label">{label}</div>'
        f'<div class="k-value k-{tone}">{value}</div>'
        f'<div class="k-sub">{sub}</div></div>'
    )


def kpi_row(cards: list[str]) -> None:
    """Render a row of KPI cards."""
    cols = st.columns(len(cards))
    for col, html in zip(cols, cards):
        col.markdown(html, unsafe_allow_html=True)


service = MarketIntelligenceService()

# --- Sidebar configuration --------------------------------------------------
with st.sidebar:
    st.markdown("### Configuration")
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

    run_btn = st.button("Run analysis", type="primary")
    st.caption("Signals are research estimates, not trading advice.")
    with st.expander("Privacy"):
        st.markdown(PRIVACY_NOTICE)


# --- Hero -------------------------------------------------------------------
st.markdown(
    '<div class="mlq-hero"><div class="eyebrow">Cross-sectional alpha · walk-forward · private &amp; local</div>'
    "<h1>ML Equity Intelligence Portal</h1>"
    "<p>Rank a universe by relative strength, validate signal quality out-of-sample, "
    "and simulate a market-neutral long/short book net of costs. Runs entirely on "
    "your machine — no login, no data collection.</p></div>",
    unsafe_allow_html=True,
)

if not run_btn:
    st.info("Configure a workflow in the sidebar and click **Run analysis**.")
    st.markdown(PRIVACY_NOTICE)
    st.stop()

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
            portfolio_returns = None
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
            portfolio_returns = simulation.portfolio_returns
            fold_metrics = simulation.fold_metrics
            current_simulation = simulation
    except Exception as exc:  # noqa: BLE001 - surface any failure to the operator
        st.error(f"Analysis failed: {exc}")
        st.stop()

is_portfolio = portfolio_summary is not None and portfolio_returns is not None

# --- 01 Overview ------------------------------------------------------------
span = f"{start_date} → {end_date}"
section("01", "Overview", f"{model_type} · {len(report)} names · {span}")

top = report.iloc[0]
if is_portfolio:
    kpis = compute_portfolio_kpis(portfolio_returns, portfolio_summary, fold_metrics)
    kpi_row(
        [
            kpi_card(
                "Net Sharpe",
                f"{kpis['sharpe_net']:.2f}",
                f"gross {kpis['sharpe_gross']:.2f}",
                "pos" if kpis["sharpe_net"] >= 0 else "neg",
            ),
            kpi_card(
                "Cumulative net",
                f"{kpis['cumulative_net'] * 100:.1f}%",
                f"gross {kpis['cumulative_gross'] * 100:.1f}%",
                "pos" if kpis["cumulative_net"] >= 0 else "neg",
            ),
            kpi_card(
                "Max drawdown",
                f"{kpis['max_drawdown'] * 100:.1f}%",
                "net equity",
                "neg",
            ),
            kpi_card(
                "Mean IC",
                f"{kpis['mean_ic']:.3f}",
                f"IR {kpis['ic_ir']:.2f}" if kpis["ic_ir"] == kpis["ic_ir"] else "IR n/a",
                "pos" if (kpis["mean_ic"] == kpis["mean_ic"] and kpis["mean_ic"] >= 0) else "neg",
            ),
        ]
    )
    kpi_row(
        [
            kpi_card("Hit rate", f"{kpis['hit_rate'] * 100:.1f}%", "days net > 0"),
            kpi_card("Mean turnover", f"{kpis['mean_turnover'] * 100:.1f}%", "per rebalance"),
            kpi_card("Cost drag", f"{kpis['cost_drag'] * 100:.2f}%", "gross − net"),
            kpi_card("Top signal", str(top["ticker"]), f"score {top['model_score']:.3f}", "pos"),
        ]
    )
else:
    kpi_row(
        [
            kpi_card("Names ranked", str(len(report)), "custom universe"),
            kpi_card("Top signal", str(top["ticker"]), f"score {top['model_score']:.3f}", "pos"),
            kpi_card(
                "Bottom signal",
                str(report.iloc[-1]["ticker"]),
                f"score {report.iloc[-1]['model_score']:.3f}",
                "neg",
            ),
            kpi_card("Model", model_type, "ranking mode"),
        ]
    )

# --- 02 Rankings ------------------------------------------------------------
section("02", "Cross-sectional rankings", "relative strength screener")

longs, shorts = long_short_buckets(ranking)
bcol1, bcol2 = st.columns(2)
bcol1.markdown(
    "**Long book (top quantile)**<br>"
    + "".join(f'<span class="mlq-chip long">{t}</span>' for t in longs),
    unsafe_allow_html=True,
)
bcol2.markdown(
    "**Short book (bottom quantile)**<br>"
    + "".join(f'<span class="mlq-chip short">{t}</span>' for t in shorts),
    unsafe_allow_html=True,
)

screener_cols = [
    c
    for c in [
        "rank",
        "ticker",
        "model_score",
        "expected_direction",
        "review_sentiment",
        "review_sentiment_score",
        "review_confidence",
    ]
    if c in report.columns
]
st.dataframe(
    report[screener_cols],
    use_container_width=True,
    hide_index=True,
    column_config={
        "rank": st.column_config.NumberColumn("Rank", format="%d"),
        "ticker": st.column_config.TextColumn("Ticker"),
        "model_score": st.column_config.NumberColumn("Score", format="%.3f"),
        "expected_direction": st.column_config.TextColumn("Direction"),
        "review_sentiment": st.column_config.TextColumn("Sentiment"),
        "review_sentiment_score": st.column_config.NumberColumn("Sent. score", format="%.2f"),
        "review_confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
    },
)

# --- 03 Signal & performance -----------------------------------------------
section("03", "Signal & performance", "score distribution, IC stability, PnL")

scol1, scol2 = st.columns([3, 2])
scol1.markdown("**Model score by name**")
scol1.plotly_chart(score_figure(report), use_container_width=True)
scol2.markdown("**Review sentiment mix**")
scol2.plotly_chart(sentiment_figure(report), use_container_width=True)

if is_portfolio:
    st.markdown("**Cumulative PnL — gross vs net of costs**")
    st.plotly_chart(cumulative_pnl_figure(portfolio_returns), use_container_width=True)

    pcol1, pcol2 = st.columns(2)
    pcol1.markdown("**Drawdown (net equity)**")
    pcol1.plotly_chart(drawdown_figure(portfolio_returns), use_container_width=True)
    pcol2.markdown("**Turnover per rebalance**")
    pcol2.plotly_chart(turnover_figure(portfolio_returns), use_container_width=True)

    if fold_metrics is not None and len(fold_metrics):
        st.markdown("**Information Coefficient by walk-forward fold**")
        st.plotly_chart(ic_by_fold_figure(fold_metrics), use_container_width=True)

# --- 04 Diagnostics & reporting --------------------------------------------
if is_portfolio:
    section("04", "Diagnostics & reporting", "fold detail, health, export")

    with st.expander("Walk-forward fold detail", expanded=True):
        st.dataframe(fold_metrics, use_container_width=True, hide_index=True)

    hcol1, hcol2, hcol3 = st.columns(3)
    latest = str(ranking["date"].max()) if "date" in ranking.columns else "n/a"
    hcol1.markdown(
        kpi_card("Universe", str(len(current_simulation.universe)), "tradable names"),
        unsafe_allow_html=True,
    )
    hcol2.markdown(
        kpi_card("Walk-forward folds", str(len(fold_metrics)), "expanding window"),
        unsafe_allow_html=True,
    )
    hcol3.markdown(
        kpi_card("Latest signal date", latest, "data freshness"),
        unsafe_allow_html=True,
    )

    with st.expander("Portfolio summary (raw)", expanded=False):
        st.json(portfolio_summary)

    if current_simulation is not None:
        report_path = Path(tempfile.gettempdir()) / "ml_equity_sp500_report.pdf"
        generate_pdf_report(current_simulation, report_path, current_simulation.news_evidence or [])
        st.download_button(
            "Download PDF report",
            data=report_path.read_bytes(),
            file_name="ml_equity_sp500_report.pdf",
            mime="application/pdf",
        )
        if current_simulation.evidence_narrative:
            st.markdown("**News & geopolitics evidence**")
            st.text(current_simulation.evidence_narrative)
