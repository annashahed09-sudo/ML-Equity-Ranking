"""S&P 500 universe helpers and simulation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from .data_loader import load_yfinance_data
from .features import compute_features, compute_forward_returns
from .market_intelligence import MarketIntelligenceService
from .news import (
    NewsEvidence,
    build_evidence_narrative,
    collect_market_news,
    filter_geopolitical_evidence,
)
from .pipeline import PipelineResult, run_walk_forward_pipeline

# Stable fallback sample for offline tests/demos. Live symbols are loaded from Yahoo Finance screeners.
FALLBACK_SP500_SAMPLE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "GOOG",
    "BRK-B",
    "LLY",
    "AVGO",
    "JPM",
    "TSLA",
    "UNH",
    "XOM",
    "V",
    "MA",
    "PG",
    "COST",
    "HD",
    "MRK",
]


def _json_records(df: pd.DataFrame) -> list[dict]:
    serializable = df.copy()
    for col in serializable.columns:
        if pd.api.types.is_datetime64_any_dtype(serializable[col]):
            serializable[col] = serializable[col].astype(str)
    return serializable.to_dict(orient="records")


@dataclass
class SP500SimulationResult:
    """Container for an S&P 500 ranking/backtest simulation."""

    universe: List[str]
    ranking: pd.DataFrame
    report: pd.DataFrame
    fold_metrics: pd.DataFrame
    portfolio_returns: pd.DataFrame
    portfolio_summary: Dict[str, float]
    news_evidence: List[NewsEvidence] | None = None
    evidence_narrative: str = ""

    def to_dict(self) -> Dict:
        """JSON-serializable representation for API responses."""
        return {
            "universe": self.universe,
            "ranking": _json_records(self.ranking),
            "report": _json_records(self.report),
            "fold_metrics": _json_records(self.fold_metrics),
            "portfolio_returns": _json_records(self.portfolio_returns),
            "portfolio_summary": {k: float(v) for k, v in self.portfolio_summary.items()},
            "news_evidence": [item.__dict__ for item in (self.news_evidence or [])],
            "evidence_narrative": self.evidence_narrative,
        }


def get_sp500_universe(limit: Optional[int] = None, use_yahoo_screener: bool = True) -> List[str]:
    """
    Return an S&P 500 simulation universe using Yahoo Finance data sources.

    Yahoo Finance does not expose a simple official free S&P 500 constituents endpoint in
    `yfinance`, so this uses Yahoo Finance screener lists to build a large-cap proxy universe
    and falls back to a stable S&P 500 sample for offline/reproducible demos.
    """
    tickers = FALLBACK_SP500_SAMPLE.copy()
    if use_yahoo_screener:
        yahoo_queries = ["portfolio_anchors", "undervalued_large_caps", "most_actives"]
        collected: list[str] = []
        for query in yahoo_queries:
            try:
                response = yf.screen(query, size=max(limit or 25, 25))
            except (OSError, ValueError, KeyError):
                response = {}
            for quote in response.get("quotes", []):
                symbol = str(quote.get("symbol", "")).replace(".", "-").upper()
                quote_type = quote.get("quoteType")
                if symbol and quote_type in {None, "EQUITY"} and symbol not in collected:
                    collected.append(symbol)
        if collected:
            tickers = collected

    if limit is not None:
        tickers = tickers[: max(1, int(limit))]
    return tickers


def prepare_sp500_dataset(
    start_date: str,
    end_date: str,
    limit: int = 25,
    use_yahoo_screener: bool = True,
) -> pd.DataFrame:
    """Load and feature-engineer S&P 500 market data."""
    tickers = get_sp500_universe(limit=limit, use_yahoo_screener=use_yahoo_screener)
    raw = load_yfinance_data(tickers=tickers, start_date=start_date, end_date=end_date)
    featured = compute_features(raw)
    return compute_forward_returns(featured).dropna().reset_index(drop=True)


def run_sp500_simulation(
    start_date: str,
    end_date: str,
    model_type: str = "advanced_ensemble",
    limit: int = 25,
    n_splits: int = 3,
    test_size: Optional[int] = None,
    min_train_size: Optional[int] = None,
    reviews_by_ticker: Optional[Dict[str, List[str]]] = None,
    use_yahoo_screener: bool = True,
    prepared_data: Optional[pd.DataFrame] = None,
    include_news: bool = False,
) -> SP500SimulationResult:
    """
    Run a practical S&P 500 ranking and walk-forward portfolio simulation.

    `prepared_data` is accepted for tests/offline demos and must already include features
    and `forward_return` if provided.
    """
    if prepared_data is None:
        dataset = prepare_sp500_dataset(
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            use_yahoo_screener=use_yahoo_screener,
        )
    else:
        dataset = prepared_data.copy()

    if dataset["ticker"].nunique() < 2:
        raise ValueError("S&P 500 simulation requires at least two tickers after data loading.")

    pipeline_result: PipelineResult = run_walk_forward_pipeline(
        dataset,
        model_type=model_type,
        n_splits=n_splits,
        test_size=test_size,
        min_train_size=min_train_size,
        model_kwargs={"prefer_gpu": True, "prefer_numba": True},
        already_prepared=True,
    )

    latest_date = pipeline_result.predictions["date"].max()
    ranking = pipeline_result.predictions[pipeline_result.predictions["date"] == latest_date].copy()
    ranking = ranking.sort_values("model_score", ascending=False).reset_index(drop=True)
    ranking["rank"] = ranking.index + 1

    service = MarketIntelligenceService()
    report = service.build_market_report(ranking, reviews_by_ticker)
    news_evidence = (
        filter_geopolitical_evidence(collect_market_news(), limit=10) if include_news else []
    )
    evidence_narrative = build_evidence_narrative(news_evidence)

    return SP500SimulationResult(
        universe=sorted(dataset["ticker"].unique().tolist()),
        ranking=ranking,
        report=report,
        fold_metrics=pipeline_result.fold_metrics,
        portfolio_returns=pipeline_result.portfolio_returns,
        portfolio_summary=pipeline_result.portfolio_summary,
        news_evidence=news_evidence,
        evidence_narrative=evidence_narrative,
    )
