"""S&P 500 universe helpers and simulation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .data_loader import load_yfinance_data
from .features import compute_features, compute_forward_returns
from .market_intelligence import MarketIntelligenceService
from .pipeline import PipelineResult, run_walk_forward_pipeline


# Stable fallback sample for offline tests/demos. The live universe can be loaded from Wikipedia.
FALLBACK_SP500_SAMPLE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK-B", "LLY", "AVGO",
    "JPM", "TSLA", "UNH", "XOM", "V", "MA", "PG", "COST", "HD", "MRK",
]


@dataclass
class SP500SimulationResult:
    """Container for an S&P 500 ranking/backtest simulation."""

    universe: List[str]
    ranking: pd.DataFrame
    report: pd.DataFrame
    fold_metrics: pd.DataFrame
    portfolio_returns: pd.DataFrame
    portfolio_summary: Dict[str, float]

    def to_dict(self) -> Dict:
        """JSON-serializable representation for API responses."""
        return {
            "universe": self.universe,
            "ranking": self.ranking.to_dict(orient="records"),
            "report": self.report.to_dict(orient="records"),
            "fold_metrics": self.fold_metrics.to_dict(orient="records"),
            "portfolio_returns": self.portfolio_returns.to_dict(orient="records"),
            "portfolio_summary": {k: float(v) for k, v in self.portfolio_summary.items()},
        }


def get_sp500_universe(limit: Optional[int] = None, use_live_wikipedia: bool = True) -> List[str]:
    """
    Return S&P 500 symbols.

    When `use_live_wikipedia` is enabled, this attempts to load current constituents from
    Wikipedia. If network/data parsing is unavailable, it falls back to a stable sample.
    """
    tickers = FALLBACK_SP500_SAMPLE.copy()
    if use_live_wikipedia:
        try:
            tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
            live = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
            if live:
                tickers = live
        except (ImportError, ValueError, OSError, KeyError, IndexError):
            tickers = FALLBACK_SP500_SAMPLE.copy()

    if limit is not None:
        tickers = tickers[: max(1, int(limit))]
    return tickers


def prepare_sp500_dataset(
    start_date: str,
    end_date: str,
    limit: int = 25,
    use_live_wikipedia: bool = True,
) -> pd.DataFrame:
    """Load and feature-engineer S&P 500 market data."""
    tickers = get_sp500_universe(limit=limit, use_live_wikipedia=use_live_wikipedia)
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
    use_live_wikipedia: bool = True,
    prepared_data: Optional[pd.DataFrame] = None,
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
            use_live_wikipedia=use_live_wikipedia,
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

    return SP500SimulationResult(
        universe=sorted(dataset["ticker"].unique().tolist()),
        ranking=ranking,
        report=report,
        fold_metrics=pipeline_result.fold_metrics,
        portfolio_returns=pipeline_result.portfolio_returns,
        portfolio_summary=pipeline_result.portfolio_summary,
    )
