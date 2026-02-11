"""High-level market intelligence service combining ML predictions and NLP reviews."""

from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd

from .data_loader import load_yfinance_data
from .features import compute_features, compute_forward_returns, get_feature_columns
from .models import create_model
from .nlp import FinancialSentimentAnalyzer


class MarketIntelligenceService:
    """
    End-to-end service for ticker ranking + qualitative review insight.

    Notes
    -----
    - Supports any ticker available via yfinance.
    - Accuracy depends on data quality, regime stability, and feature validity.
    - Returns ranked signals, not guaranteed future outcomes.
    """

    def __init__(self):
        self.sentiment = FinancialSentimentAnalyzer()

    def prepare_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        raw = load_yfinance_data(tickers=tickers, start_date=start_date, end_date=end_date)
        feat = compute_features(raw)
        feat = compute_forward_returns(feat)
        return feat.dropna().reset_index(drop=True)

    def rank_tickers(
        self,
        df: pd.DataFrame,
        model_type: str = "advanced_ensemble",
        model_kwargs: Optional[Dict] = None,
    ) -> pd.DataFrame:
        if model_kwargs is None:
            model_kwargs = {"prefer_gpu": True, "prefer_numba": True}

        feature_cols = get_feature_columns(df)
        if not feature_cols:
            raise ValueError("No engineered features found.")

        dates = sorted(df["date"].unique())
        split_idx = int(len(dates) * 0.8)
        train_dates = set(dates[:split_idx])
        test_dates = set(dates[split_idx:])

        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"].isin(test_dates)]

        model = create_model(model_type, **model_kwargs)
        model.fit(train_df[feature_cols], train_df["forward_return"])

        scored = test_df[["date", "ticker", "forward_return"]].copy()
        scored["model_score"] = model.predict(test_df[feature_cols])

        latest_date = scored["date"].max()
        latest = scored[scored["date"] == latest_date].copy()
        latest = latest.sort_values("model_score", ascending=False).reset_index(drop=True)
        latest["rank"] = latest.index + 1
        return latest

    def build_market_report(
        self,
        ranking_df: pd.DataFrame,
        reviews_by_ticker: Optional[Dict[str, List[str]]] = None,
    ) -> pd.DataFrame:
        if reviews_by_ticker is None:
            reviews_by_ticker = {}

        rows = []
        for _, r in ranking_df.iterrows():
            ticker = r["ticker"]
            reviews = reviews_by_ticker.get(ticker, [])
            insight = self.sentiment.build_insight(reviews)

            rows.append(
                {
                    "ticker": ticker,
                    "rank": int(r["rank"]),
                    "model_score": float(r["model_score"]),
                    "expected_direction": "up" if r["model_score"] > 0 else "down",
                    "review_sentiment": insight.label,
                    "review_sentiment_score": insight.sentiment_score,
                    "review_confidence": insight.confidence,
                    "review_summary": insight.summary,
                }
            )

        return pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)
