"""
Tests for research pipeline and feature engineering modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.pipeline import ResearchPipeline, PipelineResult
from research.features import compute_forward_returns, compute_features


@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    """Generate sample OHLCV market data with trending prices."""
    rng = np.random.default_rng(42)
    n_days = 250
    n_tickers = 10
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    rows = []
    for ticker in [f"T{i:02d}" for i in range(n_tickers)]:
        price = 100 + np.cumsum(rng.normal(0, 1.5, n_days))
        price = np.maximum(price, 10)
        for i, d in enumerate(dates):
            rows.append({
                "date": d,
                "ticker": ticker,
                "open": float(price[i]),
                "high": float(price[i] * (1 + abs(rng.normal(0, 0.01)))),
                "low": float(price[i] * (1 - abs(rng.normal(0, 0.01)))),
                "close": float(price[i]),
                "volume": int(rng.integers(1_000_000, 10_000_000)),
            })
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


class TestResearchPipeline:
    """Research pipeline tests."""

    def test_pipeline_initializes(self):
        pipeline = ResearchPipeline()
        assert pipeline is not None

    def test_pipeline_with_ridge(self, sample_market_data):
        """Test pipeline with ridge model and minimal factors."""
        pipeline = ResearchPipeline()
        result = pipeline.run(
            raw_df=sample_market_data,
            model_type="ridge",
            n_splits=2,
            test_size=30,
            min_train_size=60,
        )
        assert isinstance(result, PipelineResult)
        assert result.predictions is not None
        assert result.fold_metrics is not None

    def test_pipeline_result_structure(self, sample_market_data):
        """Verify pipeline result has expected columns."""
        pipeline = ResearchPipeline()
        result = pipeline.run(
            raw_df=sample_market_data,
            model_type="ridge",
            n_splits=2,
            test_size=30,
            min_train_size=60,
        )
        assert "model_score" in result.predictions.columns
        assert result.duration > 0


class TestFeatures:
    """Feature engineering tests."""

    def test_compute_forward_returns(self, sample_market_data):
        df = compute_forward_returns(sample_market_data)
        assert "forward_return" in df.columns
        assert not df["forward_return"].isna().all()

    def test_compute_features(self, sample_market_data):
        df = compute_features(sample_market_data)
        assert len(df.columns) > len(sample_market_data.columns)
