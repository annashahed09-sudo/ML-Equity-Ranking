"""
Research pipeline runner.

Provides entry points for running the research pipeline with synthetic data
or loading real market data.

Usage:
    python -m research.run --mode demo
    python -m research.run --mode sp500 --start 2020-01-01 --end 2024-01-01
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from research.pipeline import ResearchPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("research.run")


def _make_synthetic_market_data(n_days: int = 500, n_tickers: int = 20) -> pd.DataFrame:
    """Generate synthetic OHLCV market data for testing."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")
    symbols = [f"ASSET{i:02d}" for i in range(n_tickers)]

    rows = []
    for symbol in symbols:
        base = 100 + np.cumsum(rng.normal(0, 1.5, n_days))
        base = np.maximum(base, 10)
        high = base * (1 + np.abs(rng.normal(0, 0.015, n_days)))
        low = base * (1 - np.abs(rng.normal(0, 0.015, n_days)))
        open_ = low + rng.uniform(0, 1, n_days) * (high - low)
        volume = rng.integers(500_000, 5_000_000, size=n_days)
        for i, d in enumerate(dates):
            rows.append({
                "date": d,
                "ticker": symbol,
                "open": float(open_[i]),
                "high": float(high[i]),
                "low": float(low[i]),
                "close": float(base[i]),
                "volume": int(volume[i]),
            })

    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def run_demo() -> None:
    """Run the research pipeline with synthetic data."""
    logger.info("Generating synthetic market data...")
    df = _make_synthetic_market_data()

    pipeline = ResearchPipeline()
    logger.info("Running research pipeline...")
    result = pipeline.run(
        raw_df=df,
        model_type="ridge",
        n_splits=3,
        test_size=100,
        min_train_size=200,
    )

    print("\n" + "=" * 60)
    print("RESEARCH PIPELINE RESULTS")
    print("=" * 60)

    print(f"\nDuration: {result.duration:.2f}s")
    print(f"IC series: {result.ic_series.mean():.4f} mean, {result.ic_series.std():.4f} std")

    print(f"\nFold Metrics:")
    print(result.fold_metrics.to_string(index=False))

    print(f"\nPortfolio Summary:")
    for key, value in result.portfolio_summary.items():
        print(f"  {key}: {value:.6f}")

    print("\nPipeline completed successfully.")


def run_sp500(start_date: str, end_date: str) -> None:
    """Run the research pipeline with S&P 500 data."""
    from data.loader import DataLoader

    loader = DataLoader()
    logger.info(f"Loading S&P 500 data from {start_date} to {end_date}...")
    tickers = loader.get_sp500_universe(limit=50)
    df = loader.load_tickers(tickers, start_date, end_date)

    pipeline = ResearchPipeline()
    logger.info("Running research pipeline...")
    result = pipeline.run(
        raw_df=df,
        model_type="ridge",
        n_splits=3,
    )

    print(f"\nS&P 500 Pipeline Results:")
    print(f"Universe: {df['ticker'].nunique()} tickers")
    print(f"Data range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Mean IC: {result.ic_series.mean():.4f}")
    for key, value in result.portfolio_summary.items():
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research Pipeline Runner")
    parser.add_argument("--mode", choices=["demo", "sp500"], default="demo")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-01-01")

    args = parser.parse_args()

    if args.mode == "demo":
        run_demo()
    elif args.mode == "sp500":
        run_sp500(args.start, args.end)

    sys.exit(0)
