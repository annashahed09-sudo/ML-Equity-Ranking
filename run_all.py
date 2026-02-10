"""Run tests and an advanced end-to-end demo pipeline."""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from src.pipeline import run_walk_forward_pipeline, run_model_suite
from src.market_intelligence import MarketIntelligenceService


def _make_synthetic_market_data(n_days: int = 220, tickers: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    symbols = [f"T{i:02d}" for i in range(tickers)]

    rows = []
    for symbol in symbols:
        base = 100 + np.cumsum(rng.normal(0, 1, n_days))
        high = base * (1 + np.abs(rng.normal(0, 0.01, n_days)))
        low = base * (1 - np.abs(rng.normal(0, 0.01, n_days)))
        open_ = base * (1 + rng.normal(0, 0.003, n_days))
        volume = rng.integers(1_000_000, 5_000_000, size=n_days)
        for i, d in enumerate(dates):
            rows.append(
                {
                    "date": d,
                    "ticker": symbol,
                    "open": float(open_[i]),
                    "high": float(high[i]),
                    "low": float(low[i]),
                    "close": float(base[i]),
                    "volume": int(volume[i]),
                }
            )

    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


if __name__ == "__main__":
    test_rc = os.system("pytest tests/ --maxfail=1 --disable-warnings -q")

    demo_df = _make_synthetic_market_data()
    result = run_walk_forward_pipeline(
        demo_df,
        model_type="advanced_ensemble",
        n_splits=4,
        test_size=200,
        min_train_size=400,
        model_kwargs={"prefer_gpu": True, "prefer_numba": True},
    )

    print("\n=== Advanced Ensemble Fold metrics ===")
    print(result.fold_metrics)
    print("\n=== Portfolio summary ===")
    print(result.portfolio_summary)

    model_board = run_model_suite(
        demo_df,
        model_types=["ridge", "quantum_inspired", "random_forest", "histgb", "advanced_ensemble"],
        n_splits=3,
        test_size=180,
        min_train_size=360,
    )
    print("\n=== Model leaderboard ===")
    print(model_board)

    service = MarketIntelligenceService()
    latest_ranking = result.predictions.sort_values(["date", "model_score"], ascending=[False, False]).groupby("ticker").head(1)
    latest_ranking = latest_ranking.sort_values("model_score", ascending=False).reset_index(drop=True)
    latest_ranking["rank"] = latest_ranking.index + 1

    synthetic_reviews = {
        latest_ranking.iloc[0]["ticker"]: [
            "Strong earnings beat and growth outlook improves.",
            "Analysts remain bullish with upside revisions.",
        ]
    }
    report = service.build_market_report(latest_ranking[["ticker", "rank", "model_score"]], synthetic_reviews)
    print("\n=== Market review report ===")
    print(report.head())

    if test_rc != 0:
        raise SystemExit(1)
