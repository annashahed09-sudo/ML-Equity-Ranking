"""Run tests and a small end-to-end demo pipeline."""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from src.pipeline import run_walk_forward_pipeline


def _make_synthetic_market_data(n_days: int = 180, tickers: int = 8) -> pd.DataFrame:
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
        model_type="quantum_inspired",
        n_splits=4,
        test_size=160,
        min_train_size=320,
        model_kwargs={"n_components": 128, "alpha": 0.5, "prefer_gpu": True},
    )

    print("\n=== Fold metrics ===")
    print(result.fold_metrics)
    print("\n=== Portfolio summary ===")
    print(result.portfolio_summary)

    if test_rc != 0:
        raise SystemExit(1)
