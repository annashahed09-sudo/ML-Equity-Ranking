import numpy as np
import pandas as pd

from src.features import compute_features, compute_forward_returns
from src.sp500 import get_sp500_universe, run_sp500_simulation


def _prepared_data(n_days=140, n_tickers=6):
    rng = np.random.default_rng(123)
    dates = pd.date_range("2022-01-01", periods=n_days)
    rows = []
    for i in range(n_tickers):
        ticker = f"SP{i:03d}"
        close = 100 + np.cumsum(rng.normal(0, 1, n_days))
        for j, d in enumerate(dates):
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "open": close[j] * 0.995,
                    "high": close[j] * 1.01,
                    "low": close[j] * 0.99,
                    "close": close[j],
                    "volume": 1_000_000 + j,
                }
            )
    raw = pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)
    featured = compute_features(raw)
    return compute_forward_returns(featured).dropna().reset_index(drop=True)


def test_get_sp500_universe_offline_limit():
    tickers = get_sp500_universe(limit=5, use_yahoo_screener=False)
    assert len(tickers) == 5
    assert "AAPL" in tickers


def test_run_sp500_simulation_with_prepared_data():
    result = run_sp500_simulation(
        start_date="2022-01-01",
        end_date="2022-06-01",
        model_type="ridge",
        limit=6,
        n_splits=2,
        test_size=120,
        min_train_size=240,
        prepared_data=_prepared_data(),
    )
    # The simulation runs successfully with prepared data
    assert len(result.universe) >= 2
