"""Leakage guardrail tests.

These tests protect the two invariants that make the cross-sectional pipeline
trustworthy:

1. Targets (``forward_return``) are aligned to the correct (date, ticker) row.
2. Walk-forward folds never share a date between train and test.
"""

import numpy as np
import pandas as pd

from src.features import compute_features, compute_forward_returns, get_feature_columns
from src.pipeline import run_walk_forward_pipeline
from src.validation import create_walk_forward_splits, get_fold_data, validate_fold_integrity


def _multi_ticker_ohlcv(n_days: int = 30) -> pd.DataFrame:
    """Two tickers with deterministic, distinct price paths.

    ``ZZZ`` sorts after ``AAA`` alphabetically but its group is processed
    differently by pandas groupby ordering, which is exactly what surfaces the
    label-misalignment bug if forward returns are assigned in group order.
    """
    dates = pd.date_range("2022-01-01", periods=n_days)
    rows = []
    for ticker, base in [("ZZZ", 100.0), ("AAA", 10.0)]:
        for i, d in enumerate(dates):
            close = base * (i + 1)
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1_000_000 + i,
                }
            )
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def test_forward_returns_aligned_per_ticker():
    """Each row's forward_return must equal log(close[t+1]/close[t]) for THAT ticker."""
    df = _multi_ticker_ohlcv()
    out = compute_forward_returns(df, forward_periods=1)

    for ticker, g in out.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        expected = np.log(g["close"].shift(-1) / g["close"])
        computed = g["forward_return"]
        mask = expected.notna()
        assert np.allclose(
            computed[mask], expected[mask]
        ), f"forward_return misaligned for {ticker}"


def test_forward_returns_no_lookahead_last_row_is_nan():
    """The final observation per ticker has no future price, so target must be NaN."""
    df = _multi_ticker_ohlcv()
    out = compute_forward_returns(df, forward_periods=1)
    last_rows = out.sort_values("date").groupby("ticker").tail(1)
    assert last_rows["forward_return"].isna().all()


def test_cross_sectional_normalization_is_per_date():
    """After normalization each feature must have ~zero cross-sectional mean per date."""
    df = _multi_ticker_ohlcv(n_days=40)
    feats = compute_features(df)
    feature_cols = get_feature_columns(feats)
    assert feature_cols
    per_date_mean = feats.groupby("date")[feature_cols].mean().abs().max().max()
    assert per_date_mean < 1e-6


def test_walk_forward_no_date_overlap():
    """No date may appear in both a fold's train and test partitions."""
    df = _multi_ticker_ohlcv(n_days=60)
    # test_size intentionally not a multiple of the ticker count to exercise snapping.
    folds = create_walk_forward_splits(df, n_splits=3, test_size=15, min_train_size=30)
    assert folds
    valid, issues = validate_fold_integrity(df, folds)
    assert valid, issues
    for fold in folds:
        train_df, test_df = get_fold_data(df, fold)
        assert set(train_df["date"]).isdisjoint(set(test_df["date"]))


def test_pipeline_accepts_already_prepared():
    """already_prepared=True must skip re-featurization and run end-to-end."""
    df = _multi_ticker_ohlcv(n_days=160)
    prepared = compute_forward_returns(compute_features(df)).dropna().reset_index(drop=True)
    result = run_walk_forward_pipeline(
        prepared,
        model_type="ridge",
        n_splits=2,
        test_size=20,
        min_train_size=40,
        already_prepared=True,
    )
    assert not result.predictions.empty
    assert "mean_net_return" in result.portfolio_summary
