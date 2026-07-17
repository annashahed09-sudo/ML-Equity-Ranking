"""Tests for dashboard KPI computation and figure builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.dashboard_theme import (
    compute_portfolio_kpis,
    cumulative_pnl_figure,
    drawdown_figure,
    ic_by_fold_figure,
    long_short_buckets,
    score_figure,
    sentiment_figure,
    turnover_figure,
)


def _returns() -> pd.DataFrame:
    dates = pd.date_range("2022-01-03", periods=6, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "gross_return": [0.01, -0.02, 0.03, 0.00, -0.01, 0.02],
            "net_return": [0.008, -0.021, 0.028, -0.001, -0.012, 0.018],
            "turnover": [0.5, 0.3, 0.4, 0.2, 0.35, 0.25],
        }
    )


def _summary() -> dict:
    return {
        "mean_gross_return": 0.005,
        "mean_net_return": 0.003,
        "std_gross_return": 0.02,
        "std_net_return": 0.02,
        "sharpe_gross": 0.25,
        "sharpe_net": 0.15,
        "cumulative_gross": 0.03,
        "cumulative_net": 0.02,
        "mean_turnover": 0.33,
    }


def _folds() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": [0, 1, 2],
            "test_start": pd.to_datetime(["2022-02-01", "2022-03-01", "2022-04-01"]),
            "test_end": pd.to_datetime(["2022-02-28", "2022-03-31", "2022-04-30"]),
            "mean_ic": [0.05, -0.02, 0.03],
            "std_ic": [0.1, 0.12, 0.11],
            "n_test_rows": [100, 100, 100],
        }
    )


def _report() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rank": [1, 2, 3, 4, 5],
            "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE"],
            "model_score": [1.2, 0.4, 0.0, -0.3, -0.9],
            "expected_direction": ["up", "up", "up", "down", "down"],
            "review_sentiment": ["positive", "neutral", "neutral", "negative", "neutral"],
        }
    )


def test_compute_portfolio_kpis_values() -> None:
    kpis = compute_portfolio_kpis(_returns(), _summary(), _folds())
    assert kpis["n_periods"] == 6
    # cost drag is gross minus net cumulative.
    assert kpis["cost_drag"] == 0.03 - 0.02
    # max drawdown is non-positive.
    assert kpis["max_drawdown"] <= 0.0
    # hit rate is the fraction of positive net days (3 of 6).
    assert kpis["hit_rate"] == 0.5
    # mean IC matches the fold mean.
    assert np.isclose(kpis["mean_ic"], np.mean([0.05, -0.02, 0.03]))


def test_compute_portfolio_kpis_handles_missing_folds() -> None:
    kpis = compute_portfolio_kpis(_returns(), _summary(), None)
    assert np.isnan(kpis["mean_ic"])
    assert kpis["n_periods"] == 6


def test_long_short_buckets_disjoint() -> None:
    ranking = _report().rename_axis(None)
    longs, shorts = long_short_buckets(ranking, quantile=0.2)
    assert longs[0] == "AAA"
    assert shorts[-1] == "EEE"
    assert not set(longs) & set(shorts)


def test_figure_builders_return_figures() -> None:
    rets = _returns()
    assert isinstance(cumulative_pnl_figure(rets), go.Figure)
    assert isinstance(drawdown_figure(rets), go.Figure)
    assert isinstance(turnover_figure(rets), go.Figure)
    assert isinstance(ic_by_fold_figure(_folds()), go.Figure)
    assert isinstance(score_figure(_report()), go.Figure)
    assert isinstance(sentiment_figure(_report()), go.Figure)


def test_cumulative_pnl_has_two_traces() -> None:
    fig = cumulative_pnl_figure(_returns())
    assert len(fig.data) == 2
