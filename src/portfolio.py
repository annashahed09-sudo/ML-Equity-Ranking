"""
Portfolio construction and backtesting for cross-sectional equity models.

Implements long-short portfolio logic using model scores, and computes cumulative returns.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List

from .evaluation import long_short_portfolio_returns, summarize_portfolio_performance


def run_portfolio_backtest(
    df: pd.DataFrame,
    score_col: str = 'model_score',
    return_col: str = 'forward_return',
    long_quantile: float = 0.2,
    short_quantile: float = 0.2,
    transaction_cost_bps: float = 10.0
) -> Tuple[pd.DataFrame, dict]:
    """
    Run a long-short portfolio backtest using model scores and forward returns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: date, ticker, model_score, forward_return
    score_col : str, default 'model_score'
        Column with model scores
    return_col : str, default 'forward_return'
        Column with next-period returns
    long_quantile : float, default 0.2
        Fraction of assets to hold long
    short_quantile : float, default 0.2
        Fraction of assets to hold short
    transaction_cost_bps : float, default 10.0
        Transaction cost per trade (basis points)

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        (returns DataFrame, summary dict)
    """
    returns_df = long_short_portfolio_returns(
        df,
        score_col=score_col,
        return_col=return_col,
        long_quantile=long_quantile,
        short_quantile=short_quantile,
        transaction_cost_bps=transaction_cost_bps
    )
    summary = summarize_portfolio_performance(returns_df)
    return returns_df, summary
