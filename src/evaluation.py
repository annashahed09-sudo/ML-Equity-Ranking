"""
Evaluation metrics for cross-sectional equity prediction.

Implements Information Coefficient (IC), portfolio returns, and stability metrics.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Tuple, List


def information_coefficient(
    scores: pd.Series,
    future_returns: pd.Series
) -> float:
    """
    Compute the Information Coefficient (Spearman rank correlation) between model scores and next-period returns.
    Both inputs should be aligned (same index, typically multi-indexed by date/ticker or grouped by date).
    """
    mask = scores.notnull() & future_returns.notnull()
    if mask.sum() < 2:
        return np.nan
    return spearmanr(scores[mask], future_returns[mask]).correlation


def compute_ic_by_date(
    df: pd.DataFrame,
    score_col: str = 'model_score',
    return_col: str = 'forward_return'
) -> pd.Series:
    """
    Compute IC for each date (cross-sectionally).
    Returns a Series indexed by date.
    """
    grouped = df[["date", score_col, return_col]].groupby('date', group_keys=False)
    return grouped.apply(
        lambda g: information_coefficient(g[score_col], g[return_col]), include_groups=False
    )


def long_short_portfolio_returns(
    df: pd.DataFrame,
    score_col: str = 'model_score',
    return_col: str = 'forward_return',
    long_quantile: float = 0.2,
    short_quantile: float = 0.2,
    transaction_cost_bps: float = 10.0
) -> pd.DataFrame:
    """
    Construct a long-short portfolio based on model scores and compute returns.
    At each date, go long top quantile, short bottom quantile, market-neutral.
    Deduct transaction costs (bps) on turnover.
    Returns DataFrame with columns: date, gross_return, net_return, turnover
    """
    results = []
    prev_longs = set()
    prev_shorts = set()
    for date, group in df.groupby('date'):
        n = len(group)
        n_long = max(1, int(n * long_quantile))
        n_short = max(1, int(n * short_quantile))
        ranked = group.sort_values(score_col, ascending=False)
        longs = set(ranked.head(n_long)['ticker'])
        shorts = set(ranked.tail(n_short)['ticker'])
        gross_return = group[group['ticker'].isin(longs)][return_col].mean() - \
                      group[group['ticker'].isin(shorts)][return_col].mean()
        # Turnover: fraction of positions changed
        turnover = (len(longs - prev_longs) + len(shorts - prev_shorts)) / (n_long + n_short)
        # Transaction cost in return space
        tc = turnover * (transaction_cost_bps / 1e4)
        net_return = gross_return - tc
        results.append({
            'date': date,
            'gross_return': gross_return,
            'net_return': net_return,
            'turnover': turnover
        })
        prev_longs = longs
        prev_shorts = shorts
    return pd.DataFrame(results)


def summarize_portfolio_performance(
    returns_df: pd.DataFrame
) -> dict:
    """
    Summarize portfolio performance: mean, std, Sharpe, cumulative return.
    """
    gross = returns_df['gross_return']
    net = returns_df['net_return']
    summary = {
        'mean_gross_return': gross.mean(),
        'mean_net_return': net.mean(),
        'std_gross_return': gross.std(),
        'std_net_return': net.std(),
        'sharpe_gross': gross.mean() / (gross.std() + 1e-8),
        'sharpe_net': net.mean() / (net.std() + 1e-8),
        'cumulative_gross': (1 + gross).prod() - 1,
        'cumulative_net': (1 + net).prod() - 1,
        'mean_turnover': returns_df['turnover'].mean()
    }
    return summary
