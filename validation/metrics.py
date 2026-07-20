"""
Performance metrics for cross-sectional equity ranking evaluation.

Implements industry-standard metrics:
- Information Coefficient (Spearman rank correlation)
- Rank IC (cross-sectional per date)
- Quantile returns (long-short spreads)
- Sharpe / Sortino / Calmar ratios
- Maximum drawdown and drawdown duration
- Win rate and profit factor
- Portfolio turnover
- Hit rate and prediction accuracy
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def compute_information_coefficient(
    scores: pd.Series,
    forward_returns: pd.Series,
    method: str = "spearman",
) -> float:
    """
    Compute Information Coefficient between model scores and forward returns.
    
    Parameters
    ----------
    scores : pd.Series
        Model prediction scores
    forward_returns : pd.Series
        Observed forward returns
    method : str
        'spearman' (rank correlation) or 'pearson' (linear correlation)
    
    Returns
    -------
    float
        Information Coefficient (NaN if insufficient data)
    """
    mask = scores.notna() & forward_returns.notna()
    if mask.sum() < 10:
        return float("nan")
    
    if method == "spearman":
        ic, _ = scipy_stats.spearmanr(scores[mask], forward_returns[mask])
    elif method == "pearson":
        ic, _ = scipy_stats.pearsonr(scores[mask], forward_returns[mask])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return float(ic)


def compute_rank_ic(
    df: pd.DataFrame,
    score_col: str = "model_score",
    return_col: str = "forward_return",
    date_col: str = "date",
) -> pd.Series:
    """
    Compute cross-sectional rank IC for each date.
    
    For each date, computes the Spearman rank correlation between
    model scores and forward returns across all assets.
    
    Returns
    -------
    pd.Series
        Time series of daily IC values indexed by date
    """
    ic_values = {}
    
    for date, group in df.groupby(date_col):
        if len(group) < 10:
            continue
        ic = compute_information_coefficient(
            group[score_col], group[return_col], method="spearman"
        )
        ic_values[date] = ic
    
    return pd.Series(ic_values).sort_index()


def compute_quantile_returns(
    df: pd.DataFrame,
    score_col: str = "model_score",
    return_col: str = "forward_return",
    n_quantiles: int = 5,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Compute returns for each score quantile.
    
    At each date, assets are sorted by model score into quantiles.
    Returns the average forward return for each quantile.
    """
    results = []
    
    for date, group in df.groupby(date_col):
        if len(group) < n_quantiles * 2:
            continue
        
        group = group.copy()
        group["quantile"] = pd.qcut(group[score_col], n_quantiles, labels=False)
        
        for q in range(n_quantiles):
            q_returns = group[group["quantile"] == q][return_col]
            if len(q_returns) > 0:
                results.append({
                    "date": date,
                    "quantile": q,
                    "return": q_returns.mean(),
                })
    
    return pd.DataFrame(results)


def compute_long_short_returns(
    df: pd.DataFrame,
    score_col: str = "model_score",
    return_col: str = "forward_return",
    long_pct: float = 0.2,
    short_pct: float = 0.2,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Compute long-short portfolio returns from model scores.
    
    At each date: go long top long_pct, short bottom short_pct.
    Returns are equal-weighted within each leg.
    """
    results = []
    
    for date, group in df.groupby(date_col):
        if len(group) < 20:
            continue
        
        group = group.sort_values(score_col, ascending=False)
        n_long = max(1, int(len(group) * long_pct))
        n_short = max(1, int(len(group) * short_pct))
        
        long_ret = group.iloc[:n_long][return_col].mean()
        short_ret = group.iloc[-n_short:][return_col].mean()
        
        results.append({
            "date": date,
            "long_return": long_ret,
            "short_return": short_ret,
            "long_short_return": long_ret - short_ret,
        })
    
    return pd.DataFrame(results)


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized Sharpe ratio."""
    if len(returns) < 2:
        return float("nan")
    
    excess_returns = returns - risk_free_rate / periods_per_year
    if excess_returns.std() < 1e-12:
        return 0.0
    
    return float(np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std())


def compute_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252,
) -> float:
    """Compute Sortino ratio using downside deviation."""
    if len(returns) < 2:
        return float("nan")
    
    excess_returns = returns - risk_free_rate / periods_per_year
    downside = excess_returns[excess_returns < 0]
    
    if len(downside) == 0 or downside.std() < 1e-12:
        return 0.0 if excess_returns.mean() >= 0 else float("-inf")
    
    return float(np.sqrt(periods_per_year) * excess_returns.mean() / downside.std())


def compute_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Compute Calmar ratio: annualized return / max drawdown."""
    ann_return = returns.mean() * periods_per_year
    max_dd = compute_max_drawdown(returns)
    
    if abs(max_dd) < 1e-12:
        return 0.0 if ann_return >= 0 else float("-inf")
    
    return float(ann_return / abs(max_dd))


def compute_max_drawdown(returns: pd.Series) -> float:
    """Compute maximum drawdown from return series."""
    if len(returns) < 2:
        return 0.0
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    return float(drawdown.min())


def compute_drawdown_duration(returns: pd.Series) -> int:
    """Compute longest drawdown duration in periods."""
    if len(returns) < 2:
        return 0
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = cumulative < running_max
    
    # Count longest consecutive drawdown period
    max_duration = 0
    current = 0
    for is_dd in drawdown:
        if is_dd:
            current += 1
            max_duration = max(max_duration, current)
        else:
            current = 0
    
    return max_duration


def compute_win_rate(returns: pd.Series) -> float:
    """Compute fraction of positive return periods."""
    if len(returns) == 0:
        return float("nan")
    return float((returns > 0).mean())


def compute_profit_factor(returns: pd.Series) -> float:
    """Compute profit factor: sum(gains) / abs(sum(losses))."""
    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()
    
    if abs(losses) < 1e-12:
        return float("inf") if gains > 0 else float("nan")
    
    return float(gains / abs(losses))


def compute_turnover(
    current_weights: pd.Series,
    previous_weights: pd.Series,
) -> float:
    """Compute portfolio turnover as sum of absolute weight changes."""
    return float(np.abs(current_weights - previous_weights).sum())


def compute_information_ratio(
    active_returns: pd.Series,
    tracking_error: Optional[float] = None,
    periods_per_year: int = 252,
) -> float:
    """Compute Information Ratio: active return / tracking error."""
    if tracking_error is None:
        tracking_error = active_returns.std()
    
    if tracking_error < 1e-12:
        return 0.0
    
    return float(np.sqrt(periods_per_year) * active_returns.mean() / tracking_error)


def compute_ic_summary(ic_series: pd.Series) -> dict:
    """Compute comprehensive IC statistics."""
    valid_ic = ic_series.dropna()
    
    if len(valid_ic) == 0:
        return {
            "mean_ic": float("nan"),
            "std_ic": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "ic_sharpe": float("nan"),
            "pct_positive": float("nan"),
        }
    
    mean_ic = valid_ic.mean()
    std_ic = valid_ic.std()
    t_stat = mean_ic / (std_ic / np.sqrt(len(valid_ic))) if std_ic > 0 else 0
    p_value = 2 * (1 - scipy_stats.norm.cdf(abs(t_stat)))
    
    return {
        "mean_ic": float(mean_ic),
        "std_ic": float(std_ic),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "ic_sharpe": float(mean_ic / std_ic) if std_ic > 0 else 0,
        "pct_positive": float((valid_ic > 0).mean()),
        "n_dates": int(len(valid_ic)),
    }
