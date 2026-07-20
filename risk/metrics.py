"""
Risk metric computations for portfolio analysis.

Implements:
- Value at Risk (VaR) - parametric, historical, Monte Carlo
- Conditional VaR (Expected Shortfall)
- Tracking error
- Active risk
- Beta (CAPM)
- Alpha (Jensen's alpha)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def compute_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
    method: str = "historical",
    periods_per_year: int = 252,
) -> float:
    """
    Compute Value at Risk using specified method.
    
    Parameters
    ----------
    returns : pd.Series
        Portfolio return series
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% VaR)
    method : str
        'historical': empirical quantile
        'parametric': normal distribution assumption
        'monte_carlo': simulation-based
    periods_per_year : int
        Number of periods per year for scaling
    
    Returns
    -------
    float
        VaR estimate (positive number = loss amount)
    """
    if len(returns) < 10:
        return float("nan")
    
    if method == "historical":
        var = float(np.percentile(returns, (1 - confidence_level) * 100))
    
    elif method == "parametric":
        mu = returns.mean()
        sigma = returns.std()
        z = scipy_stats.norm.ppf(1 - confidence_level)
        var = mu + z * sigma
    
    elif method == "monte_carlo":
        mu = returns.mean()
        sigma = returns.std()
        n_simulations = 100000
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        simulated = rng.normal(mu, sigma, n_simulations)
        var = float(np.percentile(simulated, (1 - confidence_level) * 100))
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return abs(var)


def compute_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """
    Compute Conditional Value at Risk (Expected Shortfall).
    
    Average loss beyond VaR threshold. More coherent risk measure than VaR
    as it satisfies sub-additivity (Artzner et al., 1999).
    """
    if len(returns) < 10:
        return float("nan")
    
    var = compute_var(returns, confidence_level, method="historical")
    tail = returns[returns <= -var]
    
    if len(tail) == 0:
        return var
    
    return float(abs(tail.mean()))


def compute_tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized tracking error (active risk)."""
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common) < 10:
        return float("nan")
    
    active = portfolio_returns[common] - benchmark_returns[common]
    return float(active.std() * np.sqrt(periods_per_year))


def compute_active_risk(
    weights: pd.Series,
    benchmark_weights: pd.Series,
    covariance: pd.DataFrame,
) -> float:
    """Compute active risk from weights and covariance."""
    common = weights.index.intersection(benchmark_weights.index)
    w = weights.reindex(common, fill_value=0).values
    b = benchmark_weights.reindex(common, fill_value=0).values
    
    active_weights = w - b
    cov = covariance.loc[common, common].values
    
    active_var = active_weights @ cov @ active_weights
    return float(np.sqrt(max(0, active_var)) * np.sqrt(252))


def compute_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    periods: int = 252,
) -> float:
    """Compute rolling CAPM beta."""
    common = asset_returns.index.intersection(market_returns.index)
    if len(common) < 30:
        return float("nan")
    
    asset = asset_returns[common].iloc[-periods:]
    market = market_returns[common].iloc[-periods:]
    
    if len(asset) < 30:
        return float("nan")
    
    cov = np.cov(asset, market)[0, 1]
    mkt_var = np.var(market, ddof=1)
    
    if mkt_var < 1e-12:
        return 0.0
    
    return float(cov / mkt_var)


def compute_alpha(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.05,
) -> float:
    """Compute Jensen's alpha (excess return adjusted for market beta)."""
    common = asset_returns.index.intersection(market_returns.index)
    if len(common) < 30:
        return float("nan")
    
    asset = asset_returns[common]
    market = market_returns[common]
    
    # Annualized return
    ann_asset = (1 + asset).prod() ** (252 / len(asset)) - 1
    ann_market = (1 + market).prod() ** (252 / len(market)) - 1
    
    beta = compute_beta(asset, market)
    
    # CAPM: expected return = Rf + beta * (Rm - Rf)
    expected = risk_free_rate + beta * (ann_market - risk_free_rate)
    
    return float(ann_asset - expected)
