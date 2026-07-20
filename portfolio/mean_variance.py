"""
Mean-Variance Optimization (Markowitz, 1952).

Implements:
- Maximum Sharpe ratio portfolio (tangency portfolio)
- Minimum variance for given target return
- Efficient frontier computation
- Portfolio weights with constraints
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, LinearConstraint

from core.exceptions import PortfolioError, ConvergenceError
from risk.covariance import CovarianceEstimator, SampleCovariance


class MeanVarianceOptimizer:
    """
    Mean-Variance portfolio optimizer.
    
    Finds optimal portfolio weights by maximizing risk-adjusted returns
    subject to constraints. Supports long-only and long-short portfolios.
    """
    
    def __init__(
        self,
        covariance_estimator: Optional[CovarianceEstimator] = None,
        risk_free_rate: float = 0.05,
        max_weight: float = 0.2,
        min_weight: float = 0.0,
        weight_sum: float = 1.0,
    ):
        self.covariance_estimator = covariance_estimator or SampleCovariance()
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.weight_sum = weight_sum
    
    def max_sharpe(
        self,
        expected_returns: pd.Series,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute maximum Sharpe ratio portfolio (tangency portfolio).
        
        Parameters
        ----------
        expected_returns : pd.Series
            N-vector of expected returns for each asset
        returns : pd.DataFrame
            T × N DataFrame of historical returns for covariance estimation
        
        Returns
        -------
        pd.Series
            Optimal portfolio weights
        """
        self._validate_inputs(expected_returns, returns)
        cov = self.covariance_estimator.estimate(returns)
        n = len(expected_returns)
        
        # Objective: minimize negative Sharpe ratio
        def negative_sharpe(weights: np.ndarray) -> float:
            portfolio_return = weights @ expected_returns.values
            portfolio_var = weights @ cov.values @ weights
            portfolio_std = np.sqrt(max(portfolio_var, 1e-12))
            sharpe = (portfolio_return - self.risk_free_rate / 252) / portfolio_std
            return -sharpe
        
        # Constraints: weights sum to target
        constraints = LinearConstraint(np.ones(n), self.weight_sum, self.weight_sum)
        bounds = Bounds([self.min_weight] * n, [self.max_weight] * n)
        
        # Initial guess: equal weight
        x0 = np.array([self.weight_sum / n] * n)
        
        result = minimize(
            negative_sharpe,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        
        if not result.success:
            raise ConvergenceError(f"Max Sharpe optimization failed: {result.message}")
        
        return pd.Series(result.x, index=expected_returns.index)
    
    def min_variance_for_return(
        self,
        target_return: float,
        expected_returns: pd.Series,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute minimum variance portfolio for a target expected return.
        
        Parameters
        ----------
        target_return : float
            Target expected portfolio return (annualized)
        expected_returns : pd.Series
            N-vector of expected returns
        returns : pd.DataFrame
            Historical returns for covariance estimation
        
        Returns
        -------
        pd.Series
            Optimal portfolio weights
        """
        self._validate_inputs(expected_returns, returns)
        cov = self.covariance_estimator.estimate(returns)
        n = len(expected_returns)
        
        def portfolio_variance(weights: np.ndarray) -> float:
            return float(weights @ cov.values @ weights)
        
        # Constraints
        return_constraint = LinearConstraint(
            expected_returns.values.reshape(1, -1),
            target_return / 252,
            target_return / 252,
        )
        weight_constraint = LinearConstraint(np.ones(n), self.weight_sum, self.weight_sum)
        bounds = Bounds([self.min_weight] * n, [self.max_weight] * n)
        
        x0 = np.array([self.weight_sum / n] * n)
        
        result = minimize(
            portfolio_variance,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=[weight_constraint, return_constraint],
            options={"maxiter": 1000},
        )
        
        if not result.success:
            raise ConvergenceError(
                f"Min variance for return {target_return} failed: {result.message}"
            )
        
        return pd.Series(result.x, index=expected_returns.index)
    
    def efficient_frontier(
        self,
        expected_returns: pd.Series,
        returns: pd.DataFrame,
        n_points: int = 50,
    ) -> pd.DataFrame:
        """
        Compute the efficient frontier.
        
        Returns
        -------
        pd.DataFrame
            Efficient frontier with columns: return, volatility, sharpe, weights
        """
        self._validate_inputs(expected_returns, returns)
        
        # Compute min and max returns
        min_ret = expected_returns.min()
        max_ret = expected_returns.max()
        
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        frontier = []
        for target in target_returns:
            try:
                weights = self.min_variance_for_return(
                    target, expected_returns, returns
                )
                cov = self.covariance_estimator.estimate(returns)
                port_ret = weights @ expected_returns.values
                port_vol = np.sqrt(weights @ cov.values @ weights)
                sharpe = (port_ret - self.risk_free_rate / 252) / port_vol
                
                frontier.append({
                    "return": port_ret * 252,
                    "volatility": port_vol * np.sqrt(252),
                    "sharpe": sharpe * np.sqrt(252),
                    "weights": weights,
                })
            except (ConvergenceError, PortfolioError):
                continue
        
        return pd.DataFrame(frontier)
    
    @staticmethod
    def _validate_inputs(expected_returns: pd.Series, returns: pd.DataFrame) -> None:
        if expected_returns.empty:
            raise PortfolioError("Empty expected returns")
        if returns.empty:
            raise PortfolioError("Empty returns data")
        if len(expected_returns) != returns.shape[1]:
            raise PortfolioError(
                f"Expected returns ({len(expected_returns)}) and "
                f"returns columns ({returns.shape[1]}) mismatch"
            )
