"""
Minimum Variance portfolio optimization.

Implements the global minimum variance portfolio (GMV).
No expected return forecasts required — purely risk-based allocation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, LinearConstraint

from core.exceptions import ConvergenceError
from risk.covariance import CovarianceEstimator, SampleCovariance


class MinimumVarianceOptimizer:
    """
    Global Minimum Variance portfolio.
    
    Finds the portfolio with the lowest possible volatility,
    without regard to expected returns. This is the leftmost
    point on the efficient frontier.
    """
    
    def __init__(
        self,
        covariance_estimator: Optional[CovarianceEstimator] = None,
        max_weight: float = 0.3,
        min_weight: float = 0.0,
        long_only: bool = True,
    ):
        self.covariance_estimator = covariance_estimator or SampleCovariance()
        self.max_weight = max_weight
        self.min_weight = min_weight if long_only else -max_weight
        self.long_only = long_only
    
    def minimize_volatility(self, returns: pd.DataFrame) -> pd.Series:
        """
        Compute the global minimum variance portfolio weights.
        
        Parameters
        ----------
        returns : pd.DataFrame
            T × N DataFrame of asset returns
        
        Returns
        -------
        pd.Series
            Minimum variance portfolio weights
        """
        cov = self.covariance_estimator.estimate(returns)
        n = cov.shape[0]
        
        def portfolio_variance(weights: np.ndarray) -> float:
            return float(weights @ cov.values @ weights)
        
        constraints = LinearConstraint(np.ones(n), 1.0, 1.0)
        bounds = Bounds([self.min_weight] * n, [self.max_weight] * n)
        
        # Initial guess: inverse volatility weighted
        vols = np.sqrt(np.diag(cov.values))
        inv_vol = 1.0 / (vols + 1e-10)
        x0 = inv_vol / inv_vol.sum()
        
        result = minimize(
            portfolio_variance,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        
        if not result.success:
            raise ConvergenceError(
                f"Minimum variance optimization failed: {result.message}"
            )
        
        return pd.Series(result.x, index=cov.columns)
