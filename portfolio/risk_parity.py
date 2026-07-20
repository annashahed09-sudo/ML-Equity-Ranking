"""
Risk Parity portfolio optimization.

Implements:
- Equal Risk Contribution (ERC) portfolio
- Risk budgeting with target risk contributions
- Leveraged risk parity
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, LinearConstraint

from core.exceptions import PortfolioError, ConvergenceError
from risk.covariance import CovarianceEstimator, SampleCovariance


class RiskParityOptimizer:
    """
    Risk Parity portfolio optimizer.
    
    Allocates capital so that each asset contributes equally to
    portfolio risk. Does not rely on expected return forecasts,
    making it robust to estimation error in mean returns.
    
    Reference: Maillard, Roncalli, Teiletche (2010)
    """
    
    def __init__(
        self,
        covariance_estimator: Optional[CovarianceEstimator] = None,
        max_weight: float = 0.3,
        min_weight: float = 0.01,
        leverage: float = 1.0,
    ):
        self.covariance_estimator = covariance_estimator or SampleCovariance()
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.leverage = leverage
    
    def equal_risk_contribution(self, returns: pd.DataFrame) -> pd.Series:
        """
        Compute Equal Risk Contribution (ERC) portfolio.
        
        Each asset contributes equally to total portfolio risk.
        
        Parameters
        ----------
        returns : pd.DataFrame
            T × N DataFrame of asset returns
        
        Returns
        -------
        pd.Series
            Optimal portfolio weights
        """
        if returns.empty:
            raise PortfolioError("Empty returns data")
        
        cov = self.covariance_estimator.estimate(returns)
        n = cov.shape[0]
        
        def risk_parity_objective(weights: np.ndarray) -> float:
            """Objective: minimize sum of squared risk contribution differences."""
            portfolio_var = weights @ cov.values @ weights
            portfolio_std = np.sqrt(max(portfolio_var, 1e-12))
            
            # Marginal risk contributions
            marginal = cov.values @ weights / portfolio_std
            
            # Total risk contributions
            risk_contrib = weights * marginal
            
            # Target: equal risk contribution
            target = portfolio_std / n
            diff = risk_contrib - target
            
            return float(diff @ diff)
        
        # Constraints
        constraints = LinearConstraint(np.ones(n), 1.0, 1.0)
        bounds = Bounds([self.min_weight] * n, [self.max_weight] * n)
        
        # Initial guess: inverse volatility
        vols = np.sqrt(np.diag(cov.values))
        inv_vol = 1.0 / vols
        x0 = inv_vol / inv_vol.sum()
        
        result = minimize(
            risk_parity_objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        
        if not result.success:
            raise ConvergenceError(
                f"Risk parity optimization failed: {result.message}"
            )
        
        weights = pd.Series(result.x, index=cov.columns)
        if self.leverage != 1.0:
            weights = weights * self.leverage
        
        return weights
    
    def risk_budgeting(
        self,
        target_risk_contributions: pd.Series,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute risk budgeting portfolio with target risk contributions.
        
        Parameters
        ----------
        target_risk_contributions : pd.Series
            Target risk contribution for each asset (must sum to 1)
        returns : pd.DataFrame
            Historical returns
        
        Returns
        -------
        pd.Series
            Optimal portfolio weights
        """
        cov = self.covariance_estimator.estimate(returns)
        n = cov.shape[0]
        
        if len(target_risk_contributions) != n:
            raise PortfolioError(
                f"Target risk contributions ({len(target_risk_contributions)}) "
                f"mismatch with assets ({n})"
            )
        
        target = target_risk_contributions.values
        target = target / target.sum()  # Normalize
        
        def risk_budget_objective(weights: np.ndarray) -> float:
            portfolio_var = weights @ cov.values @ weights
            portfolio_std = np.sqrt(max(portfolio_var, 1e-12))
            marginal = cov.values @ weights / portfolio_std
            risk_contrib = weights * marginal / portfolio_std
            diff = risk_contrib - target
            return float(diff @ diff)
        
        constraints = LinearConstraint(np.ones(n), 1.0, 1.0)
        bounds = Bounds([self.min_weight] * n, [self.max_weight] * n)
        
        vols = np.sqrt(np.diag(cov.values))
        inv_vol = 1.0 / vols
        x0 = inv_vol / inv_vol.sum()
        
        result = minimize(
            risk_budget_objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )
        
        if not result.success:
            raise ConvergenceError(
                f"Risk budgeting optimization failed: {result.message}"
            )
        
        return pd.Series(result.x, index=cov.columns)
