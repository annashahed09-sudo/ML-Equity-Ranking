"""
Factor model portfolio optimization.

Implements portfolio construction using factor-based covariance estimation
and factor exposure targeting. Supports:
- Factor covariance estimation (structured)
- Factor exposure targeting
- Factor-neutral portfolio construction
- Sector/factor tilts
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, LinearConstraint

from core.exceptions import PortfolioError, ConvergenceError
from risk.covariance import CovarianceEstimator, SampleCovariance
from risk.factor_risk import FactorRiskModel


class FactorModelOptimizer:
    """
    Factor-based portfolio optimizer.

    Uses a multi-factor model for covariance estimation:
    Σ = B * F * B' + D

    where B is the N x K factor exposure matrix, F is the K x K
    factor covariance matrix, and D is the diagonal idiosyncratic
    variance matrix.

    This produces more stable portfolio weights when N is large
    and T (time periods) is limited.
    """

    def __init__(
        self,
        factor_returns: pd.DataFrame,
        covariance_estimator: Optional[CovarianceEstimator] = None,
        risk_aversion: float = 1.0,
        max_weight: float = 0.2,
        min_weight: float = -0.2,
    ):
        self.factor_returns = factor_returns
        self.risk_model = FactorRiskModel(factor_returns)
        self.covariance_estimator = covariance_estimator or SampleCovariance()
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.min_weight = min_weight

    def factor_covariance(self, asset_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute factor-based covariance matrix.

        Σ = B * F * B' + D

        Uses aligned historical factor returns to compute residual
        variances, ensuring dimensional consistency.

        Returns a structured covariance estimate that is better
        conditioned than the sample covariance when N >> T.
        """
        exposures = self.risk_model.compute_factor_exposures(asset_returns)
        factor_cov = self.factor_returns.cov()

        # Systematic component: B * F * B'
        systematic = exposures.values @ factor_cov.values @ exposures.values.T

        # Idiosyncratic variance: diagonal of residual variances from
        # the factor model R_t = B * f_t + e_t
        # We need aligned factor returns and asset returns to compute residuals
        common_idx = asset_returns.index.intersection(self.factor_returns.index)
        if len(common_idx) < 10:
            # Fall back to average off-diagonal if insufficient overlapping data
            idio_var = np.diag(asset_returns.cov().values)
        else:
            aligned_asset = asset_returns.loc[common_idx].values  # T' x N
            aligned_factors = self.factor_returns.loc[common_idx].values  # T' x K
            # Predicted returns: F @ B^T  (T' x N)
            predicted = aligned_factors @ exposures.values.T  # T' x N
            residuals = aligned_asset - predicted  # T' x N
            idio_var = np.var(residuals, axis=0, ddof=1)

        idio_diag = np.diag(idio_var)
        total_cov = systematic + idio_diag

        return pd.DataFrame(
            total_cov,
            index=exposures.index,
            columns=exposures.index,
        )

    def target_factor_exposures(
        self,
        asset_returns: pd.DataFrame,
        target_exposures: Dict[str, float],
    ) -> pd.Series:
        """
        Construct portfolio targeting specific factor exposures.

        Minimizes tracking error to a factor exposure target while
        maintaining a fully invested portfolio.

        Parameters
        ----------
        asset_returns : pd.DataFrame
            T x N historical returns
        target_exposures : Dict[str, float]
            Target exposure for each factor

        Returns
        -------
        pd.Series
            Optimal portfolio weights
        """
        exposures = self.risk_model.compute_factor_exposures(asset_returns)
        cov = self.factor_covariance(asset_returns)

        n = len(exposures)
        target = np.array([target_exposures.get(f, 0.0) for f in exposures.columns])

        def objective(weights: np.ndarray) -> float:
            """Minimize tracking error to factor target + risk-aversion weighted variance."""
            port_exposures = weights @ exposures.values
            tracking_error = np.sum((port_exposures - target) ** 2)
            portfolio_var = weights @ cov.values @ weights
            return tracking_error + self.risk_aversion * portfolio_var

        constraints = LinearConstraint(np.ones(n), 1.0, 1.0)
        bounds = Bounds([self.min_weight] * n, [self.max_weight] * n)

        x0 = np.array([1.0 / n] * n)

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )

        if not result.success:
            raise ConvergenceError(
                f"Factor targeting optimization failed: {result.message}"
            )

        return pd.Series(result.x, index=exposures.index, name="factor_weights")
