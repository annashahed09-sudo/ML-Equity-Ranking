"""
Covariance estimation methods for portfolio risk modeling.

Implements:
- Sample covariance (baseline)
- Ledoit-Wolf shrinkage estimator (optimal shrinkage)
- EWMA covariance (exponentially weighted)
- Factor model covariance (structured)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from core.exceptions import RiskError, ConvergenceError
from core.utils import ensure_array


class CovarianceEstimator(ABC):
    """Abstract base class for covariance estimation methods."""
    
    @abstractmethod
    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate covariance matrix from return data.
        
        Parameters
        ----------
        returns : pd.DataFrame
            T × N DataFrame of asset returns (T = time periods, N = assets)
        
        Returns
        -------
        pd.DataFrame
            N × N covariance matrix
        """
        ...
    
    @staticmethod
    def _validate(returns: pd.DataFrame) -> None:
        if returns.empty:
            raise RiskError("Empty returns DataFrame")
        if returns.isnull().all().all():
            raise RiskError("All return values are NaN")
        if returns.shape[1] < 2:
            raise RiskError(f"Need at least 2 assets, got {returns.shape[1]}")


class SampleCovariance(CovarianceEstimator):
    """
    Standard sample covariance matrix.
    
    Unbiased estimator using Bessel's correction (ddof=1).
    Can be numerically unstable when T < N (ill-conditioned).
    """
    
    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        self._validate(returns)
        cov = returns.cov()
        
        # Check for numerical issues
        if np.any(np.isnan(cov.values)):
            cov = cov.fillna(0)
        
        return cov


class LedoitWolfCovariance(CovarianceEstimator):
    """
    Ledoit-Wolf shrinkage covariance estimator.
    
    Shrinks sample covariance toward a structured target (identity matrix
    scaled by average variance). Provides a well-conditioned estimate even
    when T < N.
    
    Reference: Ledoit & Wolf (2004), "A well-conditioned estimator for
    large-dimensional covariance matrices"
    """
    
    def __init__(self, shrinkage_target: str = "constant_variance"):
        self.shrinkage_target = shrinkage_target
    
    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        self._validate(returns)
        
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf(assume_centered=False)
            lw.fit(returns.values)
            cov = pd.DataFrame(
                lw.covariance_,
                index=returns.columns,
                columns=returns.columns,
            )
            return cov
        except ImportError:
            return self._estimate_manual(returns)
    
    def _estimate_manual(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Manual Ledoit-Wolf implementation (Ledoit & Wolf, 2004).
        
        Implements the optimal shrinkage estimator for covariance matrices.
        Shrinks sample covariance toward a constant-variance target.
        The shrinkage intensity is analytically derived to minimize
        expected Frobenius loss.
        """
        X = returns.values
        T, N = X.shape
        
        # Sample covariance (unbiased, Bessel correction)
        sample_cov = np.cov(X, rowvar=False)
        
        # Shrinkage target: diagonal matrix with mean variance
        mean_var = np.trace(sample_cov) / N
        target = mean_var * np.eye(N)
        
        # Center the data
        X_centered = X - X.mean(axis=0)
        
        # Pi-hat matrix: sum of squared outer products (Ledoit-Wolf, eq. 14)
        pi_hat = np.zeros((N, N))
        for t in range(T):
            # Outer product of centered observations
            y_t = X_centered[t]
            outer_yt = np.outer(y_t, y_t)
            pi_hat += (outer_yt - sample_cov) ** 2
        pi_hat /= T
        
        # Gamma-hat squared: Frobenius norm of (sample - target)
        # Using squared norm to match Ledoit-Wolf notation
        gamma_hat_sq = np.sum((sample_cov - target) ** 2)
        
        # Rho-squared: sum(pi_hat) / T . The off-diagonal elements of pi_hat
        # capture the variance of the sample covariance elements.
        rho_hat_sq = np.sum(pi_hat)
        
        # Optimal shrinkage intensity (Ledoit-Wolf, eq. 19):
        # kappa* = (rho_hat_sq / T - gamma_hat_sq / T) / gamma_hat_sq
        # Simplified: kappa* = (sum(pi_hat) / T - gamma_hat_sq / T) / gamma_hat_sq
        #                = (rho_hat_sq - gamma_hat_sq) / (T * gamma_hat_sq)
        # 
        # Clipped to [0, 1] for finite-sample guarantees.
        numerator = rho_hat_sq - gamma_hat_sq
        denominator = T * gamma_hat_sq
        
        if denominator < 1e-12:
            shrinkage = 0.0
        else:
            shrinkage = max(0.0, min(1.0, numerator / denominator))
        
        # Shrunk covariance
        shrunk = (1 - shrinkage) * sample_cov + shrinkage * target
        
        return pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns)


class EWMACovariance(CovarianceEstimator):
    """
    Exponentially Weighted Moving Average covariance.
    
    Gives more weight to recent observations using a decay factor (lambda).
    Widely used in RiskMetrics.
    
    Parameter lambda: decay factor (lower = faster decay, more weight on recent)
    """
    
    def __init__(self, lambda_factor: float = 0.94):
        if not 0 < lambda_factor < 1:
            raise RiskError(f"Lambda must be in (0, 1), got {lambda_factor}")
        self.lambda_factor = lambda_factor
    
    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        self._validate(returns)
        
        X = returns.values
        T, N = X.shape
        
        # Compute EWMA weights
        weights = np.array([(1 - self.lambda_factor) * self.lambda_factor ** (T - 1 - t)
                           for t in range(T)])
        weights = weights / weights.sum()
        
        # Weighted mean
        mean = np.average(X, axis=0, weights=weights)
        
        # Weighted covariance
        X_centered = X - mean
        cov = np.zeros((N, N))
        for t in range(T):
            cov += weights[t] * np.outer(X_centered[t], X_centered[t])
        
        return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
