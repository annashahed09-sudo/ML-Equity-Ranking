"""
Comprehensive risk model for quantitative equity portfolios.

Implements:
- Covariance estimation (sample, Ledoit-Wolf, EWMA)
- Factor risk decomposition
- Value at Risk (VaR, CVaR)
- Tracking error and active risk
- Stress testing via scenario simulation
"""

from .covariance import (
    CovarianceEstimator,
    SampleCovariance,
    LedoitWolfCovariance,
    EWMACovariance,
)
from .factor_risk import FactorRiskModel
from .metrics import (
    compute_var,
    compute_cvar,
    compute_tracking_error,
    compute_active_risk,
    compute_beta,
    compute_alpha,
)

__all__ = [
    "CovarianceEstimator",
    "SampleCovariance",
    "LedoitWolfCovariance",
    "EWMACovariance",
    "FactorRiskModel",
    "compute_var",
    "compute_cvar",
    "compute_tracking_error",
    "compute_active_risk",
    "compute_beta",
    "compute_alpha",
]
