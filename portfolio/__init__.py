"""
Portfolio optimization and construction engine.

Implements institutional allocation techniques:
- Mean-Variance Optimization (Markowitz, 1952)
- Risk Parity (equal risk contribution)
- Minimum Variance portfolio (global minimum variance)
- Black-Litterman model (prior + views → posterior)
- Factor model portfolio (factor-based covariance, exposure targeting)
- Equal Risk Contribution (ERC)
- Transaction cost and turnover modeling
- Position and sector constraint handling

All optimizers accept a CovarianceEstimator to allow pluggable
covariance estimation (sample, Ledoit-Wolf, EWMA, etc.).
"""

from .mean_variance import MeanVarianceOptimizer
from .risk_parity import RiskParityOptimizer
from .min_variance import MinimumVarianceOptimizer
from .black_litterman import BlackLittermanOptimizer
from .factor_model import FactorModelOptimizer
from .constraints import PortfolioConstraints

__all__ = [
    "MeanVarianceOptimizer",
    "RiskParityOptimizer",
    "MinimumVarianceOptimizer",
    "BlackLittermanOptimizer",
    "FactorModelOptimizer",
    "PortfolioConstraints",
]
