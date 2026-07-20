"""
Factor risk model for portfolio risk decomposition.

Decomposes portfolio risk into:
- Systematic risk (factor exposures)
- Idiosyncratic risk (stock-specific)
- Sector risk
- Risk attribution by factor
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.exceptions import RiskError


class FactorRiskModel:
    """
    Multi-factor risk model.
    
    Decomposes return covariance into factor component + idiosyncratic:
    Σ = B * F * B' + D
    
    where:
    - B: N×K factor exposure matrix
    - F: K×K factor covariance matrix
    - D: N×N diagonal idiosyncratic variance matrix
    """
    
    def __init__(self, factor_returns: pd.DataFrame):
        """
        Parameters
        ----------
        factor_returns : pd.DataFrame
            T × K DataFrame of factor returns (time periods × factors)
        """
        self.factor_returns = factor_returns
        self.factor_cov = factor_returns.cov()
        self.factor_names = list(factor_returns.columns)
        self.n_factors = len(self.factor_names)
    
    def compute_factor_exposures(
        self,
        asset_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Estimate factor exposures (betas) via regression.
        
        Parameters
        ----------
        asset_returns : pd.DataFrame
            T × N DataFrame of asset returns
        
        Returns
        -------
        pd.DataFrame
            N × K DataFrame of factor betas
        """
        T_assets = asset_returns.index
        T_factors = self.factor_returns.index
        common_idx = T_assets.intersection(T_factors)
        
        if len(common_idx) < 60:
            raise RiskError(
                f"Insufficient overlapping data: {len(common_idx)} periods"
            )
        
        aligned_assets = asset_returns.loc[common_idx]
        aligned_factors = self.factor_returns.loc[common_idx]
        
        # Add constant for alpha estimation
        X = np.column_stack([np.ones(len(common_idx)), aligned_factors.values])
        
        # OLS regression for each asset
        betas = []
        for asset in aligned_assets.columns:
            y = aligned_assets[asset].values
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            betas.append(beta[1:])  # Exclude intercept
        
        return pd.DataFrame(betas, index=aligned_assets.columns, columns=self.factor_names)
    
    def decompose_risk(
        self,
        weights: pd.Series,
        asset_returns: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Decompose portfolio risk into systematic and idiosyncratic components.
        
        Parameters
        ----------
        weights : pd.Series
            N-vector of portfolio weights
        asset_returns : pd.DataFrame
            T × N DataFrame of asset returns
        
        Returns
        -------
        Dict[str, float]
            Risk decomposition with keys: systematic_risk, idiosyncratic_risk,
            total_risk, and per-factor contributions
        """
        common_idx = asset_returns.index.intersection(self.factor_returns.index)
        if len(common_idx) < 60:
            raise RiskError("Insufficient overlapping data for risk decomposition")
        
        # Compute factor betas and residuals
        factor_betas = self.compute_factor_exposures(asset_returns)
        aligned_assets = asset_returns.loc[common_idx]
        aligned_factors = self.factor_returns.loc[common_idx]
        
        # Idiosyncratic variances
        residuals = aligned_assets.values - aligned_factors.values @ factor_betas.values.T
        idio_var = np.var(residuals, axis=0, ddof=1)
        
        # Portfolio-level risk
        w = weights.reindex(factor_betas.index, fill_value=0).values
        
        # Systematic risk: w' * B * F * B' * w
        portfolio_factor_exposure = w @ factor_betas.values
        systematic_var = portfolio_factor_exposure @ self.factor_cov.values @ portfolio_factor_exposure
        
        # Idiosyncratic risk: w' * D * w
        idio_var_portfolio = np.sum(w ** 2 * idio_var)
        
        # Total risk
        total_var = systematic_var + idio_var_portfolio
        total_vol = np.sqrt(max(0, total_var))
        
        # Factor contributions
        factor_contrib = {}
        for i, factor in enumerate(self.factor_names):
            contrib = (
                portfolio_factor_exposure[i] *
                (self.factor_cov.values @ portfolio_factor_exposure)[i]
            )
            factor_contrib[factor] = contrib
        
        return {
            "total_risk": total_vol,
            "systematic_risk": np.sqrt(max(0, systematic_var)),
            "idiosyncratic_risk": np.sqrt(max(0, idio_var_portfolio)),
            "factor_contributions": factor_contrib,
            "pct_systematic": systematic_var / total_var if total_var > 0 else 0,
        }
    
    def compute_tracking_error(
        self,
        weights: pd.Series,
        benchmark_weights: pd.Series,
        asset_returns: pd.DataFrame,
    ) -> float:
        """Compute tracking error of portfolio vs benchmark."""
        common = weights.index.intersection(benchmark_weights.index)
        w_active = weights.reindex(common, fill_value=0).values - \
                   benchmark_weights.reindex(common, fill_value=0).values
        
        aligned_returns = asset_returns[common]
        active_returns = aligned_returns @ w_active
        
        return float(active_returns.std() * np.sqrt(252))
