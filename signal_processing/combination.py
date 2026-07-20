"""
Signal combination engine.

Combines multiple signals into a single composite signal using:
- Equal weighting
- Optimal weighting (maximize IC)
- Rank-based combination
- Machine learning combination
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from core.exceptions import SignalError


class SignalCombiner:
    """
    Combines multiple alpha signals into a single composite signal.
    
    Supports multiple combination methods with different tradeoffs
    between robustness and optimality.
    """
    
    @staticmethod
    def equal_weight(signals: pd.DataFrame) -> pd.Series:
        """Equal-weight combination of signals (most robust)."""
        if signals.empty:
            raise SignalError("Empty signals DataFrame")
        return signals.mean(axis=1)
    
    @staticmethod
    def rank_average(signals: pd.DataFrame) -> pd.Series:
        """Average of cross-sectional ranks of each signal."""
        ranks = signals.rank(pct=True)
        return ranks.mean(axis=1)
    
    @staticmethod
    def weighted(signals: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
        """Weighted combination with specified weights."""
        w = pd.Series(weights)
        common = signals.columns.intersection(w.index)
        if len(common) == 0:
            raise SignalError("No overlapping signal names with weights")
        
        w = w[common] / w[common].sum()
        return signals[common] @ w
    
    @staticmethod
    def ic_weighted(
        signals: pd.DataFrame,
        forward_returns: pd.Series,
        date_col: str = "date",
    ) -> pd.Series:
        """Weight by historical IC performance."""
        from validation.metrics import compute_information_coefficient
        
        # Compute IC for each signal
        ics = {}
        for col in signals.columns:
            ic = compute_information_coefficient(
                signals[col], forward_returns, method="spearman"
            )
            ics[col] = max(ic, 0)  # Only positive weights
        
        total = sum(ics.values())
        if total <= 0:
            return signals.mean(axis=1)
        
        weights = {k: v / total for k, v in ics.items()}
        return SignalCombiner.weighted(signals, weights)
    
    @staticmethod
    def optimal(
        signals: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> pd.Series:
        """
        Find optimal combination weights by maximizing IC.
        
        Uses constrained optimization to find weights that maximize
        the rank correlation between the composite signal and
        forward returns.
        """
        n = signals.shape[1]
        
        def negative_ic(weights: np.ndarray) -> float:
            composite = signals.values @ weights
            from scipy.stats import spearmanr
            ic, _ = spearmanr(composite, forward_returns)
            return -abs(ic)
        
        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
        bounds = [(0, 1)] * n
        
        result = minimize(
            negative_ic,
            x0=np.array([1/n] * n),
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
        )
        
        if not result.success:
            raise SignalError(f"Optimal signal combination failed: {result.message}")
        
        weights = pd.Series(result.x, index=signals.columns)
        return signals @ weights
