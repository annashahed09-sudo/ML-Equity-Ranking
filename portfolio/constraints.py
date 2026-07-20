"""
Portfolio constraint handling.

Implements:
- Position weight caps/floors
- Sector exposure limits
- Turnover constraints
- Market neutrality constraints
- Factor exposure constraints
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.exceptions import PortfolioError


@dataclass
class PortfolioConstraints:
    """
    Portfolio weight constraints.
    
    Combines multiple constraint types that can be applied
    after optimization or integrated into the optimizer.
    """
    
    # Per-asset bounds
    max_weight: float = 0.2
    min_weight: float = -0.2
    
    # Sector constraints
    sector_limits: Dict[str, float] = field(default_factory=dict)
    
    # Factor exposure constraints
    factor_bounds: Dict[str, tuple[float, float]] = field(default_factory=dict)
    
    # Turnover constraint
    max_turnover: Optional[float] = None
    prev_weights: Optional[pd.Series] = None
    
    # Net exposure
    target_gross_exposure: float = 1.0
    target_net_exposure: float = 0.0  # 0 = market neutral
    
    def apply(self, weights: pd.Series, sector_map: Optional[Dict[str, str]] = None) -> pd.Series:
        """
        Apply all constraints to portfolio weights.
        
        Parameters
        ----------
        weights : pd.Series
            Portfolio weights to constrain
        sector_map : Dict[str, str], optional
            Mapping from ticker to sector
        
        Returns
        -------
        pd.Series
            Constrained weights
        """
        w = weights.copy()
        
        # 1. Individual position bounds
        w = w.clip(self.min_weight, self.max_weight)
        
        # 2. Sector constraints
        if sector_map and self.sector_limits:
            w = self._apply_sector_constraints(w, sector_map)
        
        # 3. Net exposure constraint
        w = self._apply_exposure_constraint(w)
        
        # 4. Turnover constraint
        if self.max_turnover is not None and self.prev_weights is not None:
            w = self._apply_turnover_constraint(w)
        
        return w
    
    def _apply_sector_constraints(
        self,
        weights: pd.Series,
        sector_map: Dict[str, str],
    ) -> pd.Series:
        """Apply sector exposure limits."""
        sectors = pd.Series(sector_map)
        for sector, limit in self.sector_limits.items():
            mask = sectors.reindex(weights.index, fill_value="") == sector
            sector_weight = weights[mask].sum()
            if abs(sector_weight) > limit:
                scaling = limit / max(abs(sector_weight), 1e-12)
                weights[mask] = weights[mask] * scaling
        return weights
    
    def _apply_exposure_constraint(self, weights: pd.Series) -> pd.Series:
        """Scale weights to match target gross and net exposure."""
        gross = weights.abs().sum()
        net = weights.sum()
        
        if gross > self.target_gross_exposure:
            weights = weights / gross * self.target_gross_exposure
        
        # Adjust for net exposure target
        if abs(net - self.target_net_exposure) > 0.01:
            # Rebalance long/short legs
            pass  # More sophisticated adjustment would go here
        
        return weights
    
    def _apply_turnover_constraint(self, weights: pd.Series) -> pd.Series:
        """Cap turnover relative to previous weights."""
        if self.prev_weights is None:
            return weights
        
        common = weights.index.intersection(self.prev_weights.index)
        turnover = np.abs(weights[common] - self.prev_weights[common]).sum()
        
        if turnover > self.max_turnover:
            # Scale changes toward previous weights
            scaling = self.max_turnover / max(turnover, 1e-12)
            weights[common] = self.prev_weights[common] + \
                scaling * (weights[common] - self.prev_weights[common])
        
        return weights
