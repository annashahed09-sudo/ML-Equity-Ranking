"""
Abstract base class for factor computation.

All factor families inherit from FactorComputer, which provides:
- Standardized input/output interfaces
- Cross-sectional normalization
- Winsorization and outlier handling
- Sector/market neutralization
- Quality checks and diagnostics
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.exceptions import FactorError
from core.types import FactorName, FactorType
from core.utils import (
    cross_sectional_zscore,
    gaussian_rank_transform,
    winsorize_series,
    validate_window,
)


@dataclass
class FactorResult:
    """Container for computed factor values with diagnostics."""
    
    name: FactorName
    factor_type: FactorType
    values: pd.Series
    description: str = ""
    
    # Diagnostics
    mean: float = 0.0
    std: float = 0.0
    skew: float = 0.0
    kurtosis: float = 0.0
    null_fraction: float = 0.0
    extreme_fraction: float = 0.0
    
    # Cross-sectional statistics
    ic_with_forward_returns: Optional[float] = None
    rank_ic: Optional[float] = None
    long_short_spread: Optional[float] = None
    decile_spreads: Optional[pd.Series] = None
    
    @staticmethod
    def from_series(
        name: FactorName,
        factor_type: FactorType,
        values: pd.Series,
        description: str = "",
    ) -> "FactorResult":
        """Create a FactorResult from a series with computed diagnostics."""
        valid = values.dropna()
        if len(valid) > 0:
            mean = float(valid.mean())
            std = float(valid.std())
            skew = float(valid.skew())
            kurtosis = float(valid.kurtosis())
            null_fraction = float(values.isnull().mean())
            # Count values beyond 4 standard deviations
            z_scores = np.abs((valid - mean) / (std + 1e-10))
            extreme_fraction = float((z_scores > 4).mean())
        else:
            mean = std = skew = kurtosis = 0.0
            null_fraction = 1.0
            extreme_fraction = 0.0
        
        return FactorResult(
            name=name,
            factor_type=factor_type,
            values=values,
            description=description,
            mean=mean,
            std=std,
            skew=skew,
            kurtosis=kurtosis,
            null_fraction=null_fraction,
            extreme_fraction=extreme_fraction,
        )


class FactorComputer(ABC):
    """
    Abstract base class for all factor computations.
    
    Subclasses implement compute() for their specific factor family.
    The base class handles normalization, winsorization, and validation.
    """
    
    def __init__(
        self,
        name: str,
        factor_type: FactorType,
        winsorize_limits: tuple[float, float] = (0.01, 0.99),
        cross_sectional_normalize: bool = True,
        neutralization: str = "none",  # 'none', 'sector', 'market'
    ):
        self.name = name
        self.factor_type = factor_type
        self.winsorize_limits = winsorize_limits
        self.cross_sectional_normalize = cross_sectional_normalize
        self.neutralization = neutralization
    
    @abstractmethod
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        """
        Compute factor values from input data.
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with at minimum: date, ticker, close, volume
            Additional columns required depend on the factor.
        **kwargs : dict
            Factor-specific parameters (window sizes, etc.)
        
        Returns
        -------
        FactorResult
            Computed factor values with diagnostics
        """
        ...
    
    def _validate_input(self, df: pd.DataFrame, required_cols: List[str]) -> None:
        """Validate that required columns exist."""
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise FactorError(
                f"{self.name} factor: missing required columns: {missing}"
            )
    
    def _post_process(
        self,
        values: pd.Series,
        df: pd.DataFrame,
        group_col: str = "date",
    ) -> pd.Series:
        """
        Apply post-processing: winsorization, normalization, neutralization.
        
        Order matters:
        1. Winsorize extreme values
        2. Cross-sectional z-score (per time step)
        3. Sector/market neutralization (if enabled)
        """
        result = values.copy()
        
        # 1. Winsorization
        if self.winsorize_limits:
            lower = result.quantile(self.winsorize_limits[0], interpolation="higher")
            upper = result.quantile(self.winsorize_limits[1], interpolation="lower")
            result = result.clip(lower=lower, upper=upper)
        
        # 2. Cross-sectional normalization
        if self.cross_sectional_normalize and group_col in df.columns:
            temp_df = df[[group_col]].copy()
            temp_df["value"] = result
            result = cross_sectional_zscore(temp_df, "value", group_col)
        
        # 3. Neutralization
        if self.neutralization == "sector" and "sector" in df.columns:
            result = self._neutralize_by_column(result, df, group_col, "sector")
        elif self.neutralization == "market":
            result = self._market_neutralize(result, df, group_col)
        
        return result
    
    @staticmethod
    def _neutralize_by_column(
        values: pd.Series,
        df: pd.DataFrame,
        time_col: str,
        neutral_col: str,
    ) -> pd.Series:
        """Neutralize factor values by a categorical column (e.g., sector)."""
        temp = pd.DataFrame({time_col: df[time_col], neutral_col: df[neutral_col], "value": values})
        
        def _demean(group: pd.DataFrame) -> pd.Series:
            return group["value"] - group["value"].mean()
        
        return temp.groupby([time_col, neutral_col], group_keys=False).apply(_demean)
    
    @staticmethod
    def _market_neutralize(values: pd.Series, df: pd.DataFrame, time_col: str) -> pd.Series:
        """Market-neutralize by subtracting cross-sectional mean per time step."""
        temp = pd.DataFrame({time_col: df[time_col], "value": values})
        
        def _demean(group: pd.DataFrame) -> pd.Series:
            return group["value"] - group["value"].mean()
        
        return temp.groupby(time_col, group_keys=False).apply(_demean)
    
    def _check_quality(self, result: FactorResult) -> List[str]:
        """Check factor quality and return list of warnings."""
        warnings = []
        if result.null_fraction > 0.5:
            warnings.append(f"High null fraction: {result.null_fraction:.1%}")
        if result.extreme_fraction > 0.05:
            warnings.append(f"High extreme value fraction: {result.extreme_fraction:.1%}")
        if abs(result.skew) > 3:
            warnings.append(f"High skewness: {result.skew:.2f}")
        if result.kurtosis > 20:
            warnings.append(f"High kurtosis: {result.kurtosis:.1f}")
        return warnings
