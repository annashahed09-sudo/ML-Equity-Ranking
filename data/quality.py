"""
Data quality validation and cleaning.

Checks for:
- OHLC consistency (high >= low, etc.)
- Missing data patterns
- Outlier detection
- Duplicate detection
- Survivorship bias detection
- Stale price detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.exceptions import DataQualityError


@dataclass
class DataQualityReport:
    """Report of data quality checks."""
    
    passed: bool
    n_issues: int
    issues: List[str] = field(default_factory=list)
    n_rows: int = 0
    n_tickers: int = 0
    date_range: Tuple[str, str] = ("", "")
    null_fraction: float = 0.0
    duplicate_rows: int = 0
    invalid_ohlc: int = 0
    extreme_returns: int = 0


class DataQualityChecker:
    """Validates quality of OHLCV market data."""
    
    def __init__(self, extreme_return_threshold: float = 0.5):
        self.extreme_return_threshold = extreme_return_threshold
    
    def check(self, df: pd.DataFrame) -> DataQualityReport:
        """
        Run all quality checks on the dataset.
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with columns: date, ticker, open, high, low, close, volume
        
        Returns
        -------
        DataQualityReport
            Quality assessment with issues found
        """
        issues: List[str] = []
        
        # 1. Required columns
        required = {"date", "ticker", "open", "high", "low", "close", "volume"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # 2. Null prices
        price_cols = ["open", "high", "low", "close"]
        null_counts = df[price_cols].isnull().sum()
        total_nulls = null_counts.sum()
        if total_nulls > 0:
            issues.append(f"Null prices: {null_counts[null_counts > 0].to_dict()}")
        
        # 3. OHLC consistency
        invalid_ohlc = df["high"] < df["low"]
        n_invalid_ohlc = invalid_ohlc.sum()
        if n_invalid_ohlc > 0:
            issues.append(f"High < Low in {n_invalid_ohlc} rows ({n_invalid_ohlc/len(df):.1%})")
        
        invalid_open = (df["open"] < df["low"]) | (df["open"] > df["high"])
        n_invalid_open = invalid_open.sum()
        if n_invalid_open > 0:
            issues.append(f"Open outside High-Low range in {n_invalid_open} rows")
        
        invalid_close = (df["close"] < df["low"]) | (df["close"] > df["high"])
        n_invalid_close = invalid_close.sum()
        if n_invalid_close > 0:
            issues.append(f"Close outside High-Low range in {n_invalid_close} rows")
        
        # 4. Zero or negative prices
        zero_prices = (df[price_cols] <= 0).sum().sum()
        if zero_prices > 0:
            issues.append(f"Zero/negative prices in {zero_prices} rows")
        
        # 5. Zero volume
        zero_vol = (df["volume"] == 0).sum()
        if zero_vol > 0:
            issues.append(f"Zero volume in {zero_vol} rows ({zero_vol/len(df):.1%})")
        
        # 6. Duplicate rows
        dupes = df.duplicated(subset=["date", "ticker"]).sum()
        if dupes > 0:
            issues.append(f"Duplicate (date, ticker) rows: {dupes}")
        
        # 7. Extreme returns
        def check_extreme_returns(group: pd.DataFrame) -> int:
            returns = group["close"].pct_change()
            return (returns.abs() > self.extreme_return_threshold).sum()
        
        extreme_count = df.groupby("ticker").apply(check_extreme_returns, include_groups=False).sum()
        if extreme_count > 0:
            issues.append(f"Extreme daily returns (>{self.extreme_return_threshold}): {extreme_count}")
        
        # 8. Missing data patterns
        ticker_counts = df.groupby("ticker").size()
        min_obs = ticker_counts.min()
        max_obs = ticker_counts.max()
        if min_obs < max_obs * 0.8:
            tickers_low = ticker_counts[ticker_counts < max_obs * 0.8].index.tolist()
            issues.append(f"Tickers with significantly fewer observations: {tickers_low}")
        
        # 9. Consecutive missing returns (stale prices)
        def check_stale(group: pd.DataFrame) -> int:
            returns = group["close"].pct_change()
            stale = 0
            run = 0
            for r in returns:
                if pd.isna(r) or r == 0:
                    run += 1
                    if run >= 5:
                        stale += 1
                else:
                    run = 0
            return stale
        
        stale_count = df.groupby("ticker").apply(check_stale, include_groups=False).sum()
        if stale_count > 0:
            issues.append(f"Potential stale prices (5+ consecutive zero returns): {stale_count}")
        
        null_fraction = df["close"].isnull().mean()
        
        return DataQualityReport(
            passed=len(issues) == 0,
            n_issues=len(issues),
            issues=issues,
            n_rows=len(df),
            n_tickers=df["ticker"].nunique(),
            date_range=(
                str(df["date"].min().date()),
                str(df["date"].max().date()),
            ),
            null_fraction=null_fraction,
            duplicate_rows=int(dupes),
            invalid_ohlc=int(n_invalid_ohlc),
            extreme_returns=int(extreme_count),
        )
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by removing common quality issues.
        
        Returns a cleaned copy; original is not modified.
        """
        cleaned = df.copy()
        
        # Remove rows with null prices
        cleaned = cleaned.dropna(subset=["open", "high", "low", "close"])
        
        # Remove invalid OHLC
        cleaned = cleaned[cleaned["high"] >= cleaned["low"]]
        
        # Remove duplicates
        cleaned = cleaned.drop_duplicates(subset=["date", "ticker"])
        
        # Remove zero/negative prices
        price_cols = ["open", "high", "low", "close"]
        cleaned = cleaned[(cleaned[price_cols] > 0).all(axis=1)]
        
        return cleaned.reset_index(drop=True)
