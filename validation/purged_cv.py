"""
Purged cross-validation for time series data (adapted from Lopez de Prado).

Implements purged K-fold CV with embargo windows to prevent:
- Training data leakage from future test periods
- Test period contamination from nearby training data
- Look-ahead bias in feature computation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.exceptions import ValidationError


@dataclass
class PurgedFold:
    """A single purged CV fold with time-based boundaries."""
    fold_id: int
    train_mask: np.ndarray
    test_mask: np.ndarray
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int


class PurgedCrossValidation:
    """
    Purged cross-validation with embargo windows.
    
    Prevents data leakage by:
    1. Purging: removing test-adjacent training samples
    2. Embargo: removing training samples immediately after test periods
    
    Reference: Lopez de Prado, "Advances in Financial Machine Learning"
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_window: int = 5,
        embargo_window: int = 5,
    ):
        if n_splits < 2:
            raise ValidationError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo_window = embargo_window
    
    def split(self, df: pd.DataFrame, date_col: str = "date") -> List[PurgedFold]:
        """Generate purged CV fold indices."""
        # Get unique dates and compute fold boundaries
        unique_dates = sorted(df[date_col].unique())
        n_dates = len(unique_dates)
        
        if n_dates < self.n_splits * 10:
            raise ValidationError(
                f"Too few dates ({n_dates}) for {self.n_splits} splits"
            )
        
        # Create date-indexed folds
        fold_size = n_dates // self.n_splits
        date_indices = {d: i for i, d in enumerate(unique_dates)}
        
        # Get numeric positions
        positions = df[date_col].map(date_indices).values
        
        folds = []
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n_dates) - 1
            
            # Create test mask: all rows where date position is in test range
            test_mask = (positions >= test_start) & (positions <= test_end)
            
            # Create train mask: all rows NOT in test range
            train_mask = ~test_mask
            
            # Apply purging: remove test-adjacent training data
            purge_start = max(0, test_start - self.purge_window)
            purge_train_mask = (positions < purge_start) | (positions > test_end + self.purge_window)
            train_mask = train_mask & purge_train_mask
            
            # Apply embargo: remove training data immediately after test
            embargo_end = min(n_dates - 1, test_end + self.embargo_window)
            embargo_mask = positions <= embargo_end
            train_mask = train_mask & embargo_mask
            
            folds.append(PurgedFold(
                fold_id=i,
                train_mask=train_mask,
                test_mask=test_mask,
                train_start_idx=int(np.where(train_mask)[0][0]) if train_mask.any() else 0,
                train_end_idx=int(np.where(train_mask)[0][-1]) if train_mask.any() else 0,
                test_start_idx=int(test_start),
                test_end_idx=int(test_end),
            ))
        
        return folds
    
    def get_fold_data(
        self,
        df: pd.DataFrame,
        fold: PurgedFold,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract train/test DataFrames for a purged fold."""
        return (
            df[fold.train_mask].copy(),
            df[fold.test_mask].copy(),
        )
    
    def __repr__(self) -> str:
        return (
            f"PurgedCV(n_splits={self.n_splits}, "
            f"purge={self.purge_window}d, "
            f"embargo={self.embargo_window}d)"
        )
