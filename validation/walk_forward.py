"""
Walk-forward validation splitters for time-series equity data.

Implements:
- Walk-forward split: expanding training window, fixed test window
- Expanding window split: continuously expanding training window
- Proper temporal ordering enforcement with purging
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.exceptions import ValidationError


@dataclass
class Fold:
    """A single validation fold with temporal indices."""
    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_dates: Tuple[pd.Timestamp, pd.Timestamp]
    test_dates: Tuple[pd.Timestamp, pd.Timestamp]

    def __repr__(self) -> str:
        return (
            f"Fold {self.fold_id}: "
            f"train [{self.train_dates[0].date()}, {self.train_dates[1].date()}] → "
            f"test ({self.test_dates[0].date()}, {self.test_dates[1].date()}]"
        )


class WalkForwardSplitter:
    """
    Walk-forward cross-validation splitter.
    
    Creates expanding training windows with non-overlapping test windows.
    Each successive fold uses all prior data for training and predicts
    on the next time period.
    
    Pattern:
        Fold 1: train [0, T1], test (T1, T2]
        Fold 2: train [0, T2], test (T2, T3]
        ...
        Fold N: train [0, TN], test (TN, T_end]
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = 252,
        min_train_size: int = 756,
        purge_window: int = 0,
        embargo_window: int = 0,
    ):
        if n_splits < 2:
            raise ValidationError("n_splits must be >= 2")
        if test_size < 1:
            raise ValidationError("test_size must be >= 1")
        if min_train_size < test_size:
            raise ValidationError("min_train_size must be >= test_size")
        
        self.n_splits = n_splits
        self.test_size = test_size
        self.min_train_size = min_train_size
        self.purge_window = purge_window
        self.embargo_window = embargo_window
    
    def split(self, df: pd.DataFrame, date_col: str = "date") -> List[Fold]:
        """
        Generate walk-forward fold indices.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataset with date column
        date_col : str
            Name of date column
        
        Returns
        -------
        List[Fold]
            List of fold definitions with temporal indices
        """
        dates = df[date_col].unique()
        n_dates = len(dates)
        n_total = len(df)
        
        if n_dates < self.min_train_size + self.test_size:
            raise ValidationError(
                f"Not enough data: {n_dates} dates, "
                f"need at least {self.min_train_size + self.test_size}"
            )
        
        # Compute anchor points in date-space
        fold_size = (n_dates - self.min_train_size) // self.n_splits
        if fold_size < self.test_size:
            fold_size = self.test_size
        
        folds = []
        for i in range(self.n_splits):
            test_end_date = self.min_train_size + (i + 1) * fold_size
            if test_end_date > n_dates:
                test_end_date = n_dates
            test_start_date = test_end_date - self.test_size
            train_end_date = test_start_date
            
            # Apply purging
            train_end_date = max(0, train_end_date - self.purge_window)
            
            if train_end_date < 1:
                break
            
            test_start = test_start_date
            test_end = test_end_date
            
            train_start = 0
            train_end = train_end_date
            
            train_dates = (dates[train_start], dates[train_end - 1])
            test_dates = (dates[test_start], dates[test_end - 1])
            
            folds.append(Fold(
                fold_id=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_dates=train_dates,
                test_dates=test_dates,
            ))
        
        if not folds:
            raise ValidationError("No folds could be created. Increase data size or reduce n_splits.")
        
        return folds
    
    def get_fold_data(
        self,
        df: pd.DataFrame,
        fold: Fold,
        date_col: str = "date",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract train/test DataFrames for a fold.
        
        Parameters
        ----------
        df : pd.DataFrame
            Full dataset
        fold : Fold
            Fold definition
        date_col : str
            Date column name
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (train_df, test_df)
        """
        dates = df[date_col].unique()
        train_dates_set = set(dates[fold.train_start:fold.train_end])
        test_dates_set = set(dates[fold.test_start:fold.test_end])
        
        train_df = df[df[date_col].isin(train_dates_set)].copy()
        test_df = df[df[date_col].isin(test_dates_set)].copy()
        
        # Apply embargo: remove data points near the train/test boundary
        if self.embargo_window > 0 and len(test_dates_set) > 0:
            test_start_date = min(test_dates_set)
            embargo_threshold = test_start_date - pd.Timedelta(days=self.embargo_window)
            train_df = train_df[train_df[date_col] <= embargo_threshold]
        
        return train_df, test_df
    
    def get_n_folds(self) -> int:
        """Get actual number of folds after accounting for data constraints."""
        return self.n_splits
    
    def __repr__(self) -> str:
        return (
            f"WalkForwardSplitter(n_splits={self.n_splits}, "
            f"test_size={self.test_size}, "
            f"min_train_size={self.min_train_size})"
        )


class ExpandingWindowSplitter:
    """
    Expanding window splitter with fixed test window.
    
    Simpler than walk-forward: uses a single initial training period
    and expands it by one test period at a time.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        initial_train_size: int = 756,
        test_size: int = 63,
    ):
        self.n_splits = n_splits
        self.initial_train_size = initial_train_size
        self.test_size = test_size
    
    def split(self, df: pd.DataFrame, date_col: str = "date") -> List[Fold]:
        dates = df[date_col].unique()
        n_dates = len(dates)
        
        folds = []
        for i in range(self.n_splits):
            train_end = self.initial_train_size + i * self.test_size
            test_start = train_end
            test_end = min(test_start + self.test_size, n_dates)
            
            if test_end > n_dates:
                break
            
            folds.append(Fold(
                fold_id=i,
                train_start=0,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_dates=(dates[0], dates[train_end - 1]),
                test_dates=(dates[test_start], dates[test_end - 1]),
            ))
        
        return folds
    
    def get_fold_data(self, df: pd.DataFrame, fold: Fold, date_col: str = "date"):
        dates = df[date_col].unique()
        train_dates_set = set(dates[fold.train_start:fold.train_end])
        test_dates_set = set(dates[fold.test_start:fold.test_end])
        return (
            df[df[date_col].isin(train_dates_set)].copy(),
            df[df[date_col].isin(test_dates_set)].copy(),
        )
