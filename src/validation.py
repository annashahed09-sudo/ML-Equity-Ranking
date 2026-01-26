"""
Walk-forward validation framework for time-series equity data.

Walk-forward validation respects temporal ordering by splitting data into expanding
training windows and non-overlapping test windows. This prevents lookahead bias.

Pattern:
  Fold 1: train on [0, T1], test on (T1, T2]
  Fold 2: train on [0, T2], test on (T2, T3]
  ...
  Fold N: train on [0, TN], test on (TN, T_end]
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, NamedTuple
from dataclasses import dataclass


@dataclass
class WalkForwardFold:
    """Represents a single walk-forward fold."""
    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_date_range: Tuple[pd.Timestamp, pd.Timestamp]
    test_date_range: Tuple[pd.Timestamp, pd.Timestamp]
    
    def __repr__(self):
        return (
            f"WalkForwardFold(fold={self.fold_id}, "
            f"train=[{self.train_date_range[0].date()}, {self.train_date_range[1].date()}], "
            f"test=({self.test_date_range[0].date()}, {self.test_date_range[1].date()}])"
        )


def create_walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_size: int = None,
    min_train_size: int = None,
    date_column: str = 'date'
) -> List[WalkForwardFold]:
    """
    Create walk-forward validation splits respecting temporal order.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame must have a date column (sorted)
    n_splits : int, default 5
        Number of walk-forward folds
    test_size : int, optional
        Size of test set (in rows). Default: total_rows // (n_splits + 1)
    min_train_size : int, optional
        Minimum training set size. Default: 2 * test_size
    date_column : str, default 'date'
        Name of date column

    Returns
    -------
    List[WalkForwardFold]
        List of fold definitions
    """
    n_rows = len(df)
    
    if test_size is None:
        test_size = max(1, n_rows // (n_splits + 1))
    
    if min_train_size is None:
        min_train_size = max(1, 2 * test_size)
    
    if min_train_size + test_size > n_rows:
        raise ValueError(
            f"Not enough data: need at least {min_train_size + test_size} rows, "
            f"but have {n_rows}"
        )
    
    folds = []
    
    for fold_id in range(n_splits):
        test_end_idx = min_train_size + (fold_id + 1) * test_size
        
        if test_end_idx > n_rows:
            break  # Not enough data for another fold
        
        train_end_idx = test_end_idx - test_size
        test_start_idx = train_end_idx
        
        train_indices = np.arange(0, train_end_idx)
        test_indices = np.arange(test_start_idx, test_end_idx)
        
        train_dates = df.iloc[train_indices][date_column]
        test_dates = df.iloc[test_indices][date_column]
        
        fold = WalkForwardFold(
            fold_id=fold_id,
            train_indices=train_indices,
            test_indices=test_indices,
            train_date_range=(train_dates.min(), train_dates.max()),
            test_date_range=(test_dates.min(), test_dates.max()),
        )
        
        folds.append(fold)
    
    return folds


def get_fold_data(
    df: pd.DataFrame,
    fold: WalkForwardFold
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract train and test DataFrames for a fold.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    fold : WalkForwardFold
        Fold definition

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    train_df = df.iloc[fold.train_indices].copy()
    test_df = df.iloc[fold.test_indices].copy()
    
    return train_df, test_df


def validate_fold_integrity(
    df: pd.DataFrame,
    folds: List[WalkForwardFold],
    date_column: str = 'date'
) -> Tuple[bool, List[str]]:
    """
    Check for common issues in fold definitions.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    folds : List[WalkForwardFold]
        List of folds
    date_column : str, default 'date'
        Date column name

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_issues)
    """
    issues = []
    
    if not folds:
        issues.append("No folds defined")
        return False, issues
    
    # Check for overlap between folds
    all_test_indices = set()
    for i, fold in enumerate(folds):
        test_set = set(fold.test_indices)
        overlap = all_test_indices & test_set
        if overlap:
            issues.append(f"Fold {i} overlaps with previous test sets: {len(overlap)} indices")
        all_test_indices.update(test_set)
    
    # Check for data leakage (test dates in training set)
    for fold in folds:
        train_df = df.iloc[fold.train_indices]
        test_df = df.iloc[fold.test_indices]
        
        if (train_df[date_column].max() > test_df[date_column].min()):
            issues.append(
                f"Fold {fold.fold_id}: train dates extend beyond test start date"
            )
    
    return len(issues) == 0, issues
