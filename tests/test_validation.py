import pandas as pd
import numpy as np
from src.validation import create_walk_forward_splits, validate_fold_integrity

def test_walk_forward_splits():
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'ticker': ['A']*100
    })
    folds = create_walk_forward_splits(df, n_splits=3, test_size=10, min_train_size=20)
    assert len(folds) == 3
    valid, issues = validate_fold_integrity(df, folds)
    assert valid
