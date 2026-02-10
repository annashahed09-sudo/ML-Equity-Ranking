"""End-to-end training and evaluation pipeline for equity ranking models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .features import compute_features, compute_forward_returns, get_feature_columns
from .models import create_model
from .validation import create_walk_forward_splits, get_fold_data
from .evaluation import compute_ic_by_date
from .portfolio import run_portfolio_backtest


@dataclass
class PipelineResult:
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    portfolio_returns: pd.DataFrame
    portfolio_summary: Dict[str, float]


def _prepare_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    featured = compute_features(raw_df)
    featured = compute_forward_returns(featured)
    featured = featured.dropna().reset_index(drop=True)
    return featured


def run_walk_forward_pipeline(
    raw_df: pd.DataFrame,
    model_type: str = "ridge",
    n_splits: int = 5,
    test_size: Optional[int] = None,
    min_train_size: Optional[int] = None,
    model_kwargs: Optional[Dict] = None,
) -> PipelineResult:
    """Run full walk-forward model training/evaluation pipeline."""
    if model_kwargs is None:
        model_kwargs = {}

    df = _prepare_dataset(raw_df)
    feature_cols = get_feature_columns(df)
    if not feature_cols:
        raise ValueError("No feature columns found. Feature engineering failed.")

    folds = create_walk_forward_splits(
        df,
        n_splits=n_splits,
        test_size=test_size,
        min_train_size=min_train_size,
        date_column="date",
    )

    all_preds: List[pd.DataFrame] = []
    fold_metrics: List[Dict[str, float]] = []

    for fold in folds:
        train_df, test_df = get_fold_data(df, fold)

        X_train = train_df[feature_cols]
        y_train = train_df["forward_return"]
        X_test = test_df[feature_cols]

        model = create_model(model_type, **model_kwargs)
        model.fit(X_train, y_train)
        scores = model.predict(X_test)

        fold_pred = test_df[["date", "ticker", "forward_return"]].copy()
        fold_pred["model_score"] = scores
        fold_pred["fold_id"] = fold.fold_id

        fold_ic = compute_ic_by_date(fold_pred, score_col="model_score", return_col="forward_return")

        fold_metrics.append(
            {
                "fold_id": fold.fold_id,
                "test_start": fold.test_date_range[0],
                "test_end": fold.test_date_range[1],
                "mean_ic": float(np.nanmean(fold_ic.values)),
                "std_ic": float(np.nanstd(fold_ic.values)),
                "n_test_rows": int(len(fold_pred)),
            }
        )

        all_preds.append(fold_pred)

    predictions = pd.concat(all_preds, ignore_index=True).sort_values(["date", "ticker"])
    fold_metrics_df = pd.DataFrame(fold_metrics)

    portfolio_returns, portfolio_summary = run_portfolio_backtest(predictions)

    return PipelineResult(
        predictions=predictions,
        fold_metrics=fold_metrics_df,
        portfolio_returns=portfolio_returns,
        portfolio_summary=portfolio_summary,
    )
