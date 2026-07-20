"""
Research pipeline orchestrator.

End-to-end pipeline that coordinates:
1. Data loading
2. Factor computation
3. Feature engineering
4. Model training with walk-forward validation
5. Portfolio backtesting
6. Performance reporting and diagnostics
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import settings
from core.types import ExperimentResult
from core.exceptions import ValidationError

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Comprehensive result from a research pipeline run."""
    
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    portfolio_returns: pd.DataFrame
    portfolio_summary: Dict[str, float]
    backtest_result: Optional[dict] = None
    feature_importance: Optional[pd.DataFrame] = None
    ic_series: Optional[pd.Series] = None
    factor_values: Optional[pd.DataFrame] = None
    config: Dict = field(default_factory=dict)
    duration: float = 0.0


class ResearchPipeline:
    """
    End-to-end research pipeline orchestrator.
    
    Coordinates data loading, factor computation, model training,
    validation, backtesting, and reporting into a single workflow.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
    
    def run(
        self,
        raw_df: pd.DataFrame,
        model_type: str = "ridge",
        factor_names: Optional[List[str]] = None,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        min_train_size: Optional[int] = None,
        long_pct: float = 0.2,
        short_pct: float = 0.2,
        transaction_cost_bps: float = 10.0,
    ) -> PipelineResult:
        """
        Run the complete research pipeline.
        
        Parameters
        ----------
        raw_df : pd.DataFrame
            Raw OHLCV data
        model_type : str
            Model type to use
        factor_names : List[str], optional
            Factors to compute (None = all)
        n_splits : int
            Number of walk-forward splits
        test_size : int
            Test set size (rows)
        min_train_size : int
            Minimum training set size
        long_pct : float
            Fraction of assets to go long
        short_pct : float
            Fraction of assets to go short
        transaction_cost_bps : float
            Transaction cost in basis points
        
        Returns
        -------
        PipelineResult
            Complete pipeline results
        """
        start = time.perf_counter()
        
        # 1. Prepare data (features + forward returns)
        df = self._prepare_data(raw_df, factor_names)
        
        # 2. Walk-forward validation
        from validation.walk_forward import WalkForwardSplitter
        from models.factory import ModelFactory
        from validation.metrics import (
            compute_information_coefficient,
            compute_rank_ic,
            compute_long_short_returns,
        )
        
        splitter = WalkForwardSplitter(
            n_splits=n_splits,
            test_size=test_size or max(1, len(df) // (n_splits + 1)),
            min_train_size=min_train_size or max(1, 2 * (test_size or 252)),
        )
        
        folds = splitter.split(df)
        feature_cols = self._get_feature_cols(df)
        
        all_preds: List[pd.DataFrame] = []
        fold_metrics_list: List[Dict] = []
        
        for fold in folds:
            train_df, test_df = splitter.get_fold_data(df, fold)
            
            model = ModelFactory.create(model_type)
            model.fit(
                train_df[feature_cols],
                train_df["forward_return"],
            )
            
            test_preds = model.predict(test_df[feature_cols])
            
            fold_pred = test_df[["date", "ticker", "forward_return"]].copy()
            fold_pred["model_score"] = test_preds
            fold_pred["fold_id"] = fold.fold_id
            
            # Compute fold IC
            ic = compute_information_coefficient(
                fold_pred["model_score"],
                fold_pred["forward_return"],
            )
            
            fold_metrics_list.append({
                "fold_id": fold.fold_id,
                "mean_ic": ic,
                "n_test": len(fold_pred),
                "train_start": str(fold.train_dates[0].date()),
                "train_end": str(fold.train_dates[1].date()),
                "test_start": str(fold.test_dates[0].date()),
                "test_end": str(fold.test_dates[1].date()),
            })
            
            all_preds.append(fold_pred)
        
        predictions = pd.concat(all_preds, ignore_index=True)
        fold_metrics_df = pd.DataFrame(fold_metrics_list)
        
        # 3. Portfolio backtest
        portfolio_returns, portfolio_summary = self._run_backtest(
            predictions, long_pct, short_pct, transaction_cost_bps
        )
        
        # 4. IC time series
        ic_series = compute_rank_ic(predictions)
        
        elapsed = time.perf_counter() - start
        
        return PipelineResult(
            predictions=predictions,
            fold_metrics=fold_metrics_df,
            portfolio_returns=portfolio_returns,
            portfolio_summary=portfolio_summary,
            ic_series=ic_series,
            duration=elapsed,
            config={
                "model_type": model_type,
                "n_splits": n_splits,
                "factor_names": factor_names,
            },
        )
    
    def _prepare_data(
        self,
        raw_df: pd.DataFrame,
        factor_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compute features and forward returns."""
        from factors.factory import compute_all_factors
        from .features import compute_forward_returns
        
        # Compute factors
        factor_result = compute_all_factors(raw_df, factor_names)
        
        # Combine with original data
        df = raw_df.copy()
        for col in factor_result.factor_values.columns:
            df[col] = factor_result.factor_values[col]
        
        # Compute forward returns
        df = compute_forward_returns(df)
        # Only drop rows where essential columns are NaN (forward_return)
        df = df.dropna(subset=["forward_return"]).reset_index(drop=True)
        
        # Forward-fill factor NaN values within each ticker (standard quant practice)
        exclude = {"date", "ticker", "open", "high", "low", "close", "volume",
                   "forward_return", "returns", "log_return"}
        factor_cols = [c for c in df.columns if c not in exclude]
        
        # Drop columns that are entirely NaN
        for col in factor_cols:
            if df[col].isna().all():
                df = df.drop(columns=[col])
        
        # Update factor_cols after dropping all-NaN columns
        factor_cols = [c for c in factor_cols if c in df.columns]
        
        # Forward-fill NaN values within each ticker
        df[factor_cols] = df.groupby("ticker")[factor_cols].transform(lambda x: x.ffill())
        
        # Drop any remaining rows with NaN in essential columns
        df = df.dropna(subset=["forward_return"] + factor_cols).reset_index(drop=True)
        
        return df
    
    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns (exclude metadata and target)."""
        exclude = {"date", "ticker", "open", "high", "low", "close", "volume",
                   "forward_return", "returns", "log_return"}
        return [c for c in df.columns if c not in exclude]
    
    def _run_backtest(
        self,
        predictions: pd.DataFrame,
        long_pct: float,
        short_pct: float,
        tc_bps: float,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Run long-short portfolio backtest."""
        from validation.metrics import (
            compute_long_short_returns,
            compute_sharpe_ratio,
            compute_sortino_ratio,
            compute_max_drawdown,
        )
        
        ls_returns = compute_long_short_returns(
            predictions,
            score_col="model_score",
            return_col="forward_return",
            long_pct=long_pct,
            short_pct=short_pct,
        )
        
        if ls_returns.empty or "long_short_return" not in ls_returns.columns:
            # Handle empty backtest results gracefully
            ls_returns = pd.DataFrame({"long_short_return": [0.0], "net_return": [0.0]})
            return ls_returns, {
                "mean_return": 0.0,
                "std_return": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "max_drawdown": 0.0,
                "net_sharpe": 0.0,
            }
        
        ls_returns["net_return"] = ls_returns["long_short_return"] - tc_bps / 1e4
        
        summary = {
            "mean_return": float(ls_returns["long_short_return"].mean()),
            "std_return": float(ls_returns["long_short_return"].std()),
            "sharpe": compute_sharpe_ratio(ls_returns["long_short_return"]),
            "sortino": compute_sortino_ratio(ls_returns["long_short_return"]),
            "max_drawdown": compute_max_drawdown(ls_returns["long_short_return"]),
            "net_sharpe": compute_sharpe_ratio(ls_returns["net_return"]),
        }
        
        return ls_returns, summary
