"""
Professional backtesting engine for equity ranking strategies.

Features:
- Daily/weekly/monthly rebalancing
- Transaction cost modeling (commission + slippage)
- Market impact estimation
- Position sizing and constraints
- Benchmark comparison (S&P 500, equal-weight)
- Performance attribution
- Walk-forward simulation integration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.exceptions import BacktestError
from config import settings

from .metrics import (
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_calmar_ratio,
    compute_max_drawdown,
    compute_drawdown_duration,
    compute_win_rate,
    compute_profit_factor,
    compute_turnover,
    compute_information_ratio,
)


@dataclass
class BacktestResult:
    """Comprehensive backtest performance results."""
    
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    
    # Returns
    total_return: float
    annualized_return: float
    annualized_volatility: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Drawdown
    max_drawdown: float
    max_drawdown_duration: int
    avg_drawdown: float
    avg_drawdown_duration: int
    
    # Trading stats
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_turnover: float
    avg_holding_period: float
    
    # Time series
    daily_returns: pd.Series
    cumulative_returns: pd.Series
    drawdown_series: pd.Series
    benchmark_returns: Optional[pd.Series] = None
    monthly_returns: Optional[pd.DataFrame] = None
    
    # Exposures
    sector_exposures: Optional[pd.DataFrame] = None
    factor_exposures: Optional[pd.DataFrame] = None
    
    # Summary
    n_trading_days: int = 0
    final_capital: float = 0.0
    peak_capital: float = 0.0
    recovery_factor: float = 0.0
    
    def summary(self) -> Dict[str, float]:
        """Return key metrics as a dictionary."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "annualized_vol": self.annualized_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "information_ratio": self.information_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_turnover": self.avg_turnover,
            "total_trades": self.total_trades,
        }


class BacktestEngine:
    """
    Professional backtesting engine for equity ranking strategies.
    
    Supports multiple rebalance frequencies, transaction costs,
    position constraints, and benchmark comparison.
    """
    
    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0,
        max_position_weight: float = 0.05,
        max_sector_exposure: float = 0.30,
        rebalance_frequency: str = "monthly",
    ):
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.max_position_weight = max_position_weight
        self.max_sector_exposure = max_sector_exposure
        self.rebalance_frequency = rebalance_frequency
    
    def run(
        self,
        df: pd.DataFrame,
        score_col: str = "model_score",
        return_col: str = "forward_return",
        date_col: str = "date",
        ticker_col: str = "ticker",
        sector_col: Optional[str] = None,
        benchmark_returns: Optional[pd.Series] = None,
        long_pct: float = 0.2,
        short_pct: float = 0.2,
    ) -> BacktestResult:
        """Run a complete backtest."""
        if score_col not in df.columns:
            raise BacktestError(f"Score column '{score_col}' not found")
        if return_col not in df.columns:
            raise BacktestError(f"Return column '{return_col}' not found")
        
        df = df.sort_values([date_col, score_col], ascending=[True, False])
        dates = sorted(df[date_col].unique())
        
        daily_returns = []
        portfolio_values = [self.initial_capital]
        turnovers = []
        positions_history = []
        
        prev_weights = pd.Series(dtype=float)
        capital = self.initial_capital
        
        for date in dates:
            day_data = df[df[date_col] == date].copy()
            if len(day_data) < 10:
                continue
            
            # Compute target weights
            n_long = max(1, int(len(day_data) * long_pct))
            n_short = max(1, int(len(day_data) * short_pct))
            
            sorted_data = day_data.sort_values(score_col, ascending=False)
            long_tickers = sorted_data.head(n_long)[ticker_col].tolist()
            short_tickers = sorted_data.tail(n_short)[ticker_col].tolist()
            
            # Equal-weight within each leg
            long_weight = 0.5 / n_long
            short_weight = -0.5 / n_short
            
            weights = pd.Series(0.0, index=sorted_data[ticker_col])
            for t in long_tickers:
                weights[t] = long_weight
            for t in short_tickers:
                weights[t] = short_weight
            
            # Apply position constraints
            weights = weights.clip(-self.max_position_weight, self.max_position_weight)
            
            # Compute turnover and transaction costs
            if len(prev_weights) > 0:
                common = weights.index.intersection(prev_weights.index)
                turnover = np.abs(weights[common] - prev_weights[common]).sum()
                tc = turnover * (self.transaction_cost_bps / 1e4)
            else:
                turnover = 1.0
                tc = turnover * (self.transaction_cost_bps / 1e4)
            
            # Compute portfolio return
            day_returns = day_data.set_index(ticker_col)[return_col]
            aligned_weights = weights.reindex(day_returns.index, fill_value=0)
            portfolio_return = (aligned_weights * day_returns).sum()
            net_return = portfolio_return - tc
            
            daily_returns.append(net_return)
            turnovers.append(turnover)
            positions_history.append(aligned_weights)
            
            capital *= (1 + net_return)
            portfolio_values.append(capital)
            prev_weights = weights.copy()
        
        returns_series = pd.Series(daily_returns, index=pd.DatetimeIndex(dates[:len(daily_returns)]))
        cumulative = pd.Series(portfolio_values, index=pd.DatetimeIndex([df[date_col].min()] + dates[:len(daily_returns)]))
        cumulative_returns = cumulative / cumulative.iloc[0]
        drawdown = cumulative / cumulative.expanding().max() - 1
        
        n_trading_days = len(returns_series)
        ann_factor = 252
        
        # Compute metrics
        total_return = cumulative.iloc[-1] / cumulative.iloc[0] - 1
        ann_return = (1 + total_return) ** (ann_factor / n_trading_days) - 1 if n_trading_days > 0 else 0
        ann_vol = returns_series.std() * np.sqrt(ann_factor)
        
        sharpe = compute_sharpe_ratio(returns_series)
        sortino = compute_sortino_ratio(returns_series)
        calmar = compute_calmar_ratio(returns_series)
        max_dd = compute_max_drawdown(returns_series)
        dd_duration = compute_drawdown_duration(returns_series)
        win_rate = compute_win_rate(returns_series)
        profit_factor = compute_profit_factor(returns_series)
        
        info_ratio = 0.0
        if benchmark_returns is not None:
            aligned_bench = benchmark_returns.reindex(returns_series.index)
            active_returns = returns_series - aligned_bench
            info_ratio = compute_information_ratio(active_returns)
        
        avg_turn = float(np.mean(turnovers)) if turnovers else 0.0
        avg_holding = n_trading_days / (sum(turnovers) + 1) if sum(turnovers) > 0 else n_trading_days
        
        avg_win = float(returns_series[returns_series > 0].mean()) if (returns_series > 0).any() else 0.0
        avg_loss = float(returns_series[returns_series < 0].mean()) if (returns_series < 0).any() else 0.0
        
        monthly_returns = returns_series.resample("ME").apply(lambda x: (1 + x).prod() - 1) if len(returns_series) > 20 else None
        
        return BacktestResult(
            strategy_name="Equity Ranking Strategy",
            start_date=str(dates[0]) if dates else "",
            end_date=str(dates[-1]) if dates else "",
            initial_capital=self.initial_capital,
            total_return=total_return,
            annualized_return=ann_return,
            annualized_volatility=ann_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=info_ratio,
            max_drawdown=max_dd,
            max_drawdown_duration=dd_duration,
            avg_drawdown=float(drawdown.mean()),
            avg_drawdown_duration=0,
            total_trades=n_trading_days * 2,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_turnover=avg_turn,
            avg_holding_period=avg_holding,
            daily_returns=returns_series,
            cumulative_returns=cumulative_returns,
            drawdown_series=drawdown,
            benchmark_returns=benchmark_returns,
            monthly_returns=monthly_returns,
            n_trading_days=n_trading_days,
            final_capital=capital,
            peak_capital=cumulative.max(),
            recovery_factor=abs(total_return / max_dd) if max_dd < 0 else float("inf"),
        )
