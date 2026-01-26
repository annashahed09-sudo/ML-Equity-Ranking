# ML-Equity-Ranking AI Coding Agent Instructions

## Project Overview

This is a **machine learning project for cross-sectional equity return prediction**. The goal is to predict relative next-period asset performance and validate these predictions using walk-forward testing and portfolio construction.

Key constraints: **Sound problem formulation, statistical discipline, and realistic validation** take priority over headline performance metrics.

## Architecture & Core Components

### Data Flow
1. **Data Loading** → Raw OHLCV market data (time series)
2. **Feature Engineering** → Cross-sectional momentum, volatility, trend indicators
3. **Normalization** → Features standardized per time-step (preserve cross-sectional relationships)
4. **Model Training** → Ridge regression (baseline) and Gradient Boosting (nonlinear)
5. **Walk-Forward Validation** → Rolling train/test windows (temporal structure respected)
6. **Portfolio Evaluation** → Long-short ranking-based positions, before/after transaction costs

### Key Principle
**Cross-sectional prediction, not absolute return forecasting.** Features and targets are relative rankings across assets at each time step, not individual return levels.

## Project Structure (Target)

```
/workspaces/ML-Equity-Ranking/
├── data/                      # Market data and feature caches
├── src/
│   ├── data_loader.py         # OHLCV data ingestion
│   ├── features.py            # Momentum, volatility, trend indicators
│   ├── models.py              # Ridge, GradientBoosting wrappers
│   ├── validation.py          # Walk-forward split logic
│   ├── evaluation.py          # Information Coefficient, portfolio metrics
│   └── portfolio.py           # Long-short construction and backtest
├── notebooks/                 # Exploratory analysis and results
├── tests/                      # Unit tests for data pipeline, features
├── requirements.txt           # Dependencies (pandas, scikit-learn, etc.)
└── .github/copilot-instructions.md
```

## Critical Workflows & Conventions

### 1. Feature Engineering (NOT generic ML)
- **Normalize cross-sectionally**: Each feature is standardized across the asset universe at time $t$, not across time
- Common features: rolling momentum, volatility, distance from trend
- **Avoid data leakage**: ensure features at time $t$ use only data up to time $t$
- Example: Momentum for asset $i$ at time $t$ = (close_t - close_{t-20}) / close_{t-20}, then cross-section standardize

### 2. Walk-Forward Validation (NOT random CV)
- Time-series models require temporal splits
- Pattern: train on [0, T1], test on (T1, T2]; then train on [0, T2], test on (T2, T3); etc.
- Never shuffle time-series data; respect temporal ordering
- Store validation fold definitions deterministically

### 3. Evaluation Metrics
- **Information Coefficient (IC)**: Rank correlation between model scores and next-period returns
- **Portfolio returns**: Construct long-short portfolio from model rankings; report cumulative PnL
- **Stability**: Report IC across validation folds; high variance is red flag
- **Transaction costs**: Simple model: 5-10 bps per trade; deduct from portfolio returns

### 4. Model Constraints
- **Ridge regression**: Baseline for linear signals, easy interpretation
- **Gradient Boosting**: Capture nonlinearities, but tune depth/iterations carefully to avoid overfitting
- Discourage deep neural networks on small datasets (high overfitting risk in weak signal regime)

## Development Practices

### Naming & Conventions
- Use descriptive variable names: `model_scores`, `signal_values`, `rank_correlation`, not generic `X`, `y`, `pred`
- Time-indexed DataFrames: Always have a `date` column with datetime type
- Features should have suffixes: `_mom` (momentum), `_vol` (volatility), `_trend` (trend distance)

### Testing Strategy
- Unit tests for feature calculations: verify normalization, no lookahead, correct formula
- Integration tests: validate walk-forward splits don't leak future data
- Regression tests: store IC and portfolio returns for baseline; alert if performance drops unexpectedly

### Jupyter Workflows
- Keep notebooks for exploration and visualization only
- Move production logic into `src/` modules
- Notebooks should call `src/` functions, not reimplement logic

## Dependencies & Tools

**Core stack**:
- `pandas`: time-series data manipulation
- `numpy`: numerical operations
- `scikit-learn`: Ridge, GradientBoosting models, preprocessing
- `scipy.stats`: rank correlations, statistical tests
- `matplotlib` / `seaborn`: visualization

**Optional**:
- `yfinance`: public equity data source
- `pytest`: unit testing
- `jupyter`: exploration notebooks

## Common Pitfalls to Avoid

1. **Data leakage**: Using future data in feature calculations (check date logic)
2. **Cross-validation errors**: Using random CV instead of time-respecting splits
3. **Overfitting**: High performance on 1–2 assets is a red flag; validate across assets and time
4. **Ignoring transaction costs**: Long-short portfolios have meaningful rebalancing costs
5. **Non-stationarity**: Financial relationships change; track IC stability over time
6. **Ignoring data quality**: Missing data, corporate actions (splits, dividends) affect relative returns

## Key Files to Reference

- **README.md**: Motivation, high-level methodology, disclaimers
- **src/features.py** (when built): Definitive source for feature definitions and cross-section logic
- **src/validation.py** (when built): Walk-forward split implementation
- **src/evaluation.py** (when built): IC and portfolio return calculations

---

