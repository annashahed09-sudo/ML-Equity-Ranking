# Quantitative Equity Research Platform (QERP)

**Institutional-grade cross-sectional equity ranking, portfolio optimization, and risk analysis platform.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests: 209 passing](https://img.shields.io/badge/tests-209%20passing%2C%204%20skipped-brightgreen.svg)](tests/)
[![Release: v1.0.0](https://img.shields.io/badge/release-v1.0.0-blue.svg)](https://github.com/annashahed09-sudo/ML-Equity-Ranking/releases)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Quick Start](#3-quick-start)
4. [Modules](#4-modules)
   - [Configuration](#41-configuration)
   - [Core Types & Utilities](#42-core-types--utilities)
   - [Factor Engine](#43-factor-engine)
   - [Model Layer](#44-model-layer)
   - [Validation Framework](#45-validation-framework)
   - [Risk Model](#46-risk-model)
   - [Portfolio Optimization](#47-portfolio-optimization)
   - [Signal Processing](#48-signal-processing)
   - [NLP & News Intelligence](#49-nlp--news-intelligence)
   - [Explainability](#410-explainability)
   - [Research Pipeline](#411-research-pipeline)
   - [API Server](#412-api-server)
   - [Dashboard](#413-dashboard)
5. [Mathematical Foundations](#5-mathematical-foundations)
6. [Research Methodology](#6-research-methodology)
7. [Testing](#7-testing)
8. [Contributing](#8-contributing)
9. [License & Disclaimer](#9-license--disclaimer)

---

## 1. Overview

QERP is a modular, institutional-grade quantitative equity research platform designed for **cross-sectional return prediction, factor-driven analysis, portfolio optimization, and risk management**.

### Core Capabilities

- **Cross-sectional ranking** of equities using multi-model ensembles
- **Academically grounded factor models** across 7 families: value, momentum, quality, volatility, liquidity, growth, profitability
- **Institutional portfolio optimization**: Mean-Variance, Risk Parity, Black-Litterman, Factor Model
- **Comprehensive risk analytics**: VaR, CVaR, factor decomposition, shrinkage covariance estimation
- **Walk-forward validation** with purging and embargo windows (Lopez de Prado methodology)
- **Interactive research dashboard** with 12 professional pages
- **REST API** for programmatic access

### Design Principles

- **Mathematical rigor**: Every estimator is academically grounded (Ledoit-Wolf, CAPM, Black-Litterman)
- **Type safety**: Full Pydantic configuration, typed data classes, comprehensive error hierarchy
- **Modularity**: 14 independently testable modules with clear boundaries
- **Reproducibility**: Seeded random operations, configuration-driven execution, experiment tracking
- **Production readiness**: Caching, retry logic, parallel processing, comprehensive logging

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ENTRY POINTS                                 │
│              main.py · api/app.py · dashboard/app.py                    │
└──────────────────────┬──────────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────────────┐
│                        RESEARCH PIPELINE                                │
│                research/pipeline.py · research/run.py                   │
└──────┬─────────┬──────────┬──────────┬──────────┬──────────┬───────────┘
       │         │          │          │          │          │
┌──────▼──┐ ┌───▼────┐ ┌──▼─────┐ ┌──▼─────┐ ┌──▼──────┐ ┌──▼───────┐
│  DATA   │ │FACTORS │ │ MODELS │ │VALIDATN│ │ SIGNAL  │ │ PORTFOLIO│
│ loader  │ │ value  │ │ linear │ │wf, pur │ │ comb    │ │ MVO      │
│ quality │ │ momen  │ │ tree   │ │ metrics│ │ norm    │ │ RP       │
│ cache   │ │ vol    │ │ ranker │ │ backtst│ │ orthog  │ │ BL       │
│         │ │ liq    │ │ neural │ │        │ │         │ │ factor   │
└────────┘ │ growth  │ │ tuning │ └────────┘ └─────────┘ │ cons     │
           │ prof    │ └────────┘                        └──────────┘
           └─────────┘
        ┌──────┐ ┌──────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐
        │ RISK │ │ NLP  │ │  NEWS    │ │EXPLAIN   │ │  CONFIG   │
        │ cov  │ │sentim│ │ ingest   │ │SHAP      │ │ pydantic  │
        │ fa   │ │lex   │ │ events   │ │PDP, ICE  │ │ settings  │
        │ VaR  │ └──────┘ └──────────┘ │ importnc  │ │ .env      │
        └──────┘                       └──────────┘ └───────────┘
```

### Module Dependency Graph

```
config  ←  core  ←  data
                        ↓
        ┌────────── factors ──────────┐
        ↓                              ↓
    signal ←──── models ────→ validation
        ↓         ↓               ↓
    portfolio ← risk ────→ research.pipeline
        ↓         ↓               ↓
    api/app  ←───┴────── dashboard/app
```

---

## 3. Quick Start

### Installation

```bash
# Clone and install
git clone <repository-url>
cd ML-Equity-Ranking

# Core dependencies
pip install -r requirements.txt

# Optional: ML models
pip install lightgbm xgboost catboost

# Optional: NLP
pip install transformers

# Optional: Dashboard
pip install streamlit plotly
```

### Usage

```bash
# Run research pipeline with synthetic data
python main.py research

# Launch interactive dashboard
python main.py dashboard

# Start API server
python main.py api

# Run all tests
python main.py test
```

### Makefile Targets

| Command | Description |
|---------|-------------|
| `make install` | Install core dependencies |
| `make test` | Run test suite |
| `make test-cov` | Run tests with coverage |
| `make lint` | Lint all modules |
| `make api` | Start FastAPI server |
| `make dashboard` | Start Streamlit dashboard |
| `make research` | Run research pipeline |

---

## 4. Modules

### 4.1 Configuration (`config/`)

Pydantic-v2 settings with environment variable support, validation, and sensible defaults.

```python
from config import settings

# Access any setting
settings.MODEL_BACKEND        # "cpu", "gpu", or "auto"
settings.VALIDATION_STRATEGY  # Walk-forward strategy
settings.PORTFOLIO_OBJECTIVE  # Optimization objective
```

**Key settings categories:**
- Environment & paths
- Data source configuration
- Factor parameters (winsorization, neutralization)
- Model hyperparameters (LightGBM, XGBoost, CatBoost)
- Validation strategy (walk-forward, purged CV, nested CV)
- Portfolio constraints (max weight, sector limits)
- Risk parameters (confidence level, covariance method)
- API and security configuration

### 4.2 Core Types & Utilities (`core/`)

Domain types, exception hierarchy, and mathematical utilities.

**Types**: `AssetReturn`, `FactorExposure`, `PortfolioWeight`, `RankedAsset`, `BacktestResult`, `ExperimentResult`

**Exceptions**: `QuantsError` → `DataError`, `ModelError`, `PortfolioError`, `RiskError`, `ConvergenceError`, etc.

**Mathematical Utilities**:
```python
from core.utils import (
    winsorize_series,       # Clip extreme values at quantiles
    zscore_normalize,       # Standard/robust/Gaussian rank normalization
    cross_sectional_zscore, # Per-date cross-sectional normalization
    stable_softmax,         # Numerically stable softmax
    entropy, kl_divergence, # Information-theoretic measures
    jensen_shannon_distance, # Symmetric distribution divergence
)
```

### 4.3 Factor Engine (`factors/`)

**7 academically-grounded factor families** with consistent interface:

#### Value
- `EarningsYield` — Earnings / Price
- `BookToMarket` — Book Value / Market Cap
- `FreeCashFlowYield` — FCF / Enterprise Value
- `CompositeValue` — Weighted combination

#### Momentum
- `TimeSeriesMomentum` — Price change over window (configurable: 63, 126, 252 days)
- `ResidualMomentum` — Momentum orthogonalized to market
- `VolatilityAdjustedMomentum` — Momentum scaled by inverse volatility
- `High52Week` — Distance from 52-week high
- `CompositeMomentum` — Multi-signal aggregation

#### Quality
- `ReturnOnEquity` — Net Income / Equity
- `GrossProfitability` — Gross Profit / Assets
- `PiotroskiFScore` — 9-point fundamental score
- `CompositeQuality` — Multi-factor quality score

#### Volatility
- `RealizedVolatility` — Standard deviation or Parkinson estimator
- `DownsideDeviation` — Negative return deviation only
- `MarketBeta` — CAPM market sensitivity
- `IdiosyncraticVolatility` — Residual vol from market model

#### Liquidity
- `AverageDailyVolume` — Rolling average trading volume
- `AmihudIlliquidity` — Price impact per dollar volume

#### Growth
- `RevenueGrowth` — YoY revenue growth rate
- `EPSGrowth` — YoY earnings per share growth
- `CompositeGrowth` — Multi-signal growth score

#### Profitability
- `GrossMargin` — Gross Profit / Revenue
- `OperatingMargin` — Operating Income / Revenue
- `NetMargin` — Net Income / Revenue
- `CompositeProfitability` — Multi-signal profitability score

### 4.4 Model Layer (`models/`)

**12 model types** with consistent `fit` / `predict` interface:

| Model | Type | Best For |
|-------|------|----------|
| Ridge | Linear | Baseline, stable signals |
| Lasso | Linear | Sparse feature selection |
| Elastic Net | Linear | Balanced regularization |
| Random Forest | Tree | Non-linear patterns, robustness |
| XGBoost | Tree | Gradient boosting, ranking |
| LightGBM | Tree | Large datasets, efficiency |
| CatBoost | Tree | Categorical features |
| Neural MLP | Neural | Complex non-linear relationships |
| LightGBM Ranker | Learning-to-Rank | Cross-sectional ranking |
| XGBoost Ranker | Learning-to-Rank | Cross-sectional ranking |
| Stacking Ensemble | Meta | Heterogeneous model aggregation |
| Voting Ensemble | Meta | Democratic model combination |

**Model Factory**: `ModelFactory.create("ridge")` — registry-based instantiation.

**Hyperparameter Tuning**:
- `BayesianOptimizer` — Tree-structured Parzen Estimator (Optuna-style)
- `GridOptimizer` — Exhaustive grid search

### 4.5 Validation Framework (`validation/`)

**Walk-Forward Validation** (standard in quantitative finance):
- Strict temporal train/test separation
- Configurable training size, test size, purging, and embargo
- Multiple folds with expanding or sliding window

**Metrics** (15+ performance measures):
- Information Coefficient (Spearman rank correlation)
- Rank IC time series
- Quantile returns (decile spread)
- Long-short portfolio returns
- Sharpe, Sortino, Calmar ratios
- Maximum drawdown and drawdown duration
- Win rate, profit factor, turnover
- Information ratio

```python
from validation.metrics import (
    compute_information_coefficient,  # IC: Spearman rank correlation
    compute_long_short_returns,        # Top/bottom quantile spread
    compute_ic_summary,                # Comprehensive IC statistics
)
```

### 4.6 Risk Model (`risk/`)

**Covariance Estimation**:
- `SampleCovariance` — Standard unbiased estimator
- `LedoitWolfCovariance` — Optimal shrinkage (Ledoit & Wolf, 2004)
- `EWMACovariance` — Exponentially weighted (RiskMetrics)

**Factor Risk Model**:
- `FactorRiskModel` — `Σ = B·F·B' + D`
- Factor exposure estimation via OLS
- Risk decomposition into systematic + idiosyncratic

**Risk Metrics**:
- Value at Risk (Historical, Parametric, Monte Carlo)
- Conditional VaR (Expected Shortfall)
- Tracking error (active risk)
- CAPM Beta and Jensen's Alpha
- Active risk from covariance

### 4.7 Portfolio Optimization (`portfolio/`)

**Institutional allocation techniques**:

| Optimizer | Method | Reference |
|-----------|--------|-----------|
| `MeanVarianceOptimizer` | Max Sharpe, Min Vol, Efficient Frontier | Markowitz (1952) |
| `RiskParityOptimizer` | Equal Risk Contribution | Maillard, Roncalli, Telletche (2010) |
| `MinimumVarianceOptimizer` | Global Minimum Variance | Haugen & Baker (1991) |
| `BlackLittermanOptimizer` | Prior + Views → Posterior | Black & Litterman (1992) |
| `FactorModelOptimizer` | Factor-based Covariance + Exposure Targeting | Ross (1976) |

**Constraints Engine** (`PortfolioConstraints`):
- Long-only / Long-short
- Max position weight
- Sector exposure limits
- Turnover penalties

### 4.8 Signal Processing (`signal/`)

**Combination Methods**:
- Equal-weighted, rank-weighted, IC-weighted
- Optimal signal combination (constrained optimization)

**Normalization**:
- Z-score, robust (MAD), Gaussian rank transform
- Sigmoid, min-max, winsorization

**Orthogonalization**:
- Gram-Schmidt sequential orthogonalization
- PCA-based decorrelation
- Residual orthogonalization (regression-based)

### 4.9 NLP & News Intelligence (`nlp/`, `news/`)

- `FinancialSentimentAnalyzer` — FinBERT (transformer), Lexicon, Hybrid
- `LexiconSentiment` — Domain-specific financial dictionary
- `NewsIngestor` — RSS feed ingestion with entity extraction
- `EventDetector` — Earnings, M&A, guidance changes, regulatory events

### 4.10 Explainability (`explainability/`)

- `PermutationImportance` — Feature importance with statistical significance
- `PartialDependence` — Marginal effect of features on predictions
- `IndividualConditionalExpectation` — Per-instance feature effects

### 4.11 Research Pipeline (`research/`)

End-to-end pipeline orchestrator:

```
Raw Data → Factor Computation → Feature Engineering → 
Walk-Forward Validation → Model Training → Prediction → 
Portfolio Backtest → Performance Reporting
```

```python
from research.pipeline import ResearchPipeline

pipeline = ResearchPipeline()
result = pipeline.run(
    raw_df=market_data,
    model_type="lightgbm",
    n_splits=5,
)
print(f"Mean IC: {result.ic_series.mean():.4f}")
print(f"Sharpe: {result.portfolio_summary['sharpe']:.2f}")
```

### 4.12 API Server (`api/`)

FastAPI application with endpoints:
- `POST /api/v1/rank` — Rank tickers using cross-sectional model
- `POST /api/v1/simulate/sp500` — Full S&P 500 walk-forward simulation
- `POST /api/v1/optimize/portfolio` — Portfolio optimization
- `GET /api/v1/factors` — List available factors
- `GET /health` — Health check

### 4.13 Dashboard (`dashboard/`)

Premium Streamlit dashboard with 12 professional pages:

| Page | Description |
|------|-------------|
| Executive Dashboard | High-level KPIs, market overview, factor performance |
| Research Workspace | Pipeline configuration and execution |
| Factor Explorer | Factor returns, correlations, rolling metrics |
| Feature Importance | SHAP, permutation importance, PDP, ICE |
| Portfolio Construction | MVO, risk parity, Black-Litterman, efficient frontier |
| Risk Analytics | VaR/CVaR, drawdown, factor decomposition |
| Backtest Explorer | Walk-forward attribution, trade analysis |
| News Intelligence | Sentiment analysis, event detection |
| Market Heatmap | Sector treemap, factor exposure map |
| Correlation Matrix | Asset/factor correlations, clustering |
| Model Comparison | Leaderboard, CV performance, diagnostics |
| Experiment Tracking | Run history, parameter comparison |

---

## 5. Mathematical Foundations

### Covariance Estimation

**Ledoit-Wolf Shrinkage** (Ledoit & Wolf, 2004):
```
Σ_shrunk = (1 - δ) * Σ_sample + δ * Σ_target
```
where δ is the optimal shrinkage intensity minimizing expected Frobenius loss.

**EWMA Covariance** (RiskMetrics):
```
Σ_t = λ * Σ_{t-1} + (1 - λ) * r_t * r_t'
```
where λ = 0.94 for daily data (JP Morgan RiskMetrics).

### Black-Litterman Model

**Prior**: Π = λ * Σ * w_market (reverse optimization)

**Posterior**:
```
μ_post = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} * [(τΣ)^{-1}Π + P'Ω^{-1}Q]
Σ_post = Σ + [(τΣ)^{-1} + P'Ω^{-1}P]^{-1}
```

### Risk Decomposition

**Factor Model**: R_t = B·f_t + ε_t

**Portfolio Risk**: σ²_p = w'·B·F·B'·w + w'·D·w

### Information Coefficient

**Spearman Rank IC**: ρ = 1 - (6 * Σ d²_i) / (n * (n² - 1))

### Walk-Forward Validation

- Train on expanding window: [1, t_i]
- Test on fixed window: [t_i + embargo, t_i + test_size]
- Purge: Remove [t_i - purge, t_i] to prevent leakage
- Metrics averaged across folds with standard deviation

---

## 6. Research Methodology

### Pre-processing

1. **Winsorization**: Extreme values clipped at 1st/99th percentile
2. **Cross-sectional normalization**: Per-date z-scores across universe
3. **Sector neutralization**: Returns orthogonalized to sector membership
4. **Missing data**: Gap filling (forward fill) with minimum observation thresholds

### Validation Strategy

- **Primary**: Walk-forward with 5 splits, 252-day test windows, 5-day purging
- **Secondary**: Purged cross-validation for hyperparameter tuning
- **Benchmark**: Equal-weighted portfolio of universe

### Performance Evaluation

- **Primary metric**: Information Coefficient (Spearman rank correlation)
- **Secondary**: Long-short portfolio Sharpe ratio (20/20 quantile spread)
- **Robustness**: IC standard deviation, IC Sharpe ratio, percentage positive
- **Statistical significance**: t-stat of mean IC > 2.0 threshold

---

## 7. Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=term --cov-report=html

# Run specific module
pytest tests/test_factors.py -v
pytest tests/test_risk.py -v

# Run with markers
pytest -m "not slow"  # Skip slow tests
```

**Test coverage by module**: config, core, data, factors, models, validation, risk, portfolio, signal, nlp, news, explainability, research, api

---

## 8. Contributing

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black isort flake8 pre-commit

# Install pre-commit hooks
pre-commit install

# Format code
make format

# Run linters
make lint
```

### Guidelines

- All new code must include type annotations
- All modules must have corresponding test files
- Mathematical implementations must cite references
- Configuration must use Pydantic validation
- Functions must have NumPy-style docstrings

---

## 9. License & Disclaimer

**License**: MIT License

**Disclaimer**: This software is intended solely for academic research, quantitative experimentation, and simulation-based analysis. It does not constitute financial advice, investment recommendations, or trading signals for execution. Past performance and simulated results are not indicative of future results.

---

*ML-Equity-Ranking v1.0.0 — Built with Python, NumPy, Pandas, Scikit-learn, LightGBM, XGBoost, and Plotly.*

## Release Notes (v1.0.0)

- **209 tests passing**, 4 skipped (optional XGBoost/LightGBM depend on OpenMP runtime)
- 14 modular packages with clean dependency graph
- 5 academically-grounded portfolio optimizers (MVO, Risk Parity, Min Variance, Black-Litterman, Factor Model)
- 7 factor families with 25+ signals
- 12 model types with unified factory interface
- Production-grade walk-forward validation with purging and embargo
- Real-time financial news intelligence pipeline
- Interactive Streamlit dashboard with 12 pages
- FastAPI REST API with ranking, simulation, and optimization endpoints
