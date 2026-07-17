# Cross-Sectional Equity Ranking & Simulation System
## Institutional Quant Research System Specification

---

## 0. Quickstart (localhost)

Requires Python 3.9–3.11.

```bash
# 1. Create an environment and install runtime + dev dependencies
python -m venv .venv && source .venv/bin/activate
make setup                     # pip install -e ".[dev]" + pre-commit install
#   (or, minimal:  pip install -r requirements.txt)

# 2. Configure LOCAL access secrets (no paid API keys are ever required)
make secrets                   # generates strong random values into .env (gitignored)
#   (or manually:  cp .env.example .env  and edit the two values)

# 3. Run the test suite + linters
make test                      # pytest with coverage
make lint                      # ruff + isort + black checks

# 4. Run an end-to-end demo on synthetic data
make run                       # python run_all.py

# 5. Run a real S&P 500 walk-forward backtest (bundled fallback universe, offline-safe)
make backtest START=2022-01-01 END=2023-01-01 MODEL=advanced_ensemble
make report                    # same, but also writes reports/simulation_report.pdf

# 6. Launch the interfaces (bind to 127.0.0.1 only; .env is auto-loaded)
make dashboard                 # Streamlit UI on http://localhost:8501
make api                       # FastAPI service on http://localhost:8000  (GET /health)
```

> **Cost & privacy:** this project uses **no paid APIs**. Market data comes from
> `yfinance` (free) and news from public NYT/Economist RSS feeds (free); sentiment
> is computed by a local lexicon (no external LLM). The only secrets are a local
> dashboard password and API token you set yourself — nothing is billed or sent to
> a third party. Services bind to `127.0.0.1` by default, so they are not exposed
> to your network. See [§6 Security Model](#6-security-model).

Run `make help` to list all available commands. Configuration defaults live in
`configs/config.yaml` and `configs/models.yaml`; local data/feature caches go in `data/`.

---

## 1. System Overview

This system is a modular **quantitative research and signal generation platform** designed for **cross-sectional equity ranking, portfolio simulation, and factor-driven analysis**.

It supports:

- Cross-sectional return prediction (relative ranking, not absolute forecasting)
- Multi-model ensemble signal generation
- Walk-forward backtesting and validation
- NLP-enhanced sentiment integration
- Optional GPU-accelerated numerical execution
- Research-grade simulation of long/short portfolios

The system is explicitly designed for **research and alpha signal exploration**, not execution or live trading.

---

## 2. System Objectives

### Primary Objectives

- Generate statistically robust **cross-sectional equity rankings**
- Evaluate predictive signal quality using **time-series aware validation**
- Support reproducible research pipelines for factor modeling
- Enable modular experimentation across ML, NLP, and statistical models

### Non-Objectives

- Direct trade execution or brokerage integration
- Guaranteed return prediction
- Real-time low-latency trading systems
- Regulatory-compliant advisory output

---

## 3. System Architecture

### 3.1 High-Level Components

```
Data Layer → Feature Engineering → Model Layer → Signal Aggregation → Portfolio Simulation → Reporting Layer
```

---

### 3.2 Core Modules

#### (A) Data Ingestion Layer

Sources:
- Yahoo Finance (`yfinance`)
- Optional S&P 500 constituent universe (dynamic or fallback snapshot)
- External news feeds (NYT, Economist RSS metadata only)

Responsibilities:
- OHLCV retrieval
- Universe construction
- Data normalization and alignment
- Missing data handling

---

#### (B) Feature Engineering Layer

Feature categories:

- Price-based features (returns, volatility, momentum)
- Cross-sectional normalization (z-scores, ranks)
- Rolling statistical features
- Optional sentiment features from NLP pipeline

Output:
- Model-ready feature tensor indexed by (time, asset)

---

#### (C) Model Layer

Ensemble architecture:

1. Linear Models
   - Ridge regression (baseline factor model)

2. Tree-Based Models
   - Random Forest
   - Gradient Boosting
   - Histogram-based Gradient Boosting

3. Neural Models
   - Multi-layer perceptron regressor

4. Kernel Approximation Models
   - Random Fourier Features (RFF) + Ridge regression

5. Ensemble Aggregation
   - Stacked meta-model (`advanced_ensemble`)

All models operate under **cross-sectional regression objective**:
\[
y_{i,t} = f(X_{i,t}) + \epsilon
\]

where target is relative return or forward-ranked performance.

---

#### (D) Acceleration Layer

Compute backends:

- CPU: NumPy (baseline)
- GPU (optional):
  - CuPy for array operations
  - CUDA-enabled acceleration (environment-dependent)
- JIT:
  - Numba for vectorized numerical routines

Selection is runtime adaptive based on hardware availability.

---

#### (E) NLP & Sentiment Layer

Submodules:

- TF-IDF vectorization
- Logistic regression sentiment classifier
- Lexicon fallback sentiment scoring
- Text summarization (lightweight extractive methods)

Outputs:
- Sentiment score per asset
- Aggregated thematic signals
- Augmented feature vectors for model layer

Constraint:
- No full-text proprietary data redistribution (metadata only)

---

#### (F) Signal Aggregation Layer

Combines outputs from:

- ML ranking models
- Sentiment signals
- Cross-model ensemble weighting

Produces:
- Cross-sectional score per asset per time step
- Normalized rank ordering

---

#### (G) Portfolio Simulation Engine

Backtesting methodology:

- Walk-forward validation (strict temporal separation)
- Long/short portfolio construction
- Transaction cost modeling
- Turnover constraints
- Rebalancing simulation

Metrics:

- Information Coefficient (IC)
- Rank correlation stability
- Portfolio Sharpe (simulation-based)
- Drawdown statistics
- Turnover rates

---

#### (H) Reporting Layer

Outputs:

- Structured performance reports
- Ranked asset lists
- Model comparison summaries
- Optional PDF export for research documentation

---

## 4. Interfaces

### 4.1 API Service (FastAPI)

Purpose:
Programmatic access to ranking, simulation, and analytics.

Endpoints:

- `/sp500/simulate`
- `/predict_from_tickers`
- `/health`

Authentication:
- Bearer token required (`API_TOKEN`)

---

### 4.2 Research Dashboard (Streamlit)

Purpose:
Interactive exploration of rankings and simulation outputs.

Characteristics:
- Password-protected
- Non-production interface
- Intended for internal research use only

---

### 4.3 CLI Interface

Supports:

- batch ticker ranking
- full universe simulation
- offline/backtest execution
- report generation

Designed for reproducible research workflows.

---

## 5. Data Model

### Primary Entity

```
Asset-Time Feature Tensor
(index: time, asset)
```

Fields:

- OHLCV features
- engineered signals
- sentiment scores (optional)
- target variable (forward return or rank)

---

## 6. Security Model

### 6.0 Cost & data safety (no billing risk)

- **No paid third-party APIs are used.** Market data: `yfinance` (free, no key).
  News evidence: public NYT/Economist RSS feeds (free, no key). Sentiment: a
  local lexicon — no external LLM/inference service is called.
- The only secrets (`ML_EQUITY_API_TOKEN`, `ML_EQUITY_DASHBOARD_PASSWORD`) are
  **local access controls**; they are never transmitted to any third party and
  incur no charges.
- The optional `NEWS_API_KEY` / `MARKET_DATA_API_KEY` slots in `.env.example` are
  **unused by default** and only relevant if you deliberately swap in a keyed
  provider.

---

### 6.1 Authentication

- API access controlled via bearer token (`Authorization: Bearer <token>`).
- Dashboard protected via password gate.
- Tokens compared in constant time (`hmac.compare_digest`).

---

### 6.2 Localhost-first, not exposed by default

- `make dashboard` / `make api` bind to **`127.0.0.1`** (loopback) via
  `launch.sh`, so the services are reachable only from your own machine — not
  your LAN or the internet.
- To intentionally expose on your network, override the bind address
  (`make dashboard HOST=0.0.0.0`); `launch.sh` warns when you do.
- For any real deployment, put the service behind an HTTPS reverse proxy and set
  strong secrets first.

---

### 6.3 Secret Management

Required environment variables:

- `ML_EQUITY_API_TOKEN`
- `ML_EQUITY_DASHBOARD_PASSWORD`

Guidance:
- Run `make secrets` to generate strong random values into a gitignored `.env`
  (written with `chmod 600`). `launch.sh` auto-loads `.env`.
- `.env` is gitignored — never commit real secrets.
- The dashboard and `/health` warn when the insecure default credential
  (`dev-change-me`) is still in use; replace it before exposing the service.

---

## 7. Performance Considerations

- Vectorized computation preferred (NumPy / CuPy)
- Avoid Python-level loops in feature computation
- GPU acceleration optional and non-essential
- Walk-forward backtesting is computationally dominant workload

---

## 8. Limitations

- Model outputs are statistical estimations, not predictions
- Non-stationarity of financial markets limits predictive stability
- Performance is highly sensitive to:
  - feature selection
  - training window
  - regime shifts
- NLP signals are auxiliary and noisy

---

## 9. Extensibility

System is designed to support:

- Alternative asset classes (crypto, FX, commodities)
- Alternative factor models (Fama-French extensions)
- Reinforcement learning portfolio policies
- Alternative data integration (news APIs, filings, macro signals)

---

## 10. Compliance Statement

This system is intended solely for:
- academic research
- quantitative experimentation
- simulation-based analysis

It does not constitute financial advice, investment recommendations, or trading signals for execution.
