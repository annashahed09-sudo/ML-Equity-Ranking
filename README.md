# Cross-Sectional Equity Return Prediction with Machine Learning

## Overview

This project is now a **secured market-intelligence portal** for research-grade equity ranking and S&P 500 simulations. It combines:

- advanced cross-sectional ML ranking models,
- quantum-inspired and accelerated computing paths,
- NLP review analysis,
- a secured FastAPI service,
- a password-protected Streamlit portal,
- a CLI workflow for repeatable simulations.

It produces **relative ranking signals and backtest diagnostics**. It is not financial advice and cannot guarantee market prediction accuracy.

---

## What’s included

### Advanced ML
- Ridge, Gradient Boosting, Random Forest, HistGradientBoosting, Neural MLP
- Quantum-inspired model (RFF + Ridge)
- Advanced stacked ensemble

### S&P 500 simulation
- Loads live S&P 500 constituents from Wikipedia when available
- Falls back to a stable built-in S&P 500 sample for offline/demo use
- Downloads market data through `yfinance`
- Runs walk-forward model training and long/short portfolio simulation
- Returns latest ranking, fold metrics, and portfolio summary

### Accelerated computing
- NumPy default backend
- Optional CuPy GPU backend
- Optional Numba JIT accelerated routines
- Optional explicit CUDA kernel path for score normalization

### NLP + market intelligence
- Financial sentiment analyzer (TF-IDF + Logistic Regression + lexicon fallback)
- Review summarization and confidence scoring
- Unified ticker report: rank, score, expected direction, review sentiment

### Product interfaces
- **Secured FastAPI service** (`src/serving.py`)
- **Password-protected Streamlit portal** (`dashboard.py`)
- **CLI tool** (`src/cli.py`)
- Launch helper script (`launch.sh`)

---

## Install

```bash
pip install -r requirements.txt
```

---

## Security configuration

Set these before running in any shared environment:

```bash
export ML_EQUITY_API_TOKEN="replace-with-a-long-random-token"
export ML_EQUITY_DASHBOARD_PASSWORD="replace-with-a-long-random-password"
```

If these are not set, the app uses a development fallback (`dev-change-me`) and the dashboard displays a warning.

---

## Run tests

```bash
pytest tests/ -q
```

---

## Launch as a secured tool

### 1) Dashboard portal

```bash
./launch.sh dashboard
```

Open: `http://localhost:8501`

### 2) API service

```bash
./launch.sh api
```

Open: `http://localhost:8000/docs`

Use the API token as a bearer token:

```bash
curl -H "Authorization: Bearer $ML_EQUITY_API_TOKEN" http://localhost:8000/health
```

### 3) CLI custom ticker ranking

```bash
python -m src.cli --tickers AAPL,MSFT,NVDA,AMZN --start 2021-01-01 --end 2024-12-31 --model advanced_ensemble
```

### 4) CLI S&P 500 simulation

```bash
python -m src.cli --sp500 --sp500-limit 25 --start 2021-01-01 --end 2024-12-31 --model advanced_ensemble
```

For offline/demo mode:

```bash
python -m src.cli --sp500 --offline-sp500 --sp500-limit 10 --start 2021-01-01 --end 2024-12-31 --model ridge
```

---

## API quick examples

### `POST /sp500/simulate`

```json
{
  "start_date": "2021-01-01",
  "end_date": "2024-12-31",
  "model_type": "advanced_ensemble",
  "limit": 25,
  "n_splits": 3,
  "use_live_wikipedia": true
}
```

### `POST /predict_from_tickers`

```json
{
  "tickers": ["AAPL", "MSFT", "NVDA", "AMZN"],
  "start_date": "2021-01-01",
  "end_date": "2024-12-31",
  "model_type": "advanced_ensemble",
  "reviews_by_ticker": {
    "AAPL": ["Strong growth and earnings beat."]
  }
}
```

---

## Important legitimacy notes

- This tool provides model-based rankings, simulations, and sentiment context; it is **not financial advice**.
- No system can predict markets with perfect accuracy.
- Simulation results depend on data quality, model settings, date range, universe selection, and market regime.
- Always apply risk controls, diversification, position sizing, and independent validation.
- Best use: research workflow, scenario analysis, and signal triage.
