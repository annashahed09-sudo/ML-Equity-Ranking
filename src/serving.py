"""FastAPI service for secured market intelligence usage."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .features import compute_features, compute_forward_returns
from .market_intelligence import MarketIntelligenceService
from .models import create_model
from .security import get_security_settings, token_is_valid
from .sp500 import _json_records, run_sp500_simulation


class OhlcvRow(BaseModel):
    date: str
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class PredictRequest(BaseModel):
    rows: List[OhlcvRow] = Field(..., min_length=50)
    model_type: str = "advanced_ensemble"
    reviews_by_ticker: Optional[Dict[str, List[str]]] = None


class TickerRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=2)
    start_date: str
    end_date: str
    model_type: str = "advanced_ensemble"
    reviews_by_ticker: Optional[Dict[str, List[str]]] = None


class SP500SimulationRequest(BaseModel):
    start_date: str
    end_date: str
    model_type: str = "advanced_ensemble"
    limit: int = Field(25, ge=2, le=100)
    n_splits: int = Field(3, ge=1, le=10)
    test_size: Optional[int] = Field(None, ge=1)
    min_train_size: Optional[int] = Field(None, ge=1)
    use_yahoo_screener: bool = True
    include_news: bool = True
    reviews_by_ticker: Optional[Dict[str, List[str]]] = None


app = FastAPI(title="ML Equity Intelligence API", version="1.1.0")
service = MarketIntelligenceService()
bearer_scheme = HTTPBearer(auto_error=False)


def require_api_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> None:
    """Require a bearer token for non-health API routes."""
    settings = get_security_settings()
    if credentials is None or not token_is_valid(credentials.credentials, settings.api_token):
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token.")


def _validate_model_type(model_type: str) -> None:
    try:
        create_model(model_type, prefer_gpu=False, prefer_numba=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unknown model_type '{model_type}'") from exc


def _prepare_rows_df(rows: List[OhlcvRow]) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump() for r in rows])
    try:
        df["date"] = pd.to_datetime(df["date"], errors="raise")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid date format in rows: {exc}") from exc

    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    if df["ticker"].nunique() < 2:
        raise HTTPException(
            status_code=400, detail="Need at least 2 tickers for cross-sectional ranking."
        )
    return df


@app.get("/health")
def health() -> Dict[str, str]:
    settings = get_security_settings()
    return {
        "status": "ok",
        "auth": "default-dev-token" if settings.using_default_token else "configured",
    }


@app.post("/predict_from_rows", dependencies=[Depends(require_api_token)])
def predict_from_rows(payload: PredictRequest) -> Dict:
    _validate_model_type(payload.model_type)
    df = _prepare_rows_df(payload.rows)

    feat = compute_features(df)
    feat = compute_forward_returns(feat).dropna().reset_index(drop=True)
    if feat.empty:
        raise HTTPException(
            status_code=400, detail="Not enough rows after feature engineering. Increase history."
        )

    ranking = service.rank_tickers(
        feat,
        model_type=payload.model_type,
        model_kwargs={"prefer_gpu": True, "prefer_numba": True},
    )
    report = service.build_market_report(ranking, payload.reviews_by_ticker)

    return {
        "ranking": _json_records(ranking),
        "report": _json_records(report),
    }


@app.post("/predict_from_tickers", dependencies=[Depends(require_api_token)])
def predict_from_tickers(payload: TickerRequest) -> Dict:
    _validate_model_type(payload.model_type)
    if len(set([t.strip().upper() for t in payload.tickers if t.strip()])) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 unique tickers.")

    try:
        feat = service.prepare_data(payload.tickers, payload.start_date, payload.end_date)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Data loading failed: {exc}") from exc

    ranking = service.rank_tickers(
        feat,
        model_type=payload.model_type,
        model_kwargs={"prefer_gpu": True, "prefer_numba": True},
    )
    report = service.build_market_report(ranking, payload.reviews_by_ticker)

    return {
        "ranking": _json_records(ranking),
        "report": _json_records(report),
    }


@app.post("/sp500/simulate", dependencies=[Depends(require_api_token)])
def simulate_sp500(payload: SP500SimulationRequest) -> Dict:
    """Run a secured S&P 500 ranking/backtest simulation."""
    _validate_model_type(payload.model_type)
    try:
        result = run_sp500_simulation(
            start_date=payload.start_date,
            end_date=payload.end_date,
            model_type=payload.model_type,
            limit=payload.limit,
            n_splits=payload.n_splits,
            test_size=payload.test_size,
            min_train_size=payload.min_train_size,
            reviews_by_ticker=payload.reviews_by_ticker,
            use_yahoo_screener=payload.use_yahoo_screener,
            include_news=payload.include_news,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"S&P 500 simulation failed: {exc}") from exc

    return result.to_dict()
