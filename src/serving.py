"""FastAPI service for interactive market intelligence usage."""

from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .features import compute_features, compute_forward_returns
from .market_intelligence import MarketIntelligenceService
from .models import create_model


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
    tickers: List[str] = Field(..., min_length=1)
    start_date: str
    end_date: str
    model_type: str = "advanced_ensemble"
    reviews_by_ticker: Optional[Dict[str, List[str]]] = None


app = FastAPI(title="ML Equity Intelligence API", version="1.0.1")
service = MarketIntelligenceService()


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
        raise HTTPException(status_code=400, detail="Need at least 2 tickers for cross-sectional ranking.")
    return df


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict_from_rows")
def predict_from_rows(payload: PredictRequest) -> Dict:
    _validate_model_type(payload.model_type)
    df = _prepare_rows_df(payload.rows)

    feat = compute_features(df)
    feat = compute_forward_returns(feat).dropna().reset_index(drop=True)
    if feat.empty:
        raise HTTPException(status_code=400, detail="Not enough rows after feature engineering. Increase history.")

    ranking = service.rank_tickers(
        feat,
        model_type=payload.model_type,
        model_kwargs={"prefer_gpu": True, "prefer_numba": True},
    )
    report = service.build_market_report(ranking, payload.reviews_by_ticker)

    return {
        "ranking": ranking.to_dict(orient="records"),
        "report": report.to_dict(orient="records"),
    }


@app.post("/predict_from_tickers")
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
        "ranking": ranking.to_dict(orient="records"),
        "report": report.to_dict(orient="records"),
    }
