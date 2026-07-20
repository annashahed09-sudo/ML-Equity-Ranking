"""
API routes for the quantitative equity research platform.

Provides endpoints for:
- Health and status
- Ranking and prediction
- Factor analysis
- Portfolio optimization
- Risk analysis
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from config import settings
from core.exceptions import QuantsError

router = APIRouter()
bearer_scheme = HTTPBearer(auto_error=False)


def require_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)):
    """Require valid authentication for protected endpoints."""
    if credentials is None or credentials.credentials != settings.API_TOKEN:
        if settings.is_production():
            raise HTTPException(status_code=401, detail="Invalid or missing authentication")
    return True


# ── Pydantic Models ───────────────────────────────────────────────────────

class OHLCVRow(BaseModel):
    date: str
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class RankingRequest(BaseModel):
    rows: List[OHLCVRow] = Field(..., min_length=50)
    model_type: str = "ridge"
    factors: Optional[List[str]] = None


class SP500SimulationRequest(BaseModel):
    start_date: str
    end_date: str
    model_type: str = "ridge"
    limit: int = Field(50, ge=5, le=500)
    n_splits: int = Field(5, ge=1, le=10)


class PortfolioRequest(BaseModel):
    expected_returns: Dict[str, float]
    returns: Dict[str, List[float]]
    method: str = "max_sharpe"  # max_sharpe, min_variance, risk_parity


# ── Routes ────────────────────────────────────────────────────────────────

@router.get("/status")
def status():
    return {"status": "running", "version": "2.0.0"}


@router.post("/rank", dependencies=[Depends(require_auth)])
def rank_tickers(request: RankingRequest):
    """
    Rank tickers using cross-sectional model.
    
    Accepts OHLCV data and returns ranked tickers with scores.
    """
    try:
        df = pd.DataFrame([r.model_dump() for r in request.rows])
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        
        from research.pipeline import ResearchPipeline
        
        pipeline = ResearchPipeline()
        result = pipeline.run(
            raw_df=df,
            model_type=request.model_type,
            factor_names=request.factors,
            n_splits=2,
        )
        
        # Get latest ranking
        latest_date = result.predictions["date"].max()
        latest = result.predictions[result.predictions["date"] == latest_date]
        latest = latest.sort_values("model_score", ascending=False)
        latest["rank"] = range(1, len(latest) + 1)
        
        return {
            "ranking": latest[["rank", "ticker", "model_score"]].to_dict(orient="records"),
            "fold_metrics": result.fold_metrics.to_dict(orient="records"),
            "portfolio_summary": result.portfolio_summary,
        }
    except QuantsError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/simulate/sp500", dependencies=[Depends(require_auth)])
def simulate_sp500(request: SP500SimulationRequest):
    """Run S&P 500 simulation with walk-forward validation."""
    try:
        from data.loader import DataLoader
        
        loader = DataLoader()
        df = loader.load_tickers(
            tickers=loader.get_sp500_universe(limit=request.limit),
            start_date=request.start_date,
            end_date=request.end_date,
        )
        
        from research.pipeline import ResearchPipeline
        
        pipeline = ResearchPipeline()
        result = pipeline.run(
            raw_df=df,
            model_type=request.model_type,
            n_splits=request.n_splits,
        )
        
        latest_date = result.predictions["date"].max()
        latest = result.predictions[result.predictions["date"] == latest_date]
        latest = latest.sort_values("model_score", ascending=False)
        latest["rank"] = range(1, len(latest) + 1)
        
        return {
            "universe_size": df["ticker"].nunique(),
            "ranking": latest[["rank", "ticker", "model_score"]].to_dict(orient="records"),
            "fold_metrics": result.fold_metrics.to_dict(orient="records"),
            "portfolio_summary": result.portfolio_summary,
        }
    except QuantsError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/optimize/portfolio", dependencies=[Depends(require_auth)])
def optimize_portfolio(request: PortfolioRequest):
    """Optimize portfolio weights using specified method."""
    try:
        expected_returns = pd.Series(request.expected_returns)
        returns = pd.DataFrame(request.returns)
        
        if request.method == "max_sharpe":
            from portfolio.mean_variance import MeanVarianceOptimizer
            optimizer = MeanVarianceOptimizer()
            weights = optimizer.max_sharpe(expected_returns, returns)
        elif request.method == "min_variance":
            from portfolio.min_variance import MinimumVarianceOptimizer
            optimizer = MinimumVarianceOptimizer()
            weights = optimizer.minimize_volatility(returns)
        elif request.method == "risk_parity":
            from portfolio.risk_parity import RiskParityOptimizer
            optimizer = RiskParityOptimizer()
            weights = optimizer.equal_risk_contribution(returns)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")
        
        return {
            "weights": weights.to_dict(),
            "method": request.method,
        }
    except QuantsError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/factors", dependencies=[Depends(require_auth)])
def list_factors():
    """List available factors."""
    from factors.factory import FactorCatalog
    catalog = FactorCatalog.create_default()
    return {
        "factors": [
            {
                "name": name,
                "type": factor.factor_type.value,
            }
            for name, factor in catalog.factors.items()
        ]
    }
