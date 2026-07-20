"""
FastAPI application factory with middleware, security, and routes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Quantitative Equity Research Platform",
        version="2.0.0",
        description="Institutional-grade cross-sectional equity ranking, "
                    "portfolio optimization, and risk analysis API.",
        docs_url="/docs" if settings.is_development() else None,
        redoc_url="/redoc" if settings.is_development() else None,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register routes
    from .routes import router
    app.include_router(router, prefix="/api/v1")
    
    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "version": "2.0.0",
            "environment": settings.ENVIRONMENT.value,
        }
    
    return app


app = create_app()
