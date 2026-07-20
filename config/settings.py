"""
Pydantic-based configuration management for the Quantitative Equity Research Platform.

All configuration is validated at startup through environment variables with
sensible defaults for development. Production deployments should override
via .env or environment variables.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelBackend(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"


class ValidationStrategy(str, Enum):
    WALK_FORWARD = "walk_forward"
    PURGED_CV = "purged_cv"
    NESTED_CV = "nested_cv"
    EXPANDING_WINDOW = "expanding_window"


class CovarianceMethod(str, Enum):
    SAMPLE = "sample"
    LEDOIT_WOLF = "ledoit_wolf"
    SHRUNK = "shrunk"
    EWMA = "ewma"


class PortfolioObjective(str, Enum):
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    MINIMUM_VARIANCE = "minimum_variance"
    MAX_DIVERSIFICATION = "max_diversification"
    EQUAL_RISK_CONTRIBUTION = "equal_risk_contribution"
    BLACK_LITTERMAN = "black_litterman"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Environment ---
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    LOG_LEVEL: LogLevel = LogLevel.INFO
    DEBUG: bool = False

    # --- Paths ---
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
    DATA_DIR: Path = Field(default=Path("data"), description="Path to data directory")
    OUTPUT_DIR: Path = Field(default=Path("output"), description="Path to output directory")
    CACHE_DIR: Path = Field(default=Path(".cache"), description="Path to cache directory")

    # --- Data ---
    DEFAULT_START_DATE: str = "2015-01-01"
    DEFAULT_END_DATE: str = "2025-01-01"
    YFINANCE_TIMEOUT: int = 30
    YFINANCE_MAX_RETRIES: int = 3
    DATA_CACHE_ENABLED: bool = True
    DATA_CACHE_TTL_HOURS: int = 24

    # --- Universe ---
    SP500_MAX_TICKERS: int = 100
    DEFAULT_UNIVERSE_SIZE: int = 50

    # --- Factors ---
    FACTOR_WINSORIZATION_LIMITS: tuple[float, float] = (0.01, 0.99)
    FACTOR_NEUTRALIZATION: str = "sector"  # "sector", "market", "none"
    FACTOR_ZSCORE_CROSS_SECTIONAL: bool = True

    # --- Validation ---
    VALIDATION_STRATEGY: ValidationStrategy = ValidationStrategy.WALK_FORWARD
    N_SPLITS: int = 5
    TEST_SIZE: int = 252  # ~1 year of daily data
    MIN_TRAIN_SIZE: int = 756  # ~3 years of daily data
    PURGE_WINDOW: int = 5  # days to purge around test/train boundary
    EMBARGO_WINDOW: int = 5  # days to embargo after test period

    # --- Models ---
    MODEL_BACKEND: ModelBackend = ModelBackend.AUTO
    RANDOM_SEED: int = 42
    N_JOBS: int = -1
    LIGHTGBM_NUM_LEAVES: int = 31
    LIGHTGBM_LEARNING_RATE: float = 0.05
    LIGHTGBM_N_ESTIMATORS: int = 500
    XGBOOST_N_ESTIMATORS: int = 500
    XGBOOST_MAX_DEPTH: int = 6
    XGBOOST_LEARNING_RATE: float = 0.05
    CATBOOST_ITERATIONS: int = 500
    CATBOOST_DEPTH: int = 6
    CATBOOST_LEARNING_RATE: float = 0.05
    ELASTIC_NET_ALPHA: float = 0.5
    ELASTIC_NET_L1_RATIO: float = 0.5

    # --- Hyperparameter Optimization ---
    HP_OPTIMIZATION_ENABLED: bool = True
    HP_N_TRIALS: int = 50
    HP_N_STARTUP_TRIALS: int = 10
    HP_N_EI_CANDIDATES: int = 24
    HP_CV_FOLDS: int = 3

    # --- Portfolio ---
    PORTFOLIO_OBJECTIVE: PortfolioObjective = PortfolioObjective.MEAN_VARIANCE
    TRANSACTION_COST_BPS: float = 10.0
    SLIPPAGE_BPS: float = 5.0
    MAX_POSITION_WEIGHT: float = 0.05
    MAX_SECTOR_EXPOSURE: float = 0.30
    LONG_SHORT_RATIO: float = 1.0
    REBALANCE_FREQUENCY: str = "monthly"  # daily, weekly, monthly, quarterly
    TARGET_LEVERAGE: float = 1.0
    TURNOVER_PENALTY: float = 0.001

    # --- Risk ---
    RISK_FREE_RATE: float = 0.05
    CONFIDENCE_LEVEL: float = 0.95
    VAR_WINDOW: int = 252
    COVARIANCE_METHOD: CovarianceMethod = CovarianceMethod.LEDOIT_WOLF
    COVARIANCE_HALF_LIFE: int = 60

    # --- Sentiment ---
    SENTIMENT_MODEL: str = "FinBERT"  # "FinBERT", "lexicon", "hybrid"
    SENTIMENT_CONFIDENCE_THRESHOLD: float = 0.6
    NEWS_LOOKBACK_DAYS: int = 7

    # --- API ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_TOKEN: str = "change-me-in-production"
    DASHBOARD_PASSWORD: str = "change-me-in-production"
    CORS_ORIGINS: list[str] = ["http://localhost:8501", "http://localhost:3000"]

    # --- Security ---
    TOKEN_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # --- Backtesting ---
    BT_INITIAL_CAPITAL: float = 1_000_000.0
    BT_COMMISSION_PCT: float = 0.001
    BT_MIN_TRADES: int = 10

    # --- Experiment Tracking ---
    EXPERIMENT_TRACKING_ENABLED: bool = True
    EXPERIMENT_OUTPUT_DIR: Path = Field(default=Path("experiments"))

    @field_validator("DATA_DIR", "OUTPUT_DIR", "CACHE_DIR", mode="before")
    @classmethod
    def resolve_paths(cls, v: str | Path) -> Path:
        if isinstance(v, str):
            v = Path(v)
        if not v.is_absolute():
            v = Path(__file__).resolve().parents[1] / v
        v.mkdir(parents=True, exist_ok=True)
        return v

    def is_production(self) -> bool:
        return self.ENVIRONMENT == Environment.PRODUCTION

    def is_development(self) -> bool:
        return self.ENVIRONMENT == Environment.DEVELOPMENT


# Global singleton
settings = Settings()
