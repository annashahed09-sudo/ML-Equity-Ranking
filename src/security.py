"""Security helpers for API and dashboard access control."""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass

DEFAULT_DEV_TOKEN = "dev-change-me"


@dataclass(frozen=True)
class SecuritySettings:
    """Runtime security settings loaded from environment variables."""

    api_token: str
    dashboard_password: str
    using_default_token: bool
    using_default_dashboard_password: bool


def get_security_settings() -> SecuritySettings:
    """Load security settings without side effects."""
    api_token = os.getenv("ML_EQUITY_API_TOKEN", DEFAULT_DEV_TOKEN)
    dashboard_password = os.getenv("ML_EQUITY_DASHBOARD_PASSWORD", DEFAULT_DEV_TOKEN)
    return SecuritySettings(
        api_token=api_token,
        dashboard_password=dashboard_password,
        using_default_token=api_token == DEFAULT_DEV_TOKEN,
        using_default_dashboard_password=dashboard_password == DEFAULT_DEV_TOKEN,
    )


def token_is_valid(candidate: str, expected: str) -> bool:
    """Constant-time token comparison."""
    return hmac.compare_digest(candidate or "", expected or "")
