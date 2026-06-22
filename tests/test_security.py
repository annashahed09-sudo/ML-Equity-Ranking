from src.security import DEFAULT_DEV_TOKEN, get_security_settings, token_is_valid


def test_default_security_settings(monkeypatch):
    monkeypatch.delenv("ML_EQUITY_API_TOKEN", raising=False)
    monkeypatch.delenv("ML_EQUITY_DASHBOARD_PASSWORD", raising=False)
    settings = get_security_settings()
    assert settings.api_token == DEFAULT_DEV_TOKEN
    assert settings.dashboard_password == DEFAULT_DEV_TOKEN
    assert settings.using_default_token
    assert settings.using_default_dashboard_password


def test_token_is_valid():
    assert token_is_valid("abc", "abc")
    assert not token_is_valid("abc", "def")
