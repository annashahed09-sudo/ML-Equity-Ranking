from __future__ import annotations

from typing import Generator, Optional

from app.core.config import settings

try:
    from sqlalchemy import create_engine
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
except Exception:  # pragma: no cover - optional dependency fallback
    create_engine = None
    Engine = object  # type: ignore[assignment]
    Session = object  # type: ignore[assignment]
    sessionmaker = None  # type: ignore[assignment]

    class DeclarativeBase:  # type: ignore[no-redef]
        pass


class Base(DeclarativeBase):
    pass


_engine: Optional[Engine] = None
SessionLocal = None


def get_engine() -> Optional[Engine]:
    global _engine, SessionLocal

    if _engine is not None:
        return _engine

    if create_engine is None or sessionmaker is None:
        return None

    try:
        _engine = create_engine(
            settings.database_url,
            pool_pre_ping=True,
        )
        SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=_engine,
        )
    except Exception:  # pragma: no cover - keep app startup resilient
        _engine = None
        SessionLocal = None

    return _engine


def get_session_local():
    if SessionLocal is None:
        get_engine()
    return SessionLocal


def get_db() -> Generator[Optional[Session], None, None]:
    session_local = get_session_local()
    if session_local is None:
        yield None
        return

    db = session_local()
    try:
        yield db
    finally:
        db.close()