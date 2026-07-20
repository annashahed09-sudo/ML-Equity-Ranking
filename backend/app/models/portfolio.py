from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

try:
    from sqlalchemy import DateTime, Float, Integer, String
    from sqlalchemy.orm import Mapped, mapped_column

    from app.db.session import Base
except Exception:  # pragma: no cover - optional dependency fallback

    @dataclass
    class Portfolio:
        id: int | None = None
        name: str = ""
        description: str | None = None
        initial_capital: float = 100000.0
        created_at: datetime = field(default_factory=datetime.utcnow)
        updated_at: datetime = field(default_factory=datetime.utcnow)

else:

    class Portfolio(Base):
        __tablename__ = "portfolios"

        id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
        name: Mapped[str] = mapped_column(String(120), nullable=False, unique=True, index=True)
        description: Mapped[str | None] = mapped_column(String(500), nullable=True)
        initial_capital: Mapped[float] = mapped_column(Float, nullable=False, default=100000.0)
        created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
        updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)