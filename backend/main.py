from fastapi import FastAPI

from app.api.routes import router as api_router
from app.core.config import settings


app = FastAPI(title=settings.project_name)
app.include_router(api_router, prefix=settings.api_v1_str)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "running"}