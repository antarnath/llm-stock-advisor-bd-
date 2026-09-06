"""DSE Advisor API — Phase 11 thin FastAPI service.

Run:
    .venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000

CORS open for http://localhost:3000 (Next.js dev server, Phase 12).
For a real deployment, tighten allow_origins — that's v2.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.db import init_db
from backend.routers import ask, freshness, health, news, settings, stocks
from backend.settings_store import apply_to_environ
from src.utils.config import ensure_dirs


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ensure dirs + chat_log table exist + apply saved LLM
    settings to os.environ so the very first /ask picks them up. Shutdown:
    nothing."""
    ensure_dirs()
    init_db()
    apply_to_environ()
    yield


app = FastAPI(
    title="DSE Advisor API",
    version="1.0.0",
    description=(
        "Thin FastAPI wrapper around Phase 10's Orchestrator. "
        "Designed for the Phase 12 frontend."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

app.include_router(health.router)
app.include_router(stocks.router)
app.include_router(news.router)
app.include_router(freshness.router)
app.include_router(ask.router)
app.include_router(settings.router)


@app.get("/", include_in_schema=False)
def root() -> dict:
    """Tiny index — points at the built-in OpenAPI docs (/docs)."""
    return {
        "service":  "DSE Advisor API",
        "version":  "1.0.0",
        "docs":     "/docs",
        "health":   "/healthz",
        "endpoints": [
            "GET  /healthz",
            "GET  /stocks",
            "GET  /stocks/{ticker}/prediction",
            "GET  /stocks/{ticker}/news",
            "GET  /freshness",
            "POST /ask",
            "GET  /settings",
            "POST /settings",
            "DELETE /settings",
        ],
    }