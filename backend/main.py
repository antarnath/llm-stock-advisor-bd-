"""FastAPI app entry point — DSE Advisor backend.

Usage:
    .venv/bin/uvicorn backend.main:app --reload --port 8000

Swagger UI: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.utils.config import ensure_dirs

# Routers
from backend.routers import (
    ask as ask_router,
    freshness as freshness_router,
    health as health_router,
    news as news_router,
    portfolio as portfolio_router,
    settings as settings_router,
    stocks as stocks_router,
    xai as xai_router,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create dirs once at startup; release on shutdown."""
    ensure_dirs()
    logger.info("Backend ready: dirs ensured, routers mounted.")
    yield
    logger.info("Backend shutting down.")


app = FastAPI(
    title="DSE Advisor API",
    description=(
        "Backend for the LLM-Orchestrated Financial Advisor for the "
        "Bangladesh Stock Exchange. Wires the multi-agent orchestrator "
        "to an HTTP API."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Permissive CORS for local dev (frontend at :3000, docs in browser, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all routers
app.include_router(health_router.router)
app.include_router(ask_router.router)
app.include_router(freshness_router.router)
app.include_router(stocks_router.router)
app.include_router(portfolio_router.router)
app.include_router(settings_router.router)
app.include_router(news_router.router)
app.include_router(xai_router.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
