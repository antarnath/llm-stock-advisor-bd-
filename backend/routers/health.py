"""GET /healthz — liveness probe."""

from __future__ import annotations

from fastapi import APIRouter

from backend.models import FreshnessResponse

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@router.get("/")
async def root() -> dict:
    return {
        "name": "DSE Advisor API",
        "version": "1.0.0",
        "endpoints": [
            "POST /ask",
            "GET /stocks/{ticker}",
            "GET /stocks/{ticker}/prediction",
            "GET /stocks/{ticker}/news",
            "GET /stocks/{ticker}/xai",
            "POST /portfolio/optimize",
            "GET /settings",
            "POST /settings",
            "GET /freshness",
            "POST /news/refresh",
            "GET /healthz",
        ],
    }
