"""GET /freshness — how stale is the news corpus?"""

from __future__ import annotations

from fastapi import APIRouter

from backend.models import FreshnessResponse
from src.orchestrator.freshness import freshness

router = APIRouter(tags=["freshness"])


@router.get("/freshness", response_model=FreshnessResponse)
async def get_freshness() -> FreshnessResponse:
    f = freshness()
    return FreshnessResponse(
        ran_at=f.get("ran_at"),
        rows_added=f.get("rows_added"),
        stocks_updated=list(f.get("stocks_updated", []) or []),
        status=str(f.get("status", "unknown")),
        age_hours=f.get("age_hours"),
        label=str(f.get("label", "")),
    )
