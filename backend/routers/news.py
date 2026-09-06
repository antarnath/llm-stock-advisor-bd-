"""POST /news/refresh — trigger a news ingestion pass.

This is a stub for now: the real pipeline (scripts/build_news_index.py)
takes minutes. We expose an endpoint so the UI can trigger it; the actual
implementation can be wired up later via subprocess.run(...) or a Celery job.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from backend.models import RefreshResponse

router = APIRouter(tags=["news"])


@router.post("/news/refresh", response_model=RefreshResponse)
async def refresh_news() -> RefreshResponse:
    """Returns a stub response — wire to scripts/build_news_index.py later."""
    return RefreshResponse(
        triggered=False,
        message=(
            "News refresh is not yet wired up via API. "
            f"Run `.venv/bin/python scripts/build_news_index.py` manually. "
            f"(received at {datetime.now(timezone.utc).isoformat()})"
        ),
    )
