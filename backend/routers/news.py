"""GET /stocks/{ticker}/news?days=7

Wraps Phase 10's impact_reader.read_recent_impact. Days is clamped
between 1 and 30 by FastAPI's Query validation.
"""
from fastapi import APIRouter, Query

from backend.models import NewsItem, NewsOut
from src.orchestrator.impact_reader import read_recent_impact

router = APIRouter(prefix="/stocks")


@router.get("/{ticker}/news", response_model=NewsOut)
def get_news(
    ticker: str,
    days: int = Query(7, ge=1, le=30,
                      description="Lookback window in days (1..30)"),
) -> NewsOut:
    """Return Phase 9 news_impact rows for `ticker` from the last `days` days."""
    code = ticker.upper().strip()
    rows = read_recent_impact(code, days=days)
    return NewsOut(
        ticker=code, days=days,
        items=[NewsItem(**r) for r in rows],
    )