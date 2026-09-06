"""GET /freshness — combines Phase 8's _last_refresh.json with Phase 10's
freshness() helper. Single endpoint that tells the frontend "is the
model data fresh?".
"""
import json

from fastapi import APIRouter

from backend.models import FreshnessOut
from src.orchestrator.freshness import REFRESH_FILE, freshness

router = APIRouter()


def _label(status: str, age_hours: float | None) -> str:
    """Human-readable one-liner used by the UI."""
    if status == "ok" and age_hours is not None:
        return f"fresh (refreshed {age_hours}h ago)"
    if status == "stale" and age_hours is not None:
        return f"stale (last refreshed {age_hours}h ago)"
    if status == "unknown":
        return "unknown (no refresh log yet)"
    return f"broken ({status})"


@router.get("/freshness", response_model=FreshnessOut)
def get_freshness() -> FreshnessOut:
    """Compose the freshness payload. Always returns 200; an absent or
    unparseable refresh log is signalled via `status="unknown"/"broken"`
    rather than a 5xx error."""
    f = freshness()
    ran_at: str | None = None
    rows_added: int | None = None
    stocks_updated: list[str] = []
    if REFRESH_FILE.exists():
        try:
            payload = json.loads(REFRESH_FILE.read_text())
            ran_at = payload.get("ran_at")
            rows_added = payload.get("rows_added_total")
            stocks_updated = list(payload.get("stocks_updated") or [])
        except (json.JSONDecodeError, OSError):
            pass

    status = f.get("status", "unknown")
    return FreshnessOut(
        ran_at=ran_at,
        rows_added=rows_added,
        stocks_updated=stocks_updated,
        status=status,
        age_hours=f.get("age_hours"),
        label=_label(status, f.get("age_hours")),
    )