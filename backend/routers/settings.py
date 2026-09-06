"""LLM provider settings — GET to view, POST to update."""

from __future__ import annotations

from fastapi import APIRouter

from backend.models import SettingsResponse, SettingsUpdate
from backend import settings_store
from backend.db import reload_orchestrator_with_settings

router = APIRouter(tags=["settings"])


@router.get("/settings", response_model=SettingsResponse)
async def get_settings() -> SettingsResponse:
    return SettingsResponse(**settings_store.read_settings())


@router.post("/settings", response_model=SettingsResponse)
async def post_settings(update: SettingsUpdate) -> SettingsResponse:
    updates = update.model_dump(exclude_none=True)
    settings_store.update_settings(updates)

    # Rebuild the orchestrator singleton so the next /ask uses the new provider
    cfg = settings_store.read_settings()
    if cfg["has_key"] or cfg["provider"] == "stub":
        try:
            reload_orchestrator_with_settings(
                provider=cfg["provider"],
                model=cfg["model"],
                base_url=cfg["base_url"],
                api_key=cfg.get("_real_api_key", "") if False else
                        _read_real_key(),
            )
        except Exception:
            # Settings are saved even if rebuild fails
            pass
    return SettingsResponse(**cfg)


def _read_real_key() -> str:
    """Return the actual (unmasked) API key for orchestrator rebuild."""
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / ".settings.json"
    if not path.exists():
        return ""
    try:
        return str(json.loads(path.read_text()).get("api_key", "") or "")
    except Exception:
        return ""
