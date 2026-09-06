"""LLM provider settings — GET to view, POST to update."""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter

from backend.models import SettingsResponse, SettingsUpdate
from backend import settings_store
from backend.db import reload_orchestrator_with_settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["settings"])


# Module-level cache of "last applied" values so we only rebuild the
# orchestrator when something actually changed (otherwise a no-op POST
# would clobber a live provider with stub).
_LAST_APPLIED: dict = {}


@router.get("/settings", response_model=SettingsResponse)
async def get_settings() -> SettingsResponse:
    return SettingsResponse(**settings_store.read_settings())


@router.post("/settings", response_model=SettingsResponse)
async def post_settings(update: SettingsUpdate) -> SettingsResponse:
    global _LAST_APPLIED
    updates = update.model_dump(exclude_none=True)
    settings_store.update_settings(updates)

    cfg = settings_store.read_settings()
    real_key = _read_real_key() or _key_from_env(cfg["provider"])

    # Only rebuild if provider / model / key actually differ from what we
    # last applied — otherwise a no-op POST could drop a live session
    # back to stub just because the user re-sent the same settings.
    signature = (cfg["provider"], cfg["model"], cfg["base_url"], bool(real_key))
    if signature == _LAST_APPLIED:
        return SettingsResponse(**cfg)

    if cfg["provider"] != "stub" and not real_key:
        # Missing key — don't downgrade to stub; keep current singleton.
        logger.warning("Cannot switch to %s: no API key in .env or settings",
                       cfg["provider"])
        return SettingsResponse(**cfg)

    try:
        reload_orchestrator_with_settings(
            provider=cfg["provider"],
            model=cfg["model"],
            base_url=cfg["base_url"],
            api_key=real_key,
        )
        _LAST_APPLIED = signature
    except Exception as e:
        logger.exception("Orchestrator rebuild failed: %s", e)
    return SettingsResponse(**cfg)


def _read_real_key() -> str:
    """Return the actual (unmasked) API key stored on disk."""
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / ".settings.json"
    if not path.exists():
        return ""
    try:
        return str(json.loads(path.read_text()).get("api_key", "") or "")
    except Exception:
        return ""


def _key_from_env(provider: str) -> str:
    """Read {PROVIDER}_API_KEY from process env (typically populated by .env)."""
    if provider == "gemini":
        return os.getenv("GOOGLE_API_KEY", "") or ""
    return os.getenv(f"{provider.upper()}_API_KEY", "") or ""
