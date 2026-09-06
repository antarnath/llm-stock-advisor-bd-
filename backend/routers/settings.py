"""GET / POST / DELETE /settings — runtime provider config (Phase 13).

GET    : returns the current effective settings (api_key masked).
POST   : validates and persists a new record; mirrors it into os.environ
         so the very next /ask call sees the new provider without a restart.
DELETE : removes the on-disk override file. Does NOT wipe os.environ —
         the user may have started the backend with .env values they want
         to keep; the override only takes effect when the file exists.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.models import SettingsIn, SettingsOut
from backend.settings_store import (
    apply_to_environ,
    delete_settings,
    mask_key,
    read_settings,
    write_settings,
)

router = APIRouter()


@router.get("/settings", response_model=SettingsOut)
def get_settings() -> SettingsOut:
    """Return the persisted settings with api_key masked.

    If no file exists yet, returns an empty record (provider="", etc.)
    so the frontend can render an "empty form" state.
    """
    return SettingsOut(**mask_key(read_settings()))


@router.post("/settings", response_model=SettingsOut)
def post_settings(payload: SettingsIn) -> SettingsOut:
    """Validate, persist, and propagate to os.environ."""
    try:
        record = write_settings(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    apply_to_environ()
    return SettingsOut(**record)


@router.delete("/settings", response_model=SettingsOut)
def reset_settings() -> SettingsOut:
    """Remove the override file. The next /ask falls back to .env / stub."""
    delete_settings()
    return SettingsOut(provider="", model="", base_url="", api_key="", has_key=False)
