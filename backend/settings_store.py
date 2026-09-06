"""LLM settings persistence — JSON file at backend/.settings.json.

Tiny key/value store so /settings survives restarts. No auth in dev mode;
the file is gitignored.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Optional

_LOCK = Lock()


def _settings_path() -> Path:
    here = Path(__file__).resolve().parent
    return here / ".settings.json"


_DEFAULTS = {
    "provider": "stub",
    "model": "stub-template-v1",
    "base_url": "",
    "api_key": "",
}


def _mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "•" * len(key)
    return ("•" * (len(key) - 4)) + key[-4:]


def read_settings() -> dict:
    """Return the current settings dict (api_key masked for display)."""
    with _LOCK:
        path = _settings_path()
        if not path.exists():
            return {**_DEFAULTS, "has_key": False}
        try:
            data = json.loads(path.read_text())
        except Exception:
            return {**_DEFAULTS, "has_key": False}
    merged = {**_DEFAULTS, **data}
    masked = _mask_key(merged.get("api_key", "") or "")
    return {
        "provider": merged["provider"],
        "model": merged["model"],
        "base_url": merged["base_url"],
        "api_key": masked,
        "has_key": bool(merged.get("api_key")),
    }


def update_settings(updates: dict) -> dict:
    """Merge `updates` into the on-disk settings, return masked view."""
    with _LOCK:
        path = _settings_path()
        current = {**_DEFAULTS}
        if path.exists():
            try:
                current.update(json.loads(path.read_text()))
            except Exception:
                pass
        for k in ("provider", "model", "base_url", "api_key"):
            if k in updates and updates[k] is not None:
                current[k] = updates[k]
        # If api_key is masked-looking, ignore it (don't overwrite real key with bullets)
        new_key = updates.get("api_key", None)
        if new_key and set(new_key) <= {"•"}:
            pass  # user submitted the masked display value; ignore
        path.write_text(json.dumps(current, indent=2))
    return read_settings()
