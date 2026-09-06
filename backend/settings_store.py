"""Runtime-overridable LLM config (Phase 13).

Persists user-editable provider/model/base_url/api_key to a JSON file
under data/external/api/, mirroring the atomic-rename pattern already
established by src.utils.refresh_log.write_refresh_log.

Schema:
  {
    "provider":  "<stub|openrouter|groq|gemini|nim>",
    "model":     "<model id; blank means provider default>",
    "base_url":  "<NIM only; blank means integrated endpoint>",
    "api_key":   "<plaintext key; chmod 600 on disk>"
  }

Any field may be empty/missing — callers must fall back to env vars,
then to the provider's default URL, then to stub mode.

The key is masked in any value returned by `mask_key()` so that the
GET /settings endpoint never echoes the full secret to the browser.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from src.utils.config import EXTERNAL_DATA_DIR

#: Location of the on-disk config. Lives next to chat_log.sqlite.
SETTINGS_PATH: Path = EXTERNAL_DATA_DIR / "api" / "llm_settings.json"

#: Set of providers the backend understands. Anything else = ValueError.
_VALID_PROVIDERS: set[str] = {"stub", "openrouter", "groq", "gemini", "nim"}


# ---------------------------------------------------------------------------
# low-level I/O
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, payload: dict) -> None:
    """Write `payload` to `path` via a sibling .tmp + replace (atomic).

    Same pattern as src.utils.refresh_log.write_refresh_log — guarantees
    readers never see a half-written file. Sets 0600 on the temp file so
    the API key isn't world-readable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        # Some filesystems (Windows, some FUSE mounts) don't support chmod;
        # the secret is still protected by the parent directory.
        pass
    tmp.replace(path)


def read_settings() -> dict:
    """Return the persisted settings, or {} if absent/invalid.

    Never raises — the absence of a settings file is the normal first-run
    case. A corrupt JSON file is logged at import time only if it's
    malformed in a way the loader can't handle.
    """
    if not SETTINGS_PATH.exists():
        return {}
    try:
        data = json.loads(SETTINGS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        "provider": str(data.get("provider") or "").strip(),
        "model":    str(data.get("model")    or "").strip(),
        "base_url": str(data.get("base_url") or "").strip(),
        "api_key":  str(data.get("api_key")  or "").strip(),
    }


def write_settings(payload: dict) -> dict:
    """Validate, normalize, persist. Returns the saved record with key masked."""
    provider = (payload.get("provider") or "stub").lower().strip()
    if provider not in _VALID_PROVIDERS:
        raise ValueError(
            f"provider must be one of {sorted(_VALID_PROVIDERS)} (got {provider!r})"
        )

    record = {
        "provider": provider,
        "model":    (payload.get("model")    or "").strip(),
        "base_url": (payload.get("base_url") or "").strip(),
        "api_key":  (payload.get("api_key")  or "").strip(),
    }
    if record["provider"] == "stub":
        # Stub mode means no network. Drop any sensitive fields so a
        # forgotten key doesn't linger on disk.
        record["api_key"] = ""
        record["base_url"] = ""
    _atomic_write(SETTINGS_PATH, record)
    return mask_key(record)


def delete_settings() -> None:
    """Remove the on-disk config. No-op if it doesn't exist."""
    try:
        SETTINGS_PATH.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# presentation helpers
# ---------------------------------------------------------------------------

def mask_key(record: dict) -> dict:
    """Return a copy of `record` with the key masked for the browser.

    Replaces everything except the last 4 chars with `*`. Short keys are
    replaced with `****`. Includes a `has_key` boolean so the frontend
    can tell "blank input, but a key was saved" apart from "no key at all".
    """
    out = dict(record)
    k = out.get("api_key") or ""
    if not k:
        masked = ""
    elif len(k) > 4:
        masked = f"{'*' * (len(k) - 4)}{k[-4:]}"
    else:
        masked = "****"
    out["api_key"] = masked
    out["has_key"] = bool(record.get("api_key"))
    return out


# ---------------------------------------------------------------------------
# propagation to os.environ
# ---------------------------------------------------------------------------

def apply_to_environ() -> None:
    """Copy saved settings into os.environ so the existing chat() / orchestrator
    code paths can read them via os.getenv, with no other code changes.

    Called at backend startup (inside the FastAPI lifespan) and after every
    POST /settings. Safe to call repeatedly — overwrites whatever was there.

    Both ADVISOR_* (orchestrator) and NEWS_* (news classifier) env vars are
    set from the same record so the two LLM consumers stay in sync.

    For backwards-compat with the historical provider-specific key env vars
    (OPENROUTER_API_KEY / GROQ_API_KEY / GOOGLE_API_KEY), the saved api_key
    is also mirrored onto whichever of those names matches the chosen provider.
    """
    s = read_settings()
    if not s:
        return

    provider = (s.get("provider") or "").lower()

    # Provider selection — same name for both consumers.
    if provider:
        os.environ["ADVISOR_LLM_PROVIDER"] = provider
        os.environ["NEWS_LLM_PROVIDER"]    = provider

    # Model id (blank = provider default).
    if s.get("model"):
        os.environ["ADVISOR_LLM_MODEL"] = s["model"]
        os.environ["NEWS_LLM_MODEL"]    = s["model"]

    # NIM-specific URL override.
    if s.get("base_url"):
        os.environ["NIM_BASE_URL"] = s["base_url"]

    # Key: own env var (NIM_API_KEY) + legacy provider-name env vars
    # so the existing _PROVIDER_KEYS lookup keeps working.
    if s.get("api_key"):
        os.environ["NIM_API_KEY"] = s["api_key"]
        if provider == "openrouter":
            os.environ["OPENROUTER_API_KEY"] = s["api_key"]
        elif provider == "groq":
            os.environ["GROQ_API_KEY"] = s["api_key"]
        elif provider == "gemini":
            os.environ["GOOGLE_API_KEY"] = s["api_key"]
