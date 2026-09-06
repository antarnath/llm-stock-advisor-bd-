"""Singleton accessors for the heavy backend singletons.

Each function is lazy + thread-safe so the FastAPI app starts quickly even
when torch/torchvision/transformers are slow to import.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load .env once at import time so ADVISOR_LLM_PROVIDER / GROQ_API_KEY etc.
# are visible to LLMClient.__init__() and Orchestrator construction.
# override=False means real shell env vars win over .env values.
try:
    from dotenv import load_dotenv
    _ENV_PATH = _PROJECT_ROOT / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH, override=False)
        logger.info("Loaded environment from %s", _ENV_PATH)
    else:
        logger.debug("No .env file at %s — using shell env only", _ENV_PATH)
except ImportError:
    logger.debug("python-dotenv not installed; .env will not be auto-loaded")

_ORCH: Optional[object] = None
_ORCH_LOCK = Lock()


def get_orchestrator(force_reload: bool = False):
    """Return the lazily-instantiated Orchestrator singleton."""
    global _ORCH
    if _ORCH is not None and not force_reload:
        return _ORCH
    with _ORCH_LOCK:
        if _ORCH is not None and not force_reload:
            return _ORCH
        try:
            from src.orchestrator import Orchestrator
            from src.llm import LLMClient
            from src.agents.orchestrator import STOCK_UNIVERSE
            from src.orchestrator.ticker_resolver import resolve_ticker
            from src.utils.config import TOP_30_DSE_STOCKS

            # Build a custom LLMClient from persisted settings OR .env.
            # Prefer canonical ADVISOR_LLM_* names, fall back to legacy LLM_*.
            provider = (os.getenv("ADVISOR_LLM_PROVIDER")
                        or os.getenv("LLM_PROVIDER") or "stub")
            model = (os.getenv("ADVISOR_LLM_MODEL")
                     or os.getenv("LLM_MODEL") or "")
            base_url = (os.getenv(f"{provider.upper()}_BASE_URL")
                        or os.getenv("LLM_BASE_URL") or "")
            # Key resolution: provider-specific env var first, then legacy LLM_API_KEY
            api_key = (os.getenv(f"{provider.upper()}_API_KEY")
                       or (os.getenv("GOOGLE_API_KEY") if provider == "gemini" else "")
                       or os.getenv("LLM_API_KEY") or "")
            llm = LLMClient(provider=provider, model=model or None,
                            base_url=base_url or None, api_key=api_key or None)
            logger.info("Orchestrator built: provider=%s model=%s is_live=%s",
                        llm.provider, llm.model, llm.is_live)

            orch = Orchestrator(
                project_root=_PROJECT_ROOT,
                llm_client=llm,
            )
            _ORCH = orch
            return _ORCH
        except Exception as e:
            logger.exception("Failed to build Orchestrator singleton: %s", e)
            return None


def reset_orchestrator() -> None:
    """Drop the cached orchestrator so the next call rebuilds it."""
    global _ORCH
    with _ORCH_LOCK:
        _ORCH = None


def reload_orchestrator_with_settings(provider: str, model: str,
                                       base_url: str, api_key: str):
    """Apply new settings + rebuild the orchestrator singleton.

    Pushes env vars under BOTH the legacy (LLM_*) names used by older
    code paths AND the canonical ADVISOR_LLM_* names read by LLMClient,
    so every consumer sees the new values. If `api_key` is empty (e.g.
    user didn't include it in the POST body), falls back to whatever the
    process env already has — typically from .env at startup.
    """
    global _ORCH
    with _ORCH_LOCK:
        # Canonical names that LLMClient reads
        os.environ["ADVISOR_LLM_PROVIDER"] = provider
        if model:
            os.environ["ADVISOR_LLM_MODEL"] = model
        # Provider-specific base URL override
        if base_url:
            os.environ[f"{provider.upper()}_BASE_URL"] = base_url
        else:
            os.environ.pop(f"{provider.upper()}_BASE_URL", None)
        # Provider-specific API key (LLMClient looks up {PROVIDER}_API_KEY).
        # If blank, keep what env already has (e.g. from .env at boot).
        if api_key:
            os.environ[f"{provider.upper()}_API_KEY"] = api_key
            if provider == "gemini":
                os.environ["GOOGLE_API_KEY"] = api_key
        else:
            existing = os.getenv(f"{provider.upper()}_API_KEY", "")
            if existing:
                # Re-affirm so the LLMClient constructor reads it
                os.environ[f"{provider.upper()}_API_KEY"] = existing

        # Legacy aliases (kept for any old code paths that still read them)
        os.environ["LLM_PROVIDER"] = provider
        if model:
            os.environ["LLM_MODEL"] = model
        if base_url:
            os.environ["LLM_BASE_URL"] = base_url
        else:
            os.environ.pop("LLM_BASE_URL", None)
        if api_key:
            os.environ["LLM_API_KEY"] = api_key
        # (else: keep existing LLM_API_KEY)

        from src.llm import LLMClient
        from src.orchestrator import Orchestrator

        # Resolve effective key from env (in case caller passed blank)
        effective_key = (api_key
                         or os.getenv(f"{provider.upper()}_API_KEY", "")
                         or (os.getenv("GOOGLE_API_KEY") if provider == "gemini" else ""))
        llm = LLMClient(provider=provider, model=model,
                        base_url=base_url or None, api_key=effective_key or None)
        _ORCH = Orchestrator(project_root=_PROJECT_ROOT, llm_client=llm)
        logger.info("Orchestrator rebuilt: provider=%s model=%s is_live=%s",
                    llm.provider, llm.model, llm.is_live)
        return _ORCH
