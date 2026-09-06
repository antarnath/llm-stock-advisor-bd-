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

            # Build a custom LLMClient from the persisted settings
            provider = os.getenv("LLM_PROVIDER", "stub")
            model = os.getenv("LLM_MODEL", "stub-template-v1")
            base_url = os.getenv("LLM_BASE_URL", "")
            api_key = os.getenv("LLM_API_KEY", "")
            llm = LLMClient(provider=provider, model=model,
                            base_url=base_url or None, api_key=api_key or None)

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
    """Apply new settings + rebuild the orchestrator singleton."""
    global _ORCH
    with _ORCH_LOCK:
        # Push env so subsequent cold starts also pick them up
        os.environ["LLM_PROVIDER"] = provider
        os.environ["LLM_MODEL"] = model
        if base_url:
            os.environ["LLM_BASE_URL"] = base_url
        else:
            os.environ.pop("LLM_BASE_URL", None)
        if api_key:
            os.environ["LLM_API_KEY"] = api_key
        else:
            os.environ.pop("LLM_API_KEY", None)

        from src.llm import LLMClient
        from src.orchestrator import Orchestrator

        llm = LLMClient(provider=provider, model=model,
                        base_url=base_url or None, api_key=api_key or None)
        _ORCH = Orchestrator(project_root=_PROJECT_ROOT, llm_client=llm)
        return _ORCH
