"""FastAPI routers mounted by backend.main.

Each router is a small module exposing a single APIRouter instance.
"""
from . import ask, freshness, health, news, settings, stocks  # noqa: F401

__all__ = ["ask", "freshness", "health", "news", "settings", "stocks"]
