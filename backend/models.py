"""Pydantic request/response schemas for the DSE Advisor API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# Per the DSE Advisor domain we have fields like `model_type` that would
# otherwise collide with Pydantic's `model_*` protected namespace.
_BASE_CONFIG = ConfigDict(protected_namespaces=())


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

ImpactLabel = Literal["bullish", "bearish", "neutral"]


# ---------------------------------------------------------------------------
# /healthz
# ---------------------------------------------------------------------------

class HealthOut(BaseModel):
    status: Literal["ok"]


# ---------------------------------------------------------------------------
# /stocks
# ---------------------------------------------------------------------------

class StockSummary(BaseModel):
    code: str
    name: str
    sector: str


class PredictionOut(BaseModel):
    model_config = _BASE_CONFIG
    ticker: str
    available: bool
    as_of: str | None = None
    horizon: str | None = None
    current_price: float | None = None
    predicted_price: float | None = None
    predicted_return: float | None = None   # % over the horizon
    fusion: str | None = None
    model_type: str | None = None


# ---------------------------------------------------------------------------
# /stocks/{ticker}/news
# ---------------------------------------------------------------------------

class NewsItem(BaseModel):
    article_id: int
    impact: ImpactLabel
    magnitude: float
    horizon_days: int
    reason: str
    published_at: str


class NewsOut(BaseModel):
    ticker: str
    days: int
    items: list[NewsItem]


# ---------------------------------------------------------------------------
# /freshness
# ---------------------------------------------------------------------------

class FreshnessOut(BaseModel):
    ran_at: str | None
    rows_added: int | None
    stocks_updated: list[str]
    status: str            # "ok" | "stale" | "unknown" | "broken"
    age_hours: float | None = None
    label: str


# ---------------------------------------------------------------------------
# /ask
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    # min_length=1 rejects empty strings; max_length=2000 is a cheap DoS guard
    # before the request reaches the LLM.
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(None, max_length=128)


class AskResponse(BaseModel):
    reply: str
    session_id: str | None
    latency_ms: int


# ---------------------------------------------------------------------------
# /settings  (Phase 13 — runtime provider config)
# ---------------------------------------------------------------------------

class SettingsIn(BaseModel):
    """Body for POST /settings. All fields optional/blank — empty values
    mean "fall through to env, then default"."""
    model_config = _BASE_CONFIG
    provider: str = Field("", max_length=32)
    model:    str = Field("", max_length=200)
    base_url: str = Field("", max_length=500)
    api_key:  str = Field("", max_length=500)


class SettingsOut(BaseModel):
    """Response for GET / POST / DELETE /settings. The api_key is masked
    server-side (only last 4 chars visible) and has_key is a separate flag.
    All string fields default to "" so an empty settings file (first-run
    case) renders as an empty record rather than a 422."""
    model_config = _BASE_CONFIG
    provider: str = ""
    model:    str = ""
    base_url: str = ""
    api_key:  str = ""
    has_key:  bool = False
