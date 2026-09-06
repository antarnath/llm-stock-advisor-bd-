"""Pydantic request/response schemas for the backend API."""

from __future__ import annotations

from typing import Optional, Literal

from pydantic import BaseModel, Field


# ----- /ask ---------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                          description="User question in English or Bangla")


class AskResponse(BaseModel):
    reply: str
    ticker: Optional[str] = None
    tools_called: list[str] = []
    mode: Literal["llm", "template", "stub"]
    tokens: int = 0


# ----- /stocks/{ticker} ---------------------------------------------------

class StockResponse(BaseModel):
    ticker: str
    name: str
    sector: str = ""
    close: Optional[float] = None
    date: Optional[str] = None
    available: bool = True


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    ticker: str
    available: bool
    as_of: Optional[str] = None
    horizon: Optional[str] = None
    current_price: float = 0.0
    predicted_price: float = 0.0
    predicted_return: float = 0.0
    fusion: str = "unknown"
    model_type: str = "unknown"


class NewsItem(BaseModel):
    article_id: Optional[str] = None
    impact: Literal["bullish", "bearish", "neutral"]
    magnitude: float = 0.0
    horizon_days: int = 7
    reason: str = ""
    published_at: str = ""


class NewsResponse(BaseModel):
    ticker: str
    days: int = 7
    items: list[NewsItem] = []


# ----- /portfolio/optimize ----------------------------------------------

class PortfolioOptimizeRequest(BaseModel):
    capital_bdt: float = Field(100_000.0, gt=0, le=10_000_000)
    profile: Literal["conservative", "moderate", "aggressive"] = "moderate"
    method: Literal["max_sharpe", "min_variance", "risk_parity"] = "max_sharpe"
    risk_free_rate: float = 0.05


class PortfolioAllocation(BaseModel):
    ticker: str
    weight: float
    amount_bdt: float


class PortfolioResult(BaseModel):
    profile: str
    method: str
    capital_bdt: float
    expected_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    shrinkage_delta: float = 0.0
    allocations: list[PortfolioAllocation] = []


# ----- /settings ---------------------------------------------------------

class SettingsResponse(BaseModel):
    provider: str
    model: str
    base_url: str
    api_key: str = ""
    has_key: bool


class SettingsUpdate(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None


# ----- /freshness --------------------------------------------------------

class FreshnessResponse(BaseModel):
    ran_at: Optional[str] = None
    rows_added: Optional[int] = None
    stocks_updated: list[str] = []
    status: str = "unknown"
    age_hours: Optional[float] = None
    label: str = ""


# ----- /stocks/{ticker}/xai ----------------------------------------------

class XaiFeature(BaseModel):
    feature: str
    mean_abs_shap: float = 0.0
    mean_shap: float = 0.0


class XaiResponse(BaseModel):
    ticker: str
    top_features: list[XaiFeature] = []
    sample_explanation: str = ""
    method: str = "SHAP + LIME"


# ----- /news/refresh -----------------------------------------------------

class RefreshResponse(BaseModel):
    triggered: bool
    message: str
