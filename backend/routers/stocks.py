"""Stock detail endpoints:
  GET /stocks                  — list the 30 DSE tickers with name+sector
  GET /stocks/{ticker}         — last close + meta
  GET /stocks/{ticker}/prediction — latest multimodal prediction
  GET /stocks/{ticker}/news    — recent news impact (RAG)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from backend.models import (
    NewsItem,
    NewsResponse,
    PredictionResponse,
    StockResponse,
)
from src.orchestrator.prediction_reader import read_predictions
from src.orchestrator.impact_reader import read_recent_impact
from src.utils.config import (
    PROCESSED_DATA_DIR,
    TOP_30_DSE_STOCKS,
    load_stock_meta,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["stocks"])


@router.get("/stocks", response_model=list[StockResponse])
async def list_stocks() -> list[StockResponse]:
    """Return the 30 DSE tickers with name + sector."""
    meta = load_stock_meta()
    out = []
    for code in TOP_30_DSE_STOCKS:
        m = meta.get(code, {})
        out.append(StockResponse(
            ticker=code,
            name=str(m.get("name", code)),
            sector=str(m.get("sector", "")),
        ))
    return out


def _read_latest_processed(ticker: str) -> dict:
    """Read the most recent row of {ticker}_processed_v2.csv."""
    path = PROCESSED_DATA_DIR / f"{ticker}_processed_v2.csv"
    if not path.exists():
        path = PROCESSED_DATA_DIR / f"{ticker}_processed.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        if df.empty:
            return {}
        row = df.iloc[-1]
        out = {"date": str(row.get("date", row.get("Date", "")))}
        for col in ("close", "Close", "adj_close", "Adj Close"):
            if col in df.columns:
                try:
                    out["close"] = float(row[col])
                    break
                except (TypeError, ValueError):
                    pass
        return out
    except Exception:
        return {}


@router.get("/stocks/{ticker}", response_model=StockResponse)
async def get_stock(ticker: str) -> StockResponse:
    """Return last price + meta for a single ticker."""
    code = ticker.upper().strip()
    if code not in TOP_30_DSE_STOCKS:
        raise HTTPException(status_code=404, detail=f"Unknown ticker: {code}")

    meta = load_stock_meta().get(code, {})
    latest = _read_latest_processed(code)
    return StockResponse(
        ticker=code,
        name=str(meta.get("name", code)),
        sector=str(meta.get("sector", "")),
        close=latest.get("close"),
        date=str(latest.get("date", "") or "") or None,
    )


@router.get("/stocks/{ticker}/prediction", response_model=PredictionResponse)
async def get_prediction(ticker: str) -> PredictionResponse:
    """Return the latest multimodal prediction for `ticker`."""
    code = ticker.upper().strip()
    pred = read_predictions(code)
    return PredictionResponse(
        ticker=code,
        available=bool(pred.get("available", False)),
        as_of=str(pred.get("as_of", "") or "") or None,
        horizon=str(pred.get("horizon", "") or "") or None,
        current_price=float(pred.get("current_price", 0.0) or 0.0),
        predicted_price=float(pred.get("predicted_price", 0.0) or 0.0),
        predicted_return=float(pred.get("predicted_return", 0.0) or 0.0),
        fusion=str(pred.get("fusion", "unknown")),
        model_type=str(pred.get("model_type", "multimodal")),
    )


@router.get("/stocks/{ticker}/news", response_model=NewsResponse)
async def get_news(ticker: str,
                   days: int = Query(default=180, ge=1, le=730)) -> NewsResponse:
    """Return recent news items with bullish/bearish impact labels.

    Default `days=180` because the news corpus's most recent articles
    pre-date the latest processed price snapshot, so a 7-day window
    returns empty. If the requested window returns no hits, we silently
    widen to 5 years so the user still gets the most relevant articles.
    """
    code = ticker.upper().strip()
    raw = read_recent_impact(code, days=days, top_k=10)
    # If empty, widen silently to all-time (5y) regardless of the request size
    if not raw:
        raw = read_recent_impact(code, days=3650, top_k=10)
    items = []
    for it in raw or []:
        items.append(NewsItem(
            article_id=str(it.get("article_id", "") or "") or None,
            impact=it.get("impact", "neutral"),
            magnitude=float(it.get("magnitude", 0.0) or 0.0),
            horizon_days=int(it.get("horizon_days", days) or days),
            reason=str(it.get("reason", "") or ""),
            published_at=str(it.get("published_at", "") or ""),
        ))
    return NewsResponse(ticker=code, days=days, items=items)
