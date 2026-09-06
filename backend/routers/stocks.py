"""GET /stocks                     list of 30 tickers + sector + name
GET /stocks/{ticker}/prediction  latest 5-day prediction

Both delegate to Phase 10's reader functions. We only re-shape the dict
into Pydantic models for the wire format.
"""
from fastapi import APIRouter, HTTPException

from backend.models import PredictionOut, StockSummary
from src.orchestrator.prediction_reader import read_predictions
from src.utils.config import TOP_30_DSE_STOCKS, load_stock_meta

router = APIRouter(prefix="/stocks")


@router.get("", response_model=list[StockSummary])
def list_stocks() -> list[StockSummary]:
    """Return one row per tracked DSE stock."""
    meta = load_stock_meta()
    out: list[StockSummary] = []
    for code in TOP_30_DSE_STOCKS:
        m = meta.get(code, {})
        out.append(StockSummary(
            code=code,
            name=m.get("name") or code,
            sector=m.get("sector") or "",
        ))
    return out


@router.get("/{ticker}/prediction", response_model=PredictionOut)
def get_prediction(ticker: str) -> PredictionOut:
    """Return the freshest 5-day multimodal prediction for `ticker`.

    Always returns 200; missing data is signalled with `available: false`.
    """
    code = ticker.upper().strip()
    if not code:
        raise HTTPException(400, "empty ticker")
    pred = read_predictions(code)
    if not pred.get("available"):
        return PredictionOut(ticker=code, available=False)
    return PredictionOut(
        ticker=code,
        available=True,
        as_of=pred["as_of"],
        horizon="5 days",
        current_price=pred["current_price"],
        predicted_price=pred["predicted_price"],
        predicted_return=pred["predicted_return"],
        fusion=pred["fusion"],
        model_type=pred["model_type"],
    )