"""
Prediction reader — pulls the latest prediction for a ticker from the
multimodal inference output.

Reads results/multimodal/predictions_5days.csv (or 1day fallback) and
returns the most recent row for the given ticker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def read_predictions(ticker: str,
                     predictions_csv: Optional[Path] = None,
                     ) -> dict:
    """Return the latest prediction for `ticker`.

    Returns a dict with: ticker, available, as_of, horizon, current_price,
    predicted_price, predicted_return, fusion, model_type.

    If no prediction is available, returns {available: False}.
    """
    ticker = ticker.upper().strip()
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        predictions_csv,
        project_root / "results" / "multimodal" / "predictions_5days.csv",
        project_root / "results" / "deep_learning" / "predictions_1days.csv",
        project_root / "results" / "multimodal" / "predictions_1days.csv",
    ]
    csv_path = next((p for p in candidates if p and Path(p).exists()), None)
    if csv_path is None:
        return {"ticker": ticker, "available": False}

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
    except Exception:
        return {"ticker": ticker, "available": False}

    # Find the ticker column
    code_col = next((c for c in df.columns
                     if c.lower() in ("code", "ticker", "stock")), None)
    if code_col is None:
        return {"ticker": ticker, "available": False}

    rows = df[df[code_col].str.upper() == ticker]
    if rows.empty:
        return {"ticker": ticker, "available": False}

    row = rows.iloc[-1]
    horizon = csv_path.stem.replace("predictions_", "")
    return {
        "ticker": ticker,
        "available": True,
        "as_of": str(row.get("date", row.get("as_of", ""))),
        "horizon": horizon,
        "current_price": float(row.get("current_price", row.get("close", 0.0)) or 0.0),
        "predicted_price": float(row.get("predicted_price", row.get("pred_close", 0.0)) or 0.0),
        "predicted_return": float(row.get("predicted_return", row.get("pred_return", 0.0)) or 0.0),
        "fusion": str(row.get("fusion_strategy", row.get("fusion", "unknown"))),
        "model_type": str(row.get("model_type", "multimodal")),
    }


# Backwards-compat: older callers used read_latest_prediction
def read_latest_prediction(*args, **kwargs) -> dict:
    return read_predictions(*args, **kwargs)
