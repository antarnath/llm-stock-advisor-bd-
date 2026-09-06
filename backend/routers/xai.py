"""GET /stocks/{ticker}/xai — SHAP feature importance + sample explanation.

Reads pre-computed artifacts from `results/xai/`:
  - global_importance_per_stock.csv  → mean |SHAP| per feature per stock
  - natural_language_explanations.txt → per-stock NL bullets

No live recomputation needed — the heavy XAI work was done in Phase 8.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from backend.models import XaiFeature, XaiResponse
from src.utils.config import TOP_30_DSE_STOCKS

router = APIRouter(tags=["xai"])

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
XAI_DIR = _PROJECT_ROOT / "results" / "xai"

_GLOBAL_CSV = XAI_DIR / "global_importance_per_stock.csv"
_NL_FILE = XAI_DIR / "natural_language_explanations.txt"


def _load_nl_block(ticker: str) -> str:
    """Extract the natural-language paragraph for `ticker` from the NL file."""
    if not _NL_FILE.exists():
        return ""
    text = _NL_FILE.read_text(encoding="utf-8", errors="ignore")
    # Each block is preceded by a heading like: 📈 Prediction: ACI expected ...
    # Split on emoji-prefixed headings, keep each block
    parts = re.split(r"(?=📈\s*Prediction:)", text)
    for part in parts:
        m = re.search(r"📈\s*Prediction:\s*([A-Z0-9]+)", part)
        if m and m.group(1) == ticker:
            return part.strip()
    return ""


@router.get("/stocks/{ticker}/xai", response_model=XaiResponse)
async def get_xai(ticker: str,
                  top_n: int = Query(default=5, ge=1, le=27)) -> XaiResponse:
    code = ticker.upper().strip()
    if code not in TOP_30_DSE_STOCKS:
        raise HTTPException(status_code=404, detail=f"Unknown ticker: {code}")

    top_features: list[XaiFeature] = []
    if _GLOBAL_CSV.exists():
        try:
            df = pd.read_csv(_GLOBAL_CSV)
            sub = df[df["stock"].str.upper() == code]
            sub = sub.sort_values("mean_abs_shap", ascending=False).head(top_n)
            for _, row in sub.iterrows():
                top_features.append(XaiFeature(
                    feature=str(row["feature"]),
                    mean_abs_shap=float(row.get("mean_abs_shap", 0.0)),
                    mean_shap=float(row.get("mean_shap", 0.0)),
                ))
        except Exception:
            pass

    sample = _load_nl_block(code)

    return XaiResponse(
        ticker=code,
        top_features=top_features,
        sample_explanation=sample,
        method="SHAP + LIME",
    )
