"""
Impact reader — wraps the RAG retriever for the /stocks/{ticker}/news endpoint.

Returns recent news items for a ticker with impact classification
(bullish / bearish / neutral), magnitude, and short reasoning.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


def read_recent_impact(ticker: str,
                       days: int = 7,
                       top_k: int = 10,
                       rag_index_dir: Optional[Path] = None,
                       sentiment_csv: Optional[Path] = None,
                       ) -> list[dict]:
    """Return recent news items for `ticker` as a list of dicts.

    Each dict has: article_id, impact, magnitude, horizon_days, reason,
    published_at.
    """
    ticker = ticker.upper().strip()
    project_root = Path(__file__).resolve().parents[2]
    rag_index_dir = rag_index_dir or (project_root / "models" / "rag")
    sentiment_csv = sentiment_csv or (project_root / "results" / "sentiment" / "news_scored.csv")

    try:
        from src.rag.retriever import NewsRetriever
        retriever = NewsRetriever(index_dir=rag_index_dir)
        # Fetch with a wider window to ensure top_k hits even after filtering
        hits = retriever.query(
            text=f"{ticker} stock news",
            top_k=top_k * 3,
            filter_stock=ticker,
        )
    except Exception:
        hits = []

    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    items = []
    for i, h in enumerate(hits):
        if h.date < cutoff:
            continue
        # Map score → impact label
        if h.sentiment_score > 0.2:
            impact = "bullish"
        elif h.sentiment_score < -0.2:
            impact = "bearish"
        else:
            impact = "neutral"
        magnitude = min(abs(h.sentiment_score), 1.0)
        # Naive horizon: short for news, scale by magnitude
        horizon = max(1, int(round(5 * magnitude)) + 1)
        reason = (h.headline or "")[:200]
        items.append({
            "article_id": i,
            "impact": impact,
            "magnitude": float(magnitude),
            "horizon_days": int(horizon),
            "reason": reason,
            "published_at": h.date,
        })
        if len(items) >= top_k:
            break

    return items
