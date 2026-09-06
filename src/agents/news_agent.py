"""
News Analyst Agent — uses the RAG index + sentiment scores to produce a
news-driven signal for a stock.

Produces an evidence packet:
  - signal ∈ [-1, +1] based on aggregated sentiment
  - confidence proportional to article count + sentiment magnitude
  - top_features (top contributing news events)
  - RAG-retrieved news citations
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base import Agent, AgentEvidence


class NewsAgent(Agent):
    """Specialist: news sentiment + RAG retrieval."""

    name = "NewsAnalyst"

    def __init__(self,
                 sentiment_csv: Optional[Path] = None,
                 rag_index_dir: Optional[Path] = None,
                 default_top_k: int = 5):
        self.sentiment_csv = Path(sentiment_csv) if sentiment_csv else (
            Path("results/sentiment/news_scored.csv")
        )
        self.rag_index_dir = Path(rag_index_dir) if rag_index_dir else (
            Path("models/rag")
        )
        self.default_top_k = default_top_k
        self._sentiment_df: Optional[pd.DataFrame] = None
        self._retriever = None

    def _ensure_loaded(self):
        if self._sentiment_df is None and self.sentiment_csv.exists():
            self._sentiment_df = pd.read_csv(self.sentiment_csv, parse_dates=["date"])
        if self._retriever is None:
            try:
                from src.rag.retriever import NewsRetriever
                self._retriever = NewsRetriever(index_dir=self.rag_index_dir)
            except Exception:
                self._retriever = None

    def analyze(self, stock: str, query: Optional[str] = None, **kwargs) -> AgentEvidence:
        """Analyze news for the stock and produce signal."""
        self._ensure_loaded()

        if self._sentiment_df is None:
            return self._fallback(stock, "sentiment CSV missing")

        # Filter to this stock
        stock_news = self._sentiment_df[
            self._sentiment_df["stock"] == stock
        ].copy()

        if len(stock_news) == 0:
            return self._fallback(stock, "no news for this stock")

        # Aggregate: recent (last 90 days) mean weighted sentiment
        stock_news = stock_news.sort_values("date", ascending=False)
        recent = stock_news.head(20)  # last 20 articles

        mean_score = float(recent["score"].mean())
        weighted_score = float(
            (recent["score"] * recent["confidence"]).sum()
            / max(recent["confidence"].sum(), 1e-9)
        )
        n_articles = int(len(recent))
        n_pos = int((recent["pred_label"] == "positive").sum())
        n_neg = int((recent["pred_label"] == "negative").sum())

        # Confidence: more articles + higher |score| = higher confidence
        confidence = float(np.clip(0.3 + n_articles * 0.04 + abs(weighted_score) * 0.3, 0.1, 0.9))

        # Signal: weighted_score ∈ [-1, +1]
        signal = float(np.clip(weighted_score, -1.0, 1.0))

        # Build headline
        direction = "positive" if weighted_score > 0.05 else (
            "negative" if weighted_score < -0.05 else "neutral"
        )
        headline = (
            f"News sentiment for {stock}: {direction} "
            f"(weighted_score={weighted_score:+.2f}, {n_articles} recent articles, "
            f"{n_pos} positive / {n_neg} negative)"
        )

        # RAG retrieval (best-effort)
        rag_citations = []
        if self._retriever and query:
            try:
                hits = self._retriever.query(
                    query, top_k=self.default_top_k, filter_stock=stock,
                )
                for h in hits:
                    rag_citations.append(
                        f"[{h.date}] {h.headline} (sim={h.similarity:.2f})"
                    )
            except Exception:
                pass

        sources = [str(self.sentiment_csv)]
        sources.extend(rag_citations)

        return AgentEvidence(
            agent_name=self.name,
            stock=stock,
            signal=signal,
            confidence=confidence,
            headline=headline,
            details={
                "weighted_sentiment": weighted_score,
                "mean_sentiment": mean_score,
                "n_articles_recent": n_articles,
                "n_positive": n_pos,
                "n_negative": n_neg,
            },
            top_features=[
                ("positive_articles", float(n_pos)),
                ("negative_articles", float(n_neg)),
                ("sentiment_score", weighted_score),
            ],
            sources=sources,
        )

    def _fallback(self, stock: str, reason: str) -> AgentEvidence:
        return AgentEvidence(
            agent_name=self.name,
            stock=stock,
            signal=0.0,
            confidence=0.1,
            headline=f"News analysis unavailable ({reason})",
            details={"error": reason},
            sources=[],
        )
