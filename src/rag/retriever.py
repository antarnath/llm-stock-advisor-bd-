"""
NewsRetriever — friendly query interface over the news index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .embedder import NewsEmbedder
from .indexer import NewsIndex, NewsHit


class NewsRetriever:
    """Convenience wrapper combining an index with an embedder."""

    def __init__(self, index_dir: Path, model_name: Optional[str] = None):
        self.index_dir = Path(index_dir)
        self.index = NewsIndex.load(self.index_dir)
        # Use the index's dim if available, otherwise default model
        self.embedder = NewsEmbedder(
            model_name=model_name or NewsEmbedder.DEFAULT_MODEL,
            device="cpu",
        )

    def query(self, text: str, top_k: int = 5,
              filter_stock: Optional[str] = None,
              filter_language: Optional[str] = None,
              filter_date_from: Optional[str] = None,
              filter_date_to: Optional[str] = None) -> list[NewsHit]:
        """Retrieve articles matching a natural-language query."""
        q_emb = self.embedder.encode_query(text)
        return self.index.search(
            q_emb, top_k=top_k,
            filter_stock=filter_stock,
            filter_language=filter_language,
            filter_date_from=filter_date_from,
            filter_date_to=filter_date_to,
        )


def retrieve(query: str, top_k: int = 5,
             filter_stock: Optional[str] = None,
             index_dir: Path = Path("models/rag/")) -> list[NewsHit]:
    """One-shot retrieval (load index, query, return)."""
    retriever = NewsRetriever(index_dir=index_dir)
    return retriever.query(query, top_k=top_k, filter_stock=filter_stock)


def format_hits(hits: list[NewsHit], include_content: bool = False) -> str:
    """Pretty-print retrieval results for use in LLM prompts or UI."""
    if not hits:
        return "(no matches)"
    lines = []
    for i, h in enumerate(hits, 1):
        sign = "+" if h.sentiment_score > 0 else ("-" if h.sentiment_score < 0 else "0")
        lines.append(
            f"[{i}] {h.date} | {h.stock} ({h.language}) | sent={sign}{abs(h.sentiment_score):.2f}"
        )
        lines.append(f"    {h.headline}")
        if include_content:
            lines.append(f"    {h.content[:300]}{'...' if len(h.content) > 300 else ''}")
        lines.append(f"    sim={h.similarity:.3f}")
    return "\n".join(lines)
