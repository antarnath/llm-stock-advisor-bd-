"""
NewsIndex — FAISS index for DSE news articles with metadata.

Stores:
- FAISS index (InnerProduct over L2-normalized vectors → cosine similarity)
- Metadata per article: news_id, date, stock, name, sector, language,
                        headline, content, sentiment_score, sentiment_label
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import faiss
except ImportError:
    faiss = None

from .embedder import NewsEmbedder


@dataclass
class NewsHit:
    """One retrieval result."""
    news_id: str
    date: str
    stock: str
    name: str
    sector: str
    language: str
    headline: str
    content: str
    sentiment_score: float
    sentiment_label: str
    similarity: float
    rank: int = 0


class NewsIndex:
    """FAISS-backed news article index with metadata."""

    METADATA_COLS = [
        "news_id", "date", "stock", "name", "sector", "language",
        "headline", "content", "score", "pred_label",
    ]

    def __init__(self, embeddings: np.ndarray, metadata: pd.DataFrame):
        if faiss is None:
            raise ImportError("faiss-cpu is required. pip install faiss-cpu")
        assert len(embeddings) == len(metadata), \
            f"emb {len(embeddings)} vs meta {len(metadata)} mismatch"
        self.dim = embeddings.shape[1]
        self.embeddings = embeddings
        self.metadata = metadata.reset_index(drop=True)

        # Build FAISS index (InnerProduct over L2-normalized vectors)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               filter_stock: Optional[str] = None,
               filter_language: Optional[str] = None,
               filter_date_from: Optional[str] = None,
               filter_date_to: Optional[str] = None) -> list[NewsHit]:
        """Search with optional metadata filters.

        Note: filters are post-hoc (we retrieve top_k*10 then filter).
        For 1.5k docs this is fine.
        """
        # Retrieve more candidates than needed to compensate for filtering
        fetch_k = top_k * 20 if any([filter_stock, filter_language,
                                     filter_date_from, filter_date_to]) else top_k
        fetch_k = min(fetch_k, len(self.metadata))
        D, I = self.index.search(query_embedding.reshape(1, -1).astype(np.float32), fetch_k)

        hits = []
        for rank, (dist, idx) in enumerate(zip(D[0], I[0])):
            if idx < 0:
                continue
            row = self.metadata.iloc[idx]
            # Filters
            if filter_stock and row["stock"] != filter_stock:
                continue
            if filter_language and row["language"] != filter_language:
                continue
            if filter_date_from and str(row["date"]) < filter_date_from:
                continue
            if filter_date_to and str(row["date"]) > filter_date_to:
                continue
            hits.append(NewsHit(
                news_id=row["news_id"],
                date=str(row["date"]),
                stock=row["stock"],
                name=row["name"],
                sector=row["sector"],
                language=row["language"],
                headline=row["headline"],
                content=row["content"],
                sentiment_score=float(row["score"]),
                sentiment_label=row["pred_label"],
                similarity=float(dist),
                rank=len(hits),
            ))
            if len(hits) >= top_k:
                break
        return hits

    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # Save FAISS index
        faiss.write_index(self.index, str(path / "faiss.index"))
        # Save metadata
        self.metadata.to_parquet(path / "metadata.parquet")
        # Save config
        (path / "config.pkl").write_bytes(pickle.dumps({
            "dim": self.dim,
            "n_docs": len(self.metadata),
        }))

    @classmethod
    def load(cls, path: Path) -> "NewsIndex":
        path = Path(path)
        idx = faiss.read_index(str(path / "faiss.index"))
        metadata = pd.read_parquet(path / "metadata.parquet")
        embeddings = np.zeros((len(metadata), idx.d), dtype=np.float32)
        for i in range(len(metadata)):
            embeddings[i] = idx.reconstruct(i)
        return cls(embeddings, metadata)


def build_index(corpus_path: Path, index_dir: Path,
                sentiment_path: Optional[Path] = None,
                model_name: str = NewsEmbedder.DEFAULT_MODEL,
                batch_size: int = 32) -> NewsIndex:
    """Build a FAISS index from a CSV of news articles.

    Parameters
    ----------
    corpus_path : Path
        CSV with columns: news_id, date, stock, name, sector, language, headline, content
    sentiment_path : Path, optional
        CSV with sentiment scores (from Phase 6). Joins on news_id.
    index_dir : Path
        Where to save the index.
    """
    print(f"📂 Loading corpus from {corpus_path}")
    df = pd.read_csv(corpus_path)
    print(f"   {len(df):,} articles")

    # Join sentiment if provided
    if sentiment_path and Path(sentiment_path).exists():
        sent = pd.read_csv(sentiment_path)
        sent_cols = ["news_id", "score", "pred_label", "confidence"]
        sent_cols = [c for c in sent_cols if c in sent.columns]
        df = df.merge(sent[sent_cols], on="news_id", how="left")
    else:
        df["score"] = 0.0
        df["pred_label"] = "neutral"

    # Fill missing
    df["score"] = df["score"].fillna(0.0)
    df["pred_label"] = df["pred_label"].fillna("neutral")
    df["content"] = df["content"].fillna("").astype(str)
    df["headline"] = df["headline"].fillna("").astype(str)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Build text for embedding: headline + content (truncated)
    texts = (df["headline"] + ". " + df["content"]).tolist()
    print(f"   texts prepared (avg {np.mean([len(t) for t in texts]):.0f} chars)")

    # Embed
    print(f"🤖 Loading embedder {model_name}")
    embedder = NewsEmbedder(model_name=model_name, device="cpu")
    print(f"   model dim = {embedder.dim}")
    embeddings = embedder.encode(texts, batch_size=batch_size, show_progress=True)

    # Build index
    print(f"📦 Building FAISS index")
    metadata_cols = ["news_id", "date", "stock", "name", "sector", "language",
                     "headline", "content", "score", "pred_label"]
    metadata = df[metadata_cols].copy()
    news_index = NewsIndex(embeddings, metadata)

    # Save
    news_index.save(index_dir)
    print(f"💾 Saved index to {index_dir}")
    print(f"   {len(metadata):,} articles, dim={news_index.dim}")

    return news_index
