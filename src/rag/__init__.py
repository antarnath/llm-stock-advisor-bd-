"""
RAG (Retrieval-Augmented Generation) system over the DSE news corpus.

Components:
- NewsEmbedder: sentence-transformer wrapper (multilingual: en + bn)
- NewsIndex: FAISS index with stock/date/language metadata
- NewsRetriever: query interface with metadata filtering

Usage:
    from src.rag import build_index, retrieve

    # Build index once
    build_index(corpus_path="data/raw/news/news_curated.csv",
                index_dir="models/rag/")

    # Query
    results = retrieve("BATBC expansion news", top_k=5,
                       filter_stock="BATBC")
"""

from .embedder import NewsEmbedder
from .indexer import NewsIndex, build_index
from .retriever import NewsRetriever, retrieve

__all__ = ["NewsEmbedder", "NewsIndex", "NewsRetriever", "build_index", "retrieve"]
