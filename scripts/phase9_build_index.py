"""
Phase 9 — Build the RAG index over the DSE news corpus.

Reads: data/raw/news/news_curated.csv (1,560 articles)
       results/sentiment/news_scored.csv (sentiment scores)
Embeds: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
Writes: models/rag/  (FAISS index + metadata.parquet)

Usage:
    .venv/bin/python scripts/phase9_build_index.py
    .venv/bin/python scripts/phase9_build_index.py --max-articles 100  # smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.rag.indexer import build_index
from src.utils.logger import get_logger

logger = get_logger("phase9_build_index")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path,
                        default=_PROJECT_ROOT / "data/raw/news/news_curated.csv")
    parser.add_argument("--sentiment", type=Path,
                        default=_PROJECT_ROOT / "results/sentiment/news_scored.csv")
    parser.add_argument("--index-dir", type=Path,
                        default=_PROJECT_ROOT / "models/rag")
    parser.add_argument("--max-articles", type=int, default=None,
                        help="Limit corpus (debug).")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PHASE 9 — RAG INDEX BUILDER")
    logger.info("=" * 60)

    # If --max-articles is set, write a temp truncated CSV
    if args.max_articles:
        import pandas as pd
        df = pd.read_csv(args.corpus).head(args.max_articles)
        tmp = _PROJECT_ROOT / "results/rag_corpus_truncated.csv"
        tmp.parent.mkdir(exist_ok=True)
        df.to_csv(tmp, index=False)
        corpus = tmp
        logger.info(f"⚠️  Truncated corpus: {len(df)} articles at {tmp}")
    else:
        corpus = args.corpus

    t0 = time.time()
    idx = build_index(
        corpus_path=corpus,
        sentiment_path=args.sentiment,
        index_dir=args.index_dir,
        batch_size=args.batch_size,
    )
    elapsed = time.time() - t0

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"✅ Built RAG index in {elapsed:.1f}s")
    logger.info(f"📁 Index: {args.index_dir}")
    logger.info(f"📊 Articles: {len(idx.metadata):,}")
    logger.info(f"🔢 Dim: {idx.dim}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
