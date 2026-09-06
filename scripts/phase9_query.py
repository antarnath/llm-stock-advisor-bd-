"""
Phase 9 — Query the RAG index.

Demonstrates retrieval with various queries and filters.

Usage:
    .venv/bin/python scripts/phase9_query.py "BATBC dividend"
    .venv/bin/python scripts/phase9_query.py "bank sector scandal" --filter-language bn
    .venv/bin/python scripts/phase9_query.py "pharma expansion" --filter-stock SQURPHARMA
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.rag.retriever import NewsRetriever, format_hits
from src.utils.logger import get_logger

logger = get_logger("phase9_query")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="Natural-language query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--filter-stock", type=str, default=None)
    parser.add_argument("--filter-language", type=str, default=None,
                        choices=["en", "bn"])
    parser.add_argument("--filter-date-from", type=str, default=None)
    parser.add_argument("--filter-date-to", type=str, default=None)
    parser.add_argument("--index-dir", type=Path,
                        default=_PROJECT_ROOT / "models/rag")
    parser.add_argument("--include-content", action="store_true")
    args = parser.parse_args()

    logger.info(f"🔍 Query: {args.query!r}")
    if args.filter_stock:
        logger.info(f"   filter stock: {args.filter_stock}")
    if args.filter_language:
        logger.info(f"   filter language: {args.filter_language}")

    retriever = NewsRetriever(index_dir=args.index_dir)
    hits = retriever.query(
        args.query, top_k=args.top_k,
        filter_stock=args.filter_stock,
        filter_language=args.filter_language,
        filter_date_from=args.filter_date_from,
        filter_date_to=args.filter_date_to,
    )
    print()
    print(format_hits(hits, include_content=args.include_content))
    print(f"\n{len(hits)} matches")


if __name__ == "__main__":
    main()
