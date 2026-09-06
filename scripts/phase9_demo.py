"""
Phase 9 — Demo: run a curated set of RAG queries and save results.

Demonstrates the RAG system answering finance-domain questions in
both English and Bangla.

Outputs:
  results/rag/demo_queries.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.rag.retriever import NewsRetriever, format_hits
from src.utils.logger import get_logger

logger = get_logger("phase9_demo")

DEMO_QUERIES = [
    # English queries, no filter
    ("Which banks had negative news recently?", None, None),
    ("Pharmaceutical company expansion plans", None, None),
    ("Dividend announcements in 2024", None, None),

    # English with stock filter
    ("BATBC dividend announcement", "BATBC", None),
    ("GP regulatory issues", "GP", None),

    # Bangla query
    ("ব্যাংক ঋণ জালিয়াতি", None, "bn"),  # bank loan fraud
    ("কোম্পানী বিস্তার", None, "bn"),     # company expansion

    # Stock-specific Bangla
    ("BRAC Bank", "BRACBANK", "bn"),
]


def main():
    logger.info("=" * 60)
    logger.info("PHASE 9 — RAG DEMO")
    logger.info("=" * 60)

    retriever = NewsRetriever(index_dir=_PROJECT_ROOT / "models/rag")

    out_lines = []
    for query, stock, lang in DEMO_QUERIES:
        out_lines.append("=" * 60)
        out_lines.append(f"QUERY: {query}")
        if stock:
            out_lines.append(f"  filter stock: {stock}")
        if lang:
            out_lines.append(f"  filter language: {lang}")
        out_lines.append("=" * 60)
        hits = retriever.query(query, top_k=3,
                               filter_stock=stock, filter_language=lang)
        out_lines.append(format_hits(hits, include_content=False))
        out_lines.append("")

    out_path = _PROJECT_ROOT / "results/rag/demo_queries.txt"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text("\n".join(out_lines))
    print("\n".join(out_lines))
    logger.info(f"\n💾 Saved demo to {out_path}")


if __name__ == "__main__":
    main()
