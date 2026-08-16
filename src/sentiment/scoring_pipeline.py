"""
Sentiment scoring pipeline for Phase 6.

Reads `data/raw/news/news_curated.csv`, runs each headline + content through
FinBERT (English) / BanglaLexicon (Bangla), and writes per-article scores to
`results/sentiment/news_scored.csv`.

Then aggregates per-article scores → per-stock-daily sentiment:
    results/sentiment/stock_daily_sentiment.csv
        columns: date, stock, n_articles, mean_score, weighted_score,
                 pos_count, neg_count, neu_count, pos_ratio, neg_ratio

Usage:
    python src/sentiment/scoring_pipeline.py
    python src/sentiment/scoring_pipeline.py --backend vader   # fallback test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.utils.config import (
    RAW_DATA_DIR, SENTIMENT_RESULTS_DIR, SENTIMENT_PLOTS_DIR,
)
from src.utils.logger import get_logger
from src.sentiment.analyzers import AutoAnalyzer, get_analyzer


logger = get_logger("sentiment.scoring")
NEWS_CSV = RAW_DATA_DIR / "news" / "news_curated.csv"

SCORED_CSV = SENTIMENT_RESULTS_DIR / "news_scored.csv"
DAILY_CSV = SENTIMENT_RESULTS_DIR / "stock_daily_sentiment.csv"

for d in [SENTIMENT_RESULTS_DIR, SENTIMENT_PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Per-article scoring
# ---------------------------------------------------------------------------

def score_articles(backend: str = "auto") -> Path:
    """Score all news articles and write news_scored.csv.

    Returns path to output CSV.
    """
    if not NEWS_CSV.exists():
        raise FileNotFoundError(f"News CSV not found: {NEWS_CSV}. Run news_curator.py first.")

    df = pd.read_csv(NEWS_CSV)
    logger.info(f"📰 Loaded {len(df)} news articles from {NEWS_CSV}")
    logger.info(f"   Languages: {df['language'].value_counts().to_dict()}")
    logger.info(f"   Stocks: {df['stock'].nunique()}")

    logger.info(f"🔄 Loading analyzer backend: {backend}...")
    analyzer = AutoAnalyzer() if backend == "auto" else get_analyzer(backend)

    logger.info(f"⚙️  Scoring {len(df)} articles...")
    t0 = time.time()

    rows = []
    for i, r in df.iterrows():
        # Combine headline + content as one document for richer context
        text = f"{r['headline']}. {r['content']}"
        # Bangla: content is the same as headline, so the combined text is fine
        result = analyzer.analyze(text)
        rows.append({
            "news_id": r["news_id"],
            "date": r["date"],
            "stock": r["stock"],
            "name": r["name"],
            "sector": r["sector"],
            "language": r["language"],
            "event_type": r["event_type"],
            "true_label": r["true_label"],
            "pred_label": result["label"],
            "score": result["score"],
            "confidence": result["confidence"],
            "prob_pos": result["probs"]["positive"],
            "prob_neg": result["probs"]["negative"],
            "prob_neu": result["probs"]["neutral"],
        })
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(df) - i - 1) / rate
            logger.info(f"   [{i+1}/{len(df)}] {rate:.1f} art/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    logger.info(f"✅ Scored {len(rows)} articles in {elapsed:.1f}s ({(elapsed/len(rows)*1000):.0f}ms/article)")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(SCORED_CSV, index=False)
    logger.info(f"💾 Wrote {SCORED_CSV}")

    return SCORED_CSV


# ---------------------------------------------------------------------------
# Per-stock daily aggregation
# ---------------------------------------------------------------------------

def aggregate_daily(input_csv: Path = SCORED_CSV) -> Path:
    """Aggregate per-article scores → per-(stock, date) daily sentiment."""
    df = pd.read_csv(input_csv, parse_dates=["date"])
    logger.info(f"📊 Loaded {len(df)} scored articles for aggregation")

    # Weights: confidence-scaled, so high-confidence articles count more
    df["signed_weight"] = df["confidence"] * df["score"]

    grouped = df.groupby(["stock", "date"]).agg(
        n_articles=("news_id", "count"),
        mean_score=("score", "mean"),
        weighted_score=("signed_weight", "mean"),
        mean_confidence=("confidence", "mean"),
        pos_count=("pred_label", lambda s: (s == "positive").sum()),
        neg_count=("pred_label", lambda s: (s == "negative").sum()),
        neu_count=("pred_label", lambda s: (s == "neutral").sum()),
    ).reset_index()

    grouped["pos_ratio"] = grouped["pos_count"] / grouped["n_articles"]
    grouped["neg_ratio"] = grouped["neg_count"] / grouped["n_articles"]

    grouped.to_csv(DAILY_CSV, index=False)
    logger.info(f"💾 Wrote {len(grouped)} (stock, date) rows → {DAILY_CSV}")

    # Stats
    logger.info(
        f"\n📈 Daily sentiment stats:\n"
        f"   Mean weighted_score: {grouped['weighted_score'].mean():+.4f}\n"
        f"   Stocks covered: {grouped['stock'].nunique()}\n"
        f"   Date range: {grouped['date'].min()} → {grouped['date'].max()}\n"
        f"   Articles per (stock,date) median: {grouped['n_articles'].median()}"
    )

    return DAILY_CSV


# ---------------------------------------------------------------------------
# Evaluation against true labels (only meaningful for English articles, since
# Bangla lexicon doesn't match FinBERT's 3-way label set perfectly)
# ---------------------------------------------------------------------------

def evaluate_against_truth(input_csv: Path = SCORED_CSV) -> dict:
    """Compute accuracy/F1 of predicted labels vs curated true_label.

    Macro F1 because classes are roughly balanced.
    """
    df = pd.read_csv(input_csv)
    # Filter to English where the prediction is from FinBERT (best signal)
    en_df = df[df["language"] == "en"].copy()
    if len(en_df) == 0:
        return {}

    from sklearn.metrics import (
        accuracy_score, f1_score, classification_report,
    )

    y_true = en_df["true_label"]
    y_pred = en_df["pred_label"]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    logger.info(
        f"\n🎯 Sentiment classification (English, FinBERT vs curated truth):\n"
        f"   Samples:       {len(en_df)}\n"
        f"   Accuracy:      {acc:.4f}\n"
        f"   Macro F1:      {macro_f1:.4f}\n"
        f"   Weighted F1:   {weighted_f1:.4f}\n"
    )
    report = classification_report(y_true, y_pred, zero_division=0)
    logger.info("\n" + report)

    return {
        "n": int(len(en_df)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="auto", choices=["auto", "finbert", "vader", "bangla"])
    p.add_argument("--skip-scoring", action="store_true", help="Re-aggregate from existing scores")
    args = p.parse_args()

    if not args.skip_scoring:
        score_articles(backend=args.backend)
    aggregate_daily()
    evaluate_against_truth()


if __name__ == "__main__":
    main()