"""
Phase 6 — Sentiment Analysis visualization & summary report.

Reads:
    results/sentiment/news_scored.csv
    results/sentiment/stock_daily_sentiment.csv
    results/sentiment/correlation_per_stock.csv
    results/sentiment/correlation_summary.json

Writes:
    results/sentiment/plots/01_label_distribution.png
    results/sentiment/plots/02_sentiment_by_sector.png
    results/sentiment/plots/03_sentiment_by_event.png
    results/sentiment/plots/04_daily_sentiment_over_time.png
    results/sentiment/plots/05_correlation_by_lag.png
    results/sentiment/plots/06_finbert_vs_truth_confusion.png
    results/sentiment/plots/07_next_return_by_sentiment.png
    results/sentiment/summary_report.txt

Usage:
    python src/sentiment/visualize.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.config import SENTIMENT_RESULTS_DIR, SENTIMENT_PLOTS_DIR
from src.utils.logger import get_logger


logger = get_logger("sentiment.visualize")
PLOTS = SENTIMENT_PLOTS_DIR
PLOTS.mkdir(parents=True, exist_ok=True)

SCORED = SENTIMENT_RESULTS_DIR / "news_scored.csv"
DAILY = SENTIMENT_RESULTS_DIR / "stock_daily_sentiment.csv"
CORR = SENTIMENT_RESULTS_DIR / "correlation_per_stock.csv"
CORR_JSON = SENTIMENT_RESULTS_DIR / "correlation_summary.json"
REPORT_TXT = SENTIMENT_RESULTS_DIR / "summary_report.txt"


def _save(fig, name: str):
    out = PLOTS / name
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"   💾 {out.name}")


def plot_label_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    counts = df["pred_label"].value_counts()
    ax.bar(counts.index, counts.values, color=["#2ca02c", "#d62728", "#7f7f7f"])
    ax.set_title("Predicted Label Distribution (All Articles)")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 10, f"{v}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=10)

    ax = axes[1]
    counts_lang = (
        df.groupby(["language", "pred_label"]).size().unstack(fill_value=0)
    )
    counts_lang.plot(kind="bar", stacked=True, ax=ax, color=["#2ca02c", "#d62728", "#7f7f7f"])
    ax.set_title("Predicted Labels by Language")
    ax.set_ylabel("Count")
    ax.set_xlabel("Language")
    ax.legend(title="Predicted label")

    _save(fig, "01_label_distribution.png")


def plot_sentiment_by_sector(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    order = df.groupby("sector")["score"].mean().sort_values().index
    sns.boxplot(data=df, x="sector", y="score", order=order, ax=ax, palette="RdBu_r")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Sentiment Score Distribution by Sector")
    ax.set_xlabel("Sector")
    ax.set_ylabel("Sentiment score (signed)")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, "02_sentiment_by_sector.png")


def plot_sentiment_by_event(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    order = df.groupby("event_type")["score"].mean().sort_values().index
    sns.boxplot(data=df, x="event_type", y="score", order=order, ax=ax, palette="RdBu_r")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Sentiment Score Distribution by Event Type")
    ax.set_xlabel("Event type")
    ax.set_ylabel("Sentiment score (signed)")
    ax.tick_params(axis="x", rotation=30)
    _save(fig, "03_sentiment_by_event.png")


def plot_daily_sentiment_over_time(daily: pd.DataFrame):
    """Plot rolling 30-day mean sentiment across the market."""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Daily average across all stocks
    daily_avg = (
        daily.assign(date=pd.to_datetime(daily["date"]))
        .groupby("date")["weighted_score"].mean()
        .sort_index()
    )
    # 30-day rolling mean
    rolling = daily_avg.rolling(30, min_periods=1).mean()

    ax.plot(daily_avg.index, daily_avg.values, alpha=0.25, color="gray", label="Daily avg")
    ax.plot(rolling.index, rolling.values, color="#1f77b4", lw=2, label="30-day rolling mean")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Market-wide Sentiment Over Time (2010–2026)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean weighted sentiment score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "04_daily_sentiment_over_time.png")


def plot_correlation_by_lag(corr: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar: mean r per lag
    ax = axes[0]
    agg = corr.groupby("lag")["pearson_r"].mean()
    ax.bar(agg.index.astype(str), agg.values, color="#1f77b4", alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Mean Pearson r (sentiment → future return)")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Mean Pearson r across stocks")
    for i, v in enumerate(agg.values):
        ax.text(i, v + (0.005 if v >= 0 else -0.008), f"{v:+.3f}", ha="center", fontsize=9)

    # Heatmap: stock × lag
    ax = axes[1]
    pivot = corr.pivot(index="stock", columns="lag", values="pearson_r")
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                cbar_kws={"label": "Pearson r"})
    ax.set_title("Per-stock Pearson r by Lag")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Stock")

    _save(fig, "05_correlation_by_lag.png")


def plot_finbert_confusion(df: pd.DataFrame):
    """Confusion matrix of FinBERT (en) vs curated truth labels."""
    en = df[df["language"] == "en"].copy()
    if len(en) == 0:
        return

    from sklearn.metrics import confusion_matrix

    labels = ["negative", "neutral", "positive"]
    cm = confusion_matrix(en["true_label"], en["pred_label"], labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"FinBERT vs Curated Truth (English, n={len(en)})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Curated truth")
    _save(fig, "06_finbert_vs_truth_confusion.png")


def plot_next_return_by_sentiment(corr_json: dict):
    by = corr_json["next_return_by_sentiment"]
    cats = ["positive", "neutral", "negative"]
    means = [by[c]["mean"] for c in cats]
    counts = [by[c]["n"] for c in cats]
    colors = ["#2ca02c", "#7f7f7f", "#d62728"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cats, means, color=colors, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    for bar, m, n in zip(bars, means, counts):
        ax.text(bar.get_x() + bar.get_width()/2, m + (0.0001 if m >= 0 else -0.00015),
                f"{m:+.5f}\nn={n}", ha="center", fontsize=10)
    ax.set_title("Average Next-Day Return by Sentiment Polarity")
    ax.set_xlabel("Sentiment polarity")
    ax.set_ylabel("Mean next-day Target_Return_1d")
    _save(fig, "07_next_return_by_sentiment.png")


def write_summary_report(scored: pd.DataFrame, daily: pd.DataFrame, corr_json: dict) -> None:
    """Write a human-readable summary of the entire Phase 6."""
    lines = []
    lines.append("=" * 70)
    lines.append("📊 PHASE 6 — SENTIMENT ANALYSIS SUMMARY")
    lines.append("=" * 70)
    lines.append(f"\n� Corpus: {len(scored)} articles (2010-2026)")
    lines.append(f"   Stocks covered: {scored['stock'].nunique()}")
    lines.append(f"   Languages: {scored['language'].value_counts().to_dict()}")
    lines.append(f"   Event types: {scored['event_type'].value_counts().to_dict()}")

    lines.append("\n� Predicted-label distribution:")
    for label, n in scored["pred_label"].value_counts().items():
        pct = n / len(scored) * 100
        lines.append(f"   {label:10s} {n:5d} ({pct:5.1f}%)")

    # Mean by event type
    lines.append("\n📈 Mean sentiment score by event type:")
    by_event = scored.groupby("event_type")["score"].mean().sort_values(ascending=False)
    for et, m in by_event.items():
        lines.append(f"   {et:12s} {m:+.4f}")

    # Mean by sector
    lines.append("\n� Mean sentiment score by sector:")
    by_sector = scored.groupby("sector")["score"].mean().sort_values(ascending=False)
    for sec, m in by_sector.items():
        lines.append(f"   {sec:15s} {m:+.4f}")

    # Daily aggregation
    lines.append("\n📅 Daily aggregation:")
    lines.append(f"   (stock, date) pairs: {len(daily)}")
    lines.append(f"   Mean weighted_score: {daily['weighted_score'].mean():+.4f}")
    lines.append(f"   Median articles per (stock,date): {daily['n_articles'].median()}")

    # FinBERT accuracy vs truth (English)
    en = scored[scored["language"] == "en"]
    if len(en) > 0:
        acc = (en["true_label"] == en["pred_label"]).mean()
        lines.append("\n🤖 FinBERT classification quality (English vs curated truth):")
        lines.append(f"   Samples:  {len(en)}")
        lines.append(f"   Accuracy: {acc:.4f}")

    # Correlation summary
    lines.append("\n📊 Sentiment-price correlation:")
    lines.append("   Mean Pearson r by lag (across stocks):")
    for lag, vals in corr_json["mean_r_by_lag"].items():
        lines.append(
            f"      lag {lag:>2}  mean r={vals['mean']:+.4f}  std={vals['std']:.4f}  n={vals['n']}"
        )
    lines.append(f"\n   Lag-1 significant stocks (p<0.05): {corr_json['lag1_significant_count']}/{corr_json['lag1_total']}")

    lines.append("\n📈 Average next-day return by sentiment polarity:")
    by = corr_json["next_return_by_sentiment"]
    for cat in ["positive", "neutral", "negative"]:
        lines.append(
            f"   {cat:8s} mean={by[cat]['mean']:+.5f}  (n={by[cat]['n']})"
        )
    lines.append(f"   spread (pos - neg) = {by['spread_pos_minus_neg']:+.5f}")

    lines.append("\n" + "=" * 70)
    lines.append("📁 OUTPUTS")
    lines.append("=" * 70)
    lines.append(f"   {SCORED}")
    lines.append(f"   {DAILY}")
    lines.append(f"   {CORR}")
    lines.append(f"   {REPORT_TXT}")
    lines.append(f"   {PLOTS}/ (7 PNG plots)")

    report = "\n".join(lines)
    logger.info("\n" + report)
    REPORT_TXT.write_text(report + "\n")
    logger.info(f"💾 Wrote {REPORT_TXT}")


def main():
    logger.info("=" * 70)
    logger.info("📊 Phase 6 Sentiment Visualization")
    logger.info("=" * 70)

    scored = pd.read_csv(SCORED)
    daily = pd.read_csv(DAILY)
    corr = pd.read_csv(CORR)
    corr_json = json.loads(CORR_JSON.read_text())

    logger.info(f"📰 {len(scored)} scored articles")
    logger.info(f"📅 {len(daily)} (stock, date) daily rows")
    logger.info(f"📈 {len(corr)} (stock, lag) correlation cells")

    plot_label_distribution(scored)
    plot_sentiment_by_sector(scored)
    plot_sentiment_by_event(scored)
    plot_daily_sentiment_over_time(daily)
    plot_correlation_by_lag(corr)
    plot_finbert_confusion(scored)
    plot_next_return_by_sentiment(corr_json)

    write_summary_report(scored, daily, corr_json)

    logger.info("\n✅ Phase 6 visualization complete")
    logger.info(f"📁 Plots: {PLOTS}")
    logger.info(f"📄 Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()