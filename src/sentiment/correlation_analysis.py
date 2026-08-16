"""
Sentiment-price correlation analysis for Phase 6.

Joins per-stock-daily sentiment with processed stock prices (Target_Return_1d
and next-day returns) and reports:
  - Pearson correlation between same-day sentiment score and same-day return
  - Lagged correlations: sentiment at t vs return at t+1, t+2, t+5, t+10
  - Granger-style lead-lag observation (we run simple lag correlations; full
    Granger F-tests are reported as future work)
  - Per-stock and aggregate summary

Outputs:
  results/sentiment/correlation_per_stock.csv
  results/sentiment/correlation_summary.txt
  results/sentiment/correlation_summary.json

Usage:
    python src/sentiment/correlation_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from src.utils.config import PROCESSED_DATA_DIR, SENTIMENT_RESULTS_DIR, SENTIMENT_PLOTS_DIR
from src.utils.logger import get_logger


logger = get_logger("sentiment.correlation")

DAILY_SENTIMENT = SENTIMENT_RESULTS_DIR / "stock_daily_sentiment.csv"
PER_STOCK_OUT = SENTIMENT_RESULTS_DIR / "correlation_per_stock.csv"
SUMMARY_TXT = SENTIMENT_RESULTS_DIR / "correlation_summary.txt"
SUMMARY_JSON = SENTIMENT_RESULTS_DIR / "correlation_summary.json"

LAGS = [0, 1, 2, 5, 10]  # days ahead — does sentiment predict future return?


def _load_returns(stock: str) -> pd.DataFrame:
    """Load processed_v2 CSV with Target_Return_1d column."""
    df = pd.read_csv(
        PROCESSED_DATA_DIR / f"{stock}_processed_v2.csv",
        parse_dates=["date"],
    )
    return df[["date", "Target_Return_1d"]].dropna()


def _corr_for_pair(sent: pd.DataFrame, ret: pd.DataFrame, lag: int) -> dict:
    """Pearson r between sentiment_score (today) and return (today+lag).

    `lag=0` → same day
    `lag=1` → sentiment today, return tomorrow
    """
    s = sent.copy()
    r = ret.copy()
    if lag > 0:
        # shift returns backward so today's sentiment aligns with future return
        r["ret_future"] = r["Target_Return_1d"].shift(-lag)
        merged = s.merge(
            r[["date", "ret_future"]],
            on="date", how="inner",
        ).dropna()
        x = merged["weighted_score"].values
        y = merged["ret_future"].values
        label = f"return_t+{lag}"
    else:
        merged = s.merge(r, on="date", how="inner").dropna()
        x = merged["weighted_score"].values
        y = merged["Target_Return_1d"].values
        label = "return_t"

    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return {"lag": lag, "n": int(len(x)), "pearson_r": float("nan"), "p_value": float("nan"), "target": label}

    r_val, p_val = pearsonr(x, y)
    return {
        "lag": lag,
        "n": int(len(x)),
        "pearson_r": float(r_val),
        "p_value": float(p_val),
        "target": label,
    }


def analyze_stock(stock: str, sent: pd.DataFrame) -> dict:
    """Compute lag correlations for one stock."""
    stock_sent = sent[sent["stock"] == stock].copy()
    if len(stock_sent) == 0:
        return {"stock": stock, "results": []}

    ret = _load_returns(stock)
    if len(ret) == 0:
        return {"stock": stock, "results": []}

    stock_sent["date"] = pd.to_datetime(stock_sent["date"])
    results = []
    for lag in LAGS:
        results.append(_corr_for_pair(stock_sent, ret, lag))

    return {"stock": stock, "results": results}


def main():
    if not DAILY_SENTIMENT.exists():
        logger.error(f"❌ {DAILY_SENTIMENT} not found. Run scoring_pipeline.py first.")
        return

    sent = pd.read_csv(DAILY_SENTIMENT, parse_dates=["date"])
    stocks = sorted(sent["stock"].unique())
    logger.info(f"📊 Analyzing {len(stocks)} stocks × {len(LAGS)} lags")

    rows = []
    for stock in stocks:
        out = analyze_stock(stock, sent)
        for r in out["results"]:
            row = {"stock": stock, **r}
            rows.append(row)
        # per-stock lag-0 log
        lag0 = next((r for r in out["results"] if r["lag"] == 0), None)
        if lag0:
            logger.info(
                f"   {stock:15s} lag0 r={lag0['pearson_r']:+.3f}  "
                f"p={lag0['p_value']:.3f}  n={lag0['n']}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(PER_STOCK_OUT, index=False)
    logger.info(f"\n💾 Per-stock correlations: {PER_STOCK_OUT}")

    # ---- Aggregate stats ----
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("📊 SENTIMENT–PRICE CORRELATION SUMMARY (Phase 6)")
    summary_lines.append("=" * 70)
    summary_lines.append(f"\nStocks analyzed: {len(stocks)}")
    summary_lines.append(f"Lags tested: {LAGS} (days)")
    summary_lines.append(f"Total (stock, lag) cells: {len(df)}")

    # mean r per lag (across stocks)
    summary_lines.append("\n📈 Mean Pearson r by lag (across stocks):")
    agg = df.groupby("lag")["pearson_r"].agg(["mean", "median", "std", "count"]).reset_index()
    for _, a in agg.iterrows():
        summary_lines.append(
            f"   lag {int(a['lag']):2d}  mean r={a['mean']:+.4f}  "
            f"median={a['median']:+.4f}  std={a['std']:.4f}  n={int(a['count'])}"
        )

    # significant stocks at lag=1 (sentiment today predicts return tomorrow)
    lag1 = df[df["lag"] == 1].dropna()
    if len(lag1) > 0:
        sig = lag1[lag1["p_value"] < 0.05]
        summary_lines.append(
            f"\n📌 Significant lead-lag (lag=1, p<0.05): "
            f"{len(sig)}/{len(lag1)} stocks"
        )
        if len(sig) > 0:
            summary_lines.append("   Top 5 by |r|:")
            top = sig.reindex(sig["pearson_r"].abs().sort_values(ascending=False).index).head(5)
            for _, r in top.iterrows():
                summary_lines.append(
                    f"      {r['stock']:15s}  r={r['pearson_r']:+.3f}  p={r['p_value']:.4f}  n={int(r['n'])}"
                )

    # positive vs negative sentiment: do they predict direction?
    # We approximate with: mean next-day return on positive-sentiment days vs negative-sentiment days
    summary_lines.append("\n📊 Average next-day return by sentiment polarity:")
    sent["date"] = pd.to_datetime(sent["date"])
    enriched_rows = []
    for stock in stocks:
        ss = sent[sent["stock"] == stock].copy()
        rr = _load_returns(stock)
        m = ss.merge(rr, on="date", how="inner")
        m["next_return"] = m["Target_Return_1d"].shift(-1)
        enriched_rows.append(m)
    enriched = pd.concat(enriched_rows, ignore_index=True).dropna(subset=["next_return"])
    pos = enriched[enriched["weighted_score"] > 0.05]["next_return"]
    neg = enriched[enriched["weighted_score"] < -0.05]["next_return"]
    neu = enriched[enriched["weighted_score"].between(-0.05, 0.05)]["next_return"]
    summary_lines.append(
        f"   Positive sentiment days:  mean next-day return = {pos.mean():+.5f}  (n={len(pos)})"
    )
    summary_lines.append(
        f"   Negative sentiment days:  mean next-day return = {neg.mean():+.5f}  (n={len(neg)})"
    )
    summary_lines.append(
        f"   Neutral sentiment days:   mean next-day return = {neu.mean():+.5f}  (n={len(neu)})"
    )
    summary_lines.append(
        f"   Spread (pos - neg):       {pos.mean() - neg.mean():+.5f}"
    )

    summary = "\n".join(summary_lines)
    logger.info("\n" + summary)

    # write summary files
    with open(SUMMARY_TXT, "w") as f:
        f.write(summary + "\n")
    logger.info(f"\n💾 Wrote {SUMMARY_TXT}")

    summary_json = {
        "stocks_analyzed": len(stocks),
        "lags": LAGS,
        "mean_r_by_lag": {
            int(row["lag"]): {
                "mean": float(row["mean"]),
                "median": float(row["median"]),
                "std": float(row["std"]),
                "n": int(row["count"]),
            }
            for _, row in agg.iterrows()
        },
        "lag1_significant_count": int(((df["lag"] == 1) & (df["p_value"] < 0.05)).sum()),
        "lag1_total": int((df["lag"] == 1).sum()),
        "next_return_by_sentiment": {
            "positive": {"mean": float(pos.mean()), "n": int(len(pos))},
            "negative": {"mean": float(neg.mean()), "n": int(len(neg))},
            "neutral":  {"mean": float(neu.mean()), "n": int(len(neu))},
            "spread_pos_minus_neg": float(pos.mean() - neg.mean()),
        },
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary_json, f, indent=2)
    logger.info(f"💾 Wrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()