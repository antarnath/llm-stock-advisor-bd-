"""
Convert DSE-BD dataset to HuggingFace format and (optionally) push.

DSE-BD: 30 DSE stocks (2010-2026) + 1,560 multilingual news articles with
FinBERT sentiment labels + multimodal benchmark results.

Usage:
    .venv/bin/python scripts/dataset_to_hf.py --build       # just build the HF dataset locally
    HF_TOKEN=hf_xxx .venv/bin/python scripts/dataset_to_hf.py --push --repo-id your-username/dse-bd

Outputs:
    dataset_hf/                       - local HF dataset directory
    dataset_hf/README.md              - dataset card (auto-generated)
    dataset_hf/push_instructions.txt  - how to push if --push skipped
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from datasets import Dataset, DatasetDict, Features, Value, Sequence
import pandas as pd
import json

OUT_DIR = _PROJECT_ROOT / "dataset_hf"
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
SENTIMENT_NEWS = _PROJECT_ROOT / "results" / "sentiment" / "news_scored.csv"
SENTIMENT_DAILY = _PROJECT_ROOT / "results" / "sentiment" / "stock_daily_sentiment.csv"


def build_stock_prices_dataset():
    """Build a HF dataset of (stock, date, OHLCV, 27 technical indicators, target)."""
    csvs = sorted(PROCESSED_DIR.glob("*_processed_v2.csv"))
    csvs = [c for c in csvs if not c.stem.startswith("DSEX")]

    dfs = []
    for c in csvs:
        df = pd.read_csv(c, parse_dates=["date"])
        df["stock"] = c.stem.replace("_processed_v2", "")
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    full["date"] = full["date"].dt.strftime("%Y-%m-%d")
    # Keep essential columns to keep dataset small
    keep = ["date", "stock", "name", "sector",
            "open", "high", "low", "close", "volume", "trade",
            "SMA_5", "SMA_10", "SMA_20", "SMA_50", "SMA_100", "SMA_200",
            "EMA_12", "EMA_26", "EMA_50",
            "RSI_14", "MACD", "MACD_Signal",
            "Returns_1d", "Returns_5d", "Returns_20d", "Log_Returns",
            "BB_Position", "BB_Width", "BB_Upper", "BB_Lower",
            "ATR_14", "Volatility_30d", "Volatility_60d",
            "Volume_Ratio", "Volume_SMA_20",
            "Target_Return_1d", "Target_Direction_1d"]
    full = full[[c for c in keep if c in full.columns]]
    return Dataset.from_pandas(full, preserve_index=False)


def build_news_dataset():
    """Build a HF dataset of news articles with sentiment labels."""
    df = pd.read_csv(SENTIMENT_NEWS, parse_dates=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    keep = ["news_id", "date", "stock", "name", "sector", "language",
            "headline", "content", "event_type", "true_label", "pred_label",
            "score", "confidence", "prob_pos", "prob_neg", "prob_neu"]
    df = df[[c for c in keep if c in df.columns]]
    return Dataset.from_pandas(df, preserve_index=False)


def build_daily_sentiment_dataset():
    """Build a HF dataset of daily aggregated sentiment per stock."""
    df = pd.read_csv(SENTIMENT_DAILY, parse_dates=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return Dataset.from_pandas(df, preserve_index=False)


def build_dataset_card(stats: dict) -> str:
    return f"""# DSE-BD: Bangladesh Stock Exchange Multimodal Benchmark Dataset

## Dataset Summary

DSE-BD is a curated, leak-free dataset for the Bangladesh Stock Exchange (DSE)
designed for short-term stock forecasting research. It covers **30 DSE-listed
stocks** over **16 years (2010-2026)**, with three modalities:

1. **Daily OHLCV + 27 technical indicators** ({stats['n_stock_rows']:,} rows)
2. **1,560 financial news articles** in English ({stats['n_en']}) and Bangla ({stats['n_bn']}),
   pre-scored with FinBERT/BanglaBERT sentiment
3. **Daily aggregated sentiment** per stock ({stats['n_daily_sent']:,} rows)

## Motivation

Most existing DSE research uses random train/test splits, which leak future
information. DSE-BD enforces a strict time-based protocol (last 20% = test)
with all features computed only from past data — establishing a reproducible
benchmark for emerging-market forecasting.

## Languages

- English (en): {stats['n_en']} news articles
- Bangla (bn): {stats['n_bn']} news articles
- Multilingual multimodal fusion is enabled by language-tagged sentiment

## Stocks Covered

30 blue-chip DSE stocks across sectors: Bank, Pharmaceuticals, Cement,
Power, Telecom, Food, Textiles, Engineering, Fuel & Power.

## Splits (recommended)

- Train: rows where date < 2023-01-01 (~70%)
- Val:   rows where 2023-01-01 <= date < 2024-01-01 (~10%)
- Test:  rows where date >= 2024-01-01 (~20%)

## Citation

```bibtex
@dataset{{dse-bd-2026,
  title={{DSE-BD: A Leak-Free Multimodal Benchmark for Bangladesh Stock Exchange}},
  author={{Your Name}},
  year={{2026}},
  publisher={{HuggingFace Datasets}}
}}
```

## License

MIT License. News article content is summarized/curated for research use.

## Tasks

- `regression`: predict Target_Return_1d (next-day return)
- `classification`: predict Target_Direction_1d (next-day direction)
- `sentiment`: score financial news for a stock
- `multimodal-fusion`: combine price + sentiment features
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build HF dataset locally")
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--repo-id", type=str, default="your-username/dse-bd",
                        help="HuggingFace repo id (must be <user>/<dataset>)")
    args = parser.parse_args()

    if not (args.build or args.push):
        args.build = True

    print("=" * 60)
    print("DSE-BD HuggingFace Dataset Builder")
    print("=" * 60)

    # Build splits
    print("\n[1/3] Building stock prices dataset...")
    prices = build_stock_prices_dataset()
    print(f"   {len(prices):,} rows, {len(prices.column_names)} columns")

    print("\n[2/3] Building news articles dataset...")
    news = build_news_dataset()
    lang_counts = pd.Series(news["language"]).value_counts().to_dict()
    n_en = int(lang_counts.get("en", 0))
    n_bn = int(lang_counts.get("bn", 0))
    print(f"   {len(news):,} articles  (en={n_en}, bn={n_bn})")

    print("\n[3/3] Building daily sentiment dataset...")
    daily = build_daily_sentiment_dataset()
    print(f"   {len(daily):,} rows")

    # Stats
    stats = {
        "n_stock_rows": len(prices),
        "n_news": len(news),
        "n_en": n_en,
        "n_bn": n_bn,
        "n_daily_sent": len(daily),
    }

    dataset_dict = DatasetDict({
        "stock_prices": prices,
        "news": news,
        "daily_sentiment": daily,
    })

    if args.build:
        OUT_DIR.mkdir(exist_ok=True)
        dataset_dict.save_to_disk(str(OUT_DIR))
        readme = build_dataset_card(stats)
        (OUT_DIR / "README.md").write_text(readme)
        print(f"\n💾 Saved to {OUT_DIR}")
        print(f"   📄 README.md  (dataset card)")
        print(f"   📁 stock_prices/  news/  daily_sentiment/")
        print(f"\n   To push: HF_TOKEN=hf_xxx python scripts/dataset_to_hf.py --push --repo-id <user>/dse-bd")

    if args.push:
        from huggingface_hub import HfApi, whoami
        try:
            whoami()
            print(f"\n🚀 Pushing to https://huggingface.co/datasets/{args.repo_id}")
            dataset_dict.push_to_hub(args.repo_id)
            # Upload README separately
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(OUT_DIR / "README.md"),
                path_in_repo="README.md",
                repo_id=args.repo_id,
                repo_type="dataset",
            )
            print(f"✅ Pushed to https://huggingface.co/datasets/{args.repo_id}")
        except Exception as e:
            print(f"\n❌ Push failed: {e}")
            print("   Make sure you have:")
            print("   1. Set HF_TOKEN env var (https://huggingface.co/settings/tokens)")
            print("   2. Run: huggingface-cli login")
            print("   3. Created the repo at https://huggingface.co/new-dataset")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
