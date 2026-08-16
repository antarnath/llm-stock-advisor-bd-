"""
Multimodal sequence dataset for Phase 7 (price + sentiment).

Builds (price_window, sentiment_window) -> next-day return pairs from:
- Per-stock price CSV: data/processed/{STOCK}_processed_v2.csv (Phase 2 output)
- Per-(stock, date) sentiment: results/sentiment/stock_daily_sentiment.csv (Phase 6)

Critical leak-free rules (mirrors Phase 4):
1.   EXCLUDE_COLS drops OHLCV + metadata + target columns from the PRICE stream only.
2.   Sentiment columns have their OWN StandardScaler, fit on TRAIN ONLY.
3.   Sentiment NaN policy: per-stock forward-fill on the time axis (sticky),
     with bfill once for leading NaNs and 0.0 fallback for stocks with zero news.
4.   Target is Target_Return_1d (next-day return), same as Phase 4.
5.   Time-based split identical to Phase 4 (0.72 / 0.08 / 0.20 chronological).

What this module exposes:
- MultimodalSequenceDataset (returns (price, sentiment, target) per window)
- prepare_multimodal_sequences(csv_path, sentiment_csv, stock_code, ...)
  -> (train_ds, val_ds, test_ds, price_scaler, sentiment_scaler,
      price_features, sentiment_features)

Usage:
    from src.data_processing.multimodal_dataset import (
        MultimodalSequenceDataset, prepare_multimodal_sequences,
    )
    train_ds, val_ds, test_ds, ps, ss, pf, sf = prepare_multimodal_sequences(
        csv_path, sentiment_csv, stock_code,
    )
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from src.utils.config import (
    MM_SEQUENCE_LENGTH,
    MM_SENTIMENT_COLS,
    MM_FILL_NA,
    DL_TRAIN_VAL_SPLIT,
    TEST_SIZE,
)
from src.data_processing.sequence_dataset import EXCLUDE_COLS


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultimodalSequenceDataset(Dataset):
    """Sliding-window dataset over (price, sentiment, target) for one stock.

    Each item is:
        price     : float32 (sequence_length, n_price_features)
        sentiment : float32 (sequence_length, n_sentiment_features)
        target    : float32 (1,) — next-day return at the row immediately AFTER
                    the window.

    Both modalities use the same window endpoints so they stay temporally aligned.
    """

    def __init__(
        self,
        price_features: np.ndarray,
        sentiment_features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int = MM_SEQUENCE_LENGTH,
    ):
        assert len(price_features) == len(sentiment_features) == len(targets), (
            f"price ({len(price_features)}), sentiment ({len(sentiment_features)}), "
            f"and targets ({len(targets)}) must be equal length"
        )
        self.price = price_features.astype(np.float32)
        self.sentiment = sentiment_features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return max(0, len(self.price) - self.sequence_length)

    def __getitem__(self, idx: int):
        price = self.price[idx : idx + self.sequence_length]
        sent = self.sentiment[idx : idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        return price, sent, y


# ---------------------------------------------------------------------------
# Time-based split (same boundaries as Phase 4)
# ---------------------------------------------------------------------------

def _time_split_indices(n: int, test_size: float, train_val_split: float):
    test_end = n
    val_end = int(n * (1.0 - test_size))
    train_end = int(val_end * (1.0 - train_val_split))
    return train_end, val_end, test_end


# ---------------------------------------------------------------------------
# Sentiment preparation: per-stock forward-fill, NaN cleanup
# ---------------------------------------------------------------------------

def _prepare_sentiment_for_stock(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    stock_code: str,
    sentiment_cols: list[str],
) -> pd.DataFrame:
    """Left-merge sentiment into price timeline and apply forward-fill NaN policy.

    Returns a DataFrame aligned to price_df with columns sentiment_cols added.
    Sentiment NaNs are forward-filled within the stock's timeline (sticky),
    then bfilled once for leading NaNs. Any residual NaN (entire stock has
    zero news) is replaced with 0.0.
    """
    # Restrict sentiment to the target stock
    sent = sentiment_df[sentiment_df["stock"] == stock_code].copy()
    if sent.empty:
        # No news for this stock at all — all-zero sentiment
        out = price_df[["date"]].copy()
        for c in sentiment_cols:
            out[c] = MM_FILL_NA
        return out

    sent["date"] = pd.to_datetime(sent["date"])
    sent = sent[["date"] + sentiment_cols].drop_duplicates(subset=["date"], keep="last")

    # Left-merge onto price timeline (preserves price df ordering, NaN where no news)
    price_dates = price_df[["date"]].copy()
    price_dates["date"] = pd.to_datetime(price_dates["date"])
    merged = price_dates.merge(sent, on="date", how="left")

    # Forward-fill per stock (single stock here, but defensive)
    merged = merged.sort_values("date").reset_index(drop=True)
    merged[sentiment_cols] = (
        merged.groupby("stock_code" if "stock_code" in merged.columns else merged.index // 1)[
            sentiment_cols
        ].ffill()
    )
    # The above groupby is a no-op for a single stock — do the ffill directly:
    merged[sentiment_cols] = merged[sentiment_cols].ffill()

    # Backfill once for leading NaNs (the earliest few rows)
    merged[sentiment_cols] = merged[sentiment_cols].bfill()

    # Final defensive fill: any residual NaN (entire stock had zero news) → 0.0
    residual_nan = merged[sentiment_cols].isna().any().any()
    if residual_nan:
        merged[sentiment_cols] = merged[sentiment_cols].fillna(MM_FILL_NA)

    return merged


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def prepare_multimodal_sequences(
    csv_path: str | Path,
    sentiment_csv: str | Path,
    stock_code: str,
    sequence_length: int = MM_SEQUENCE_LENGTH,
    test_size: float = TEST_SIZE,
    train_val_split: float = DL_TRAIN_VAL_SPLIT,
    sentiment_cols: list[str] | None = None,
):
    """Build scaled (price, sentiment) sequence datasets for one stock.

    Args:
        csv_path:        Path to {stock}_processed_v2.csv
        sentiment_csv:   Path to stock_daily_sentiment.csv (Phase 6 output)
        stock_code:      Stock code used for filtering sentiment rows
        sequence_length: Window length in trading days
        test_size:       Time-based test fraction (default 0.20)
        train_val_split: Fraction of train+val to use for training (default 0.10 → 72/8/20)
        sentiment_cols:  Which sentiment columns to use (default MM_SENTIMENT_COLS)

    Returns:
        train_ds, val_ds, test_ds : MultimodalSequenceDataset
        price_scaler              : fitted StandardScaler (for inference)
        sentiment_scaler          : fitted StandardScaler (for inference)
        price_features            : list[str] of price feature column names
        sentiment_features        : list[str] of sentiment feature column names
    """
    if sentiment_cols is None:
        sentiment_cols = list(MM_SENTIMENT_COLS)

    # Load price data
    price_df = pd.read_csv(csv_path, parse_dates=["date"])
    if "code" in price_df.columns:
        actual_code = str(price_df["code"].iloc[0])
        if actual_code != stock_code:
            # Trust the code column if available
            stock_code = actual_code

    # Load sentiment data
    sentiment_df = pd.read_csv(sentiment_csv, parse_dates=["date"])

    # Merge sentiment onto price timeline with forward-fill NaN policy
    sentiment_aligned = _prepare_sentiment_for_stock(
        price_df, sentiment_df, stock_code, sentiment_cols,
    )

    # Concatenate sentiment columns onto the price df
    price_df = price_df.reset_index(drop=True)
    sentiment_aligned = sentiment_aligned.reset_index(drop=True)
    assert len(price_df) == len(sentiment_aligned), (
        f"price ({len(price_df)}) and sentiment-aligned ({len(sentiment_aligned)}) "
        f"lengths must match after merge"
    )
    for c in sentiment_cols:
        price_df[c] = sentiment_aligned[c].values

    # Drop rows where target is NaN (mirrors Phase 4)
    price_df = price_df.dropna(subset=["Target_Return_1d"]).reset_index(drop=True)

    # Build price feature list (mirrors Phase 4 EXCLUDE_COLS, plus never includes sentiment)
    price_feature_cols = [c for c in price_df.columns if c not in EXCLUDE_COLS]
    # Defensive: explicitly remove sentiment columns from price features
    price_feature_cols = [c for c in price_feature_cols if c not in sentiment_cols]
    # Confirm no sentiment column leaks into price features
    leaked = set(sentiment_cols) & set(price_feature_cols)
    if leaked:
        raise RuntimeError(
            f"Sentiment columns {leaked} leaked into price feature list! "
            f"This is a bug — sentiment must have its own scaler."
        )

    target_col = "Target_Return_1d"
    X_price = price_df[price_feature_cols].values.astype(np.float32)
    X_sent = price_df[sentiment_cols].values.astype(np.float32)
    y = price_df[target_col].values.astype(np.float32)

    # Defensive NaN cleanup (mirrors Phase 4)
    for arr_name, arr in [("X_price", X_price), ("X_sent", X_sent), ("y", y)]:
        if np.isnan(arr).any():
            X_price = np.nan_to_num(X_price, nan=MM_FILL_NA)
            X_sent = np.nan_to_num(X_sent, nan=MM_FILL_NA)
            y = np.nan_to_num(y, nan=0.0)
            break

    # Time-based split (identical boundaries to Phase 4)
    train_end, val_end, test_end = _time_split_indices(
        len(price_df), test_size=test_size, train_val_split=train_val_split,
    )

    X_price_train, X_sent_train, y_train = (
        X_price[:train_end], X_sent[:train_end], y[:train_end],
    )
    X_price_val, X_sent_val, y_val = (
        X_price[train_end:val_end], X_sent[train_end:val_end], y[train_end:val_end],
    )
    X_price_test, X_sent_test, y_test = (
        X_price[val_end:test_end], X_sent[val_end:test_end], y[val_end:test_end],
    )

    # Two StandardScalers, both fit on TRAIN ONLY (no leakage)
    price_scaler = StandardScaler()
    X_price_train = price_scaler.fit_transform(X_price_train)
    X_price_val = price_scaler.transform(X_price_val)
    X_price_test = price_scaler.transform(X_price_test)

    sentiment_scaler = StandardScaler()
    X_sent_train = sentiment_scaler.fit_transform(X_sent_train)
    X_sent_val = sentiment_scaler.transform(X_sent_val)
    X_sent_test = sentiment_scaler.transform(X_sent_test)

    train_ds = MultimodalSequenceDataset(
        X_price_train, X_sent_train, y_train, sequence_length=sequence_length,
    )
    val_ds = MultimodalSequenceDataset(
        X_price_val, X_sent_val, y_val, sequence_length=sequence_length,
    )
    test_ds = MultimodalSequenceDataset(
        X_price_test, X_sent_test, y_test, sequence_length=sequence_length,
    )

    return (
        train_ds, val_ds, test_ds,
        price_scaler, sentiment_scaler,
        price_feature_cols, list(sentiment_cols),
    )


__all__ = [
    "MultimodalSequenceDataset",
    "prepare_multimodal_sequences",
]
