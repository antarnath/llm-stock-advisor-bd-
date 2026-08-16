"""
Sequence Dataset for Deep Learning (Phase 4 — leak-free).

Builds sliding-window tensors of (seq_len, n_features) -> next-day return
from the same *_processed_v2.csv files used by Phase 3 baselines.

Critical leak-free rules (MUST match baseline_trainer_v2.EXCLUDE_COLS):
1.   EXCLUDE_COLS drops OHLCV + metadata + target columns.
2.   StandardScaler is fit on TRAIN ONLY.
3.   Target is Target_Return_1d (next-day return), not raw price.
4.   Sequence windows are built from scaled features (not raw OHLCV).

What this module exposes:
- EXCLUDE_COLS  (mirror of the baseline set; do not change without SSOT update)
- StockSequenceDataset  (torch Dataset returning (X, y) per window)
- prepare_stock_sequences(df, ...) -> (train_ds, val_ds, test_ds, scaler, feature_cols)

Usage:
    from src.data_processing.sequence_dataset import (
        StockSequenceDataset, prepare_stock_sequences,
    )
    train_ds, val_ds, test_ds, scaler, feats = prepare_stock_sequences(df)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

# Project config
from src.utils.config import (
    DL_SEQUENCE_LENGTH,
    DL_TRAIN_VAL_SPLIT,
    TEST_SIZE,
    RANDOM_STATE,
)


# Source of truth for leakage-prone columns. Must match baseline_trainer_v2.
EXCLUDE_COLS = {
    # Metadata
    "date", "code", "name", "sector",
    # Targets (the variable we're predicting)
    "Target_Return_1d", "Target_Price_1d",
    # Same-day raw prices / volume — leaking current-day info into
    # the prediction of tomorrow's return.
    "open", "high", "low", "close", "volume", "trade", "value",
}


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the columns that are safe to use as DL features."""
    return [c for c in df.columns if c not in EXCLUDE_COLS]


class StockSequenceDataset(Dataset):
    """Sliding-window dataset over a single stock's scaled features.

    Each item is:
        X: float32 tensor of shape (sequence_length, n_features)
        y: float32 tensor of shape (1,) — the next-day return at the
           timestamp immediately AFTER the window.

    The dataset is built from a CONTIGUOUS scaled feature matrix (no
    shuffling — temporal order is preserved). The window ending at index i
    contains features [i-seq_len+1, ..., i] and the target is the row
    target value at index i (next-day return defined as in the v2 processors).
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int = DL_SEQUENCE_LENGTH,
    ):
        assert len(features) == len(targets), (
            f"features and targets must be equal length, got {len(features)} vs {len(targets)}"
        )
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        # If we have N rows, we can form (N - seq_len) windows:
        # window k = rows [k, k+seq_len) features, target = row k+seq_len.
        return max(0, len(self.features) - self.sequence_length)

    def __getitem__(self, idx: int):
        # X is the past `seq_len` rows ending at idx+seq_len-1.
        # y is the target at idx+seq_len (the day to predict).
        x = self.features[idx : idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        return x, y


def _time_split_indices(n: int, test_size: float, train_val_split: float):
    """Time-based split indices (no shuffle).

    Returns (train_end, val_end, test_end) where:
      - train rows: [0, train_end)
      - val rows:   [train_end, val_end)
      - test rows:  [val_end, test_end = n)
    """
    test_end = n
    val_end = int(n * (1.0 - test_size))
    train_end = int(val_end * (1.0 - train_val_split))
    return train_end, val_end, test_end


def prepare_stock_sequences(
    df: pd.DataFrame,
    sequence_length: int = DL_SEQUENCE_LENGTH,
    test_size: float = TEST_SIZE,
    train_val_split: float = DL_TRAIN_VAL_SPLIT,
):
    """Build scaled train/val/test sequence datasets for one stock.

    Steps:
      1.  Drop rows with NaN target (Target_Return_1d).
      2.  Select feature columns (EXCLUDE_COLS applied).
      3.  Time-based split (no shuffle).
      4.  Fit StandardScaler on TRAIN features only.
      5.  Transform train/val/test features.
      6.  Wrap each split in a StockSequenceDataset.

    Returns:
        train_ds, val_ds, test_ds : StockSequenceDataset
        scaler  : fitted StandardScaler (for inference)
        feature_cols : list[str] of feature column names
    """
    # Drop rows where target is NaN (first/last rows of the return series).
    df = df.dropna(subset=["Target_Return_1d"]).reset_index(drop=True)

    feature_cols = select_feature_columns(df)
    target_col = "Target_Return_1d"

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    # Final defensive NaN check on features (should be clean after v2 processing).
    if np.isnan(X).any() or np.isnan(y).any():
        # Replace any residual NaNs with 0 to avoid sequence construction errors.
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)

    train_end, val_end, test_end = _time_split_indices(
        len(df), test_size=test_size, train_val_split=train_val_split
    )

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:test_end], y[val_end:test_end]

    # Fit scaler on TRAIN ONLY (no leakage).
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_ds = StockSequenceDataset(X_train, y_train, sequence_length=sequence_length)
    val_ds = StockSequenceDataset(X_val, y_val, sequence_length=sequence_length)
    test_ds = StockSequenceDataset(X_test, y_test, sequence_length=sequence_length)

    return train_ds, val_ds, test_ds, scaler, feature_cols


__all__ = [
    "EXCLUDE_COLS",
    "StockSequenceDataset",
    "prepare_stock_sequences",
    "select_feature_columns",
]
