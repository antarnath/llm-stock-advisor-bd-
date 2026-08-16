"""
Deep Learning Inference — Phase 4 (LSTM, leak-free).

Loads each {STOCK}_best_lstm.pkl sidecar + matching .pt state dict,
reconstructs the StockLSTM, and produces N-day-ahead rolling return
predictions using the iterative feature-update trick from
src.inference.predict_v2 (so day-2..N differ from day-1).

Output: results/deep_learning/predictions_5days.csv

Usage:
    python src/inference/dl_predict.py
    python src/inference/dl_predict.py --days 5
    python src/inference/dl_predict.py --days 1
"""

from __future__ import annotations

import argparse
import pickle
import sys
from datetime import timedelta
from pathlib import Path

# Ensure `src` is importable when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

import torch

from src.utils.config import (
    PROCESSED_DATA_DIR as DATA_DIR,
    DEEP_LEARNING_MODELS_DIR as MODELS_DIR,
    DL_RESULTS_DIR as RESULTS_DIR,
    DL_SEQUENCE_LENGTH,
    get_device,
)
from src.utils.logger import get_logger
from src.training.architectures.lstm import StockLSTM


logger = get_logger("dl_predict")


# Same updateable-features convention as src.inference.predict_v2
UPDATEABLE_RETURNS = ["Returns_1d", "Returns_5d", "Returns_20d", "Log_Returns"]


def load_dl_model(stock_code: str, suffix: str = "_lstm"):
    """Load {stock}_best_lstm.pkl sidecar + matching state_dict.

    Returns (sidecar_dict, StockLSTM model) or None if not found.
    """
    sidecar_path = MODELS_DIR / f"{stock_code}_best{suffix}.pkl"
    state_path = MODELS_DIR / f"{stock_code}_best{suffix}.pt"
    if not sidecar_path.exists() or not state_path.exists():
        return None

    with open(sidecar_path, "rb") as f:
        sidecar = pickle.load(f)

    cfg = sidecar["config"]
    model = StockLSTM(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        output_dim=cfg["output_dim"],
        dropout=cfg["dropout"],
    )
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return sidecar, model


def predict_stock(stock_code: str, days: int = 5, suffix: str = "_lstm"):
    """Predict next N days of returns (and convert to price) for a stock.

    Returns: list[dict] or None.
    """
    data_path = DATA_DIR / f"{stock_code}_processed_v2.csv"
    if not data_path.exists():
        logger.warning(f"   ⚠️  No processed data: {data_path.name}")
        return None

    df = pd.read_csv(data_path, parse_dates=["date"])
    if df.empty:
        logger.warning(f"   ⚠️  Empty processed data for {stock_code}")
        return None

    loaded = load_dl_model(stock_code, suffix=suffix)
    if loaded is None:
        logger.warning(f"   ⚠️  No model for {stock_code}")
        return None

    sidecar, model = loaded
    feature_cols = sidecar["features"]
    scaler = sidecar["scaler"]

    # Drop rows with NaN target so the last "known" row matches what the model saw
    df = df.dropna(subset=["Target_Return_1d"]).reset_index(drop=True)

    # Build the last `sequence_length` feature rows (scaled) and the working row.
    # We need the most-recent seq_len rows of *scaled* features.
    feature_matrix = df[feature_cols].values.astype(np.float32)
    feature_matrix = scaler.transform(feature_matrix)
    if len(feature_matrix) < DL_SEQUENCE_LENGTH:
        logger.warning(
            f"   ⚠️  {stock_code}: only {len(feature_matrix)} rows, "
            f"need >= {DL_SEQUENCE_LENGTH}"
        )
        return None

    # Window for first prediction: last seq_len scaled rows.
    window = feature_matrix[-DL_SEQUENCE_LENGTH:].copy()  # shape (seq_len, n_features)

    # Track un-scaled "last seen" values for the updateable features so we can
    # update them after each predicted step.
    last_unscaled = df.iloc[-1][feature_cols].astype(float).copy()
    current_close = float(df.iloc[-1]["close"])
    current_date = df.iloc[-1]["date"]

    # Track recent Returns_1d history (un-scaled) for Returns_5d / Returns_20d.
    recent_returns = list(df["Returns_1d"].dropna().tail(20).astype(float).values)

    device = get_device() or torch.device("cpu")
    model.to(device)

    predictions = []
    working_close = current_close

    for day in range(1, days + 1):
        x = torch.from_numpy(window).float().unsqueeze(0).to(device)  # (1, seq_len, n_feat)
        with torch.no_grad():
            pred_return = float(model(x).view(-1).cpu().numpy()[0])

        # Cap absurd predictions to ±20% (sanity)
        pred_return = float(np.clip(pred_return, -0.20, 0.20))

        pred_close = working_close * (1.0 + pred_return)
        pred_date = current_date + timedelta(days=day)
        while pred_date.weekday() >= 5:
            pred_date += timedelta(days=1)

        predictions.append({
            "stock": stock_code,
            "prediction_date": pred_date.strftime("%Y-%m-%d"),
            "days_ahead": day,
            "current_price": round(current_close, 2),
            "predicted_price": round(pred_close, 2),
            "predicted_return_%": round(pred_return * 100, 4),
            "model_type": "lstm_v1",
        })

        # ---- Update unscaled working row ----
        recent_returns.append(pred_return)
        if len(recent_returns) > 20:
            recent_returns.pop(0)
        if "Returns_1d" in last_unscaled.index:
            last_unscaled["Returns_1d"] = pred_return
        if "Returns_5d" in last_unscaled.index and len(recent_returns) >= 5:
            last_unscaled["Returns_5d"] = float(
                np.prod(1.0 + np.array(recent_returns[-5:])) - 1.0
            )
        if "Returns_20d" in last_unscaled.index and len(recent_returns) >= 20:
            last_unscaled["Returns_20d"] = float(
                np.prod(1.0 + np.array(recent_returns[-20:])) - 1.0
            )
        if "Log_Returns" in last_unscaled.index:
            last_unscaled["Log_Returns"] = (
                float(np.log(1.0 + pred_return)) if pred_return > -1.0 else 0.0
            )

        # Rebuild SCALED row and slide the window.
        new_row_unscaled = last_unscaled.values.astype(np.float32).reshape(1, -1)
        new_row_scaled = scaler.transform(new_row_unscaled)[0]
        window = np.vstack([window[1:], new_row_scaled])

        working_close = pred_close

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Generate LSTM predictions (Phase 4).")
    parser.add_argument("--days", type=int, default=5, help="Days ahead to predict (default: 5).")
    parser.add_argument("--max-stocks", type=int, default=None, help="Limit to N stocks (debug).")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("📈 LSTM Predictions (Phase 4 — Leak-Free)")
    logger.info("=" * 70)

    files = sorted(DATA_DIR.glob("*_processed_v2.csv"))
    if not files:
        logger.error(f"❌ No *_processed_v2.csv files in {DATA_DIR}")
        return

    if args.max_stocks:
        files = files[: args.max_stocks]

    stock_codes = [f.stem.replace("_processed_v2", "") for f in files]
    logger.info(f"📊 Predicting {args.days}-day returns for {len(stock_codes)} stocks\n")

    all_predictions = []
    successful = 0
    failed = []

    for i, code in enumerate(stock_codes, 1):
        logger.info(f"[{i}/{len(stock_codes)}] {code}...")
        try:
            preds = predict_stock(code, days=args.days)
            if preds:
                all_predictions.extend(preds)
                successful += 1
                d1 = preds[0]
                direction = "📈" if d1["predicted_return_%"] > 0 else "📉"
                logger.info(
                    f"   {direction} Day 1: ৳{d1['current_price']:.2f} → "
                    f"৳{d1['predicted_price']:.2f} ({d1['predicted_return_%']:+.2f}%)"
                )
            else:
                failed.append(code)
        except Exception as e:
            logger.error(f"   ❌ Error predicting {code}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(code)

    if not all_predictions:
        logger.error("❌ No predictions generated.")
        return

    pred_df = pd.DataFrame(all_predictions)
    output_path = RESULTS_DIR / f"predictions_{args.days}days.csv"
    pred_df.to_csv(output_path, index=False)

    logger.info("\n" + "=" * 70)
    logger.info(
        f"📊 PREDICTION SUMMARY (Day 1, {successful}/{len(stock_codes)} stocks)"
    )
    logger.info("=" * 70)

    day1 = pred_df[pred_df["days_ahead"] == 1].sort_values(
        "predicted_return_%", ascending=False
    )

    logger.info("\n🟢 TOP 10 EXPECTED GAINERS (Day 1):")
    logger.info("-" * 70)
    for _, row in day1.head(10).iterrows():
        logger.info(
            f"  {row['stock']:15s} ৳{row['current_price']:>8.2f} → "
            f"�{row['predicted_price']:>8.2f} ({row['predicted_return_%']:+.2f}%)"
        )

    logger.info("\n🔴 TOP 10 EXPECTED LOSERS (Day 1):")
    logger.info("-" * 70)
    for _, row in day1.tail(10).iterrows():
        logger.info(
            f"  {row['stock']:15s} ৳{row['current_price']:>8.2f} → "
            f"৳{row['predicted_price']:>8.2f} ({row['predicted_return_%']:+.2f}%)"
        )

    # Loop sanity
    n_distinct = (
        pred_df.groupby("stock")["predicted_return_%"]
        .nunique()
        .reindex(stock_codes, fill_value=0)
    )
    n_all_same = int((n_distinct == 1).sum())
    logger.info(
        f"\n🔁 Loop sanity: {n_all_same}/{len(stock_codes)} stocks had IDENTICAL "
        "returns across all days (should be 0)."
    )

    logger.info(f"\n💾 Predictions saved: {output_path}")
    logger.info(
        f"� Total rows: {len(pred_df)} "
        f"({len(stock_codes)} stocks × {args.days} days)"
    )
    logger.info("=" * 70)
    logger.info("✨ LSTM prediction complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
