"""
Multimodal Inference — Phase 7 (price + sentiment).

Loads each {STOCK}_best_mm_{fusion}.pkl sidecar + matching .pt state_dict,
reconstructs the multimodal LSTM (early or late fusion), and produces N-day
return predictions using the iterative feature-update trick from dl_predict.

KEY DIFFERENCE vs Phase 4 inference:
- Sentiment window is FROZEN at the last observed 60-day window and replicated
  across all forecast steps. Sentiment is exogenous (driven by news events,
  not modeled by the LSTM), so we do not iterate it.
- Price features are iteratively updated for Returns_1d/5d/20d/Log_Returns,
  matching Phase 4.

Output: results/multimodal/predictions_{N}days.csv (same schema as Phase 4,
plus a `fusion_strategy` column).

Usage:
    python src/inference/mm_predict.py --fusion early
    python src/inference/mm_predict.py --fusion both --days 5
    python src/inference/mm_predict.py --stock GP --fusion early --days 5
"""

from __future__ import annotations

import argparse
import pickle
import sys
from datetime import timedelta
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.utils.config import (
    PROCESSED_DATA_DIR as DATA_DIR,
    SENTIMENT_RESULTS_DIR,
    MULTIMODAL_MODELS_DIR as MODELS_DIR,
    MULTIMODAL_RESULTS_DIR as RESULTS_DIR,
    MM_SEQUENCE_LENGTH,
    MM_FILL_NA,
    MM_SENTIMENT_COLS,
    get_device,
)
from src.utils.logger import get_logger
from src.training.architectures.multimodal_lstm import build_multimodal
from src.data_processing.multimodal_dataset import _prepare_sentiment_for_stock


logger = get_logger("mm_predict")

UPDATEABLE_RETURNS = ["Returns_1d", "Returns_5d", "Returns_20d", "Log_Returns"]


def load_mm_model(stock_code: str, fusion: str):
    """Load {stock}_best_mm_{fusion}.pkl sidecar + matching state_dict.

    Returns (sidecar_dict, model) or None if not found.
    """
    sidecar_path = MODELS_DIR / f"{stock_code}_best_mm_{fusion}.pkl"
    state_path = MODELS_DIR / f"{stock_code}_best_mm_{fusion}.pt"
    if not sidecar_path.exists() or not state_path.exists():
        return None

    with open(sidecar_path, "rb") as f:
        sidecar = pickle.load(f)

    cfg = sidecar["config"]
    model = build_multimodal(cfg)
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return sidecar, model


def _build_last_sentiment_window(
    df: pd.DataFrame,
    sentiment_csv: Path,
    stock_code: str,
    sentiment_cols: list[str],
    sequence_length: int,
) -> np.ndarray:
    """Build the last `sequence_length` sentiment rows (scaled) for inference.

    Mirrors the dataset preparation: left-merge sentiment onto price timeline,
    forward-fill per-stock, residual-fill with 0.0. Then apply the trained
    sentiment scaler and take the last seq_len rows.
    """
    sentiment_df = pd.read_csv(sentiment_csv, parse_dates=["date"])
    aligned = _prepare_sentiment_for_stock(df, sentiment_df, stock_code, sentiment_cols)
    # Re-align to df index order
    aligned = aligned.reset_index(drop=True)
    df_reset = df.reset_index(drop=True)
    if len(aligned) != len(df_reset):
        # Should not happen but be defensive
        n = min(len(aligned), len(df_reset))
        aligned = aligned.iloc[:n].reset_index(drop=True)
        df_reset = df_reset.iloc[:n].reset_index(drop=True)
    raw = aligned[sentiment_cols].values.astype(np.float32)
    raw = np.nan_to_num(raw, nan=MM_FILL_NA)
    return raw


def predict_stock(stock_code: str, days: int = 5, fusion: str = "early") -> list[dict] | None:
    """Predict next N days of returns for a stock using the multimodal model.

    Returns list[dict] with same schema as Phase 4 + `fusion_strategy`.
    """
    data_path = DATA_DIR / f"{stock_code}_processed_v2.csv"
    if not data_path.exists():
        logger.warning(f"   ⚠️  No processed data: {data_path.name}")
        return None

    df = pd.read_csv(data_path, parse_dates=["date"])
    if df.empty:
        logger.warning(f"   ⚠️  Empty processed data for {stock_code}")
        return None

    loaded = load_mm_model(stock_code, fusion=fusion)
    if loaded is None:
        logger.warning(f"   ⚠️  No model for {stock_code} ({fusion})")
        return None

    sidecar, model = loaded
    price_features = sidecar["features"]
    sentiment_features = sidecar["sentiment_features"]
    price_scaler = sidecar["scaler"]
    sentiment_scaler = sidecar["sentiment_scaler"]

    # Drop NaN target rows (matches what training saw)
    df = df.dropna(subset=["Target_Return_1d"]).reset_index(drop=True)

    # Build price window (scaled)
    price_matrix = df[price_features].values.astype(np.float32)
    price_matrix = price_scaler.transform(price_matrix)
    if len(price_matrix) < MM_SEQUENCE_LENGTH:
        logger.warning(
            f"   ⚠️  {stock_code}: only {len(price_matrix)} rows, need >= {MM_SEQUENCE_LENGTH}"
        )
        return None
    price_window = price_matrix[-MM_SEQUENCE_LENGTH:].copy()

    # Build sentiment window (scaled, FROZEN at last-known)
    sentiment_raw = _build_last_sentiment_window(
        df, SENTIMENT_RESULTS_DIR / "stock_daily_sentiment.csv",
        stock_code, sentiment_features, MM_SEQUENCE_LENGTH,
    )
    sentiment_matrix = sentiment_scaler.transform(sentiment_raw)
    # Defensive NaN handling post-transform (zero-variance columns may have NaN)
    sentiment_matrix = np.nan_to_num(sentiment_matrix, nan=0.0)
    sentiment_window = sentiment_matrix[-MM_SEQUENCE_LENGTH:].copy()

    # Working un-scaled row for iterative price update
    last_unscaled = df.iloc[-1][price_features].astype(float).copy()
    current_close = float(df.iloc[-1]["close"])
    current_date = df.iloc[-1]["date"]

    # Track recent Returns_1d history for Returns_5d / Returns_20d
    recent_returns = list(df["Returns_1d"].dropna().tail(20).astype(float).values)

    device = get_device() or torch.device("cpu")
    model.to(device)

    predictions = []
    working_close = current_close

    for day in range(1, days + 1):
        # Build model input (both modalities)
        x_price = torch.from_numpy(price_window).float().unsqueeze(0).to(device)  # (1, seq, P)
        x_sent = torch.from_numpy(sentiment_window).float().unsqueeze(0).to(device)  # (1, seq, S)
        with torch.no_grad():
            pred_return = float(model(x_price, x_sent).view(-1).cpu().numpy()[0])

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
            "model_type": f"multimodal_{fusion}",
            "fusion_strategy": fusion,
        })

        # ---- Iteratively update PRICE features only ----
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

        new_row_unscaled = last_unscaled.values.astype(np.float32).reshape(1, -1)
        new_row_scaled = price_scaler.transform(new_row_unscaled)[0]
        price_window = np.vstack([price_window[1:], new_row_scaled])

        # Sentiment window STAYS FROZEN — no update.
        working_close = pred_close

    return predictions


def predict_all_stocks(fusion: str, days: int = 5, max_stocks: int | None = None) -> Path:
    """Predict N-day returns for all stocks that have a multimodal checkpoint."""
    files = sorted(DATA_DIR.glob("*_processed_v2.csv"))
    if not files:
        logger.error(f"❌ No *_processed_v2.csv in {DATA_DIR}")
        return None

    if max_stocks:
        files = files[:max_stocks]

    stock_codes = [f.stem.replace("_processed_v2", "") for f in files]
    logger.info(f"📊 Predicting {days}-day returns for {len(stock_codes)} stocks ({fusion})\n")

    all_predictions = []
    successful = 0
    failed = []

    for i, code in enumerate(stock_codes, 1):
        logger.info(f"[{i}/{len(stock_codes)}] {code}...")
        try:
            preds = predict_stock(code, days=days, fusion=fusion)
            if preds:
                all_predictions.extend(preds)
                successful += 1
                d1 = preds[0]
                direction = "�" if d1["predicted_return_%"] > 0 else "📉"
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
        return None

    pred_df = pd.DataFrame(all_predictions)
    output_path = RESULTS_DIR / f"predictions_{days}days.csv"
    # If predicting both fusions into one CSV, append mode with merge
    if output_path.exists():
        existing = pd.read_csv(output_path)
        # Avoid duplicates if re-running
        keep_mask = ~(
            existing.set_index(["stock", "days_ahead", "fusion_strategy"]).index
            .isin(pred_df.set_index(["stock", "days_ahead", "fusion_strategy"]).index)
        )
        existing = existing[keep_mask]
        pred_df = pd.concat([existing, pred_df], ignore_index=True)
    pred_df.to_csv(output_path, index=False)

    logger.info("\n" + "=" * 70)
    logger.info(
        f"📊 PREDICTION SUMMARY ({fusion}, Day 1, {successful}/{len(stock_codes)} stocks)"
    )
    logger.info("=" * 70)

    day1 = pred_df[(pred_df["days_ahead"] == 1) & (pred_df["fusion_strategy"] == fusion)].sort_values(
        "predicted_return_%", ascending=False,
    )

    logger.info("\n� TOP 10 EXPECTED GAINERS (Day 1):")
    logger.info("-" * 70)
    for _, row in day1.head(10).iterrows():
        logger.info(
            f"  {row['stock']:15s} ৳{row['current_price']:>8.2f} → "
            f"৳{row['predicted_price']:>8.2f} ({row['predicted_return_%']:+.2f}%)"
        )

    logger.info("\n🔴 TOP 10 EXPECTED LOSERS (Day 1):")
    logger.info("-" * 70)
    for _, row in day1.tail(10).iterrows():
        logger.info(
            f"  {row['stock']:15s} ৳{row['current_price']:>8.2f} → "
            f"৳{row['predicted_price']:>8.2f} ({row['predicted_return_%']:+.2f}%)"
        )

    # Loop sanity
    this_fusion = pred_df[pred_df["fusion_strategy"] == fusion]
    n_distinct = (
        this_fusion.groupby("stock")["predicted_return_%"]
        .nunique()
        .reindex(stock_codes, fill_value=0)
    )
    n_all_same = int((n_distinct == 1).sum())
    logger.info(
        f"\n🔁 Loop sanity: {n_all_same}/{len(stock_codes)} stocks had IDENTICAL "
        "returns across all days (should be 0)."
    )

    logger.info(f"\n💾 Predictions saved: {output_path}")
    logger.info(f"📊 Total rows: {len(pred_df)}")
    logger.info("=" * 70)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate Multimodal LSTM predictions (Phase 7).")
    parser.add_argument("--fusion", choices=["early", "late", "both"], default="both",
                        help="Which fusion(s) to run (default: both)")
    parser.add_argument("--days", type=int, default=5, help="Days ahead (default: 5).")
    parser.add_argument("--stock", type=str, default=None, help="Predict a single stock only.")
    parser.add_argument("--max-stocks", type=int, default=None, help="Limit to N stocks (debug).")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("📈 Multimodal Predictions (Phase 7 — Price + Sentiment)")
    logger.info("=" * 70)

    if args.stock:
        fusions = ["early", "late"] if args.fusion == "both" else [args.fusion]
        for fusion in fusions:
            preds = predict_stock(args.stock, days=args.days, fusion=fusion)
            if preds:
                logger.info(
                    f"\n{pd.DataFrame(preds).to_string(index=False)}"
                )
            else:
                logger.error(f"❌ No predictions for {args.stock} ({fusion})")
        return

    fusions = ["early", "late"] if args.fusion == "both" else [args.fusion]
    for fusion in fusions:
        logger.info(f"\n{'=' * 70}\n▶ Fusion: {fusion.upper()}\n{'=' * 70}")
        predict_all_stocks(fusion=fusion, days=args.days, max_stocks=args.max_stocks)


if __name__ == "__main__":
    main()