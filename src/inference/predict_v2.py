"""
Generate Stock Predictions - Phase 3 (v2, LEAK-FREE)
Uses v2 trained models (which predict Target_Return_1d, not raw price) to
produce 5-day rolling return predictions for all stocks.

What changed vs v1:
- Loads {STOCK}_best_v2.pkl (trained on returns, lag-1 features only).
- Loads {STOCK}_processed_v2.csv (OHLCV + lagged indicators).
- Predicts next-day RETURN (not price). Converts to price only for display.
- Iteratively updates the last row's features using the predicted return so
  day-2, day-3, ... predictions differ from day-1 (fixes the v1 bug where the
  loop produced 5 identical predictions).
- Updates features that depend on `Returns_1d` / `Returns_5d` / `Returns_20d`
  by rolling the predicted return into the appropriate window.

Output: results/baseline/predictions_5days_v2.csv with columns
  stock, prediction_date, days_ahead, current_price, predicted_price,
  predicted_return_%, model_type

Usage:
    python src/inference/predict_v2.py
    python src/inference/predict_v2.py --days 5
    python src/inference/predict_v2.py --days 1    # 1-step only
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import argparse
from datetime import timedelta

# Project paths (centralized)
from src.utils.config import (
    PROJECT_ROOT,
    PROCESSED_DATA_DIR as DATA_DIR,
    BASELINE_MODELS_DIR as MODELS_DIR,
    BASELINE_RESULTS_DIR as RESULTS_DIR,
)
from src.utils.logger import get_logger

logger = get_logger("predict_v2")


# Features that depend on recent close/return history — these need to be
# updated after each predicted step so the loop produces non-identical values.
UPDATEABLE_RETURNS = ["Returns_1d", "Returns_5d", "Returns_20d", "Log_Returns"]


def load_model(stock_code, suffix="_v2"):
    """Load trained v2 model for a stock."""
    model_path = MODELS_DIR / f"{stock_code}_best{suffix}.pkl"
    if not model_path.exists():
        return None

    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_stock(stock_code, days=5, suffix="_v2"):
    """Predict next N days of returns (and convert to price) for a stock.

    Iteratively updates the input feature vector so that day-2 features
    reflect day-1 predicted return. This breaks the v1 bug where the same
    X was fed 5 times.
    """
    data_path = DATA_DIR / f"{stock_code}_processed{suffix}.csv"
    if not data_path.exists():
        logger.warning(f"   ⚠️  No processed data: {data_path.name}")
        return None

    df = pd.read_csv(data_path, parse_dates=["date"])
    if df.empty:
        logger.warning(f"   ⚠️  Empty processed data for {stock_code}")
        return None

    model_data = load_model(stock_code, suffix=suffix)
    if model_data is None:
        logger.warning(f"   ⚠️  No model for {stock_code}")
        return None

    model = model_data["model"]
    feature_cols = model_data["features"]

    # Last known row (yesterday, after lag-1 shift — today is unknown)
    last_row = df.iloc[-1].copy()
    current_close = float(last_row["close"])
    current_date = last_row["date"]

    # Maintain a working feature vector that we update after each step
    x_row = last_row[feature_cols].astype(float).copy()
    # Track the rolling history of recent returns for window features
    # We approximate future returns using the predicted returns.
    recent_returns = list(df["Returns_1d"].dropna().tail(20).values)  # buffer

    predictions = []
    working_close = current_close

    for day in range(1, days + 1):
        # Predict next-day return (fraction, e.g. 0.0123 = +1.23%)
        pred_return = float(model.predict(x_row.values.reshape(1, -1))[0])

        # Convert to predicted price
        pred_close = working_close * (1.0 + pred_return)

        # Predicted date (skip weekends)
        pred_date = current_date + timedelta(days=day)
        while pred_date.weekday() >= 5:  # Sat=5, Sun=6
            pred_date += timedelta(days=1)

        predictions.append({
            "stock": stock_code,
            "prediction_date": pred_date.strftime("%Y-%m-%d"),
            "days_ahead": day,
            "current_price": round(current_close, 2),
            "predicted_price": round(pred_close, 2),
            "predicted_return_%": round(pred_return * 100, 4),
            "model_type": "best_v2",
        })

        # ---- Update features for next iteration ----
        # 1) Update rolling close-derived indicators.
        #    We approximate by treating the predicted return as the latest
        #    observed return. This is the standard simple approach in
        #    recursive forecasting (and honest about it).
        recent_returns.append(pred_return)
        if len(recent_returns) > 20:
            recent_returns.pop(0)

        if "Returns_1d" in x_row.index:
            x_row["Returns_1d"] = pred_return
        if "Returns_5d" in x_row.index and len(recent_returns) >= 5:
            # Geometric cumulative over last 5 days
            x_row["Returns_5d"] = float(np.prod(1.0 + np.array(recent_returns[-5:])) - 1.0)
        if "Returns_20d" in x_row.index and len(recent_returns) >= 20:
            x_row["Returns_20d"] = float(np.prod(1.0 + np.array(recent_returns[-20:])) - 1.0)
        if "Log_Returns" in x_row.index:
            x_row["Log_Returns"] = float(np.log(1.0 + pred_return)) if pred_return > -1.0 else 0.0

        # 2) Update working close for the next iteration's price conversion
        working_close = pred_close

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Generate v2 stock predictions.")
    parser.add_argument("--days", type=int, default=5, help="Days ahead to predict (default: 5).")
    parser.add_argument("--max-stocks", type=int, default=None, help="Limit to N stocks (debug).")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("📈 Stock Predictions (v2 — Leak-Free, Returns Target)")
    logger.info("=" * 70)

    files = sorted(DATA_DIR.glob("*_processed_v2.csv"))
    if not files:
        logger.error(f"❌ No *_processed_v2.csv files in {DATA_DIR}")
        logger.error("   Run src/data_processing/technical_indicators_v2.py first.")
        return

    if args.max_stocks:
        files = files[: args.max_stocks]

    stock_codes = [f.stem.replace("_processed_v2", "") for f in files]
    logger.info(f"📊 Predicting {args.days}-day returns for {len(stock_codes)} stocks\n")

    all_predictions = []
    successful = 0
    failed = []

    for i, code in enumerate(stock_codes, 1):
        logger.info(f"[{i}/{len(stock_codes)}] {code}...", extra=None)
        try:
            preds = predict_stock(code, days=args.days)
            if preds:
                all_predictions.extend(preds)
                successful += 1
                # Show next-day direction in logs
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
    output_path = RESULTS_DIR / f"predictions_{args.days}days_v2.csv"
    pred_df.to_csv(output_path, index=False)

    logger.info("\n" + "=" * 70)
    logger.info(f"📊 PREDICTION SUMMARY (Day 1, {successful}/{len(stock_codes)} stocks)")
    logger.info("=" * 70)

    day1 = pred_df[pred_df["days_ahead"] == 1].sort_values("predicted_return_%", ascending=False)

    logger.info("\n🟢 TOP 10 EXPECTED GAINERS (Day 1):")
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

    # Quick sanity check: are day-2..day-N predictions different from day-1?
    n_distinct = (
        pred_df.groupby("stock")["predicted_return_%"]
        .nunique()
        .reindex(stock_codes, fill_value=0)
    )
    n_all_same = int((n_distinct == 1).sum())
    logger.info(
        f"\n🔁 Loop sanity: {n_all_same}/{len(stock_codes)} stocks had IDENTICAL "
        "returns across all 5 days (should be 0 after the v2 fix)."
    )

    logger.info(f"\n💾 Predictions saved: {output_path}")
    logger.info(f"📊 Total rows: {len(pred_df)} ({len(stock_codes)} stocks × {args.days} days)")
    logger.info("=" * 70)
    logger.info("✨ Prediction complete (v2)!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
