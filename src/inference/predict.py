"""
Generate Stock Predictions - Phase 3
Uses trained models to predict next-day prices for all stocks

Output: results/predictions.csv with predictions for next 5 days
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime, timedelta


# Project paths (centralized)
from src.utils.config import (
    PROJECT_ROOT,
    PROCESSED_DATA_DIR as DATA_DIR,
    BASELINE_MODELS_DIR as MODELS_DIR,
    BASELINE_RESULTS_DIR as RESULTS_DIR,
)


def load_model(stock_code):
    """Load trained model for a stock"""
    model_path = MODELS_DIR / f"{stock_code}_best.pkl"
    if not model_path.exists():
        return None

    with open(model_path, 'rb') as f:
        return pickle.load(f)


def predict_stock(stock_code, days=5):
    """Predict next N days for a stock"""
    # Load processed data
    data_path = DATA_DIR / f"{stock_code}_processed.csv"
    if not data_path.exists():
        return None

    df = pd.read_csv(data_path, parse_dates=['date'])

    # Load model
    model_data = load_model(stock_code)
    if model_data is None:
        return None

    model = model_data['model']
    feature_cols = model_data['features']

    # Get last row (most recent data)
    last_row = df.iloc[-1]
    current_price = last_row['close']
    current_date = last_row['date']

    # Prepare features
    X = df[feature_cols].iloc[-1:].values

    # Predict iteratively
    predictions = []
    for day in range(1, days + 1):
        pred_price = model.predict(X)[0]
        pred_date = current_date + timedelta(days=day)

        # Skip weekends (business days only)
        while pred_date.weekday() >= 5:  # Saturday=5, Sunday=6
            pred_date += timedelta(days=1)

        predictions.append({
            'stock': stock_code,
            'prediction_date': pred_date.strftime('%Y-%m-%d'),
            'days_ahead': day,
            'current_price': round(current_price, 2),
            'predicted_price': round(pred_price, 2),
            'expected_return_%': round(((pred_price / current_price) - 1) * 100, 2),
        })

    return predictions


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║   Stock Price Predictions - Next 5 Days                      ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Get all stock codes
    files = sorted(DATA_DIR.glob("*_processed.csv"))
    stock_codes = [f.stem.replace('_processed', '') for f in files]

    print(f"📊 Predicting for {len(stock_codes)} stocks\n")

    all_predictions = []

    for i, code in enumerate(stock_codes, 1):
        print(f"[{i}/{len(stock_codes)}] {code}...", end=" ")
        try:
            preds = predict_stock(code, days=5)
            if preds:
                all_predictions.extend(preds)
                # Show next day prediction
                next_day = preds[0]
                direction = "📈" if next_day['expected_return_%'] > 0 else "📉"
                print(f"{direction} Day 1: ৳{next_day['predicted_price']:.2f} "
                      f"({next_day['expected_return_%']:+.2f}%)")
            else:
                print("❌ No model")
        except Exception as e:
            print(f"❌ Error: {e}")

    # Save predictions
    if all_predictions:
        pred_df = pd.DataFrame(all_predictions)
        output_path = RESULTS_DIR / "predictions_5days.csv"
        pred_df.to_csv(output_path, index=False)

        print("\n" + "=" * 70)
        print("📊 PREDICTION SUMMARY (Next Day)")
        print("=" * 70)

        # Next day predictions
        next_day_df = pred_df[pred_df['days_ahead'] == 1].sort_values('expected_return_%', ascending=False)

        print("\n🟢 TOP 10 EXPECTED GAINERS:")
        print("-" * 70)
        for _, row in next_day_df.head(10).iterrows():
            print(f"  {row['stock']:15s} ৳{row['current_price']:>8.2f} → ৳{row['predicted_price']:>8.2f} "
                  f"({row['expected_return_%']:+.2f}%)")

        print("\n🔴 TOP 10 EXPECTED LOSERS:")
        print("-" * 70)
        for _, row in next_day_df.tail(10).iterrows():
            print(f"  {row['stock']:15s} ৳{row['current_price']:>8.2f} → ৳{row['predicted_price']:>8.2f} "
                  f"({row['expected_return_%']:+.2f}%)")

        print(f"\n💾 Full predictions saved: {output_path}")
        print(f"📊 Total predictions: {len(pred_df)} (30 stocks × 5 days)")
        print("=" * 70)


if __name__ == "__main__":
    main()
