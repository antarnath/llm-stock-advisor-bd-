"""
Quick Test / Demo for Phase 3 v2 Baseline Models
================================================

This script lets you test the trained models in 3 modes:
  1. PREDICT  - Predict next-day return for a single stock
  2. COMPARE  - Show metrics for all 4 models on one stock
  3. SCAN     - Show top 10 gainers/losers across all 30 stocks

It addresses the common question: "how do I actually use these models?"

Usage (from project root):
    python test.py                          # default: PREDICT 1 day for GP
    python test.py --stock BEXIMCO          # 1-day forecast for BEXIMCO
    python test.py --days 5                 # 5-day forecast for GP (default)
    python test.py --days 10 --stock BEXIMCO
    python test.py --mode forecast --stock GP --days 7
    python test.py --mode compare --stock GP
    python test.py --mode scan
    python test.py --mode scan --top 5      # top 5 gainers/losers
    python test.py --list                   # show all 30 trained stocks
    python test.py --inspect --stock GP     # show features used + leakage check

Q: Why do we have 30 separate .pkl files (one per stock)?
A: Each stock has its own volatility, drift, and sector dynamics. A model
   trained on GP's price behavior doesn't transfer well to BEXIMCO. This
   is the standard practice in production finance — one model per asset.
"""

import argparse
import sys
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

# Allow running from project root without installing the package
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    BASELINE_MODELS_DIR,
    PROCESSED_DATA_DIR,
    BASELINE_RESULTS_DIR,
    TOP_30_DSE_STOCKS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(stock_code):
    """Load the v2 trained model + metadata for a stock."""
    model_path = BASELINE_MODELS_DIR / f"{stock_code}_best_v2.pkl"
    if not model_path.exists():
        return None, f"No model found for {stock_code} at {model_path}"
    with open(model_path, "rb") as f:
        return pickle.load(f), None


def load_processed(stock_code):
    """Load the v2 processed CSV for a stock."""
    data_path = PROCESSED_DATA_DIR / f"{stock_code}_processed_v2.csv"
    if not data_path.exists():
        return None, f"No processed file for {stock_code} at {data_path}"
    return pd.read_csv(data_path, parse_dates=["date"]), None


def get_last_features(df, feature_cols):
    """Extract the most-recent feature vector from the processed CSV.

    Because v2 indicators are already .shift(1)-ed, the last row already
    contains yesterday's indicators — exactly what we need to predict
    today's return (which the model treats as "next day").
    """
    last_row = df.iloc[-1][feature_cols].astype(float)
    return last_row


def predict_n_days(stock_code, days=5):
    """Recursive N-day forecast.

    Iteratively updates the feature vector after each step so day-2 features
    reflect the day-1 predicted return — fixes the "5 identical predictions"
    bug from v1. Returns a list of dicts:
        [{day, date, predicted_return, predicted_close}, ...]
    """
    model_data, err = load_model(stock_code)
    if err:
        raise RuntimeError(err)
    df, err = load_processed(stock_code)
    if err:
        raise RuntimeError(err)

    model = model_data["model"]
    feature_cols = model_data["features"]
    last_date = df.iloc[-1]["date"]
    current_close = float(df.iloc[-1]["close"])

    # Working copy of the feature row + a buffer of recent returns
    x_row = get_last_features(df, feature_cols)
    recent_returns = list(df["Returns_1d"].dropna().tail(20).values)

    preds = []
    working_close = current_close
    cursor_date = last_date

    for day in range(1, days + 1):
        # Predict next-day return
        pred_return = float(model.predict(x_row.values.reshape(1, -1))[0])
        pred_close = working_close * (1.0 + pred_return)

        # Skip weekends
        cursor_date = cursor_date + pd.Timedelta(days=1)
        while cursor_date.weekday() >= 5:
            cursor_date += pd.Timedelta(days=1)

        preds.append({
            "day": day,
            "date": cursor_date.strftime("%Y-%m-%d"),
            "predicted_return_%": pred_return * 100,
            "predicted_close": pred_close,
            "predicted_return_raw": pred_return,
        })

        # ----- Update features for next step -----
        recent_returns.append(pred_return)
        if len(recent_returns) > 20:
            recent_returns.pop(0)

        if "Returns_1d" in x_row.index:
            x_row["Returns_1d"] = pred_return
        if "Returns_5d" in x_row.index and len(recent_returns) >= 5:
            x_row["Returns_5d"] = float(np.prod(1.0 + np.array(recent_returns[-5:])) - 1.0)
        if "Returns_20d" in x_row.index and len(recent_returns) >= 20:
            x_row["Returns_20d"] = float(np.prod(1.0 + np.array(recent_returns[-20:])) - 1.0)
        if "Log_Returns" in x_row.index:
            x_row["Log_Returns"] = float(np.log(1.0 + pred_return)) if pred_return > -1.0 else 0.0

        working_close = pred_close

    return preds, current_close, last_date, model, feature_cols


def leakage_check(model_data):
    """Return a list of forbidden (OHLCV/target) columns found in features."""
    forbidden = {"open", "high", "low", "close", "volume", "trade", "value",
                 "Target_Return_1d", "Target_Price_1d"}
    return [f for f in model_data["features"] if f in forbidden]


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_predict(stock_code, days=1):
    """Predict next-day (or N-day) return(s) and convert to price.

    Iteratively updates the feature vector per step so day-2, day-3, ...
    predictions differ from day-1 (fixes the v1 bug where all 5 days were
    identical).
    """
    title = f"PREDICT MODE — {stock_code} ({days}-day forecast)"
    print("=" * 70)
    print(f"📈  {title}")
    print("=" * 70)

    try:
        preds, current_close, last_date, model, feature_cols = predict_n_days(stock_code, days=days)
    except RuntimeError as e:
        print(f"❌ {e}")
        return

    print(f"   Stock            : {stock_code}")
    print(f"   Last data date   : {last_date.strftime('%Y-%m-%d')}")
    print(f"   Current close    : ৳{current_close:,.2f}")
    print(f"   Model type       : {type(model).__name__}")
    print(f"   Features used    : {len(feature_cols)}")
    print()
    print(f"   {'Day':<5} {'Date':<12} {'Return %':>11} {'Predicted Price':>18}")
    print("   " + "-" * 50)
    for p in preds:
        direction = "📈" if p["predicted_return_raw"] > 0 else "📉"
        print(f"   {p['day']:<5} {p['date']:<12} {p['predicted_return_%']:>+10.4f}% "
              f"৳{p['predicted_close']:>14,.2f}  {direction}")

    # Summary stats
    rets = [p["predicted_return_raw"] for p in preds]
    cumulative = float(np.prod(1.0 + np.array(rets)) - 1.0) * 100
    up_days = sum(1 for r in rets if r > 0)
    print()
    print(f"   📊 Cumulative return over {days} days : {cumulative:+.4f}%")
    print(f"   📊 Up days vs down days               : {up_days} / {days - up_days}")

    # Sanity: leakage check
    model_data, _ = load_model(stock_code)
    leaks = leakage_check(model_data)
    if leaks:
        print(f"\n   ⚠️  LEAKAGE DETECTED in saved model: {leaks}")
    else:
        print(f"\n   ✅  Leakage check passed (no OHLCV/target in features).")
    print("=" * 70)


def mode_forecast(stock_code, days=5):
    """Alias for predict mode but always multi-day and explicitly named."""
    mode_predict(stock_code, days=days)


def mode_compare(stock_code):
    """Show metrics for all 4 models on a single stock from baseline_results_v2.csv."""
    print("=" * 70)
    print(f"📊  COMPARE MODE — {stock_code}")
    print("=" * 70)

    results_path = BASELINE_RESULTS_DIR / "baseline_results_v2.csv"
    if not results_path.exists():
        print(f"❌ Results CSV not found at {results_path}")
        return

    df = pd.read_csv(results_path)
    row = df[df["stock"] == stock_code]
    if row.empty:
        print(f"❌ No results for {stock_code}")
        return

    row = row.iloc[0]
    models = ["LinearRegression", "RandomForest", "XGBoost", "LightGBM"]
    print(f"   {'Model':<20} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'Dir_Acc':>10}")
    print("   " + "-" * 62)
    for m in models:
        rmse = row.get(f"{m}_RMSE", float("nan"))
        mae = row.get(f"{m}_MAE", float("nan"))
        r2 = row.get(f"{m}_R²", float("nan"))
        dacc = row.get(f"{m}_Dir_Acc", float("nan"))
        print(f"   {m:<20} {rmse:>10.6f} {mae:>10.6f} {r2:>+10.4f} {dacc:>9.1f}%")

    best = row.get("best_model", "N/A")
    print(f"\n   🏆 Best model (by RMSE): {best}")
    print("=" * 70)


def mode_scan(top_n=10):
    """Show top-N gainers and losers across all 30 stocks (Day 1 prediction)."""
    print("=" * 70)
    print(f"🔍  SCAN MODE — Top {top_n} gainers/losers (Day 1 prediction)")
    print("=" * 70)

    rows = []
    for stock in TOP_30_DSE_STOCKS:
        model_data, err = load_model(stock)
        if err:
            continue
        df, err = load_processed(stock)
        if err:
            continue
        feats = model_data["features"]
        x = get_last_features(df, feats)
        ret = float(model_data["model"].predict(x.values.reshape(1, -1))[0])
        cur = float(df.iloc[-1]["close"])
        rows.append({
            "stock": stock,
            "current_price": cur,
            "predicted_return_%": ret * 100,
            "predicted_price": cur * (1.0 + ret),
        })

    if not rows:
        print("❌ No predictions produced.")
        return

    scan = pd.DataFrame(rows).sort_values("predicted_return_%", ascending=False)

    print(f"\n🟢 TOP {top_n} EXPECTED GAINERS (Day 1):")
    print("-" * 70)
    for _, r in scan.head(top_n).iterrows():
        print(f"   {r['stock']:<15} ৳{r['current_price']:>9,.2f} → "
              f"৳{r['predicted_price']:>9,.2f}  ({r['predicted_return_%']:+.2f}%)")

    print(f"\n🔴 TOP {top_n} EXPECTED LOSERS (Day 1):")
    print("-" * 70)
    for _, r in scan.tail(top_n).iloc[::-1].iterrows():
        print(f"   {r['stock']:<15} ৳{r['current_price']:>9,.2f} → "
              f"৳{r['predicted_price']:>9,.2f}  ({r['predicted_return_%']:+.2f}%)")
    print("=" * 70)


def mode_list():
    """List all 30 stocks with model + processed file status."""
    print("=" * 70)
    print("📋  TRAINED STOCKS (one model per stock)")
    print("=" * 70)
    print(f"   {'Stock':<15} {'Model .pkl':<15} {'Processed CSV':<15} {'Size (KB)':>10}")
    print("   " + "-" * 58)
    total = 0
    for stock in TOP_30_DSE_STOCKS:
        mp = BASELINE_MODELS_DIR / f"{stock}_best_v2.pkl"
        cp = PROCESSED_DATA_DIR / f"{stock}_processed_v2.csv"
        m_exists = mp.exists()
        c_exists = cp.exists()
        size_kb = mp.stat().st_size / 1024 if m_exists else 0
        total += 1 if m_exists and c_exists else 0
        print(f"   {stock:<15} "
              f"{'✓' if m_exists else '✗':<15} "
              f"{'✓' if c_exists else '✗':<15} "
              f"{size_kb:>10.1f}")
    print(f"\n   ✅ {total}/{len(TOP_30_DSE_STOCKS)} stocks fully trained.")
    print("=" * 70)


def mode_inspect(stock_code):
    """Show feature list + leakage check for a saved model."""
    print("=" * 70)
    print(f"🔬  INSPECT MODE — {stock_code}")
    print("=" * 70)

    model_data, err = load_model(stock_code)
    if err:
        print(f"❌ {err}")
        return

    print(f"   Pickle keys  : {list(model_data.keys())}")
    print(f"   Stock        : {model_data.get('stock', 'N/A')}")
    print(f"   Target       : {model_data.get('target', 'N/A')}")
    print(f"   Version      : {model_data.get('version', 'N/A')}")
    print(f"   Model class  : {type(model_data['model']).__name__}")
    print(f"   # features   : {len(model_data['features'])}")

    leaks = leakage_check(model_data)
    if leaks:
        print(f"   ⚠️  LEAKAGE : {leaks}")
    else:
        print(f"   ✅ Leakage  : none (clean)")

    print(f"\n   Features:")
    for i, f in enumerate(model_data["features"], 1):
        marker = " ⚠️" if f in {"open", "high", "low", "close", "volume"} else ""
        print(f"     {i:>3}. {f}{marker}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Quick test/demo for Phase 3 v2 baseline stock models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--stock", default="GP",
                        help="Stock code (default: GP). Use --list to see all 30.")
    parser.add_argument("--mode", default="predict",
                        choices=["predict", "forecast", "compare", "scan", "list", "inspect"],
                        help="Operation mode (default: predict).")
    parser.add_argument("--days", type=int, default=1,
                        help="Days ahead to forecast for predict/forecast modes (default: 1).")
    parser.add_argument("--top", type=int, default=10,
                        help="Top-N for scan mode (default: 10).")
    parser.add_argument("--list", action="store_true",
                        help="Shortcut for --mode list.")
    args = parser.parse_args()

    if args.list:
        args.mode = "list"

    if args.mode == "predict":
        mode_predict(args.stock, days=args.days)
    elif args.mode == "forecast":
        mode_forecast(args.stock, days=args.days)
    elif args.mode == "compare":
        mode_compare(args.stock)
    elif args.mode == "scan":
        mode_scan(top_n=args.top)
    elif args.mode == "list":
        mode_list()
    elif args.mode == "inspect":
        mode_inspect(args.stock)


if __name__ == "__main__":
    main()
