"""
Natural-Language Explanation Generator.

Turns SHAP/LIME feature weights into a beginner-friendly explanation:

    📈 Prediction: GP +0.85% next-day return
    🔑 Top drivers:
       • Returns_5d = +0.04 (pushes UP, weight=0.42)
       • RSI_14 = 65 (pushes DOWN, weight=-0.31)
       ...
    🎯 Confidence: 64%
"""

from __future__ import annotations

import warnings
from typing import Iterable

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


# Human-readable feature name mapping (extend as needed)
FEATURE_NICE_NAMES = {
    "SMA_5": "5-day moving average",
    "SMA_10": "10-day moving average",
    "SMA_20": "20-day moving average",
    "SMA_50": "50-day moving average",
    "SMA_100": "100-day moving average",
    "SMA_200": "200-day moving average",
    "EMA_12": "12-day exponential MA",
    "EMA_26": "26-day exponential MA",
    "EMA_50": "50-day exponential MA",
    "RSI_14": "RSI (14-day momentum)",
    "MACD": "MACD",
    "MACD_Signal": "MACD signal line",
    "MACD_Histogram": "MACD histogram",
    "BB_Middle": "Bollinger middle band",
    "BB_Upper": "Bollinger upper band",
    "BB_Lower": "Bollinger lower band",
    "BB_Width": "Bollinger band width",
    "BB_Position": "Bollinger position",
    "ATR_14": "Average True Range (14d)",
    "Returns_1d": "1-day return",
    "Returns_5d": "5-day return",
    "Returns_20d": "20-day return",
    "Log_Returns": "log return",
    "Volatility_30d": "30-day volatility",
    "Volatility_60d": "60-day volatility",
    "Volume_SMA_20": "20-day avg volume",
    "Volume_Ratio": "today's volume vs avg",
}


def format_feature_name(name: str) -> str:
    """Strip LIME bucket suffixes (>, <, ranges) and pretty-print."""
    if "|" in name:  # time-series flattened
        parts = name.split("|", 1)
        return f"{parts[0]} {FEATURE_NICE_NAMES.get(parts[1], parts[1])}"
    # Strip LIME conditions like "> 0.5" or "0.1 < x <= 0.5"
    base = name.split(" ")[0]
    return FEATURE_NICE_NAMES.get(base, name)


def generate_explanation(
    stock: str,
    predicted_return: float,
    feature_weights: pd.DataFrame,
    top_k: int = 5,
) -> str:
    """Generate a human-readable prediction explanation.

    Args:
        stock: Ticker symbol (e.g., "GP")
        predicted_return: Predicted next-day return (e.g., 0.0085)
        feature_weights: DataFrame with columns [feature, shap_value] sorted by |weight|
        top_k: Number of top features to show
    Returns:
        Multi-line explanation string
    """
    direction_word = "increase" if predicted_return > 0 else "decrease"
    pred_pct = abs(predicted_return) * 100

    lines = []
    lines.append(f"📈 Prediction: {stock} expected to {direction_word} by {pred_pct:.2f}% next day")
    lines.append("")

    top = feature_weights.head(top_k)
    if top.empty:
        lines.append("   (no features)")
        return "\n".join(lines)

    lines.append("🔑 Key drivers (top {}):".format(top_k))
    for _, row in top.iterrows():
        feat = row["feature"]
        val = row.get("feature_value", None)
        weight = row["shap_value"]
        nice = format_feature_name(feat)
        sign_word = "pushes UP" if weight > 0 else "pushes DOWN"
        strength = "strongly" if abs(weight) > 0.001 else "moderately"
        val_str = f" (value={val:.4f})" if val is not None and isinstance(val, (int, float)) else ""
        lines.append(f"   • {nice}{val_str}: {sign_word} {strength} (weight={weight:+.6f})")

    # Confidence: top-1 weight / sum of all weights, capped at 1
    total_weight = float(np.abs(feature_weights["shap_value"]).sum())
    if total_weight > 0:
        top1 = float(np.abs(top.iloc[0]["shap_value"]))
        confidence = min(top1 / total_weight * 1.5, 1.0)  # scale so a clear winner gets ~high
    else:
        confidence = 0.0
    lines.append("")
    lines.append(f"🎯 Concentration: top feature explains {confidence*100:.0f}% of total explanation weight")
    return "\n".join(lines)


def explain_dataframe(
    stock: str,
    predicted_return: float,
    shap_values_row: np.ndarray,
    feature_names: list[str],
    feature_values_row: np.ndarray | None = None,
    top_k: int = 5,
) -> str:
    """Convenience: pass numpy arrays directly."""
    df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": np.asarray(shap_values_row).flatten(),
    })
    if feature_values_row is not None:
        df["feature_value"] = np.asarray(feature_values_row).flatten()
    df = df.iloc[df["shap_value"].abs().values.argsort()[::-1]].reset_index(drop=True)
    return generate_explanation(stock, predicted_return, df, top_k)


__all__ = ["generate_explanation", "explain_dataframe", "format_feature_name", "FEATURE_NICE_NAMES"]
