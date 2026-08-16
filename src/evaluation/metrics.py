"""
Centralized metric calculation utilities.
Reused by training, evaluation, and inference modules.

Usage:
    from src.evaluation.metrics import calculate_metrics, format_metrics
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true, y_pred) -> dict:
    """
    Calculate standard regression metrics.

    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values

    Returns:
        Dict with RMSE, MAE, MAPE, R²
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Avoid division by zero in MAPE
    mask = y_true != 0
    mape = (
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        if mask.any() else float("inf")
    )

    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(mape),
        "R²": float(r2_score(y_true, y_pred)),
    }


def format_metrics(metrics: dict, indent: str = "   ") -> str:
    """Format metrics dict as human-readable string."""
    lines = []
    for key, val in metrics.items():
        if key == "R²":
            lines.append(f"{indent}{key}: {val:.4f}")
        else:
            lines.append(f"{indent}{key}: {val:.2f}")
    return "\n".join(lines)


def aggregate_metrics(metrics_list: list) -> dict:
    """Aggregate (mean) metrics across multiple runs/stocks."""
    if not metrics_list:
        return {}

    keys = metrics_list[0].keys()
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}


__all__ = ["calculate_metrics", "format_metrics", "aggregate_metrics"]