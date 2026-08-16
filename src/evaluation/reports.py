"""
Text-based summary report generation for ML experiments.

Usage:
    from src.evaluation.reports import generate_summary_report
    generate_summary_report(results_df, output_path)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def generate_summary_report(
    df: pd.DataFrame,
    output_path: Path,
    models: list = None,
    title: str = "BASELINE ML MODELS - FINAL SUMMARY REPORT",
) -> None:
    """
    Generate a text-based summary report of ML experiment results.

    Args:
        df: DataFrame with columns: stock, {model}_RMSE, {model}_MAE, {model}_MAPE, {model}_R2, best_model
        output_path: Where to write the report
        models: List of model names (auto-detected if None)
        title: Report title
    """
    if models is None:
        # Auto-detect models from columns
        models = sorted({
            col.replace("_RMSE", "").replace("_MAE", "").replace("_MAPE", "").replace("_R2", "").replace("_R²", "")
            for col in df.columns
            if col.endswith("_RMSE")
        })

    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"{title}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Project: LLM-Orchestrated Financial Advisor for Bangladesh Stock Market\n")
        f.write(f"Total Stocks: {len(df)}\n")
        f.write(f"Models: {', '.join(models)}\n\n")
        f.write("=" * 70 + "\n")
        f.write("MODEL PERFORMANCE\n")
        f.write("=" * 70 + "\n\n")

        for model in models:
            rmse = df[f"{model}_RMSE"].mean()
            mae = df[f"{model}_MAE"].mean()
            mape = df[f"{model}_MAPE"].mean()
            r2 = df[f"{model}_R2"].mean()

            f.write(f"{model}:\n")
            f.write(f"  Average RMSE:  {rmse:.2f}\n")
            f.write(f"  Average MAE:   {mae:.2f}\n")
            f.write(f"  Average MAPE:  {mape:.2f}%\n")
            f.write(f"  Average R²:    {r2:.4f}\n\n")

        if "best_model" in df.columns:
            f.write("=" * 70 + "\n")
            f.write("BEST MODEL DISTRIBUTION\n")
            f.write("=" * 70 + "\n\n")
            best_counts = df["best_model"].value_counts()
            for model, count in best_counts.items():
                pct = count / len(df) * 100
                f.write(f"  {model}: {count} stocks ({pct:.1f}%)\n")

        # Top 10 stocks by best R²
        f.write("\n" + "=" * 70 + "\n")
        f.write("TOP 10 STOCKS BY R² (Best Predictions)\n")
        f.write("=" * 70 + "\n\n")

        best_r2_per_stock = []
        for _, row in df.iterrows():
            r2_values = {m: row[f"{m}_R2"] for m in models if f"{m}_R2" in row.index}
            best_r2 = max(r2_values.values())
            best_model_name = max(r2_values, key=r2_values.get)
            best_r2_per_stock.append({
                "stock": row["stock"],
                "best_model": best_model_name,
                "best_r2": best_r2,
            })

        best_df = (
            pd.DataFrame(best_r2_per_stock)
            .sort_values("best_r2", ascending=False)
            .head(10)
        )
        f.write(best_df.to_string(index=False))
        f.write("\n\n")

        f.write("=" * 70 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("=" * 70 + "\n\n")

        if models:
            best_overall = models[np.argmin([df[f"{m}_RMSE"].mean() for m in models])]
            f.write(f"1. Best Overall Model: {best_overall}\n")
            f.write(f"   Average RMSE: {df[f'{best_overall}_RMSE'].mean():.2f}\n")
            f.write(f"   Average R²: {df[f'{best_overall}_R2'].mean():.4f}\n\n")

            for m in models:
                count = (df["best_model"] == m).sum() if "best_model" in df.columns else 0
                f.write(f"2. {m} performed best for: {count} stocks\n\n")

        f.write("=" * 70 + "\n")


__all__ = ["generate_summary_report"]