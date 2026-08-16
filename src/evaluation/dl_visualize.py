"""
Visualize Deep Learning Results — Phase 4 (LSTM, leak-free).

Reads results/deep_learning/deep_learning_results.csv (one row per stock,
LSTM only) and produces 6 PNGs + summary_report.txt:

    01_model_summary.png            — overall metrics card (RMSE/MAE/R²/Dir_Acc)
    02_per_stock_performance.png    — RMSE & Dir_Acc per stock (sorted)
    03_metrics_distribution.png     — distributions of RMSE / R² / Dir_Acc
    04_directional_accuracy.png     — Dir_Acc vs 50% baseline (per stock)
    05_epochs_trained.png           — epochs trained per stock (early stopping)
    06_training_curves.png          — averaged train/val loss curve (across stocks)

Color palette mirrors src/evaluation/visualize.py v2 for consistency.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure `src` is importable when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import (
    DL_RESULTS_DIR as RESULTS_DIR,
    DL_PLOTS_DIR as PLOTS_DIR,
    DEEP_LEARNING_MODELS_DIR as MODELS_DIR,
)
from src.utils.logger import get_logger


sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
logger = get_logger("dl_visualize")

# Single arch (LSTM) — colors mirror baseline palette.
LSTM_COLOR = "#9b59b6"  # purple — distinct from baseline blue/green/red/amber


def load_results() -> pd.DataFrame | None:
    results_file = RESULTS_DIR / "deep_learning_results.csv"
    if not results_file.exists():
        logger.error(f"❌ Results file not found: {results_file}")
        return None
    df = pd.read_csv(results_file)
    logger.info(f"📊 Loaded {len(df)} stock results from {results_file.name}")
    return df


def plot_model_summary(df: pd.DataFrame):
    """4-panel card: RMSE / MAE / R² / Dir_Acc with 50% baseline."""
    metrics = ["test_rmse", "test_mae", "test_r2", "test_dir_acc"]
    titles = [
        "Average RMSE\n(Lower is Better — Target = next-day return)",
        "Average MAE\n(Lower is Better)",
        "Average R²\n(Higher is Better)",
        "Average Directional Accuracy\n(Higher is Better — 50% = coin flip)",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    for ax, m, title in zip(axes, metrics, titles):
        v = df[m].mean()
        ax.bar(["LSTM"], [v], color=LSTM_COLOR, width=0.4)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(m.replace("test_", "").replace("_", " "))
        ax.grid(True, alpha=0.3)
        ax.text(0, v, f"{v:.4f}" if "r2" in m or "rmse" in m or "mae" in m else f"{v:.1f}%",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
        if m == "test_dir_acc":
            ax.axhline(50, color="red", linestyle="--", linewidth=1.5, label="50% (random)")
            ax.legend()
            ax.set_ylim(40, 60)
        if m == "test_r2":
            ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    out = PLOTS_DIR / "01_model_summary.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"✅ Saved: {out.name}")


def plot_per_stock_performance(df: pd.DataFrame):
    """Per-stock RMSE and Dir_Acc (sorted by stock code)."""
    df_sorted = df.sort_values("stock").reset_index(drop=True)

    fig, axes = plt.subplots(2, 1, figsize=(20, 12))

    axes[0].plot(df_sorted["stock"], df_sorted["test_rmse"], marker="o",
                 linewidth=2, color=LSTM_COLOR, markersize=7)
    axes[0].set_title("LSTM — RMSE by Stock", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("RMSE (return)")
    axes[0].set_xlabel("Stock")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df_sorted["stock"], df_sorted["test_dir_acc"], marker="o",
                 linewidth=2, color=LSTM_COLOR, markersize=7, label="LSTM")
    axes[1].axhline(50, color="red", linestyle="--", linewidth=1, alpha=0.6,
                    label="50% (random)")
    axes[1].set_title("LSTM — Directional Accuracy by Stock", fontsize=14,
                      fontweight="bold")
    axes[1].set_ylabel("Dir_Acc (%)")
    axes[1].set_xlabel("Stock")
    axes[1].legend(fontsize=12, loc="lower right")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(30, 70)

    plt.tight_layout()
    out = PLOTS_DIR / "02_per_stock_performance.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"✅ Saved: {out.name}")


def plot_metrics_distribution(df: pd.DataFrame):
    """Histograms of RMSE / R² / Dir_Acc."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, m, title in zip(
        axes,
        ["test_rmse", "test_r2", "test_dir_acc"],
        ["RMSE", "R²", "Dir_Acc (%)"],
    ):
        v = df[m].dropna()
        ax.hist(v, bins=12, color=LSTM_COLOR, alpha=0.7, edgecolor="black")
        ax.set_title(f"LSTM — {title}", fontweight="bold", fontsize=12)
        ax.set_xlabel(title)
        ax.axvline(v.mean(), color="red", linestyle="--", linewidth=2,
                   label=f"Mean: {v.mean():.4f}" if m != "test_dir_acc"
                   else f"Mean: {v.mean():.1f}%")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if m == "test_dir_acc":
            ax.axvline(50, color="black", linewidth=0.5, linestyle=":")
        if m == "test_r2":
            ax.axvline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    out = PLOTS_DIR / "03_metrics_distribution.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"✅ Saved: {out.name}")


def plot_directional_accuracy(df: pd.DataFrame):
    """Per-stock Dir_Acc bar chart, sorted descending."""
    df_sorted = df.sort_values("test_dir_acc", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(16, 8))
    colors = [LSTM_COLOR if v >= 50 else "#95a5a6" for v in df_sorted["test_dir_acc"]]
    bars = ax.bar(df_sorted["stock"], df_sorted["test_dir_acc"], color=colors,
                  edgecolor="black", linewidth=0.4)
    ax.axhline(50, color="red", linestyle="--", linewidth=1.5, label="50% (random)")
    ax.set_title("LSTM — Directional Accuracy per Stock\n(Purple = above 50%)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Dir_Acc (%)")
    ax.set_xlabel("Stock (sorted)")
    ax.set_ylim(30, 65)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    plt.tight_layout()
    out = PLOTS_DIR / "04_directional_accuracy.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"✅ Saved: {out.name}")


def plot_epochs_trained(df: pd.DataFrame):
    """Bar chart of epochs trained per stock (early-stopping visualization)."""
    df_sorted = df.sort_values("epochs_trained", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(df_sorted["stock"], df_sorted["epochs_trained"], color=LSTM_COLOR,
           edgecolor="black", linewidth=0.4)
    ax.axhline(df["epochs_trained"].mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean: {df['epochs_trained'].mean():.1f}")
    ax.set_title("LSTM — Epochs Trained per Stock (Early Stopping)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Epochs trained")
    ax.set_xlabel("Stock (sorted by epochs)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    plt.tight_layout()
    out = PLOTS_DIR / "05_epochs_trained.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"✅ Saved: {out.name}")


def plot_training_curves():
    """Average train/val loss curve across all stocks.

    Reads each sidecar's `history` dict and averages epochs-aligned.
    """
    histories = []
    for p in sorted(MODELS_DIR.glob("*_best_lstm.pkl")):
        with open(p, "rb") as f:
            sidecar = pickle.load(f)
        hist = sidecar.get("history")
        if hist and "train_loss" in hist and "val_loss" in hist:
            histories.append(hist)

    if not histories:
        logger.warning("⚠️  No training histories found, skipping training curves plot.")
        return

    # Truncate each to min length so we can average epoch-by-epoch
    min_len = min(len(h["train_loss"]) for h in histories)
    train_avg = np.mean([h["train_loss"][:min_len] for h in histories], axis=0)
    val_avg = np.mean([h["val_loss"][:min_len] for h in histories], axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    epochs = np.arange(1, min_len + 1)
    ax.plot(epochs, train_avg, color="#3498db", linewidth=2.5,
            label=f"Avg Train Loss (n={len(histories)})")
    ax.plot(epochs, val_avg, color="#e74c3c", linewidth=2.5,
            label=f"Avg Val Loss (n={len(histories)})")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("LSTM — Average Training Curves Across All Stocks\n(Avg of {} trainings, log scale)".format(len(histories)),
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    out = PLOTS_DIR / "06_training_curves.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"✅ Saved: {out.name}")


def generate_summary_report(df: pd.DataFrame) -> str:
    """Write human-readable summary text."""
    n = len(df)
    above_50 = int((df["test_dir_acc"] >= 50).sum())
    above_52 = int((df["test_dir_acc"] >= 52).sum())
    above_55 = int((df["test_dir_acc"] >= 55).sum())

    report = []
    report.append("=" * 70)
    report.append("DEEP LEARNING (LSTM) — FINAL SUMMARY REPORT (Phase 4)")
    report.append("=" * 70)
    report.append("")
    report.append("Architecture: StockLSTM (3 layers, hidden=128, dropout=0.2)")
    report.append(f"Total Stocks: {n}")
    report.append("Target: Target_Return_1d (next-day return)")
    report.append("Features: Lag-1 technical indicators (NO same-day OHLCV)")
    report.append("")
    report.append("-" * 70)
    report.append("OVERALL METRICS")
    report.append("-" * 70)
    report.append(f"  Average RMSE:      {df['test_rmse'].mean():.6f}")
    report.append(f"  Average MAE:       {df['test_mae'].mean():.6f}")
    report.append(f"  Average R²:        {df['test_r2'].mean():.4f}")
    report.append(f"  Average Dir_Acc:   {df['test_dir_acc'].mean():.1f}%")
    report.append(f"  Median Dir_Acc:    {df['test_dir_acc'].median():.1f}%")
    report.append(f"  Avg Epochs Run:    {df['epochs_trained'].mean():.1f}")
    report.append(f"  Avg Train Windows: {df['n_train_windows'].mean():.0f}")
    report.append(f"  Avg Test Windows:  {df['n_test_windows'].mean():.0f}")
    report.append("")
    report.append("-" * 70)
    report.append("TRADING-RELEVANT BREAKDOWN (Directional Accuracy)")
    report.append("-" * 70)
    report.append(f"  Stocks with Dir_Acc >= 50%: {above_50}/{n} ({above_50/n*100:.1f}%)")
    report.append(f"  Stocks with Dir_Acc >= 52%: {above_52}/{n} ({above_52/n*100:.1f}%)")
    report.append(f"  Stocks with Dir_Acc >= 55%: {above_55}/{n} ({above_55/n*100:.1f}%)")
    report.append("")
    report.append("-" * 70)
    report.append("TOP 10 STOCKS BY Dir_Acc")
    report.append("-" * 70)
    report.append(f"  {'Stock':15s}  {'Dir_Acc':>10s}  {'RMSE':>10s}  {'Epochs':>8s}")
    top = df.nlargest(10, "test_dir_acc")
    for _, r in top.iterrows():
        report.append(
            f"  {r['stock']:15s}  {r['test_dir_acc']:>9.1f}%  "
            f"{r['test_rmse']:>10.6f}  {int(r['epochs_trained']):>8d}"
        )
    report.append("")
    report.append("-" * 70)
    report.append("BOTTOM 10 STOCKS BY Dir_Acc")
    report.append("-" * 70)
    bot = df.nsmallest(10, "test_dir_acc")
    for _, r in bot.iterrows():
        report.append(
            f"  {r['stock']:15s}  {r['test_dir_acc']:>9.1f}%  "
            f"{r['test_rmse']:>10.6f}  {int(r['epochs_trained']):>8d}"
        )
    report.append("")
    report.append("=" * 70)
    report.append("KEY INSIGHTS")
    report.append("=" * 70)
    report.append("1. Phase 4 (LSTM) mirrors Phase 3 v2 leak-free discipline.")
    report.append("2. Daily return prediction is genuinely hard — Dir_Acc near 50%")
    report.append("   is realistic for emerging markets.")
    report.append("3. Any LSTM with Dir_Acc > 52% is a candidate for direction bets.")
    report.append("4. Comparison vs Phase 3 best baseline is in summary (below).")
    report.append("")
    report.append("=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)

    text = "\n".join(report)
    out = RESULTS_DIR / "summary_report.txt"
    out.write_text(text)
    logger.info(f"✅ Saved: {out.name}")
    return text


def main():
    df = load_results()
    if df is None or df.empty:
        logger.error("❌ No results to visualize.")
        return

    logger.info("=" * 70)
    logger.info("📊 Generating LSTM visualizations...")
    logger.info("=" * 70)

    plot_model_summary(df)
    plot_per_stock_performance(df)
    plot_metrics_distribution(df)
    plot_directional_accuracy(df)
    plot_epochs_trained(df)
    plot_training_curves()
    generate_summary_report(df)

    logger.info("=" * 70)
    logger.info("✨ All visualizations generated!")
    logger.info(f"📁 Plots: {PLOTS_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
