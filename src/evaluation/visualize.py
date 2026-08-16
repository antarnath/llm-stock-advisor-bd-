"""
Visualize ML Model Results - Phase 3 (v2, LEAK-FREE)
Reads results from baseline_results_v2.csv (4 models, returns target).

Generates 5 PNGs + summary_report.txt:
  01_model_comparison.png   — RMSE & R² across models
  02_per_stock_performance.png — RMSE & R² per stock
  03_best_model_distribution.png — best-model wins per stock
  04_metrics_distribution.png — distributions of RMSE / R² / Dir_Acc
  05_directional_accuracy.png — NEW: per-model Dir_Acc with 50% baseline

Color palette: blue, green, red, amber for LR/RF/XGB/LGBM.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths (centralized)
from src.utils.config import (
    PROJECT_ROOT,
    BASELINE_RESULTS_DIR as RESULTS_DIR,
    BASELINE_PLOTS_DIR as PLOTS_DIR,
)
from src.utils.logger import get_logger

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
logger = get_logger("visualize_v2")

# Style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# 4 baseline models (v2)
MODELS = ['LinearRegression', 'RandomForest', 'XGBoost', 'LightGBM']
COLORS = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']


def load_results(suffix="_v2"):
    """Load and flatten the v2 results CSV."""
    results_file = RESULTS_DIR / f"baseline_results{suffix}.csv"
    if not results_file.exists():
        logger.error(f"❌ Results file not found: {results_file}")
        return None

    df = pd.read_csv(results_file)
    logger.info(f"📊 Loaded {len(df)} stock results from {results_file.name}")
    return df


def plot_model_comparison(df):
    """Compare RMSE, R², and Dir_Acc across models (3-panel)."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    rmse_means = [df[f'{m}_RMSE'].mean() for m in MODELS]
    bars = axes[0].bar(MODELS, rmse_means, color=COLORS)
    axes[0].set_title('Average RMSE by Model\n(Lower is Better — Target = next-day return)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('RMSE')
    axes[0].grid(True, alpha=0.3)
    for bar, val in zip(bars, rmse_means):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{val:.5f}', ha='center', fontsize=11, fontweight='bold')

    r2_means = [df[f'{m}_R²'].mean() for m in MODELS]
    bars = axes[1].bar(MODELS, r2_means, color=COLORS)
    axes[1].set_title('Average R² by Model\n(Higher is Better)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('R² Score')
    axes[1].axhline(0, color='black', linewidth=0.5)
    axes[1].grid(True, alpha=0.3)
    for bar, val in zip(bars, r2_means):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')

    dir_means = [df[f'{m}_Dir_Acc'].mean() for m in MODELS]
    bars = axes[2].bar(MODELS, dir_means, color=COLORS)
    axes[2].axhline(50, color='red', linestyle='--', linewidth=1.5, label='50% (random)')
    axes[2].set_title('Average Directional Accuracy\n(Higher is Better — 50% = coin flip)', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('Dir_Acc (%)')
    axes[2].set_ylim(40, 70)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    for bar, val in zip(bars, dir_means):
        axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                     f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    out = PLOTS_DIR / "01_model_comparison.png"
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Saved: {out.name}")


def plot_per_stock_performance(df):
    """Show per-stock RMSE and Dir_Acc."""
    fig, axes = plt.subplots(2, 1, figsize=(20, 12))

    for i, model in enumerate(MODELS):
        axes[0].plot(df['stock'], df[f'{model}_RMSE'], marker='o', label=model,
                     linewidth=2, color=COLORS[i], markersize=7)
    axes[0].set_title('RMSE by Stock', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('RMSE (return)')
    axes[0].set_xlabel('Stock')
    axes[0].legend(fontsize=12, loc='upper right')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)

    for i, model in enumerate(MODELS):
        axes[1].plot(df['stock'], df[f'{model}_Dir_Acc'], marker='o', label=model,
                     linewidth=2, color=COLORS[i], markersize=7)
    axes[1].axhline(50, color='red', linestyle='--', linewidth=1, alpha=0.6)
    axes[1].set_title('Directional Accuracy by Stock', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Dir_Acc (%)')
    axes[1].set_xlabel('Stock')
    axes[1].legend(fontsize=12, loc='lower right')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(30, 75)

    plt.tight_layout()
    out = PLOTS_DIR / "02_per_stock_performance.png"
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Saved: {out.name}")


def plot_best_model_distribution(df):
    """Show which model wins for each stock."""
    if 'best_model' not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    best_counts = df['best_model'].value_counts()
    color_map = dict(zip(MODELS, COLORS))
    colors_used = [color_map.get(m, '#95a5a6') for m in best_counts.index]

    bars = ax.barh(best_counts.index, best_counts.values, color=colors_used)
    ax.set_title('Best Model Distribution\n(Which Model Won for Each Stock — by RMSE)', fontsize=15, fontweight='bold')
    ax.set_xlabel('Number of Stocks', fontsize=13)
    ax.set_ylabel('Model', fontsize=13)

    total = len(df)
    for bar, count in zip(bars, best_counts.values):
        pct = count / total * 100
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{int(count)} stocks ({pct:.1f}%)',
                va='center', fontsize=12, fontweight='bold')

    ax.set_xlim(0, max(best_counts.values) * 1.2)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    out = PLOTS_DIR / "03_best_model_distribution.png"
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Saved: {out.name}")


def plot_metrics_distribution(df):
    """Distribution of RMSE, R², Dir_Acc across all stocks."""
    fig, axes = plt.subplots(3, len(MODELS), figsize=(24, 12))

    for j, model in enumerate(MODELS):
        # RMSE
        ax = axes[0, j]
        v = df[f'{model}_RMSE'].dropna()
        ax.hist(v, bins=12, color=COLORS[j], alpha=0.7, edgecolor='black')
        ax.set_title(f'{model} - RMSE', fontweight='bold', fontsize=12)
        ax.set_xlabel('RMSE')
        ax.axvline(v.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {v.mean():.5f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # R²
        ax = axes[1, j]
        v = df[f'{model}_R²'].dropna()
        ax.hist(v, bins=12, color=COLORS[j], alpha=0.7, edgecolor='black')
        ax.set_title(f'{model} - R²', fontweight='bold', fontsize=12)
        ax.set_xlabel('R²')
        ax.axvline(v.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {v.mean():.4f}')
        ax.axvline(0, color='black', linewidth=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Dir_Acc
        ax = axes[2, j]
        v = df[f'{model}_Dir_Acc'].dropna()
        ax.hist(v, bins=12, color=COLORS[j], alpha=0.7, edgecolor='black')
        ax.set_title(f'{model} - Dir_Acc', fontweight='bold', fontsize=12)
        ax.set_xlabel('Dir_Acc (%)')
        ax.axvline(v.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {v.mean():.1f}%')
        ax.axvline(50, color='black', linewidth=0.5, linestyle=':')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / "04_metrics_distribution.png"
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Saved: {out.name}")


def plot_directional_accuracy(df):
    """NEW v2 plot: per-model average Dir_Acc with 50% baseline."""
    dir_means = [df[f'{m}_Dir_Acc'].mean() for m in MODELS]
    dir_stds = [df[f'{m}_Dir_Acc'].std() for m in MODELS]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(MODELS, dir_means, yerr=dir_stds, color=COLORS, capsize=8,
                  edgecolor='black', linewidth=1)
    ax.axhline(50, color='red', linestyle='--', linewidth=2, label='50% (random guess)')
    ax.set_ylim(40, 65)
    ax.set_ylabel('Directional Accuracy (%)', fontsize=12)
    ax.set_title('Directional Accuracy by Model\n(v2 — Leak-Free Features, Returns Target)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)

    for bar, mean, std in zip(bars, dir_means, dir_stds):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{mean:.1f}% ± {std:.1f}', ha='center', fontsize=11, fontweight='bold')

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    out = PLOTS_DIR / "05_directional_accuracy.png"
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Saved: {out.name}")


def generate_summary_report(df):
    """Write summary_report.txt (overwrites any old version)."""
    report_path = RESULTS_DIR / "summary_report.txt"

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("BASELINE ML MODELS — FINAL SUMMARY REPORT (v2, LEAK-FREE)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Project: LLM-Orchestrated Financial Advisor for Bangladesh Stock Market\n")
        f.write("Phase: 3 - Baseline Forecasting (v2)\n")
        f.write(f"Total Stocks: {len(df)}\n")
        f.write("Models: Linear Regression, Random Forest, XGBoost, LightGBM\n")
        f.write("Target: Target_Return_1d (next-day return)\n")
        f.write("Features: Lag-1 indicators only (no same-day OHLCV)\n\n")

        f.write("=" * 70 + "\n")
        f.write("LEAKAGE FIX — v2 vs v1\n")
        f.write("=" * 70 + "\n\n")
        f.write("v1 (BUGGY) used Target_Price_1d and kept current-day OHLCV + all\n")
        f.write("    today's indicators as features. Linear Regression got R²=0.89\n")
        f.write("    because today_close[t] ≈ today_close[t+1].  Not skill — leak.\n")
        f.write("v2 (FIXED) predicts Target_Return_1d using ONLY yesterday's\n")
        f.write("    indicators. R² near zero is now realistic for daily returns.\n\n")

        f.write("=" * 70 + "\n")
        f.write("MODEL PERFORMANCE\n")
        f.write("=" * 70 + "\n\n")
        for model in MODELS:
            rmse = df[f'{model}_RMSE'].mean()
            mae = df[f'{model}_MAE'].mean()
            mape = df[f'{model}_MAPE'].mean() if f'{model}_MAPE' in df.columns else float('nan')
            r2 = df[f'{model}_R²'].mean()
            diracc = df[f'{model}_Dir_Acc'].mean()
            f.write(f"{model}:\n")
            f.write(f"  Average RMSE:        {rmse:.6f}\n")
            f.write(f"  Average MAE:         {mae:.6f}\n")
            f.write(f"  Average MAPE:        {mape:.2f}%\n")
            f.write(f"  Average R²:          {r2:.4f}\n")
            f.write(f"  Average Dir_Acc:     {diracc:.1f}% (50% = random)\n\n")

        if 'best_model' in df.columns:
            f.write("=" * 70 + "\n")
            f.write("BEST MODEL DISTRIBUTION (by RMSE)\n")
            f.write("=" * 70 + "\n\n")
            best_counts = df['best_model'].value_counts()
            for model, count in best_counts.items():
                pct = count / len(df) * 100
                f.write(f"  {model}: {count} stocks ({pct:.1f}%)\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("TOP 10 STOCKS BY Dir_Acc (best LightGBM or whichever model wins)\n")
        f.write("=" * 70 + "\n\n")

        # Pick max Dir_Acc across all 4 models per stock
        dir_cols = [f'{m}_Dir_Acc' for m in MODELS]
        df['_best_dir'] = df[dir_cols].max(axis=1)
        df['_best_dir_model'] = df[dir_cols].idxmax(axis=1).str.replace('_Dir_Acc', '')

        top10 = df.sort_values('_best_dir', ascending=False).head(10)
        top10_view = top10[['stock', '_best_dir_model', '_best_dir']].copy()
        top10_view.columns = ['stock', 'best_model_dir', 'best_dir_acc_%']
        f.write(top10_view.to_string(index=False))
        f.write("\n\n")

        f.write("=" * 70 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"1. Leakage fixed: target is now next-day return, features are lag-1.\n")
        f.write(f"2. R² values are realistic (close to zero) — daily return prediction is\n")
        f.write(f"   genuinely hard for an emerging market.\n")
        f.write(f"3. Directional Accuracy is the most useful metric for trading: any model\n")
        f.write(f"   with Dir_Acc > 50% has positive expected value on direction bets.\n")

        # Compare to v1 if backup exists
        backup_csv = RESULTS_DIR.parent / 'baseline_LEAKY_backup' / 'baseline_results.csv'
        if backup_csv.exists():
            try:
                v1_df = pd.read_csv(backup_csv)
                f.write(f"\nv1 (LEAKY) summary for comparison:\n")
                for model in ['LinearRegression', 'RandomForest', 'XGBoost']:
                    if f'{model}_R2' in v1_df.columns:
                        f.write(f"  {model} v1 R²: {v1_df[f'{model}_R2'].mean():.4f}  →  v2 R²: {df[f'{model}_R²'].mean():.4f}\n")
            except Exception:
                pass

        f.write("\n" + "=" * 70 + "\n")

    logger.info(f"✅ Summary report: {report_path}")


def main():
    logger.info("=" * 70)
    logger.info("📊 Generating Visualizations (v2 — Leak-Free)")
    logger.info("=" * 70)

    df = load_results(suffix="_v2")
    if df is None:
        logger.error("Run baseline_trainer_v2.py first.")
        return

    logger.info(f"📁 Output: {PLOTS_DIR}\n")
    plot_model_comparison(df)
    plot_per_stock_performance(df)
    plot_best_model_distribution(df)
    plot_metrics_distribution(df)
    plot_directional_accuracy(df)
    generate_summary_report(df)

    logger.info("\n✨ All visualizations generated!")
    logger.info(f"📁 Check: {PLOTS_DIR}")


if __name__ == "__main__":
    main()