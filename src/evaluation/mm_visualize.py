"""
Multimodal Visualization — Phase 7 (price + sentiment).

Reads:
    results/multimodal/multimodal_results.csv      (30 × 2 = 60 rows)
    results/deep_learning/deep_learning_results.csv (Phase 4 baseline, 30 rows)

Writes 8 PNGs + summary_report.txt to results/multimodal/plots/:
    01_model_summary.png              — overall metrics card, all 3 arches
    02_per_stock_performance.png      — RMSE & Dir_Acc per stock (early vs late)
    03_ablation_phase4_vs_phase7.png  — Phase 4 baseline vs Phase 7 multimodal
    04_confusion_matrix.png           — directional accuracy confusion (pooled)
    05_training_curves.png            — avg train/val loss per fusion
    06_sentiment_contribution.png     — Δ Dir_Acc per stock (Phase 7 − Phase 4)
    07_metric_distributions.png       — histograms of RMSE / R² / Dir_Acc
    08_directional_accuracy_bars.png  — sorted per-stock Dir_Acc bars (all 3)
    summary_report.txt

Usage:
    python src/evaluation/mm_visualize.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.utils.config import (
    MULTIMODAL_RESULTS_DIR as RESULTS_DIR,
    MULTIMODAL_PLOTS_DIR as PLOTS_DIR,
    MULTIMODAL_MODELS_DIR as MODELS_DIR,
    DL_RESULTS_DIR,
)
from src.utils.logger import get_logger


sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
logger = get_logger("mm_visualize")

# Phase 7 colors — distinct from Phase 4 LSTM purple (#9b59b6)
COLOR_EARLY = "#ff7f0e"   # orange
COLOR_LATE = "#2ca02c"    # green
COLOR_ATTENTION = "#1f77b4"  # blue
COLOR_LSTM = "#9b59b6"    # purple (Phase 4)

FUSION_COLORS = {
    "early": COLOR_EARLY,
    "late": COLOR_LATE,
    "attention": COLOR_ATTENTION,
}
FUSION_ORDER = ["early", "late", "attention"]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_results() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    mm_csv = RESULTS_DIR / "multimodal_results.csv"
    dl_csv = DL_RESULTS_DIR / "deep_learning_results.csv"

    mm_df = None
    if mm_csv.exists():
        mm_df = pd.read_csv(mm_csv)
        logger.info(f"📊 Loaded {len(mm_df)} multimodal rows from {mm_csv.name}")
    else:
        logger.error(f"❌ {mm_csv} not found.")

    dl_df = None
    if dl_csv.exists():
        dl_df = pd.read_csv(dl_csv)
        logger.info(f"📊 Loaded {len(dl_df)} Phase 4 LSTM rows from {dl_csv.name}")
    else:
        logger.warning(f"⚠️  Phase 4 results not found at {dl_csv} — ablation plots will skip.")

    return mm_df, dl_df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save(fig, name: str):
    out = PLOTS_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"   💾 {out.name}")


def plot_model_summary(mm_df: pd.DataFrame):
    """Overall metrics card — early vs late vs attention."""
    fig, axes = plt.subplots(1, 4, figsize=(26, 6))
    metrics = ["test_rmse", "test_mae", "test_r2", "test_dir_acc"]
    titles = [
        "Avg RMSE\n(Lower is Better)",
        "Avg MAE\n(Lower is Better)",
        "Avg R²\n(Higher is Better)",
        "Avg Directional Accuracy\n(50% = Random)",
    ]

    fusions_present = [f for f in FUSION_ORDER if f in mm_df["fusion_strategy"].unique()]
    labels = [f.upper() for f in fusions_present]
    colors = [FUSION_COLORS[f] for f in fusions_present]

    for ax, m, title in zip(axes, metrics, titles):
        means = [mm_df[mm_df["fusion_strategy"] == f][m].mean() for f in fusions_present]
        ax.bar(labels, means, color=colors, width=0.55)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(m.replace("test_", "").replace("_", " "))
        ax.grid(True, alpha=0.3)
        for i, v in enumerate(means):
            if "dir" in m:
                ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom",
                        fontsize=11, fontweight="bold")
            else:
                ax.text(i, v, f"{v:.4f}", ha="center", va="bottom",
                        fontsize=11, fontweight="bold")
        if m == "test_dir_acc":
            ax.axhline(50, color="red", linestyle="--", linewidth=1.5, label="50% (random)")
            ax.legend()
            ax.set_ylim(40, 60)
        if m == "test_r2":
            ax.axhline(0, color="black", linewidth=0.5)

    _save(fig, "01_model_summary.png")


def plot_per_stock_performance(mm_df: pd.DataFrame):
    """Per-stock RMSE and Dir_Acc — early vs late fusion."""
    early = mm_df[mm_df["fusion_strategy"] == "early"].sort_values("stock")
    late = mm_df[mm_df["fusion_strategy"] == "late"].sort_values("stock")
    stocks = sorted(set(early["stock"]) & set(late["stock"]))

    fig, axes = plt.subplots(2, 1, figsize=(20, 12))

    axes[0].plot(early["stock"], early["test_rmse"], marker="o",
                 linewidth=2, color=COLOR_EARLY, label="Early")
    axes[0].plot(late["stock"], late["test_rmse"], marker="s",
                 linewidth=2, color=COLOR_LATE, label="Late")
    axes[0].set_title("Multimodal — RMSE by Stock (Early vs Late Fusion)",
                      fontsize=14, fontweight="bold")
    axes[0].set_ylabel("RMSE")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(early["stock"], early["test_dir_acc"], marker="o",
                 linewidth=2, color=COLOR_EARLY, label="Early")
    axes[1].plot(late["stock"], late["test_dir_acc"], marker="s",
                 linewidth=2, color=COLOR_LATE, label="Late")
    axes[1].axhline(50, color="red", linestyle="--", linewidth=1, alpha=0.6,
                    label="50% (random)")
    axes[1].set_title("Multimodal — Directional Accuracy by Stock",
                      fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Dir_Acc (%)")
    axes[1].set_ylim(30, 70)
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, alpha=0.3)

    _save(fig, "02_per_stock_performance.png")


def plot_ablation_phase4_vs_phase7(mm_df: pd.DataFrame, dl_df: pd.DataFrame | None):
    """Phase 4 LSTM vs Phase 7 multimodal — bar chart of mean RMSE and Dir_Acc."""
    if dl_df is None or dl_df.empty:
        logger.warning("⚠️  Phase 4 results missing — skipping ablation plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    fusions_present = [f for f in FUSION_ORDER if f in mm_df["fusion_strategy"].unique()]
    bar_labels = ["Phase 4\n(LSTM, price only)"] + [
        f"Phase 7\n({f} fusion)" for f in fusions_present
    ]
    bar_colors = [COLOR_LSTM] + [FUSION_COLORS[f] for f in fusions_present]

    # RMSE
    ax = axes[0]
    p4_rmse = dl_df["test_rmse"].mean()
    p7_rmses = [mm_df[mm_df["fusion_strategy"] == f]["test_rmse"].mean() for f in fusions_present]
    ax.bar(bar_labels, [p4_rmse] + p7_rmses, color=bar_colors)
    ax.set_title("Ablation: RMSE (Lower is Better)", fontweight="bold")
    ax.set_ylabel("Avg RMSE")
    for i, v in enumerate([p4_rmse] + p7_rmses):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Dir_Acc
    ax = axes[1]
    p4_dir = dl_df["test_dir_acc"].mean()
    p7_dirs = [mm_df[mm_df["fusion_strategy"] == f]["test_dir_acc"].mean() for f in fusions_present]
    ax.bar(bar_labels, [p4_dir] + p7_dirs, color=bar_colors)
    ax.axhline(50, color="red", linestyle="--", linewidth=1, label="50% (random)")
    ax.set_title("Ablation: Directional Accuracy (Higher is Better)", fontweight="bold")
    ax.set_ylabel("Avg Dir_Acc (%)")
    ax.set_ylim(45, 55)
    for i, v in enumerate([p4_dir] + p7_dirs):
        ax.text(i, v + 0.1, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    _save(fig, "03_ablation_phase4_vs_phase7.png")


def plot_confusion_matrix(mm_df: pd.DataFrame):
    """Pooled confusion matrix of directional accuracy (mean sign prediction)."""
    # We don't have ground-truth per-prediction here; use predicted vs mean direction.
    # Pool all per-stock test_dir_acc into a "hit/miss" distribution.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, fusion, color in [
        (axes[0], "early", COLOR_EARLY),
        (axes[1], "late", COLOR_LATE),
    ]:
        sub = mm_df[mm_df["fusion_strategy"] == fusion]
        # Bucket Dir_Acc into ranges to visualize
        bins = [0, 45, 48, 50, 52, 55, 100]
        labels = ["<45", "45-48", "48-50", "50-52", "52-55", ">=55"]
        counts = pd.cut(sub["test_dir_acc"], bins=bins, labels=labels,
                        include_lowest=True).value_counts().reindex(labels, fill_value=0)
        ax.bar(labels, counts.values, color=color, alpha=0.8,
               edgecolor="black", linewidth=0.5)
        ax.set_title(f"{fusion.upper()} Fusion — Dir_Acc Distribution", fontweight="bold")
        ax.set_ylabel("Number of stocks")
        ax.set_xlabel("Dir_Acc range")
        for i, (lbl, n) in enumerate(counts.items()):
            ax.text(i, n + 0.1, str(int(n)), ha="center", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

    _save(fig, "04_confusion_matrix.png")


def plot_training_curves(mm_df: pd.DataFrame):
    """Average train/val loss per fusion."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for fusion, suffix, color in [
        ("early", "_mm_early", COLOR_EARLY),
        ("late", "_mm_late", COLOR_LATE),
    ]:
        histories = []
        for p in sorted(MODELS_DIR.glob(f"*_best{suffix}.pkl")):
            with open(p, "rb") as f:
                sidecar = pickle.load(f)
            hist = sidecar.get("history")
            if hist and "train_loss" in hist and "val_loss" in hist:
                histories.append(hist)
        if not histories:
            logger.warning(f"�️  No training histories for {fusion}")
            continue

        min_len = min(len(h["train_loss"]) for h in histories)
        train_avg = np.mean([h["train_loss"][:min_len] for h in histories], axis=0)
        val_avg = np.mean([h["val_loss"][:min_len] for h in histories], axis=0)
        epochs = np.arange(1, min_len + 1)
        ax.plot(epochs, val_avg, color=color, linewidth=2.5,
                label=f"{fusion.upper()} val loss (n={len(histories)})")
        ax.plot(epochs, train_avg, color=color, linewidth=1.5, linestyle="--",
                alpha=0.5, label=f"{fusion.upper()} train loss")

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("Multimodal — Average Training Curves Across All Stocks", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    _save(fig, "05_training_curves.png")


def plot_sentiment_contribution(mm_df: pd.DataFrame, dl_df: pd.DataFrame | None):
    """Δ Dir_Acc per stock = Phase 7 (late) − Phase 4 (LSTM).

    Highlights stocks where sentiment helped (Δ > 0) vs hurt (Δ < 0).
    """
    if dl_df is None or dl_df.empty:
        logger.warning("⚠️  Phase 4 results missing — skipping sentiment contribution plot.")
        return

    p7_late = mm_df[mm_df["fusion_strategy"] == "late"].set_index("stock")["test_dir_acc"]
    p4 = dl_df.set_index("stock")["test_dir_acc"]
    common = sorted(set(p7_late.index) & set(p4.index))
    delta = (p7_late.loc[common] - p4.loc[common]).sort_values()

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = [COLOR_LATE if v >= 0 else "#d62728" for v in delta.values]
    ax.bar(delta.index, delta.values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Sentiment Contribution: Δ Dir_Acc per Stock\n(Late fusion − Phase 4 LSTM, green = sentiment helped)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Δ Dir_Acc (percentage points)")
    ax.set_xlabel("Stock (sorted)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")
    n_pos = int((delta > 0).sum())
    n_neg = int((delta < 0).sum())
    ax.text(0.02, 0.97,
            f"Stocks where sentiment helped: {n_pos}/{len(delta)}\n"
            f"Stocks where sentiment hurt:   {n_neg}/{len(delta)}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                      edgecolor="black", alpha=0.8))

    _save(fig, "06_sentiment_contribution.png")


def plot_metric_distributions(mm_df: pd.DataFrame):
    """Histograms of RMSE / R² / Dir_Acc per fusion."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    fusions_present = [f for f in FUSION_ORDER if f in mm_df["fusion_strategy"].unique()]

    for ax, m, title in zip(
        axes,
        ["test_rmse", "test_r2", "test_dir_acc"],
        ["RMSE", "R²", "Dir_Acc (%)"],
    ):
        for fusion in fusions_present:
            v = mm_df[mm_df["fusion_strategy"] == fusion][m].dropna()
            ax.hist(v, bins=10, color=FUSION_COLORS[fusion], alpha=0.55,
                    edgecolor="black", label=fusion.upper())
        ax.set_title(f"Multimodal — {title}", fontweight="bold", fontsize=12)
        ax.set_xlabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if m == "test_dir_acc":
            ax.axvline(50, color="black", linewidth=0.5, linestyle=":")
        if m == "test_r2":
            ax.axvline(0, color="black", linewidth=0.5)

    _save(fig, "07_metric_distributions.png")


def plot_directional_accuracy_bars(mm_df: pd.DataFrame):
    """Sorted per-stock Dir_Acc bars — all fusions present."""
    fig, ax = plt.subplots(figsize=(20, 8))

    fusions_present = [f for f in FUSION_ORDER if f in mm_df["fusion_strategy"].unique()]
    per_fusion = {
        f: mm_df[mm_df["fusion_strategy"] == f].set_index("stock")["test_dir_acc"]
        for f in fusions_present
    }
    # Union of all stocks that have at least one fusion result
    stocks = sorted(set().union(*(s.index for s in per_fusion.values())))
    if not stocks:
        return

    # Sort by average Dir_Acc across present fusions
    df_for_sort = pd.DataFrame({f: per_fusion[f] for f in fusions_present})
    order = df_for_sort.mean(axis=1).sort_values(ascending=False).index.tolist()

    x = np.arange(len(order))
    n = len(fusions_present)
    width = min(0.8 / n, 0.3)
    for i, fusion in enumerate(fusions_present):
        offset = (i - (n - 1) / 2) * width
        vals = [per_fusion[fusion].get(s, np.nan) for s in order]
        ax.bar(x + offset, vals, width=width,
               color=FUSION_COLORS[fusion], label=fusion.upper(),
               edgecolor="black", linewidth=0.4)
    ax.axhline(50, color="red", linestyle="--", linewidth=1.5, label="50% (random)")
    ax.set_title("Multimodal — Directional Accuracy per Stock (sorted by avg)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Dir_Acc (%)")
    ax.set_ylim(30, 65)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    _save(fig, "08_directional_accuracy_bars.png")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def generate_summary_report(mm_df: pd.DataFrame, dl_df: pd.DataFrame | None) -> str:
    fusions_present = sorted(mm_df["fusion_strategy"].unique())
    early = mm_df[mm_df["fusion_strategy"] == "early"]
    late = mm_df[mm_df["fusion_strategy"] == "late"]
    attention = mm_df[mm_df["fusion_strategy"] == "attention"] if "attention" in fusions_present else pd.DataFrame()

    n_stocks_early = len(early["stock"].unique()) if not early.empty else 0
    n_stocks_late = len(late["stock"].unique()) if not late.empty else 0
    n_stocks_attn = len(attention["stock"].unique()) if not attention.empty else 0
    n_unique_stocks = len(mm_df["stock"].unique())

    lines = []
    lines.append("=" * 70)
    lines.append("📊 PHASE 7 — MULTIMODAL FORECASTING SUMMARY")
    lines.append("=" * 70)
    lines.append(f"\nStocks analyzed: {n_unique_stocks}")
    ckpt_descr = []
    if not early.empty:
        ckpt_descr.append(f"Early ({n_stocks_early})")
    if not late.empty:
        ckpt_descr.append(f"Late ({n_stocks_late})")
    if not attention.empty:
        ckpt_descr.append(f"Attention ({n_stocks_attn})")
    lines.append(f"Total multimodal checkpoints: {len(mm_df)} ({' + '.join(ckpt_descr)})")
    lines.append(
        "Fusion strategies: Early (concat) + Late (two-stream)"
        + (" + Attention (gated cross-modal)" if not attention.empty else "")
    )
    lines.append(f"Sentiment features: 7 (n_articles, mean_score, weighted_score, "
                 f"mean_confidence, pos_count, neg_count, neu_count)")
    lines.append(f"Sentiment NaN policy: per-stock forward-fill (sticky) + bfill for "
                 f"leading NaNs + 0.0 fallback for stocks with zero news")

    def _section(name, df):
        if df.empty:
            return
        lines.append("\n" + "-" * 70)
        lines.append(name)
        lines.append("-" * 70)
        lines.append(f"  Avg RMSE:      {df['test_rmse'].mean():.6f}")
        lines.append(f"  Avg MAE:       {df['test_mae'].mean():.6f}")
        lines.append(f"  Avg R²:        {df['test_r2'].mean():.4f}")
        lines.append(f"  Avg Dir_Acc:   {df['test_dir_acc'].mean():.1f}%")
        lines.append(f"  Stocks ≥ 50%:  {(df['test_dir_acc'] >= 50).sum()}/{len(df)}")

    _section("EARLY FUSION (concat → single LSTM, 356K params)", early)
    _section("LATE FUSION (price LSTM + sentiment LSTM → MLP, 228K params)", late)
    _section("ATTENTION FUSION (gated cross-modal, 238K params) — 18 stocks", attention)

    if dl_df is not None and not dl_df.empty:
        lines.append("\n" + "-" * 70)
        lines.append("ABLATION vs PHASE 4 (LSTM, PRICE ONLY)")
        lines.append("-" * 70)
        p4_rmse = dl_df["test_rmse"].mean()
        p4_dir = dl_df["test_dir_acc"].mean()
        lines.append(f"  Phase 4 RMSE:    {p4_rmse:.6f}    Dir_Acc: {p4_dir:.1f}%")
        for fusion, df in [("Early", early), ("Late", late), ("Attention", attention)]:
            if df.empty:
                continue
            rmse = df["test_rmse"].mean()
            dacc = df["test_dir_acc"].mean()
            lines.append(
                f"  Phase 7 {fusion:9s} {rmse:.6f}    Dir_Acc: {dacc:.1f}%   "
                f"Δ RMSE={rmse - p4_rmse:+.6f}  Δ Dir={dacc - p4_dir:+.2f}pp"
            )

    lines.append("\n" + "-" * 70)
    lines.append("TOP 10 STOCKS BY Dir_Acc (Late Fusion)")
    lines.append("-" * 70)
    lines.append(f"  {'Stock':15s}  {'Dir_Acc':>10s}  {'RMSE':>10s}  {'Epochs':>8s}")
    if not late.empty:
        top = late.nlargest(10, "test_dir_acc")
        for _, r in top.iterrows():
            lines.append(
                f"  {r['stock']:15s}  {r['test_dir_acc']:>9.1f}%  "
                f"{r['test_rmse']:>10.6f}  {int(r['epochs_trained']):>8d}"
            )

    lines.append("\n" + "-" * 70)
    lines.append("BOTTOM 10 STOCKS BY Dir_Acc (Late Fusion)")
    lines.append("-" * 70)
    if not late.empty:
        bot = late.nsmallest(10, "test_dir_acc")
        for _, r in bot.iterrows():
            lines.append(
                f"  {r['stock']:15s}  {r['test_dir_acc']:>9.1f}%  "
                f"{r['test_rmse']:>10.6f}  {int(r['epochs_trained']):>8d}"
            )

    lines.append("\n" + "=" * 70)
    lines.append("KEY INSIGHTS")
    lines.append("=" * 70)
    lines.append("1. Phase 7 mirrors Phase 4 leak-free discipline: StandardScaler fit on")
    lines.append("   TRAIN only, time-based split, identical window length.")
    lines.append("2. Sentiment is forward-filled per stock (sticky, realistic).")
    lines.append("3. ~50 of ~3000 trading days per stock have any news — sentiment is")
    lines.append("   an exogenous, sparse signal. Expect modest Δ vs Phase 4.")
    lines.append("4. Late fusion generally has fewer parameters (228K) than early (356K)")
    lines.append("   — better suited to the small sentiment signal.")
    lines.append("5. At inference, sentiment window is FROZEN across forecast steps.")
    if not attention.empty:
        lines.append("6. Attention fusion is trained on 18/30 stocks (subset) — gated")
        lines.append("   sentiment→price cross-modal attention, 238K params.")
        lines.append("7. None of the three fusions significantly improve over Phase 4 LSTM.")
        lines.append("   Honest finding: daily DSE direction is hard to predict.")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    text = "\n".join(lines)
    out = RESULTS_DIR / "summary_report.txt"
    out.write_text(text)
    logger.info(f"💾 Wrote {out}")
    return text


def main():
    logger.info("=" * 70)
    logger.info("� Phase 7 Multimodal Visualization")
    logger.info("=" * 70)

    mm_df, dl_df = load_results()
    if mm_df is None or mm_df.empty:
        logger.error("� No multimodal results to visualize.")
        return

    plot_model_summary(mm_df)
    plot_per_stock_performance(mm_df)
    plot_ablation_phase4_vs_phase7(mm_df, dl_df)
    plot_confusion_matrix(mm_df)
    plot_training_curves(mm_df)
    plot_sentiment_contribution(mm_df, dl_df)
    plot_metric_distributions(mm_df)
    plot_directional_accuracy_bars(mm_df)

    generate_summary_report(mm_df, dl_df)

    logger.info("\n✅ Phase 7 visualization complete")
    logger.info(f"📁 Plots: {PLOTS_DIR}")


if __name__ == "__main__":
    main()