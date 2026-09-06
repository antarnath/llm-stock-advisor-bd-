"""
Phase 8 — XAI Master Runner.

Runs SHAP (LinearExplainer / TreeExplainer) on all 30 DSE stocks using the
best Phase 3 baseline model per stock. Computes:

  - Global feature importance (mean |SHAP| per feature, per stock)
  - Per-stock local explanations (top-10 features for 5 sample predictions)
  - Cross-stock aggregate (top-5 features overall, top-5 features per stock)
  - Natural-language explanations for sample predictions
  - LIME comparison for top-3 stocks

Outputs (results/xai/):
  - global_importance_per_stock.csv   : (30 stocks × 27 features)
  - aggregate_importance.csv          : mean |SHAP| per feature across 30 stocks
  - local_explanations_sample.csv     : 5 samples × top-10 features per stock
  - natural_language_explanations.txt : 30 sample NL explanations
  - lime_top3_stocks.csv              : LIME top features for 3 representative stocks
  - summary_report.txt                : aggregate stats + key insights
  - plots/                            : 5 PNGs

Usage:
    .venv/bin/python scripts/phase8_xai.py
    .venv/bin/python scripts/phase8_xai.py --max-stocks 5  # smoke test
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config import (
    PROCESSED_DATA_DIR as DATA_DIR,
    BASELINE_MODELS_DIR as MODELS_DIR,
    BASELINE_RESULTS_DIR as RESULTS_DIR,
)
from src.utils.logger import get_logger
from src.xai.shap_explainer import ShapExplainer
from src.xai.lime_explainer import LimeExplainer
from src.xai.natural_language import explain_dataframe


RESULTS_DIR = RESULTS_DIR  # results/baseline (override below if needed)
XAI_DIR = _PROJECT_ROOT / "results" / "xai"
XAI_PLOTS = XAI_DIR / "plots"
XAI_DIR.mkdir(parents=True, exist_ok=True)
XAI_PLOTS.mkdir(parents=True, exist_ok=True)

logger = get_logger("phase8_xai")


# ---------------------------------------------------------------------------
# Data prep helpers
# ---------------------------------------------------------------------------

def load_baseline_for_stock(stock: str, models_dir: Path):
    """Find the best_v2 model for the stock and load it + its feature list."""
    # Prefer _v2.pkl (the leak-free protocol models)
    candidates = sorted(models_dir.glob(f"{stock}_best*.pkl"))
    if not candidates:
        return None, None, None
    # Prefer the v2 over v1
    candidates_v2 = [p for p in candidates if "_v2" in p.name]
    model_path = candidates_v2[0] if candidates_v2 else candidates[0]
    try:
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
    except Exception as e:
        logger.warning(f"   ⚠️  Failed to load {model_path.name}: {e}")
        return None, None, None
    if not isinstance(bundle, dict) or "model" not in bundle:
        return None, None, None
    return bundle["model"], bundle.get("features", []), model_path


def prepare_stock_data(stock: str, features: list[str], data_dir: Path):
    """Load processed CSV, build X matrix in the saved feature order, time-split."""
    csv_path = data_dir / f"{stock}_processed_v2.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, parse_dates=["date"])
    drop_cols = [c for c in df.columns if c in {"date", "code", "name", "sector"}
                 or c.startswith("Target_")]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    # Ensure column order matches the saved feature list
    missing = [f for f in features if f not in X.columns]
    if missing:
        return None
    X = X[features]
    # Time-based split (last 20% is test, matches Phase 3)
    n = len(X)
    test_size = 0.20
    val_end = int(n * (1.0 - test_size))
    X_background = X.iloc[:val_end].sample(n=min(100, val_end), random_state=42).values
    X_test = X.iloc[val_end:].values
    return X_background, X_test, df.iloc[val_end:].copy()


def shap_one_stock(stock: str, models_dir: Path, data_dir: Path, n_test: int = 200):
    """Run SHAP on one stock, return (global_df, list of local_dfs)."""
    model, features, model_path = load_baseline_for_stock(stock, models_dir)
    if model is None:
        return None, None, None
    prep = prepare_stock_data(stock, features, data_dir)
    if prep is None:
        return None, None, None
    X_background, X_test, test_df = prep
    if len(X_test) > n_test:
        idx = np.random.default_rng(42).choice(len(X_test), size=n_test, replace=False)
        X_test = X_test[idx]
        test_df = test_df.iloc[idx].reset_index(drop=True)

    try:
        explainer = ShapExplainer(
            model=model,
            X_background=X_background,
            X_test=X_test,
            feature_names=features,
            model_type=None,
            n_test=len(X_test),
        )
        global_df = explainer.global_importance()
    except Exception as e:
        logger.warning(f"   ⚠️  SHAP failed for {stock}: {e}")
        return None, None, None

    # Capture 5 local explanations
    local_dfs = []
    for i in range(min(5, len(X_test))):
        # Build a feature→value map for this test instance
        feat_value_map = dict(zip(features, X_test[i]))
        try:
            local = explainer.local_importance(idx=i, top_k=10).copy()
            local["stock"] = stock
            local["sample_idx"] = i
            # Re-align feature_value column to the (sorted top-10) features
            local["feature_value"] = local["feature"].map(feat_value_map)
            local_dfs.append(local)
        except Exception as e:
            logger.warning(f"   ⚠️  local SHAP failed for {stock} sample {i}: {e}")
            continue

    # Also capture predicted returns + base value for NL explanations
    try:
        preds = model.predict(X_test)
        base = explainer.base_value()
    except Exception:
        preds = np.zeros(len(X_test))
        base = 0.0
    return global_df, local_dfs, {"preds": preds, "base": base, "X_test": X_test}


def lime_one_stock(stock: str, models_dir: Path, data_dir: Path, n_samples: int = 500):
    """Run LIME on one stock for a single instance (sanity comparison)."""
    model, features, _ = load_baseline_for_stock(stock, models_dir)
    if model is None:
        return None
    prep = prepare_stock_data(stock, features, data_dir)
    if prep is None:
        return None
    X_background, X_test, _ = prep
    try:
        explainer = LimeExplainer(model, X_background[:100], features)
        weights = explainer.explain_as_dict(X_test[0], num_features=10, num_samples=n_samples)
        return [(stock, desc, w) for desc, w in weights]
    except Exception as e:
        logger.warning(f"   ⚠️  LIME failed for {stock}: {e}")
        return None


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_aggregate_importance(agg: pd.DataFrame, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = agg.head(15)
    signed_col = "signed_sum" if "signed_sum" in top.columns else "mean_shap"
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in top[signed_col]]
    ax.barh(range(len(top)), top["mean_abs_shap"], color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP| across 30 DSE stocks")
    ax.set_title("Phase 8 — Global Feature Importance (XAI)\nTop-15 features, sign indicates direction",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_per_stock_top_features(per_stock: pd.DataFrame, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Pivot: rows = stocks, cols = features (use rank, not raw SHAP)
    pivot = per_stock.pivot_table(
        index="stock", columns="feature", values="mean_abs_shap", fill_value=0,
    )
    # Pick top-10 most important features across all stocks
    top_feats = pivot.mean(axis=0).sort_values(ascending=False).head(10).index.tolist()
    sub = pivot[top_feats]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        sub, cmap="viridis", ax=ax, cbar_kws={"label": "Mean |SHAP|"},
        linewidths=0.3, linecolor="gray",
    )
    ax.set_title("Phase 8 — Per-Stock Top-10 Feature Importance\n(rows=stocks, cols=features)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Stock")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_top5_per_stock(per_stock: pd.DataFrame, out_path: Path):
    """For each stock, find the top-1 feature — visualize stability."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top1 = per_stock.sort_values("mean_abs_shap", ascending=False).groupby("stock").head(1)
    counts = top1["feature"].value_counts()

    fig, ax = plt.subplots(figsize=(12, 6))
    counts.plot(kind="barh", ax=ax, color="#1f77b4", edgecolor="black")
    ax.set_xlabel(f"Number of stocks for which this feature is #1 (n={len(top1)})")
    ax.set_title("Phase 8 — Top Feature per Stock (stability check)\n"
                 "If one feature dominates, models rely on a common signal.",
                 fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_sign_distribution(agg: pd.DataFrame, out_path: Path):
    """Show how often each feature pushes up vs down."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = agg.head(15).copy()
    signed_col = "signed_sum" if "signed_sum" in top.columns else "mean_shap"
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in top[signed_col]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top)), top[signed_col], color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel(f"Mean {signed_col} (positive = pushes prediction UP)")
    ax.set_title("Phase 8 — Direction of Feature Effects",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_shap_vs_lime(lime_df: pd.DataFrame, per_stock: pd.DataFrame, out_path: Path):
    """Compare LIME top-1 vs SHAP top-1 per stock."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lime_top1 = lime_df.sort_values("abs_weight", ascending=False).groupby("stock").head(1)
    lime_top1_features = set(lime_top1["feature"].tolist())
    shap_top1 = per_stock.sort_values("mean_abs_shap", ascending=False).groupby("stock").head(1)
    shap_top1_features = set(shap_top1["feature"].tolist())
    common = lime_top1_features & shap_top1_features
    only_lime = lime_top1_features - shap_top1_features
    only_shap = shap_top1_features - lime_top1_features

    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["Both SHAP & LIME", "Only SHAP top-1", "Only LIME top-1"]
    values = [len(common), len(only_shap), len(only_lime)]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]
    ax.bar(categories, values, color=colors, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(values):
        ax.text(i, v + 0.1, str(v), ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of unique features")
    ax.set_title("Phase 8 — SHAP vs LIME Top-Feature Agreement",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def write_summary_report(
    per_stock: pd.DataFrame,
    agg: pd.DataFrame,
    nl_examples: list[str],
    out_path: Path,
):
    top_overall = agg.head(5)
    top1 = per_stock.sort_values("mean_abs_shap", ascending=False).groupby("stock").head(1)
    top1_counts = top1["feature"].value_counts()

    lines = []
    lines.append("=" * 70)
    lines.append("📊 PHASE 8 — EXPLAINABLE AI (XAI) SUMMARY")
    lines.append("=" * 70)
    lines.append(f"\nStocks analyzed: {per_stock['stock'].nunique()}")
    lines.append(f"Features per stock: {per_stock['feature'].nunique()}")
    lines.append(f"Method: SHAP (LinearExplainer / TreeExplainer based on best model type)")
    lines.append(f"Test samples per stock: 200 (subsampled from time-based test split)")
    lines.append("")
    lines.append("-" * 70)
    lines.append("TOP-5 FEATURES OVERALL (mean |SHAP| across all 30 stocks)")
    lines.append("-" * 70)
    for _, r in top_overall.iterrows():
        lines.append(
            f"  {r['feature']:20s}  mean |SHAP|={r['mean_abs_shap']:.6f}  "
            f"mean SHAP={r['mean_shap']:+.6f}"
        )

    lines.append("")
    lines.append("-" * 70)
    lines.append("TOP-1 FEATURE STABILITY (how often each feature is #1 for a stock)")
    lines.append("-" * 70)
    for feat, n in top1_counts.head(5).items():
        lines.append(f"  {feat:20s}  #1 for {n}/30 stocks")

    lines.append("")
    lines.append("-" * 70)
    lines.append("SAMPLE NATURAL-LANGUAGE EXPLANATIONS (first 5 of 30)")
    lines.append("-" * 70)
    for ex in nl_examples[:5]:
        lines.append(ex)
        lines.append("")

    lines.append("=" * 70)
    lines.append("KEY INSIGHTS")
    lines.append("=" * 70)
    lines.append("1. SHAP exposes what Phase 3 baseline models actually learned.")
    lines.append("2. If top features have low SHAP magnitudes → models learned weak signals.")
    lines.append("3. If top-1 feature varies wildly across stocks → no universal driver.")
    lines.append("4. This mechanistic evidence complements the ~50% Dir_Acc finding from")
    lines.append("   Phases 3-7: even when a model picks a feature, its effect is small.")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 8 XAI — SHAP + LIME.")
    parser.add_argument("--max-stocks", type=int, default=None,
                        help="Limit to N stocks (debug).")
    parser.add_argument("--n-test", type=int, default=200,
                        help="Test samples per stock for SHAP.")
    parser.add_argument("--skip-lime", action="store_true",
                        help="Skip LIME comparison (faster).")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("🔍 PHASE 8 — EXPLAINABLE AI (SHAP + LIME)")
    logger.info("=" * 70)

    # Discover stocks
    csv_files = sorted(DATA_DIR.glob("*_processed_v2.csv"))
    stocks = [p.stem.replace("_processed_v2", "") for p in csv_files]
    if args.max_stocks:
        stocks = stocks[: args.max_stocks]
    logger.info(f"📊 Stocks to analyze: {len(stocks)}")
    logger.info(f"📁 Models dir: {MODELS_DIR}")
    logger.info(f"📁 Data dir:   {DATA_DIR}")
    logger.info(f"📁 Output dir: {XAI_DIR}")
    logger.info("=" * 70)

    # ---- Run SHAP on all stocks ----
    logger.info("\n▶ Running SHAP per stock...")
    per_stock_rows = []
    all_local = []
    nl_examples = []

    for i, stock in enumerate(stocks, 1):
        logger.info(f"\n[{i}/{len(stocks)}] {stock}")
        global_df, local_dfs, extras = shap_one_stock(
            stock, MODELS_DIR, DATA_DIR, n_test=args.n_test,
        )
        if global_df is None:
            logger.warning(f"   ⏭️  skipped (no model or data)")
            continue
        global_df["stock"] = stock
        per_stock_rows.append(global_df)

        if local_dfs:
            all_local.extend(local_dfs)

        # Build one NL explanation per stock using sample 0
        if extras and len(local_dfs) > 0:
            first_local = local_dfs[0]
            try:
                pred = float(extras["preds"][0]) if extras["preds"] is not None else 0.0
            except Exception:
                pred = 0.0
            nl = explain_dataframe(
                stock=stock,
                predicted_return=pred,
                shap_values_row=first_local["shap_value"].values,
                feature_names=first_local["feature"].tolist(),
                feature_values_row=first_local.get("feature_value", pd.Series([None]*len(first_local))).values,
                top_k=5,
            )
            nl_examples.append(nl)

    if not per_stock_rows:
        logger.error("❌ No stocks processed.")
        return

    per_stock = pd.concat(per_stock_rows, ignore_index=True)
    per_stock.to_csv(XAI_DIR / "global_importance_per_stock.csv", index=False)
    logger.info(f"\n💾 Saved global_importance_per_stock.csv ({len(per_stock)} rows)")

    # Aggregate across stocks
    agg = per_stock.groupby("feature", as_index=False).agg(
        mean_abs_shap=("mean_abs_shap", "mean"),
        mean_shap=("mean_shap", "mean"),
        n_stocks=("stock", "count"),
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    agg.to_csv(XAI_DIR / "aggregate_importance.csv", index=False)
    logger.info(f"💾 Saved aggregate_importance.csv ({len(agg)} features)")

    # Local explanations
    if all_local:
        local_df = pd.concat(all_local, ignore_index=True)
        local_df.to_csv(XAI_DIR / "local_explanations_sample.csv", index=False)
        logger.info(f"💾 Saved local_explanations_sample.csv ({len(local_df)} rows)")

    # NL explanations
    nl_text = "\n\n" + ("\n\n" + "-" * 70 + "\n\n").join(nl_examples)
    (XAI_DIR / "natural_language_explanations.txt").write_text(nl_text)
    logger.info(f"💾 Saved natural_language_explanations.txt ({len(nl_examples)} explanations)")

    # ---- Run LIME on top-3 stocks (GP, BATBC, SQURPHARMA) ----
    lime_rows = []
    if not args.skip_lime:
        logger.info("\n▶ Running LIME on 3 representative stocks for cross-method check...")
        for stock in ["GP", "BATBC", "SQURPHARMA"]:
            if stock not in stocks:
                continue
            logger.info(f"   LIME on {stock}...")
            items = lime_one_stock(stock, MODELS_DIR, DATA_DIR)
            if items:
                for stk, desc, w in items:
                    base_feat = desc.split(" ")[0]
                    lime_rows.append({
                        "stock": stk,
                        "description": desc,
                        "feature": base_feat,
                        "weight": w,
                        "abs_weight": abs(w),
                    })
        if lime_rows:
            lime_df = pd.DataFrame(lime_rows)
            lime_df.to_csv(XAI_DIR / "lime_top3_stocks.csv", index=False)
            logger.info(f"💾 Saved lime_top3_stocks.csv ({len(lime_df)} rows)")

    # ---- Plots ----
    logger.info("\n▶ Generating plots...")
    plot_aggregate_importance(agg, XAI_PLOTS / "01_aggregate_importance.png")
    plot_per_stock_top_features(per_stock, XAI_PLOTS / "02_per_stock_heatmap.png")
    plot_top5_per_stock(per_stock, XAI_PLOTS / "03_top_feature_per_stock.png")
    plot_sign_distribution(agg, XAI_PLOTS / "04_signed_effects.png")
    if lime_rows:
        plot_shap_vs_lime(pd.DataFrame(lime_rows), per_stock, XAI_PLOTS / "05_shap_vs_lime.png")
    logger.info("   💾 5 plots saved")

    # ---- Summary report ----
    write_summary_report(per_stock, agg, nl_examples, XAI_DIR / "summary_report.txt")
    logger.info(f"💾 Saved summary_report.txt")

    logger.info("\n" + "=" * 70)
    logger.info("✅ Phase 8 XAI complete")
    logger.info(f"📁 Results: {XAI_DIR}")
    logger.info(f"📁 Plots:   {XAI_PLOTS}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()