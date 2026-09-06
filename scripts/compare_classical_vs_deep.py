"""
Compare Phase 3 baselines vs Phase 4 LSTM side-by-side.

For each stock, joins baseline_results_v2.csv (best of LR/RF/XGB/LGBM by RMSE)
with deep_learning_results.csv (LSTM) and prints a per-stock comparison.

Usage:
    python scripts/compare_phase3_vs_phase4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.utils.config import BASELINE_RESULTS_DIR, DL_RESULTS_DIR
from src.utils.logger import get_logger


logger = get_logger("phase3_vs_phase4")


def main():
    baseline_csv = BASELINE_RESULTS_DIR / "baseline_results_v2.csv"
    dl_csv = DL_RESULTS_DIR / "deep_learning_results.csv"

    if not baseline_csv.exists():
        logger.error(f"❌ Missing: {baseline_csv}")
        return
    if not dl_csv.exists():
        logger.error(f"❌ Missing: {dl_csv}")
        return

    base = pd.read_csv(baseline_csv)
    dl = pd.read_csv(dl_csv)

    # For each stock, find the Phase 3 baseline with the lowest RMSE
    metric_cols = [c for c in base.columns if c.endswith("_RMSE")]
    base["baseline_rmse"] = base[metric_cols].min(axis=1)
    base["baseline_dir_acc"] = base[
        [c.replace("RMSE", "Dir_Acc") for c in metric_cols]
    ].max(axis=1)

    merged = pd.merge(
        dl[["stock", "name", "test_rmse", "test_dir_acc", "epochs_trained"]],
        base[["stock", "best_model", "baseline_rmse", "baseline_dir_acc"]],
        on="stock",
        how="inner",
    )

    merged["lstm_better_dir"] = merged["test_dir_acc"] > merged["baseline_dir_acc"]
    merged["lstm_better_rmse"] = merged["test_rmse"] < merged["baseline_rmse"]

    logger.info("=" * 70)
    logger.info("📊 PHASE 3 (BASELINE) vs PHASE 4 (LSTM) — HEAD-TO-HEAD")
    logger.info("=" * 70)

    logger.info(f"\n{'Stock':15s}  {'Best Baseline':15s}  {'Base RMSE':>11s}  {'Base DirAcc':>11s}  "
                f"{'LSTM RMSE':>10s}  {'LSTM DirAcc':>11s}  {'LSTM Wins?':>11s}")
    logger.info("-" * 100)
    for _, r in merged.iterrows():
        wins = []
        if r["lstm_better_rmse"]:
            wins.append("RMSE")
        if r["lstm_better_dir"]:
            wins.append("DirAcc")
        wins_str = "+".join(wins) if wins else "—"
        logger.info(
            f"{r['stock']:15s}  {r['best_model']:15s}  "
            f"{r['baseline_rmse']:>11.6f}  {r['baseline_dir_acc']:>10.1f}%  "
            f"{r['test_rmse']:>10.6f}  {r['test_dir_acc']:>10.1f}%  "
            f"{wins_str:>11s}"
        )

    n = len(merged)
    n_rmse_win = int(merged["lstm_better_rmse"].sum())
    n_dir_win = int(merged["lstm_better_dir"].sum())
    n_both = int((merged["lstm_better_rmse"] & merged["lstm_better_dir"]).sum())

    logger.info("-" * 100)
    logger.info(f"\nLSTM wins on RMSE:    {n_rmse_win}/{n} ({n_rmse_win/n*100:.1f}%)")
    logger.info(f"LSTM wins on DirAcc:  {n_dir_win}/{n} ({n_dir_win/n*100:.1f}%)")
    logger.info(f"LSTM wins on both:    {n_both}/{n} ({n_both/n*100:.1f}%)")
    logger.info(
        f"\nMean baseline RMSE: {merged['baseline_rmse'].mean():.6f} | "
        f"Mean LSTM RMSE: {merged['test_rmse'].mean():.6f}"
    )
    logger.info(
        f"Mean baseline DirAcc: {merged['baseline_dir_acc'].mean():.1f}% | "
        f"Mean LSTM DirAcc: {merged['test_dir_acc'].mean():.1f}%"
    )

    out = DL_RESULTS_DIR / "phase3_vs_phase4_comparison.csv"
    merged.to_csv(out, index=False)
    logger.info(f"\n💾 Saved: {out}")


if __name__ == "__main__":
    main()
