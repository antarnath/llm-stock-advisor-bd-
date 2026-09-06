"""
Demo: end-to-end Portfolio Optimization using agent signals.

Pipeline:
  1. Multi-agent orchestrator produces ensemble signals for all 30 stocks
  2. Build covariance matrix from last 252 days of returns
  3. Run Mean-Variance + Risk-Parity optimizers under each risk profile
  4. Print the resulting portfolio weights + risk/return stats
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.agents import Orchestrator, AdvisorQuery
from src.portfolio import (
    annualized_covariance,
    from_agent_signals,
    min_variance_portfolio,
    risk_parity_weights,
)


def load_returns(project_root: Path) -> pd.DataFrame:
    data_dir = project_root / "data" / "processed"
    frames = {}
    for p in sorted(data_dir.glob("*_processed_v2.csv")):
        s = p.stem.replace("_processed_v2", "")
        df = pd.read_csv(p, parse_dates=["date"])
        if "Returns_1d" in df.columns:
            frames[s] = df.set_index("date")["Returns_1d"]
    return pd.DataFrame(frames).sort_index()


def show_portfolio(name, result, top_n=10):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"  method={result.method}   expected_return={result.expected_return*100:.2f}%/yr")
    print(f"  volatility={result.volatility*100:.2f}%/yr   sharpe={result.sharpe:.3f}")
    print(f"  shrinkage_delta={result.shrinkage_delta:.3f}   converged={result.converged}")
    print(f"{'=' * 60}")
    top = sorted(result.weights.items(), key=lambda kv: -kv[1])[:top_n]
    print(f"  Top {top_n} holdings:")
    for t, w in top:
        if w > 1e-4:
            print(f"    {t:12s}  {w*100:6.2f}%")
    n_held = sum(1 for w in result.weights.values() if w > 1e-4)
    print(f"  ({n_held} positions > 0.01%)")


def main():
    project_root = PROJECT_ROOT
    returns = load_returns(project_root)

    print(f"\nReturns matrix: {returns.shape[0]:,} days × {returns.shape[1]} stocks")
    print(f"Date range:     {returns.index.min().date()} → {returns.index.max().date()}")

    # Step 1: get ensemble signals from the orchestrator for all 30 stocks
    print("\n--- Step 1: Running multi-agent orchestrator on all 30 stocks ---")
    orch = Orchestrator(
        models_dir=project_root / "models/baseline",
        data_dir=project_root / "data/processed",
        sentiment_csv=project_root / "results/sentiment/news_scored.csv",
        rag_index_dir=project_root / "models/rag",
    )
    df_rank = orch.rank_all(profile="moderate", capital_bdt=100_000.0, top_k=30)
    signals = dict(zip(df_rank["stock"], df_rank["ensemble_signal"]))

    print("\n  Top 5 agent signals (most bullish):")
    for t, s in sorted(signals.items(), key=lambda kv: -kv[1])[:5]:
        print(f"    {t:12s}  {s:+.3f}")
    print("  Bottom 5 (most bearish):")
    for t, s in sorted(signals.items(), key=lambda kv: kv[1])[:5]:
        print(f"    {t:12s}  {s:+.3f}")

    # Step 2: covariance
    print("\n--- Step 2: Annualized covariance (Ledoit-Wolf) ---")
    cov, delta = annualized_covariance(returns, lookback=252, shrink=True)
    print(f"  Shrinkage intensity δ = {delta:.3f}")
    print(f"  Avg correlation: {((cov.values.sum() - np.trace(cov.values)) / (30*29)):.3f}")

    # Step 3: portfolios under each risk profile
    for profile in ["conservative", "moderate", "aggressive"]:
        # 3a: max-Sharpe using agent signals
        result_ms = from_agent_signals(
            ensemble_signals=signals,
            returns=returns,
            method="max_sharpe",
            profile=profile,
        )
        show_portfolio(
            f"Max-Sharpe (agent views) — {profile} profile",
            result_ms,
        )

        # 3b: risk-parity (no views needed)
        max_w = {"conservative": 0.05, "moderate": 0.10, "aggressive": 0.20}[profile]
        result_rp = risk_parity_weights(cov, max_weight=max_w)
        show_portfolio(
            f"Risk Parity (equal risk contribution) — {profile} profile",
            result_rp,
        )

    # Step 4: compare with min-variance
    result_mv = min_variance_portfolio(cov, max_weight=0.10)
    show_portfolio("Min-Variance (no views) — moderate profile", result_mv)

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    import numpy as np
    main()
