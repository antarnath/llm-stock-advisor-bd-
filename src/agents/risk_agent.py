"""
Risk Analyst Agent — computes a risk-adjusted signal for a stock using
historical volatility, drawdown, and Sharpe-like metrics.

Produces an evidence packet:
  - signal ∈ [-1, +1] (negative = high risk, positive = favorable risk-adjusted profile)
  - confidence based on data availability
  - details: volatility, max drawdown, downside frequency
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import Agent, AgentEvidence


class RiskAgent(Agent):
    """Specialist: stock-specific risk profile from historical returns."""

    name = "RiskAnalyst"

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def analyze(self, stock: str, **kwargs) -> AgentEvidence:
        csv_path = self.data_dir / f"{stock}_processed_v2.csv"
        if not csv_path.exists():
            return self._fallback(stock, "data not found")

        df = pd.read_csv(csv_path, parse_dates=["date"])
        if "Returns_1d" not in df.columns or len(df) < 100:
            return self._fallback(stock, "insufficient history")

        rets = df["Returns_1d"].dropna().values
        # Use last 252 trading days (~1 year)
        rets = rets[-252:] if len(rets) > 252 else rets

        vol = float(np.std(rets))
        mean = float(np.mean(rets))
        downside = float(np.std(rets[rets < 0])) if (rets < 0).any() else vol

        # Max drawdown
        cum = np.cumprod(1 + rets)
        peak = np.maximum.accumulate(cum)
        drawdown = (cum - peak) / peak
        max_dd = float(np.min(drawdown))

        # Sharpe-like (annualized; assume 252 trading days, 0 risk-free)
        sharpe = float((mean / vol) * np.sqrt(252)) if vol > 1e-9 else 0.0

        # Build signal: positive Sharpe → positive signal; high vol → negative signal
        # We combine into one score
        sharpe_signal = np.clip(sharpe / 2.0, -1.0, 1.0)        # Sharpe ∈ [-2, +2] → [-1, +1]
        vol_signal = -np.clip((vol - 0.02) / 0.02, -1.0, 1.0)  # vol>2% → negative
        drawdown_signal = np.clip(max_dd / -0.30, -1.0, 1.0)   # -30% DD → -1, 0% → 0

        # Combine
        signal = float(0.5 * sharpe_signal + 0.2 * vol_signal + 0.3 * drawdown_signal)
        signal = float(np.clip(signal, -1.0, 1.0))

        # Confidence: more data = more confidence
        confidence = float(np.clip(len(rets) / 500, 0.3, 0.95))

        if signal > 0.2:
            verdict = "favorable risk-adjusted profile"
        elif signal < -0.2:
            verdict = "elevated risk — caution"
        else:
            verdict = "neutral risk profile"

        headline = (
            f"Risk profile for {stock}: {verdict} "
            f"(vol={vol*100:.2f}%/day, Sharpe={sharpe:.2f}, max DD={max_dd*100:.1f}%)"
        )

        return AgentEvidence(
            agent_name=self.name,
            stock=stock,
            signal=signal,
            confidence=confidence,
            headline=headline,
            details={
                "annualized_volatility": vol * np.sqrt(252),
                "daily_volatility": vol,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "mean_return_daily": mean,
                "downside_volatility": downside,
            },
            top_features=[
                ("annualized_volatility", vol * np.sqrt(252)),
                ("max_drawdown", max_dd),
                ("sharpe", sharpe),
            ],
            sources=[str(csv_path)],
        )

    def _fallback(self, stock: str, reason: str) -> AgentEvidence:
        return AgentEvidence(
            agent_name=self.name,
            stock=stock,
            signal=0.0,
            confidence=0.1,
            headline=f"Risk analysis unavailable ({reason})",
            details={"error": reason},
            sources=[],
        )
