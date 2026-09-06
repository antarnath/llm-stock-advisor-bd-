"""
Portfolio Manager Agent — sizes a position in a single stock based on
signal strength, confidence, and the user's risk profile.

Produces an evidence packet:
  - signal ∈ [-1, +1] (suggested direction: +1 = full buy, -1 = full sell)
  - confidence
  - details: suggested_position_pct, suggested_position_bdt, cash_reserve

Allocation rules (rule-of-thumb; not formal MPT — that's a separate module):
  - Conservative: cap at 5% per stock; hold 60% cash reserve
  - Moderate:     cap at 10% per stock; hold 40% cash reserve
  - Aggressive:   cap at 20% per stock; hold 20% cash reserve

Position size = min(signal_strength × cap, cap)
"""

from __future__ import annotations

import numpy as np

from .base import Agent, AgentEvidence


PROFILES = {
    "conservative": {"max_per_stock": 0.05, "cash_reserve": 0.60},
    "moderate":     {"max_per_stock": 0.10, "cash_reserve": 0.40},
    "aggressive":   {"max_per_stock": 0.20, "cash_reserve": 0.20},
    "balanced":     {"max_per_stock": 0.10, "cash_reserve": 0.40},  # alias for moderate
}


class PortfolioAgent(Agent):
    """Specialist: position sizing given a user's risk profile."""

    name = "PortfolioManager"

    def __init__(self, default_profile: str = "moderate"):
        self.default_profile = default_profile.lower()

    def analyze(self, stock: str,
                combined_signal: float = 0.0,
                combined_confidence: float = 0.5,
                capital_bdt: float = 100_000.0,
                user_profile: str | None = None,
                **kwargs) -> AgentEvidence:
        """Decide a position size for the stock.

        combined_signal and combined_confidence come from other agents.
        """
        profile = (user_profile or self.default_profile).lower()
        rules = PROFILES.get(profile, PROFILES["moderate"])

        max_pct = rules["max_per_stock"]
        cash_reserve = rules["cash_reserve"]

        # Direction: positive signal → buy, negative → sell (or stay out)
        # Position size scales with |signal| × confidence
        position_pct = abs(combined_signal) * combined_confidence * max_pct
        position_pct = min(position_pct, max_pct)

        # Direction label
        if combined_signal > 0.2:
            direction = "BUY"
        elif combined_signal < -0.2:
            direction = "SELL"
        else:
            direction = "HOLD"
            position_pct = 0.0

        # Convert to BDT
        investable_capital = capital_bdt * (1 - cash_reserve)
        position_bdt = investable_capital * position_pct
        cash_bdt = capital_bdt * cash_reserve + investable_capital * (1 - position_pct)

        # Confidence for portfolio agent = how strong the consensus signal was
        confidence = float(combined_confidence)

        headline = (
            f"Portfolio suggestion: {direction} {stock}, "
            f"allocate {position_pct*100:.1f}% (৳{position_bdt:,.0f}) "
            f"of capital under {profile} profile"
        )

        return AgentEvidence(
            agent_name=self.name,
            stock=stock,
            signal=combined_signal,  # portfolio doesn't have own signal — passes through
            confidence=confidence,
            headline=headline,
            details={
                "direction": direction,
                "position_pct": position_pct,
                "position_bdt": position_bdt,
                "cash_bdt": cash_bdt,
                "user_profile": profile,
                "max_per_stock_pct": max_pct,
                "cash_reserve_pct": cash_reserve,
            },
            top_features=[],
            sources=[],
        )
