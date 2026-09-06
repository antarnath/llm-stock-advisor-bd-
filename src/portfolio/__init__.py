"""
Portfolio Optimization — Mean-Variance + Risk-Parity over the 30-stock DSE
universe.

Provides defensible, finance-grade position sizing that the multi-agent
Orchestrator can use instead of (or alongside) the simple rule-based
sizing in `src/agents/portfolio_agent.py`.

Components:
  - covariance: Ledoit-Wolf shrinkage estimator for Σ
  - mean_variance: Markowitz tangency / min-variance / max-Sharpe
  - risk_parity: equal risk contribution portfolio (no return forecasts needed)

Each component returns a `PortfolioResult` dataclass with weights, expected
return, volatility, Sharpe, and a human-readable explanation.

Constraints honored (per user risk profile):
  - Conservative: max 5% per stock, 60% cash reserve
  - Moderate:     max 10% per stock, 40% cash reserve
  - Aggressive:   max 20% per stock, 20% cash reserve
"""

from .covariance import (
    ledoit_wolf_shrinkage,
    annualized_covariance,
)
from .mean_variance import (
    PortfolioResult,
    mean_variance_optimize,
    max_sharpe_portfolio,
    min_variance_portfolio,
    from_agent_signals,
)
from .risk_parity import risk_parity_weights


__all__ = [
    "ledoit_wolf_shrinkage",
    "annualized_covariance",
    "PortfolioResult",
    "mean_variance_optimize",
    "max_sharpe_portfolio",
    "min_variance_portfolio",
    "from_agent_signals",
    "risk_parity_weights",
]
