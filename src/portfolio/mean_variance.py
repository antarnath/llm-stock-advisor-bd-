"""
Mean-Variance Optimization (Markowitz) over the DSE stock universe.

Implements three classical problems using scipy.optimize.minimize:
  1. min_variance_portfolio  — minimize w^T Σ w (no return forecast)
  2. max_sharpe_portfolio   — maximize (μᵀw − rf) / √(wᵀΣw)
  3. mean_variance_optimize — generic target-return / target-vol problem

All respect:
  - Long-only: w >= 0
  - Fully invested: Σ w = 1
  - Per-stock cap: w_i <= max_weight (from user risk profile)

The orchestrator calls these with the multi-agent ensemble signal as the
expected-return vector ("agent views" → portfolio optimization).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class PortfolioResult:
    """Portfolio construction output."""
    weights: dict[str, float]               # ticker → weight in [0, max_weight]
    expected_return: float                  # annualized μᵀw
    volatility: float                       # annualized √(wᵀΣw)
    sharpe: float                           # (μᵀw − rf) / vol, rf ≈ 0
    method: str                             # "min_variance" / "max_sharpe" / "target_return"
    shrinkage_delta: float = 0.0            # LW shrinkage intensity used
    n_iterations: int = 0                   # optimizer iterations
    converged: bool = True
    constraint_violations: dict = field(default_factory=dict)


def _portfolio_stats(w: np.ndarray,
                     mu: np.ndarray,
                     cov: np.ndarray,
                     rf: float = 0.0) -> tuple[float, float, float]:
    """Return (expected_return, volatility, sharpe) for a weight vector."""
    er = float(mu @ w)
    var = float(w @ cov @ w)
    vol = float(np.sqrt(max(var, 1e-12)))
    sharpe = (er - rf) / vol if vol > 1e-9 else 0.0
    return er, vol, sharpe


def _solve_with_constraints(mu: np.ndarray,
                            cov: np.ndarray,
                            objective: str,
                            target: Optional[float] = None,
                            max_weight: float = 0.10,
                            rf: float = 0.0
                            ) -> tuple[np.ndarray, dict]:
    """Solve a portfolio optimization problem with long-only + cap constraints.

    objective ∈ {"min_variance", "max_sharpe", "target_return"}
    """
    n = len(mu)
    # Initial weights: equal-weight
    w0 = np.full(n, 1.0 / n)

    # Constraints
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if objective == "target_return" and target is not None:
        constraints.append({
            "type": "eq",
            "fun": lambda w: mu @ w - target,
        })

    bounds = [(0.0, max_weight)] * n

    if objective == "min_variance":
        fun = lambda w: w @ cov @ w
    elif objective == "max_sharpe":
        # Maximize (μᵀw − rf)/√(wᵀΣw) ⟺ minimize negative sharpe
        # Use a quadratic surrogate: wᵀΣw − κ·(μᵀw − rf)  with adaptive κ
        # Cleaner: direct 1/vol minus penalty for negative return
        def fun(w):
            var = w @ cov @ w
            er = mu @ w
            if var <= 1e-12 or er <= 0:
                return 1e6
            # minimize -sharpe = -er/sqrt(var)
            return -(er - rf) / np.sqrt(var)
    elif objective == "target_return":
        # Minimize variance given target return — already constrained above
        fun = lambda w: w @ cov @ w
    else:
        raise ValueError(f"Unknown objective: {objective}")

    result = minimize(
        fun, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-9},
    )
    info = {"n_iter": int(result.nit), "converged": bool(result.success)}
    return result.x, info


def min_variance_portfolio(cov: pd.DataFrame,
                           max_weight: float = 0.10,
                           ) -> PortfolioResult:
    """Minimum-variance portfolio (no return forecast needed)."""
    tickers = list(cov.columns)
    cov_mat = cov.values
    n = len(tickers)
    w, info = _solve_with_constraints(
        mu=np.zeros(n), cov=cov_mat,
        objective="min_variance", max_weight=max_weight,
    )
    er, vol, sharpe = _portfolio_stats(w, np.zeros(n), cov_mat)
    return PortfolioResult(
        weights={t: float(w[i]) for i, t in enumerate(tickers)},
        expected_return=er, volatility=vol, sharpe=sharpe,
        method="min_variance",
        n_iterations=info["n_iter"], converged=info["converged"],
    )


def max_sharpe_portfolio(mu: pd.Series,
                         cov: pd.DataFrame,
                         rf: float = 0.0,
                         max_weight: float = 0.10,
                         ) -> PortfolioResult:
    """Tangency (max-Sharpe) portfolio. mu = expected returns (annualized)."""
    # Align
    tickers = list(cov.columns)
    mu_vec = np.array([mu.get(t, 0.0) for t in tickers])
    cov_mat = cov.values
    w, info = _solve_with_constraints(
        mu=mu_vec, cov=cov_mat,
        objective="max_sharpe", max_weight=max_weight, rf=rf,
    )
    er, vol, sharpe = _portfolio_stats(w, mu_vec, cov_mat, rf=rf)
    return PortfolioResult(
        weights={t: float(w[i]) for i, t in enumerate(tickers)},
        expected_return=er, volatility=vol, sharpe=sharpe,
        method="max_sharpe",
        n_iterations=info["n_iter"], converged=info["converged"],
    )


def mean_variance_optimize(mu: pd.Series,
                           cov: pd.DataFrame,
                           target_return: float,
                           max_weight: float = 0.10,
                           ) -> PortfolioResult:
    """Minimize variance subject to a target annualized return."""
    tickers = list(cov.columns)
    mu_vec = np.array([mu.get(t, 0.0) for t in tickers])
    cov_mat = cov.values
    w, info = _solve_with_constraints(
        mu=mu_vec, cov=cov_mat,
        objective="target_return", target=target_return,
        max_weight=max_weight,
    )
    er, vol, sharpe = _portfolio_stats(w, mu_vec, cov_mat)
    return PortfolioResult(
        weights={t: float(w[i]) for i, t in enumerate(tickers)},
        expected_return=er, volatility=vol, sharpe=sharpe,
        method="target_return",
        n_iterations=info["n_iter"], converged=info["converged"],
    )


def from_agent_signals(ensemble_signals: dict[str, float],
                       returns: pd.DataFrame,
                       method: str = "max_sharpe",
                       profile: str = "moderate",
                       lookback: int = 252,
                       rf: float = 0.0,
                       shrink: bool = True,
                       ) -> PortfolioResult:
    """End-to-end: agent signals → covariance → MPT → weights.

    Parameters
    ----------
    ensemble_signals : dict[str, float]
        ticker → ensemble signal in [-1, +1] from the multi-agent orchestrator.
    returns : pd.DataFrame
        Daily returns for all stocks, columns = tickers.
    method : str
        "max_sharpe" or "min_variance".
    profile : str
        User risk profile — determines max per-stock weight.
    lookback : int
        Days of history for covariance.
    rf : float
        Annualized risk-free rate (Bangladesh T-bill ≈ 0 for simplicity).
    shrink : bool
        Apply Ledoit-Wolf shrinkage.
    """
    from .covariance import annualized_covariance

    PROFILE_CAPS = {
        "conservative": 0.05,
        "moderate": 0.10,
        "aggressive": 0.20,
    }
    max_w = PROFILE_CAPS.get(profile.lower(), 0.10)

    cov, delta = annualized_covariance(returns, lookback=lookback, shrink=shrink)
    # Map agent signals to expected returns. Use sign-preserving tanh mapping,
    # then scale to a reasonable annualized return magnitude.
    mu = {}
    for t in cov.columns:
        s = ensemble_signals.get(t, 0.0)
        # Convert [-1, +1] signal to annualized return in [-15%, +25%]
        mu[t] = float(np.tanh(s) * 0.20)
    mu_s = pd.Series(mu)

    if method == "max_sharpe":
        result = max_sharpe_portfolio(mu_s, cov, rf=rf, max_weight=max_w)
    elif method == "min_variance":
        result = min_variance_portfolio(cov, max_weight=max_w)
    else:
        raise ValueError(f"Unknown method: {method}")

    result.shrinkage_delta = delta
    return result
