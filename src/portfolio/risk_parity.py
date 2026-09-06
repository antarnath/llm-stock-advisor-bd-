"""
Risk Parity portfolio construction.

Goal: each asset contributes equally to total portfolio variance. No return
forecasts required — works directly off the covariance matrix.

Algorithm: iterative proportional scaling (Spinu 2013). For long-only with
per-stock caps, project onto the constraint set after each update.

Reference:
  Spinu, F. (2013). "An Algorithm for Portfolio Risk Parity."
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .mean_variance import PortfolioResult


def risk_parity_weights(cov: pd.DataFrame,
                        max_weight: float = 0.10,
                        max_iter: int = 200,
                        tol: float = 1e-6,
                        ) -> PortfolioResult:
    """Compute risk-parity weights using iterative proportional scaling.

    At convergence, each asset's marginal risk contribution
        RC_i = w_i · (Σw)_i
    is equal across all i. We solve
        w_i ∝ 1 / sqrt((Σw)_i)
    iteratively, then project onto the constraint set {w ≥ 0, Σw = 1, w ≤ cap}.
    """
    tickers = list(cov.columns)
    n = len(tickers)
    cov_mat = cov.values

    # Initial equal-weight
    w = np.full(n, 1.0 / n)

    for it in range(max_iter):
        # Marginal risk contribution (without 1/vol normalization)
        sigma_w = cov_mat @ w
        rc = w * sigma_w                  # RC_i = w_i * (Σw)_i
        # Target: equal RC across assets
        # Update: w_i ∝ 1 / sqrt((Σw)_i)
        # Avoid division by zero
        denom = np.sqrt(np.maximum(sigma_w, 1e-12))
        w_new = 1.0 / denom
        # Normalize then project
        w_new = w_new / w_new.sum()
        w_new = np.minimum(w_new, max_weight)
        w_new = w_new / w_new.sum()       # renormalize after cap
        # Check convergence
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            return _make_result(w, cov_mat, tickers, it, converged=True)
        w = w_new

    return _make_result(w, cov_mat, tickers, max_iter, converged=False)


def _make_result(w: np.ndarray, cov_mat: np.ndarray, tickers: list[str],
                 iterations: int, converged: bool) -> PortfolioResult:
    er = 0.0   # no return forecast
    var = float(w @ cov_mat @ w)
    vol = float(np.sqrt(max(var, 1e-12)))
    return PortfolioResult(
        weights={t: float(w[i]) for i, t in enumerate(tickers)},
        expected_return=er, volatility=vol, sharpe=0.0,
        method="risk_parity",
        n_iterations=int(iterations), converged=bool(converged),
    )
