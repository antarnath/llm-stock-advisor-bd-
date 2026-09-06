"""
Covariance estimation for portfolio construction.

The standard sample covariance matrix is noisy and ill-conditioned for n≈30
stocks × T≈252 days. Ledoit-Wolf shrinkage pulls Σ̂ toward a structured
target (scaled identity), trading some bias for a large reduction in
estimation error.

References:
  Ledoit, O., & Wolf, M. (2004). "A well-conditioned estimator for
  large-dimensional covariance matrices." Journal of Multivariate
  Analysis, 88(2), 365-411.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ledoit_wolf_shrinkage(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage of sample covariance toward scaled identity.

    Σ_shrunk = δ · F + (1 − δ) · S

    where
      - S = sample covariance
      - F = structured target (here: μ_F · I, where μ_F = trace(S)/n)
      - δ = optimal shrinkage intensity (analytical formula)

    Parameters
    ----------
    returns : np.ndarray, shape (T, n)
        T observations of n asset returns.

    Returns
    -------
    shrunk : np.ndarray, shape (n, n)
        Shrunk covariance matrix (positive semi-definite by construction).
    delta : float
        Optimal shrinkage intensity in [0, 1].
    """
    T, n = returns.shape
    X = returns - returns.mean(axis=0, keepdims=True)
    S = (X.T @ X) / T                              # sample cov

    # Structured target: scaled identity
    mu = np.trace(S) / n
    F = mu * np.eye(n)

    # Optimal shrinkage intensity (Ledoit-Wolf 2004 formula)
    # delta* = sum_{i,j} Var(s_ij) / sum_{i,j} (s_ij - f_ij)^2
    s_ij = (X[:, :, None] * X[:, None, :]).mean(axis=0)  # outer products averaged
    # Simpler formulation using double sums
    var_s = 0.0
    for i in range(n):
        for j in range(n):
            x_ix_j = X[:, i] * X[:, j]
            var_s += np.var(x_ix_j, ddof=1) / T
    diff_sq = np.sum((S - F) ** 2)
    delta = var_s / diff_sq if diff_sq > 0 else 1.0
    delta = float(np.clip(delta, 0.0, 1.0))

    shrunk = delta * F + (1.0 - delta) * S

    # Ensure symmetric (numerical safety)
    shrunk = 0.5 * (shrunk + shrunk.T)
    return shrunk, delta


def annualized_covariance(returns: pd.DataFrame,
                          lookback: int = 252,
                          shrink: bool = True
                          ) -> tuple[pd.DataFrame, float]:
    """Compute annualized covariance matrix over a rolling window.

    Parameters
    ----------
    returns : pd.DataFrame, shape (T, n)
        Columns = stock codes, values = daily simple returns.
    lookback : int
        Number of trading days to use (default 252 = ~1 year).
    shrink : bool
        If True, apply Ledoit-Wolf shrinkage. Recommended for n > 10.

    Returns
    -------
    cov_annual : pd.DataFrame, shape (n, n)
        Annualized covariance matrix (× 252).
    delta : float
        Shrinkage intensity used (0 if shrink=False).
    """
    # Use the last `lookback` complete observations
    R = returns.dropna(how="any").tail(lookback).values
    if shrink:
        cov_daily, delta = ledoit_wolf_shrinkage(R)
    else:
        cov_daily = np.cov(R, rowvar=False, ddof=1)
        delta = 0.0
    cov_annual = pd.DataFrame(cov_daily * 252.0, index=returns.columns,
                              columns=returns.columns)
    return cov_annual, delta
