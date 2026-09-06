"""POST /portfolio/optimize — Mean-Variance / Risk-Parity over agent signals.

Reads the latest daily returns for all 30 stocks from
data/processed/*_processed_v2.csv, computes a covariance matrix with
Ledoit-Wolf shrinkage, then optimizes with the requested method using
a momentum-based view as the expected-return proxy.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.models import (
    PortfolioAllocation,
    PortfolioOptimizeRequest,
    PortfolioResult,
)
from src.portfolio.covariance import annualized_covariance
from src.portfolio.mean_variance import (
    max_sharpe_portfolio,
    min_variance_portfolio,
)
from src.portfolio.risk_parity import risk_parity_weights
from src.utils.config import PROCESSED_DATA_DIR, TOP_30_DSE_STOCKS

logger = logging.getLogger(__name__)
router = APIRouter(tags=["portfolio"])

PROFILE_CAP = {
    "conservative": 0.05,
    "moderate": 0.10,
    "aggressive": 0.20,
}


def _load_returns(lookback_days: int = 252) -> pd.DataFrame:
    """Return a (T x N) DataFrame of daily returns for the 30 stocks."""
    series = {}
    for code in TOP_30_DSE_STOCKS:
        path = PROCESSED_DATA_DIR / f"{code}_processed_v2.csv"
        if not path.exists():
            path = PROCESSED_DATA_DIR / f"{code}_processed.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "close" not in df.columns and "Close" not in df.columns:
            continue
        col = "close" if "close" in df.columns else "Close"
        df = df.dropna(subset=[col]).tail(lookback_days)
        if len(df) < 30:
            continue
        ret = df[col].pct_change().dropna()
        series[code] = ret
    if not series:
        return pd.DataFrame()
    out = pd.DataFrame(series)
    return out.dropna(how="all").fillna(0.0)


def _agent_views(returns: pd.DataFrame) -> pd.Series:
    """Per-stock annualized expected return from a 60-day momentum proxy.

    Placeholder for the real multi-agent ensemble views.
    """
    if returns.empty:
        return pd.Series(dtype=float)
    last = returns.tail(60)
    if last.empty:
        last = returns
    mu = (1.0 + last.mean()) ** 252 - 1.0
    return mu


def _format_allocations(tickers: list[str], weights: dict[str, float],
                        capital: float, cap: float
                        ) -> list[PortfolioAllocation]:
    """Convert a weights dict → list of allocations, enforcing cap + cleanup."""
    pairs = [(t, min(float(w), cap)) for t, w in weights.items()]
    total = sum(w for _, w in pairs)
    if total > 0:
        pairs = [(t, w / total) for t, w in pairs]
    pairs.sort(key=lambda x: -x[1])
    out = []
    for t, w in pairs[:15]:
        if w < 1e-4:
            continue
        out.append(PortfolioAllocation(
            ticker=t,
            weight=round(float(w), 6),
            amount_bdt=round(float(w) * capital, 2),
        ))
    return out


@router.post("/portfolio/optimize", response_model=PortfolioResult)
async def optimize(req: PortfolioOptimizeRequest) -> PortfolioResult:
    returns = _load_returns(252)
    if returns.shape[1] < 5:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough stocks with returns: {returns.shape[1]}",
        )

    cap = PROFILE_CAP[req.profile]
    rf = req.risk_free_rate
    cov_mat, delta = annualized_covariance(returns, lookback=252, shrink=True)
    views = _agent_views(returns)

    if req.method == "risk_parity":
        result = risk_parity_weights(cov_mat, max_weight=cap)
        return PortfolioResult(
            profile=req.profile,
            method=req.method,
            capital_bdt=req.capital_bdt,
            expected_return=float(views.mean()) if not views.empty else 0.0,
            volatility=float(result.volatility),
            sharpe_ratio=float(result.sharpe),
            shrinkage_delta=float(delta),
            allocations=_format_allocations(
                list(result.weights.keys()), result.weights, req.capital_bdt, cap),
        )

    if req.method == "min_variance":
        result = min_variance_portfolio(cov_mat, max_weight=cap)
        return PortfolioResult(
            profile=req.profile,
            method=req.method,
            capital_bdt=req.capital_bdt,
            expected_return=float(views.mean()) if not views.empty else 0.0,
            volatility=float(result.volatility),
            sharpe_ratio=float(result.sharpe),
            shrinkage_delta=float(delta),
            allocations=_format_allocations(
                list(result.weights.keys()), result.weights, req.capital_bdt, cap),
        )

    # max_sharpe (default)
    mu = views.reindex(cov_mat.columns).fillna(0.0)
    result = max_sharpe_portfolio(mu=mu, cov=cov_mat, rf=rf, max_weight=cap)
    return PortfolioResult(
        profile=req.profile,
        method=req.method,
        capital_bdt=req.capital_bdt,
        expected_return=float(result.expected_return),
        volatility=float(result.volatility),
        sharpe_ratio=float(result.sharpe),
        shrinkage_delta=float(delta),
        allocations=_format_allocations(
            list(result.weights.keys()), result.weights, req.capital_bdt, cap),
    )
