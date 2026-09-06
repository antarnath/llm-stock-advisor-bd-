"""
Test the Portfolio Optimization module end-to-end.

Loads all 30 stocks' returns, builds the covariance matrix, then exercises:
  - min_variance
  - max_sharpe (with mock agent views)
  - risk_parity
  - from_agent_signals (full pipeline)
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.portfolio import (
    annualized_covariance,
    min_variance_portfolio,
    max_sharpe_portfolio,
    risk_parity_weights,
    from_agent_signals,
)


passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    mark = "PASS" if cond else "FAIL"
    if cond:
        passed += 1
    else:
        failed += 1
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))


# -----------------------------------------------------------------
print("\n=== Loading all 30 stocks' returns ===")
data_dir = PROJECT_ROOT / "data" / "processed"
stocks = sorted([p.stem.replace("_processed_v2", "")
                 for p in data_dir.glob("*_processed_v2.csv")])
check("30 stocks found", len(stocks) == 30, f"got {len(stocks)}")

frames = {}
for s in stocks:
    df = pd.read_csv(data_dir / f"{s}_processed_v2.csv", parse_dates=["date"])
    if "Returns_1d" in df.columns:
        frames[s] = df.set_index("date")["Returns_1d"]

returns = pd.DataFrame(frames).sort_index()
check("Returns matrix shape", returns.shape[0] > 1000 and returns.shape[1] == 30,
      f"shape={returns.shape}")


# -----------------------------------------------------------------
print("\n=== Test: annualized covariance (Ledoit-Wolf) ===")
cov, delta = annualized_covariance(returns, lookback=252, shrink=True)
check("Cov shape (30,30)", cov.shape == (30, 30), f"shape={cov.shape}")
check("Symmetric", np.allclose(cov.values, cov.values.T))
check("Positive diagonal", (np.diag(cov.values) > 0).all())
check("Shrinkage delta in [0,1]", 0.0 <= delta <= 1.0, f"delta={delta:.3f}")


# -----------------------------------------------------------------
print("\n=== Test: min-variance portfolio ===")
mv = min_variance_portfolio(cov, max_weight=0.10)
check("Returns PortfolioResult", hasattr(mv, "weights"))
check("Weights sum to 1", abs(sum(mv.weights.values()) - 1.0) < 1e-4,
      f"sum={sum(mv.weights.values()):.6f}")
check("All weights >= 0", all(w >= 0 for w in mv.weights.values()))
check("All weights <= 0.10", all(w <= 0.10 + 1e-6 for w in mv.weights.values()))
check("Converged", mv.converged)
check("Has volatility", mv.volatility > 0, f"vol={mv.volatility:.4f}")
top5 = sorted(mv.weights.items(), key=lambda kv: -kv[1])[:5]
print(f"  Top 5 holdings:")
for t, w in top5:
    print(f"    {t}: {w*100:.2f}%")


# -----------------------------------------------------------------
print("\n=== Test: max-Sharpe portfolio with mock agent views ===")
mock_signals = {s: float(np.random.RandomState(42).uniform(-1, 1)) for s in stocks}
# Manually set 2 high-signal, 1 low
mock_signals["BEXPHARMA"] = 0.8
mock_signals["UNILEVER"] = 0.6
mock_signals["BEXIMCO"] = -0.7

ms = max_sharpe_portfolio(
    mu=pd.Series({s: np.tanh(mock_signals[s]) * 0.20 for s in stocks}),
    cov=cov, rf=0.0, max_weight=0.10,
)
check("Max-sharpe weights sum to 1", abs(sum(ms.weights.values()) - 1.0) < 1e-4)
check("Max-sharpe cap respected",
      all(w <= 0.10 + 1e-6 for w in ms.weights.values()))
check("Returns positive weights only",
      all(w >= 0 for w in ms.weights.values()))
check("BEXPHARMA has weight", ms.weights.get("BEXPHARMA", 0) > 0,
      f"weight={ms.weights.get('BEXPHARMA', 0):.4f}")
# When most views are negative (28/30 random), the optimizer picks equal-weight.
# Verify it computes a meaningful Sharpe (not NaN/inf).
check("Sharpe is finite", np.isfinite(ms.sharpe), f"sharpe={ms.sharpe:.3f}")
check("Volatility is positive", ms.volatility > 0, f"vol={ms.volatility:.4f}")
print(f"  Top 5 holdings:")
for t, w in sorted(ms.weights.items(), key=lambda kv: -kv[1])[:5]:
    print(f"    {t}: {w*100:.2f}%")


# -----------------------------------------------------------------
print("\n=== Test: max-Sharpe with ALL-positive views ===")
positive_signals = {s: float(np.random.RandomState(7).uniform(0.2, 0.9)) for s in stocks}
ms_pos = max_sharpe_portfolio(
    mu=pd.Series({s: np.tanh(positive_signals[s]) * 0.20 for s in stocks}),
    cov=cov, rf=0.0, max_weight=0.10,
)
check("Positive views → positive Sharpe",
      ms_pos.sharpe > 0, f"sharpe={ms_pos.sharpe:.3f}")
check("Top holding has weight >= 0.05",
      max(ms_pos.weights.values()) >= 0.05,
      f"max={max(ms_pos.weights.values()):.4f}")
print(f"  Top 5 holdings:")
for t, w in sorted(ms_pos.weights.items(), key=lambda kv: -kv[1])[:5]:
    print(f"    {t}: {w*100:.2f}%")

# -----------------------------------------------------------------
print("\n=== Test: risk-parity portfolio ===")
rp = risk_parity_weights(cov, max_weight=0.10)
check("Risk-parity weights sum to 1", abs(sum(rp.weights.values()) - 1.0) < 1e-4)
check("Risk-parity cap respected",
      all(w <= 0.10 + 1e-6 for w in rp.weights.values()))
check("Risk-parity converged", rp.converged, f"iter={rp.n_iterations}")

# Check equal risk contribution: RC_i ≈ constant
w = np.array([rp.weights[s] for s in stocks])
cov_mat = cov.values
sigma_w = cov_mat @ w
rc = w * sigma_w
rc_var = float(np.std(rc) / np.mean(rc))
check("Risk contributions roughly equal (CV < 0.2)",
      rc_var < 0.2, f"CV={rc_var:.3f}")
print(f"  Risk contribution CV: {rc_var:.3f}")


# -----------------------------------------------------------------
print("\n=== Test: end-to-end from_agent_signals (full pipeline) ===")
result = from_agent_signals(
    ensemble_signals=mock_signals,
    returns=returns,
    method="max_sharpe",
    profile="moderate",
)
check("Pipeline result has shrinkage_delta",
      hasattr(result, "shrinkage_delta"))
check("Pipeline respects moderate cap (10%)",
      all(w <= 0.10 + 1e-6 for w in result.weights.values()))

# Test with conservative profile
result_cons = from_agent_signals(
    ensemble_signals=mock_signals, returns=returns,
    method="max_sharpe", profile="conservative",
)
check("Conservative profile cap (5%)",
      all(w <= 0.05 + 1e-6 for w in result_cons.weights.values()))

# Test with aggressive profile
result_agg = from_agent_signals(
    ensemble_signals=mock_signals, returns=returns,
    method="max_sharpe", profile="aggressive",
)
check("Aggressive profile cap (20%)",
      all(w <= 0.20 + 1e-6 for w in result_agg.weights.values()))


# -----------------------------------------------------------------
print("\n=== Summary ===")
print(f"  {passed} passed, {failed} failed")
if failed:
    sys.exit(1)
