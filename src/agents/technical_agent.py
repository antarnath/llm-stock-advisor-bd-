"""
Technical Analyst Agent — produces a technical signal for a stock using the
trained baseline models and recent price data.

Produces an evidence packet:
  - signal ∈ [-1, +1]
  - confidence ∈ [0, 1]
  - top_features (SHAP-grounded)
  - details (predicted return, model agreement, recent volatility)
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base import Agent, AgentEvidence

warnings.filterwarnings("ignore")


class TechnicalAgent(Agent):
    """Specialist: price-only technical analysis using baseline models."""

    name = "TechnicalAnalyst"

    def __init__(self, models_dir: Path, data_dir: Path,
                 top_features: int = 5):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.top_features = top_features
        # Cache for SHAP outputs if computed elsewhere
        self._shap_cache = {}

    def analyze(self, stock: str, **kwargs) -> AgentEvidence:
        """Load the stock's best model, predict next-day return, compute signal."""
        model, features, model_path = self._load_model(stock)
        if model is None:
            return self._fallback(stock, "model not found")

        csv_path = self.data_dir / f"{stock}_processed_v2.csv"
        if not csv_path.exists():
            return self._fallback(stock, "processed CSV not found")

        df = pd.read_csv(csv_path, parse_dates=["date"])
        # Drop non-feature cols
        drop = [c for c in df.columns if c.startswith("Target_")]
        drop += [c for c in ("date", "code", "name", "sector") if c in df.columns]
        X = df.drop(columns=[c for c in drop if c in df.columns]).copy()

        # Use last 60 rows for context
        recent = X[features].tail(60).values

        # Predict next-day return (single-row prediction using last row)
        X_last = X[features].tail(1).values
        try:
            pred = float(model.predict(X_last)[0])
        except Exception:
            return self._fallback(stock, "predict failed")

        # Compute agreement: if model is XGBoost, use tree variance; else use
        # recent volatility as proxy
        recent_returns = df["Returns_1d"].tail(20).values if "Returns_1d" in df.columns else np.array([])
        volatility = float(np.nanstd(recent_returns)) if len(recent_returns) else 0.02

        # Recent direction (last 5 days)
        recent_dir = float(np.sign(recent_returns[-5:]).mean()) if len(recent_returns) >= 5 else 0.0

        # Build signal
        # pred is a return magnitude (typically ~0.01 for 1%)
        # Convert to [-1, +1] signal using tanh scaling
        signal = float(np.tanh(pred / max(volatility, 0.005)))

        # Confidence: model recency + low recent volatility = higher confidence
        conf = float(np.clip(0.6 - volatility * 8, 0.2, 0.9))

        # Headline
        if pred > 0:
            headline = f"Model predicts {stock} up {pred*100:.2f}% next day (model: {model_path.name.split('_')[0]})"
        else:
            headline = f"Model predicts {stock} down {abs(pred)*100:.2f}% next day (model: {model_path.name.split('_')[0]})"

        details = {
            "predicted_return": pred,
            "model_name": model_path.name,
            "recent_volatility_20d": volatility,
            "recent_5d_direction": recent_dir,
            "n_features_used": len(features),
        }

        # SHAP-top features: optional - we don't compute SHAP here to keep
        # this agent fast. The orchestrator can layer SHAP separately.
        top_feats = []

        sources = [
            f"models/baseline/{model_path.name}",
            f"data/processed/{stock}_processed_v2.csv",
        ]

        return AgentEvidence(
            agent_name=self.name,
            stock=stock,
            signal=signal,
            confidence=conf,
            headline=headline,
            details=details,
            top_features=top_feats,
            sources=sources,
        )

    def _load_model(self, stock: str):
        """Load best_v2 model bundle."""
        candidates = sorted(self.models_dir.glob(f"{stock}_best*.pkl"))
        if not candidates:
            return None, None, None
        candidates_v2 = [p for p in candidates if "_v2" in p.name]
        model_path = candidates_v2[0] if candidates_v2 else candidates[0]
        try:
            with open(model_path, "rb") as f:
                bundle = pickle.load(f)
        except Exception:
            return None, None, None
        if not isinstance(bundle, dict) or "model" not in bundle:
            return None, None, None
        return bundle["model"], bundle.get("features", []), model_path

    def _fallback(self, stock: str, reason: str) -> AgentEvidence:
        return AgentEvidence(
            agent_name=self.name,
            stock=stock,
            signal=0.0,
            confidence=0.1,
            headline=f"Technical analysis unavailable ({reason})",
            details={"error": reason},
            sources=[],
        )
