"""
LIME Explainer.

Local Interpretable Model-agnostic Explanations. Works for any model with
a .predict() method. For time-series LSTM models, the 60-day window is
flattened to a 1D feature vector of (seq_len * n_features) per instance.

Usage:
    from src.xai.lime_explainer import LimeExplainer

    explainer = LimeExplainer(model, X_train, feature_names)
    exp = explainer.explain_instance(x_instance, num_features=10)
    weights = explainer.explain_as_dict(x_instance, num_features=10)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Callable

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import lime  # noqa: E402
import lime.lime_tabular  # noqa: E402


class LimeExplainer:
    """LIME Tabular explainer for tabular regression models."""

    def __init__(
        self,
        model,
        X_train: np.ndarray,
        feature_names: list[str],
        mode: str = "regression",
        discretize_continuous: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            model: Trained model with .predict()
            X_train: Training data for LIME background (subsampled internally)
            feature_names: list of feature names matching X columns
            mode: "regression" (default) or "classification"
            discretize_continuous: Whether to discretize continuous features
        """
        self.model = model
        self.feature_names = list(feature_names)
        self.predict_fn = model.predict
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.asarray(X_train),
            feature_names=self.feature_names,
            mode=mode,
            discretize_continuous=discretize_continuous,
            random_state=seed,
            verbose=False,
        )

    def explain_instance(
        self,
        x_instance: np.ndarray,
        num_features: int = 10,
        num_samples: int = 1000,
    ) -> lime.lime_tabular.Explanation:
        """Return a LIME Explanation object for one instance."""
        return self.explainer.explain_instance(
            data_row=np.asarray(x_instance).flatten(),
            predict_fn=self.predict_fn,
            num_features=num_features,
            num_samples=num_samples,
        )

    def explain_as_dict(
        self,
        x_instance: np.ndarray,
        num_features: int = 10,
        num_samples: int = 1000,
    ) -> list[tuple[str, float]]:
        """Return a list of (feature_description, weight) sorted by |weight|."""
        exp = self.explain_instance(x_instance, num_features, num_samples)
        return list(exp.as_list())


class LimeTimeSeriesExplainer:
    """LIME for time-series models (LSTM/GRU/CNN-LSTM).

    Flattens the (seq_len, n_features) window to (seq_len * n_features) for
    LIME, and reshapes predictions back. The returned feature names are
    prefixed with `t-{lag}|` so you can tell which timestep each weight
    belongs to.
    """

    def __init__(
        self,
        model,
        X_train: np.ndarray,           # shape (n, seq_len, n_features)
        feature_names: list[str],
        predict_fn: Callable | None = None,
        seed: int = 42,
    ):
        """
        Args:
            model: Trained model with .predict()
            X_train: Training sequences (n, seq_len, n_features)
            feature_names: list of feature names
            predict_fn: Optional custom predict (e.g., wrapping a torch model)
        """
        self.model = model
        self.feature_names = list(feature_names)
        seq_len = X_train.shape[1]
        n_feat = X_train.shape[2]
        self.seq_len = seq_len
        self.n_feat = n_feat
        # Build flat feature names like "t-59|RSI_14", "t-58|RSI_14", ...
        self.flat_names = [
            f"t-{seq_len - 1 - i}|{name}"
            for i in range(seq_len)
            for name in feature_names
        ]
        # Flatten training data
        X_flat = X_train.reshape(X_train.shape[0], -1)
        self.predict_fn = predict_fn or (lambda x: model.predict(x))
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_flat,
            feature_names=self.flat_names,
            mode="regression",
            discretize_continuous=False,  # too many features; keep continuous
            random_state=seed,
            verbose=False,
        )

    def explain_as_dict(
        self,
        x_instance: np.ndarray,
        num_features: int = 15,
        num_samples: int = 500,
    ) -> list[tuple[str, float]]:
        """Explain one (seq_len, n_features) window. Returns flat-name weights."""
        x_flat = np.asarray(x_instance).reshape(-1)
        exp = self.explainer.explain_instance(
            data_row=x_flat,
            predict_fn=lambda z: np.asarray(self.predict_fn(
                z.reshape(-1, self.seq_len, self.n_feat)
            )).flatten(),
            num_features=num_features,
            num_samples=num_samples,
        )
        return list(exp.as_list())

    def aggregate_by_feature(
        self,
        x_instance: np.ndarray,
        num_features: int = 30,
        num_samples: int = 500,
    ) -> pd.DataFrame:
        """Aggregate LIME weights across all timesteps per base feature.

        Returns DataFrame with columns: feature, total_abs_weight, signed_sum
        sorted by total_abs_weight desc.
        """
        items = self.explain_as_dict(x_instance, num_features, num_samples)
        rows = []
        for desc, w in items:
            # desc is like "t-3|RSI_14" or "RSI_14 < 0.5"
            base = desc.split("|", 1)[-1].split(" ")[0] if "|" in desc else desc.split(" ")[0]
            rows.append({"feature": base, "weight": w})
        if not rows:
            return pd.DataFrame(columns=["feature", "total_abs_weight", "signed_sum"])
        df = pd.DataFrame(rows)
        agg = df.groupby("feature")["weight"].agg(
            total_abs_weight=lambda s: float(np.abs(s).sum()),
            signed_sum="sum",
        ).reset_index().sort_values("total_abs_weight", ascending=False)
        return agg.reset_index(drop=True)


__all__ = ["LimeExplainer", "LimeTimeSeriesExplainer"]
