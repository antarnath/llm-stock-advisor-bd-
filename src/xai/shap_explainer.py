"""
SHAP Explainer.

Provides unified SHAP-based feature importance for all model types:

- TreeSHAP for XGBoost / LightGBM / RandomForest (baseline models)
- KernelSHAP for deep learning models (LSTM / GRU / CNN-LSTM / Multimodal LSTM)
- Per-stock + per-feature SHAP values, aggregated into a global importance matrix

Usage:
    from src.xai.shap_explainer import ShapExplainer

    explainer = ShapExplainer(model, X_background, X_test, feature_names)
    shap_values = explainer.explain(model_type="tree")
    importance_df = explainer.global_importance()
    local_df = explainer.local_importance(idx=0)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Literal

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import shap  # noqa: E402


ModelType = Literal["tree", "kernel"]


class ShapExplainer:
    """Unified SHAP explainer for tree + deep models."""

    def __init__(
        self,
        model,
        X_background: np.ndarray,
        X_test: np.ndarray,
        feature_names: list[str],
        model_type: ModelType | None = None,
        n_background: int = 100,
        n_test: int = 200,
        seed: int = 42,
    ):
        """
        Args:
            model: Trained model with .predict() (and optionally feature_names_in_)
            X_background: Background dataset for SHAP (subsampled internally)
            X_test: Test instances to explain
            feature_names: list of feature names matching X columns
            model_type: "tree" for tree models, "kernel" for everything else.
                        Auto-detected if None.
            n_background: Max background samples (subsampled)
            n_test: Max test samples (subsampled)
        """
        self.model = model
        self.feature_names = list(feature_names)
        self.model_type = model_type or self._auto_detect(model)
        self.seed = seed

        # Subsample background and test
        rng = np.random.default_rng(seed)
        if len(X_background) > n_background:
            idx = rng.choice(len(X_background), size=n_background, replace=False)
            self.X_background = X_background[idx]
        else:
            self.X_background = X_background
        if len(X_test) > n_test:
            idx = rng.choice(len(X_test), size=n_test, replace=False)
            self.X_test = X_test[idx]
        else:
            self.X_test = X_test

        self._explainer = None
        self._shap_values = None

    @staticmethod
    def _auto_detect(model) -> ModelType:
        """Auto-detect whether to use TreeSHAP or KernelSHAP."""
        cls_name = type(model).__name__
        tree_classes = (
            "XGBRegressor", "XGBClassifier",
            "LGBMRegressor", "LGBMClassifier",
            "RandomForestRegressor", "RandomForestClassifier",
            "GradientBoostingRegressor", "GradientBoostingClassifier",
        )
        if any(t in cls_name for t in tree_classes):
            return "tree"
        return "kernel"

    def _build_explainer(self):
        """Lazy-build the SHAP explainer."""
        if self._explainer is not None:
            return self._explainer
        cls_name = type(self.model).__name__
        if self.model_type == "tree":
            self._explainer = shap.TreeExplainer(self.model)
        elif cls_name == "LinearRegression" and self.X_background is not None:
            # Exact linear SHAP — much faster and more accurate than KernelSHAP
            self._explainer = shap.LinearExplainer(
                self.model, self.X_background,
                feature_perturbation="correlation_dependent",
            )
        else:
            self._explainer = shap.KernelExplainer(
                self.model.predict,
                self.X_background,
                link="identity",
                silent=True,
            )
        return self._explainer

    def explain(self) -> np.ndarray:
        """Compute SHAP values for the test set.

        Returns:
            shap_values: shape (n_test, n_features)
        """
        if self._shap_values is not None:
            return self._shap_values
        explainer = self._build_explainer()
        cls_name = type(explainer).__name__
        if cls_name == "TreeExplainer":
            self._shap_values = explainer.shap_values(self.X_test)
        elif cls_name == "LinearExplainer":
            self._shap_values = explainer.shap_values(self.X_test)
        else:
            # KernelSHAP: batched to avoid huge memory
            n = len(self.X_test)
            batch = min(50, n)
            shap_chunks = []
            for i in range(0, n, batch):
                chunk = self.X_test[i : i + batch]
                vals = explainer.shap_values(chunk, nsamples=100, silent=True)
                # KernelSHAP for regression returns shape (n, features)
                if isinstance(vals, list):
                    vals = vals[0]
                shap_chunks.append(np.asarray(vals))
            self._shap_values = np.concatenate(shap_chunks, axis=0)
        # Normalize to (n, features)
        if isinstance(self._shap_values, list):
            self._shap_values = np.asarray(self._shap_values[0])
        self._shap_values = np.asarray(self._shap_values)
        # Some explainers return (n, features) or (features,) for single instance
        if self._shap_values.ndim == 1:
            self._shap_values = self._shap_values.reshape(1, -1)
        return self._shap_values

    def global_importance(self) -> pd.DataFrame:
        """Compute global feature importance: mean |SHAP| per feature.

        Returns DataFrame with columns: feature, mean_abs_shap, mean_shap
        """
        shap_values = self.explain()
        mean_abs = np.abs(shap_values).mean(axis=0)
        mean_signed = shap_values.mean(axis=0)
        df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs,
            "mean_shap": mean_signed,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        return df

    def local_importance(self, idx: int, top_k: int = 10) -> pd.DataFrame:
        """Compute local feature importance for a single test instance.

        Returns DataFrame with columns: feature, shap_value, feature_value
        """
        shap_values = self.explain()
        if idx >= len(shap_values):
            raise IndexError(f"idx {idx} >= {len(shap_values)}")
        row = shap_values[idx]
        df = pd.DataFrame({
            "feature": self.feature_names,
            "shap_value": row,
            "feature_value": self.X_test[idx],
        }).sort_values("shap_value", key=np.abs, ascending=False).reset_index(drop=True)
        if top_k:
            df = df.head(top_k)
        return df

    def base_value(self) -> float:
        """Return the SHAP expected/base value."""
        explainer = self._build_explainer()
        if hasattr(explainer, "expected_value"):
            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                ev = np.asarray(ev).flatten()
                if len(ev) > 0:
                    return float(ev[0])
                return 0.0
            return float(ev)
        return 0.0


def explain_xgboost_stock(
    model_path: Path,
    csv_path: Path,
    target_col: str = "Target_Return_1d",
    n_test: int = 200,
    save_csv: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end: load XGBoost pickle, prepare data, run TreeSHAP, return CSVs.

    Args:
        model_path: Path to *_best_v2.pkl (XGBoost)
        csv_path: Path to {STOCK}_processed_v2.csv
        target_col: Target column to drop from features
        n_test: Number of test instances to explain
        save_csv: If set, write global + local CSVs here

    Returns:
        (global_df, local_df)
    """
    import pickle

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    features = bundle["features"]

    df = pd.read_csv(csv_path, parse_dates=["date"])
    drop_cols = [c for c in df.columns if c in {"date", "code", "name", "sector", target_col}
                 or c.startswith("Target_")]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    X = X[features]

    # Time-based test split (last 20%, matches baseline training protocol)
    test_size = 0.20
    n = len(X)
    test_end = n
    val_end = int(n * (1.0 - test_size))
    X_test = X.iloc[val_end:test_end].values

    if len(X_test) > n_test:
        idx = np.random.default_rng(42).choice(len(X_test), size=n_test, replace=False)
        X_test = X_test[idx]

    X_background = X.iloc[:val_end].sample(
        n=min(100, val_end), random_state=42,
    ).values

    explainer = ShapExplainer(
        model=model,
        X_background=X_background,
        X_test=X_test,
        feature_names=features,
        model_type=None,  # auto-detect (tree vs linear vs kernel)
        n_test=len(X_test),
    )
    global_df = explainer.global_importance()
    local_df = explainer.local_importance(idx=0, top_k=15)

    if save_csv is not None:
        save_csv.mkdir(parents=True, exist_ok=True)
        global_df.to_csv(save_csv / "global_importance.csv", index=False)
        local_df.to_csv(save_csv / "local_importance.csv", index=False)

    return global_df, local_df


__all__ = ["ShapExplainer", "explain_xgboost_stock", "ModelType"]
