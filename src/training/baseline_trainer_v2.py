"""
Baseline ML Models Training - Phase 3 (v2, LEAK-FREE)
Trains 4 baseline models on cleaned features, predicting next-day returns.

Models trained:
- Linear Regression
- Random Forest
- XGBoost
- LightGBM  (NEW — already in requirements.txt)

What changed vs v1:
- Features exclude OHLCV raw columns AND any column that directly encodes
  the close price on day t (target is Target_Return_1d = close[t+1]/close[t]-1).
- Target is Target_Return_1d (1-step forward return), not raw price.
- Adds Directional Accuracy (% correct sign prediction) — most useful metric
  for return prediction.
- Reuses src.evaluation.metrics.calculate_metrics (no duplication).
- Loads *_processed_v2.csv and saves models as *_best_v2.pkl.

Input:  data/processed/{STOCK}_processed_v2.csv (30 files)
Output: models/baseline/{STOCK}_best_v2.pkl + results/baseline/baseline_results_v2.csv

Usage:
    python src/training/baseline_trainer_v2.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import pickle
from datetime import datetime

# Project paths and constants (centralized)
from src.utils.config import (
    PROJECT_ROOT,
    PROCESSED_DATA_DIR as DATA_DIR,
    BASELINE_MODELS_DIR as MODELS_DIR,
    BASELINE_RESULTS_DIR as RESULTS_DIR,
    LOGS_DIR,
    TEST_SIZE,
    RANDOM_STATE,
    TOP_30_DSE_STOCKS,
)
from src.evaluation.metrics import calculate_metrics
from src.utils.logger import get_logger

logger = get_logger("baseline_trainer_v2")
warnings = None  # placeholder; we import below
import warnings as _warnings
_warnings.filterwarnings('ignore')


# Columns that must NEVER appear in features (target leakage sources).
EXCLUDE_COLS = {
    # Metadata
    'date', 'code', 'name', 'sector',
    # Targets (the variable we're predicting)
    'Target_Return_1d', 'Target_Price_1d',
    # Same-day raw prices / volume — leaking current-day info into
    # the prediction of tomorrow's return.
    'open', 'high', 'low', 'close', 'volume', 'trade', 'value',
}


class BaselineMLTrainerV2:
    """Train leak-free baseline ML models for stock return prediction."""

    SUFFIX = "_v2"

    def __init__(self):
        self.data_dir = DATA_DIR
        self.models_dir = MODELS_DIR
        self.results_dir = RESULTS_DIR
        self.logs_dir = LOGS_DIR

        for d in [self.models_dir, self.results_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.feature_columns = None
        self.all_results = []

    # ----- Feature preparation -----

    def prepare_features(self, df):
        """Return feature columns excluding OHLCV and targets.

        The processed v2 CSVs have all indicators already shifted by 1 day,
        so by construction every column except OHLCV is leak-free. We
        additionally drop raw OHLCV here for defense in depth.
        """
        feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
        return feature_cols

    def train_test_split_time(self, df, test_size=TEST_SIZE):
        """Time-based train/test split (no shuffle!)."""
        split_idx = int(len(df) * (1 - test_size))
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
        return train, test

    # ----- Metrics -----

    def directional_accuracy(self, y_true, y_pred):
        """Percentage of cases where sign(y_true) == sign(y_pred).

        Most useful metric for return prediction: does the model predict
        up-days as up and down-days as down?
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100)

    def evaluate(self, y_true, y_pred):
        """Full evaluation: standard regression metrics + directional accuracy."""
        m = calculate_metrics(y_true, y_pred)
        m['Dir_Acc'] = self.directional_accuracy(y_true, y_pred)
        return m

    # ----- Model trainers -----

    def train_linear_regression(self, X_train, y_train):
        model = _import_sklearn_lr()
        model.fit(X_train, y_train)
        return model

    def train_random_forest(self, X_train, y_train):
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )
        model.fit(X_train, y_train)
        return model

    def train_xgboost(self, X_train, y_train):
        try:
            import xgboost as xgb
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method='hist',
                verbosity=0,
            )
            model.fit(X_train, y_train, verbose=False)
            return model
        except ImportError:
            logger.warning("XGBoost not installed. Skipping XGBoost.")
            return None

    def train_lightgbm(self, X_train, y_train):
        try:
            import lightgbm as lgb
            model = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(X_train, y_train)
            return model
        except ImportError:
            logger.warning("LightGBM not installed. Skipping LightGBM.")
            return None

    # ----- Per-stock training -----

    def train_stock(self, filepath):
        """Train all 4 models on a single stock."""
        try:
            df = pd.read_csv(filepath, parse_dates=['date'])
            stock_code = df['code'].iloc[0] if 'code' in df.columns else filepath.stem.replace('_processed_v2', '')
            stock_name = df['name'].iloc[0] if 'name' in df.columns else stock_code

            logger.info(f"\n🚀 Training v2 models for {stock_code} ({stock_name})...")

            # Features and target (returns, not price)
            feature_cols = self.prepare_features(df)
            target_col = 'Target_Return_1d'

            X = df[feature_cols].values
            y = df[target_col].values

            # Time-based split
            train_data, test_data = self.train_test_split_time(df)
            train_idx = len(train_data)

            X_train, y_train = X[:train_idx], y[:train_idx]
            X_test, y_test = X[train_idx:], y[train_idx:]

            logger.info(f"   Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(feature_cols)}")
            logger.info(f"   Target: {target_col} (next-day return)")

            results = {'stock': stock_code, 'name': stock_name}
            models = {}

            # 1) Linear Regression
            logger.info("   🔄 Linear Regression...")
            lr = self.train_linear_regression(X_train, y_train)
            results['LinearRegression'] = self.evaluate(y_test, lr.predict(X_test))
            models['lr_model'] = lr
            _log_metrics(results['LinearRegression'])

            # 2) Random Forest
            logger.info("   🔄 Random Forest...")
            rf = self.train_random_forest(X_train, y_train)
            results['RandomForest'] = self.evaluate(y_test, rf.predict(X_test))
            models['rf_model'] = rf
            _log_metrics(results['RandomForest'])

            # 3) XGBoost
            logger.info("   🔄 XGBoost...")
            xgb_model = self.train_xgboost(X_train, y_train)
            if xgb_model is not None:
                results['XGBoost'] = self.evaluate(y_test, xgb_model.predict(X_test))
                models['xgb_model'] = xgb_model
                _log_metrics(results['XGBoost'])

            # 4) LightGBM (NEW)
            logger.info("   🔄 LightGBM...")
            lgbm = self.train_lightgbm(X_train, y_train)
            if lgbm is not None:
                results['LightGBM'] = self.evaluate(y_test, lgbm.predict(X_test))
                models['lgbm_model'] = lgbm
                _log_metrics(results['LightGBM'])

            # Best by RMSE (lower is better)
            model_results = {k: v for k, v in results.items() if isinstance(v, dict)}
            if model_results:
                best_name, best_metrics = min(model_results.items(), key=lambda x: x[1]['RMSE'])
                results['best_model'] = best_name
                results['best_rmse'] = best_metrics['RMSE']
                logger.info(f"   🏆 Best: {best_name} (RMSE: {best_metrics['RMSE']:.6f}, Dir_Acc: {best_metrics['Dir_Acc']:.1f}%)")

            # Save best model
            model_map = {
                'LinearRegression': 'lr_model',
                'RandomForest': 'rf_model',
                'XGBoost': 'xgb_model',
                'LightGBM': 'lgbm_model',
            }
            best_model_obj = models.get(model_map.get(results.get('best_model')))
            if best_model_obj is not None:
                model_path = self.models_dir / f"{stock_code}_best{self.SUFFIX}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump({
                        'model': best_model_obj,
                        'features': feature_cols,
                        'stock': stock_code,
                        'target': target_col,
                        'version': 'v2',
                    }, f)

            return results, {
                'models': models,
                'feature_cols': feature_cols,
                'y_test': y_test,
            }

        except Exception as e:
            logger.error(f"   ❌ Error training {filepath.name}: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def train_all_stocks(self, max_stocks=None):
        """Train v2 models on all stocks."""
        logger.info("=" * 70)
        logger.info("📊 Baseline ML Models Training - Phase 3 (v2 — LEAK-FREE)")
        logger.info("   Models: Linear Regression, Random Forest, XGBoost, LightGBM")
        logger.info("   Target: Target_Return_1d (next-day return)")
        logger.info("=" * 70)

        files = sorted(self.data_dir.glob(f"*_processed{self.SUFFIX}.csv"))
        if not files:
            logger.error(f"❌ No *_processed{self.SUFFIX}.csv files found in {self.data_dir}")
            logger.error("   Run technical_indicators_v2.py first.")
            return []

        if max_stocks:
            files = files[:max_stocks]

        logger.info(f"📁 Data Directory: {self.data_dir}")
        logger.info(f"📊 Stocks to Process: {len(files)}")
        logger.info(f"💾 Models Directory: {self.models_dir}")
        logger.info(f"📈 Results Directory: {self.results_dir}")
        logger.info("=" * 70)

        all_results = []
        successful = 0
        failed = []

        for i, filepath in enumerate(files, 1):
            logger.info(f"\n[{i}/{len(files)}]")
            result, _ = self.train_stock(filepath)
            if result:
                all_results.append(result)
                successful += 1
            else:
                failed.append(filepath.name)

        # Save and summarize
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_path = self.results_dir / f"baseline_results{self.SUFFIX}.csv"
            # Expand metrics dicts to separate columns for downstream readability
            for model in ['LinearRegression', 'RandomForest', 'XGBoost', 'LightGBM']:
                if model in results_df.columns:
                    metrics_df = results_df[model].apply(pd.Series)
                    metrics_df.columns = [f'{model}_{c}' for c in metrics_df.columns]
                    results_df = pd.concat([results_df.drop(columns=[model]), metrics_df], axis=1)
            results_df.to_csv(results_path, index=False)

            logger.info("\n" + "=" * 70)
            logger.info("📊 TRAINING SUMMARY (v2)")
            logger.info("=" * 70)
            logger.info(f"Total Stocks: {len(files)}")
            logger.info(f"✅ Successful: {successful}")
            logger.info(f"❌ Failed: {len(failed)}")

            for model in ['LinearRegression', 'RandomForest', 'XGBoost', 'LightGBM']:
                rmse_col = f'{model}_RMSE'
                r2_col = f'{model}_R²'
                dir_col = f'{model}_Dir_Acc'
                if rmse_col in results_df.columns:
                    avg_rmse = results_df[rmse_col].mean()
                    avg_r2 = results_df[r2_col].mean() if r2_col in results_df.columns else float('nan')
                    avg_dir = results_df[dir_col].mean() if dir_col in results_df.columns else float('nan')
                    logger.info(
                        f"\n📈 {model}:\n"
                        f"   Avg RMSE:    {avg_rmse:.6f}\n"
                        f"   Avg R²:      {avg_r2:.4f}\n"
                        f"   Avg Dir_Acc: {avg_dir:.1f}%"
                    )

            if 'best_model' in results_df.columns:
                best_counts = results_df['best_model'].value_counts().to_dict()
                logger.info(f"\n🏆 Best Model Distribution:")
                for m, c in sorted(best_counts.items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"   {m}: {c} stocks")

            logger.info(f"\n💾 Results saved: {results_path}")
            logger.info("=" * 70)
            logger.info("✨ Training complete (v2)!")

        return all_results


# ----- Helpers -----

def _import_sklearn_lr():
    """LinearRegression (small wrapper to allow mocking in tests)."""
    from sklearn.linear_model import LinearRegression
    return LinearRegression()


def _log_metrics(metrics):
    logger.info(
        f"      RMSE: {metrics['RMSE']:.6f} | "
        f"MAE: {metrics['MAE']:.6f} | "
        f"R²: {metrics['R²']:.4f} | "
        f"Dir_Acc: {metrics['Dir_Acc']:.1f}%"
    )


def main():
    trainer = BaselineMLTrainerV2()
    trainer.train_all_stocks()


if __name__ == "__main__":
    main()