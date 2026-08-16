"""
Baseline ML Models Training - Phase 3
Trains 3 baseline models on processed stock data:
- Linear Regression
- Random Forest
- XGBoost

Goal: Predict next-day stock price/return

Input:  data/processed/{STOCK}_processed.csv
Output: models/saved/ trained models + predictions

Usage:
    python scripts/train_baseline.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


# Project paths (centralized)
from src.utils.config import (
    PROJECT_ROOT,
    PROCESSED_DATA_DIR as DATA_DIR,
    BASELINE_MODELS_DIR as MODELS_DIR,
    BASELINE_RESULTS_DIR as RESULTS_DIR,
    LOGS_DIR,
)


class BaselineMLTrainer:
    """Train baseline ML models for stock prediction"""

    def __init__(self):
        self.data_dir = DATA_DIR
        self.models_dir = MODELS_DIR
        self.results_dir = RESULTS_DIR
        self.logs_dir = LOGS_DIR

        # Create directories
        for d in [self.models_dir, self.results_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.feature_columns = None
        self.all_results = []

    def prepare_features(self, df):
        """Select feature columns"""
        # Exclude non-numeric and target columns
        exclude_cols = ['date', 'code', 'name', 'sector', 'Target_Return_1d', 'Target_Price_1d']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        return feature_cols

    def train_test_split_time(self, df, test_size=0.2):
        """Time-based train/test split (no shuffle!)"""
        split_idx = int(len(df) * (1 - test_size))
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]

        return train, test

    def calculate_metrics(self, y_true, y_pred):
        """Calculate regression metrics"""
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'R²': r2_score(y_true, y_pred),
        }
        return metrics

    def train_linear_regression(self, X_train, y_train):
        """Train Linear Regression model"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        model.fit(X_train, y_train)
        return model

    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model"""
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
                random_state=42,
                n_jobs=-1,
                tree_method='hist',
                verbosity=0
            )
            model.fit(X_train, y_train, verbose=False)
            return model
        except ImportError:
            print("XGBoost not installed. Skipping.")
            return None

    def train_stock(self, filepath):
        """Train all models on a single stock"""
        try:
            df = pd.read_csv(filepath, parse_dates=['date'])
            stock_code = df['code'].iloc[0] if 'code' in df.columns else filepath.stem
            stock_name = df['name'].iloc[0] if 'name' in df.columns else stock_code

            print(f"\n Training models for {stock_code} ({stock_name})...")

            # Get features
            feature_cols = self.prepare_features(df)
            target_col = 'Target_Price_1d'

            X = df[feature_cols].values
            y = df[target_col].values

            # Train/test split (time-based)
            train_data, test_data = self.train_test_split_time(df, test_size=0.2)
            train_idx = len(train_data)

            X_train = X[:train_idx]
            y_train = y[:train_idx]
            X_test = X[train_idx:]
            y_test = y[train_idx:]

            print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
            print(f"   Features: {len(feature_cols)}")

            results = {'stock': stock_code, 'name': stock_name}

            # Train Linear Regression
            print(f"   🔄 Training Linear Regression...")
            lr_model = self.train_linear_regression(X_train, y_train)
            lr_pred = lr_model.predict(X_test)
            lr_metrics = self.calculate_metrics(y_test, lr_pred)
            results['LinearRegression'] = lr_metrics
            print(f"      RMSE: {lr_metrics['RMSE']:.2f}, R²: {lr_metrics['R²']:.4f}")

            # Train Random Forest
            print(f"   🔄 Training Random Forest...")
            rf_model = self.train_random_forest(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            rf_metrics = self.calculate_metrics(y_test, rf_pred)
            results['RandomForest'] = rf_metrics
            print(f"      RMSE: {rf_metrics['RMSE']:.2f}, R²: {rf_metrics['R²']:.4f}")

            # Train XGBoost
            print(f"   🔄 Training XGBoost...")
            xgb_model = self.train_xgboost(X_train, y_train)

            if xgb_model is not None:
                xgb_pred = xgb_model.predict(X_test)
                xgb_metrics = self.calculate_metrics(y_test, xgb_pred)
                results['XGBoost'] = xgb_metrics
                print(f"      RMSE: {xgb_metrics['RMSE']:.2f}, R²: {xgb_metrics['R²']:.4f}")

            # Determine best model
            all_models = {k: v for k, v in results.items() if isinstance(v, dict)}
            if all_models:
                best_model = min(all_models.items(), key=lambda x: x[1]['RMSE'])
                results['best_model'] = best_model[0]
                results['best_rmse'] = best_model[1]['RMSE']
                print(f"   🏆 Best: {best_model[0]} (RMSE: {best_model[1]['RMSE']:.2f})")

            return results, {
                'lr_model': lr_model,
                'rf_model': rf_model,
                'xgb_model': xgb_model,
                'feature_cols': feature_cols,
                'y_test': y_test,
                'predictions': {
                    'LinearRegression': lr_pred,
                    'RandomForest': rf_pred,
                    'XGBoost': xgb_pred if xgb_model is not None else None
                }
            }

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def train_all_stocks(self, max_stocks=None):
        """Train models on all stocks"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║   Baseline ML Models Training - Phase 3                     ║
║   Models: Linear Regression, Random Forest, XGBoost         ║
╚══════════════════════════════════════════════════════════════╝
        """)

        # Get files
        files = sorted(self.data_dir.glob("*_processed.csv"))

        if max_stocks:
            files = files[:max_stocks]

        print(f"📁 Data Directory: {self.data_dir}")
        print(f"📊 Stocks to Process: {len(files)}")
        print(f"💾 Models Directory: {self.models_dir}")
        print(f"📈 Results Directory: {self.results_dir}")
        print("=" * 70)

        all_results = []
        successful = 0
        failed = []

        for i, filepath in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}]", end="")
            result, models = self.train_stock(filepath)

            if result:
                all_results.append(result)
                successful += 1

                # Save the best model
                if models and result.get('best_model'):
                    best_name = result['best_model']
                    best_model = models.get(f"{best_name.lower().replace(' ', '_')}_model")
                    if best_model is None:
                        # Map model names correctly
                        model_map = {
                            'LinearRegression': 'lr_model',
                            'RandomForest': 'rf_model',
                            'XGBoost': 'xgb_model'
                        }
                        model_key = model_map.get(best_name)
                        best_model = models.get(model_key)

                    if best_model:
                        model_path = self.models_dir / f"{result['stock']}_best.pkl"
                        with open(model_path, 'wb') as f:
                            pickle.dump({
                                'model': best_model,
                                'features': models['feature_cols'],
                                'stock': result['stock']
                            }, f)
            else:
                failed.append(filepath.name)

        # Save results
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_path = self.results_dir / "baseline_results.csv"
            results_df.to_csv(results_path, index=False)

            # Create summary
            print("\n" + "=" * 70)
            print("📊 TRAINING SUMMARY")
            print("=" * 70)
            print(f"Total Stocks: {len(files)}")
            print(f"✅ Successful: {successful}")
            print(f"❌ Failed: {len(failed)}")

            # Average metrics per model
            model_names = ['LinearRegression', 'RandomForest', 'XGBoost']
            for model_name in model_names:
                model_results = [r for r in all_results if model_name in r and isinstance(r[model_name], dict)]
                if model_results:
                    avg_rmse = np.mean([r[model_name]['RMSE'] for r in model_results])
                    avg_r2 = np.mean([r[model_name]['R²'] for r in model_results])
                    print(f"\n📈 {model_name}:")
                    print(f"   Avg RMSE: {avg_rmse:.2f}")
                    print(f"   Avg R²:   {avg_r2:.4f}")

            # Best models distribution
            if any('best_model' in r for r in all_results):
                best_counts = {}
                for r in all_results:
                    if 'best_model' in r:
                        best_counts[r['best_model']] = best_counts.get(r['best_model'], 0) + 1

                print(f"\n🏆 Best Model Distribution:")
                for model, count in sorted(best_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"   {model}: {count} stocks")

            print(f"\n💾 Results saved: {results_path}")
            print("=" * 70)
            print("✨ Training complete!")
            print("=" * 70)

        return all_results


def main():
    trainer = BaselineMLTrainer()

    # Train on subset first (10 stocks) for quick test
    # trainer.train_all_stocks(max_stocks=10)

    # Train on all stocks
    trainer.train_all_stocks()


if __name__ == "__main__":
    main()
