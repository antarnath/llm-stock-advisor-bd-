"""
Deep Learning Trainer — Phase 4 (LSTM only, leak-free).

Trains a single StockLSTM per stock on next-day returns (Target_Return_1d)
using lag-1 features from *_processed_v2.csv. Mirrors the API of
BaselineMLTrainerV2 so downstream phases can compare them.

Per-stock artifact layout:
    models/deep_learning/{STOCK}_best_lstm.pt     # state_dict
    models/deep_learning/{STOCK}_best_lstm.pkl    # sidecar with config + features + scaler

Result CSV:
    results/deep_learning/deep_learning_results.csv

Columns: stock, name, arch, test_rmse, test_mae, test_mape, test_r2, test_dir_acc,
         best_val_loss, epochs_trained, n_features, n_train_windows, n_test_windows

Usage:
    python src/training/deep_learning_trainer.py
    python src/training/deep_learning_trainer.py --max-stocks 1
    python src/training/deep_learning_trainer.py --epochs 5   # smoke test
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure `src` is importable when run as a script (python src/training/deep_learning_trainer.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Project config
from src.utils.config import (
    PROJECT_ROOT,
    PROCESSED_DATA_DIR as DATA_DIR,
    DEEP_LEARNING_MODELS_DIR as MODELS_DIR,
    DL_RESULTS_DIR as RESULTS_DIR,
    LOGS_DIR,
    DL_BATCH_SIZE,
    DL_EPOCHS,
    DL_HIDDEN_DIM,
    DL_LEARNING_RATE,
    DL_NUM_LAYERS,
    DL_PATIENCE,
    DL_SEQUENCE_LENGTH,
    get_device,
)

# Reuse the v2 evaluation helpers (no metric duplication)
from src.evaluation.metrics import calculate_metrics
from src.utils.logger import get_logger
from src.data_processing.sequence_dataset import (
    prepare_stock_sequences,
    select_feature_columns,
)
from src.training.architectures.lstm import StockLSTM


logger = get_logger("deep_learning_trainer")

# Ensure `src` is importable when run as a script (python src/training/deep_learning_trainer.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def directional_accuracy(y_true, y_pred) -> float:
    """Percent of cases where sign(y_true) == sign(y_pred)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100)


def evaluate_predictions(y_true, y_pred) -> dict:
    """RMSE/MAE/MAPE/R² + Dir_Acc on a flat 1-D pair of arrays."""
    m = calculate_metrics(y_true, y_pred)
    m["Dir_Acc"] = directional_accuracy(y_true, y_pred)
    return m


# ---------------------------------------------------------------------------
# Per-stock training
# ---------------------------------------------------------------------------

class DeepLearningTrainer:
    """Trains one StockLSTM per stock on next-day returns."""

    SUFFIX = "_v2"   # load processed_v2 CSVs
    ARCH = "lstm"

    def __init__(
        self,
        batch_size: int = DL_BATCH_SIZE,
        epochs: int = DL_EPOCHS,
        learning_rate: float = DL_LEARNING_RATE,
        patience: int = DL_PATIENCE,
        device: Optional[torch.device] = None,
    ):
        self.data_dir = DATA_DIR
        self.models_dir = MODELS_DIR
        self.results_dir = RESULTS_DIR
        self.logs_dir = LOGS_DIR

        for d in [self.models_dir, self.results_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.device = device if device is not None else get_device()
        if self.device is None:
            raise RuntimeError("PyTorch is not installed. Run: pip install torch==2.2.0")

        self.all_results: list[dict] = []

    # -- Single LSTM training pass on one (train, val) pair --

    def train_lstm(self, train_ds, val_ds, n_features: int):
        """Train a single StockLSTM with early stopping.

        Returns:
            (model, best_val_loss, history_dict, epochs_trained)
        """
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False, drop_last=False,
        )

        model = StockLSTM(input_dim=n_features).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate, weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10,
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, self.epochs + 1):
            # ---- train ----
            model.train()
            train_loss_sum, train_n = 0.0, 0
            for X, y in train_loader:
                X = X.to(self.device)
                y = y.to(self.device).view(-1, 1)
                optimizer.zero_grad()
                pred = model(X)
                loss = criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss_sum += loss.item() * X.size(0)
                train_n += X.size(0)
            train_loss = train_loss_sum / max(train_n, 1)

            # ---- validate ----
            model.eval()
            val_loss_sum, val_n = 0.0, 0
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.to(self.device)
                    y = y.to(self.device).view(-1, 1)
                    pred = model(X)
                    val_loss_sum += criterion(pred, y).item() * X.size(0)
                    val_n += X.size(0)
            val_loss = val_loss_sum / max(val_n, 1)

            scheduler.step(val_loss)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(
                        f"      ⏹  Early stopping at epoch {epoch} "
                        f"(best val loss: {best_val_loss:.6f})"
                    )
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        return model, best_val_loss, history, epoch

    # -- Full per-stock pipeline --

    def _predict_dataset(self, model, dataset) -> np.ndarray:
        """Run a model over a sequence dataset and return flat predictions."""
        if len(dataset) == 0:
            return np.zeros(0, dtype=np.float32)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        model.eval()
        preds = []
        with torch.no_grad():
            for X, _ in loader:
                X = X.to(self.device)
                pred = model(X).view(-1).cpu().numpy()
                preds.append(pred)
        return np.concatenate(preds) if preds else np.zeros(0, dtype=np.float32)

    def train_stock(self, filepath: Path) -> Optional[dict]:
        """Train LSTM on a single stock. Returns result dict or None on failure."""
        try:
            df = pd.read_csv(filepath, parse_dates=["date"])
            stock_code = (
                df["code"].iloc[0]
                if "code" in df.columns
                else filepath.stem.replace("_processed_v2", "")
            )
            stock_name = df["name"].iloc[0] if "name" in df.columns else stock_code

            logger.info(f"\n🚀 Training LSTM for {stock_code} ({stock_name})...")

            # Prepare sequences (leak-free: scaler fit on train only)
            train_ds, val_ds, test_ds, scaler, feature_cols = prepare_stock_sequences(df)

            n_features = len(feature_cols)
            logger.info(
                f"   Features: {n_features} | " 
                f"train windows: {len(train_ds)} | "
                f"val windows: {len(val_ds)} | "
                f"test windows: {len(test_ds)}"
            )

            if len(train_ds) < self.batch_size or len(test_ds) == 0:
                logger.warning(
                    f"   ⚠️  Skipping {stock_code}: not enough data "
                    f"(train={len(train_ds)}, test={len(test_ds)})"
                )
                return None

            t0 = time.time()
            model, best_val_loss, history, epochs_trained = self.train_lstm(
                train_ds, val_ds, n_features=n_features,
            )
            train_time = time.time() - t0

            # ---- Evaluate on test ----
            X_test = np.concatenate(
                [test_ds.features[i:i + test_ds.sequence_length] for i in range(len(test_ds))],
                axis=0,
            ).reshape(len(test_ds), test_ds.sequence_length, -1)
            # Easier: just run through DataLoader.
            test_preds = self._predict_dataset(model, test_ds)
            test_targets = np.array(
                [
                    test_ds.targets[i + test_ds.sequence_length]
                    for i in range(len(test_ds))
                ],
                dtype=np.float32,
            )
            metrics = evaluate_predictions(test_targets, test_preds)

            logger.info(
                f"   ✅ LSTM | epochs={epochs_trained} "
                f"val_loss={best_val_loss:.6f} "
                f"RMSE={metrics['RMSE']:.6f} Dir_Acc={metrics['Dir_Acc']:.1f}% "
                f"({train_time:.1f}s)"
            )

            # ---- Save checkpoint + sidecar ----
            model_path = self.models_dir / f"{stock_code}_best_{self.ARCH}.pt"
            sidecar_path = self.models_dir / f"{stock_code}_best_{self.ARCH}.pkl"
            torch.save(model.state_dict(), model_path)

            sidecar = {
                "stock": stock_code,
                "name": stock_name,
                "target": "Target_Return_1d",
                "features": feature_cols,
                "arch": "StockLSTM",
                "config": model.config_dict(),
                "scaler": scaler,  # StandardScaler (needed for inference)
                "history": history,
                "epochs_trained": epochs_trained,
                "best_val_loss": best_val_loss,
                "version": "v1",
            }
            with open(sidecar_path, "wb") as f:
                pickle.dump(sidecar, f)

            return {
                "stock": stock_code,
                "name": stock_name,
                "arch": "LSTM",
                "test_rmse": metrics["RMSE"],
                "test_mae": metrics["MAE"],
                "test_mape": metrics["MAPE"],
                "test_r2": metrics["R²"],
                "test_dir_acc": metrics["Dir_Acc"],
                "best_val_loss": best_val_loss,
                "epochs_trained": epochs_trained,
                "n_features": n_features,
                "n_train_windows": len(train_ds),
                "n_test_windows": len(test_ds),
            }

        except Exception as e:
            logger.error(f"   ❌ Error training {filepath.name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    # -- All stocks --

    def train_all_stocks(self, max_stocks: Optional[int] = None):
        """Train LSTM on all available *_processed_v2.csv files."""
        logger.info("=" * 70)
        logger.info("🧠 Deep Learning Training - Phase 4 (LSTM, LEAK-FREE)")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Target: Target_Return_1d (next-day return)")
        logger.info(f"   Sequence length: {DL_SEQUENCE_LENGTH}")
        logger.info(f"   Hidden dim: {DL_HIDDEN_DIM} | Layers: {DL_NUM_LAYERS}")
        logger.info(f"   Epochs: {self.epochs} | Patience: {self.patience}")
        logger.info("=" * 70)

        files = sorted(self.data_dir.glob(f"*_processed{self.SUFFIX}.csv"))
        if not files:
            logger.error(f"❌ No *_processed{self.SUFFIX}.csv files found in {self.data_dir}")
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

        # Load existing results if any (for resume support)
        results_path = self.results_dir / "deep_learning_results.csv"
        existing_results = []
        already_trained = set()
        if results_path.exists():
            try:
                existing_df = pd.read_csv(results_path)
                existing_results = existing_df.to_dict("records")
                already_trained = set(existing_df["stock"].tolist())
                logger.info(
                    f"♻️  Resuming from {len(already_trained)} already-trained stocks"
                )
            except Exception:
                logger.warning("⚠️  Could not parse existing results CSV, starting fresh")

        for i, filepath in enumerate(files, 1):
            stock_code = filepath.stem.replace(f"_processed{self.SUFFIX}", "")
            if stock_code in already_trained:
                logger.info(f"\n[{i}/{len(files)}] ⏭️  Skipping {stock_code} (already trained)")
                continue

            logger.info(f"\n[{i}/{len(files)}]")
            result = self.train_stock(filepath)
            if result:
                all_results.append(result)
                successful += 1
                # Save incrementally so results are visible during long runs
                combined = existing_results + all_results
                pd.DataFrame(combined).to_csv(results_path, index=False)
            else:
                failed.append(filepath.name)

        # Merge with existing results for the final summary
        all_results = existing_results + all_results

        if all_results:
            results_df = pd.DataFrame(all_results)
            results_path = self.results_dir / "deep_learning_results.csv"
            results_df.to_csv(results_path, index=False)

            logger.info("\n" + "=" * 70)
            logger.info("📊 LSTM TRAINING SUMMARY (Phase 4)")
            logger.info("=" * 70)
            logger.info(f"Total Stocks: {len(files)}")
            logger.info(f"✅ Successful: {successful}")
            logger.info(f"❌ Failed: {len(failed)}")
            logger.info(
                f"\n📈 LSTM (overall):\n"
                f"   Avg RMSE:      {results_df['test_rmse'].mean():.6f}\n"
                f"   Avg MAE:       {results_df['test_mae'].mean():.6f}\n"
                f"   Avg R²:        {results_df['test_r2'].mean():.4f}\n"
                f"   Avg Dir_Acc:   {results_df['test_dir_acc'].mean():.1f}%\n"
                f"   Avg Epochs:    {results_df['epochs_trained'].mean():.1f}"
            )

            # Top/bottom by directional accuracy (the trading-relevant metric)
            top = results_df.nlargest(5, "test_dir_acc")[["stock", "test_dir_acc", "test_rmse"]]
            bot = results_df.nsmallest(5, "test_dir_acc")[["stock", "test_dir_acc", "test_rmse"]]
            logger.info("\n🟢 TOP 5 by Dir_Acc:")
            for _, r in top.iterrows():
                logger.info(
                    f"   {r['stock']:15s} Dir_Acc={r['test_dir_acc']:.1f}%  "
                    f"RMSE={r['test_rmse']:.6f}"
                )
            logger.info("\n🔴 BOTTOM 5 by Dir_Acc:")
            for _, r in bot.iterrows():
                logger.info(
                    f"   {r['stock']:15s} Dir_Acc={r['test_dir_acc']:.1f}%  "
                    f"RMSE={r['test_rmse']:.6f}"
                )

            if failed:
                logger.info(f"\n⚠️  Failed: {', '.join(failed)}")

            logger.info(f"\n💾 Results saved: {results_path}")
            logger.info("=" * 70)
            logger.info("✨ Deep Learning training complete!")

        return all_results


def main():
    parser = argparse.ArgumentParser(description="Train LSTM per stock (Phase 4).")
    parser.add_argument("--max-stocks", type=int, default=None, help="Limit to N stocks (debug).")
    parser.add_argument("--epochs", type=int, default=DL_EPOCHS, help="Max epochs per stock.")
    parser.add_argument("--batch-size", type=int, default=DL_BATCH_SIZE, help="Batch size.")
    parser.add_argument("--patience", type=int, default=DL_PATIENCE, help="Early-stopping patience.")
    args = parser.parse_args()

    trainer = DeepLearningTrainer(
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
    )
    trainer.train_all_stocks(max_stocks=args.max_stocks)


if __name__ == "__main__":
    main()
