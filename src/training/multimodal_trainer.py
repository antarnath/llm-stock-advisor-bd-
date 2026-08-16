"""
Multimodal Trainer — Phase 7 (price + sentiment, leak-free).

Trains one multimodal LSTM per stock per fusion strategy (early / late)
on next-day returns (Target_Return_1d) using a 60-day price window +
a 60-day sentiment window. Mirrors the API of deep_learning_trainer.py
so downstream phases can compare them.

Per-stock artifact layout:
    models/multimodal/{STOCK}_best_mm_{early|late}.pt   # state_dict
    models/multimodal/{STOCK}_best_mm_{early|late}.pkl  # sidecar

Result CSV:
    results/multimodal/multimodal_results.csv
        Columns: stock, name, arch, fusion_strategy, test_rmse, test_mae,
                 test_mape, test_r2, test_dir_acc, best_val_loss,
                 epochs_trained, n_features, n_train_windows, n_test_windows

Usage:
    python src/training/multimodal_trainer.py                      # both fusions, all stocks
    python src/training/multimodal_trainer.py --fusion early       # only early
    python src/training/multimodal_trainer.py --max-stocks 1 --epochs 5  # smoke test
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config import (
    PROCESSED_DATA_DIR,
    SENTIMENT_RESULTS_DIR,
    MULTIMODAL_MODELS_DIR,
    MULTIMODAL_RESULTS_DIR,
    MM_BATCH_SIZE,
    MM_EPOCHS,
    MM_LEARNING_RATE,
    MM_PATIENCE,
    MM_SEQUENCE_LENGTH,
    MM_SENTIMENT_COLS,
    get_device,
)
from src.evaluation.metrics import calculate_metrics
from src.utils.logger import get_logger
from src.data_processing.multimodal_dataset import prepare_multimodal_sequences
from src.training.architectures.multimodal_lstm import (
    MultimodalLSTMEarly,
    MultimodalLSTMLate,
)


logger = get_logger("multimodal_trainer")


def directional_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100)


def evaluate_predictions(y_true, y_pred) -> dict:
    m = calculate_metrics(y_true, y_pred)
    m["Dir_Acc"] = directional_accuracy(y_true, y_pred)
    return m


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MultimodalTrainer:
    """Trains one multimodal LSTM per stock per fusion strategy."""

    SUFFIX = "_v2"
    SENTIMENT_CSV_NAME = "stock_daily_sentiment.csv"

    def __init__(
        self,
        sentiment_csv: Optional[Path] = None,
        batch_size: int = MM_BATCH_SIZE,
        epochs: int = MM_EPOCHS,
        learning_rate: float = MM_LEARNING_RATE,
        patience: int = MM_PATIENCE,
        device: Optional[torch.device] = None,
    ):
        self.data_dir = PROCESSED_DATA_DIR
        self.sentiment_csv = sentiment_csv or (SENTIMENT_RESULTS_DIR / self.SENTIMENT_CSV_NAME)
        self.models_dir = MULTIMODAL_MODELS_DIR
        self.results_dir = MULTIMODAL_RESULTS_DIR

        for d in [self.models_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)

        if not self.sentiment_csv.exists():
            raise FileNotFoundError(
                f"Sentiment CSV not found: {self.sentiment_csv}. "
                f"Run Phase 6 first (scoring_pipeline.py)."
            )

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.device = device if device is not None else get_device()
        if self.device is None:
            raise RuntimeError("PyTorch is not installed. Run: pip install torch==2.2.0")

        self.all_results: list[dict] = []

    # ---- Model factory ----

    def _build_model(self, fusion: str, n_price: int, n_sent: int) -> nn.Module:
        if fusion == "early":
            return MultimodalLSTMEarly(
                input_dim_price=n_price, input_dim_sentiment=n_sent,
            ).to(self.device)
        if fusion == "late":
            return MultimodalLSTMLate(
                input_dim_price=n_price, input_dim_sentiment=n_sent,
            ).to(self.device)
        raise ValueError(f"Unknown fusion: {fusion!r}. Choose 'early' or 'late'.")

    # ---- Single training pass ----

    def _train_pass(
        self,
        model: nn.Module,
        train_ds,
        val_ds,
    ) -> tuple[nn.Module, float, dict, int]:
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False, drop_last=False,
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate, weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3,
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}
        epoch = 0

        for epoch in range(1, self.epochs + 1):
            model.train()
            train_loss_sum, train_n = 0.0, 0
            for price, sent, y in train_loader:
                price = price.to(self.device)
                sent = sent.to(self.device)
                y = y.to(self.device).view(-1, 1)
                optimizer.zero_grad()
                pred = model(price, sent)
                loss = criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss_sum += loss.item() * price.size(0)
                train_n += price.size(0)
            train_loss = train_loss_sum / max(train_n, 1)

            model.eval()
            val_loss_sum, val_n = 0.0, 0
            with torch.no_grad():
                for price, sent, y in val_loader:
                    price = price.to(self.device)
                    sent = sent.to(self.device)
                    y = y.to(self.device).view(-1, 1)
                    pred = model(price, sent)
                    val_loss_sum += criterion(pred, y).item() * price.size(0)
                    val_n += price.size(0)
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

    # ---- Test prediction ----

    def _predict_test(self, model: nn.Module, test_ds) -> tuple[np.ndarray, np.ndarray]:
        if len(test_ds) == 0:
            return np.zeros(0), np.zeros(0)
        loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for price, sent, y in loader:
                price = price.to(self.device)
                sent = sent.to(self.device)
                pred = model(price, sent).view(-1).cpu().numpy()
                preds.append(pred)
                targets.append(y.numpy())
        preds = np.concatenate(preds) if preds else np.zeros(0)
        targets = np.concatenate(targets) if targets else np.zeros(0)
        return preds, targets

    # ---- Single-stock full pipeline ----

    def train_stock(self, filepath: Path, fusion: str) -> Optional[dict]:
        try:
            df = pd.read_csv(filepath, parse_dates=["date"])
            stock_code = (
                df["code"].iloc[0]
                if "code" in df.columns
                else filepath.stem.replace(f"_processed{self.SUFFIX}", "")
            )
            stock_name = df["name"].iloc[0] if "name" in df.columns else stock_code

            logger.info(
                f"\n🚀 Training MM-{fusion.upper()} for {stock_code} ({stock_name})..."
            )

            # Prepare multimodal sequences
            (
                train_ds, val_ds, test_ds,
                price_scaler, sentiment_scaler,
                price_features, sentiment_features,
            ) = prepare_multimodal_sequences(
                csv_path=filepath,
                sentiment_csv=self.sentiment_csv,
                stock_code=stock_code,
            )

            n_price = len(price_features)
            n_sent = len(sentiment_features)
            logger.info(
                f"   Price features: {n_price} | Sentiment features: {n_sent} | "
                f"train windows: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}"
            )

            if len(train_ds) < self.batch_size or len(test_ds) == 0:
                logger.warning(
                    f"   ⚠️  Skipping {stock_code}: not enough data "
                    f"(train={len(train_ds)}, test={len(test_ds)})"
                )
                return None

            model = self._build_model(fusion, n_price, n_sent)
            t0 = time.time()
            model, best_val_loss, history, epochs_trained = self._train_pass(
                model, train_ds, val_ds,
            )
            train_time = time.time() - t0

            preds, targets = self._predict_test(model, test_ds)
            metrics = evaluate_predictions(targets, preds)

            logger.info(
                f"   ✅ MM-{fusion} | epochs={epochs_trained} "
                f"val_loss={best_val_loss:.6f} "
                f"RMSE={metrics['RMSE']:.6f} Dir_Acc={metrics['Dir_Acc']:.1f}% "
                f"({train_time:.1f}s)"
            )

            # Save checkpoint + sidecar
            model_path = self.models_dir / f"{stock_code}_best_mm_{fusion}.pt"
            sidecar_path = self.models_dir / f"{stock_code}_best_mm_{fusion}.pkl"
            torch.save(model.state_dict(), model_path)

            sidecar = {
                "stock": stock_code,
                "name": stock_name,
                "target": "Target_Return_1d",
                "features": price_features,
                "sentiment_features": sentiment_features,
                "arch": model.arch_name,
                "fusion": fusion,
                "config": model.config_dict(),
                "scaler": price_scaler,         # backward-compat with Phase 4 loaders
                "sentiment_scaler": sentiment_scaler,
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
                "arch": model.arch_name,
                "fusion_strategy": fusion,
                "test_rmse": metrics["RMSE"],
                "test_mae": metrics["MAE"],
                "test_mape": metrics["MAPE"],
                "test_r2": metrics["R²"],
                "test_dir_acc": metrics["Dir_Acc"],
                "best_val_loss": best_val_loss,
                "epochs_trained": epochs_trained,
                "n_features": n_price + n_sent,
                "n_train_windows": len(train_ds),
                "n_test_windows": len(test_ds),
            }

        except Exception as e:
            logger.error(f"   ❌ Error training {filepath.name} ({fusion}): {e}")
            import traceback
            traceback.print_exc()
            return None

    # ---- All stocks ----

    def train_all_stocks(
        self,
        fusion: str = "early",
        max_stocks: Optional[int] = None,
    ):
        logger.info("=" * 70)
        logger.info(f"🧠 Multimodal Training — Phase 7 (fusion={fusion.upper()}, LEAK-FREE)")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Target: Target_Return_1d (next-day return)")
        logger.info(f"   Sequence length: {MM_SEQUENCE_LENGTH}")
        logger.info(f"   Epochs: {self.epochs} | Patience: {self.patience}")
        logger.info(f"   Sentiment CSV: {self.sentiment_csv}")
        logger.info("=" * 70)

        files = sorted(self.data_dir.glob(f"*_processed{self.SUFFIX}.csv"))
        if not files:
            logger.error(
                f"� No *_processed{self.SUFFIX}.csv files found in {self.data_dir}"
            )
            return []

        if max_stocks:
            files = files[:max_stocks]

        logger.info(f"📁 Data Directory: {self.data_dir}")
        logger.info(f"📊 Stocks to Process: {len(files)}")
        logger.info(f"💾 Models Directory: {self.models_dir}")
        logger.info(f"📈 Results Directory: {self.results_dir}")
        logger.info("=" * 70)

        results_path = self.results_dir / "multimodal_results.csv"

        # Load the FULL existing CSV (all fusions), not just this fusion's rows,
        # so we preserve other fusions when writing incrementally.
        all_existing_df = pd.DataFrame()
        existing_for_this_fusion = []
        already_trained = set()
        if results_path.exists():
            try:
                all_existing_df = pd.read_csv(results_path)
                # Filter to rows matching THIS fusion for resume detection
                same_fusion = all_existing_df[all_existing_df["fusion_strategy"] == fusion]
                existing_for_this_fusion = same_fusion.to_dict("records")
                already_trained = set(same_fusion["stock"].tolist())
                logger.info(
                    f"♻️  Resuming {fusion}: {len(already_trained)} stocks already trained"
                    f" ({len(all_existing_df)} total rows in CSV across all fusions)"
                )
            except Exception:
                logger.warning("⚠️  Could not parse existing results CSV, starting fresh")

        all_results = []
        successful = 0
        failed = []

        for i, filepath in enumerate(files, 1):
            stock_code = filepath.stem.replace(f"_processed{self.SUFFIX}", "")
            if stock_code in already_trained:
                logger.info(f"\n[{i}/{len(files)}] ⏭️  Skipping {stock_code} ({fusion}, already trained)")
                continue

            logger.info(f"\n[{i}/{len(files)}]")
            result = self.train_stock(filepath, fusion)
            if result:
                all_results.append(result)
                successful += 1
                # Write incremental snapshot: existing rows for THIS fusion + new rows for THIS fusion,
                # plus all rows for OTHER fusions preserved from the original CSV.
                this_fusion_rows = existing_for_this_fusion + all_results
                other_fusion_rows = (
                    all_existing_df[all_existing_df["fusion_strategy"] != fusion]
                    if not all_existing_df.empty else pd.DataFrame()
                )
                if not other_fusion_rows.empty:
                    full_snapshot = pd.concat(
                        [other_fusion_rows, pd.DataFrame(this_fusion_rows)],
                        ignore_index=True,
                    )
                else:
                    full_snapshot = pd.DataFrame(this_fusion_rows)
                full_snapshot.to_csv(results_path, index=False)
            else:
                failed.append(filepath.name)

        all_results = existing_for_this_fusion + all_results

        if all_results:
            results_df = pd.DataFrame(all_results)
            results_path = self.results_dir / "multimodal_results.csv"

            # Final merge: this fusion's results + all other fusions' existing rows
            other_fusion_rows = (
                all_existing_df[all_existing_df["fusion_strategy"] != fusion]
                if not all_existing_df.empty else pd.DataFrame()
            )
            if not other_fusion_rows.empty:
                full_df = pd.concat(
                    [other_fusion_rows, results_df],
                    ignore_index=True,
                )
            else:
                full_df = results_df
            full_df.to_csv(results_path, index=False)

            logger.info("\n" + "=" * 70)
            logger.info(f"📊 MM-{fusion.upper()} SUMMARY (Phase 7)")
            logger.info("=" * 70)
            logger.info(f"Total Stocks: {len(files)}")
            logger.info(f"✅ Successful: {successful}")
            logger.info(f"❌ Failed: {len(failed)}")
            logger.info(
                f"\n📈 MM-{fusion} (overall):\n"
                f"   Avg RMSE:      {results_df['test_rmse'].mean():.6f}\n"
                f"   Avg MAE:       {results_df['test_mae'].mean():.6f}\n"
                f"   Avg R²:        {results_df['test_r2'].mean():.4f}\n"
                f"   Avg Dir_Acc:   {results_df['test_dir_acc'].mean():.1f}%\n"
                f"   Avg Epochs:    {results_df['epochs_trained'].mean():.1f}"
            )

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
                logger.info(f"\n�️  Failed: {', '.join(failed)}")

            logger.info(f"\n💾 Results saved: {results_path}")
            logger.info("=" * 70)

        return all_results


def main():
    parser = argparse.ArgumentParser(description="Train Multimodal LSTM per stock (Phase 7).")
    parser.add_argument("--fusion", choices=["early", "late", "both"], default="both",
                        help="Fusion strategy to train (default: both)")
    parser.add_argument("--max-stocks", type=int, default=None, help="Limit to N stocks (debug).")
    parser.add_argument("--epochs", type=int, default=MM_EPOCHS, help="Max epochs per stock.")
    parser.add_argument("--batch-size", type=int, default=MM_BATCH_SIZE, help="Batch size.")
    parser.add_argument("--patience", type=int, default=MM_PATIENCE, help="Early-stopping patience.")
    args = parser.parse_args()

    trainer = MultimodalTrainer(
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
    )

    fusions = ["early", "late"] if args.fusion == "both" else [args.fusion]
    for fusion in fusions:
        trainer.train_all_stocks(fusion=fusion, max_stocks=args.max_stocks)


if __name__ == "__main__":
    main()