"""
Re-score existing LSTM checkpoints to build the results CSV.

Useful when training was interrupted but checkpoints exist. For each
*_best_lstm.pkl sidecar, loads the model and evaluates on the test split
of the *_processed_v2.csv, then writes results/deep_learning/deep_learning_results.csv.

Usage:
    python scripts/dl_rescore_existing.py
"""

from __future__ import annotations

import sys
import pickle
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils.config import (
    PROCESSED_DATA_DIR as DATA_DIR,
    DEEP_LEARNING_MODELS_DIR as MODELS_DIR,
    DL_RESULTS_DIR as RESULTS_DIR,
    DL_BATCH_SIZE,
    get_device,
)
from src.utils.logger import get_logger
from src.data_processing.sequence_dataset import prepare_stock_sequences
from src.training.architectures.lstm import StockLSTM
from src.training.deep_learning_trainer import evaluate_predictions


logger = get_logger("dl_rescore")


def rescore_one(stock_code: str, device) -> dict | None:
    sidecar_path = MODELS_DIR / f"{stock_code}_best_lstm.pkl"
    state_path = MODELS_DIR / f"{stock_code}_best_lstm.pt"
    if not sidecar_path.exists() or not state_path.exists():
        return None

    with open(sidecar_path, "rb") as f:
        sidecar = pickle.load(f)

    cfg = sidecar["config"]
    model = StockLSTM(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        output_dim=cfg["output_dim"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()

    data_path = DATA_DIR / f"{stock_code}_processed_v2.csv"
    df = pd.read_csv(data_path, parse_dates=["date"])
    _, _, test_ds, _, _ = prepare_stock_sequences(df)
    if len(test_ds) == 0:
        logger.warning(f"   ⚠️  No test data for {stock_code}")
        return None

    loader = DataLoader(test_ds, batch_size=DL_BATCH_SIZE, shuffle=False)
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            p = model(X).view(-1).cpu().numpy()
            preds.append(p)
            targets.append(y.numpy())
    preds = np.concatenate(preds) if preds else np.zeros(0)
    targets = np.concatenate(targets) if targets else np.zeros(0)
    metrics = evaluate_predictions(targets, preds)

    logger.info(
        f"   ✅ {stock_code:15s}  RMSE={metrics['RMSE']:.6f}  "
        f"Dir_Acc={metrics['Dir_Acc']:.1f}%  "
        f"epochs={sidecar.get('epochs_trained', '?')}  "
        f"val_loss={sidecar.get('best_val_loss', float('nan')):.6f}"
    )

    return {
        "stock": stock_code,
        "name": sidecar.get("name", stock_code),
        "arch": "LSTM",
        "test_rmse": metrics["RMSE"],
        "test_mae": metrics["MAE"],
        "test_mape": metrics["MAPE"],
        "test_r2": metrics["R²"],
        "test_dir_acc": metrics["Dir_Acc"],
        "best_val_loss": sidecar.get("best_val_loss", float("nan")),
        "epochs_trained": sidecar.get("epochs_trained", 0),
        "n_features": cfg["input_dim"],
        "n_train_windows": 0,  # not stored per-window
        "n_test_windows": len(test_ds),
    }


def main():
    device = get_device() or torch.device("cpu")
    logger.info(f"🔄 Re-scoring existing LSTM checkpoints on {device}...")

    pt_files = sorted(MODELS_DIR.glob("*_best_lstm.pt"))
    logger.info(f"📁 Found {len(pt_files)} checkpoints in {MODELS_DIR}")

    results = []
    for p in pt_files:
        stock = p.stem.replace("_best_lstm", "")
        r = rescore_one(stock, device)
        if r:
            results.append(r)

    if not results:
        logger.error("❌ No results to save.")
        return

    df = pd.DataFrame(results)
    out = RESULTS_DIR / "deep_learning_results.csv"
    df.to_csv(out, index=False)
    logger.info(f"\n💾 Saved {len(results)} rows -> {out}")
    logger.info(
        f"\n📊 Aggregate:\n"
        f"   Avg RMSE:    {df['test_rmse'].mean():.6f}\n"
        f"   Avg R²:      {df['test_r2'].mean():.4f}\n"
        f"   Avg Dir_Acc: {df['test_dir_acc'].mean():.1f}%"
    )


if __name__ == "__main__":
    main()
