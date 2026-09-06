"""
Eval Phase 7 attention fusion — wires up the 18 orphaned attention checkpoints.

Loads each *_best_mm_attention.pt checkpoint and evaluates it on the same
leak-free test split used during Phase 7 training. Appends rows to
results/multimodal/multimodal_results.csv with fusion_strategy='attention'.

Usage:
    .venv/bin/python scripts/eval_phase7_attention.py
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.metrics import calculate_metrics
from src.training.architectures.multimodal_lstm import build_multimodal
from src.data_processing.multimodal_dataset import prepare_multimodal_sequences

MODELS_DIR = _PROJECT_ROOT / "models" / "multimodal"
RESULTS_DIR = _PROJECT_ROOT / "results" / "multimodal"
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
SENTIMENT_CSV = _PROJECT_ROOT / "results" / "sentiment" / "stock_daily_sentiment.csv"
RESULTS_CSV = RESULTS_DIR / "multimodal_results.csv"


def directional_accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100)


def evaluate_one(stock: str) -> dict | None:
    pt_path = MODELS_DIR / f"{stock}_best_mm_attention.pt"
    pkl_path = MODELS_DIR / f"{stock}_best_mm_attention.pkl"
    if not (pt_path.exists() and pkl_path.exists()):
        return None

    # Sidecar
    with open(pkl_path, "rb") as f:
        sidecar = pickle.load(f)

    # Reconstruct test split identically to Phase 7
    price_csv = PROCESSED_DIR / f"{stock}_processed_v2.csv"
    if not price_csv.exists():
        print(f"   ⚠️  {stock}: no processed CSV at {price_csv}")
        return None

    try:
        (
            _train_ds, _val_ds, test_ds,
            _ps, _ss, _pf, _sf,
        ) = prepare_multimodal_sequences(
            csv_path=price_csv,
            sentiment_csv=SENTIMENT_CSV,
            stock_code=stock,
        )
    except Exception as e:
        print(f"   ⚠️  {stock}: dataset prep failed — {e}")
        return None

    if len(test_ds) == 0:
        print(f"   ⚠️  {stock}: empty test split")
        return None

    # Load model
    model = build_multimodal(sidecar["config"])
    model.load_state_dict(
        torch.load(pt_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    # Predict
    preds, targets = [], []
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    with torch.no_grad():
        for price, sent, y in loader:
            out = model(price, sent).view(-1).numpy()
            preds.append(out)
            targets.append(y.numpy())
    preds = np.concatenate(preds) if preds else np.zeros(0)
    targets = np.concatenate(targets) if targets else np.zeros(0)

    if len(preds) == 0:
        return None

    m = calculate_metrics(targets, preds)
    m["Dir_Acc"] = directional_accuracy(targets, preds)

    return {
        "stock": stock,
        "name": sidecar.get("name", stock),
        "arch": "MultimodalLSTMAttention",
        "fusion_strategy": "attention",
        "test_rmse": m["RMSE"],
        "test_mae": m["MAE"],
        "test_mape": m["MAPE"],
        "test_r2": m["R²"],
        "test_dir_acc": m["Dir_Acc"],
        "best_val_loss": sidecar.get("best_val_loss", float("nan")),
        "epochs_trained": sidecar.get("epochs_trained", 0),
        "n_features": 27 + 7,
        "n_train_windows": len(_train_ds),
        "n_test_windows": len(test_ds),
    }


def main():
    print("=" * 70)
    print("🔍 PHASE 7 — ATTENTION FUSION EVALUATION")
    print("=" * 70)

    # Discover attention checkpoints
    attention_stocks = sorted(
        p.stem.replace("_best_mm_attention", "")
        for p in MODELS_DIR.glob("*_best_mm_attention.pt")
    )
    print(f"📁 Found {len(attention_stocks)} attention checkpoints: {attention_stocks}")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing results CSV to preserve early + late rows
    existing = pd.DataFrame()
    if RESULTS_CSV.exists():
        existing = pd.read_csv(RESULTS_CSV)
        print(f"♻️  Loaded existing results CSV ({len(existing)} rows)")

    # Eval each stock
    new_rows = []
    for i, stock in enumerate(attention_stocks, 1):
        print(f"\n[{i}/{len(attention_stocks)}] {stock}")
        row = evaluate_one(stock)
        if row is None:
            continue
        print(
            f"   ✅ RMSE={row['test_rmse']:.6f}  Dir_Acc={row['test_dir_acc']:.1f}%  "
            f"epochs={row['epochs_trained']}"
        )
        new_rows.append(row)

    if not new_rows:
        print("\n❌ No attention checkpoints evaluated.")
        return

    new_df = pd.DataFrame(new_rows)

    # Append / merge: drop any pre-existing attention rows to avoid duplicates,
    # then concat fresh attention rows + all other rows.
    non_attention = existing[existing["fusion_strategy"] != "attention"]
    merged = pd.concat([non_attention, new_df], ignore_index=True)
    merged.to_csv(RESULTS_CSV, index=False)

    print("\n" + "=" * 70)
    print(f"📊 ATTENTION FUSION SUMMARY (across {len(new_df)} stocks)")
    print("=" * 70)
    print(f"   Avg RMSE:    {new_df['test_rmse'].mean():.6f}")
    print(f"   Avg MAE:     {new_df['test_mae'].mean():.6f}")
    print(f"   Avg R²:      {new_df['test_r2'].mean():.4f}")
    print(f"   Avg Dir_Acc: {new_df['test_dir_acc'].mean():.1f}%")
    print(f"   Stocks ≥ 50%: {(new_df['test_dir_acc'] >= 50).sum()}/{len(new_df)}")
    print("=" * 70)
    print(f"💾 Results appended to: {RESULTS_CSV}")
    print(f"   Total rows now: {len(merged)} (was {len(existing)})")


if __name__ == "__main__":
    main()