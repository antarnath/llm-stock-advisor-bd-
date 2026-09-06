"""
Bangla-only vs English-only vs Bilingual ablation.

Trains the MultimodalLSTMEarly model on 3 sentiment variants per stock:
  - bilingual: all articles (already exists in stock_daily_sentiment.csv)
  - en_only:   English-only sentiment
  - bn_only:   Bangla-only sentiment

Then compares Dir_Acc to demonstrate that Bangla news adds value beyond
English alone — paper claim #3.

Outputs:
  results/multimodal/bangla_ablation_results.csv
  results/multimodal/bangla_ablation_summary.txt

Usage:
  .venv/bin/python scripts/eval_bangla_ablation.py --max-stocks 5
  .venv/bin/python scripts/eval_bangla_ablation.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config import (
    PROCESSED_DATA_DIR as DATA_DIR,
    MULTIMODAL_RESULTS_DIR,
)
from src.utils.logger import get_logger
from src.training.architectures.multimodal_lstm import MultimodalLSTMEarly

logger = get_logger("bangla_ablation")

SENT_CSV = _PROJECT_ROOT / "results" / "sentiment" / "news_scored.csv"
SEED = 42
WINDOW = 60
EPOCHS = 8
BATCH = 64


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_lang_sentiment_csv(out_path: Path, language: str | None):
    """Rebuild stock_daily_sentiment.csv keeping only one language."""
    df = pd.read_csv(SENT_CSV, parse_dates=["date"])
    if language is not None:
        df = df[df["language"] == language].copy()
    agg = df.groupby(["stock", "date"]).agg(
        n_articles=("score", "count"),
        mean_score=("score", "mean"),
        weighted_score=(
            "score",
            lambda x: (x * df.loc[x.index, "confidence"]).sum()
            / max(df.loc[x.index, "confidence"].sum(), 1e-9),
        ),
        mean_confidence=("confidence", "mean"),
        pos_count=("pred_label", lambda x: (x == "positive").sum()),
        neg_count=("pred_label", lambda x: (x == "negative").sum()),
        neu_count=("pred_label", lambda x: (x == "neutral").sum()),
    ).reset_index()
    agg["pos_ratio"] = agg["pos_count"] / agg["n_articles"]
    agg["neg_ratio"] = agg["neg_count"] / agg["n_articles"]
    agg.to_csv(out_path, index=False)
    return agg


def build_sequences(stock: str, sent_csv: Path):
    """Build (price_seq, sent_seq, target) for one stock using given sentiment CSV."""
    price_path = DATA_DIR / f"{stock}_processed_v2.csv"
    if not price_path.exists() or not sent_csv.exists():
        return None
    price_df = pd.read_csv(price_path, parse_dates=["date"])
    sent_df = pd.read_csv(sent_csv, parse_dates=["date"])

    # Drop target & non-feature cols
    target_col = "Target_Return_1d"
    drop = [c for c in price_df.columns if c.startswith("Target_") or c == target_col]
    drop += [c for c in ("date", "code", "name", "sector", "stock") if c in price_df.columns]
    price_feat = price_df.drop(columns=[c for c in drop if c in price_df.columns])

    # Sentiment features (use 7 standardized cols)
    sent_feat_cols = ["mean_score", "weighted_score", "mean_confidence",
                      "n_articles", "pos_ratio", "neg_ratio", "neu_count"]
    sent_feat = sent_df[["date"] + sent_feat_cols].copy()
    sent_feat[sent_feat_cols] = (
        sent_feat[sent_feat_cols].fillna(0.0).astype(float)
    )

    # Keep target separately
    target_series = price_df[["date", target_col]].dropna(subset=[target_col]).reset_index(drop=True)

    # Build base frame from price_df date index, keep price_feat cols + sentiment
    df = pd.DataFrame({"date": price_df["date"].values})
    for c in price_feat.columns:
        df[c] = price_feat[c].values
    df = df.merge(sent_feat, on="date", how="left")
    df[sent_feat_cols] = df[sent_feat_cols].fillna(0.0)

    # Drop rows where target is missing
    df = df.merge(target_series[["date", target_col]], on="date", how="inner")
    df = df.reset_index(drop=True)
    if len(df) < WINDOW + 50:
        return None

    n_price = price_feat.shape[1]
    n_sent = len(sent_feat_cols)
    Xp = df[price_feat.columns.tolist()].values.astype(np.float32)
    Xs = df[sent_feat_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    # Standardize (leak-free: fit on train portion only)
    n_train = int(len(df) * 0.7)
    pm = Xp[:n_train].mean(axis=0)
    ps = Xp[:n_train].std(axis=0) + 1e-9
    Xp = (Xp - pm) / ps
    Xs_mean = Xs[:n_train].mean(axis=0)
    Xs_std = Xs[:n_train].std(axis=0) + 1e-9
    Xs = (Xs - Xs_mean) / Xs_std

    # Sliding windows
    P, S, Y = [], [], []
    for i in range(WINDOW, len(df)):
        P.append(Xp[i - WINDOW:i])
        S.append(Xs[i - WINDOW:i])
        Y.append(y[i])
    return np.array(P), np.array(S), np.array(Y), n_price, n_sent


def train_eval(stock: str, sent_csv: Path, variant: str):
    """Train one MultimodalLSTMEarly model, return test Dir_Acc."""
    data = build_sequences(stock, sent_csv)
    if data is None:
        return None
    P, S, y, n_p, n_s = data
    n = len(P)
    n_train = int(n * 0.7)
    n_val = int(n * 0.85)
    Xp_tr, Xp_va, Xp_te = P[:n_train], P[n_train:n_val], P[n_val:]
    Xs_tr, Xs_va, Xs_te = S[:n_train], S[n_train:n_val], S[n_val:]
    y_tr, y_va, y_te = y[:n_train], y[n_train:n_val], y[n_val:]

    device = torch.device("cpu")  # avoid GPU + numpy issues
    set_seed()
    model = MultimodalLSTMEarly(input_dim_price=n_p, input_dim_sentiment=n_s).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    bs = BATCH
    Xp_tr_t = torch.from_numpy(Xp_tr)
    Xs_tr_t = torch.from_numpy(Xs_tr)
    y_tr_t = torch.from_numpy(y_tr)
    Xp_va_t = torch.from_numpy(Xp_va)
    Xs_va_t = torch.from_numpy(Xs_va)
    y_va_t = torch.from_numpy(y_va)

    best_val = float("inf")
    best_state = None
    patience_left = 3
    n_train_b = len(Xp_tr_t)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_train_b)
        for i in range(0, n_train_b, bs):
            idx = perm[i:i+bs]
            xb_p = Xp_tr_t[idx]
            xb_s = Xs_tr_t[idx]
            yb = y_tr_t[idx]
            opt.zero_grad()
            pred = model(xb_p, xb_s)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
        # Val
        model.eval()
        with torch.no_grad():
            val_preds = model(Xp_va_t, Xs_va_t)
            val_loss = crit(val_preds, y_va_t).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = 3
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test Dir_Acc
    model.eval()
    with torch.no_grad():
        Xp_te_t = torch.from_numpy(Xp_te)
        Xs_te_t = torch.from_numpy(Xs_te)
        preds = model(Xp_te_t, Xs_te_t).numpy()
    dir_acc = float(np.mean(np.sign(preds) == np.sign(y_te))) * 100
    return {"stock": stock, "variant": variant, "dir_acc": dir_acc,
            "best_val_loss": best_val, "n_test": len(y_te)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-stocks", type=int, default=None)
    args = parser.parse_args()

    # Build language-specific sentiment CSVs
    bn_csv = _PROJECT_ROOT / "results" / "sentiment" / "stock_daily_sentiment_bn.csv"
    en_csv = _PROJECT_ROOT / "results" / "sentiment" / "stock_daily_sentiment_en.csv"
    bi_csv = _PROJECT_ROOT / "results" / "sentiment" / "stock_daily_sentiment.csv"
    logger.info("Building language-filtered sentiment CSVs...")
    bn_df = build_lang_sentiment_csv(bn_csv, "bn")
    en_df = build_lang_sentiment_csv(en_csv, "en")
    logger.info(f"  bn: {len(bn_df)} rows   en: {len(en_df)} rows   bi: {bi_csv.stat().st_size}B (existing)")

    # Discover stocks
    csvs = sorted(DATA_DIR.glob("*_processed_v2.csv"))
    stocks = [p.stem.replace("_processed_v2", "") for p in csvs
              if not p.stem.startswith("DSEX")]
    if args.max_stocks:
        stocks = stocks[: args.max_stocks]
    logger.info(f"Running ablation on {len(stocks)} stocks, 3 variants each")

    rows = []
    for variant_name, csv in [("bilingual", bi_csv), ("en_only", en_csv), ("bn_only", bn_csv)]:
        for i, stock in enumerate(stocks, 1):
            logger.info(f"  [{variant_name}] {i}/{len(stocks)} {stock}")
            try:
                r = train_eval(stock, csv, variant_name)
                if r:
                    rows.append(r)
            except Exception as e:
                logger.warning(f"     failed: {e}")

    df = pd.DataFrame(rows)
    out_csv = MULTIMODAL_RESULTS_DIR / "bangla_ablation_results.csv"
    df.to_csv(out_csv, index=False)

    # Summary
    summary = []
    summary.append("=" * 60)
    summary.append("BANGLA-ONLY vs ENGLISH-ONLY vs BILINGUAL ABLATION")
    summary.append("=" * 60)
    summary.append(f"Stocks tested: {df['stock'].nunique()}")
    summary.append(f"Variants: bilingual, en_only, bn_only")
    summary.append("")
    summary.append("Mean test directional accuracy (Dir_Acc %) by variant:")
    summary.append("-" * 60)
    for v in ["bilingual", "en_only", "bn_only"]:
        sub = df[df.variant == v]
        if len(sub):
            summary.append(f"  {v:12s}  mean={sub['dir_acc'].mean():.2f}%  "
                           f"std={sub['dir_acc'].std():.2f}%  n={len(sub)}")
    summary.append("")
    summary.append("KEY FINDINGS")
    summary.append("-" * 60)
    bi = df[df.variant == "bilingual"]["dir_acc"].mean()
    en = df[df.variant == "en_only"]["dir_acc"].mean()
    bn = df[df.variant == "bn_only"]["dir_acc"].mean()
    summary.append(f"  Bilingual beats English-only by: {bi - en:+.2f}pp")
    summary.append(f"  Bilingual beats Bangla-only by:  {bi - bn:+.2f}pp")
    summary.append(f"  Bangla-only beats English-only by: {bn - en:+.2f}pp")
    summary.append("")
    summary.append("Interpretation:")
    if bi > en and bi > bn:
        summary.append("  BILINGUAL fusion is strictly best — combining bn+en helps.")
        summary.append("  This justifies the multilingual design choice of our system.")
    elif abs(bi - en) < 0.5 and bn < en:
        summary.append("  Bilingual ≈ English-only; Bangla adds little on its own.")
        summary.append("  The Bangla contribution is marginal — discuss as limitation.")
    summary.append("=" * 60)

    out_txt = MULTIMODAL_RESULTS_DIR / "bangla_ablation_summary.txt"
    out_txt.write_text("\n".join(summary))
    logger.info(f"\n💾 Saved {out_csv}")
    logger.info(f"💾 Saved {out_txt}")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
