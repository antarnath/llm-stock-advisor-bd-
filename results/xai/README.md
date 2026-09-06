# Phase 8 — Explainable AI (XAI)

## Purpose
Expose what the Phase 3 baseline models actually learned. SHAP (and LIME for cross-method sanity) attribute every prediction to the 27 technical indicators, answering:
- Which features drive predictions most?
- Do different stocks rely on the same signals?
- Is the effect large enough to be meaningful, or is the model essentially noise?

## Files

| File | Rows | Description |
|------|------|-------------|
| `global_importance_per_stock.csv` | 810 | mean |SHAP| per (stock, feature) — 30 stocks × 27 features |
| `aggregate_importance.csv` | 27 | mean |SHAP| aggregated across all 30 stocks |
| `local_explanations_sample.csv` | 1500 | top-10 features for 5 sample predictions per stock (30×5×10) |
| `natural_language_explanations.txt` | 30 | human-readable per-stock sample explanation |
| `lime_top3_stocks.csv` | 30 | LIME top-10 features for GP, BATBC, SQURPHARMA |
| `summary_report.txt` | — | top-5 features + top-1 stability + sample NL + insights |

## Plots (`plots/`)

1. `01_aggregate_importance.png` — top-15 features, signed bars (green=up, red=down)
2. `02_per_stock_heatmap.png` — stock × feature SHAP magnitude heatmap
3. `03_top_feature_per_stock.png` — which feature is #1 for how many stocks
4. `04_signed_effects.png` — mean signed SHAP per feature (direction only)
5. `05_shap_vs_lime.png` — top-1 feature agreement between SHAP and LIME

## Method

For each stock:
1. Load `models/baseline/{STOCK}_best_v2.pkl` (the leak-free Phase 3 model)
2. Auto-detect model type:
   - `LinearRegression` → `shap.LinearExplainer` (correlation_dependent perturbation)
   - `XGBRegressor` / tree ensembles → `shap.TreeExplainer`
   - otherwise → `shap.KernelExplainer` (500 samples)
3. Background: 100 rows sampled from the train split
4. Test: 200 rows sampled from the time-based test split (last 20%)
5. Compute `shap_values` once → aggregate mean |SHAP| for global view, top-10 for local

For 3 representative stocks (GP, BATBC, SQURPHARMA), also run LIME on a single instance with 500 perturbations, and compare top-1 features with SHAP.

## How to reproduce

```bash
.venv/bin/python scripts/phase8_xai.py                # full run, ~30s
.venv/bin/python scripts/phase8_xai.py --max-stocks 5 # smoke test
.venv/bin/python scripts/phase8_xai.py --skip-lime    # SHAP only
```

## Key insights

- **Top global drivers** (mean |SHAP| across 30 stocks):
  `Volume_SMA_20`, `Returns_20d`, `BB_Position`, `Volatility_60d`, `Volume_Ratio`
- **No universal #1**: 10 different features lead for different stocks (Volume_SMA_20 is #1 for only 10/30 stocks).
- **Effect magnitudes are tiny** (≤ 5e-4) — consistent with the ~50% directional accuracy finding from Phases 3-7.
- **SHAP & LIME agree** on the dominant feature category (volume + momentum), validating the conclusions.
