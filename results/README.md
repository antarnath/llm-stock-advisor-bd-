# 📊 Results

Output artifacts from experiments, training, and inference.

## Structure

```
results/
├── baseline/             # Phase 3 baseline ML results
│   ├── baseline_results.csv      # Per-stock, per-model metrics
│   ├── predictions_5days.csv     # 5-day forecasts (30 stocks × 5 days)
│   ├── summary_report.txt        # Human-readable summary
│   └── plots/
│       ├── 01_model_comparison.png       # RMSE/R² bar chart
│       ├── 02_per_stock_performance.png  # Per-stock line plots
│       ├── 03_best_model_distribution.png # Which model won where
│       └── 04_metrics_distribution.png   # RMSE/R² histograms
│
└── [future phase folders]
    ├── deep_learning/
    ├── transformers/
    ├── multimodal/
    └── ...
```

## Regenerating Results

```bash
# Phase 3 baseline (full pipeline)
python src/training/baseline_trainer.py   # → baseline_results.csv
python src/inference/predict.py           # → predictions_5days.csv
python src/evaluation/visualize.py        # → plots/*.png + summary_report.txt
```

Or use the end-to-end pipeline:
```bash
python scripts/run_pipeline.py
```

## Gitignore

All generated artifacts (CSVs, PNGs, TXTs, PDFs) are gitignored — they're reproducible from code.

## Latest Results (Phase 3, 2026-08-13)

| Model | Avg RMSE | Avg MAE | Avg MAPE | Avg R² |
|-------|----------|---------|----------|--------|
| **LinearRegression** | **16.47** | 12.87 | 5.07% | **0.8947** |
| RandomForest | 28.03 | 20.80 | 12.73% | 0.2107 |
| XGBoost | 29.43 | 21.80 | 13.36% | 0.1440 |

**Best model distribution:** RF won 17 stocks, LR won 9, XGBoost won 4.
See `baseline/summary_report.txt` for per-stock details.