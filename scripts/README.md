# 🛠️ Scripts

Ad-hoc, debug, and utility scripts. Production code lives in [`src/`](../src/).

## Structure

```
scripts/
├── README.md              # ← you are here
├── run_pipeline.py        # End-to-end pipeline runner
└── debug/                 # One-off debug scripts (kept for reproducibility)
    ├── check_data_gaps.py
    ├── detailed_gap_analysis.py
    ├── train_quick_test.py
    └── verify_data_integrity.py
```

## When to Use

- **`run_pipeline.py`** — Run all phases end-to-end (use this most often)
- **`debug/`** — Scripts used during development to verify data, debug issues, etc.
  - These are intentionally kept around for reproducibility and re-runs

## Running

From project root:
```bash
# Run full pipeline
python scripts/run_pipeline.py

# Run specific phase only
python scripts/run_pipeline.py --phase 3

# Skip training (use existing models)
python scripts/run_pipeline.py --skip-train

# Run a debug script directly
python scripts/debug/check_data_gaps.py
```

## Adding New Debug Scripts

Drop them in `debug/` with a clear docstring explaining:
1. What the script does
2. Why it was created
3. Expected output
4. When it can be removed (or "keep forever for reproducibility")