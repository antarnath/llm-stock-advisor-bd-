# Report Directory — `report/`

This directory contains the **live-maintained thesis report** for the
*LLM-Orchestrated Financial Advisor for the Bangladesh Stock Market*
project.

## Contents

```
report/
├── main.tex                 # The thesis source (build with pdflatex).
├── README.md                # ← you are here.
├── REPORT_MAINTENANCE.md    # Maintenance protocol for AI agents + humans.
├── CHANGELOG.md             # Per-change history of the report.
└── figures/                 # PNG plots referenced by main.tex.
```

## Build

```bash
# Local
bash scripts/report/build_report.sh

# Docker (if pdflatex is not installed)
docker run --rm -v "$(pwd)/..":/src -w /src/report \
    texlive/texlive:latest pdflatex -interaction=nonstopmode main.tex
```

## Source of truth

Every quantitative claim in `main.tex` is sourced from one of:

- `results/baseline/baseline_results_v2.csv`
- `results/deep_learning/deep_learning_results.csv`
- `results/multimodal/multimodal_results.csv`
- `results/sentiment/correlation_summary.json`
- The corresponding `summary_report.txt` files.

If the report disagrees with the artifact, **the artifact wins** and
the report is updated. See `REPORT_MAINTENANCE.md` for the protocol.

## Helper scripts

| Script | Purpose |
|---|---|
| `scripts/report/phase_summary.py` | Print aggregate metrics per phase; copy-paste directly into the report. |
| `scripts/report/sync_figures.sh` | Re-copy all source plots into `figures/` with stable names. |
| `scripts/report/build_report.sh` | Two-pass `pdflatex` build with strict reference checking. |
| `scripts/report/check_report_aligned.sh` | Pre-commit guard: refuse a commit that touches `results/` without touching `main.tex`. |