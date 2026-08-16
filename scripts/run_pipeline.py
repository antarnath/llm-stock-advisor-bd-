#!/usr/bin/env python3
"""
End-to-End Pipeline Runner
Runs all completed phases in sequence with proper error handling.

Phases executed (if data is available):
  Phase 1: Data collection (only if data/raw/ is empty)
  Phase 2: Data processing (only if data/processed/ is empty)
  Phase 3: Baseline ML training, prediction, visualization

Usage:
    python scripts/run_pipeline.py              # run all phases
    python scripts/run_pipeline.py --skip-train # skip training, just visualize
    python scripts/run_pipeline.py --phase 3    # run only phase 3
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    RAW_STOCKS_DIR, PROCESSED_DATA_DIR, BASELINE_MODELS_DIR,
    BASELINE_RESULTS_DIR, ensure_dirs,
)
from src.utils.logger import get_logger

logger = get_logger("pipeline")


def run_step(label: str, script_path: str) -> bool:
    """Run a Python script as a subprocess and return success status."""
    logger.info(f"\n{'=' * 70}\n▶ {label}\n{'=' * 70}")
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=False,
        )
        logger.info(f"✅ {label} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {label} failed with exit code {e.returncode}")
        return False


def phase_1_collect():
    """Phase 1: Collect DSE stock data (skip if already collected)"""
    existing = list(RAW_STOCKS_DIR.glob("*.csv"))
    if len(existing) >= 30:
        logger.info(f"Phase 1: Already have {len(existing)} stock files. Skipping.")
        return True
    return run_step(
        "Phase 1: Collect DSE Stock Data",
        "src/data_collection/collect_stocks.py",
    )


def phase_2_process():
    """Phase 2: Process data and add technical indicators (skip if processed)"""
    existing = list(PROCESSED_DATA_DIR.glob("*_processed.csv"))
    if len(existing) >= 30:
        logger.info(f"Phase 2: Already have {len(existing)} processed files. Skipping.")
        return True
    return run_step(
        "Phase 2: Process Data & Add Technical Indicators",
        "src/data_processing/technical_indicators.py",
    )


def phase_3_baseline(skip_train: bool = False):
    """Phase 3: Train, predict, visualize"""
    steps = []
    if not skip_train:
        # Train only if no models exist
        existing_models = list(BASELINE_MODELS_DIR.glob("*_best.pkl"))
        if len(existing_models) < 30:
            steps.append(("Phase 3a: Train Baseline Models", "src/training/baseline_trainer.py"))
        else:
            logger.info(f"Phase 3a: Already have {len(existing_models)} trained models. Skipping.")
    steps.append(("Phase 3b: Generate Predictions", "src/inference/predict.py"))
    steps.append(("Phase 3c: Visualize Results", "src/evaluation/visualize.py"))

    for label, script in steps:
        if not run_step(label, script):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the full pipeline")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], help="Run only a specific phase")
    parser.add_argument("--skip-train", action="store_true", help="Skip training (use existing models)")
    parser.add_argument("--force", action="store_true", help="Force re-run all phases")
    args = parser.parse_args()

    ensure_dirs()

    logger.info("""
╔══════════════════════════════════════════════════════════════╗
║  LLM Financial Advisor - End-to-End Pipeline                 ║
║  Project: Bangladesh Stock Market                            ║
╚══════════════════════════════════════════════════════════════╝
    """)

    success = True

    if args.phase is None or args.phase == 1:
        success = phase_1_collect() and success
    if args.phase is None or args.phase == 2:
        success = phase_2_process() and success
    if args.phase is None or args.phase == 3:
        success = phase_3_baseline(skip_train=args.skip_train) and success

    if success:
        logger.info("\n" + "=" * 70)
        logger.info("✨ Pipeline completed successfully!")
        logger.info(f"📊 Results: {BASELINE_RESULTS_DIR}")
        logger.info("=" * 70)
        sys.exit(0)
    else:
        logger.error("\n❌ Pipeline failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()