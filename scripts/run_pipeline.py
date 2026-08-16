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
    BASELINE_RESULTS_DIR, DEEP_LEARNING_MODELS_DIR, DL_RESULTS_DIR,
    SENTIMENT_RESULTS_DIR, MULTIMODAL_MODELS_DIR, MULTIMODAL_RESULTS_DIR,
    ensure_dirs,
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


def phase_4_deep_learning(skip_train: bool = False):
    """Phase 4: Train LSTM, predict, visualize."""
    steps = []
    if not skip_train:
        existing_models = list(DEEP_LEARNING_MODELS_DIR.glob("*_best_lstm.pt"))
        if len(existing_models) < 30:
            steps.append((
                "Phase 4a: Train LSTM (Deep Learning)",
                "src/training/deep_learning_trainer.py",
            ))
        else:
            logger.info(
                f"Phase 4a: Already have {len(existing_models)} LSTM checkpoints. Skipping."
            )
    steps.append(("Phase 4b: Generate LSTM Predictions", "src/inference/dl_predict.py"))
    steps.append(("Phase 4c: Visualize LSTM Results", "src/evaluation/dl_visualize.py"))

    for label, script in steps:
        if not run_step(label, script):
            return False
    return True


def phase_6_sentiment():
    """Phase 6: Sentiment Analysis — curate news, score, correlate, visualize."""
    steps = [
        ("Phase 6a: Curate News Dataset", "src/data_collection/news_curator.py"),
        ("Phase 6b: Score Articles (FinBERT/Bangla)", "src/sentiment/scoring_pipeline.py"),
        ("Phase 6c: Sentiment-Price Correlation", "src/sentiment/correlation_analysis.py"),
        ("Phase 6d: Visualize Sentiment Results", "src/sentiment/visualize.py"),
    ]
    for label, script in steps:
        if not run_step(label, script):
            return False
    return True


def phase_7_multimodal(skip_train: bool = False):
    """Phase 7: Multimodal LSTM (early + late fusion) — train, predict, visualize."""
    steps = []
    if not skip_train:
        # 30 stocks × 2 fusions = 60 checkpoints
        existing = list(MULTIMODAL_MODELS_DIR.glob("*_best_mm_*.pt"))
        if len(existing) < 60:
            steps.append((
                "Phase 7a: Train Multimodal LSTM (early + late fusion)",
                "src/training/multimodal_trainer.py",
            ))
        else:
            logger.info(
                f"Phase 7a: Already have {len(existing)} multimodal checkpoints. Skipping."
            )
    steps.append(("Phase 7b: Generate Multimodal Predictions",
                  "src/inference/mm_predict.py"))
    steps.append(("Phase 7c: Visualize Multimodal Results (with ablation)",
                  "src/evaluation/mm_visualize.py"))

    for label, script in steps:
        if not run_step(label, script):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the full pipeline")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5, 6, 7], help="Run only a specific phase")
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
    if args.phase is None or args.phase == 4:
        success = phase_4_deep_learning(skip_train=args.skip_train) and success
    if args.phase is None or args.phase == 6:
        success = phase_6_sentiment() and success
    if args.phase is None or args.phase == 7:
        success = phase_7_multimodal(skip_train=args.skip_train) and success

    if success:
        logger.info("\n" + "=" * 70)
        logger.info("✨ Pipeline completed successfully!")
        logger.info(f"📊 Results: {BASELINE_RESULTS_DIR}, {DL_RESULTS_DIR}, "
                    f"{SENTIMENT_RESULTS_DIR}, {MULTIMODAL_RESULTS_DIR}")
        logger.info("=" * 70)
        sys.exit(0)
    else:
        logger.error("\n❌ Pipeline failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()