"""
Centralized configuration and project paths.
All scripts should import paths from here instead of computing them locally.

Usage:
    from src.utils.config import DATA_DIR, MODELS_DIR
    df = pd.read_csv(DATA_DIR / "raw" / "stocks" / "ACI.csv")
"""

from pathlib import Path
import os


# Project root: Dataset/
# This file is at: src/utils/config.py
# So root = parents[2] = Dataset/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_STOCKS_DIR = RAW_DATA_DIR / "stocks"
RAW_INDICES_DIR = RAW_DATA_DIR / "indices"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_MODELS_DIR = MODELS_DIR / "baseline"
DEEP_LEARNING_MODELS_DIR = MODELS_DIR / "deep_learning"
TRANSFORMER_MODELS_DIR = MODELS_DIR / "transformers"
EXPERIMENT_MODELS_DIR = MODELS_DIR / "experiments"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
BASELINE_RESULTS_DIR = RESULTS_DIR / "baseline"
BASELINE_PLOTS_DIR = BASELINE_RESULTS_DIR / "plots"

# Logs directory
LOGS_DIR = PROJECT_ROOT / "logs"

# Documentation
DOCS_DIR = PROJECT_ROOT / "docs"
PHASES_DIR = DOCS_DIR / "phases"

# Source code
SRC_DIR = PROJECT_ROOT / "src"

# Tests
TESTS_DIR = PROJECT_ROOT / "tests"

# Notebooks
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Top 30 DSE stocks (canonical list, single source of truth)
TOP_30_DSE_STOCKS = [
    "ACI", "BANKASIA", "BATBC", "BEXIMCO", "BEXPHARMA",
    "BRACBANK", "BSCCL", "CUSTOMERS", "DBBL", "DSEX",
    "DUTCHBANGL", "EBL", "GP", "HEIDELBCEM", "ISLAMI BANK",
    "JAMUNAOIL", "LAFARGECEM", "MARICO", "MUTUALTRUST", "NCCBANK",
    "POWERGRID", "PRIMEBANK", "RENATA", "ROBI", "SIBL",
    "SQURPHARMA", "SUMITPOWER", "TITASGAS", "UNILEVER", "WALTONHIL",
]

# Market indices
INDICES = ["DSEX", "DS30", "DSES"]

# Technical indicator parameters
SMA_WINDOWS = [5, 10, 20, 50, 200]
EMA_WINDOWS = [12, 26, 50]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
ATR_PERIOD = 14

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Deep learning (Phase 4)
DL_SEQUENCE_LENGTH = 60
DL_HIDDEN_DIM = 128
DL_NUM_LAYERS = 3
DL_DROPOUT = 0.2
DL_LEARNING_RATE = 1e-3
DL_BATCH_SIZE = 64
DL_EPOCHS = 100
DL_PATIENCE = 20  # early stopping
DL_TRAIN_VAL_SPLIT = 0.1  # % of train hold-out for validation during training
DL_RESULTS_DIR = RESULTS_DIR / "deep_learning"
DL_PLOTS_DIR = DL_RESULTS_DIR / "plots"

# Sentiment Analysis (Phase 6)
FINBERT_MODEL_NAME = "ProsusAI/finbert"   # ~440MB, CPU-friendly
SENTIMENT_RESULTS_DIR = RESULTS_DIR / "sentiment"
SENTIMENT_PLOTS_DIR = SENTIMENT_RESULTS_DIR / "plots"

# Multimodal Forecasting (Phase 7) — price + sentiment
MULTIMODAL_MODELS_DIR = MODELS_DIR / "multimodal"
MULTIMODAL_RESULTS_DIR = RESULTS_DIR / "multimodal"
MULTIMODAL_PLOTS_DIR = MULTIMODAL_RESULTS_DIR / "plots"
MM_SEQUENCE_LENGTH = DL_SEQUENCE_LENGTH   # 60, reuse price window length
MM_EPOCHS = 50                            # lower than DL_EPOCHS=100 (sentiment is sparse)
MM_PATIENCE = 7
MM_BATCH_SIZE = DL_BATCH_SIZE             # 64
MM_LEARNING_RATE = DL_LEARNING_RATE       # 1e-3
MM_FILL_NA = 0.0                          # defensive, after forward-fill residual
MM_SENTIMENT_COLS = [
    "n_articles", "mean_score", "weighted_score", "mean_confidence",
    "pos_count", "neg_count", "neu_count",
]

# Logging
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def get_device():
    """Return torch device (CPU/CUDA) once torch is installed.

    Lazy import so this module doesn't break if torch isn't installed
    yet (e.g. during Phase 3 baseline training).
    """
    try:
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        return None


def get_raw_stock_path(stock_code: str) -> Path:
    """Get path to raw stock CSV"""
    return RAW_STOCKS_DIR / f"{stock_code}.csv"


def get_processed_stock_path(stock_code: str) -> Path:
    """Get path to processed stock CSV"""
    return PROCESSED_DATA_DIR / f"{stock_code}_processed.csv"


def get_baseline_model_path(stock_code: str) -> Path:
    """Get path to baseline model pickle"""
    return BASELINE_MODELS_DIR / f"{stock_code}_best.pkl"


def ensure_dirs():
    """Ensure all required directories exist (idempotent)"""
    for d in [
        RAW_STOCKS_DIR, RAW_INDICES_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
        BASELINE_MODELS_DIR, DEEP_LEARNING_MODELS_DIR, TRANSFORMER_MODELS_DIR, EXPERIMENT_MODELS_DIR,
        BASELINE_RESULTS_DIR, BASELINE_PLOTS_DIR, DL_RESULTS_DIR, DL_PLOTS_DIR,
        SENTIMENT_RESULTS_DIR, SENTIMENT_PLOTS_DIR,
        MULTIMODAL_MODELS_DIR, MULTIMODAL_RESULTS_DIR, MULTIMODAL_PLOTS_DIR,
        LOGS_DIR, PHASES_DIR, TESTS_DIR, NOTEBOOKS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    ensure_dirs()
    print("✅ All directories exist")
