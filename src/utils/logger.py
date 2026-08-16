"""
Centralized logging configuration.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Training started...")
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from .config import LOGS_DIR, LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL


def get_logger(name: str, log_to_file: bool = False) -> logging.Logger:
    """
    Get a configured logger.

    Args:
        name: Typically __name__ from the calling module
        log_to_file: If True, also write logs to logs/{name}_{timestamp}.log

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_to_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = name.replace(".", "_").replace("/", "_")
        file_handler = logging.FileHandler(LOGS_DIR / f"{safe_name}_{timestamp}.log")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        logger.addHandler(file_handler)

    return logger


__all__ = ["get_logger"]
