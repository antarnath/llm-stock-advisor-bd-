"""
Freshness checker — reports how stale the current news + predictions are.

Returns:
  - ran_at: timestamp of last successful news refresh
  - rows_added: number of articles added in the last refresh
  - stocks_updated: list of stocks with new news in the last refresh
  - status: "ok" | "stale" | "unknown" | "broken"
  - age_hours: hours since last refresh
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# Path to the on-disk refresh log (JSONL).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFRESH_FILE: Path = PROJECT_ROOT / "data" / "external" / "refresh_log.json"


def freshness(refresh_log_path: Optional[Path] = None) -> dict:
    """Compute freshness status from the last refresh log entry.

    Returns a dict with keys: status, age_hours, label, ran_at, rows_added,
    stocks_updated.
    """
    path = refresh_log_path or REFRESH_FILE
    if not path.exists():
        return {
            "ran_at": None,
            "rows_added": None,
            "stocks_updated": [],
            "status": "unknown",
            "age_hours": None,
            "label": "No refresh recorded yet.",
        }

    try:
        # File is JSONL; take the last entry
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        if not lines:
            return _unknown("empty log file")
        last = json.loads(lines[-1])
        ran_at = last.get("ran_at")
        if not ran_at:
            return _unknown("no timestamp in last entry")

        ran_dt = datetime.fromisoformat(ran_at.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        age_h = (now - ran_dt).total_seconds() / 3600.0

        # Status thresholds
        if age_h < 24:
            status = "ok"
        elif age_h < 72:
            status = "stale"
        else:
            status = "broken"

        return {
            "ran_at": ran_at,
            "rows_added": int(last.get("rows_added", 0) or 0),
            "stocks_updated": list(last.get("stocks_updated", []) or []),
            "status": status,
            "age_hours": float(age_h),
            "label": f"Last refresh {age_h:.1f}h ago",
        }
    except Exception as e:
        return {
            "ran_at": None,
            "rows_added": None,
            "stocks_updated": [],
            "status": "broken",
            "age_hours": None,
            "label": f"Could not read refresh log: {e}",
        }


def read_freshness_log(refresh_log_path: Optional[Path] = None) -> dict:
    """Alias kept for backwards-compat with earlier callers."""
    return freshness(refresh_log_path=refresh_log_path)


def _unknown(reason: str) -> dict:
    return {
        "ran_at": None,
        "rows_added": None,
        "stocks_updated": [],
        "status": "unknown",
        "age_hours": None,
        "label": reason,
    }