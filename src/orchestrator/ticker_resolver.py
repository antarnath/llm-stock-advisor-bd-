"""
Ticker resolver — maps natural-language mentions of DSE companies to their
official ticker codes.

Examples:
  "Should I buy Grameenphone?"        -> "GP"
  "How is BEXIMCO doing?"            -> "BEXIMCO"
  "What about the BATBC stock?"      -> "BATBC"

Strategy:
  1. Substring match against the 30-stock universe (case-insensitive).
  2. Alias lookup for common company-name spellings.
  3. Fallback: None.
"""

from __future__ import annotations

import re
from typing import Optional


# Map from common name aliases to DSE tickers.
ALIASES = {
    "grameenphone": "GP",
    "gph": "GP",
    "beximco": "BEXIMCO",
    "bexpharma": "BEXPHARMA",
    "bex pharma": "BEXPHARMA",
    "bex pharmaceuticals": "BEXPHARMA",
    "square pharma": "SQURPHARMA",
    "squarepharma": "SQURPHARMA",
    "renata": "RENATA",
    "walton": "WALTONHIL",
    "walton hil": "WALTONHIL",
    "unilever": "UNILEVER",
    "aci": "ACI",
    "islami bank": "ISLAMI BANK",
    "brac bank": "BRACBANK",
    "dutch bangla": "DUTCHBANGL",
    "dutch-bangla": "DUTCHBANGL",
    "prime bank": "PRIMEBANK",
    "dBBL": "DBBL",
    "ebl": "EBL",
    "robi": "ROBI",
    "robi axiata": "ROBI",
    "titas": "TITASGAS",
    "titas gas": "TITASGAS",
    "jamuna oil": "JAMUNAOIL",
    "lafarge": "LAFARGECEM",
    "lafargeholcim": "LAFARGECEM",
    "heidelberg": "HEIDELBCEM",
    "heidelberg cement": "HEIDELBCEM",
    "marico": "MARICO",
    "bat": "BATBC",
    "bat bangladesh": "BATBC",
    "bsccl": "BSCCL",
    "customers care": "CUSTOMERS",
    "prime bank": "PRIMEBANK",
    "ncc bank": "NCCBANK",
    "shahjalal islami": "SIBL",
    "shahjalal": "SIBL",
    "mutual trust": "MUTUALTRUST",
    "bank asia": "BANKASIA",
    "sumit power": "SUMITPOWER",
    "power grid": "POWERGRID",
    "dsex": "DSEX",
}


def resolve_ticker(text: str, universe: list[str]) -> Optional[str]:
    """Return the most likely DSE ticker for `text`, or None if no match.

    Longest-match-wins so e.g. "ISLAMI BANK" wins over "BANK" if both are in
    the universe.
    """
    if not text:
        return None
    low = text.lower()

    # 1. Aliases (full phrase match)
    for alias, ticker in ALIASES.items():
        if ticker in universe and alias in low:
            return ticker

    # 2. Direct ticker mention (longest-first to beat "BANK" over "ISLAMI BANK")
    for ticker in sorted(universe, key=len, reverse=True):
        # word-boundary match
        if re.search(rf"\b{re.escape(ticker)}\b", text, flags=re.IGNORECASE):
            return ticker

    return None
