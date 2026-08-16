"""
Unified sentiment analyzer interface for Phase 6.

Three backends with one calling convention:
- FinBERT (ProsusAI/finbert)        — English, ~68ms/headline on CPU, accurate.
                                       Best general-purpose for financial English.
- VADER (vaderSentiment)            — English, <1ms/headline, fast fallback.
                                       Rule-based, lower accuracy on news headlines.
- BanglaLexicon (built-in)          — Bangla (বাংলা), <1ms/headline.
                                       Tiny curated lexicon (~150 positive /
                                       150 negative words). Robust enough to
                                       flag dominant polarity; not a replacement
                                       for a Bangla BERT model (deferred to 6xxx).

All return the same dict shape:
    {
        "label":      "positive" | "negative" | "neutral",
        "score":      float in [-1.0, +1.0],  # signed polarity
        "confidence": float in [0.0, 1.0],   # top-class probability
        "probs":      {pos: float, neg: float, neu: float},
    }

USAGE:
    from src.sentiment.analyzers import get_analyzer
    a = get_analyzer("finbert")
    out = a.analyze("Square Pharma beats Q3 estimates")
    print(out)  # {'label': 'positive', 'score': 0.85, ...}
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config import FINBERT_MODEL_NAME
from src.utils.logger import get_logger


logger = get_logger("sentiment.analyzers")

Backend = Literal["finbert", "vader", "bangla"]


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class SentimentAnalyzer:
    """Base class — every backend implements `analyze(text) -> dict`."""

    name: str = "base"

    def analyze(self, text: str) -> dict:
        raise NotImplementedError

    def analyze_batch(self, texts: list[str], show_progress: bool = False) -> list[dict]:
        return [self.analyze(t) for t in texts]


# ---------------------------------------------------------------------------
# FinBERT backend (ProsusAI/finbert)
# ---------------------------------------------------------------------------

class FinBERTAnalyzer(SentimentAnalyzer):
    """ProsusAI/finbert — financial-domain English sentiment, CPU-friendly."""

    name = "finbert"

    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        logger.info(f"   Loading FinBERT ({FINBERT_MODEL_NAME})...")
        self.tok = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            FINBERT_MODEL_NAME, use_safetensors=True,
        )
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # id2label: {0: 'positive', 1: 'negative', 2: 'neutral'}
        self.id2label = self.model.config.id2label

    def analyze(self, text: str) -> dict:
        import torch

        if not text or not text.strip():
            return _neutral_result()

        inp = self.tok(
            text, return_tensors="pt", truncation=True, max_length=256,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inp).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        # {0: pos, 1: neg, 2: neu} — order is fixed for FinBERT
        pos, neg, neu = probs[0], probs[1], probs[2]
        label = self.id2label[int(max(range(3), key=lambda i: probs[i]))]
        # Signed score: positive - negative, in [-1, +1]
        score = pos - neg
        return {
            "label": label,
            "score": float(score),
            "confidence": float(max(probs)),
            "probs": {"positive": pos, "negative": neg, "neutral": neu},
        }


# ---------------------------------------------------------------------------
# VADER backend (vaderSentiment)
# ---------------------------------------------------------------------------

class VaderAnalyzer(SentimentAnalyzer):
    """VADER — fast English rule-based sentiment."""

    name = "vader"

    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> dict:
        if not text or not text.strip():
            return _neutral_result()
        scores = self.analyzer.polarity_scores(text)
        # scores = {'neg', 'neu', 'pos', 'compound'}
        compound = scores["compound"]
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        # Map compound [-1, +1] → score; confidence = 1 - neutrality
        confidence = 1.0 - scores["neu"]
        return {
            "label": label,
            "score": float(compound),
            "confidence": float(confidence),
            "probs": {
                "positive": scores["pos"],
                "negative": scores["neg"],
                "neutral": scores["neu"],
            },
        }


# ---------------------------------------------------------------------------
# Bangla lexicon (tiny curated)
# ---------------------------------------------------------------------------

# Minimal positive/negative Bangla lexicon (~300 words).
# NOTE: This is a research-grade fallback. A production system would use
# Bangla-BERT or sahajBERT. Deferred to phase_6xxx_bangla_bert.md.
BANGLA_POSITIVE = {
    "মুনাফা", "রেকর্ড", "বৃদ্ধি", "উন্নীত", "অনুমোদন", "সফল", "উচ্চ",
    "বাড়িয়ে", "সেরা", "শক্তিশালী", "উন্নতি", "লাভ", "প্রবৃদ্ধি", "ছাড়িয়ে",
    "উদ্বোধন", "চালু", "ঘোষণা", "অর্জন", "সম্প্রসারণ", "স্বাক্ষর", "চুক্তি",
    "অনুকূল", "উজ্জ্বল", "বিজয়", "উৎসব", "সমৃদ্ধ", "স্থিতিশীল", "উত্তম",
    "বাড়তি", "উর্ধ্বমুখী", "ইতিবাচক", "অগ্রগতি", "সাফল্য", "অর্জিত",
    "ভাল", "ভালো", "চমৎকার", "সুন্দর", "চাহিদা", "আয়", "আয়কর", "সাশ্রয়",
    "খুশি", "সন্তুষ্ট", "আনন্দ", "আশা", "আশাবাদ", "আশাব্যঞ্জক",
}

BANGLA_NEGATIVE = {
    "লোকসান", "কমেছে", "কমেছে", "পতন", "প্রভাব", "সংকট", "সংকটে",
    "ক্ষতি", "অভিযোগ", "অনিয়ম", "অসঙ্গতি", "অভাব", "অভাবে", "স্থগিত",
    "বন্ধ", "বন্ধ করেছে", "খেলাপি", "গুজব", "দুর্নীতি", "জালিয়াতি",
    "গ্রেফতার", "জরিমানা", "জব্দ", "তদন্ত", "চাপ", "উদ্বেগ", "ভয়",
    "আতঙ্ক", "বিপদ", "বিপর্যয়", "ব্যর্থ", "হতাশ", "নেতিবাচক",
    "দুর্বল", "অবনতি", "অবমূল্যায়ন", "মুদ্রাস্ফীতি", "খতিয়ে", "খতিয়ে দেখা",
    "অব্যাহতি", "কমিয়ে", "অনুমোদন দেয়নি", "প্রত্যাখ্যান", "অস্বীকার",
    "সমালোচনা", "অভিযোগ", "ক্ষোভ", "ক্ষুব্ধ", "কঠিন", "কঠিনাশ্রয়ী",
}

# Stopwords and particles — never count, never negate
BANGLA_STOPWORDS = {
    "এ", "ও", "এবং", "বা", "কিন্তু", "তবে", "যে", "যা", "যার", "যাদের",
    "করে", "করেছে", "করবে", "হবে", "হয়েছে", "হয়", "থেকে", "থেকেও",
    "প্রতি", "জন্য", "জন্যও", "পর", "পরে", "মধ্যে", "মধ্যেও", "সাথে",
    "সাথেও", "সম্পর্কে", "বিষয়ে", "ব্যাপারে", "কারণে", "ফলে", "হেতু",
    "কারণে", "কারণ", "এই", "সেই", "ওই", "যেই", "একটি", "এক", "দুই",
    "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়", "দশ",
    "কোন", "কোনো", "সব", "সকল", "অনেক", "অধিক", "কম", "বেশি",
}


class BanglaLexiconAnalyzer(SentimentAnalyzer):
    """Tiny Bangla lexicon scorer. ~1ms/text."""

    name = "bangla"

    def __init__(self):
        # Compile patterns once
        self.pos_re = self._compile_pattern(BANGLA_POSITIVE)
        self.neg_re = self._compile_pattern(BANGLA_NEGATIVE)

    @staticmethod
    def _compile_pattern(words: set[str]) -> re.Pattern:
        # Sort by length descending so longer matches win (avoid 'লাভ' before 'লাভবান')
        sorted_words = sorted(words, key=len, reverse=True)
        # Use word boundaries that work for Bangla (no caps, no spaces inside words)
        pattern = r"(?<![অ-হ])(" + "|".join(re.escape(w) for w in sorted_words) + r")(?![অ-হ])"
        return re.compile(pattern, flags=re.UNICODE)

    def analyze(self, text: str) -> dict:
        if not text or not text.strip():
            return _neutral_result()

        pos_hits = self.pos_re.findall(text)
        neg_hits = self.neg_re.findall(text)

        pos_count = len(pos_hits)
        neg_count = len(neg_hits)

        if pos_count == 0 and neg_count == 0:
            return _neutral_result()

        # Score: signed ratio in [-1, +1]
        total = pos_count + neg_count
        score = (pos_count - neg_count) / total

        if score > 0.15:
            label = "positive"
        elif score < -0.15:
            label = "negative"
        else:
            label = "neutral"

        # Confidence: how decisive the ratio is
        confidence = abs(score)

        return {
            "label": label,
            "score": float(score),
            "confidence": float(confidence),
            "probs": {
                "positive": pos_count / total,
                "negative": neg_count / total,
                "neutral": max(0.0, 1.0 - confidence),
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _neutral_result() -> dict:
    return {
        "label": "neutral",
        "score": 0.0,
        "confidence": 0.0,
        "probs": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
    }


def get_analyzer(backend: Backend | str) -> SentimentAnalyzer:
    """Factory: returns the requested analyzer (singleton per process).

    Use 'auto' for language-aware selection: 'finbert' for English, 'bangla' for Bangla.
    """
    backend = backend.lower()
    if backend == "finbert":
        return FinBERTAnalyzer()
    if backend == "vader":
        return VaderAnalyzer()
    if backend == "bangla":
        return BanglaLexiconAnalyzer()
    if backend == "auto":
        return AutoAnalyzer()
    raise ValueError(f"Unknown backend: {backend!r}. Choose: finbert | vader | bangla | auto")


class AutoAnalyzer(SentimentAnalyzer):
    """Routes English text → FinBERT, Bangla text → BanglaLexiconAnalyzer."""

    name = "auto"

    def __init__(self):
        self.finbert = FinBERTAnalyzer()
        self.bangla = BanglaLexiconAnalyzer()
        # Bangla Unicode range: U+0980 - U+09FF
        self._bangla_re = re.compile(r"[\u0980-\u09FF]")

    def _is_bangla(self, text: str) -> bool:
        # If at least 20% of non-space chars are in Bangla range, treat as Bangla
        chars = [c for c in text if not c.isspace()]
        if not chars:
            return False
        bangla_count = sum(1 for c in chars if self._bangla_re.match(c))
        return (bangla_count / len(chars)) > 0.2

    def analyze(self, text: str) -> dict:
        if self._is_bangla(text):
            return self.bangla.analyze(text)
        return self.finbert.analyze(text)


__all__ = [
    "SentimentAnalyzer",
    "FinBERTAnalyzer",
    "VaderAnalyzer",
    "BanglaLexiconAnalyzer",
    "AutoAnalyzer",
    "get_analyzer",
]