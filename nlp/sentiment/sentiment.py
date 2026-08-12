"""
Sentiment analysis for PS-87 diary and free-text input.

Uses VADER (Valence Aware Dictionary and sEntiment Reasoner), a lightweight
lexicon/rule-based analyzer that handles negation (e.g. "not happy") without
a large ML model download.

VADER is isolated behind :func:`_compound_score`; swap the backend here only
if the team deliberately chooses a different analyzer.
"""

from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Documented score range for DiaryEntry.sentiment_score and ML features.
SENTIMENT_MIN = -1.0
SENTIMENT_MAX = 1.0
NEUTRAL_SENTIMENT = 0.0

_analyzer: SentimentIntensityAnalyzer | None = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    """Lazy singleton so import side effects are deferred until first use."""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def _compound_score(text: str) -> float:
    """Return the raw VADER compound score for non-empty ``text``."""
    return _get_analyzer().polarity_scores(text)["compound"]


def analyze_sentiment(text: str | None) -> float:
    """
    Compute a deterministic sentiment score for ``text``.

    Score meaning:
    - Range: ``[-1.0, 1.0]`` (VADER compound score, clamped).
    - ``-1.0`` = strongest negative valence.
    - ``0.0`` = neutral (also returned for empty/whitespace-only input).
    - ``+1.0`` = strongest positive valence.

    This is an analytical valence signal, not a mental-health diagnosis.

    Expects caller-provided text (typically after
    :func:`nlp.preprocessing.clean_text.clean_text`). This function only
    checks for empty input; it does not duplicate preprocessing.

    Args:
        text: Diary or free-text string.

    Returns:
        Float sentiment score in ``[-1.0, 1.0]``.
    """
    if text is None or not text.strip():
        return NEUTRAL_SENTIMENT

    compound = _compound_score(text)
    clamped = max(SENTIMENT_MIN, min(SENTIMENT_MAX, compound))
    return round(clamped, 4)
