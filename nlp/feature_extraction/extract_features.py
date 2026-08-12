"""
NLP feature extraction for the PS-87 ML feature pipeline.

Orchestrates preprocessing, sentiment, and emotion modules into a single
internal feature dict.

IMPORTANT — this is NOT a locked public JSON contract (see docs/CONTRACTS.md).
``NLPFeatures`` is an internal handoff proposal for the ML owner. Field names
may change after team agreement; do not treat this TypedDict as a public API.
"""

from __future__ import annotations

from typing import TypedDict

from ..emotion.emotion import EMOTION_LABELS, tag_emotion
from ..preprocessing.clean_text import clean_text
from ..sentiment.sentiment import analyze_sentiment

# ml/src/features/feature_pipeline.py is still a placeholder (2026-08).


class NLPFeatures(TypedDict):
    """
    Internal NLP → ML feature bundle (proposal, not a locked contract).

    Fields:
    - ``sentiment_score``: VADER compound in ``[-1.0, 1.0]``.
    - ``emotion_tags``: sorted unique labels from :func:`tag_emotion`.
    - ``emotion_count``: ``len(emotion_tags)``.
    - ``word_count`` / ``char_count``: basic text statistics on cleaned input.
    - ``emotion_flags``: ``{label: 0|1}`` for every :data:`EMOTION_LABELS` entry.
    """

    sentiment_score: float
    emotion_tags: list[str]
    emotion_count: int
    word_count: int
    char_count: int
    emotion_flags: dict[str, int]


def _emotion_flags(tags: list[str]) -> dict[str, int]:
    present = set(tags)
    return {label: int(label in present) for label in EMOTION_LABELS}


def extract_features(text: str | None) -> NLPFeatures:
    """
    Build NLP features from raw diary/free-text input.

    Pipeline::

        clean_text → analyze_sentiment + tag_emotion → feature dict

    Args:
        text: Raw diary entry text.

    Returns:
        :class:`NLPFeatures` dict. Deterministic for the same input.
    """
    cleaned = clean_text(text)
    sentiment = analyze_sentiment(cleaned)
    emotions = tag_emotion(cleaned)
    words = cleaned.split() if cleaned else []

    return NLPFeatures(
        sentiment_score=sentiment,
        emotion_tags=emotions,
        emotion_count=len(emotions),
        word_count=len(words),
        char_count=len(cleaned),
        emotion_flags=_emotion_flags(emotions),
    )
