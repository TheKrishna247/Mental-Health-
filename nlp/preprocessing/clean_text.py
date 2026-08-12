"""
Conservative text normalization for PS-87 NLP modules.

Shared preprocessing before sentiment analysis, emotion tagging, and feature
extraction. Preserves negation and meaningful emotional language.
"""

from __future__ import annotations

import re
import unicodedata

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")
_MULTI_PUNCT_PATTERN = re.compile(r"([!?.,;:\-])\1{2,}")
_WHITESPACE_PATTERN = re.compile(r"\s+")

# Normalize common Unicode punctuation to ASCII equivalents without altering words.
_UNICODE_REPLACEMENTS = str.maketrans(
    {
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2026": "...",  # ellipsis
    }
)


def clean_text(text: str | None) -> str:
    """
    Normalize diary/free-text input conservatively.

    Steps:
    - Treat ``None`` like empty input.
    - Apply Unicode NFC normalization and common punctuation normalization.
    - Strip leading/trailing whitespace.
    - Remove URLs and email addresses (noise, not emotional content).
    - Collapse repeated punctuation (``!!!`` → ``!``) without removing negation.
    - Collapse internal runs of whitespace to a single space.

    Does **not** remove stop words, negation words, stemming/lemmatization,
    or emoji. Mixed-script student diary text (e.g. Hinglish) is preserved.

    Args:
        text: Raw user text.

    Returns:
        Cleaned text, or ``""`` for empty/whitespace-only input.
    """
    if text is None:
        return ""

    cleaned = unicodedata.normalize("NFC", text).translate(_UNICODE_REPLACEMENTS)
    cleaned = cleaned.strip()
    if not cleaned:
        return ""

    cleaned = _URL_PATTERN.sub(" ", cleaned)
    cleaned = _EMAIL_PATTERN.sub(" ", cleaned)
    cleaned = _MULTI_PUNCT_PATTERN.sub(r"\1", cleaned)
    cleaned = _WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned
