"""
Lexicon-based emotion tagging for PS-87 diary and free-text input.

Maps text to descriptive emotion labels (signals, not clinical diagnoses).

Matching strategy (deterministic, explainable):
1. Multi-word phrases are matched first (longest phrase wins; no overlap).
2. Remaining single-token lexicon matches are applied.
3. Any match preceded by a negation word within a short token window is
   excluded — we tag affirmed emotional language only, not negated mentions.
"""

from __future__ import annotations

import re
from typing import Final

# Stable label set used across emotion tagging and feature extraction.
EMOTION_LABELS: Final[tuple[str, ...]] = (
    "joy",
    "sadness",
    "anger",
    "fear",
    "anxiety",
    "loneliness",
    "exhaustion",
    "overwhelm",
    "calm",
    "guilt",
    "hope",
)

_NEGATION_WINDOW = 3

_NEGATION_WORDS: Final[frozenset[str]] = frozenset(
    {
        "not",
        "no",
        "never",
        "none",
        "nobody",
        "nothing",
        "nowhere",
        "neither",
        "nor",
        "without",
        "hardly",
        "barely",
        "scarcely",
        "cannot",
        "can't",
        "cant",
        "won't",
        "wont",
        "wouldn't",
        "wouldnt",
        "shouldn't",
        "shouldnt",
        "couldn't",
        "couldnt",
        "don't",
        "dont",
        "doesn't",
        "doesnt",
        "didn't",
        "didnt",
        "isn't",
        "isnt",
        "aren't",
        "arent",
        "wasn't",
        "wasnt",
        "weren't",
        "werent",
        "ain't",
        "aint",
    }
)

# Phrase → label. Longest phrases are matched first to avoid partial overlaps.
EMOTION_PHRASES: Final[dict[str, str]] = {
    # Exhaustion / burnout
    "burned out": "exhaustion",
    "burnt out": "exhaustion",
    "wiped out": "exhaustion",
    # Anxiety
    "stressed out": "anxiety",
    "on edge": "anxiety",
    # Anger
    "fed up": "anger",
    # Overwhelm
    "too much": "overwhelm",
    # Calm
    "at ease": "calm",
    "feel calm": "calm",
    "feeling calm": "calm",
    # Hope
    "looking forward": "hope",
    # Loneliness — contextual (avoids bare "alone" false positives)
    "feel alone": "loneliness",
    "feeling alone": "loneliness",
    "so alone": "loneliness",
    "all alone": "loneliness",
    "left alone": "loneliness",
    # Sadness — contextual (avoids bare "down"/"low"/"empty")
    "feel down": "sadness",
    "feeling down": "sadness",
    "so down": "sadness",
    "feel low": "sadness",
    "feeling low": "sadness",
    "feel empty": "sadness",
    "feeling empty": "sadness",
    # Joy — contextual (avoids bare "great"/"amazing")
    "feel great": "joy",
    "feeling great": "joy",
}

# Single-token lexicon. Ambiguous tokens (alone, down, low, empty, great, amazing)
# are intentionally omitted; those emotions use contextual phrases above.
EMOTION_LEXICON: Final[dict[str, tuple[str, ...]]] = {
    "joy": (
        "happy",
        "joyful",
        "joy",
        "excited",
        "glad",
        "delighted",
        "cheerful",
        "grateful",
        "thankful",
        "content",
        "pleased",
        "thrilled",
        "wonderful",
    ),
    "sadness": (
        "sad",
        "unhappy",
        "miserable",
        "heartbroken",
        "grief",
        "grieving",
        "tearful",
        "crying",
        "hopeless",
    ),
    "anger": (
        "angry",
        "mad",
        "furious",
        "irritated",
        "annoyed",
        "frustrated",
        "rage",
        "resentful",
        "bitter",
    ),
    "fear": (
        "afraid",
        "scared",
        "frightened",
        "terrified",
        "fearful",
        "panic",
        "panicking",
    ),
    "anxiety": (
        "anxious",
        "anxiety",
        "worried",
        "worry",
        "nervous",
        "uneasy",
        "restless",
        "tense",
        "stress",
        "stressed",
        "overthinking",
    ),
    "loneliness": (
        "lonely",
        "loneliness",
        "isolated",
        "isolation",
        "disconnected",
    ),
    "exhaustion": (
        "exhausted",
        "exhaustion",
        "tired",
        "drained",
        "fatigued",
        "weary",
        "burnout",
        "depleted",
    ),
    "overwhelm": (
        "overwhelmed",
        "overwhelming",
        "overwhelm",
        "swamped",
    ),
    "calm": (
        "calm",
        "peaceful",
        "relaxed",
        "serene",
        "settled",
    ),
    "guilt": (
        "guilty",
        "guilt",
        "ashamed",
        "shame",
        "regret",
        "remorse",
    ),
    "hope": (
        "hopeful",
        "hope",
        "optimistic",
        "encouraged",
    ),
}

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _tokenize(normalized: str) -> list[str]:
    return _TOKEN_PATTERN.findall(normalized)


def _is_negated(tokens: list[str], match_start: int) -> bool:
    """True if a negation word appears within the window before ``match_start``."""
    window_start = max(0, match_start - _NEGATION_WINDOW)
    for token in tokens[window_start:match_start]:
        if token in _NEGATION_WORDS:
            return True
    return False


def _find_non_overlapping_phrases(
    normalized: str,
) -> list[tuple[int, int, str]]:
    """Return ``(start, end, label)`` for matched phrases; longest match first."""
    matches: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []

    for phrase, label in sorted(EMOTION_PHRASES.items(), key=lambda item: -len(item[0])):
        start = 0
        while True:
            idx = normalized.find(phrase, start)
            if idx == -1:
                break
            end = idx + len(phrase)
            if not any(not (end <= o_start or idx >= o_end) for o_start, o_end in occupied):
                matches.append((idx, end, label))
                occupied.append((idx, end))
            start = idx + 1

    return sorted(matches, key=lambda item: item[0])


def _token_index_at_char(tokens: list[str], normalized: str, char_index: int) -> int:
    """Map a character offset to the token index at that position."""
    prefix = normalized[:char_index]
    return len(_tokenize(prefix))


def _match_phrases(
    normalized: str,
    tokens: list[str],
    found: set[str],
    phrase_spans: list[tuple[int, int, str]],
) -> None:
    for start, _end, label in phrase_spans:
        token_index = _token_index_at_char(tokens, normalized, start)
        if _is_negated(tokens, token_index):
            continue
        found.add(label)


def _build_token_lexicon() -> dict[str, set[str]]:
    token_to_emotions: dict[str, set[str]] = {}
    for label, words in EMOTION_LEXICON.items():
        for word in words:
            token_to_emotions.setdefault(word, set()).add(label)
    return token_to_emotions


def _char_in_spans(char_index: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= char_index < end for start, end in spans)


def _match_lexicon_tokens(
    normalized: str,
    tokens: list[str],
    found: set[str],
    phrase_spans: list[tuple[int, int, str]],
) -> None:
    token_to_emotions = _build_token_lexicon()
    span_ranges = [(start, end) for start, end, _ in phrase_spans]

    search_from = 0
    for index, token in enumerate(tokens):
        labels = token_to_emotions.get(token)
        if not labels:
            continue

        token_start = normalized.find(token, search_from)
        if token_start == -1:
            continue
        search_from = token_start + len(token)

        if _char_in_spans(token_start, span_ranges):
            continue

        if _is_negated(tokens, index):
            continue

        found.update(labels)


def tag_emotion(text: str | None) -> list[str]:
    """
    Extract emotion labels from diary/free-text input.

    Returns descriptive emotion tags (e.g. ``joy``, ``sadness``, ``loneliness``).
    These are NLP signals for downstream ML/support — not psychiatric diagnoses.

    Expects caller-provided text (typically after
    :func:`nlp.preprocessing.clean_text.clean_text`). Empty or whitespace-only
    input returns ``[]``.

    Negation rule: a match is ignored when a negation word appears within
    three tokens before the phrase or word (e.g. ``not happy``, ``don't feel calm``,
    ``not stressed out``).

    Args:
        text: Diary or free-text string.

    Returns:
        Sorted list of unique emotion labels from :data:`EMOTION_LABELS`.
    """
    if text is None or not text.strip():
        return []

    normalized = text.lower()
    tokens = _tokenize(normalized)
    phrase_spans = _find_non_overlapping_phrases(normalized)
    found: set[str] = set()

    _match_phrases(normalized, tokens, found, phrase_spans)
    _match_lexicon_tokens(normalized, tokens, found, phrase_spans)

    label_order = {label: index for index, label in enumerate(EMOTION_LABELS)}
    return sorted(found, key=lambda label: label_order[label])
