"""
PHQ-9 questionnaire scoring for the PS-87 NLP layer.

Accepts nine item responses already normalized to integers 0–3
(Not at all → 0, Several days → 1, More than half the days → 2,
Nearly every day → 3). Returns total score and severity only; callers
(e.g. backend) attach user_id, type, and submitted_at for AssessmentResult.
"""

from __future__ import annotations

from typing import Sequence, TypedDict

PHQ9_ITEM_COUNT = 9
MIN_ITEM_SCORE = 0
MAX_ITEM_SCORE = 3


class PHQ9ScoringResult(TypedDict):
    """NLP output for PHQ-9; maps to AssessmentResult score and severity fields."""

    score: int
    severity: str


def _severity_from_total(total: int) -> str:
    """
    Map PHQ-9 total score (0–27) to severity band.

    Bands: 0–4 minimal, 5–9 mild, 10–14 moderate, 15–19 moderately severe,
    20–27 severe.
    """
    if total <= 4:
        return "minimal"
    if total <= 9:
        return "mild"
    if total <= 14:
        return "moderate"
    if total <= 19:
        return "moderately severe"
    return "severe"


def _validate_responses(responses: Sequence[int]) -> None:
    if len(responses) != PHQ9_ITEM_COUNT:
        raise ValueError(
            f"PHQ-9 requires exactly {PHQ9_ITEM_COUNT} responses; got {len(responses)}."
        )
    for index, value in enumerate(responses):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"PHQ-9 item {index + 1} must be an integer in "
                f"{MIN_ITEM_SCORE}–{MAX_ITEM_SCORE}; got {value!r}."
            )
        if not MIN_ITEM_SCORE <= value <= MAX_ITEM_SCORE:
            raise ValueError(
                f"PHQ-9 item {index + 1} must be in "
                f"{MIN_ITEM_SCORE}–{MAX_ITEM_SCORE}; got {value}."
            )


def score_phq9(responses: Sequence[int]) -> PHQ9ScoringResult:
    """
    Score a completed PHQ-9 from nine normalized item responses.

    Args:
        responses: Sequence of exactly nine integers, each in 0–3.

    Returns:
        Dict with ``score`` (0–27) and ``severity`` (minimal | mild | moderate
        | moderately severe | severe).

    Raises:
        ValueError: If the response count or any item value is invalid.
    """
    _validate_responses(responses)
    total = sum(responses)
    return PHQ9ScoringResult(score=total, severity=_severity_from_total(total))
