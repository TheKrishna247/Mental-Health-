"""
WHO-5 questionnaire scoring for the PS-87 NLP layer.

Accepts five item responses already normalized to integers 0–5 on the
six-point WHO-5 response scale. Raw score is the sum of item scores (0–25).
PS-87 severity uses the standard <13 cutoff as ``poor`` (0–12) or
``not poor`` (13–25). Returns score and severity only; callers (e.g.
backend) attach user_id, type, and submitted_at for AssessmentResult.
"""

from __future__ import annotations

from typing import Sequence, TypedDict

WHO5_ITEM_COUNT = 5
MIN_ITEM_SCORE = 0
MAX_ITEM_SCORE = 5
POOR_MAX_RAW_SCORE = 12


class WHO5ScoringResult(TypedDict):
    """NLP output for WHO-5; maps to AssessmentResult score and severity fields."""

    score: int
    severity: str


def _severity_from_total(total: int) -> str:
    """
    Map WHO-5 raw score (0–25) to PS-87 severity.

    Scores 0–12 → ``poor``; 13–25 → ``not poor`` (WHO-5 <13 well-being cutoff).
    """
    if total <= POOR_MAX_RAW_SCORE:
        return "poor"
    return "not poor"


def _validate_responses(responses: Sequence[int]) -> None:
    if len(responses) != WHO5_ITEM_COUNT:
        raise ValueError(
            f"WHO-5 requires exactly {WHO5_ITEM_COUNT} responses; got {len(responses)}."
        )
    for index, value in enumerate(responses):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"WHO-5 item {index + 1} must be an integer in "
                f"{MIN_ITEM_SCORE}–{MAX_ITEM_SCORE}; got {value!r}."
            )
        if not MIN_ITEM_SCORE <= value <= MAX_ITEM_SCORE:
            raise ValueError(
                f"WHO-5 item {index + 1} must be in "
                f"{MIN_ITEM_SCORE}–{MAX_ITEM_SCORE}; got {value}."
            )


def score_who5(responses: Sequence[int]) -> WHO5ScoringResult:
    """
    Score a completed WHO-5 from five normalized item responses.

    Args:
        responses: Sequence of exactly five integers, each in 0–5.

    Returns:
        Dict with ``score`` (raw total 0–25) and ``severity`` (``poor`` |
        ``not poor``).

    Raises:
        ValueError: If the response count or any item value is invalid.
    """
    _validate_responses(responses)
    total = sum(responses)
    return WHO5ScoringResult(score=total, severity=_severity_from_total(total))
