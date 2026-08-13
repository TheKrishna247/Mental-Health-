"""
PS-87 explainability helpers for the baseline Logistic Regression model.

Uses coefficient-based feature contributions. These are model signals, not
clinical causation claims.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from ml.src.features.feature_pipeline import FEATURE_NAMES
from ml.src.models.risk_model import RiskModel

DEFAULT_TOP_N: int = 5


class TopFactor(TypedDict):
    factor: str
    weight: float


FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "phq9_norm": "PHQ-9 symptom burden",
    "phq9_missing": "PHQ-9 unavailable",
    "gad7_norm": "GAD-7 anxiety burden",
    "gad7_missing": "GAD-7 unavailable",
    "who5_risk_norm": "WHO-5 wellbeing risk",
    "who5_missing": "WHO-5 unavailable",
    "mood_mean": "Recent mood",
    "mood_latest": "Latest mood",
    "mood_trend": "Mood trend",
    "mood_available": "Mood check-ins available",
    "sleep_mean": "Recent sleep",
    "sleep_latest": "Latest sleep",
    "sleep_trend": "Sleep trend",
    "sleep_available": "Sleep check-ins available",
    "routine_mean": "Recent routine",
    "routine_latest": "Latest routine",
    "routine_trend": "Routine trend",
    "routine_available": "Routine check-ins available",
    "diary_sentiment_mean": "Recent diary sentiment",
    "diary_sentiment_latest": "Latest diary sentiment",
    "diary_sentiment_trend": "Diary sentiment trend",
    "diary_available": "Diary entries available",
    "emotion_joy": "Joy signal",
    "emotion_sadness": "Sadness signal",
    "emotion_anger": "Anger signal",
    "emotion_fear": "Fear signal",
    "emotion_anxiety": "Anxiety signal",
    "emotion_loneliness": "Loneliness signal",
    "emotion_exhaustion": "Exhaustion signal",
    "emotion_overwhelm": "Overwhelm signal",
    "emotion_calm": "Calm signal",
    "emotion_guilt": "Guilt signal",
    "emotion_hope": "Hope signal",
}


def humanize_feature_name(feature_name: str) -> str:
    """Map internal feature name to a readable label."""
    return FEATURE_DISPLAY_NAMES.get(
        feature_name,
        feature_name.replace("_", " ").capitalize(),
    )


def compute_feature_contributions(
    model: RiskModel,
    feature_vector: list[float] | np.ndarray,
) -> dict[str, float]:
    """
    Compute per-feature contribution as coefficient * scaled_feature_value.

    Uses the same scaled feature space as the trained classifier.
    """
    array = np.asarray(feature_vector, dtype=float).reshape(1, -1)
    if array.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} features; got {array.shape[1]}."
        )
    scaled = model.transformed_features(array)[0]
    coefficients = model.coefficients()
    contributions = {
        name: float(coef * value)
        for name, coef, value in zip(FEATURE_NAMES, coefficients, scaled, strict=True)
    }
    return contributions


def explain_instance(
    model: RiskModel,
    feature_vector: list[float] | np.ndarray,
    *,
    top_n: int = DEFAULT_TOP_N,
) -> list[TopFactor]:
    """
    Rank features by absolute contribution and return top factors.

    Compatible with the public RiskScore ``top_factors`` shape.
    """
    contributions = compute_feature_contributions(model, feature_vector)
    ranked = sorted(
        contributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    top = ranked[:top_n]
    return [
        TopFactor(
            factor=humanize_feature_name(name),
            weight=round(weight, 4),
        )
        for name, weight in top
    ]
