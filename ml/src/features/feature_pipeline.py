"""
PS-87 ML feature pipeline.

Builds a fixed numerical feature vector from questionnaire scores, check-ins,
and diary/NLP signals. Independent of FastAPI, Backend, and NLP runtime imports.

The internal feature vector is NOT a locked public contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Final, Sequence

OBSERVATION_WINDOW_DAYS: Final[int] = 7

PHQ9_MAX: Final[int] = 27
GAD7_MAX: Final[int] = 21
WHO5_MAX: Final[int] = 25

CHECKIN_MIN: Final[int] = 1
CHECKIN_MAX: Final[int] = 5

EMOTION_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "emotion_joy",
    "emotion_sadness",
    "emotion_anger",
    "emotion_fear",
    "emotion_anxiety",
    "emotion_loneliness",
    "emotion_exhaustion",
    "emotion_overwhelm",
    "emotion_calm",
    "emotion_guilt",
    "emotion_hope",
)

# Maps NLP emotion_flags keys to ML feature column names.
EMOTION_FLAG_TO_FEATURE: Final[dict[str, str]] = {
    "joy": "emotion_joy",
    "sadness": "emotion_sadness",
    "anger": "emotion_anger",
    "fear": "emotion_fear",
    "anxiety": "emotion_anxiety",
    "loneliness": "emotion_loneliness",
    "exhaustion": "emotion_exhaustion",
    "overwhelm": "emotion_overwhelm",
    "calm": "emotion_calm",
    "guilt": "emotion_guilt",
    "hope": "emotion_hope",
}

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "phq9_norm",
    "phq9_missing",
    "gad7_norm",
    "gad7_missing",
    "who5_risk_norm",
    "who5_missing",
    "mood_mean",
    "mood_latest",
    "mood_trend",
    "mood_available",
    "sleep_mean",
    "sleep_latest",
    "sleep_trend",
    "sleep_available",
    "routine_mean",
    "routine_latest",
    "routine_trend",
    "routine_available",
    "diary_sentiment_mean",
    "diary_sentiment_latest",
    "diary_sentiment_trend",
    "diary_available",
    *EMOTION_FEATURE_NAMES,
)


@dataclass(frozen=True)
class CheckInRecord:
    """Contract-compatible check-in record for feature engineering."""

    type: str
    value: int
    timestamp: datetime


@dataclass(frozen=True)
class DiaryRecord:
    """Diary/NLP signals for feature engineering (no raw text)."""

    sentiment_score: float
    emotion_flags: dict[str, int]
    timestamp: datetime


@dataclass
class StudentFeatureInput:
    """Internal ML input bundle for one student at a reference time."""

    phq9_score: int | None = None
    gad7_score: int | None = None
    who5_score: int | None = None
    checkins: list[CheckInRecord] = field(default_factory=list)
    diary_entries: list[DiaryRecord] = field(default_factory=list)
    as_of: datetime | None = None


def normalize_phq9(score: int) -> float:
    """Map PHQ-9 score (0–27) to [0, 1] risk-oriented normalization."""
    _validate_questionnaire_score(score, PHQ9_MAX, "PHQ-9")
    return score / PHQ9_MAX


def normalize_gad7(score: int) -> float:
    """Map GAD-7 score (0–21) to [0, 1] risk-oriented normalization."""
    _validate_questionnaire_score(score, GAD7_MAX, "GAD-7")
    return score / GAD7_MAX


def normalize_who5_risk(score: int) -> float:
    """Map WHO-5 score (0–25) to [0, 1] risk signal (lower wellbeing → higher risk)."""
    _validate_questionnaire_score(score, WHO5_MAX, "WHO-5")
    return 1.0 - (score / WHO5_MAX)


def normalize_checkin_value(value: int) -> float:
    """
    Map check-in value (1–5) to [0, 1] risk signal.

    1 → 1.00, 2 → 0.75, 3 → 0.50, 4 → 0.25, 5 → 0.00
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Check-in value must be an integer in 1–5; got {value!r}.")
    if not CHECKIN_MIN <= value <= CHECKIN_MAX:
        raise ValueError(f"Check-in value must be in 1–5; got {value}.")
    return (CHECKIN_MAX - value) / (CHECKIN_MAX - CHECKIN_MIN)


def sentiment_to_risk(sentiment_score: float) -> float:
    """
    Map NLP sentiment_score [-1, 1] to [0, 1] risk-oriented signal.

    +1 → 0.0, 0 → 0.5, -1 → 1.0
    """
    if not -1.0 <= sentiment_score <= 1.0:
        raise ValueError(
            f"sentiment_score must be in [-1, 1]; got {sentiment_score}."
        )
    return (1.0 - sentiment_score) / 2.0


def compute_trend(values: Sequence[float]) -> float:
    """
    Deterministic recent-vs-earlier trend.

    Compares the mean of the later half to the mean of the earlier half.
    Returns 0.0 when fewer than two values are available.
    """
    if len(values) < 2:
        return 0.0
    midpoint = len(values) // 2
    earlier = values[:midpoint]
    later = values[midpoint:]
    if not earlier or not later:
        return 0.0
    earlier_mean = sum(earlier) / len(earlier)
    later_mean = sum(later) / len(later)
    return later_mean - earlier_mean


def _validate_questionnaire_score(score: int, maximum: int, label: str) -> None:
    if isinstance(score, bool) or not isinstance(score, int):
        raise ValueError(f"{label} score must be an integer in 0–{maximum}; got {score!r}.")
    if not 0 <= score <= maximum:
        raise ValueError(f"{label} score must be in 0–{maximum}; got {score}.")


def _window_start(as_of: datetime) -> datetime:
    return as_of - timedelta(days=OBSERVATION_WINDOW_DAYS)


def _filter_checkins(
    checkins: Sequence[CheckInRecord],
    checkin_type: str,
    as_of: datetime,
) -> list[tuple[datetime, float]]:
    start = _window_start(as_of)
    filtered: list[tuple[datetime, float]] = []
    for record in checkins:
        if record.type != checkin_type:
            continue
        if record.timestamp < start or record.timestamp > as_of:
            continue
        filtered.append((record.timestamp, normalize_checkin_value(record.value)))
    filtered.sort(key=lambda item: item[0])
    return filtered


def _aggregate_checkin_type(
    checkins: Sequence[CheckInRecord],
    checkin_type: str,
    as_of: datetime,
) -> tuple[float, float, float, float]:
    """Return mean, latest, trend, available for one check-in type."""
    series = _filter_checkins(checkins, checkin_type, as_of)
    if not series:
        return 0.0, 0.0, 0.0, 0.0
    values = [value for _, value in series]
    mean_value = sum(values) / len(values)
    latest_value = values[-1]
    trend_value = compute_trend(values)
    return mean_value, latest_value, trend_value, 1.0


def _filter_diary_entries(
    entries: Sequence[DiaryRecord],
    as_of: datetime,
) -> list[DiaryRecord]:
    start = _window_start(as_of)
    filtered = [
        entry
        for entry in entries
        if start <= entry.timestamp <= as_of
    ]
    filtered.sort(key=lambda entry: entry.timestamp)
    return filtered


def _aggregate_diary_sentiment(
    entries: Sequence[DiaryRecord],
    as_of: datetime,
) -> tuple[float, float, float, float]:
    filtered = _filter_diary_entries(entries, as_of)
    if not filtered:
        return 0.0, 0.0, 0.0, 0.0
    risk_values = [sentiment_to_risk(entry.sentiment_score) for entry in filtered]
    mean_value = sum(risk_values) / len(risk_values)
    latest_value = risk_values[-1]
    trend_value = compute_trend(risk_values)
    return mean_value, latest_value, trend_value, 1.0


def _aggregate_emotion_flags(
    entries: Sequence[DiaryRecord],
    as_of: datetime,
) -> dict[str, float]:
    filtered = _filter_diary_entries(entries, as_of)
    aggregated = {name: 0.0 for name in EMOTION_FEATURE_NAMES}
    if not filtered:
        return aggregated
    for entry in filtered:
        for flag_key, feature_name in EMOTION_FLAG_TO_FEATURE.items():
            if entry.emotion_flags.get(flag_key, 0) == 1:
                aggregated[feature_name] = 1.0
    return aggregated


def _questionnaire_features(
    score: int | None,
    maximum: int,
    label: str,
    *,
    who5: bool = False,
) -> tuple[float, float]:
    if score is None:
        return 0.0, 1.0
    if who5:
        return normalize_who5_risk(score), 0.0
    if label == "PHQ-9":
        return normalize_phq9(score), 0.0
    if label == "GAD-7":
        return normalize_gad7(score), 0.0
    raise ValueError(f"Unsupported questionnaire label: {label}")


def build_feature_vector(student: StudentFeatureInput) -> dict[str, float]:
    """
    Build the fixed ML feature vector for one student.

    Missing questionnaires use norm=0.0 with missing=1.0 so the model can
    distinguish absence from a genuine zero score (missing=0.0).
    """
    as_of = student.as_of or datetime.now()

    phq9_norm, phq9_missing = _questionnaire_features(
        student.phq9_score, PHQ9_MAX, "PHQ-9"
    )
    gad7_norm, gad7_missing = _questionnaire_features(
        student.gad7_score, GAD7_MAX, "GAD-7"
    )
    who5_risk_norm, who5_missing = _questionnaire_features(
        student.who5_score, WHO5_MAX, "WHO-5", who5=True
    )

    mood_mean, mood_latest, mood_trend, mood_available = _aggregate_checkin_type(
        student.checkins, "mood", as_of
    )
    sleep_mean, sleep_latest, sleep_trend, sleep_available = _aggregate_checkin_type(
        student.checkins, "sleep", as_of
    )
    routine_mean, routine_latest, routine_trend, routine_available = _aggregate_checkin_type(
        student.checkins, "routine", as_of
    )

    (
        diary_sentiment_mean,
        diary_sentiment_latest,
        diary_sentiment_trend,
        diary_available,
    ) = _aggregate_diary_sentiment(student.diary_entries, as_of)

    emotion_features = _aggregate_emotion_flags(student.diary_entries, as_of)

    features: dict[str, float] = {
        "phq9_norm": phq9_norm,
        "phq9_missing": phq9_missing,
        "gad7_norm": gad7_norm,
        "gad7_missing": gad7_missing,
        "who5_risk_norm": who5_risk_norm,
        "who5_missing": who5_missing,
        "mood_mean": mood_mean,
        "mood_latest": mood_latest,
        "mood_trend": mood_trend,
        "mood_available": mood_available,
        "sleep_mean": sleep_mean,
        "sleep_latest": sleep_latest,
        "sleep_trend": sleep_trend,
        "sleep_available": sleep_available,
        "routine_mean": routine_mean,
        "routine_latest": routine_latest,
        "routine_trend": routine_trend,
        "routine_available": routine_available,
        "diary_sentiment_mean": diary_sentiment_mean,
        "diary_sentiment_latest": diary_sentiment_latest,
        "diary_sentiment_trend": diary_sentiment_trend,
        "diary_available": diary_available,
        **emotion_features,
    }
    return features


def feature_dict_to_vector(features: dict[str, float]) -> list[float]:
    """Convert a feature dict to a list ordered by FEATURE_NAMES."""
    return [float(features[name]) for name in FEATURE_NAMES]


def validate_feature_dict(features: dict[str, float]) -> None:
    """Ensure a feature dict contains exactly the expected feature keys."""
    missing = set(FEATURE_NAMES) - set(features)
    extra = set(features) - set(FEATURE_NAMES)
    if missing or extra:
        raise ValueError(
            f"Feature dict keys mismatch. Missing: {sorted(missing)}; extra: {sorted(extra)}."
        )
