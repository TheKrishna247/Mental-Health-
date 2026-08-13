"""Tests for ml/src/features/feature_pipeline.py."""

from datetime import datetime, timedelta

import pytest

from ml.src.features.feature_pipeline import (
    FEATURE_NAMES,
    CheckInRecord,
    DiaryRecord,
    StudentFeatureInput,
    build_feature_vector,
    compute_trend,
    feature_dict_to_vector,
    normalize_checkin_value,
    normalize_gad7,
    normalize_phq9,
    normalize_who5_risk,
    sentiment_to_risk,
    validate_feature_dict,
)

AS_OF = datetime(2026, 8, 12, 12, 0, 0)


def test_phq9_normalization():
    assert normalize_phq9(0) == 0.0
    assert normalize_phq9(27) == pytest.approx(1.0)
    assert normalize_phq9(13) == pytest.approx(13 / 27)


def test_gad7_normalization():
    assert normalize_gad7(0) == 0.0
    assert normalize_gad7(21) == pytest.approx(1.0)


def test_who5_risk_normalization():
    assert normalize_who5_risk(25) == pytest.approx(0.0)
    assert normalize_who5_risk(0) == pytest.approx(1.0)
    assert normalize_who5_risk(12) == pytest.approx(1 - (12 / 25))


def test_checkin_normalization():
    assert normalize_checkin_value(1) == pytest.approx(1.0)
    assert normalize_checkin_value(5) == pytest.approx(0.0)
    assert normalize_checkin_value(3) == pytest.approx(0.5)


def test_checkin_validation_rejects_out_of_range():
    with pytest.raises(ValueError):
        normalize_checkin_value(0)
    with pytest.raises(ValueError):
        normalize_checkin_value(6)


def test_sentiment_transformation():
    assert sentiment_to_risk(1.0) == pytest.approx(0.0)
    assert sentiment_to_risk(0.0) == pytest.approx(0.5)
    assert sentiment_to_risk(-1.0) == pytest.approx(1.0)


def test_compute_trend_requires_two_values():
    assert compute_trend([0.5]) == 0.0
    assert compute_trend([0.2, 0.8]) == pytest.approx(0.6)


def test_multiple_checkin_aggregation():
    checkins = [
        CheckInRecord("mood", 2, AS_OF - timedelta(days=3)),
        CheckInRecord("mood", 4, AS_OF - timedelta(days=1)),
    ]
    student = StudentFeatureInput(checkins=checkins, as_of=AS_OF)
    features = build_feature_vector(student)
    assert features["mood_available"] == 1.0
    assert features["mood_latest"] == pytest.approx(0.25)
    assert features["mood_mean"] == pytest.approx((0.75 + 0.25) / 2)


def test_multiple_diary_aggregation():
    entries = [
        DiaryRecord(0.5, {"sadness": 1}, AS_OF - timedelta(days=4)),
        DiaryRecord(-0.5, {"anxiety": 1}, AS_OF - timedelta(days=1)),
    ]
    student = StudentFeatureInput(diary_entries=entries, as_of=AS_OF)
    features = build_feature_vector(student)
    assert features["diary_available"] == 1.0
    assert features["diary_sentiment_latest"] == pytest.approx(sentiment_to_risk(-0.5))
    assert features["emotion_sadness"] == 1.0
    assert features["emotion_anxiety"] == 1.0


def test_emotion_flag_aggregation_any_entry():
    entries = [
        DiaryRecord(0.0, {"exhaustion": 1}, AS_OF - timedelta(days=2)),
        DiaryRecord(0.2, {"joy": 1}, AS_OF - timedelta(days=1)),
    ]
    features = build_feature_vector(StudentFeatureInput(diary_entries=entries, as_of=AS_OF))
    assert features["emotion_exhaustion"] == 1.0
    assert features["emotion_joy"] == 1.0


def test_missing_questionnaire_indicators():
    features = build_feature_vector(StudentFeatureInput(as_of=AS_OF))
    assert features["phq9_norm"] == 0.0
    assert features["phq9_missing"] == 1.0
    assert features["gad7_missing"] == 1.0
    assert features["who5_missing"] == 1.0


def test_present_zero_questionnaire_not_marked_missing():
    student = StudentFeatureInput(phq9_score=0, gad7_score=0, who5_score=25, as_of=AS_OF)
    features = build_feature_vector(student)
    assert features["phq9_norm"] == 0.0
    assert features["phq9_missing"] == 0.0
    assert features["who5_risk_norm"] == 0.0
    assert features["who5_missing"] == 0.0


def test_missing_checkins_and_diary():
    features = build_feature_vector(StudentFeatureInput(as_of=AS_OF))
    assert features["mood_available"] == 0.0
    assert features["sleep_available"] == 0.0
    assert features["routine_available"] == 0.0
    assert features["diary_available"] == 0.0
    assert features["emotion_sadness"] == 0.0


def test_feature_ordering_and_dimensionality():
    features = build_feature_vector(StudentFeatureInput(as_of=AS_OF))
    validate_feature_dict(features)
    vector = feature_dict_to_vector(features)
    assert len(FEATURE_NAMES) == 33
    assert len(vector) == 33
    for index, name in enumerate(FEATURE_NAMES):
        assert vector[index] == features[name]


def test_out_of_window_entries_are_excluded():
    checkins = [CheckInRecord("mood", 1, AS_OF - timedelta(days=10))]
    features = build_feature_vector(StudentFeatureInput(checkins=checkins, as_of=AS_OF))
    assert features["mood_available"] == 0.0
