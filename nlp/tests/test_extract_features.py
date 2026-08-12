"""Tests for nlp/feature_extraction/extract_features.py."""

from nlp.emotion.emotion import EMOTION_LABELS
from nlp.feature_extraction.extract_features import extract_features


def test_extract_features_normal_diary():
    result = extract_features("Today was good. I feel grateful and calm.")
    assert -1.0 <= result["sentiment_score"] <= 1.0
    assert result["word_count"] > 0
    assert result["char_count"] > 0
    assert isinstance(result["emotion_tags"], list)
    assert set(result["emotion_flags"].keys()) == set(EMOTION_LABELS)


def test_extract_features_empty():
    result = extract_features("")
    assert result["sentiment_score"] == 0.0
    assert result["emotion_tags"] == []
    assert result["emotion_count"] == 0
    assert result["word_count"] == 0
    assert result["char_count"] == 0
    assert all(value == 0 for value in result["emotion_flags"].values())


def test_extract_features_none():
    result = extract_features(None)
    assert result["sentiment_score"] == 0.0
    assert result["emotion_tags"] == []


def test_extract_features_multiple_emotions():
    result = extract_features("I am sad, anxious, and completely exhausted.")
    assert result["sentiment_score"] < 0
    assert result["emotion_count"] >= 2
    assert result["emotion_flags"]["sadness"] == 1
    assert result["emotion_flags"]["anxiety"] == 1
    assert result["emotion_flags"]["exhaustion"] == 1


def test_extract_features_neutral():
    result = extract_features("I went to class and ate lunch.")
    assert result["emotion_tags"] == []
    assert result["emotion_count"] == 0
    assert all(value == 0 for value in result["emotion_flags"].values())


def test_extract_features_deterministic():
    text = "I feel lonely but hopeful about tomorrow."
    assert extract_features(text) == extract_features(text)


def test_extract_features_emotion_count_matches_tags():
    result = extract_features("I feel angry and guilty.")
    assert result["emotion_count"] == len(result["emotion_tags"])


def test_extract_features_word_and_char_counts():
    text = "I feel calm."
    result = extract_features(text)
    assert result["word_count"] == 3
    assert result["char_count"] == len("I feel calm.")


def test_extract_features_sentiment_range():
    negative = extract_features("I feel terrible and hopeless.")
    positive = extract_features("I am delighted and grateful!")
    assert -1.0 <= negative["sentiment_score"] <= 1.0
    assert -1.0 <= positive["sentiment_score"] <= 1.0
    assert negative["sentiment_score"] < positive["sentiment_score"]
