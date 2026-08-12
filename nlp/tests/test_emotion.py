"""Tests for nlp/emotion/emotion.py."""

import pytest

from nlp.emotion.emotion import EMOTION_LABELS, tag_emotion


def test_emotion_single():
    assert tag_emotion("I feel lonely today.") == ["loneliness"]


def test_emotion_multiple():
    tags = tag_emotion("I feel lonely, overwhelmed and exhausted.")
    assert tags == ["loneliness", "exhaustion", "overwhelm"]


def test_emotion_neutral():
    assert tag_emotion("I went to the library and studied.") == []


def test_emotion_empty():
    assert tag_emotion("") == []
    assert tag_emotion(None) == []


def test_emotion_whitespace_only():
    assert tag_emotion("   ") == []


def test_emotion_positive():
    assert tag_emotion("I am happy.") == ["joy"]


def test_emotion_negative():
    tags = tag_emotion("I am sad and anxious.")
    assert "sadness" in tags
    assert "anxiety" in tags


def test_emotion_negated_positive_word():
    assert tag_emotion("I am not happy.") == []


def test_emotion_negated_phrase_stressed_out():
    assert tag_emotion("I am stressed out.") == ["anxiety"]
    assert tag_emotion("I am not stressed out.") == []


def test_emotion_feel_calm():
    assert tag_emotion("I feel calm.") == ["calm"]


def test_emotion_negated_feel_calm():
    assert tag_emotion("I don't feel calm.") == []


def test_emotion_multi_word_phrases():
    assert "exhaustion" in tag_emotion("I am burned out.")
    assert "hope" in tag_emotion("I am looking forward to the break.")
    assert "calm" in tag_emotion("I am at ease now.")
    assert "overwhelm" in tag_emotion("This is too much for me.")


def test_emotion_ambiguous_alone_no_false_positive():
    assert tag_emotion("I went alone to the library.") == []


def test_emotion_contextual_alone():
    assert tag_emotion("I feel so alone tonight.") == ["loneliness"]


def test_emotion_ambiguous_great_no_bare_match():
    assert tag_emotion("The exam was great.") == []
    assert tag_emotion("I feel great today.") == ["joy"]


def test_emotion_labels_are_stable():
    tags = tag_emotion("I am anxious but hopeful.")
    assert all(tag in EMOTION_LABELS for tag in tags)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I am happy.", ["joy"]),
        ("I am not happy.", []),
        ("I feel calm.", ["calm"]),
        ("I don't feel calm.", []),
        ("I am stressed out.", ["anxiety"]),
        ("I am not stressed out.", []),
        ("I feel lonely, overwhelmed and exhausted.", ["loneliness", "exhaustion", "overwhelm"]),
        ("I went to class and ate lunch.", []),
        ("I went alone to the library.", []),
    ],
)
def test_emotion_regression_cases(text: str, expected: list[str]):
    assert tag_emotion(text) == expected
