"""Tests for nlp/sentiment/sentiment.py."""

import pytest

from nlp.preprocessing.clean_text import clean_text
from nlp.sentiment.sentiment import (
    NEUTRAL_SENTIMENT,
    SENTIMENT_MAX,
    SENTIMENT_MIN,
    analyze_sentiment,
)


def test_sentiment_positive():
    score = analyze_sentiment("I am so happy and grateful for today!")
    assert score > 0.3
    assert SENTIMENT_MIN <= score <= SENTIMENT_MAX


def test_sentiment_negative():
    score = analyze_sentiment("I feel terrible, hopeless, and miserable.")
    assert score < -0.3
    assert SENTIMENT_MIN <= score <= SENTIMENT_MAX


def test_sentiment_neutral():
    score = analyze_sentiment("I went to class and ate lunch.")
    assert -0.2 <= score <= 0.2


def test_sentiment_empty():
    assert analyze_sentiment("") == NEUTRAL_SENTIMENT
    assert analyze_sentiment(None) == NEUTRAL_SENTIMENT


def test_sentiment_whitespace_only():
    assert analyze_sentiment("   \n\t  ") == NEUTRAL_SENTIMENT


def test_sentiment_negation():
    positive = analyze_sentiment("I am happy")
    negated = analyze_sentiment("I am not happy")
    assert negated < positive
    assert negated < 0


def test_sentiment_punctuation_emphasis():
    plain = analyze_sentiment("I am happy")
    emphatic = analyze_sentiment("I am happy!!!")
    assert emphatic >= plain


def test_sentiment_mixed_case():
    lower = analyze_sentiment("i am happy")
    mixed = analyze_sentiment("I Am HaPpY")
    assert lower > 0
    assert mixed > 0


def test_sentiment_after_preprocessing():
    raw = "I am not happy!!!  https://noise.com"
    score = analyze_sentiment(clean_text(raw))
    assert score < 0


def test_sentiment_repeated_punctuation_normalized():
    cleaned = clean_text("This is awful!!!")
    score = analyze_sentiment(cleaned)
    assert score < -0.2


@pytest.mark.parametrize(
    "text",
    [
        "I am so happy and grateful for today!",
        "I feel terrible, hopeless, and miserable.",
        "I went to class and ate lunch.",
        "",
    ],
)
def test_sentiment_score_within_documented_range(text):
    score = analyze_sentiment(text)
    assert SENTIMENT_MIN <= score <= SENTIMENT_MAX
