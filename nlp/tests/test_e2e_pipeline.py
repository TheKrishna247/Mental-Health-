"""End-to-end diary NLP pipeline tests."""

from nlp.emotion.emotion import EMOTION_LABELS
from nlp.feature_extraction.extract_features import extract_features
from nlp.preprocessing.clean_text import clean_text
from nlp.sentiment.sentiment import analyze_sentiment
from nlp.emotion.emotion import tag_emotion


DIARY_TEXT = (
    "Long day. I'm not stressed out anymore, but I still feel tired "
    "and a bit lonely. Trying to stay hopeful."
)


def test_e2e_diary_pipeline_steps_are_consistent():
    cleaned = clean_text(DIARY_TEXT)
    sentiment = analyze_sentiment(cleaned)
    emotions = tag_emotion(cleaned)
    features = extract_features(DIARY_TEXT)

    assert features["sentiment_score"] == sentiment
    assert features["emotion_tags"] == emotions
    assert features["emotion_count"] == len(emotions)
    assert features["word_count"] == len(cleaned.split())
    assert features["char_count"] == len(cleaned)
    assert set(features["emotion_flags"].keys()) == set(EMOTION_LABELS)


def test_e2e_diary_expected_signals():
    features = extract_features(DIARY_TEXT)

    # Negated phrase should not tag anxiety; affirmed emotions should appear.
    assert "anxiety" not in features["emotion_tags"]
    assert features["emotion_flags"]["anxiety"] == 0
    assert "exhaustion" in features["emotion_tags"]
    assert "loneliness" in features["emotion_tags"]
    assert "hope" in features["emotion_tags"]
    assert -1.0 <= features["sentiment_score"] <= 1.0


def test_e2e_empty_diary():
    cleaned = clean_text("")
    assert cleaned == ""
    assert analyze_sentiment(cleaned) == 0.0
    assert tag_emotion(cleaned) == []
    features = extract_features("")
    assert features["emotion_count"] == 0
    assert features["word_count"] == 0
