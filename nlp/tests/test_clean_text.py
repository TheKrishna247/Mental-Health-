"""Tests for nlp/preprocessing/clean_text.py."""

from nlp.preprocessing.clean_text import clean_text


def test_clean_text_normal_diary():
    text = "  Today was hectic. I feel tired but okay overall.  "
    assert clean_text(text) == "Today was hectic. I feel tired but okay overall."


def test_clean_text_extra_whitespace():
    assert clean_text("too   many    spaces") == "too many spaces"


def test_clean_text_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_clean_text_whitespace_only():
    assert clean_text("   \t\n  ") == ""


def test_clean_text_url():
    assert clean_text("I feel sad https://example.com/page today") == "I feel sad today"


def test_clean_text_email():
    assert clean_text("Contact me at user@example.com about stress") == "Contact me at about stress"


def test_clean_text_repeated_punctuation():
    assert clean_text("I am not happy!!!") == "I am not happy!"
    assert clean_text("Why me???!!!") == "Why me?!"


def test_clean_text_preserves_negation():
    result = clean_text("I am not happy.")
    assert "not" in result
    assert "happy" in result


def test_clean_text_unicode_punctuation():
    # Smart quotes and em dash → ASCII equivalents; words preserved.
    assert clean_text("I\u2019m not happy \u2014 really.") == "I'm not happy - really."


def test_clean_text_unicode_diary_hinglish():
    text = "Aaj mood off tha, but I'm trying \U0001f614"
    result = clean_text(text)
    assert "Aaj mood off tha" in result
    assert "\U0001f614" in result


def test_clean_text_preserves_mixed_case():
    assert clean_text("I Am NOT Happy") == "I Am NOT Happy"
