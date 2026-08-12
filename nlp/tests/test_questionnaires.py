"""Smoke tests for completed questionnaire scorers."""

import pytest

from nlp.questionnaire.gad7.gad7 import score_gad7
from nlp.questionnaire.phq9.phq9 import score_phq9
from nlp.questionnaire.who5.who5 import score_who5


def test_phq9_minimal():
    result = score_phq9([0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert result["score"] == 0
    assert result["severity"] == "minimal"


def test_phq9_severe():
    result = score_phq9([3, 3, 3, 3, 3, 3, 3, 3, 3])
    assert result["score"] == 27
    assert result["severity"] == "severe"


def test_phq9_invalid_length():
    with pytest.raises(ValueError):
        score_phq9([0, 1, 2])


def test_gad7_minimal():
    result = score_gad7([0, 0, 0, 0, 0, 0, 0])
    assert result["score"] == 0
    assert result["severity"] == "minimal"


def test_gad7_severe():
    result = score_gad7([3, 3, 3, 3, 3, 3, 3])
    assert result["score"] == 21
    assert result["severity"] == "severe"


def test_gad7_invalid_value():
    with pytest.raises(ValueError):
        score_gad7([0, 0, 0, 0, 0, 0, 4])


def test_who5_poor():
    result = score_who5([0, 1, 2, 3, 4])
    assert result["score"] == 10
    assert result["severity"] == "poor"


def test_who5_not_poor():
    result = score_who5([3, 4, 4, 4, 4])
    assert result["score"] == 19
    assert result["severity"] == "not poor"


def test_who5_invalid_length():
    with pytest.raises(ValueError):
        score_who5([1, 2, 3])
