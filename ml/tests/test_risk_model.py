"""Tests for ml/src/models/risk_model.py and explainability."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from ml.src.explainability.explainability import explain_instance
from ml.src.features.feature_pipeline import (
    FEATURE_NAMES,
    CheckInRecord,
    DiaryRecord,
    StudentFeatureInput,
    build_feature_vector,
    feature_dict_to_vector,
)
from ml.src.models.risk_model import (
    RISK_LEVEL_LOW_MAX,
    RISK_LEVEL_MODERATE_MAX,
    RiskModel,
    probability_to_score,
    score_to_level,
)
from ml.src.training.train import generate_synthetic_training_data


def test_model_training_and_inference():
    frame = generate_synthetic_training_data(n_samples=300, random_state=42)
    X = frame[list(FEATURE_NAMES)].to_numpy(dtype=float)
    y = frame["risk_target"].to_numpy(dtype=int)

    model = RiskModel()
    model.fit(X, y)
    probabilities = model.predict_proba(X)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_risk_score_range_and_levels():
    assert probability_to_score(0.0) == 0
    assert probability_to_score(1.0) == 100
    assert score_to_level(0) == "low"
    assert score_to_level(RISK_LEVEL_LOW_MAX) == "low"
    assert score_to_level(RISK_LEVEL_LOW_MAX + 1) == "moderate"
    assert score_to_level(RISK_LEVEL_MODERATE_MAX) == "moderate"
    assert score_to_level(RISK_LEVEL_MODERATE_MAX + 1) == "high"


def test_explainability_returns_top_factors():
    frame = generate_synthetic_training_data(n_samples=200, random_state=7)
    X = frame[list(FEATURE_NAMES)].to_numpy(dtype=float)
    y = frame["risk_target"].to_numpy(dtype=int)
    model = RiskModel()
    model.fit(X, y)

    sample = X[0]
    factors = explain_instance(model, sample, top_n=3)
    assert len(factors) == 3
    assert all("factor" in item and "weight" in item for item in factors)


def test_model_save_and_load(tmp_path):
    frame = generate_synthetic_training_data(n_samples=120, random_state=1)
    X = frame[list(FEATURE_NAMES)].to_numpy(dtype=float)
    y = frame["risk_target"].to_numpy(dtype=int)

    model = RiskModel()
    model.fit(X, y)
    model.save(tmp_path)
    loaded = RiskModel.load(tmp_path)

    prob_original = model.predict_proba(X[:5])
    prob_loaded = loaded.predict_proba(X[:5])
    np.testing.assert_allclose(prob_original, prob_loaded)


def test_predict_one_elevated_student():
    as_of = datetime(2026, 8, 12, 12, 0, 0)
    student = StudentFeatureInput(
        phq9_score=20,
        gad7_score=18,
        who5_score=6,
        checkins=[
            CheckInRecord("mood", 1, as_of - timedelta(days=1)),
            CheckInRecord("sleep", 2, as_of - timedelta(days=2)),
            CheckInRecord("routine", 2, as_of - timedelta(days=1)),
        ],
        diary_entries=[
            DiaryRecord(-0.8, {"sadness": 1, "anxiety": 1}, as_of - timedelta(days=1)),
        ],
        as_of=as_of,
    )
    features = build_feature_vector(student)
    vector = feature_dict_to_vector(features)

    frame = generate_synthetic_training_data(n_samples=400, random_state=99)
    X = frame[list(FEATURE_NAMES)].to_numpy(dtype=float)
    y = frame["risk_target"].to_numpy(dtype=int)
    model = RiskModel().fit(X, y)

    prediction = model.predict_one(vector)
    assert 0 <= prediction.score <= 100
    assert prediction.level in {"low", "moderate", "high"}
    factors = explain_instance(model, vector, top_n=5)
    assert len(factors) == 5


def test_no_fastapi_or_database_imports_in_ml_modules():
    import ast
    import importlib

    modules = [
        "ml.src.features.feature_pipeline",
        "ml.src.models.risk_model",
        "ml.src.explainability.explainability",
        "ml.src.training.train",
    ]
    forbidden = {"fastapi", "sqlalchemy", "psycopg2", "backend"}
    for module_name in modules:
        module = importlib.import_module(module_name)
        source_path = module.__file__
        assert source_path is not None
        tree = ast.parse(open(source_path, encoding="utf-8-sig").read())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0].lower())
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0].lower())
        assert forbidden.isdisjoint(imported)
