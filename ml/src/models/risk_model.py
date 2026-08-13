"""
PS-87 baseline risk model wrapper.

Uses scikit-learn Logistic Regression for an explainable prototype baseline.
Independent of FastAPI and Backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.src.features.feature_pipeline import FEATURE_NAMES

RANDOM_STATE: int = 42
DEFAULT_C: float = 1.0

# IMPLEMENTATION DECISION: prototype risk-level thresholds (not clinical thresholds).
RISK_LEVEL_LOW_MAX: int = 33
RISK_LEVEL_MODERATE_MAX: int = 66

MODEL_FILENAME: str = "risk_model.joblib"
METADATA_FILENAME: str = "risk_model_metadata.joblib"


@dataclass(frozen=True)
class RiskPrediction:
    """Internal ML prediction bundle (maps to public RiskScore by Backend adapter)."""

    probability: float
    score: int
    level: str


def probability_to_score(probability: float) -> int:
    """Convert class-1 probability to an integer score in [0, 100]."""
    clamped = max(0.0, min(1.0, float(probability)))
    return int(round(clamped * 100))


def score_to_level(score: int) -> str:
    """Map integer score to low | moderate | high using prototype thresholds."""
    clamped = max(0, min(100, int(score)))
    if clamped <= RISK_LEVEL_LOW_MAX:
        return "low"
    if clamped <= RISK_LEVEL_MODERATE_MAX:
        return "moderate"
    return "high"


def build_model_pipeline(
    *,
    random_state: int = RANDOM_STATE,
    c: float = DEFAULT_C,
) -> Pipeline:
    """Create an untrained preprocessing + Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    random_state=random_state,
                    C=c,
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


class RiskModel:
    """Trainable and persistable baseline risk model."""

    def __init__(self, pipeline: Pipeline | None = None) -> None:
        self.pipeline = pipeline or build_model_pipeline()
        self.feature_names: tuple[str, ...] = FEATURE_NAMES

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RiskModel":
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features; got {X.shape[1]}."
            )
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.pipeline.predict_proba(X)
        return probabilities[:, 1]

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_one(self, feature_vector: list[float] | np.ndarray) -> RiskPrediction:
        array = np.asarray(feature_vector, dtype=float).reshape(1, -1)
        probability = float(self.predict_proba(array)[0])
        score = probability_to_score(probability)
        level = score_to_level(score)
        return RiskPrediction(probability=probability, score=score, level=level)

    def save(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, model_dir / MODEL_FILENAME)
        joblib.dump(
            {"feature_names": self.feature_names},
            model_dir / METADATA_FILENAME,
        )

    @classmethod
    def load(cls, model_dir: str | Path) -> "RiskModel":
        model_dir = Path(model_dir)
        pipeline = joblib.load(model_dir / MODEL_FILENAME)
        metadata: dict[str, Any] = joblib.load(model_dir / METADATA_FILENAME)
        model = cls(pipeline=pipeline)
        model.feature_names = tuple(metadata["feature_names"])
        return model

    def transformed_features(self, X: np.ndarray) -> np.ndarray:
        """Return scaled features as used by the classifier."""
        scaler = self.pipeline.named_steps["scaler"]
        return scaler.transform(X)

    def coefficients(self) -> np.ndarray:
        """Return classifier coefficients aligned with FEATURE_NAMES."""
        classifier = self.pipeline.named_steps["classifier"]
        return np.asarray(classifier.coef_[0], dtype=float)
