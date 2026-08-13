"""
PS-87 ML training pipeline.

Loads synthetic training data, fits preprocessing + Logistic Regression,
evaluates on a held-out test split, and saves the model artifact.

Synthetic target generation uses a latent risk process over multiple signals
with controlled noise. It is NOT a deterministic function of one feature.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from ml.src.features.feature_pipeline import (
    FEATURE_NAMES,
    CheckInRecord,
    DiaryRecord,
    StudentFeatureInput,
    build_feature_vector,
    feature_dict_to_vector,
    validate_feature_dict,
)
from ml.src.models.risk_model import METADATA_FILENAME, MODEL_FILENAME, RiskModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "synthetic" / "training_data.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "ml" / "models"

RANDOM_STATE = 42
DEFAULT_N_SAMPLES = 1800
REFERENCE_AS_OF = datetime(2026, 8, 12, 12, 0, 0)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _sample_questionnaire_score(
    rng: np.random.Generator,
    maximum: int,
    risk_bias: float,
) -> int:
    """Sample a questionnaire score skewed by latent risk."""
    mean = maximum * (0.25 + 0.55 * risk_bias)
    score = int(round(rng.normal(mean, maximum * 0.12)))
    return int(max(0, min(maximum, score)))


def _sample_checkin_value(rng: np.random.Generator, risk_bias: float) -> int:
    """Sample a 1–5 check-in value; higher risk_bias → worse values."""
    weights = np.array(
        [
            0.05 + 0.30 * risk_bias,
            0.10 + 0.20 * risk_bias,
            0.20,
            0.25 - 0.10 * risk_bias,
            0.40 - 0.40 * risk_bias,
        ],
        dtype=float,
    )
    weights = np.clip(weights, 0.01, None)
    weights /= weights.sum()
    return int(rng.choice(np.arange(1, 6), p=weights))


def _sample_emotion_flags(
    rng: np.random.Generator,
    risk_bias: float,
) -> dict[str, int]:
    negative = ("sadness", "anxiety", "loneliness", "exhaustion", "overwhelm", "fear")
    positive = ("joy", "calm", "hope")
    flags = {name: 0 for name in (*negative, *positive, "anger", "guilt")}
    for name in negative:
        if rng.random() < 0.10 + 0.50 * risk_bias:
            flags[name] = 1
    for name in positive:
        if rng.random() < 0.20 - 0.12 * risk_bias:
            flags[name] = 1
    if rng.random() < 0.05 + 0.20 * risk_bias:
        flags["anger"] = 1
    if rng.random() < 0.05 + 0.15 * risk_bias:
        flags["guilt"] = 1
    return flags


def _sample_latent_target(
    rng: np.random.Generator,
    *,
    phq9_score: int | None,
    gad7_score: int | None,
    who5_score: int | None,
    mood_values: list[int],
    sleep_values: list[int],
    routine_values: list[int],
    sentiment_scores: list[float],
    emotion_flags_list: list[dict[str, int]],
) -> int:
    """
    Generate risk_target from a latent process combining multiple signals.

    Uses underlying generative values plus noise; not a single-feature rule.
    """
    latent = 0.0
    weight_total = 0.0

    if phq9_score is not None:
        latent += 0.22 * (phq9_score / 27)
        weight_total += 0.22
    if gad7_score is not None:
        latent += 0.22 * (gad7_score / 21)
        weight_total += 0.22
    if who5_score is not None:
        latent += 0.18 * (1 - who5_score / 25)
        weight_total += 0.18

    if mood_values:
        mood_risk = np.mean([(5 - v) / 4 for v in mood_values])
        latent += 0.12 * mood_risk
        weight_total += 0.12
    if sleep_values:
        sleep_risk = np.mean([(5 - v) / 4 for v in sleep_values])
        latent += 0.10 * sleep_risk
        weight_total += 0.10
    if routine_values:
        routine_risk = np.mean([(5 - v) / 4 for v in routine_values])
        latent += 0.08 * routine_risk
        weight_total += 0.08
    if sentiment_scores:
        sentiment_risk = np.mean([(1 - s) / 2 for s in sentiment_scores])
        latent += 0.08 * sentiment_risk
        weight_total += 0.08

    negative_emotion_count = 0
    for flags in emotion_flags_list:
        negative_emotion_count += sum(
            flags.get(name, 0)
            for name in ("sadness", "anxiety", "loneliness", "exhaustion", "overwhelm")
        )
    if emotion_flags_list:
        latent += 0.05 * min(1.0, negative_emotion_count / len(emotion_flags_list))
        weight_total += 0.05

    if weight_total > 0:
        latent /= weight_total

    latent += rng.normal(0.0, 0.12)
    probability = _sigmoid((latent - 0.52) * 5.0)
    return int(rng.random() < probability)


def generate_synthetic_training_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    *,
    random_state: int = RANDOM_STATE,
    as_of: datetime = REFERENCE_AS_OF,
) -> pd.DataFrame:
    """
    Generate reproducible synthetic training rows.

    Each row contains the fixed ML feature vector plus internal ``risk_target``.
    """
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | int]] = []

    for _ in range(n_samples):
        latent_risk = float(rng.beta(2.0, 2.5))

        phq9_missing = rng.random() < 0.12
        gad7_missing = rng.random() < 0.12
        who5_missing = rng.random() < 0.12

        phq9_score = None if phq9_missing else _sample_questionnaire_score(rng, 27, latent_risk)
        gad7_score = None if gad7_missing else _sample_questionnaire_score(rng, 21, latent_risk)
        who5_score = None if who5_missing else _sample_questionnaire_score(rng, 25, 1 - latent_risk)

        checkins: list[CheckInRecord] = []
        mood_values: list[int] = []
        sleep_values: list[int] = []
        routine_values: list[int] = []

        for checkin_type, bucket in (
            ("mood", mood_values),
            ("sleep", sleep_values),
            ("routine", routine_values),
        ):
            if rng.random() < 0.18:
                continue
            n_entries = int(rng.integers(1, 8))
            for day_offset in sorted(rng.choice(7, size=n_entries, replace=False)):
                value = _sample_checkin_value(rng, latent_risk)
                bucket.append(value)
                timestamp = as_of - timedelta(days=int(day_offset), hours=int(rng.integers(0, 24)))
                checkins.append(CheckInRecord(type=checkin_type, value=value, timestamp=timestamp))

        diary_entries: list[DiaryRecord] = []
        sentiment_scores: list[float] = []
        emotion_flags_list: list[dict[str, int]] = []

        if rng.random() >= 0.20:
            n_diaries = int(rng.integers(1, 6))
            for day_offset in sorted(rng.choice(7, size=n_diaries, replace=False)):
                sentiment = float(np.clip(rng.normal(0.35 - 0.9 * latent_risk, 0.35), -1, 1))
                flags = _sample_emotion_flags(rng, latent_risk)
                sentiment_scores.append(sentiment)
                emotion_flags_list.append(flags)
                timestamp = as_of - timedelta(days=int(day_offset), hours=int(rng.integers(0, 24)))
                diary_entries.append(
                    DiaryRecord(
                        sentiment_score=round(sentiment, 4),
                        emotion_flags=flags,
                        timestamp=timestamp,
                    )
                )

        student = StudentFeatureInput(
            phq9_score=phq9_score,
            gad7_score=gad7_score,
            who5_score=who5_score,
            checkins=checkins,
            diary_entries=diary_entries,
            as_of=as_of,
        )
        features = build_feature_vector(student)
        validate_feature_dict(features)

        target = _sample_latent_target(
            rng,
            phq9_score=phq9_score,
            gad7_score=gad7_score,
            who5_score=who5_score,
            mood_values=mood_values,
            sleep_values=sleep_values,
            routine_values=routine_values,
            sentiment_scores=sentiment_scores,
            emotion_flags_list=emotion_flags_list,
        )

        row = {name: features[name] for name in FEATURE_NAMES}
        row["risk_target"] = target
        rows.append(row)

    return pd.DataFrame(rows)


def write_synthetic_training_data(
    output_path: str | Path = DEFAULT_DATA_PATH,
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate and persist synthetic training data."""
    frame = generate_synthetic_training_data(
        n_samples=n_samples,
        random_state=random_state,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def load_training_frame(data_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    frame = pd.read_csv(data_path)
    expected = list(FEATURE_NAMES) + ["risk_target"]
    missing = set(expected) - set(frame.columns)
    extra = set(frame.columns) - set(expected)
    if missing or extra:
        raise ValueError(
            f"Training data columns mismatch. Missing: {sorted(missing)}; extra: {sorted(extra)}."
        )
    return frame


def train_model(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    *,
    random_state: int = RANDOM_STATE,
    test_size: float = 0.2,
) -> dict[str, float | str | int]:
    """Train, evaluate, and save the baseline risk model."""
    frame = load_training_frame(data_path)
    X = frame[list(FEATURE_NAMES)].to_numpy(dtype=float)
    y = frame["risk_target"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = RiskModel()
    model.fit(X_train, y_train)

    y_pred = model.predict_class(X_test)
    y_prob = model.predict_proba(X_test)

    metrics: dict[str, float | str | int] = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "class_0_count": int((y == 0).sum()),
        "class_1_count": int((y == 1).sum()),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    model_dir = Path(model_dir)
    model.save(model_dir)

    return metrics


def print_training_report(metrics: dict[str, float | str | int], model_dir: Path) -> None:
    print("=== PS-87 ML Training Report ===")
    print(f"Train size: {metrics['train_size']}")
    print(f"Test size: {metrics['test_size']}")
    print(f"Class 0 count: {metrics['class_0_count']}")
    print(f"Class 1 count: {metrics['class_1_count']}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print()
    print(metrics["classification_report"])
    print(f"Saved model artifact: {model_dir / MODEL_FILENAME}")
    print(f"Saved metadata:         {model_dir / METADATA_FILENAME}")


def main() -> None:
    write_synthetic_training_data()
    metrics = train_model()
    print_training_report(metrics, DEFAULT_MODEL_DIR)


if __name__ == "__main__":
    main()
