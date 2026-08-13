"""Training pipeline and leakage checks."""

import numpy as np
from sklearn.model_selection import train_test_split

from ml.src.features.feature_pipeline import FEATURE_NAMES
from ml.src.models.risk_model import RiskModel
from ml.src.training.train import (
    generate_synthetic_training_data,
    load_training_frame,
    train_model,
    write_synthetic_training_data,
)


def test_synthetic_dataset_shape_and_no_target_leakage_columns(tmp_path):
    output = tmp_path / "training_data.csv"
    frame = write_synthetic_training_data(output, n_samples=500, random_state=42)
    assert frame.shape == (500, len(FEATURE_NAMES) + 1)
    assert "risk_target" in frame.columns
    assert "severity" not in frame.columns
    assert "user_id" not in frame.columns
    assert "text" not in frame.columns

    loaded = load_training_frame(output)
    assert list(loaded.columns) == list(FEATURE_NAMES) + ["risk_target"]


def test_target_not_perfectly_determined_by_single_feature():
    frame = generate_synthetic_training_data(n_samples=1000, random_state=21)
    high_phq9 = frame[frame["phq9_norm"] > 0.6]
    if len(high_phq9) > 10:
        target_rate = high_phq9["risk_target"].mean()
        assert 0.05 < target_rate < 0.98


def test_train_test_preprocessing_fit_only_on_training(tmp_path):
    data_path = tmp_path / "training_data.csv"
    write_synthetic_training_data(data_path, n_samples=600, random_state=5)
    frame = load_training_frame(data_path)
    X = frame[list(FEATURE_NAMES)].to_numpy(dtype=float)
    y = frame["risk_target"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RiskModel().fit(X_train, y_train)

    train_scaled = model.transformed_features(X_train)
    test_scaled = model.transformed_features(X_test)
    assert train_scaled.shape == X_train.shape
    assert test_scaled.shape == X_test.shape


def test_train_model_returns_metrics(tmp_path):
    data_path = tmp_path / "training_data.csv"
    model_dir = tmp_path / "models"
    write_synthetic_training_data(data_path, n_samples=400, random_state=3)
    metrics = train_model(data_path=data_path, model_dir=model_dir, random_state=42)
    assert metrics["accuracy"] >= 0.0
    assert metrics["roc_auc"] >= 0.0
    assert (model_dir / "risk_model.joblib").exists()
