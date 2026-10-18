"""Unit tests verifying probability calibration sanity and isotonic monotonicity."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.evaluation.calibration import calibrate_model, compare_calibration_methods


@pytest.fixture
def uncalibrated_synthetic_data() -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    """Train uncalibrated model with distorted margins."""
    rng = np.random.default_rng(42)
    n = 400
    X = rng.normal(0, 1, (n, 5))
    # Extreme class imbalance: 5% positive
    y = (rng.uniform(0, 1, n) < 0.05).astype(int)

    clf = LogisticRegression(class_weight={0: 1, 1: 50}, random_state=42)
    clf.fit(X, y)

    X_val = rng.normal(0, 1, (n, 5))
    y_val = (rng.uniform(0, 1, n) < 0.05).astype(int)

    return clf, X_val, y_val


def test_calibrate_model_isotonic_produces_valid_probabilities(uncalibrated_synthetic_data: tuple) -> None:
    """Verify calibrated model outputs bounded [0, 1] probabilities."""
    clf, X_val, y_val = uncalibrated_synthetic_data
    calibrated = calibrate_model(clf, X_val, y_val, method="isotonic")

    probs = calibrated.predict_proba(X_val)[:, 1]
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert probs.shape[0] == len(y_val)


def test_compare_calibration_methods_reports_valid_structure(uncalibrated_synthetic_data: tuple) -> None:
    """Verify compare_calibration_methods evaluates both isotonic and sigmoid methods."""
    clf, X_val, y_val = uncalibrated_synthetic_data
    comparison = compare_calibration_methods(clf, X_val, y_val)

    assert "isotonic" in comparison
    assert "sigmoid" in comparison

    for method in ("isotonic", "sigmoid"):
        assert "brier_score" in comparison[method]
        assert comparison[method]["brier_score"] >= 0.0
        assert "calibrated_model" in comparison[method]
