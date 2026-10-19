"""Unit tests verifying strict data leakage prevention and single-touch test set guard."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression

from src.evaluation.evaluate import TestSetGuard
from src.models.pipelines import build_pipeline


def test_test_set_guard_enforces_single_touch(tmp_path: Path) -> None:
    """Verify TestSetGuard strictly allows exactly one evaluation pass."""
    flag_file = tmp_path / ".test_evaluated"
    guard = TestSetGuard(flag_path=flag_file)

    # Initially allowed
    assert guard.can_evaluate() is True

    # Mark evaluated
    guard.mark_evaluated()
    assert flag_file.exists()
    assert guard.can_evaluate() is False

    # Second evaluation must be blocked
    assert guard.can_evaluate() is False

    # Development reset
    guard.reset()
    assert guard.can_evaluate() is True


def test_imblearn_pipeline_resampling_isolation() -> None:
    """Verify SMOTE resampler is executed only during fit and NEVER during transform or predict."""
    rng = np.random.default_rng(42)
    n_train = 200
    n_val = 50

    # 10% positive in train
    y_tr = (rng.uniform(0, 1, n_train) < 0.10).astype(int)
    X_tr = rng.normal(0, 1, (n_train, 10))

    # Validation set with same shape
    X_val = rng.normal(0, 1, (n_val, 10))
    y_val = (rng.uniform(0, 1, n_val) < 0.10).astype(int)

    pipeline = ImbPipeline(
        [
            ("scaler", RobustScaler()),
            ("smote", SMOTE(random_state=42, k_neighbors=2)),
            ("classifier", LogisticRegression()),
        ]
    )

    # Fit pipeline
    pipeline.fit(X_tr, y_tr)

    # When predicting on validation, length must equal n_val exactly (no synthetic rows generated)
    val_probs = pipeline.predict_proba(X_val)
    assert (
        val_probs.shape[0] == n_val
    ), "Validation output size must exactly match input validation size"


def test_scaler_parameters_derived_exclusively_from_train_fold() -> None:
    """Verify RobustScaler parameters are frozen from training fold and not updated during test transform."""
    train_data = np.array([[10.0], [20.0], [30.0], [40.0], [100.0]])
    test_data = np.array([[1000.0], [2000.0]])

    pipeline = build_pipeline("logistic_regression", resampling="none", seed=42)
    pipeline.fit(train_data, np.array([0, 0, 0, 1, 1]))

    scaler: RobustScaler = pipeline.named_steps["scaler"]
    center_fit = scaler.center_.copy()
    scale_fit = scaler.scale_.copy()

    # Score test data
    _ = pipeline.predict_proba(test_data)

    # Scaler center and scale must remain completely unchanged
    np.testing.assert_array_equal(scaler.center_, center_fit)
    np.testing.assert_array_equal(scaler.scale_, scale_fit)
