"""Unit tests for feature transformers and preprocessing pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.transformers import (
    HourOfDayTransformer,
    LogAmountTransformer,
    build_preprocessor,
)


def test_log_amount_transformer_transform() -> None:
    """Verify LogAmountTransformer correctly computes log1p without negative amounts."""
    df = pd.DataFrame({"Amount": [0.0, 9.0, 99.0]})
    transformer = LogAmountTransformer()
    transformed = transformer.fit_transform(df)

    expected = np.log1p([0.0, 9.0, 99.0])
    np.testing.assert_allclose(transformed["Amount"].values, expected, rtol=1e-5)


def test_hour_of_day_transformer_calculation() -> None:
    """Verify HourOfDayTransformer accurately extracts cyclical 0-23 hours."""
    # 0s -> 0.0 hr, 3600s -> 1.0 hr, 86400s -> 0.0 hr
    df = pd.DataFrame({"Time": [0.0, 3600.0, 86400.0, 90000.0]})
    transformer = HourOfDayTransformer(drop_original_time=False)
    transformed = transformer.fit_transform(df)

    assert "HourOfDay" in transformed.columns
    assert "Time" in transformed.columns

    expected_hours = [0.0, 1.0, 0.0, 1.0]
    np.testing.assert_allclose(transformed["HourOfDay"].values, expected_hours)


def test_preprocessor_pipeline_fit_transform() -> None:
    """Verify full preprocessor pipeline transforms arrays with RobustScaler and preserves sample count."""
    rng = np.random.default_rng(42)
    n = 100
    data = {
        "Time": rng.uniform(0, 100000, n),
        "Amount": rng.exponential(50, n),
    }
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1, n)

    df = pd.DataFrame(data)

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(df)

    assert transformed.shape[0] == n
    assert not np.isnan(transformed).any()
