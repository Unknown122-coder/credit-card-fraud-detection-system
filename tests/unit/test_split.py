"""Unit tests for train/val/test splitting and leakage prevention."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.data.split import split_data, time_based_split, stratified_split


@pytest.fixture
def mock_dataset() -> pd.DataFrame:
    """Generate deterministic synthetic transaction dataset with class imbalance."""
    rng = np.random.default_rng(42)
    n_samples = 1000

    times = np.sort(rng.uniform(0, 172800, size=n_samples))
    amounts = rng.exponential(scale=50.0, size=n_samples)

    # Extreme imbalance: ~1% fraud in synthetic fixture
    classes = (rng.uniform(0, 1, size=n_samples) < 0.02).astype(int)

    data = {"Time": times, "Amount": amounts, "Class": classes}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1, size=n_samples)

    return pd.DataFrame(data)


def test_time_based_split_chronological_ordering(mock_dataset: pd.DataFrame) -> None:
    """Verify time-based split preserves strict temporal ordering with zero forward leakage."""
    train, val, test = time_based_split(mock_dataset)

    # Ensure train max time is strictly <= val min time
    assert train["Time"].max() <= val["Time"].min()
    # Ensure val max time is strictly <= test min time
    assert val["Time"].max() <= test["Time"].min()

    # Verify partition size integrity (70% / 15% / 15%)
    total_len = len(mock_dataset)
    assert len(train) == int(total_len * 0.70)
    assert len(val) == int(total_len * 0.15)
    assert len(test) == total_len - len(train) - len(val)


def test_stratified_split_preserves_class_ratio(mock_dataset: pd.DataFrame) -> None:
    """Verify stratified split maintains equal fraud proportions across partitions."""
    train, val, test = stratified_split(mock_dataset)

    overall_ratio = mock_dataset["Class"].mean()
    train_ratio = train["Class"].mean()
    val_ratio = val["Class"].mean()
    test_ratio = test["Class"].mean()

    # All partition ratios should closely match the overall ratio
    assert pytest.approx(train_ratio, abs=0.015) == overall_ratio
    assert pytest.approx(val_ratio, abs=0.015) == overall_ratio
    assert pytest.approx(test_ratio, abs=0.015) == overall_ratio


def test_no_row_overlap_across_splits(mock_dataset: pd.DataFrame) -> None:
    """Assert zero row overlap or duplicate indices across train, val, and test splits."""
    train, val, test = split_data(mock_dataset)

    # Compute row string hashes
    def get_hashes(df: pd.DataFrame) -> set[int]:
        return set(pd.util.hash_pandas_object(df, index=False))

    h_train = get_hashes(train)
    h_val = get_hashes(val)
    h_test = get_hashes(test)

    assert len(h_train.intersection(h_val)) == 0, "Train and validation splits share identical rows!"
    assert len(h_train.intersection(h_test)) == 0, "Train and test splits share identical rows!"
    assert len(h_val.intersection(h_test)) == 0, "Validation and test splits share identical rows!"
