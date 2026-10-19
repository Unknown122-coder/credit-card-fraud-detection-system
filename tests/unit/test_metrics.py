"""Unit tests for cost functions, metrics, and threshold optimization."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    bootstrap_confidence_intervals,
    compute_all_metrics,
    compute_expected_cost,
    find_optimal_threshold,
)


def test_compute_expected_cost_asymmetric_calculation() -> None:
    """Verify asymmetric cost: FN = Amount + 25, FP = 10."""
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array(
        [0, 1, 1, 0]
    )  # Index 0: FN; Index 1: FP; Index 2: TP; Index 3: TN
    amounts = np.array([100.0, 50.0, 200.0, 30.0])

    # Expected:
    # idx 0: FN -> 100.0 + 25.0 = 125.0
    # idx 1: FP -> 10.0
    # idx 2: TP -> 0.0
    # idx 3: TN -> 0.0
    # Total = 135.0
    cost = compute_expected_cost(
        y_true=y_true,
        y_pred=y_pred,
        amounts=amounts,
        fp_cost=10.0,
        fn_base_cost=25.0,
    )
    assert cost == 135.0


def test_find_optimal_threshold_minimizes_cost() -> None:
    """Ensure threshold search picks a threshold that strictly reduces or matches expected loss vs 0.50."""
    rng = np.random.default_rng(42)
    n = 500
    y_true = (rng.uniform(0, 1, n) < 0.05).astype(int)
    # Simulated calibrated probabilities with moderate signal
    y_prob = np.clip(y_true * 0.7 + rng.normal(0.1, 0.15, n), 0.0, 1.0)
    amounts = rng.uniform(10.0, 300.0, n)

    opt_thresh, min_cost = find_optimal_threshold(
        y_true=y_true,
        y_prob=y_prob,
        amounts=amounts,
        fp_cost=10.0,
        fn_base_cost=25.0,
        n_steps=100,
    )

    # Cost at default 0.5 threshold
    default_pred = (y_prob >= 0.5).astype(int)
    default_cost = compute_expected_cost(y_true, default_pred, amounts, 10.0, 25.0)

    assert (
        min_cost <= default_cost
    ), "Optimized threshold should achieve equal or lower cost than 0.50"
    assert 0.01 <= opt_thresh <= 0.99


def test_compute_all_metrics_keys_and_ranges() -> None:
    """Verify compute_all_metrics produces valid bounded metric outputs."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])

    metrics = compute_all_metrics(y_true=y_true, y_prob=y_prob, threshold=0.5)

    required_keys = {
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "f2",
        "expected_cost",
        "brier_score",
    }
    assert required_keys.issubset(metrics.keys())

    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["f2"] <= 1.0
    assert metrics["brier_score"] >= 0.0


def test_bootstrap_confidence_intervals_format() -> None:
    """Verify bootstrap returns valid (mean, lower, upper) intervals."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=100)
    y_prob = rng.uniform(0, 1, size=100)

    cis = bootstrap_confidence_intervals(
        y_true=y_true,
        y_prob=y_prob,
        threshold=0.5,
        n_iterations=50,  # Fast for unit test
        confidence=0.95,
        seed=42,
    )

    for metric_name, (mean_val, lower, upper) in cis.items():
        assert lower <= mean_val <= upper or pytest.approx(lower, abs=1e-4) == mean_val
