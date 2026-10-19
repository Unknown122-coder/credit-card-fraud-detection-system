"""Unit tests for cost function monotonicity and threshold sensitivity properties."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import find_optimal_threshold


def test_threshold_shifts_lower_when_fn_cost_increases() -> None:
    """As missed fraud cost increases, optimal threshold must shift lower to catch more fraud."""
    rng = np.random.default_rng(42)
    n = 1000
    y_true = (rng.uniform(0, 1, n) < 0.05).astype(int)
    # Probabilities with positive signal
    y_prob = np.clip(y_true * 0.6 + rng.uniform(0.01, 0.4, n), 1e-4, 0.99)
    amounts = rng.uniform(50.0, 500.0, n)

    # Low FN base cost
    t_low_fn, _ = find_optimal_threshold(
        y_true, y_prob, amounts=amounts, fp_cost=20.0, fn_base_cost=10.0, n_steps=200
    )

    # High FN base cost (catastrophic fraud penalty)
    t_high_fn, _ = find_optimal_threshold(
        y_true, y_prob, amounts=amounts, fp_cost=20.0, fn_base_cost=1000.0, n_steps=200
    )

    assert (
        t_high_fn <= t_low_fn
    ), "Elevated FN penalty must drive optimal threshold lower or equal"


def test_threshold_shifts_higher_when_fp_cost_increases() -> None:
    """As false alarm review cost increases, optimal threshold must shift higher to decline fewer users."""
    rng = np.random.default_rng(42)
    n = 1000
    y_true = (rng.uniform(0, 1, n) < 0.05).astype(int)
    y_prob = np.clip(y_true * 0.6 + rng.uniform(0.01, 0.4, n), 1e-4, 0.99)
    amounts = rng.uniform(50.0, 200.0, n)

    # Low FP cost
    t_low_fp, _ = find_optimal_threshold(
        y_true, y_prob, amounts=amounts, fp_cost=5.0, fn_base_cost=100.0, n_steps=200
    )

    # Very high FP cost (declining legitimate users is severely penalized)
    t_high_fp, _ = find_optimal_threshold(
        y_true, y_prob, amounts=amounts, fp_cost=500.0, fn_base_cost=100.0, n_steps=200
    )

    assert (
        t_high_fp >= t_low_fp
    ), "Elevated FP friction cost must drive optimal threshold higher or equal"


def test_zero_fraud_leads_to_upper_boundary_threshold() -> None:
    """When all transactions are legitimate, cost-minimizing threshold must avoid all false alarms."""
    y_true = np.zeros(200, dtype=int)
    y_prob = np.linspace(0.05, 0.85, 200)

    opt_thresh, min_cost = find_optimal_threshold(
        y_true, y_prob, fp_cost=10.0, fn_base_cost=50.0, n_steps=100
    )

    # Because no fraud exists, optimal policy is to predict 0 for all (threshold > max(y_prob))
    assert min_cost == 0.0
    assert opt_thresh >= np.max(y_prob) or pytest.approx(
        opt_thresh, abs=0.05
    ) == np.max(y_prob)
