"""Evaluation metrics, cost functions, and bootstrap confidence intervals.

Provides cost-sensitive evaluation tailored for credit-card fraud detection
where false negatives (missed fraud) are far more expensive than false
positives (legitimate transactions flagged for review).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ── Core metric computation ──────────────────────────────────────────────


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    amounts: Optional[np.ndarray] = None,
    fp_cost: float = 10.0,
    fn_base_cost: float = 25.0,
) -> dict[str, float]:
    """Compute a comprehensive set of classification metrics.

    Threshold-independent metrics (PR-AUC, ROC-AUC, Brier score) are
    computed directly from predicted probabilities.  All other metrics
    use the binary predictions obtained by applying *threshold*.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 = legitimate, 1 = fraud).
    y_prob:
        Predicted probability of the positive (fraud) class.
    threshold:
        Decision threshold applied to *y_prob* to produce binary
        predictions.
    amounts:
        Per-transaction monetary amounts.  When provided, false-negative
        cost includes the transaction amount.
    fp_cost:
        Fixed cost incurred for each false positive.
    fn_base_cost:
        Base cost incurred for each false negative (added to the
        transaction amount when *amounts* is provided).

    Returns
    -------
    dict[str, float]
        Dictionary with keys: ``pr_auc``, ``roc_auc``, ``precision``,
        ``recall``, ``f1``, ``f2``, ``expected_cost``, ``brier_score``.
    """
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "expected_cost": compute_expected_cost(
            y_true, y_pred, amounts=amounts, fp_cost=fp_cost, fn_base_cost=fn_base_cost
        ),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


# ── Cost-sensitive evaluation ────────────────────────────────────────────


def compute_expected_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amounts: Optional[np.ndarray] = None,
    fp_cost: float = 10.0,
    fn_base_cost: float = 25.0,
) -> float:
    """Calculate total expected cost given a cost matrix.

    Cost model:
    * **False positive** → fixed cost of *fp_cost* per instance (manual
      review of a legitimate transaction).
    * **False negative** → ``amount_i + fn_base_cost`` per instance when
      *amounts* is available; otherwise just *fn_base_cost*.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_pred:
        Predicted binary labels.
    amounts:
        Per-transaction monetary amounts.
    fp_cost:
        Cost per false positive.
    fn_base_cost:
        Base cost per false negative.

    Returns
    -------
    float
        Total expected cost across all samples.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)

    total_fp_cost = float(fp_mask.sum()) * fp_cost

    if amounts is not None:
        amounts = np.asarray(amounts)
        total_fn_cost = float((amounts[fn_mask] + fn_base_cost).sum())
    else:
        total_fn_cost = float(fn_mask.sum()) * fn_base_cost

    return total_fp_cost + total_fn_cost


# ── Optimal threshold search ────────────────────────────────────────────


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: Optional[np.ndarray] = None,
    fp_cost: float = 10.0,
    fn_base_cost: float = 25.0,
    n_steps: int = 1000,
) -> tuple[float, float]:
    """Find the decision threshold that minimises expected cost.

    Sweeps thresholds linearly from 0.01 to 0.99 and evaluates
    :func:`compute_expected_cost` at each point.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_prob:
        Predicted probabilities for the positive class.
    amounts:
        Per-transaction monetary amounts.
    fp_cost:
        Cost per false positive.
    fn_base_cost:
        Base cost per false negative.
    n_steps:
        Number of threshold values to evaluate.

    Returns
    -------
    tuple[float, float]
        ``(optimal_threshold, minimum_cost)``.
    """
    # Sweep thresholds covering extreme low base rates (1e-4 to 0.99)
    # using log-spaced low thresholds and linear high thresholds
    min_prob = max(
        1e-4, float(np.min(y_prob[y_prob > 0])) if np.any(y_prob > 0) else 1e-4
    )
    low_steps = max(20, n_steps // 4)
    high_steps = max(20, n_steps - low_steps)
    low_sweep = np.geomspace(min_prob, 0.05, low_steps)
    high_sweep = np.linspace(0.05, 0.99, high_steps)
    thresholds: np.ndarray = np.unique(
        np.clip(
            np.concatenate(
                (np.array([1e-4]), low_sweep, high_sweep, np.array([0.99]))
            ),
            1e-4,
            0.99,
        )
    )
    best_threshold: float = 0.5
    best_cost: float = float("inf")

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cost = compute_expected_cost(
            y_true, y_pred, amounts=amounts, fp_cost=fp_cost, fn_base_cost=fn_base_cost
        )
        if cost < best_cost:
            best_cost = cost
            best_threshold = float(t)

    logger.info(
        "Optimal threshold found: %.4f (min cost: %.2f)", best_threshold, best_cost
    )
    return best_threshold, best_cost


# ── Bootstrap confidence intervals ──────────────────────────────────────


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    amounts: Optional[np.ndarray] = None,
    n_iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    fp_cost: float = 10.0,
    fn_base_cost: float = 25.0,
) -> dict[str, tuple[float, float, float]]:
    """Compute bootstrap confidence intervals for all metrics.

    Uses stratified resampling with replacement so that each bootstrap
    sample preserves the positive-class ratio of the original data.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_prob:
        Predicted probabilities for the positive class.
    threshold:
        Decision threshold to apply.
    amounts:
        Per-transaction monetary amounts.
    n_iterations:
        Number of bootstrap iterations.
    confidence:
        Confidence level (e.g. 0.95 for 95 %).
    seed:
        Random seed for reproducibility.
    fp_cost:
        Cost per false positive.
    fn_base_cost:
        Base cost per false negative.

    Returns
    -------
    dict[str, tuple[float, float, float]]
        Mapping from metric name to ``(mean, lower_ci, upper_ci)``.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    metric_names = [
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "f2",
        "expected_cost",
        "brier_score",
    ]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        logger.warning("Bootstrap requires both classes; returning point estimates.")
        base_metrics = compute_all_metrics(
            y_true, y_prob, threshold, amounts, fp_cost, fn_base_cost
        )
        return {
            name: (base_metrics[name], base_metrics[name], base_metrics[name])
            for name in metric_names
        }

    results: dict[str, list[float]] = {name: [] for name in metric_names}

    logger.info(
        "Running %d bootstrap iterations (confidence=%.2f)", n_iterations, confidence
    )

    for _ in range(n_iterations):
        # Stratified resample: independently resample positives and negatives
        boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        boot_idx = np.concatenate([boot_pos, boot_neg])

        boot_y_true = y_true[boot_idx]
        boot_y_prob = y_prob[boot_idx]
        boot_amounts = amounts[boot_idx] if amounts is not None else None

        metrics = compute_all_metrics(
            boot_y_true,
            boot_y_prob,
            threshold=threshold,
            amounts=boot_amounts,
            fp_cost=fp_cost,
            fn_base_cost=fn_base_cost,
        )
        for name in metric_names:
            results[name].append(metrics[name])

    alpha = 1.0 - confidence
    lower_pct = (alpha / 2.0) * 100
    upper_pct = (1.0 - alpha / 2.0) * 100

    ci_results: dict[str, tuple[float, float, float]] = {}
    for name in metric_names:
        values = np.array(results[name])
        mean_val = float(np.mean(values))
        lower_ci = float(np.percentile(values, lower_pct))
        upper_ci = float(np.percentile(values, upper_pct))
        ci_results[name] = (mean_val, lower_ci, upper_ci)
        logger.debug(
            "  %s: mean=%.4f, CI=[%.4f, %.4f]", name, mean_val, lower_ci, upper_ci
        )

    logger.info(
        "Bootstrap confidence intervals computed for %d metrics", len(ci_results)
    )
    return ci_results


# ── Threshold curve for UI ──────────────────────────────────────────────


def compute_threshold_metrics_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: Optional[np.ndarray] = None,
    fp_cost: float = 10.0,
    fn_base_cost: float = 25.0,
    n_steps: int = 100,
) -> pd.DataFrame:
    """Compute metrics across a range of thresholds for UI visualisation.

    Designed to back a threshold-slider widget: at each threshold the
    caller can look up precision, recall, and cost in the resulting
    DataFrame.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_prob:
        Predicted probabilities for the positive class.
    amounts:
        Per-transaction monetary amounts.
    fp_cost:
        Cost per false positive.
    fn_base_cost:
        Base cost per false negative.
    n_steps:
        Number of evenly-spaced threshold values.

    Returns
    -------
    pd.DataFrame
        Columns: ``threshold``, ``precision``, ``recall``, ``f1``,
        ``f2``, ``expected_cost``, ``tp``, ``fp``, ``fn``, ``tn``.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thresholds = np.linspace(0.01, 0.99, n_steps)

    rows: list[dict[str, float]] = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(t),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
                "expected_cost": compute_expected_cost(
                    y_true,
                    y_pred,
                    amounts=amounts,
                    fp_cost=fp_cost,
                    fn_base_cost=fn_base_cost,
                ),
                "tp": float(tp),
                "fp": float(fp),
                "fn": float(fn),
                "tn": float(tn),
            }
        )

    logger.info("Threshold metrics curve computed with %d steps", n_steps)
    return pd.DataFrame(rows)
