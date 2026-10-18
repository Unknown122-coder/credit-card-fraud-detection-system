"""Probability calibration using CalibratedClassifierCV.

Provides post-hoc probability calibration for fraud-detection models so
that predicted probabilities reflect true event frequencies — critical
for cost-sensitive decision-making where the threshold is chosen to
minimise expected monetary loss.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


# ── Model calibration ───────────────────────────────────────────────────


def calibrate_model(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "isotonic",
) -> CalibratedClassifierCV:
    """Calibrate a pre-fitted model on held-out validation data.

    Wraps *model* in a :class:`~sklearn.calibration.CalibratedClassifierCV`
    with ``cv='prefit'`` so the underlying estimator is **not** re-trained.

    Parameters
    ----------
    model:
        A fitted scikit-learn-compatible classifier that exposes
        ``predict_proba`` or ``decision_function``.
    X_val:
        Feature matrix for calibration (validation split).
    y_val:
        Ground-truth labels for calibration.
    method:
        Calibration mapping — ``'isotonic'`` (non-parametric) or
        ``'sigmoid'`` (Platt scaling).

    Returns
    -------
    CalibratedClassifierCV
        Calibrated model ready for ``predict_proba`` calls.
    """
    logger.info("Calibrating model with method='%s' on %d samples", method, len(y_val))
    calibrated = CalibratedClassifierCV(estimator=model, cv="prefit", method=method)
    calibrated.fit(X_val, y_val)
    logger.info("Calibration complete")
    return calibrated


# ── Comparison of calibration methods ────────────────────────────────────


def compare_calibration_methods(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Fit both isotonic and sigmoid calibration and compare.

    For each method the function reports the Brier score and the data
    needed to draw a reliability diagram (fraction of positives and mean
    predicted probability per bin).

    Parameters
    ----------
    model:
        A fitted classifier.
    X_val:
        Validation feature matrix.
    y_val:
        Validation labels.

    Returns
    -------
    dict[str, dict[str, Any]]
        Outer keys are ``'isotonic'`` and ``'sigmoid'``.  Each inner
        dict has:

        * ``brier_score`` — scalar Brier score.
        * ``calibrated_model`` — the fitted CalibratedClassifierCV.
        * ``y_prob`` — calibrated predicted probabilities.
        * ``fraction_of_positives`` — array for reliability diagram.
        * ``mean_predicted_value`` — array for reliability diagram.
    """
    results: dict[str, dict[str, Any]] = {}

    for method in ("isotonic", "sigmoid"):
        cal_model = calibrate_model(model, X_val, y_val, method=method)
        y_prob = cal_model.predict_proba(X_val)[:, 1]
        brier = float(brier_score_loss(y_val, y_prob))

        fraction_pos, mean_pred = calibration_curve(y_val, y_prob, n_bins=10)

        results[method] = {
            "brier_score": brier,
            "calibrated_model": cal_model,
            "y_prob": y_prob,
            "fraction_of_positives": fraction_pos,
            "mean_predicted_value": mean_pred,
        }
        logger.info("Calibration method='%s': Brier=%.6f", method, brier)

    best_method = min(results, key=lambda m: results[m]["brier_score"])
    logger.info(
        "Best calibration method: '%s' (Brier=%.6f)",
        best_method,
        results[best_method]["brier_score"],
    )
    return results


# ── Calibration curve plotting ──────────────────────────────────────────


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob_uncalibrated: np.ndarray,
    y_prob_calibrated: np.ndarray,
    method: str,
    save_path: Optional[str] = None,
) -> None:
    """Plot reliability diagram comparing uncalibrated vs calibrated probs.

    Draws the perfectly-calibrated diagonal, the uncalibrated curve, and
    the calibrated curve side-by-side.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_prob_uncalibrated:
        Uncalibrated predicted probabilities.
    y_prob_calibrated:
        Calibrated predicted probabilities.
    method:
        Name of the calibration method used (for labelling).
    save_path:
        If provided, the figure is saved to this path.  Parent
        directories are created automatically.
    """
    frac_uncal, mean_uncal = calibration_curve(y_true, y_prob_uncalibrated, n_bins=10)
    frac_cal, mean_cal = calibration_curve(y_true, y_prob_calibrated, n_bins=10)

    brier_uncal = brier_score_loss(y_true, y_prob_uncalibrated)
    brier_cal = brier_score_loss(y_true, y_prob_calibrated)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -- Reliability diagram --
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(
        mean_uncal,
        frac_uncal,
        "s-",
        label=f"Uncalibrated (Brier={brier_uncal:.4f})",
    )
    ax.plot(
        mean_cal,
        frac_cal,
        "o-",
        label=f"{method.capitalize()} (Brier={brier_cal:.4f})",
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability Diagram")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # -- Histogram of predicted probabilities --
    ax2 = axes[1]
    ax2.hist(
        y_prob_uncalibrated,
        bins=50,
        alpha=0.5,
        label="Uncalibrated",
        density=True,
    )
    ax2.hist(
        y_prob_calibrated,
        bins=50,
        alpha=0.5,
        label=f"{method.capitalize()}",
        density=True,
    )
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Density")
    ax2.set_title("Distribution of Predicted Probabilities")
    ax2.legend()

    plt.tight_layout()

    if save_path is not None:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Calibration curve saved to %s", save_path)

    plt.close(fig)
