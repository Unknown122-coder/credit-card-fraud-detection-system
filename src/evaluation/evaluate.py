"""Full evaluation pipeline with test-set guard.

Orchestrates model evaluation end-to-end: validation metrics, cost-based
threshold optimisation, probability calibration, **single-use** test-set
evaluation with bootstrap confidence intervals, diagnostic plots, and
MLflow experiment logging.

.. important::

   The test set is touched **exactly once** — all threshold selection and
   calibration happen on the validation split.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import PROJECT_ROOT, Settings, get_settings
from src.evaluation.calibration import calibrate_model, plot_calibration_curve
from src.evaluation.metrics import (
    bootstrap_confidence_intervals,
    compute_all_metrics,
    compute_threshold_metrics_curve,
    find_optimal_threshold,
)
from src.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


# ── Test-set guard ──────────────────────────────────────────────────────


class TestSetGuard:
    """Prevents accidental re-use of the held-out test set.

    Uses a sentinel flag file on disk so the guard persists across
    process restarts.

    Parameters
    ----------
    flag_path:
        Path to the sentinel file.  Defaults to
        ``models/registry/.test_evaluated`` under the project root.
    """

    def __init__(self, flag_path: Optional[Path] = None) -> None:
        if flag_path is None:
            flag_path = PROJECT_ROOT / "models" / "registry" / ".test_evaluated"
        self._flag_path = flag_path

    def can_evaluate(self) -> bool:
        """Return ``True`` only if the test set has NOT been evaluated yet."""
        return not self._flag_path.exists()

    def mark_evaluated(self) -> None:
        """Create the sentinel file, locking out future test evaluations."""
        self._flag_path.parent.mkdir(parents=True, exist_ok=True)
        self._flag_path.write_text("evaluated", encoding="utf-8")
        logger.info("Test-set guard flag written to %s", self._flag_path)

    def reset(self) -> None:
        """Remove the sentinel file.  **Development only.**"""
        if self._flag_path.exists():
            self._flag_path.unlink()
        logger.warning(
            "Test-set guard RESET — flag removed at %s. "
            "This should only happen during development!",
            self._flag_path,
        )


# ── Plotting helpers ────────────────────────────────────────────────────


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str | Path,
) -> None:
    """Plot and save a precision–recall curve.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_prob:
        Predicted probabilities for the positive class.
    save_path:
        Destination file path (e.g. ``docs/figures/pr_curve.png``).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, lw=2, label=f"PR curve (AP = {ap:.4f})")
    ax.fill_between(recall, precision, alpha=0.2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("PR curve saved to %s", save_path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str | Path,
) -> None:
    """Plot and save a receiver operating characteristic (ROC) curve.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_prob:
        Predicted probabilities for the positive class.
    save_path:
        Destination file path.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.2)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curve saved to %s", save_path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path,
) -> None:
    """Plot and save a confusion matrix heatmap.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_pred:
        Predicted binary labels.
    save_path:
        Destination file path.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Legitimate", "Fraud"]
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path)


# ── Full evaluation pipeline ────────────────────────────────────────────


def run_full_evaluation(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Settings,
    amounts_val: Optional[np.ndarray] = None,
    amounts_test: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Run the complete evaluation pipeline.

    Pipeline steps:

    1. Compute validation metrics at the default threshold (0.5).
    2. Find the cost-optimal threshold on the validation set.
    3. Calibrate the model on the validation set.
    4. Re-compute validation metrics with calibrated probabilities.
    5. **If** the :class:`TestSetGuard` allows: evaluate on the test set
       with bootstrap confidence intervals (single-use).
    6. Generate and save diagnostic plots.
    7. Log all results to MLflow.

    Parameters
    ----------
    model:
        A fitted scikit-learn-compatible classifier.
    X_train:
        Training feature matrix (used only for MLflow dataset logging).
    y_train:
        Training labels.
    X_val:
        Validation feature matrix.
    y_val:
        Validation labels.
    X_test:
        Test feature matrix.
    y_test:
        Test labels.
    config:
        Project :class:`~src.config.Settings` object.
    amounts_val:
        Per-transaction amounts for validation samples.
    amounts_test:
        Per-transaction amounts for test samples.

    Returns
    -------
    dict[str, Any]
        Comprehensive results dictionary containing validation metrics,
        optimal threshold, calibration data, and (if evaluated) test
        metrics with confidence intervals.
    """
    eval_cfg = config.evaluation
    figures_dir = config.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}

    # ── Step 1: Validation metrics at default threshold (Uncalibrated) ──
    logger.info("Step 1: Computing uncalibrated validation metrics at threshold=0.5")
    y_val_prob_raw = model.predict_proba(X_val)[:, 1]

    val_metrics_default = compute_all_metrics(
        y_val,
        y_val_prob_raw,
        threshold=0.5,
        amounts=amounts_val,
        fp_cost=eval_cfg.fp_cost,
        fn_base_cost=eval_cfg.fn_base_cost,
    )
    results["val_metrics_default"] = val_metrics_default
    logger.info("Validation metrics uncalibrated (t=0.5): %s", val_metrics_default)

    # ── Step 2: Split validation partition into Calibration & Threshold splits ──
    # Prevents in-sample calibration overfitting and ensures threshold is selected on calibrated holdout
    from sklearn.model_selection import train_test_split
    val_indices = np.arange(len(y_val))
    if np.sum(y_val == 1) >= 2:
        cal_idx, thresh_idx = train_test_split(
            val_indices, test_size=0.5, stratify=y_val, random_state=config.project.seed
        )
    else:
        cal_idx, thresh_idx = train_test_split(
            val_indices, test_size=0.5, random_state=config.project.seed
        )

    X_val_cal, y_val_cal = X_val[cal_idx], y_val[cal_idx]
    X_val_thresh, y_val_thresh = X_val[thresh_idx], y_val[thresh_idx]
    amounts_val_thresh = amounts_val[thresh_idx] if amounts_val is not None else None

    # ── Step 3: Calibrate model on calibration sub-split ──────────────
    logger.info("Step 3: Calibrating model with method='%s' on calibration sub-split", eval_cfg.calibration_method)
    calibrated_model = calibrate_model(
        model, X_val_cal, y_val_cal, method=eval_cfg.calibration_method
    )
    results["calibrated_model"] = calibrated_model

    # ── Step 4: Find optimal threshold on calibrated probabilities ────
    logger.info("Step 4: Finding cost-minimizing threshold on calibrated holdout probabilities")
    y_thresh_prob_cal = calibrated_model.predict_proba(X_val_thresh)[:, 1]

    optimal_threshold, min_cost = find_optimal_threshold(
        y_val_thresh,
        y_thresh_prob_cal,
        amounts=amounts_val_thresh,
        fp_cost=eval_cfg.fp_cost,
        fn_base_cost=eval_cfg.fn_base_cost,
        n_steps=eval_cfg.threshold_steps,
    )
    results["optimal_threshold"] = optimal_threshold
    results["optimal_threshold_cost"] = min_cost

    val_metrics_optimal = compute_all_metrics(
        y_val_thresh,
        y_thresh_prob_cal,
        threshold=optimal_threshold,
        amounts=amounts_val_thresh,
        fp_cost=eval_cfg.fp_cost,
        fn_base_cost=eval_cfg.fn_base_cost,
    )
    results["val_metrics_optimal"] = val_metrics_optimal
    logger.info(
        "Validation metrics on calibrated sub-split (t=%.4f): %s", optimal_threshold, val_metrics_optimal
    )

    # Full validation set calibrated metrics and diagnostic calibration curve
    y_val_prob_cal = calibrated_model.predict_proba(X_val)[:, 1]
    val_metrics_calibrated = compute_all_metrics(
        y_val,
        y_val_prob_cal,
        threshold=optimal_threshold,
        amounts=amounts_val,
        fp_cost=eval_cfg.fp_cost,
        fn_base_cost=eval_cfg.fn_base_cost,
    )
    results["val_metrics_calibrated"] = val_metrics_calibrated
    logger.info("Full calibrated validation metrics: %s", val_metrics_calibrated)

    # Save calibration curve
    plot_calibration_curve(
        y_val,
        y_val_prob_raw,
        y_val_prob_cal,
        method=eval_cfg.calibration_method,
        save_path=str(figures_dir / "calibration_curve.png"),
    )

    # ── Step 5: Test-set evaluation (single use) ─────────────────────
    guard = TestSetGuard()
    if guard.can_evaluate():
        logger.info("Step 5: Evaluating on held-out test set (ONE-TIME)")
        y_test_prob = calibrated_model.predict_proba(X_test)[:, 1]

        test_metrics = compute_all_metrics(
            y_test,
            y_test_prob,
            threshold=optimal_threshold,
            amounts=amounts_test,
            fp_cost=eval_cfg.fp_cost,
            fn_base_cost=eval_cfg.fn_base_cost,
        )
        results["test_metrics"] = test_metrics
        logger.info("Test metrics: %s", test_metrics)

        # Bootstrap confidence intervals
        test_ci = bootstrap_confidence_intervals(
            y_test,
            y_test_prob,
            threshold=optimal_threshold,
            amounts=amounts_test,
            n_iterations=eval_cfg.bootstrap_iterations,
            confidence=eval_cfg.bootstrap_confidence,
            seed=config.project.seed,
            fp_cost=eval_cfg.fp_cost,
            fn_base_cost=eval_cfg.fn_base_cost,
        )
        results["test_confidence_intervals"] = test_ci

        guard.mark_evaluated()
    else:
        logger.warning(
            "Step 5: SKIPPED — test set has already been evaluated. "
            "Using previously recorded results."
        )
        results["test_metrics"] = None
        results["test_confidence_intervals"] = None

    # ── Step 6: Generate and save diagnostic plots ───────────────────
    logger.info("Step 6: Generating diagnostic plots")

    # Use calibrated probabilities for plots
    prob_for_plots = y_val_prob_cal
    pred_for_cm = (prob_for_plots >= optimal_threshold).astype(int)

    plot_pr_curve(y_val, prob_for_plots, figures_dir / "pr_curve.png")
    plot_roc_curve(y_val, prob_for_plots, figures_dir / "roc_curve.png")
    plot_confusion_matrix(y_val, pred_for_cm, figures_dir / "confusion_matrix.png")

    # Threshold metrics curve for UI slider
    threshold_curve = compute_threshold_metrics_curve(
        y_val,
        prob_for_plots,
        amounts=amounts_val,
        fp_cost=eval_cfg.fp_cost,
        fn_base_cost=eval_cfg.fn_base_cost,
    )
    threshold_curve_path = figures_dir / "threshold_metrics_curve.csv"
    threshold_curve.to_csv(threshold_curve_path, index=False)
    results["threshold_curve"] = threshold_curve
    logger.info("Threshold curve saved to %s", threshold_curve_path)

    # ── Step 7: Log to MLflow ────────────────────────────────────────
    logger.info("Step 7: Logging results to MLflow")
    _log_to_mlflow(results, config)

    # ── Save threshold config for serving ────────────────────────────
    threshold_config = {
        "optimal_threshold": optimal_threshold,
        "fp_cost": eval_cfg.fp_cost,
        "fn_base_cost": eval_cfg.fn_base_cost,
        "calibration_method": eval_cfg.calibration_method,
    }
    threshold_path = config.model_dir / config.artifacts.threshold_filename
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump(threshold_config, f, indent=2)
    logger.info("Threshold config saved to %s", threshold_path)

    # Save calibrated model
    calibrator_path = config.model_dir / config.artifacts.calibrator_filename
    joblib.dump(calibrated_model, calibrator_path)
    logger.info("Calibrated model saved to %s", calibrator_path)

    logger.info("Full evaluation pipeline complete")
    return results


# ── MLflow logging helper ───────────────────────────────────────────────


def _log_to_mlflow(results: dict[str, Any], config: Settings) -> None:
    """Log evaluation results to MLflow.

    Parameters
    ----------
    results:
        Results dictionary produced by :func:`run_full_evaluation`.
    config:
        Project settings containing MLflow configuration.
    """
    try:
        import mlflow  # noqa: WPS433 — optional dependency
    except ImportError:
        logger.warning("MLflow not installed — skipping experiment logging")
        return

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    with mlflow.start_run(run_name="evaluation"):
        # Log optimal threshold
        mlflow.log_param("optimal_threshold", results["optimal_threshold"])

        # Log validation metrics (default threshold)
        for key, value in results["val_metrics_default"].items():
            mlflow.log_metric(f"val_default_{key}", value)

        # Log validation metrics (optimal threshold)
        for key, value in results["val_metrics_optimal"].items():
            mlflow.log_metric(f"val_optimal_{key}", value)

        # Log calibrated validation metrics
        for key, value in results["val_metrics_calibrated"].items():
            mlflow.log_metric(f"val_calibrated_{key}", value)

        # Log test metrics if available
        if results.get("test_metrics") is not None:
            for key, value in results["test_metrics"].items():
                mlflow.log_metric(f"test_{key}", value)

        # Log confidence intervals if available
        if results.get("test_confidence_intervals") is not None:
            for metric_name, (mean, lower, upper) in results[
                "test_confidence_intervals"
            ].items():
                mlflow.log_metric(f"test_ci_{metric_name}_mean", mean)
                mlflow.log_metric(f"test_ci_{metric_name}_lower", lower)
                mlflow.log_metric(f"test_ci_{metric_name}_upper", upper)

        # Log figures as artifacts
        figures_dir = config.figures_dir
        if figures_dir.exists():
            mlflow.log_artifacts(str(figures_dir), artifact_path="figures")

    logger.info("MLflow logging complete")


# ── CLI entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.utils.logger import setup_logging
    from src.data.split import load_splits

    settings = get_settings()
    setup_logging(settings.project.log_level)
    set_global_seed(settings.project.seed)

    # ── Load data splits from Parquet ─────────────────────────────────
    processed_dir = settings.processed_dir
    logger.info("Loading data splits from %s", processed_dir)

    train_df, val_df, test_df = load_splits(settings)
    target_column = settings.data.target_column

    if target_column not in train_df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in training data."
        )

    # Use the same feature selection strategy as the tuning script.
    feature_columns = [
        column for column in train_df.columns
        if column != target_column
    ]

    # Verify that validation and test contain the same features.
    for split_name, split_df in (
        ("validation", val_df),
        ("test", test_df),
    ):
        missing = [
            column for column in feature_columns
            if column not in split_df.columns
        ]
        if missing:
            raise ValueError(
                f"{split_name} data is missing features: {missing}"
            )

    # Build feature matrices and target arrays.
    X_train = train_df[feature_columns].to_numpy()
    y_train = train_df[target_column].to_numpy()

    X_val = val_df[feature_columns].to_numpy()
    y_val = val_df[target_column].to_numpy()

    X_test = test_df[feature_columns].to_numpy()
    y_test = test_df[target_column].to_numpy()

    # Preserve transaction amounts for cost-sensitive evaluation.
    amounts_val: Optional[np.ndarray] = (
        val_df["Amount"].to_numpy()
        if "Amount" in val_df.columns
        else None
    )
    amounts_test: Optional[np.ndarray] = (
        test_df["Amount"].to_numpy()
        if "Amount" in test_df.columns
        else None
    )

    logger.info(
        "Loaded splits: train=%s, validation=%s, test=%s",
        X_train.shape,
        X_val.shape,
        X_test.shape,
    )
    logger.info(
        "Feature count: %d; training fraud ratio: %.6f",
        len(feature_columns),
        float(np.mean(y_train)),
    )

    # ── Load tuned XGBoost model ──────────────────────────────────────
    model_path = (
        settings.model_dir / "tuned" / "xgboost_pipeline.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Tuned XGBoost model not found: {model_path}"
        )

    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    # Check feature compatibility before running evaluation.
    if hasattr(model, "n_features_in_"):
        if model.n_features_in_ != X_train.shape[1]:
            raise ValueError(
                f"Model expects {model.n_features_in_} features, "
                f"but the data contains {X_train.shape[1]}."
            )

    # ── Run evaluation ────────────────────────────────────────────────
    eval_results = run_full_evaluation(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        config=settings,
        amounts_val=amounts_val,
        amounts_test=amounts_test,
    )

    # ── Print summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(
        f"Optimal threshold: "
        f"{eval_results['optimal_threshold']:.4f}"
    )

    print("\nValidation metrics (calibrated, optimal threshold):")
    for key, value in eval_results["val_metrics_calibrated"].items():
        print(f"  {key:20s}: {value:.6f}")

    if eval_results.get("test_metrics") is not None:
        print("\nTest metrics:")
        for key, value in eval_results["test_metrics"].items():
            print(f"  {key:20s}: {value:.6f}")

        print("\nTest confidence intervals:")
        for key, (mean, lower, upper) in (
            eval_results["test_confidence_intervals"].items()
        ):
            print(
                f"  {key:20s}: {mean:.4f} "
                f"[{lower:.4f}, {upper:.4f}]"
            )
    else:
        print("\nTest set: already evaluated (skipped)")

    print("=" * 60)
