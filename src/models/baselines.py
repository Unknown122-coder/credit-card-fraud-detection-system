"""Baseline model evaluation and class imbalance comparison.

Implements and evaluates:
- Logistic Regression (vanilla, class_weight='balanced')
- Random Forest (vanilla, class_weight='balanced')
- XGBoost (vanilla, scale_pos_weight, SMOTE, SMOTETomek, RandomUnderSampler)
- Balanced Random Forest

Cross-validation is stratified, with all resampling and scaling isolated
strictly inside the training folds using imblearn.pipeline.Pipeline.
Results are logged to MLflow and saved as CSV and Markdown comparison tables.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import Settings, get_settings
from src.data.split import load_splits
from src.evaluation.metrics import compute_all_metrics
from src.models.pipelines import build_pipeline, get_baseline_configs
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


def evaluate_pipeline_cv(
    pipeline_builder: Any,
    X: np.ndarray,
    y: np.ndarray,
    amounts: Optional[np.ndarray] = None,
    n_splits: int = 5,
    seed: int = 42,
    fp_cost: float = 10.0,
    fn_base_cost: float = 25.0,
) -> tuple[dict[str, float], float]:
    """Perform Stratified K-Fold CV evaluating an imblearn pipeline.

    Accumulates out-of-fold probability predictions to ensure an unbiased
    estimate of out-of-fold performance.

    Parameters
    ----------
    pipeline_builder:
        Callable returning a fresh imblearn Pipeline.
    X:
        Feature matrix.
    y:
        Binary target vector.
    amounts:
        Monetary amount array for expected cost calculation.
    n_splits:
        Number of stratified folds.
    seed:
        Random state for CV splitter.
    fp_cost:
        False positive unit cost.
    fn_base_cost:
        False negative base overhead.

    Returns
    -------
    tuple[dict[str, float], float]
        Dictionary of mean CV metrics and total training duration in seconds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_probs = np.zeros(len(y), dtype=float)

    start_time = time.perf_counter()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va = X[val_idx]

        # Construct a fresh pipeline per fold to guarantee no leakage
        pipeline = pipeline_builder()
        pipeline.fit(X_tr, y_tr)

        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X_va)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            raw_scores = pipeline.decision_function(X_va)
            # Sigmoidal mapping for decision function
            probs = 1.0 / (1.0 + np.exp(-raw_scores))
        else:
            probs = pipeline.predict(X_va).astype(float)

        oof_probs[val_idx] = probs

    total_time = time.perf_counter() - start_time

    # Compute metrics on out-of-fold predictions
    metrics = compute_all_metrics(
        y_true=y,
        y_prob=oof_probs,
        threshold=0.5,
        amounts=amounts,
        fp_cost=fp_cost,
        fn_base_cost=fn_base_cost,
    )
    return metrics, total_time


def run_baseline_comparison(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    config: Optional[Settings] = None,
    amounts_train: Optional[np.ndarray] = None,
    output_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Run cross-validated baseline experiments and save comparison table.

    Parameters
    ----------
    X_train:
        Training features.
    y_train:
        Training targets.
    config:
        Project configuration.
    amounts_train:
        Amounts corresponding to training observations.
    output_dir:
        Directory to write comparison tables.

    Returns
    -------
    pd.DataFrame
        Master model comparison table.
    """
    if config is None:
        config = get_settings()

    set_global_seed(config.project.seed)

    # Convert to NumPy arrays for faster indexing in CV
    X_arr = X_train.values if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)
    y_arr = y_train.values if isinstance(y_train, pd.Series) else np.asarray(y_train)

    if (
        amounts_train is None
        and isinstance(X_train, pd.DataFrame)
        and "Amount" in X_train.columns
    ):
        amounts_arr = X_train["Amount"].to_numpy()
    elif amounts_train is not None:
        amounts_arr = np.asarray(amounts_train)
    else:
        amounts_arr = None

    # Setup MLflow
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    configs = get_baseline_configs()
    records: list[dict[str, Any]] = []

    logger.info(
        "Starting baseline comparison across %d configurations...", len(configs)
    )

    for exp in configs:
        exp_name: str = exp["name"]
        model_name: str = exp["model_name"]
        resampling: str = exp["resampling"]
        model_params: dict[str, Any] = exp["model_params"]

        logger.info(
            "Evaluating configuration: %s (model=%s, resampling=%s)",
            exp_name,
            model_name,
            resampling,
        )

        def make_pipe() -> Any:
            return build_pipeline(
                model_name=model_name,
                resampling=resampling,
                seed=config.project.seed,
                **model_params,
            )

        with mlflow.start_run(run_name=f"baseline_{exp_name}"):
            mlflow.log_params(
                {
                    "model_name": model_name,
                    "resampling": resampling,
                    **model_params,
                }
            )

            metrics, train_time = evaluate_pipeline_cv(
                pipeline_builder=make_pipe,
                X=X_arr,
                y=y_arr,
                amounts=amounts_arr,
                n_splits=config.training.cv_folds,
                seed=config.project.seed,
                fp_cost=config.evaluation.fp_cost,
                fn_base_cost=config.evaluation.fn_base_cost,
            )

            metrics["train_time_sec"] = train_time
            mlflow.log_metrics(metrics)

            record = {
                "Experiment": exp_name,
                "Model": model_name,
                "Resampling": resampling,
                "PR-AUC": round(metrics["pr_auc"], 4),
                "ROC-AUC": round(metrics["roc_auc"], 4),
                "Precision": round(metrics["precision"], 4),
                "Recall": round(metrics["recall"], 4),
                "F1": round(metrics["f1"], 4),
                "F2": round(metrics["f2"], 4),
                "Expected Cost ($)": round(metrics["expected_cost"], 2),
                "Train Time (s)": round(train_time, 2),
            }
            records.append(record)

    results_df = pd.DataFrame(records)
    results_df = results_df.sort_values(by="PR-AUC", ascending=False).reset_index(
        drop=True
    )

    # Save results
    out_path = Path(output_dir) if output_dir else config.model_dir
    out_path.mkdir(parents=True, exist_ok=True)
    save_comparison_tables(results_df, out_path)

    logger.info(
        "Baseline comparison completed successfully:\n%s", results_df.to_string()
    )
    return results_df


def save_comparison_tables(df: pd.DataFrame, output_dir: Path) -> None:
    """Save results as CSV and clean Markdown tables."""
    csv_file = output_dir / "baseline_comparison.csv"
    md_file = output_dir / "baseline_comparison.md"

    df.to_csv(csv_file, index=False)

    md_content = "# Model Imbalance & Baseline Comparison Table\n\n"
    md_content += df.to_markdown(index=False)
    md_content += "\n\n*Note: Evaluated with 5-fold Stratified CV out-of-fold predictions. Threshold = 0.50.*\n"

    md_file.write_text(md_content, encoding="utf-8")
    logger.info("Saved baseline comparison tables to %s and %s", csv_file, md_file)


if __name__ == "__main__":
    settings = get_settings()
    train_df, val_df, _ = load_splits(settings)
    feature_cols = [c for c in train_df.columns if c != settings.data.target_column]
    run_baseline_comparison(
        X_train=train_df[feature_cols],
        y_train=train_df[settings.data.target_column],
        config=settings,
    )
