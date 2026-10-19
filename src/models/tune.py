"""Optuna-based hyperparameter tuning for gradient boosted models.

Provides factory functions that create Optuna objective callables for
XGBoost, LightGBM, and CatBoost.  Each objective maximises mean PR-AUC
(average precision) across stratified CV folds and supports Optuna's
pruning interface for early stopping of unpromising trials.

Typical usage
-------------
>>> from src.config import get_settings
>>> settings = get_settings()
>>> studies = run_tuning(X_train, y_train, settings)
>>> best_models = build_best_models(studies, X_train, y_train, seed=42)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
from catboost import CatBoostClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from optuna.visualization import (
    plot_contour,
    plot_optimization_history,
    plot_param_importances,
)
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from src.config import Settings, get_settings
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed

logger: logging.Logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
_ArrayLike = NDArray[np.floating[Any]]
_ObjectiveCallable = Callable[[optuna.Trial], float]


# ═══════════════════════════════════════════════════════════════════════════
# 1. XGBoost objective
# ═══════════════════════════════════════════════════════════════════════════


def create_xgboost_objective(
    X_train: _ArrayLike,
    y_train: _ArrayLike,
    cv: int,
    seed: int,
) -> _ObjectiveCallable:
    """Return an Optuna objective that maximises XGBoost PR-AUC.

    Parameters
    ----------
    X_train:
        Training feature matrix.
    y_train:
        Binary target vector.
    cv:
        Number of cross-validation folds.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Callable[[optuna.Trial], float]
        Objective function suitable for ``study.optimize``.
    """
    # Pre-compute class imbalance ratio for scale_pos_weight
    n_negative: int = int(np.sum(y_train == 0))
    n_positive: int = int(np.sum(y_train == 1))
    scale_pos_weight: float = n_negative / max(n_positive, 1)

    def _objective(trial: optuna.Trial) -> float:
        """Evaluate a single XGBoost hyper-parameter configuration."""
        params: dict[str, Any] = {
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": scale_pos_weight,
            "use_label_encoder": False,
            "eval_metric": "aucpr",
            "random_state": seed,
            "verbosity": 0,
            "n_jobs": 1,
        }

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        fold_scores: list[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = XGBClassifier(**params)
            model.fit(X_tr, y_tr, verbose=False)

            y_prob = model.predict_proba(X_val)[:, 1]
            score = average_precision_score(y_val, y_prob)
            fold_scores.append(score)

            # Report intermediate value for pruning
            trial.report(score, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    return _objective


# ═══════════════════════════════════════════════════════════════════════════
# 2. LightGBM objective
# ═══════════════════════════════════════════════════════════════════════════


def create_lightgbm_objective(
    X_train: _ArrayLike,
    y_train: _ArrayLike,
    cv: int,
    seed: int,
) -> _ObjectiveCallable:
    """Return an Optuna objective that maximises LightGBM PR-AUC.

    Parameters
    ----------
    X_train:
        Training feature matrix.
    y_train:
        Binary target vector.
    cv:
        Number of cross-validation folds.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Callable[[optuna.Trial], float]
        Objective function suitable for ``study.optimize``.
    """

    def _objective(trial: optuna.Trial) -> float:
        """Evaluate a single LightGBM hyper-parameter configuration."""
        params: dict[str, Any] = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 500),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "is_unbalance": True,
            "random_state": seed,
            "verbosity": -1,
            "n_jobs": 1,
        }

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        fold_scores: list[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = LGBMClassifier(**params)
            model.fit(X_tr, y_tr)

            y_prob = model.predict_proba(X_val)[:, 1]
            score = average_precision_score(y_val, y_prob)
            fold_scores.append(score)

            trial.report(score, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    return _objective


# ═══════════════════════════════════════════════════════════════════════════
# 3. CatBoost objective
# ═══════════════════════════════════════════════════════════════════════════


def create_catboost_objective(
    X_train: _ArrayLike,
    y_train: _ArrayLike,
    cv: int,
    seed: int,
) -> _ObjectiveCallable:
    """Return an Optuna objective that maximises CatBoost PR-AUC.

    Parameters
    ----------
    X_train:
        Training feature matrix.
    y_train:
        Binary target vector.
    cv:
        Number of cross-validation folds.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Callable[[optuna.Trial], float]
        Objective function suitable for ``study.optimize``.
    """

    def _objective(trial: optuna.Trial) -> float:
        """Evaluate a single CatBoost hyper-parameter configuration."""
        params: dict[str, Any] = {
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "iterations": trial.suggest_int("iterations", 100, 1000, step=100),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature", 0.0, 10.0
            ),
            "auto_class_weights": "Balanced",
            "random_seed": seed,
            "verbose": 0,
            "thread_count": 1,
        }

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        fold_scores: list[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = CatBoostClassifier(**params)
            model.fit(X_tr, y_tr)

            y_prob = model.predict_proba(X_val)[:, 1]
            score = average_precision_score(y_val, y_prob)
            fold_scores.append(score)

            trial.report(score, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    return _objective


# ═══════════════════════════════════════════════════════════════════════════
# 4. run_tuning
# ═══════════════════════════════════════════════════════════════════════════


def run_tuning(
    X_train: _ArrayLike,
    y_train: _ArrayLike,
    settings: Settings,
) -> dict[str, optuna.Study]:
    """Create and optimise Optuna studies for each model family.

    Each study is run as a nested MLflow run so that trial-level metrics
    are grouped under the parent experiment.

    Parameters
    ----------
    X_train:
        Training feature matrix.
    y_train:
        Binary target vector.
    settings:
        Project configuration object.

    Returns
    -------
    dict[str, optuna.Study]
        Studies keyed by ``"xgboost"``, ``"lightgbm"``, ``"catboost"``.
    """
    seed: int = settings.project.seed
    cv: int = settings.training.cv_folds
    tuning_cfg = settings.tuning

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=5,
    )

    # Map model name → (objective factory, per-model trial count)
    model_registry: dict[str, tuple[_ObjectiveCallable, int]] = {
        "xgboost": (
            create_xgboost_objective(X_train, y_train, cv, seed),
            tuning_cfg.xgboost_trials,
        ),
        "lightgbm": (
            create_lightgbm_objective(X_train, y_train, cv, seed),
            tuning_cfg.lightgbm_trials,
        ),
        "catboost": (
            create_catboost_objective(X_train, y_train, cv, seed),
            tuning_cfg.catboost_trials,
        ),
    }

    # Configure MLflow
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)

    studies: dict[str, optuna.Study] = {}

    with mlflow.start_run(run_name="hyperparameter_tuning") as parent_run:
        logger.info(
            "Started parent MLflow run %s for hyperparameter tuning",
            parent_run.info.run_id,
        )

        for model_name, (objective_fn, n_trials) in model_registry.items():
            logger.info("Starting tuning for %s — %d trials", model_name, n_trials)

            study = optuna.create_study(
                study_name=f"{model_name}_tuning",
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
            )

            with mlflow.start_run(
                run_name=f"tune_{model_name}", nested=True
            ) as child_run:
                logger.info(
                    "Nested MLflow run %s for %s",
                    child_run.info.run_id,
                    model_name,
                )

                # Suppress Optuna's own logging to avoid clutter
                optuna.logging.set_verbosity(optuna.logging.WARNING)

                study.optimize(
                    objective_fn,
                    n_trials=n_trials,
                    timeout=tuning_cfg.timeout_seconds,
                    show_progress_bar=True,
                )

                # Log best results to MLflow
                best_trial = study.best_trial
                mlflow.log_params(
                    {f"best_{k}": v for k, v in best_trial.params.items()}
                )
                mlflow.log_metric("best_pr_auc", best_trial.value)
                mlflow.log_metric("n_completed_trials", len(study.trials))

                logger.info(
                    "%s best PR-AUC=%.5f  params=%s",
                    model_name,
                    best_trial.value,
                    best_trial.params,
                )

            studies[model_name] = study

        # Log a summary metric for the overall best
        overall_best_name = max(studies, key=lambda k: studies[k].best_value)
        mlflow.log_param("overall_best_model", overall_best_name)
        mlflow.log_metric("overall_best_pr_auc", studies[overall_best_name].best_value)
        logger.info(
            "Overall best model: %s with PR-AUC=%.5f",
            overall_best_name,
            studies[overall_best_name].best_value,
        )

    return studies


# ═══════════════════════════════════════════════════════════════════════════
# 5. save_optuna_visualizations
# ═══════════════════════════════════════════════════════════════════════════


def save_optuna_visualizations(
    studies: dict[str, optuna.Study],
    output_dir: str,
) -> None:
    """Persist interactive Optuna visualisation plots as HTML files.

    For each study the following plots are saved:

    * Optimisation history
    * Parameter importances
    * Contour plot for the two most important parameters

    Parameters
    ----------
    studies:
        Studies keyed by model name (e.g. ``"xgboost"``).
    output_dir:
        Directory where HTML files will be written.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for model_name, study in studies.items():
        logger.info("Saving visualisations for %s to %s", model_name, out_path)

        # 1. Optimisation history -------------------------------------------
        fig_history = plot_optimization_history(study)
        history_path = out_path / f"{model_name}_optimization_history.html"
        fig_history.write_html(str(history_path))
        logger.info("  ↳ optimization history → %s", history_path)

        # 2. Parameter importances ------------------------------------------
        try:
            fig_importance = plot_param_importances(study)
            importance_path = out_path / f"{model_name}_param_importances.html"
            fig_importance.write_html(str(importance_path))
            logger.info("  ↳ param importances   → %s", importance_path)
        except (ValueError, RuntimeError) as exc:
            logger.warning("Skipping param importances for %s: %s", model_name, exc)

        # 3. Contour plot (top-2 params) ------------------------------------
        try:
            importances = optuna.importance.get_param_importances(study)
            top_params = list(importances.keys())[:2]
            if len(top_params) == 2:
                fig_contour = plot_contour(study, params=top_params)
                contour_path = out_path / f"{model_name}_contour_top2.html"
                fig_contour.write_html(str(contour_path))
                logger.info("  ↳ contour plot        → %s", contour_path)
            else:
                logger.warning("Not enough params for contour plot for %s", model_name)
        except (ValueError, RuntimeError) as exc:
            logger.warning("Skipping contour plot for %s: %s", model_name, exc)


# ═══════════════════════════════════════════════════════════════════════════
# 6. build_best_models
# ═══════════════════════════════════════════════════════════════════════════


_MODEL_BUILDERS: dict[str, type[BaseEstimator]] = {
    "xgboost": XGBClassifier,
    "lightgbm": LGBMClassifier,
    "catboost": CatBoostClassifier,
}


def _make_xgboost_params(
    best_params: dict[str, Any],
    y_train: _ArrayLike,
    seed: int,
) -> dict[str, Any]:
    """Build full XGBoost constructor kwargs from Optuna best params.

    Parameters
    ----------
    best_params:
        Best trial parameters from Optuna.
    y_train:
        Binary target vector (used for ``scale_pos_weight``).
    seed:
        Random seed.

    Returns
    -------
    dict[str, Any]
        Merged parameter dict ready for ``XGBClassifier(**params)``.
    """
    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    return {
        **best_params,
        "scale_pos_weight": n_neg / max(n_pos, 1),
        "use_label_encoder": False,
        "eval_metric": "aucpr",
        "random_state": seed,
        "verbosity": 0,
        "n_jobs": 1,
    }


def _make_lightgbm_params(
    best_params: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Build full LightGBM constructor kwargs from Optuna best params.

    Parameters
    ----------
    best_params:
        Best trial parameters from Optuna.
    seed:
        Random seed.

    Returns
    -------
    dict[str, Any]
        Merged parameter dict ready for ``LGBMClassifier(**params)``.
    """
    return {
        **best_params,
        "is_unbalance": True,
        "random_state": seed,
        "verbosity": -1,
        "n_jobs": 1,
    }


def _make_catboost_params(
    best_params: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Build full CatBoost constructor kwargs from Optuna best params.

    Parameters
    ----------
    best_params:
        Best trial parameters from Optuna.
    seed:
        Random seed.

    Returns
    -------
    dict[str, Any]
        Merged parameter dict ready for ``CatBoostClassifier(**params)``.
    """
    return {
        **best_params,
        "auto_class_weights": "Balanced",
        "random_seed": seed,
        "verbose": 0,
        "thread_count": 1,
    }


def build_best_models(
    studies: dict[str, optuna.Study],
    X_train: _ArrayLike,
    y_train: _ArrayLike,
    seed: int,
) -> dict[str, BaseEstimator]:
    """Construct and fit imblearn pipelines using best params from each study.

    Each pipeline consists of:

    1. :class:`~sklearn.preprocessing.RobustScaler`
    2. The best-found classifier for that model family

    Parameters
    ----------
    studies:
        Completed Optuna studies keyed by model name.
    X_train:
        Full training feature matrix.
    y_train:
        Full training binary target vector.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict[str, BaseEstimator]
        Fitted imblearn pipelines keyed by model name.
    """
    fitted_models: dict[str, BaseEstimator] = {}

    for model_name, study in studies.items():
        best_params = study.best_trial.params.copy()
        logger.info("Building best %s model with params: %s", model_name, best_params)

        # Build model-specific full parameter dicts
        if model_name == "xgboost":
            full_params = _make_xgboost_params(best_params, y_train, seed)
            clf = XGBClassifier(**full_params)
        elif model_name == "lightgbm":
            full_params = _make_lightgbm_params(best_params, seed)
            clf = LGBMClassifier(**full_params)
        elif model_name == "catboost":
            full_params = _make_catboost_params(best_params, seed)
            clf = CatBoostClassifier(**full_params)
        else:
            logger.warning("Unknown model name %s — skipping", model_name)
            continue

        pipeline = ImbPipeline(
            steps=[
                ("scaler", RobustScaler()),
                ("classifier", clf),
            ]
        )
        pipeline.fit(X_train, y_train)
        fitted_models[model_name] = pipeline

        logger.info("Fitted %s pipeline on full training data", model_name)

    return fitted_models


# ═══════════════════════════════════════════════════════════════════════════
# __main__ entry-point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import joblib

    # ── Setup ─────────────────────────────────────────────────────────
    settings: Settings = get_settings()
    set_global_seed(settings.project.seed)

    processed_dir: Path = settings.processed_dir

    # ── Load splits ───────────────────────────────────────────────────
    logger.info("Loading training splits from %s", processed_dir)
    from src.data.split import load_splits

    try:
        train_df, _, _ = load_splits(settings)
        feat_cols = [c for c in train_df.columns if c != settings.data.target_column]
        X_train_arr = train_df[feat_cols].values
        y_train_arr = train_df[settings.data.target_column].values
    except Exception:
        X_train_arr = np.load(processed_dir / "X_train.npy")
        y_train_arr = np.load(processed_dir / "y_train.npy")

    logger.info(
        "Loaded X_train shape=%s, y_train shape=%s, fraud ratio=%.4f",
        X_train_arr.shape,
        y_train_arr.shape,
        float(y_train_arr.mean()),
    )

    # ── Run tuning ────────────────────────────────────────────────────
    all_studies: dict[str, optuna.Study] = run_tuning(
        X_train_arr, y_train_arr, settings
    )

    # ── Save visualizations ───────────────────────────────────────────
    viz_dir: str = str(settings.figures_dir / "optuna")
    save_optuna_visualizations(all_studies, viz_dir)

    # ── Build best models ─────────────────────────────────────────────
    best_models: dict[str, BaseEstimator] = build_best_models(
        all_studies,
        X_train_arr,
        y_train_arr,
        seed=settings.project.seed,
    )

    # ── Persist fitted pipelines ──────────────────────────────────────
    model_out_dir: Path = settings.model_dir / "tuned"
    model_out_dir.mkdir(parents=True, exist_ok=True)

    for name, pipeline in best_models.items():
        out_path = model_out_dir / f"{name}_pipeline.joblib"
        joblib.dump(pipeline, out_path)
        logger.info("Saved %s pipeline → %s", name, out_path)

    # ── Log fitted models to MLflow ───────────────────────────────────
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)

    with mlflow.start_run(run_name="best_tuned_models"):
        for name, pipeline in best_models.items():
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path=f"models/{name}",
                registered_model_name=f"fraud-{name}-tuned",
            )
            logger.info("Logged %s model to MLflow registry", name)

    logger.info("Hyperparameter tuning complete.")
