"""Model pipeline factory using imblearn to isolate resampling within training folds.

Imblearn's :class:`~imblearn.pipeline.Pipeline` ensures that any resampling
step (SMOTE, SMOTETomek, etc.) is applied **only** to the training fold during
cross-validation, preventing data leakage from synthetic samples into the
validation fold.

Each pipeline follows the pattern::

    RobustScaler → (optional Resampler) → Classifier
"""

from __future__ import annotations

import logging
from typing import Any

from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported model names and resampling strategies
# ---------------------------------------------------------------------------
_SUPPORTED_MODELS: set[str] = {
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "balanced_rf",
}

_SUPPORTED_RESAMPLERS: set[str] = {
    "none",
    "smote",
    "smote_tomek",
    "random_undersample",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_pipeline(
    model_name: str,
    resampling: str = "none",
    seed: int = 42,
    **model_params: Any,
) -> ImbPipeline:
    """Build an imblearn pipeline: scaler → (optional resampler) → model.

    The pipeline is compatible with :func:`sklearn.model_selection.cross_val_predict`
    and automatically applies resampling **only** within training folds.

    Parameters
    ----------
    model_name:
        One of ``'logistic_regression'``, ``'random_forest'``, ``'xgboost'``,
        ``'lightgbm'``, ``'catboost'``, ``'balanced_rf'``.
    resampling:
        One of ``'none'``, ``'smote'``, ``'smote_tomek'``,
        ``'random_undersample'``.  Class-weight / scale_pos_weight are
        model-level parameters, **not** resampling steps.
    seed:
        Random seed for reproducibility.
    **model_params:
        Extra keyword arguments forwarded to the model constructor.

    Returns
    -------
    ImbPipeline
        Ready-to-fit imblearn pipeline.

    Raises
    ------
    ValueError
        If *model_name* or *resampling* is not supported.
    """
    _validate_model_name(model_name)
    _validate_resampling_strategy(resampling)

    steps: list[tuple[str, Any]] = [("scaler", RobustScaler())]

    resampler = get_resampler(resampling, seed=seed)
    if resampler is not None:
        steps.append(("resampler", resampler))

    model = get_model(model_name, seed=seed, **model_params)
    steps.append(("model", model))

    pipeline = ImbPipeline(steps=steps)
    logger.info(
        "Built pipeline: model=%s, resampling=%s, extra_params=%s",
        model_name,
        resampling,
        model_params or "{}",
    )
    return pipeline


def get_model(
    model_name: str,
    seed: int = 42,
    **params: Any,
) -> BaseEstimator:
    """Instantiate a classifier by name with sensible defaults.

    Parameters
    ----------
    model_name:
        Identifier for the classifier — see :data:`_SUPPORTED_MODELS`.
    seed:
        Random state seed.
    **params:
        Keyword arguments forwarded to the model constructor.

    Returns
    -------
    BaseEstimator
        Scikit-learn-compatible estimator.

    Raises
    ------
    ValueError
        If *model_name* is not recognised.
    """
    _validate_model_name(model_name)

    factories: dict[str, type[BaseEstimator]] = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "xgboost": XGBClassifier,
        "lightgbm": LGBMClassifier,
        "catboost": CatBoostClassifier,
        "balanced_rf": BalancedRandomForestClassifier,
    }

    # ── per-model default kwargs ──────────────────────────────────────
    defaults: dict[str, dict[str, Any]] = {
        "logistic_regression": {
            "max_iter": 1000,
            "solver": "lbfgs",
            "random_state": seed,
        },
        "random_forest": {
            "n_estimators": 100,
            "n_jobs": -1,
            "random_state": seed,
        },
        "xgboost": {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "n_estimators": 300,
            "use_label_encoder": False,
            "random_state": seed,
            "n_jobs": -1,
            "verbosity": 0,
        },
        "lightgbm": {
            "objective": "binary",
            "verbose": -1,
            "n_estimators": 300,
            "n_jobs": -1,
            "random_state": seed,
        },
        "catboost": {
            "loss_function": "Logloss",
            "verbose": 0,
            "iterations": 300,
            "random_seed": seed,
        },
        "balanced_rf": {
            "n_estimators": 100,
            "n_jobs": -1,
            "random_state": seed,
            "replacement": True,
        },
    }

    kwargs = {**defaults[model_name], **params}
    model = factories[model_name](**kwargs)
    logger.debug("Instantiated %s with %s", model_name, kwargs)
    return model


def get_resampler(
    strategy: str,
    seed: int = 42,
    **params: Any,
) -> Any | None:
    """Return a resampling transformer or ``None`` for ``'none'``.

    Parameters
    ----------
    strategy:
        One of ``'none'``, ``'smote'``, ``'smote_tomek'``,
        ``'random_undersample'``.
    seed:
        Random state seed.
    **params:
        Extra keyword arguments forwarded to the resampler constructor.

    Returns
    -------
    object | None
        An imblearn sampler instance, or ``None`` when *strategy* is
        ``'none'``.

    Raises
    ------
    ValueError
        If *strategy* is not recognised.
    """
    _validate_resampling_strategy(strategy)

    if strategy == "none":
        return None

    resampler_defaults: dict[str, dict[str, Any]] = {
        "smote": {
            "k_neighbors": 5,
            "random_state": seed,
        },
        "smote_tomek": {
            "random_state": seed,
        },
        "random_undersample": {
            "random_state": seed,
        },
    }

    factories: dict[str, type] = {
        "smote": SMOTE,
        "smote_tomek": SMOTETomek,
        "random_undersample": RandomUnderSampler,
    }

    kwargs = {**resampler_defaults[strategy], **params}
    resampler = factories[strategy](**kwargs)
    logger.debug("Instantiated resampler %s with %s", strategy, kwargs)
    return resampler


def get_baseline_configs() -> list[dict[str, Any]]:
    """Return a list of experiment configurations for baseline comparison.

    Each configuration dictionary has the keys:

    - ``name``: human-readable experiment label.
    - ``model_name``: identifier passed to :func:`get_model`.
    - ``resampling``: identifier passed to :func:`get_resampler`.
    - ``model_params``: extra keyword arguments for the model constructor.

    Returns
    -------
    list[dict[str, Any]]
        Ten baseline experiment configurations.
    """
    configs: list[dict[str, Any]] = [
        # ── Logistic Regression baselines ─────────────────────────────
        {
            "name": "LogReg",
            "model_name": "logistic_regression",
            "resampling": "none",
            "model_params": {},
        },
        {
            "name": "LogReg_balanced",
            "model_name": "logistic_regression",
            "resampling": "none",
            "model_params": {"class_weight": "balanced"},
        },
        # ── Random Forest baselines ───────────────────────────────────
        {
            "name": "RF",
            "model_name": "random_forest",
            "resampling": "none",
            "model_params": {},
        },
        {
            "name": "RF_balanced",
            "model_name": "random_forest",
            "resampling": "none",
            "model_params": {"class_weight": "balanced"},
        },
        # ── XGBoost baselines ─────────────────────────────────────────
        {
            "name": "XGB",
            "model_name": "xgboost",
            "resampling": "none",
            "model_params": {},
        },
        {
            "name": "XGB_weighted",
            "model_name": "xgboost",
            "resampling": "none",
            "model_params": {"scale_pos_weight": 100.0},
        },
        {
            "name": "XGB_SMOTE",
            "model_name": "xgboost",
            "resampling": "smote",
            "model_params": {},
        },
        {
            "name": "XGB_SMOTETomek",
            "model_name": "xgboost",
            "resampling": "smote_tomek",
            "model_params": {},
        },
        {
            "name": "XGB_RUS",
            "model_name": "xgboost",
            "resampling": "random_undersample",
            "model_params": {},
        },
        # ── Balanced Random Forest ────────────────────────────────────
        {
            "name": "BalancedRF",
            "model_name": "balanced_rf",
            "resampling": "none",
            "model_params": {},
        },
    ]
    return configs


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------


def _validate_model_name(model_name: str) -> None:
    """Raise ``ValueError`` if *model_name* is not supported.

    Parameters
    ----------
    model_name:
        Model identifier to validate.
    """
    if model_name not in _SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Supported: {sorted(_SUPPORTED_MODELS)}"
        )


def _validate_resampling_strategy(strategy: str) -> None:
    """Raise ``ValueError`` if *strategy* is not supported.

    Parameters
    ----------
    strategy:
        Resampling strategy identifier to validate.
    """
    if strategy not in _SUPPORTED_RESAMPLERS:
        raise ValueError(
            f"Unknown resampling strategy '{strategy}'. "
            f"Supported: {sorted(_SUPPORTED_RESAMPLERS)}"
        )
