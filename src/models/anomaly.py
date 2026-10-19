"""Unsupervised Anomaly Detection benchmark and feature augmentation module.

Implements Isolation Forest anomaly scoring to evaluate standalone unsupervised
fraud interception vs. supervised gradient boosting, and provides an anomaly-score
feature generator for stacked hybrid architectures.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest

from src.config import Settings, get_settings
from src.evaluation.metrics import compute_all_metrics, find_optimal_threshold
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


class IsolationForestScoreTransformer(BaseEstimator, TransformerMixin):
    """Appends inverted IsolationForest anomaly score as an auxiliary feature column.

    Inverts sklearn's decision_function so that larger positive scores indicate
    higher anomaly/fraud risk, bounded in [0, 1] via min-max normalization.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        contamination: float = 0.002,
        max_samples: float = 0.8,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state
        self.model_: Optional[IsolationForest] = None
        self.min_score_: float = 0.0
        self.max_score_: float = 1.0

    def fit(
        self, X: pd.DataFrame | np.ndarray, y: Optional[Any] = None
    ) -> IsolationForestScoreTransformer:
        """Fit IsolationForest on training data."""
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=1,
        )
        self.model_.fit(X_arr)
        raw_scores = -self.model_.decision_function(X_arr)
        self.min_score_ = float(np.min(raw_scores))
        self.max_score_ = float(np.max(raw_scores))
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Append normalized anomaly score as an extra column."""
        if self.model_ is None:
            raise RuntimeError("Transformer must be fitted before calling transform.")
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        raw_scores = -self.model_.decision_function(X_arr)
        denom = max(self.max_score_ - self.min_score_, 1e-8)
        norm_scores = np.clip((raw_scores - self.min_score_) / denom, 0.0, 1.0).reshape(
            -1, 1
        )
        return np.hstack([X_arr, norm_scores])


def evaluate_isolation_forest_baseline(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
    y_val: pd.Series | np.ndarray,
    amounts_val: Optional[np.ndarray] = None,
    config: Optional[Settings] = None,
) -> dict[str, Any]:
    """Train unsupervised Isolation Forest on train data and benchmark against validation holdout.

    Parameters
    ----------
    X_train, y_train:
        Training data.
    X_val, y_val:
        Validation data.
    amounts_val:
        Amounts for validation expected cost.
    config:
        Project configuration.

    Returns
    -------
    dict[str, Any]
        Dictionary of standalone anomaly benchmark metrics.
    """
    if config is None:
        config = get_settings()

    set_global_seed(config.project.seed)

    X_tr = X_train.values if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)
    X_va = X_val.values if isinstance(X_val, pd.DataFrame) else np.asarray(X_val)
    y_va = y_val.values if isinstance(y_val, pd.Series) else np.asarray(y_val)

    logger.info("Training standalone Isolation Forest baseline...")
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.002,
        random_state=config.project.seed,
        n_jobs=-1,
    )
    iso.fit(X_tr)

    # Inverted decision function: higher score -> higher anomaly
    raw_scores = -iso.decision_function(X_va)
    # Min-max scale to [0, 1] pseudo-probability
    s_min, s_max = np.min(raw_scores), np.max(raw_scores)
    pseudo_probs = (raw_scores - s_min) / max(s_max - s_min, 1e-8)

    opt_thresh, min_cost = find_optimal_threshold(
        y_va,
        pseudo_probs,
        amounts=amounts_val,
        fp_cost=config.evaluation.fp_cost,
        fn_base_cost=config.evaluation.fn_base_cost,
    )

    metrics = compute_all_metrics(
        y_va,
        pseudo_probs,
        threshold=opt_thresh,
        amounts=amounts_val,
        fp_cost=config.evaluation.fp_cost,
        fn_base_cost=config.evaluation.fn_base_cost,
    )
    metrics["optimal_threshold"] = opt_thresh
    logger.info("Isolation Forest Validation Benchmark: %s", metrics)
    return metrics
