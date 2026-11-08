"""Unit tests for version-resilient SHAP explainer construction.

Serialized explainers and boosters are version-sensitive: a deployment that
resolves a newer shap or xgboost than the one that trained the model can fail to
read them. These tests pin the behaviour of the rebuild + model-agnostic
fallback chain that keeps local attributions available in that situation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import shap
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from src.explain import shap_utils
from src.explain.shap_utils import (
    build_explainer_for_model,
    compute_local_explanation,
    extract_background,
    extract_tree_estimator,
    transform_features_through_pipeline,
)

FEATURE_COLS: list[str] = ["Time"] + [f"V{i}" for i in range(1, 6)] + ["Amount"]


@pytest.fixture
def fitted_pipeline() -> ImbPipeline:
    """Fit a small scaler + XGBoost pipeline mirroring the champion's structure."""
    rng = np.random.default_rng(7)
    n = 150
    X = pd.DataFrame(
        rng.normal(0, 1, size=(n, len(FEATURE_COLS))), columns=FEATURE_COLS
    )
    y = ((X["V1"] + X["V3"] + rng.normal(0, 0.25, n)) > 0.6).astype(int)

    pipe = ImbPipeline(
        [
            ("scaler", RobustScaler()),
            (
                "classifier",
                XGBClassifier(n_estimators=15, max_depth=3, verbosity=0),
            ),
        ]
    )
    pipe.fit(X, y)
    return pipe


@pytest.fixture
def single_row() -> pd.DataFrame:
    """Single transaction to explain."""
    row = {col: 0.0 for col in FEATURE_COLS}
    row.update({"V1": 1.6, "V3": 1.2, "Amount": 220.0, "Time": 48000.0})
    return pd.DataFrame([row])[FEATURE_COLS]


def _raw_margin(pipeline: ImbPipeline, X: pd.DataFrame) -> float:
    """Raw log-odds margin the explanation must reconstruct."""
    estimator = extract_tree_estimator(pipeline)
    transformed = transform_features_through_pipeline(pipeline, X)
    return float(estimator.predict(transformed, output_margin=True)[0])


def test_extract_tree_estimator_unwraps_pipeline(fitted_pipeline: ImbPipeline) -> None:
    """Verify the final classifier is unwrapped from the pipeline, bare models pass through."""
    estimator = extract_tree_estimator(fitted_pipeline)
    assert isinstance(estimator, XGBClassifier)
    assert extract_tree_estimator(estimator) is estimator


def test_extract_background_recovers_matrix_and_rejects_empty(
    fitted_pipeline: ImbPipeline, single_row: pd.DataFrame
) -> None:
    """Verify background recovery returns a 2-D float matrix, or None when absent."""
    background = transform_features_through_pipeline(fitted_pipeline, single_row)
    explainer = shap.TreeExplainer(
        extract_tree_estimator(fitted_pipeline), data=background
    )

    recovered = extract_background(explainer)
    assert recovered is not None
    assert recovered.ndim == 2
    assert recovered.shape[1] == len(FEATURE_COLS)

    # An explainer carrying no background data yields None rather than raising
    assert extract_background(object()) is None


def test_build_explainer_prefers_tree_backend(
    fitted_pipeline: ImbPipeline, single_row: pd.DataFrame
) -> None:
    """Verify the primary path builds a TreeExplainer whose attributions are additive."""
    explainer = build_explainer_for_model(fitted_pipeline)
    assert isinstance(explainer, shap.TreeExplainer)

    explanation = compute_local_explanation(
        explainer, single_row, top_n=5, pipeline=fitted_pipeline
    )
    assert len(explanation["top_attributions"]) == 5
    assert len(explanation["all_attributions"]) == len(FEATURE_COLS)
    assert explanation["summary_text"]
    # Local accuracy: base value + attributions reconstruct the model margin
    assert explanation["prediction_score"] == pytest.approx(
        _raw_margin(fitted_pipeline, single_row), abs=1e-3
    )


def test_model_agnostic_fallback_when_tree_backend_unavailable(
    fitted_pipeline: ImbPipeline,
    single_row: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a non-tree explainer still yields additive attributions when TreeExplainer fails.

    Simulates the real shap/xgboost version skew, where shap cannot parse the
    serialized booster and TreeExplainer construction raises.
    """

    def _unsupported_booster(*args: object, **kwargs: object) -> object:
        raise ValueError("could not convert string to float: '[5E-1]'")

    tree_explainer_cls = shap.TreeExplainer  # captured before patching
    monkeypatch.setattr(shap_utils.shap, "TreeExplainer", _unsupported_booster)

    background = transform_features_through_pipeline(
        fitted_pipeline,
        pd.DataFrame(
            np.tile(single_row.to_numpy(), (20, 1)) + 0.1, columns=FEATURE_COLS
        ),
    )
    explainer = build_explainer_for_model(fitted_pipeline, background=background)

    assert explainer is not None
    assert not isinstance(explainer, tree_explainer_cls)
    # Fallback backends expose no expected_value; the base value must come
    # from the returned Explanation instead
    assert not hasattr(explainer, "expected_value")

    explanation = compute_local_explanation(
        explainer, single_row, top_n=5, pipeline=fitted_pipeline
    )
    assert len(explanation["top_attributions"]) == 5
    assert explanation["prediction_score"] == pytest.approx(
        _raw_margin(fitted_pipeline, single_row), abs=0.05
    )


def test_build_explainer_returns_none_without_any_backend(
    fitted_pipeline: ImbPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify an unbuildable explainer degrades to None instead of raising."""

    def _boom(*args: object, **kwargs: object) -> object:
        raise ValueError("no backend available")

    monkeypatch.setattr(shap_utils.shap, "TreeExplainer", _boom)
    # No background sample means the model-agnostic fallback cannot be built either
    assert build_explainer_for_model(fitted_pipeline, background=None) is None
