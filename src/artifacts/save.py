"""Model artifact persistence and versioned metadata registry.

Saves and loads all production artifacts:
- champion_pipeline.joblib: Preprocessor + trained estimator
- calibrator.joblib: CalibratedClassifierCV wrapper
- threshold_config.json: Cost-optimal operational threshold and matrix parameters
- feature_schema.json: Column names, dtypes, and expected empirical bounds
- model_metadata.json: Version, git commit hash, training timestamp, test-set metrics
- shap_explainer.joblib: Serialized SHAP explainer
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from src.config import Settings, get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelArtifacts:
    """Container holding loaded production artifacts."""

    pipeline: Any
    calibrator: Optional[Any]
    threshold: float
    metadata: dict[str, Any]
    feature_schema: dict[str, Any]
    explainer: Optional[Any] = None


def get_git_hash() -> str:
    """Retrieve short Git commit hash if in a Git repository."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "development-local"


def create_feature_schema(X_train: pd.DataFrame) -> dict[str, Any]:
    """Extract column types and statistical boundaries for runtime input validation.

    Parameters
    ----------
    X_train:
        Reference training DataFrame.

    Returns
    -------
    dict[str, Any]
        Feature schema mapping column names to validation constraints.
    """
    schema: dict[str, Any] = {
        "columns": list(X_train.columns),
        "n_features": len(X_train.columns),
        "properties": {},
    }

    for col in X_train.columns:
        s = X_train[col]
        schema["properties"][col] = {
            "dtype": str(s.dtype),
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "std": float(s.std()),
        }

    return schema


def save_model_artifacts(
    pipeline: Any,
    calibrator: Optional[Any],
    threshold: float,
    metrics: dict[str, Any],
    X_train: pd.DataFrame,
    config: Optional[Settings] = None,
    explainer: Optional[Any] = None,
    output_dir: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Persist all production deployment artifacts to disk.

    Parameters
    ----------
    pipeline:
        Fitted model or imblearn Pipeline.
    calibrator:
        Fitted CalibratedClassifierCV or None.
    threshold:
        Optimal decision threshold.
    metrics:
        Evaluation metrics on validation/test holdout.
    X_train:
        Training data for feature schema generation.
    config:
        Project configuration.
    explainer:
        Fitted SHAP explainer.
    output_dir:
        Target directory for model artifacts. Defaults to config.model_dir.

    Returns
    -------
    dict[str, Any]
        Metadata dictionary.
    """
    if config is None:
        config = get_settings()

    target_dir = Path(output_dir) if output_dir else config.model_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pipeline
    pipeline_path = target_dir / config.artifacts.pipeline_filename
    joblib.dump(pipeline, pipeline_path, compress=3)
    logger.info("Saved champion pipeline → %s", pipeline_path)

    # 2. Calibrator
    if calibrator is not None:
        calibrator_path = target_dir / config.artifacts.calibrator_filename
        joblib.dump(calibrator, calibrator_path, compress=3)
        logger.info("Saved calibrator → %s", calibrator_path)

    # 3. Threshold Configuration
    threshold_data = {
        "optimal_threshold": float(threshold),
        "fp_cost": float(config.evaluation.fp_cost),
        "fn_base_cost": float(config.evaluation.fn_base_cost),
        "selection_strategy": "expected_cost_minimization_validation",
    }
    threshold_path = target_dir / config.artifacts.threshold_filename
    threshold_path.write_text(json.dumps(threshold_data, indent=2), encoding="utf-8")
    logger.info("Saved threshold configuration → %s", threshold_path)

    # 4. Feature Schema
    schema = create_feature_schema(X_train)
    schema_path = target_dir / config.artifacts.feature_schema_filename
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    logger.info("Saved feature schema → %s", schema_path)

    # 5. Explainer
    if explainer is not None:
        explainer_path = target_dir / config.artifacts.explainer_filename
        joblib.dump(explainer, explainer_path, compress=3)
        logger.info("Saved SHAP explainer → %s", explainer_path)

    # 6. Model Metadata
    metadata = {
        "model_name": config.project.name,
        "version": config.project.version,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_hash(),
        "python_version": platform.python_version(),
        "operating_threshold": float(threshold),
        "metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in metrics.items()},
        "config_snapshot": {
            "dataset_name": config.data.dataset_name,
            "split_strategy": config.data.split_strategy,
            "cv_folds": config.training.cv_folds,
            "calibration_method": config.evaluation.calibration_method,
        },
    }
    metadata_path = target_dir / config.artifacts.metadata_filename
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved versioned model metadata → %s", metadata_path)

    return metadata


def load_model_artifacts(model_dir: str | Path) -> ModelArtifacts:
    """Load all production artifacts from directory into ModelArtifacts container.

    Parameters
    ----------
    model_dir:
        Directory containing serialized artifacts.

    Returns
    -------
    ModelArtifacts
        Loaded pipeline, calibrator, threshold config, metadata, schema, and explainer.
    """
    p = Path(model_dir)

    pipeline_file = p / "champion_pipeline.joblib"
    if not pipeline_file.exists():
        raise FileNotFoundError(f"Missing pipeline artifact: {pipeline_file}")
    pipeline = joblib.load(pipeline_file)

    calibrator_file = p / "calibrator.joblib"
    calibrator = joblib.load(calibrator_file) if calibrator_file.exists() else None

    threshold_file = p / "threshold_config.json"
    if threshold_file.exists():
        thresh_info = json.loads(threshold_file.read_text(encoding="utf-8"))
        threshold = float(thresh_info.get("optimal_threshold", 0.5))
    else:
        threshold = 0.5

    metadata_file = p / "model_metadata.json"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.exists() else {}

    schema_file = p / "feature_schema.json"
    schema = json.loads(schema_file.read_text(encoding="utf-8")) if schema_file.exists() else {}

    explainer_file = p / "shap_explainer.joblib"
    explainer = joblib.load(explainer_file) if explainer_file.exists() else None

    return ModelArtifacts(
        pipeline=pipeline,
        calibrator=calibrator,
        threshold=threshold,
        metadata=metadata,
        feature_schema=schema,
        explainer=explainer,
    )


if __name__ == "__main__":
    settings = get_settings()
    logger.info("Artifact registry module ready.")
