"""Pydantic-based configuration management for the fraud detection system.

Loads settings from ``configs/config.yaml`` and exposes them as strongly-typed
Pydantic models.  A module-level :func:`get_settings` helper provides a cached
singleton so the config is read from disk exactly once.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root — two levels up from this file: src/config.py → project root
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


# ── Sub-models ────────────────────────────────────────────────────────────


class ProjectConfig(BaseModel):
    """Top-level project metadata."""

    name: str = "credit-card-fraud-detection"
    version: str = "1.0.0"
    seed: int = 42
    log_level: str = "INFO"


class DataConfig(BaseModel):
    """Data ingestion & splitting knobs."""

    dataset_name: str = "ulb_creditcard"
    raw_path: str = "data/raw/creditcard.csv"
    processed_dir: str = "data/processed"
    split_strategy: str = "time_based"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    target_column: str = "Class"
    kaggle_dataset: str = "mlg-ulb/creditcardfraud"

    @field_validator("split_strategy")
    @classmethod
    def _validate_strategy(cls, v: str) -> str:
        """Ensure split strategy is one of the supported values."""
        allowed = {"time_based", "stratified"}
        if v not in allowed:
            raise ValueError(f"split_strategy must be one of {allowed}, got '{v}'")
        return v

    @field_validator("train_ratio", "val_ratio", "test_ratio")
    @classmethod
    def _validate_ratio(cls, v: float) -> float:
        """Ensure ratios are in (0, 1)."""
        if not 0.0 < v < 1.0:
            raise ValueError(f"Ratio must be in (0, 1), got {v}")
        return v


class FeaturesConfig(BaseModel):
    """Feature engineering settings."""

    pca_columns: list[str] = Field(
        default_factory=lambda: [f"V{i}" for i in range(1, 29)]
    )
    raw_columns: list[str] = Field(default_factory=lambda: ["Time", "Amount"])
    log_transform_amount: bool = True
    add_hour_feature: bool = True


class TrainingConfig(BaseModel):
    """Training hyper-parameters & resampling strategies."""

    cv_folds: int = 5
    resampling_strategies: list[str] = Field(
        default_factory=lambda: [
            "none",
            "class_weight",
            "smote",
            "smote_tomek",
            "random_undersample",
            "balanced_rf",
        ]
    )
    smote_k_neighbors: int = 5
    smote_sampling_strategy: float = 0.1


class TuningConfig(BaseModel):
    """Optuna hyperparameter tuning settings."""

    n_trials: int = 100
    timeout_seconds: int = 3600
    pruner: str = "median"
    sampler: str = "tpe"
    xgboost_trials: int = 100
    lightgbm_trials: int = 100
    catboost_trials: int = 75


class EvaluationConfig(BaseModel):
    """Evaluation & calibration knobs."""

    fp_cost: float = 10.0
    fn_base_cost: float = 25.0
    threshold_steps: int = 1000
    bootstrap_iterations: int = 1000
    bootstrap_confidence: float = 0.95
    calibration_method: str = "isotonic"


class ExplainabilityConfig(BaseModel):
    """SHAP / explainability settings."""

    shap_background_samples: int = 100
    top_features_display: int = 10
    save_global_plots: bool = True
    figures_dir: str = "docs/figures"


class ServingConfig(BaseModel):
    """API / UI serving settings."""

    host: str = "0.0.0.0"
    api_port: int = 8000
    ui_port: int = 7860
    max_upload_bytes: int = 10_485_760
    max_batch_rows: int = 10_000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class MLflowConfig(BaseModel):
    """MLflow experiment tracking settings."""

    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "fraud-detection"
    artifact_location: str = "mlruns"


class ArtifactsConfig(BaseModel):
    """Paths for saved model artefacts."""

    model_dir: str = "models/registry"
    pipeline_filename: str = "champion_pipeline.joblib"
    calibrator_filename: str = "calibrator.joblib"
    threshold_filename: str = "threshold_config.json"
    metadata_filename: str = "model_metadata.json"
    explainer_filename: str = "shap_explainer.joblib"
    feature_schema_filename: str = "feature_schema.json"


# ── Root settings model ──────────────────────────────────────────────────


class Settings(BaseModel):
    """Root configuration container for the entire project.

    All relative paths stored in sub-configs are resolved against
    :pydata:`PROJECT_ROOT` via the helper properties on this class.
    """

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)

    # ── Convenience path helpers ──────────────────────────────────────

    @property
    def raw_data_path(self) -> Path:
        """Absolute path to the raw CSV file."""
        return PROJECT_ROOT / self.data.raw_path

    @property
    def processed_dir(self) -> Path:
        """Absolute path to the processed-data directory."""
        return PROJECT_ROOT / self.data.processed_dir

    @property
    def model_dir(self) -> Path:
        """Absolute path to the model registry directory."""
        return PROJECT_ROOT / self.artifacts.model_dir

    @property
    def figures_dir(self) -> Path:
        """Absolute path to the figures directory."""
        return PROJECT_ROOT / self.explainability.figures_dir

    def get_all_feature_columns(self) -> list[str]:
        """Return the ordered list of all feature column names (PCA + raw)."""
        return self.features.pca_columns + self.features.raw_columns


# ── Loader helpers ────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read and parse a YAML file.

    Parameters
    ----------
    path:
        Absolute or relative path to the YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed YAML as a dictionary.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@functools.lru_cache(maxsize=1)
def get_settings(
    config_path: str | Path | None = None,
) -> Settings:
    """Load and cache the project :class:`Settings`.

    Parameters
    ----------
    config_path:
        Override path to the YAML config.  Defaults to
        ``<PROJECT_ROOT>/configs/config.yaml``.

    Returns
    -------
    Settings
        Validated, frozen settings object.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)

    raw = _load_yaml(config_path)
    settings = Settings(**raw)

    # Configure root logger level based on the project config.
    logging.basicConfig(
        level=getattr(logging, settings.project.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info(
        "Settings loaded from %s  (project=%s, version=%s)",
        config_path,
        settings.project.name,
        settings.project.version,
    )
    return settings
