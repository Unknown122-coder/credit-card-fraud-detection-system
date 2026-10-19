"""Script to generate and persist SHAP explainability artifacts."""

from __future__ import annotations

from typing import Optional

import joblib
import pandas as pd

from src.config import Settings, get_settings
from src.data.split import load_splits
from src.explain.shap_utils import (
    create_explainer,
    generate_global_plots,
    save_explainer,
)
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


def generate_all_explanations(config: Optional[Settings] = None) -> None:
    """Generate global SHAP figures and serialize TreeExplainer artifact.

    Parameters
    ----------
    config:
        Project configuration. Defaults to get_settings().
    """
    if config is None:
        config = get_settings()

    set_global_seed(config.project.seed)
    model_dir = config.model_dir
    pipeline_path = model_dir / config.artifacts.pipeline_filename

    # Prefer the configured champion; fall back to the tuned XGBoost model.
    if not pipeline_path.exists():
        fallback_path = model_dir / "tuned" / "xgboost_pipeline.joblib"

        if fallback_path.exists():
            pipeline_path = fallback_path
            logger.info(
                "Champion pipeline not found; using tuned XGBoost model at %s",
                pipeline_path,
            )
        else:
            logger.warning(
                "No model found. Checked champion at %s and fallback at %s.",
                model_dir / config.artifacts.pipeline_filename,
                fallback_path,
            )
            return

    logger.info("Loading model from %s", pipeline_path)
    pipeline = joblib.load(pipeline_path)

    # Load validation data for background and explanation samples
    _, val_df, _ = load_splits(config)
    feature_cols = [c for c in val_df.columns if c != config.data.target_column]
    X_val = val_df[feature_cols]

    # Initialize SHAP explainer
    from src.explain.shap_utils import transform_features_through_pipeline

    # Sample validation data once, preserving original feature names.
    X_sample = X_val.sample(
        n=min(300, len(X_val)),
        random_state=config.project.seed,
    )

    # Initialize SHAP using the pipeline's scaled feature space.
    explainer = create_explainer(
        model=pipeline,
        X_background=X_val,
        max_samples=config.explainability.shap_background_samples,
        seed=config.project.seed,
    )

    # Transform explanation rows into the same space used by the tree.
    X_sample_scaled = transform_features_through_pipeline(
        pipeline,
        X_sample,
    )

    # Restore feature names for readable SHAP plots.
    X_sample_scaled = pd.DataFrame(
        X_sample_scaled,
        columns=feature_cols,
        index=X_sample.index,
    )

    # Generate and save global plots.
    figures_dir = config.figures_dir
    logger.info("Generating global SHAP visualizations in %s", figures_dir)

    generate_global_plots(
        explainer=explainer,
        X_sample=X_sample_scaled,
        save_dir=figures_dir,
    )

    # Serialize explainer artifact
    explainer_path = model_dir / config.artifacts.explainer_filename
    save_explainer(explainer, explainer_path)
    logger.info("Saved SHAP explainer artifact → %s", explainer_path)


if __name__ == "__main__":
    generate_all_explanations()
