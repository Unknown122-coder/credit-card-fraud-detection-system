"""SHAP-based model explainability for credit card fraud detection.

Provides:
- Global feature importance and beeswarm distributions
- Local per-transaction explanations and waterfall data for UI rendering
- Domain mapping of PCA-anonymized features (V1-V28) to proxy risk archetypes
- Narrative synthesis explaining fraud flags to operational auditors
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# PCA Feature to Risk Archetype Domain Mapping
# ---------------------------------------------------------------------------
FEATURE_GROUP_MAP: dict[str, str] = {
    # V1 - V5: Velocity Proxies
    "V1": "Velocity Proxy (Burst Frequency)",
    "V2": "Velocity Proxy (Transaction Gap)",
    "V3": "Velocity Proxy (Card Velocity)",
    "V4": "Velocity Proxy (Session Frequency)",
    "V5": "Velocity Proxy (Rapid Retry)",
    # V6 - V12: Terminal / Location Proxies
    "V6": "Terminal/Location Proxy (POS Hardware)",
    "V7": "Terminal/Location Proxy (Merchant Terminal)",
    "V8": "Terminal/Location Proxy (Geo-Distance)",
    "V9": "Terminal/Location Proxy (IP Subnet)",
    "V10": "Terminal/Location Proxy (Device Fingerprint)",
    "V11": "Terminal/Location Proxy (Country Mismatch)",
    "V12": "Terminal/Location Proxy (Zipcode Cluster)",
    # V13 - V28: Account & Behavioral Spending Proxies
    "V13": "Account History Proxy (Tenure)",
    "V14": "Account Risk Proxy (Historical Flag)",
    "V15": "Behavioral Profile Proxy (Channel Shift)",
    "V16": "Behavioral Profile Proxy (Spending Deviation)",
    "V17": "Account Risk Proxy (Suspicious Merchant Category)",
    "V18": "Behavioral Profile Proxy (Peer Variance)",
    "V19": "Account Profile Proxy (Limit Proximity)",
    "V20": "Basket Composition Proxy (High-Risk SKU)",
    "V21": "Authentication Proxy (3DS Challenge)",
    "V22": "Authentication Proxy (AVS Mismatch)",
    "V23": "Cardholder Pattern Proxy (Weekend Shift)",
    "V24": "Cardholder Pattern Proxy (Time Dispersion)",
    "V25": "Device Proxy (Browser Profile)",
    "V26": "Payment Network Proxy (Authorization Route)",
    "V27": "Terminal Profile Proxy (MCC Velocity)",
    "V28": "Cross-Border Settlement Proxy",
    # Raw features
    "Amount": "Transaction Magnitude (USD)",
    "Time": "Temporal Timestamp (Seconds)",
    "HourOfDay": "Diurnal Cycle (Hour of Day)",
}


def get_feature_group(feature_name: str) -> str:
    """Return proxy risk archetype for a given feature."""
    return FEATURE_GROUP_MAP.get(feature_name, "General Transaction Attribute")


def create_explainer(
    model: BaseEstimator,
    X_background: pd.DataFrame | np.ndarray,
    max_samples: int = 100,
    seed: int = 42,
) -> shap.Explainer:
    """Create a SHAP explainer instance with memory-bounded background samples.

    Extracts the underlying tree estimator if passed an imblearn/sklearn Pipeline.

    Parameters
    ----------
    model:
        Trained classifier or Pipeline.
    X_background:
        Background reference dataset.
    max_samples:
        Maximum background instances to sample for computational efficiency.
    seed:
        Random state for sampling.

    Returns
    -------
    shap.Explainer
        Fitted SHAP explainer (TreeExplainer when possible).
    """
def transform_features_through_pipeline(model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Transform features through all pipeline steps prior to the final classifier.

    Ensures that TreeExplainer evaluates features in the exact numerical space
    (e.g., scaled by RobustScaler) on which the tree estimator was trained.
    """
    if hasattr(model, "steps") and len(model.steps) > 1:
        X_trans = X
        for name, step in model.steps[:-1]:
            if hasattr(step, "transform"):
                X_trans = step.transform(X_trans)
        return np.asarray(X_trans)
    return np.asarray(X)


def create_explainer(
    model: BaseEstimator,
    X_background: pd.DataFrame | np.ndarray,
    max_samples: int = 100,
    seed: int = 42,
) -> shap.Explainer:
    """Create a SHAP explainer instance with memory-bounded background samples.

    Extracts the underlying tree estimator and transforms the background data
    through the pipeline's preprocessing steps to guarantee attribution validity.
    """
    estimator = model
    if hasattr(model, "named_steps"):
        if "classifier" in model.named_steps:
            estimator = model.named_steps["classifier"]
        elif len(model.steps) > 0:
            estimator = model.steps[-1][1]

    # Convert background to DataFrame/array and subsample
    if isinstance(X_background, pd.DataFrame):
        if len(X_background) > max_samples:
            bg_data = X_background.sample(n=max_samples, random_state=seed)
        else:
            bg_data = X_background.copy()
    else:
        bg_arr = np.asarray(X_background)
        if len(bg_arr) > max_samples:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(bg_arr), size=max_samples, replace=False)
            bg_data = bg_arr[indices]
        else:
            bg_data = bg_arr

    # Preprocess background through pipeline transformers (e.g. RobustScaler)
    transformed_bg = transform_features_through_pipeline(model, bg_data)

    logger.info(
        "Initializing SHAP explainer on estimator %s with %d background rows",
        type(estimator).__name__,
        len(bg_data),
    )

    try:
        explainer = shap.TreeExplainer(estimator, data=transformed_bg)
        logger.info("Created TreeExplainer successfully on transformed background space")
        return explainer
    except Exception as exc:
        logger.warning("TreeExplainer failed (%s). Falling back to shap.Explainer", exc)
        return shap.Explainer(
            estimator.predict_proba if hasattr(estimator, "predict_proba") else estimator,
            transformed_bg,
        )


def compute_shap_values(
    explainer: shap.Explainer,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Compute raw SHAP attribution matrix for positive (fraud) class.

    Parameters
    ----------
    explainer:
        SHAP explainer.
    X:
        Evaluation instances.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, n_features).
    """
    shap_vals = explainer(X) if hasattr(explainer, "__call__") else explainer.shap_values(X)

    if hasattr(shap_vals, "values"):
        vals = shap_vals.values
    else:
        vals = shap_vals

    # Handle multi-class / binary classification output
    if isinstance(vals, list):
        # List of [class_0, class_1]
        vals = vals[1] if len(vals) > 1 else vals[0]
    elif isinstance(vals, np.ndarray) and vals.ndim == 3:
        # Array of shape (n_samples, n_features, n_classes)
        vals = vals[:, :, 1] if vals.shape[2] > 1 else vals[:, :, 0]

    return np.asarray(vals)


def generate_global_plots(
    explainer: shap.Explainer,
    X_sample: pd.DataFrame,
    save_dir: Path | str,
) -> list[str]:
    """Generate and save global beeswarm and feature importance bar plots.

    Parameters
    ----------
    explainer:
        SHAP explainer.
    X_sample:
        Sample DataFrame for global distribution estimation.
    save_dir:
        Output folder for figures.

    Returns
    -------
    list[str]
        List of saved PNG file paths.
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[str] = []

    shap_vals = compute_shap_values(explainer, X_sample)

    # 1. Summary Beeswarm Plot
    fig1 = plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_vals,
        X_sample,
        show=False,
        max_display=15,
        plot_type="dot",
    )
    plt.title("SHAP Global Feature Impact (Beeswarm)", fontsize=13, pad=15)
    plt.tight_layout()
    beeswarm_path = out_dir / "shap_summary_beeswarm.png"
    fig1.savefig(beeswarm_path, dpi=300)
    plt.close(fig1)
    saved_files.append(str(beeswarm_path))

    # 2. Bar Plot (Mean |SHAP|)
    fig2 = plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_vals,
        X_sample,
        show=False,
        max_display=15,
        plot_type="bar",
    )
    plt.title("Mean Absolute SHAP Feature Importance", fontsize=13, pad=15)
    plt.tight_layout()
    bar_path = out_dir / "shap_importance_bar.png"
    fig2.savefig(bar_path, dpi=300)
    plt.close(fig2)
    saved_files.append(str(bar_path))

    logger.info("Global SHAP plots saved: %s", saved_files)
    return saved_files


def compute_local_explanation(
    explainer: shap.Explainer,
    X_single: pd.DataFrame,
    top_n: int = 10,
    pipeline: Optional[Any] = None,
) -> dict[str, Any]:
    """Compute local SHAP explanation and return formatted attributions for UI/API.

    Parameters
    ----------
    explainer:
        SHAP explainer.
    X_single:
        Single-row DataFrame of features.
    top_n:
        Number of top contributing features to highlight.
    pipeline:
        Optional trained pipeline whose preprocessing transformers will be applied
        to align the evaluation vector with the tree estimator's feature space.

    Returns
    -------
    dict[str, Any]
        Structured explanation payload.
    """
    if pipeline is not None:
        X_eval = transform_features_through_pipeline(pipeline, X_single)
    else:
        X_eval = X_single

    shap_vals = compute_shap_values(explainer, X_eval)[0]

    # Expected value (base value)
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[1]) if len(base_val) > 1 else float(base_val[0])
    else:
        base_val = float(base_val)

    feature_names = list(X_single.columns)
    feature_values = X_single.iloc[0].values

    attributions: list[dict[str, Any]] = []
    for name, val, shap_val in zip(feature_names, feature_values, shap_vals):
        attributions.append({
            "feature_name": str(name),
            "feature_value": float(val),
            "shap_value": float(shap_val),
            "feature_group": get_feature_group(str(name)),
        })

    # Sort descending by absolute SHAP impact
    attributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    top_attributions = attributions[:top_n]

    # Compute prediction score
    prediction_score = base_val + float(np.sum(shap_vals))

    narrative = generate_explanation_narrative(top_attributions)

    return {
        "base_value": round(base_val, 4),
        "prediction_score": round(prediction_score, 4),
        "top_attributions": top_attributions,
        "all_attributions": attributions,
        "summary_text": narrative,
    }


def generate_explanation_narrative(attributions: list[dict[str, Any]]) -> str:
    """Generate clear, audit-compliant English narrative explaining the risk drivers.

    Parameters
    ----------
    attributions:
        Ranked feature attribution dictionaries.

    Returns
    -------
    str
        Markdown write-up template synthesizing top risk signals.
    """
    pos_drivers = [a for a in attributions if a["shap_value"] > 0]
    neg_drivers = [a for a in attributions if a["shap_value"] < 0]

    narrative_parts: list[str] = []

    if pos_drivers:
        top_pos = pos_drivers[:3]
        pos_desc = ", ".join(
            f"**{a['feature_name']}** ({a['feature_group']}: +{a['shap_value']:.2f} risk)"
            for a in top_pos
        )
        narrative_parts.append(f"Primary elevated risk indicators: {pos_desc}.")

    if neg_drivers:
        top_neg = neg_drivers[:2]
        neg_desc = ", ".join(
            f"**{a['feature_name']}** ({a['feature_group']}: {a['shap_value']:.2f} risk reduction)"
            for a in top_neg
        )
        narrative_parts.append(f"Mitigating factors: {neg_desc}.")

    if not narrative_parts:
        return "Transaction risk score reflects baseline population distribution with no extreme feature deviations."

    return " ".join(narrative_parts)


def save_explainer(explainer: shap.Explainer, save_path: str | Path) -> None:
    """Serialize fitted explainer using joblib."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(explainer, p, compress=3)
    logger.info("Saved SHAP explainer → %s", p)


def load_explainer(load_path: str | Path) -> shap.Explainer:
    """Load serialized SHAP explainer."""
    p = Path(load_path)
    if not p.exists():
        raise FileNotFoundError(f"SHAP explainer artifact not found at {p}")
    return joblib.load(p)
