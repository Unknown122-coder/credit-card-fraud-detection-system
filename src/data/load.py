"""Data loading module with swappable dataset support.

Provides config-driven loading and schema validation for the credit card
fraud detection dataset.  Currently supports the ``ulb_creditcard`` dataset
(Kaggle ULB Credit Card Fraud).  New datasets can be added by extending the
``_DATASET_LOADERS`` dispatch table.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected dtypes for schema validation
# ---------------------------------------------------------------------------
_EXPECTED_DTYPES: dict[str, type] = {
    "Time": np.floating,
    "Amount": np.floating,
    "Class": np.integer,
}
# V1–V28 are all float
for _i in range(1, 29):
    _EXPECTED_DTYPES[f"V{_i}"] = np.floating


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_feature_columns(config: Settings) -> list[str]:
    """Return the ordered list of feature column names from the config.

    The list comprises the 28 PCA columns (V1–V28) followed by the two
    raw columns (Time, Amount), as declared in ``config.features``.

    Parameters
    ----------
    config:
        Project settings object.

    Returns
    -------
    list[str]
        Ordered feature column names.
    """
    return config.get_all_feature_columns()


def validate_schema(df: pd.DataFrame, config: Settings) -> bool:
    """Validate that *df* conforms to the expected schema.

    Checks:
    1. All feature columns exist.
    2. The target column exists.
    3. Column dtypes match expectations (numeric types).

    Parameters
    ----------
    df:
        DataFrame to validate.
    config:
        Project settings object.

    Returns
    -------
    bool
        ``True`` if the schema is valid.

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    expected_cols = get_feature_columns(config) + [config.data.target_column]

    # 1. Check for missing columns.
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in DataFrame: {sorted(missing)}.  "
            f"Available columns: {sorted(df.columns.tolist())}"
        )

    # 2. Check dtypes.
    dtype_errors: list[str] = []
    for col, expected_base in _EXPECTED_DTYPES.items():
        if col not in df.columns:
            continue
        if not np.issubdtype(df[col].dtype, expected_base):
            dtype_errors.append(
                f"  {col}: expected {expected_base.__name__}, got {df[col].dtype}"
            )

    if dtype_errors:
        raise ValueError("Dtype mismatches detected:\n" + "\n".join(dtype_errors))

    logger.info("Schema validation passed — %d columns verified.", len(expected_cols))
    return True


def load_dataset(config: Settings | None = None) -> pd.DataFrame:
    """Load the dataset specified in *config* and validate its schema.

    The function dispatches to the appropriate loader based on
    ``config.data.dataset_name``.  After loading the raw CSV the schema
    is validated, and summary statistics are logged.

    Parameters
    ----------
    config:
        Project settings.  If ``None``, :func:`get_settings` is called to
        obtain the default settings.

    Returns
    -------
    pd.DataFrame
        Loaded and validated DataFrame.

    Raises
    ------
    FileNotFoundError
        If the raw data file does not exist.
    ValueError
        If the dataset name is unknown or the schema is invalid.
    """
    if config is None:
        config = get_settings()

    dataset_name: str = config.data.dataset_name
    if dataset_name not in _DATASET_LOADERS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'.  "
            f"Supported datasets: {sorted(_DATASET_LOADERS.keys())}"
        )

    loader = _DATASET_LOADERS[dataset_name]
    df = loader(config)

    # Validate schema
    validate_schema(df, config)

    # Operational deduplication policy (retain first occurrence chronologically)
    initial_rows = len(df)
    dup_cols = [c for c in df.columns if c != "Time"]
    df = (
        df.sort_values("Time", kind="mergesort")
        .drop_duplicates(subset=dup_cols, keep="first")
        .reset_index(drop=True)
    )
    n_dropped = initial_rows - len(df)
    if n_dropped > 0:
        logger.info(
            "Deduplication policy applied: removed %d duplicate transaction(s) (retained first occurrence)",
            n_dropped,
        )

    # Log summary
    target = config.data.target_column
    n_total = len(df)
    n_fraud = int(df[target].sum())
    n_legit = n_total - n_fraud
    fraud_pct = n_fraud / n_total * 100 if n_total > 0 else 0.0

    logger.info(
        "Dataset '%s' loaded — shape=%s  |  legit=%d  fraud=%d (%.4f%%)",
        dataset_name,
        df.shape,
        n_legit,
        n_fraud,
        fraud_pct,
    )

    return df


# ---------------------------------------------------------------------------
# Per-dataset loader implementations
# ---------------------------------------------------------------------------


def _load_ulb_creditcard(config: Settings) -> pd.DataFrame:
    """Load the ULB Credit Card Fraud dataset from a CSV file.

    Parameters
    ----------
    config:
        Project settings (``config.raw_data_path`` is used).

    Returns
    -------
    pd.DataFrame
        Raw DataFrame read from disk.

    Raises
    ------
    FileNotFoundError
        If the CSV file is missing.
    """
    raw_path: Path = config.raw_data_path
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {raw_path}.  "
            f"Download it with:  kaggle datasets download -d {config.data.kaggle_dataset}  "
            f"and extract it to {raw_path.parent}"
        )

    logger.info("Reading CSV from %s …", raw_path)
    df = pd.read_csv(raw_path)
    logger.debug("Raw CSV shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Dispatch table — add new datasets here
# ---------------------------------------------------------------------------
_DATASET_LOADERS: dict[str, Callable[[Settings], pd.DataFrame]] = {
    "ulb_creditcard": _load_ulb_creditcard,
}
