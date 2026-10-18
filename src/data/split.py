"""Train / validation / test splitting with pluggable strategies.

Supports two split strategies selectable via ``config.data.split_strategy``:

* **time_based** — sort by ``Time`` column; first 70 % → train, next 15 % →
  val, last 15 % → test.  Preserves temporal ordering so that the model is
  never evaluated on "past" data.
* **stratified** — random split using :func:`sklearn.model_selection.train_test_split`
  with stratification on the target column to preserve class ratios.

Splits are persisted as Parquet files in the configured processed directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split_data(
    df: pd.DataFrame,
    config: Settings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split *df* into train / validation / test sets.

    Dispatches to the strategy defined in ``config.data.split_strategy``.

    Parameters
    ----------
    df:
        Full dataset to split.
    config:
        Project settings.  Falls back to :func:`get_settings` when ``None``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train, val, test)`` DataFrames.

    Raises
    ------
    ValueError
        If an unsupported strategy is specified.
    """
    if config is None:
        config = get_settings()

    strategy = config.data.split_strategy
    dispatch: dict[
        str,
        type[
            # Callable signature — accepts (df, config) → 3-tuple
            ...,  # type: ignore[type-arg]
        ],
    ] = {
        "time_based": time_based_split,
        "stratified": stratified_split,
    }

    if strategy not in dispatch:
        raise ValueError(
            f"Unknown split strategy '{strategy}'.  "
            f"Supported: {sorted(dispatch.keys())}"
        )

    logger.info("Splitting data using '%s' strategy …", strategy)
    dup_cols = [c for c in df.columns if c != "Time"]
    if df.duplicated(subset=dup_cols).any():
        logger.warning(
            "Input DataFrame contains duplicate transactions across feature columns. "
            "Deduplicating (keeping first occurrence chronologically) to prevent cross-split leakage."
        )
        df = (
            df.sort_values("Time", kind="mergesort")
            .drop_duplicates(subset=dup_cols, keep="first")
            .reset_index(drop=True)
        )

    train, val, test = dispatch[strategy](df, config)  # type: ignore[operator]

    _assert_no_overlap(train, val, test)
    _log_split_summary(train, val, test, config)

    return train, val, test


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def time_based_split(
    df: pd.DataFrame,
    config: Settings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by sorting on the ``Time`` column and cutting at ratio boundaries.

    Parameters
    ----------
    df:
        Full dataset.
    config:
        Project settings.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train, val, test)`` — each a contiguous temporal slice.
    """
    if config is None:
        config = get_settings()

    df_sorted = df.sort_values("Time", kind="mergesort").reset_index(drop=True)

    n = len(df_sorted)
    train_end = int(n * config.data.train_ratio)
    val_end = train_end + int(n * config.data.val_ratio)

    train = df_sorted.iloc[:train_end].copy()
    val = df_sorted.iloc[train_end:val_end].copy()
    test = df_sorted.iloc[val_end:].copy()

    logger.debug(
        "Time-based split boundaries: train=[0, %d)  val=[%d, %d)  test=[%d, %d)",
        train_end,
        train_end,
        val_end,
        val_end,
        n,
    )

    return train, val, test


def stratified_split(
    df: pd.DataFrame,
    config: Settings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified random split preserving class ratios in each fold.

    First splits into train vs. (val + test), then splits the remainder
    into val and test.

    Parameters
    ----------
    df:
        Full dataset.
    config:
        Project settings.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train, val, test)`` — each retains the global fraud ratio.
    """
    if config is None:
        config = get_settings()

    target = config.data.target_column
    seed = config.project.seed
    train_ratio = config.data.train_ratio
    val_ratio = config.data.val_ratio
    test_ratio = config.data.test_ratio

    # First split: train vs. (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_df, val_test_df = train_test_split(
        df,
        test_size=val_test_ratio,
        stratify=df[target],
        random_state=seed,
    )

    # Second split: val vs. test  (proportion within the remainder)
    relative_test = test_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=relative_test,
        stratify=val_test_df[target],
        random_state=seed,
    )

    logger.debug(
        "Stratified split ratios requested: train=%.2f  val=%.2f  test=%.2f",
        train_ratio,
        val_ratio,
        test_ratio,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

_SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    config: Settings | None = None,
) -> None:
    """Persist train / val / test DataFrames as Parquet files.

    Files are saved to ``<processed_dir>/{train,val,test}.parquet``.

    Parameters
    ----------
    train, val, test:
        The three split DataFrames.
    config:
        Project settings.
    """
    if config is None:
        config = get_settings()

    out_dir: Path = config.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, split_df in zip(_SPLIT_NAMES, (train, val, test), strict=True):
        path = out_dir / f"{name}.parquet"
        split_df.to_parquet(path, index=False, engine="pyarrow")
        logger.info("Saved %s split → %s  (%d rows)", name, path, len(split_df))


def load_splits(
    config: Settings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load previously saved train / val / test Parquet files.

    Parameters
    ----------
    config:
        Project settings.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train, val, test)`` DataFrames.

    Raises
    ------
    FileNotFoundError
        If any of the Parquet files are missing.
    """
    if config is None:
        config = get_settings()

    out_dir: Path = config.processed_dir
    splits: list[pd.DataFrame] = []

    for name in _SPLIT_NAMES:
        path = out_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Split file not found: {path}.  "
                f"Run split_data() and save_splits() first."
            )
        split_df = pd.read_parquet(path, engine="pyarrow")
        logger.info("Loaded %s split ← %s  (%d rows)", name, path, len(split_df))
        splits.append(split_df)

    return splits[0], splits[1], splits[2]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assert_no_overlap(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Assert that the three splits have no overlapping row indices.

    Uses a hash of each row to detect overlap even after ``reset_index``.

    Parameters
    ----------
    train, val, test:
        DataFrames to check.

    Raises
    ------
    AssertionError
        If any two splits share rows.
    """
    # Use object_id-based approach: compare by hashing all column values.
    def _row_hashes(df: pd.DataFrame) -> set[int]:
        """Return a set of hashes for each row in *df*."""
        return set(
            pd.util.hash_pandas_object(df, index=False).values.tolist()
        )

    h_train = _row_hashes(train)
    h_val = _row_hashes(val)
    h_test = _row_hashes(test)

    overlap_tv = h_train & h_val
    overlap_tt = h_train & h_test
    overlap_vt = h_val & h_test

    assert not overlap_tv, (
        f"Train and val share {len(overlap_tv)} row hash(es)."
    )
    assert not overlap_tt, (
        f"Train and test share {len(overlap_tt)} row hash(es)."
    )
    assert not overlap_vt, (
        f"Val and test share {len(overlap_vt)} row hash(es)."
    )
    logger.debug("No-overlap assertion passed for all splits.")


def _log_split_summary(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    config: Settings,
) -> None:
    """Log size and fraud-ratio summary for each split.

    Parameters
    ----------
    train, val, test:
        DataFrames to summarise.
    config:
        Project settings (for target column name).
    """
    target = config.data.target_column
    total = len(train) + len(val) + len(test)

    for name, split_df in zip(_SPLIT_NAMES, (train, val, test), strict=True):
        n = len(split_df)
        n_fraud = int(split_df[target].sum())
        pct = n / total * 100 if total > 0 else 0.0
        fraud_rate = n_fraud / n * 100 if n > 0 else 0.0
        logger.info(
            "  %-5s — %7d rows (%5.1f%%)  |  fraud=%5d (%.4f%%)",
            name,
            n,
            pct,
            n_fraud,
            fraud_rate,
        )
