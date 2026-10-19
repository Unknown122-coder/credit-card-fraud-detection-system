"""Custom scikit-learn-compatible transformers for feature engineering.

All transformers follow the sklearn estimator contract:

* ``fit(X, y=None)`` — learns nothing or stores metadata; returns ``self``.
* ``transform(X)`` — applies the transformation and returns a DataFrame.
* ``get_params`` / ``set_params`` — inherited from :class:`BaseEstimator`.

They are fully picklable and compatible with :mod:`joblib` serialisation,
:class:`sklearn.pipeline.Pipeline`, and :func:`sklearn.utils.validation.check_is_fitted`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.utils.validation import check_is_fitted

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LogAmountTransformer
# ---------------------------------------------------------------------------


class LogAmountTransformer(BaseEstimator, TransformerMixin):
    """Apply ``log1p`` to the *Amount* column.

    After transformation the original ``Amount`` column is replaced by
    ``Amount`` containing ``log1p(Amount)``.

    Parameters
    ----------
    amount_col : str
        Name of the amount column (default ``"Amount"``).

    Attributes
    ----------
    fitted_ : bool
        Set to ``True`` after :meth:`fit` is called.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"Amount": [0.0, 1.0, 100.0], "V1": [0.1, 0.2, 0.3]})
    >>> transformer = LogAmountTransformer()
    >>> transformer.fit_transform(df)["Amount"].tolist()  # doctest: +SKIP
    [0.0, 0.6931471805599453, 4.61512051684126]
    """

    def __init__(self, amount_col: str = "Amount") -> None:
        self.amount_col = amount_col

    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
    ) -> LogAmountTransformer:
        """Fit the transformer (no-op, but required by the sklearn contract).

        Parameters
        ----------
        X:
            Input DataFrame.  Must contain ``self.amount_col``.
        y:
            Ignored.

        Returns
        -------
        LogAmountTransformer
            ``self``
        """
        if self.amount_col not in X.columns:
            raise ValueError(
                f"Column '{self.amount_col}' not found in input DataFrame. "
                f"Available: {X.columns.tolist()}"
            )
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply ``log1p`` to the amount column.

        Parameters
        ----------
        X:
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            Copy of *X* with the amount column log-transformed.
        """
        check_is_fitted(self, "fitted_")
        X_out = X.copy()
        X_out[self.amount_col] = np.log1p(X_out[self.amount_col])
        logger.debug("Applied log1p to '%s'.", self.amount_col)
        return X_out


# ---------------------------------------------------------------------------
# HourOfDayTransformer
# ---------------------------------------------------------------------------


class HourOfDayTransformer(BaseEstimator, TransformerMixin):
    """Convert the ``Time`` column (seconds since first transaction) to
    *hour of day* (``Time % 86400 / 3600``).

    Adds a ``HourOfDay`` column.  Optionally drops the original ``Time``
    column.

    Parameters
    ----------
    time_col : str
        Name of the time column (default ``"Time"``).
    drop_original : bool
        If ``True`` (default), the original *time_col* is removed.

    Attributes
    ----------
    fitted_ : bool
        Set to ``True`` after :meth:`fit` is called.
    """

    SECONDS_PER_DAY: int = 86_400
    SECONDS_PER_HOUR: int = 3_600

    def __init__(
        self,
        time_col: str = "Time",
        drop_original: bool = True,
        drop_original_time: bool | None = None,
    ) -> None:
        self.time_col = time_col
        self.drop_original = (
            drop_original if drop_original_time is None else drop_original_time
        )
        self.drop_original_time = drop_original_time

    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
    ) -> HourOfDayTransformer:
        """Fit the transformer (validates the time column exists).

        Parameters
        ----------
        X:
            Input DataFrame.
        y:
            Ignored.

        Returns
        -------
        HourOfDayTransformer
            ``self``
        """
        if self.time_col not in X.columns:
            raise ValueError(
                f"Column '{self.time_col}' not found in input DataFrame. "
                f"Available: {X.columns.tolist()}"
            )
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute *HourOfDay* from the time column.

        Parameters
        ----------
        X:
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            Copy of *X* with ``HourOfDay`` added (and ``Time`` optionally
            dropped).
        """
        check_is_fitted(self, "fitted_")
        X_out = X.copy()
        X_out["HourOfDay"] = (
            X_out[self.time_col] % self.SECONDS_PER_DAY
        ) / self.SECONDS_PER_HOUR
        if self.drop_original:
            X_out = X_out.drop(columns=[self.time_col])
            logger.debug(
                "Created 'HourOfDay' and dropped '%s'.", self.time_col
            )
        else:
            logger.debug("Created 'HourOfDay' (kept '%s').", self.time_col)
        return X_out


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------


def build_preprocessor(config: Settings | None = None) -> Pipeline:
    """Build a scikit-learn :class:`Pipeline` for feature preprocessing.

    The pipeline is assembled dynamically based on the config flags:

    1. If ``config.features.log_transform_amount`` → :class:`LogAmountTransformer`.
    2. If ``config.features.add_hour_feature`` → :class:`HourOfDayTransformer`.
    3. Always appends a :class:`~sklearn.preprocessing.RobustScaler` as the
       final step (robust to outliers / extreme fraud amounts).

    Parameters
    ----------
    config:
        Project settings.  Falls back to :func:`get_settings` when ``None``.

    Returns
    -------
    Pipeline
        Ready-to-use sklearn ``Pipeline``.

    Examples
    --------
    >>> from src.config import get_settings
    >>> pipe = build_preprocessor(get_settings())
    >>> pipe.steps  # doctest: +SKIP
    [('log_amount', LogAmountTransformer()), ('hour_of_day', HourOfDayTransformer()), ('robust_scaler', RobustScaler())]
    """
    if config is None:
        config = get_settings()

    steps: list[tuple[str, BaseEstimator]] = []

    if config.features.log_transform_amount:
        steps.append(("log_amount", LogAmountTransformer(amount_col="Amount")))
        logger.info("Preprocessor: added LogAmountTransformer.")

    if config.features.add_hour_feature:
        steps.append(
            (
                "hour_of_day",
                HourOfDayTransformer(time_col="Time", drop_original=True),
            )
        )
        logger.info("Preprocessor: added HourOfDayTransformer (drop Time).")

    steps.append(("robust_scaler", RobustScaler()))
    logger.info("Preprocessor: added RobustScaler as final step.")

    pipeline = Pipeline(steps)
    logger.info(
        "Built preprocessor pipeline with %d step(s): %s",
        len(steps),
        [name for name, _ in steps],
    )
    return pipeline
