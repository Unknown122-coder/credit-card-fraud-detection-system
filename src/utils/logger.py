"""Structured logging setup for the fraud-detection system.

Provides a JSON-style formatter so that every log record is a parseable
JSON object — useful for shipping logs to ELK / CloudWatch / Grafana.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone


class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        """Format *record* as a JSON string.

        Parameters
        ----------
        record : logging.LogRecord
            Standard library log record.

        Returns
        -------
        str
            Serialised JSON string (single line, no trailing newline).
        """
        log_entry: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Append exception info when present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure the root logger with JSON structured output.

    Calling this function more than once is safe — it removes any
    previously-attached handlers before adding a fresh one.

    Parameters
    ----------
    log_level : str, optional
        Python log-level name (``"DEBUG"``, ``"INFO"``, etc.).
        Defaults to ``"INFO"``.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root = logging.getLogger()

    # Avoid duplicate handlers on repeated calls
    root.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(_JSONFormatter())
    handler.setLevel(numeric_level)

    root.setLevel(numeric_level)
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger with the given *name*.

    The logger inherits the root configuration set by :func:`setup_logging`.
    If ``setup_logging`` has not been called yet, the logger will still work
    with Python's default settings.

    Parameters
    ----------
    name : str
        Dotted logger name — typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    return logging.getLogger(name)
