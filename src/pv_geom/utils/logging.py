"""Logging setup for the command line.

The library only ever calls ``logging.getLogger(__name__)``; nothing is
configured on import, so an application embedding pv-geom keeps control. The
CLI calls :func:`configure_logging` to get readable progress on the console,
and can add a JSON-lines file for a run's permanent record.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PACKAGE_LOGGER = "pv_geom"


class JSONFormatter(logging.Formatter):
    """One JSON object per record: timestamp, level, logger, message."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """``[pv-geom] message``, with the level shown only when it matters."""

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if record.levelno >= logging.WARNING:
            msg = f"{record.levelname}: {msg}"
        if record.exc_info:
            msg = f"{msg}\n{self.formatException(record.exc_info)}"
        return f"[pv-geom] {msg}"


def configure_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Console logging for the CLI: pv-geom's own progress at INFO (DEBUG with
    ``verbose``, WARNING with ``quiet``); third-party libraries only when they
    warn, unless ``verbose``."""
    pkg = logging.getLogger(PACKAGE_LOGGER)
    pkg.setLevel(logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ConsoleFormatter())
    pkg.handlers[:] = [handler]
    pkg.propagate = False

    root = logging.getLogger()
    root.setLevel(logging.INFO if verbose else logging.WARNING)
    if not verbose:
        # Dask sets its own loggers to INFO when first imported, from its
        # config; tell it not to before that happens.
        try:
            import dask

            dask.config.set({"logging": {"distributed": "warning"}})
        except ImportError:
            pass
    if not root.handlers:
        fallback = logging.StreamHandler(sys.stderr)
        fallback.setFormatter(logging.Formatter("%(name)s %(levelname)s: %(message)s"))
        root.addHandler(fallback)


def add_file_log(path: str | Path) -> logging.Handler:
    """Also write pv-geom's records to ``path`` as JSON lines. Returns the
    handler so the caller can remove it when the run ends."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(JSONFormatter())
    handler.setLevel(logging.DEBUG)
    logging.getLogger(PACKAGE_LOGGER).addHandler(handler)
    return handler
