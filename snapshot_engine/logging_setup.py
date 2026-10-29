"""
Central logging setup — stdlib ``logging``, structured, INFO by default.

Call :func:`setup_logging` once at program start (``run.py``). All modules then
use ``logging.getLogger(__name__)``. No secrets are ever logged.
"""

from __future__ import annotations

import logging
from typing import Optional

_CONFIGURED = False

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"


def setup_logging(level: str = "INFO", *, force: bool = False,
                  log_file: Optional[str] = None) -> None:
    """Configure the root logger once. ``level`` is a name like ``INFO``/``DEBUG``.

    ``log_file`` adds a size-rotated file next to the console (for background
    services that have no console, e.g. a background job).
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    import sys

    # pythonw / launchd without a terminal: no stderr to write to.
    handlers: list[logging.Handler] = [logging.StreamHandler()] if sys.stderr else []
    if log_file:
        from logging.handlers import RotatingFileHandler
        from pathlib import Path

        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3,
                                            encoding="utf-8"))
    logging.basicConfig(level=numeric, format=_FORMAT, datefmt=_DATEFMT, force=force,
                        handlers=handlers)
    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module logger (configuring logging with defaults if needed)."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name if name else "snapshot")
