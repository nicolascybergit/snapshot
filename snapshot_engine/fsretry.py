"""
Reading files that a sync client is busy with: wait instead of failing.

While a cloud sync client syncs a file that was just changed, macOS can answer a read with
``OSError [Errno 11] Resource deadlock avoided`` (sometimes EAGAIN). The read works again a moment later. Everything that reads such a file goes
through ``read_text`` / ``retrying``: it tries again every ``STEP`` seconds for up to ``WAIT``
seconds and only then gives up with the original error.

The two values are fixed here on purpose: ``Config.load`` itself reads through this module, so
they cannot come from ``parameters.yaml``.
"""

from __future__ import annotations

import errno
import time
from pathlib import Path
from typing import Callable, TypeVar

WAIT = 20.0          # seconds a busy file is waited for in total
STEP = 0.5           # seconds between two tries
BUSY = {errno.EDEADLK, errno.EAGAIN}

T = TypeVar("T")


def busy(exc: BaseException) -> bool:
    """Whether an error only means "the file is busy right now"."""
    return isinstance(exc, OSError) and exc.errno in BUSY


def retrying(fn: Callable[..., T], *args, wait: float = WAIT, **kwargs) -> T:
    """``fn(*args, **kwargs)``, tried again while it fails with a busy file."""
    end = time.monotonic() + wait
    while True:
        try:
            return fn(*args, **kwargs)
        except OSError as exc:
            if not busy(exc) or time.monotonic() >= end:
                raise
            time.sleep(STEP)


def read_text(path: str | Path, encoding: str = "utf-8", wait: float = WAIT) -> str:
    return retrying(Path(path).read_text, encoding=encoding, wait=wait)

