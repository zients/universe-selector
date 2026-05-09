from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from filelock import FileLock, Timeout

from universe_selector.errors import BusyError


@contextmanager
def batch_lock(path: str, timeout_seconds: float = 1.0) -> Iterator[None]:
    lock = FileLock(path)
    try:
        with lock.acquire(timeout=timeout_seconds):
            yield
    except Timeout as exc:
        raise BusyError("another batch process is already running") from exc
