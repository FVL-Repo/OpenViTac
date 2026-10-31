"""Run-scoped logging for one collect episode.

The logger is a runtime handle, not task/spec data: it is created inside the
collect tee context, writes to fd 2 (so ``tee_output`` also writes
``collect.log``), and is removed on exit.  Task authors only receive adapters;
they never create or store loggers themselves.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

_LOG_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s "
    "[task=%(task)s seed=%(seed)s run=%(run_id)s] %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class RunLog:
    """The two adapters exposed for one run."""

    run: logging.LoggerAdapter[logging.Logger]
    task: logging.LoggerAdapter[logging.Logger]


class _ContextFilter(logging.Filter):
    """Fill run context on records from either logger in the run."""

    def __init__(self, *, task: str, seed: int, run_id: str) -> None:
        super().__init__()
        self._task = task
        self._seed = seed
        self._run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "task"):
            record.task = self._task
        if not hasattr(record, "seed"):
            record.seed = self._seed
        if not hasattr(record, "run_id"):
            record.run_id = self._run_id
        return True


class _FdStream:
    """Text sink over a private duplicate of fd 2.

    ``logging.StreamHandler`` needs only ``write``, ``flush`` and ``close``.
    Text accumulates in a linear buffer and is written straight to the
    descriptor, so output follows ``tee_output``'s descriptor-level contract
    instead of whatever object happens to back ``sys.stderr``.  ``close``
    releases the duplicate only; the process's fd 2 stays untouched.
    """

    encoding = "utf-8"
    errors = "backslashreplace"

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._pending = bytearray()

    def fileno(self) -> int:
        return self._fd

    def write(self, text: str) -> int:
        if self._fd < 0:
            raise ValueError("write to a closed run log stream")
        self._pending += text.encode(self.encoding, self.errors)
        return len(text)

    def flush(self) -> None:
        if self._fd < 0 or not self._pending:
            return
        data = bytes(self._pending)
        self._pending.clear()
        view = memoryview(data)
        try:
            while view:
                written = os.write(self._fd, view)
                if written <= 0:
                    break
                view = view[written:]
        except OSError:
            # A closed console or tee must not abort the run.
            pass

    def close(self) -> None:
        if self._fd < 0:
            return
        try:
            self.flush()
        finally:
            fd, self._fd = self._fd, -1
            os.close(fd)


class _Fd2Handler(logging.StreamHandler[Any]):
    """Stream handler that owns the fd 2 duplicate used by its stream."""

    def __init__(self) -> None:
        super().__init__(_FdStream(os.dup(2)))

    def close(self) -> None:
        stream = self.stream
        self.stream = None
        try:
            if stream is not None:
                stream.close()
        finally:
            super().close()


@contextmanager
def run_logger(
    *,
    task: str,
    seed: int,
    run_id: str,
    level: int = logging.INFO,
) -> Generator[RunLog, None, None]:
    """Bind ``openvitac.run`` and ``openvitac.task.<task>`` to one tee stream.

    ``level`` defaults to INFO; callers may opt into DEBUG.  The handler is
    attached to both named loggers with propagation disabled, then removed in
    ``finally`` so repeated collects never accumulate handlers or duplicate
    lines.
    """
    handler = _Fd2Handler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    handler.addFilter(_ContextFilter(task=task, seed=seed, run_id=run_id))

    run = logging.getLogger("openvitac.run")
    task_logger = logging.getLogger(f"openvitac.task.{task}")
    loggers = (run, task_logger)
    previous = [(logger, logger.level, logger.propagate) for logger in loggers]
    extra = {"task": task, "seed": seed, "run_id": run_id}

    for logger in loggers:
        logger.setLevel(level)
        logger.propagate = False
        logger.addHandler(handler)
    try:
        yield RunLog(
            run=logging.LoggerAdapter(run, extra),
            task=logging.LoggerAdapter(task_logger, extra),
        )
    finally:
        for logger, old_level, old_propagate in previous:
            logger.removeHandler(handler)
            logger.setLevel(old_level)
            logger.propagate = old_propagate
        handler.close()
