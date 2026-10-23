"""Duplicate process output to the console and a log file."""
from __future__ import annotations

import contextlib
import os
import select
import sys
import threading
from collections.abc import Generator
from pathlib import Path
from typing import BinaryIO

_CHUNK_SIZE = 65536
_JOIN_TIMEOUT = 5.0


def _write_all(fd: int, data: bytes) -> None:
    """Write every byte of ``data`` to ``fd``; ``os.write`` may write partially."""
    while data:
        data = data[os.write(fd, data) :]


def _copy_output(
    out_read: int,
    out_console: int,
    err_read: int,
    err_console: int,
    log_fd: int,
    stop: threading.Event,
) -> None:
    """Pump both pipes to their console fd and the log until the pipes close."""
    sources = {out_read: out_console, err_read: err_console}
    try:
        while sources:
            try:
                ready, _, _ = select.select(list(sources), [], [], 0.1)
            except OSError:
                break
            if not ready:
                if stop.is_set():
                    break
                continue
            for source in ready:
                try:
                    data = os.read(source, _CHUNK_SIZE)
                except OSError:
                    data = b""
                if not data:
                    del sources[source]
                    os.close(source)
                    continue
                # A closed console must not stop the log from being written.
                with contextlib.suppress(OSError):
                    _write_all(sources[source], data)
                with contextlib.suppress(OSError):
                    _write_all(log_fd, data)
    finally:
        for source in sources:
            with contextlib.suppress(OSError):
                os.close(source)


@contextlib.contextmanager
def tee_output(log_path: Path) -> Generator[None, None, None]:
    """Capture fd 1 and 2 into ``log_path`` while still writing the console.

    Kit replaces ``sys.stdout`` after launch and its C++ log sink writes fd 1
    directly, so redirecting the Python streams misses nearly all output.
    Teeing at the descriptor level captures everything, and the original
    descriptors are restored on every exit path, leaving no global state.
    """
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()

    saved_out = os.dup(1)
    saved_err = os.dup(2)
    out_read = out_write = err_read = err_write = -1
    log: BinaryIO | None = None
    reader: threading.Thread | None = None
    stop = threading.Event()
    try:
        log = log_path.open("wb", buffering=0)
        out_read, out_write = os.pipe()
        err_read, err_write = os.pipe()
        os.dup2(out_write, 1)
        os.dup2(err_write, 2)
        thread = threading.Thread(
            target=_copy_output,
            args=(out_read, saved_out, err_read, saved_err, log.fileno(), stop),
            name="openvitac-tee",
            daemon=True,
        )
        thread.start()
        reader = thread
        yield
    finally:
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(Exception):
                stream.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        for fd in (out_write, err_write):
            if fd >= 0:
                with contextlib.suppress(OSError):
                    os.close(fd)
        if reader is not None:
            # Every write end is closed now, so the reader drains and sees EOF.
            stop.set()
            reader.join(timeout=_JOIN_TIMEOUT)
        else:
            for fd in (out_read, err_read):
                if fd >= 0:
                    with contextlib.suppress(OSError):
                        os.close(fd)
        if log is not None:
            log.close()
