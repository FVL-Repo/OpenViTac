"""Parent-side tactile runtime: SHM + subprocess control, no pyuipc import."""

from __future__ import annotations

import importlib.util
import json
import os
import selectors
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from multiprocessing import shared_memory
from typing import Any, Self, cast

from .config import TactileCfg
from .protocol import (
    FINGER_SHM_SIZE,
    STATE_SHM_SIZE,
    FingerSample,
    TactileState,
    header,
    read_state,
    write_fingers,
)


class TactileError(RuntimeError):
    """Base class for tactile runtime failures."""


class TactileUnavailable(TactileError):
    """The worker/package is not available or failed to start."""


class TactileTimeout(TactileError):
    """A control/state operation exceeded its deadline."""


class TactileSequenceGap(TactileError):
    """The state sequence did not advance as required."""


@dataclass(frozen=True)
class TactileHealth:
    alive: bool
    pid: int | None
    rc: int | None
    gpu: int | str | None
    last_seq: int
    last_physx_step: int
    last_error: str | None
    abort_count: int
    shm_names: tuple[str, str]


@dataclass
class _RuntimeState:
    fingerprint_seq: int = 0
    fingerprint_slot: int = 0
    last_state_seq: int = 0
    last_physx_step: int = 0
    abort_count: int = 0
    last_error: str | None = None


class TactileRuntime:
    """Own one worker process and its two SHM channels.

    The parent only imports this module + numpy; ``find_spec("uipc")`` is a
    cheap availability check before spawning.  The worker is the only process
    that may import pyuipc.
    """

    def __init__(
        self,
        cfg: TactileCfg,
        *,
        run_id: str,
        group_id: str = "0",
        check_uipc: bool = True,
        worker_module: str = "openvitac_tactile.worker",
    ) -> None:
        self.cfg = cfg
        self.run_id = self._safe(run_id)
        self.group_id = self._safe(group_id)
        self._check_uipc = check_uipc
        self._worker_module = worker_module
        self._proc: subprocess.Popen[str] | None = None
        self._state_shm: shared_memory.SharedMemory | None = None
        self._finger_shm: shared_memory.SharedMemory | None = None
        self._state_name = f"{cfg.channel_prefix}_{self.run_id}_g{self.group_id}_state"
        self._finger_name = (
            f"{cfg.channel_prefix}_{self.run_id}_g{self.group_id}_finger"
        )
        self._rs = _RuntimeState()
        self._closed = False

    @staticmethod
    def _safe(value: str) -> str:
        cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in value)
        return cleaned[:80] or "0"

    @property
    def shm_names(self) -> tuple[str, str]:
        return (self._state_name, self._finger_name)

    def _require_started(self) -> None:
        if self._proc is None or self._state_shm is None or self._finger_shm is None:
            raise TactileUnavailable("TactileRuntime.start() was not called")

    def start(self) -> None:
        if self._proc is not None:
            return
        if self._check_uipc and importlib.util.find_spec("uipc") is None:
            raise TactileUnavailable(
                "pyuipc is not installed; install the `uipc` extra or select a "
                "worker interpreter that has it"
            )
        self.cfg.validate()
        if not self.cfg.enabled:
            raise TactileUnavailable("TactileCfg.enabled is False")
        self._state_shm = shared_memory.SharedMemory(
            name=self._state_name, create=True, size=STATE_SHM_SIZE
        )
        try:
            self._finger_shm = shared_memory.SharedMemory(
                name=self._finger_name, create=True, size=FINGER_SHM_SIZE
            )
            state_buf = self._state_shm.buf
            finger_buf = self._finger_shm.buf
            assert state_buf is not None and finger_buf is not None
            for hdr in (header(state_buf), header(finger_buf)):
                hdr["seq"] = 0
                hdr["slot"] = 0
                hdr["done"] = 0
                hdr["frame"] = -1
                hdr["checksum"] = 0.0
                hdr["ts_ns"] = 0

            env = os.environ.copy()
            env.update(self.cfg.worker.env)
            env["PYTHONUNBUFFERED"] = "1"
            if self.cfg.worker.gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(self.cfg.worker.gpu)
            cmd = [
                self.cfg.worker.python,
                "-m",
                self._worker_module,
                "--state-shm",
                self._state_name,
                "--finger-shm",
                self._finger_name,
                "--workspace",
                self.cfg.worker.workspace,
            ]
            if self._check_uipc and self.cfg.enabled and bool(self.cfg.pads):
                cmd.append("--require-uipc")
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=None,
                    text=True,
                    bufsize=1,
                    env=env,
                )
            except OSError as exc:
                raise TactileUnavailable(
                    f"failed to start tactile worker: {exc}"
                ) from exc

            ready = self._recv(timeout_s=self.cfg.worker.startup_timeout_s)
            if ready.get("op") != "ready":
                raise TactileUnavailable(f"worker did not send READY: {ready!r}")
        except BaseException:
            # A failed start must not leave a worker process or named SHM that a
            # retry with the same run_id would collide with.
            self._kill()
            self._cleanup_shm()
            self._proc = None
            raise

    def _send(self, payload: Mapping[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise TactileUnavailable("worker stdin is unavailable")
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()

    def _recv(self, timeout_s: float) -> dict[str, Any]:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise TactileUnavailable("worker stdout is unavailable")
        selector = selectors.DefaultSelector()
        try:
            selector.register(proc.stdout, selectors.EVENT_READ)
            if not selector.select(timeout_s):
                raise TactileTimeout(f"worker control timeout after {timeout_s:.1f}s")
            line = proc.stdout.readline()
        finally:
            selector.close()
        if not line:
            rc = proc.poll()
            raise TactileUnavailable(f"worker exited before response (rc={rc})")
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TactileUnavailable(f"invalid worker response: {line!r}") from exc
        if not isinstance(parsed, dict):
            raise TactileUnavailable(f"worker response is not an object: {parsed!r}")
        return cast(dict[str, Any], parsed)

    def reset(self, request: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self._require_started()
        payload = dict(request or {})
        # The worker owns the scene but must not import the parent's config
        # module state; serialize the validated config once per reset.
        payload.setdefault("config", asdict(self.cfg))
        self._send({"op": "reset", "request": payload})
        msg = self._recv(self.cfg.worker.reset_timeout_s)
        if msg.get("op") != "reset_ack":
            raise TactileUnavailable(f"unexpected reset response: {msg!r}")
        ack_value = msg.get("ack")
        if not isinstance(ack_value, dict):
            raise TactileUnavailable(f"reset_ack missing ack object: {msg!r}")
        ack = cast(dict[str, Any], ack_value)
        if ack.get("status") != "ready":
            raise TactileUnavailable(f"reset failed: {ack!r}")
        self._rs.last_state_seq = 0
        self._rs.last_physx_step = 0
        return ack

    def exchange(
        self, sample: FingerSample, *, wait: bool = False
    ) -> TactileState | None:
        self._require_started()
        assert self._finger_shm is not None
        finger_buf = self._finger_shm.buf
        assert finger_buf is not None
        _, self._rs.fingerprint_slot = write_fingers(
            finger_buf,
            sample,
            write_slot=self._rs.fingerprint_slot,
        )
        self._rs.fingerprint_seq = sample.seq
        if not wait:
            return self.latest()
        # The prototype's verified pipeline lets Isaac run up to
        # ``max_lag_steps`` finger frames ahead of UIPC.  Waiting for that
        # trailing target (rather than the current step) keeps UIPC in the
        # loop without serialising every physics step.
        if self.cfg.sync.mode == "lockstep":
            target_step = sample.physx_step
        else:
            target_step = max(0, sample.physx_step - self.cfg.sync.max_lag_steps)
        if target_step <= 0:
            # Pipeline fill: UIPC is still consuming the initial lead frames.
            return self.latest()
        deadline = time.monotonic() + self.cfg.sync.state_timeout_ms / 1e3
        while time.monotonic() <= deadline:
            state = self.latest()
            if state is not None and state.physx_step >= target_step:
                return state
            time.sleep(0.0005)
        raise TactileTimeout(
            f"state did not reach physx_step {target_step} within "
            f"{self.cfg.sync.state_timeout_ms:.1f} ms"
        )

    def latest(self) -> TactileState | None:
        self._require_started()
        assert self._state_shm is not None
        state_buf = self._state_shm.buf
        assert state_buf is not None
        state = read_state(state_buf, self._rs.last_state_seq)
        if state is None:
            return None
        if state.seq < self._rs.last_state_seq:
            self._rs.last_error = (
                f"seq regression {state.seq} < {self._rs.last_state_seq}"
            )
            return None
        self._rs.last_state_seq = state.seq
        self._rs.last_physx_step = max(self._rs.last_physx_step, state.physx_step)
        return state

    @property
    def health(self) -> TactileHealth:
        proc = self._proc
        rc = proc.poll() if proc is not None else None
        return TactileHealth(
            alive=proc is not None and rc is None,
            pid=proc.pid if proc is not None else None,
            rc=rc,
            gpu=self.cfg.worker.gpu,
            last_seq=self._rs.last_state_seq,
            last_physx_step=self._rs.last_physx_step,
            last_error=self._rs.last_error,
            abort_count=self._rs.abort_count,
            shm_names=self.shm_names,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                self._send({"op": "stop"})
                self._recv(timeout_s=min(self.cfg.worker.stop_timeout_s, 5.0))
            except (TactileError, OSError):
                pass
        self._kill()
        self._cleanup_shm()
        self._proc = None

    def _kill(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=min(self.cfg.worker.stop_timeout_s, 5.0))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
        for stream in (proc.stdin, proc.stdout):
            if stream is not None:
                stream.close()
        self._rs.abort_count = int(proc.returncode not in (0, None))

    def _cleanup_shm(self) -> None:
        for shm in (self._state_shm, self._finger_shm):
            if shm is None:
                continue
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
        self._state_shm = None
        self._finger_shm = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def runtime_env_snapshot() -> dict[str, str]:
    """Small helper for tests/diagnostics without touching CUDA."""
    return {
        "python": sys.executable,
        "cwd": os.getcwd(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
