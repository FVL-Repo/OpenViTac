"""Process-level coordination helpers for distributed RFCL training."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


def expand_worker_devices(
    devices: Sequence[str], workers_per_device: int
) -> list[str]:
    if not devices:
        raise ValueError("devices cannot be empty")
    if int(workers_per_device) <= 0:
        raise ValueError("workers_per_device must be positive")
    return [
        str(device)
        for device in devices
        for _ in range(int(workers_per_device))
    ]


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def export_actor_policy(trainer: Any, path: str | Path, *, version: int) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "rfcl_distributed_actor_v1",
        "version": int(version),
        "state_dim": int(trainer.state_dim),
        "action_dim": int(trainer.action_dim),
        "initial_log_std": float(trainer.initial_log_std),
        "actor": {
            name: value.detach().cpu()
            for name, value in trainer.actor.state_dict().items()
        },
        "state_mean": trainer.state_mean.detach().cpu(),
        "state_std": trainer.state_std.detach().cpu(),
    }
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def save_worker_result(
    path: str | Path,
    *,
    metadata: dict[str, Any],
    transitions: list[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]],
    replay_eligible: Sequence[bool],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if len(transitions) != len(replay_eligible):
        raise ValueError("replay_eligible must match the transition count")
    if transitions:
        states = np.stack([item[0] for item in transitions]).astype(np.float32)
        actions = np.stack([item[1] for item in transitions]).astype(np.float32)
        rewards = np.asarray([item[2] for item in transitions], dtype=np.float32)
        next_states = np.stack([item[3] for item in transitions]).astype(np.float32)
        terminated = np.asarray([item[4] for item in transitions], dtype=np.bool_)
    else:
        state_dim = int(metadata.get("state_dim", 0))
        action_dim = int(metadata.get("action_dim", 0))
        states = np.empty((0, state_dim), dtype=np.float32)
        actions = np.empty((0, action_dim), dtype=np.float32)
        rewards = np.empty((0,), dtype=np.float32)
        next_states = np.empty((0, state_dim), dtype=np.float32)
        terminated = np.empty((0,), dtype=np.bool_)
    serializable = {
        name: np.asarray(value)
        for name, value in metadata.items()
    }
    temporary = destination.with_name(f".{destination.name}.tmp.npz")
    np.savez_compressed(
        temporary,
        schema=np.asarray("rfcl_distributed_worker_result_v1"),
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        terminated=terminated,
        replay_eligible=np.asarray(replay_eligible, dtype=np.bool_),
        **serializable,
    )
    os.replace(temporary, destination)


def load_worker_result(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as payload:
        result = {name: payload[name].copy() for name in payload.files}
    schema = str(np.asarray(result.pop("schema")).item())
    if schema != "rfcl_distributed_worker_result_v1":
        raise ValueError(f"Unsupported distributed result schema {schema!r}")
    return result


@dataclass
class WorkerSchedule:
    demo_index: int | None = None
    episodes_in_block: int = 0
    contact_valid: bool = False


class DistributedDemoScheduler:
    """Round-robin demo blocks with one persistent state per actor worker."""

    def __init__(
        self,
        demo_count: int,
        worker_count: int,
        *,
        block_size: int = 1,
    ) -> None:
        if int(demo_count) <= 0:
            raise ValueError("demo_count must be positive")
        if int(worker_count) <= 0:
            raise ValueError("worker_count must be positive")
        if int(block_size) <= 0:
            raise ValueError("block_size must be positive")
        self.demo_count = int(demo_count)
        self.worker_count = int(worker_count)
        self.block_size = int(block_size)
        self.cursor = 0
        self.workers = [WorkerSchedule() for _ in range(self.worker_count)]
        self.visit_counts = np.zeros(self.demo_count, dtype=np.int64)

    def _worker(self, worker_id: int) -> WorkerSchedule:
        worker_id = int(worker_id)
        if not 0 <= worker_id < self.worker_count:
            raise IndexError(f"worker_id out of range: {worker_id}")
        return self.workers[worker_id]

    def select_demo(
        self, worker_id: int, unavailable: Sequence[bool]
    ) -> tuple[int, bool, bool]:
        unavailable_array = np.asarray(unavailable, dtype=bool)
        if unavailable_array.shape != (self.demo_count,):
            raise ValueError("unavailable must contain one item per demo")
        if bool(unavailable_array.all()):
            raise StopIteration("all demos are unavailable")
        worker = self._worker(worker_id)
        if (
            worker.demo_index is not None
            and worker.episodes_in_block < self.block_size
            and not unavailable_array[worker.demo_index]
        ):
            return worker.demo_index, False, not worker.contact_valid

        active_elsewhere = {
            state.demo_index
            for index, state in enumerate(self.workers)
            if index != int(worker_id)
            and state.demo_index is not None
            and state.episodes_in_block < self.block_size
        }
        candidates = [
            (self.cursor + offset) % self.demo_count
            for offset in range(self.demo_count)
            if not unavailable_array[(self.cursor + offset) % self.demo_count]
        ]
        distinct = [candidate for candidate in candidates if candidate not in active_elsewhere]
        demo_index = int((distinct or candidates)[0])
        self.cursor = (demo_index + 1) % self.demo_count
        worker.demo_index = demo_index
        worker.episodes_in_block = 0
        worker.contact_valid = False
        return demo_index, True, True

    def record_episode(
        self, worker_id: int, demo_index: int, *, success: bool
    ) -> None:
        worker = self._worker(worker_id)
        demo_index = int(demo_index)
        if worker.demo_index != demo_index:
            raise ValueError(
                f"Worker {worker_id} cannot record demo {demo_index}; "
                f"active demo is {worker.demo_index}"
            )
        worker.episodes_in_block += 1
        worker.contact_valid = bool(success)
        self.visit_counts[demo_index] += 1

    def invalidate_worker(self, worker_id: int) -> None:
        self._worker(worker_id).contact_valid = False

    def state_dict(self) -> dict[str, Any]:
        return {
            "demo_count": self.demo_count,
            "worker_count": self.worker_count,
            "block_size": self.block_size,
            "cursor": self.cursor,
            "workers": [
                {
                    "demo_index": state.demo_index,
                    "episodes_in_block": state.episodes_in_block,
                    "contact_valid": False,
                }
                for state in self.workers
            ],
            "visit_counts": self.visit_counts.copy(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "demo_count": self.demo_count,
            "worker_count": self.worker_count,
            "block_size": self.block_size,
        }
        for name, value in expected.items():
            if int(state[name]) != int(value):
                raise ValueError(
                    f"Scheduler {name} mismatch: checkpoint={state[name]!r}, "
                    f"current={value!r}"
                )
        cursor = int(state["cursor"])
        if not 0 <= cursor < self.demo_count:
            raise ValueError("Invalid scheduler cursor")
        worker_states = list(state["workers"])
        if len(worker_states) != self.worker_count:
            raise ValueError("Invalid scheduler worker count")
        restored = []
        for worker_state in worker_states:
            demo_index = worker_state.get("demo_index")
            demo_index = None if demo_index is None else int(demo_index)
            if demo_index is not None and not 0 <= demo_index < self.demo_count:
                raise ValueError("Invalid scheduler demo index")
            episodes_in_block = int(worker_state["episodes_in_block"])
            if not 0 <= episodes_in_block <= self.block_size:
                raise ValueError("Invalid scheduler block progress")
            restored.append(
                WorkerSchedule(
                    demo_index=demo_index,
                    episodes_in_block=episodes_in_block,
                    contact_valid=False,
                )
            )
        visit_counts = np.asarray(state["visit_counts"], dtype=np.int64)
        if visit_counts.shape != (self.demo_count,) or np.any(visit_counts < 0):
            raise ValueError("Invalid scheduler visit counts")
        self.cursor = cursor
        self.workers = restored
        self.visit_counts[...] = visit_counts
