"""Pure helpers shared by frozen-policy RFCL rollout collectors."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Sequence

import numpy as np


RFCL_TRAJECTORY_NAMESPACE = uuid.UUID("a50a9bf1-904f-4e41-9e72-c52ef4e26286")
RFCL_SNAPSHOT_SCHEMA = "rfcl_snapshot_dataset_v2"


def resolve_snapshot_identity(
    snapshot_root: str | Path,
    *,
    task_name: str | None = None,
    task_config: str | None = None,
) -> tuple[str, str]:
    manifest_path = Path(snapshot_root) / "rfcl_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != RFCL_SNAPSHOT_SCHEMA:
        raise ValueError(
            f"Unsupported snapshot schema {manifest.get('schema')!r}; "
            "generate a fresh v2 dataset"
        )
    manifest_task = str(manifest["task"])
    manifest_config = str(manifest["task_config"])
    if task_name is not None and str(task_name) != manifest_task:
        raise ValueError(
            f"Task override {task_name!r} does not match snapshot task "
            f"{manifest_task!r}"
        )
    if task_config is not None and str(task_config) != manifest_config:
        raise ValueError(
            f"Task config override {task_config!r} does not match snapshot "
            f"config {manifest_config!r}"
        )
    return manifest_task, manifest_config


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trajectory_uuid(
    *,
    checkpoint_digest: str,
    adapter_id: str,
    worker_seed: int,
    attempt: int,
) -> str:
    identity = (
        f"{checkpoint_digest}:{adapter_id}:{int(worker_seed)}:{int(attempt)}"
    )
    return str(uuid.uuid5(RFCL_TRAJECTORY_NAMESPACE, identity))


def balanced_quotas(total: int, workers: int) -> list[int]:
    if int(total) <= 0:
        raise ValueError("total must be positive")
    if int(workers) <= 0:
        raise ValueError("workers must be positive")
    worker_count = min(int(workers), int(total))
    quotient, remainder = divmod(int(total), worker_count)
    return [
        quotient + int(worker_index < remainder)
        for worker_index in range(worker_count)
    ]


def sample_rollout_start(
    *,
    frontiers: Sequence[int],
    state_counts: Sequence[int],
    attempt: int,
    worker_seed: int,
    window: int,
    minimum_remaining_steps: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    frontiers = np.asarray(frontiers, dtype=np.int64)
    state_counts = np.asarray(state_counts, dtype=np.int64)
    if frontiers.ndim != 1 or state_counts.shape != frontiers.shape:
        raise ValueError("frontiers and state_counts must be matching vectors")
    if len(frontiers) == 0:
        raise ValueError("At least one demo is required")
    if np.any(state_counts <= 1):
        raise ValueError("Every demo must contain at least two states")
    if np.any(frontiers < 0) or np.any(frontiers >= state_counts):
        raise ValueError("frontiers are outside their demo state ranges")
    if int(window) < 0:
        raise ValueError("window cannot be negative")
    if int(minimum_remaining_steps) <= 0:
        raise ValueError("minimum_remaining_steps must be positive")

    demo_index = int((int(attempt) + int(worker_seed)) % len(frontiers))
    lower = int(frontiers[demo_index])
    latest_long_start = max(
        lower,
        int(state_counts[demo_index]) - 1 - int(minimum_remaining_steps),
    )
    upper = min(
        int(state_counts[demo_index]) - 1,
        lower + int(window),
        latest_long_start,
    )
    upper = max(lower, upper)
    state_index = int(rng.integers(lower, upper + 1))
    return demo_index, state_index
