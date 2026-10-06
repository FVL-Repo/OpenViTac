import json
from pathlib import Path

import numpy as np
import pytest

from policy.RL.rfcl_collection import (
    balanced_quotas,
    resolve_snapshot_identity,
    sample_rollout_start,
    trajectory_uuid,
)


def test_snapshot_identity_requires_v2_and_rejects_overrides(tmp_path):
    manifest = {
        "schema": "rfcl_snapshot_dataset_v2",
        "task": "insert_USB",
        "task_config": "gelsight",
    }
    (tmp_path / "rfcl_manifest.json").write_text(json.dumps(manifest))
    assert resolve_snapshot_identity(tmp_path) == ("insert_USB", "gelsight")
    with pytest.raises(ValueError, match="Task config override"):
        resolve_snapshot_identity(tmp_path, task_config="xense")

    manifest["schema"] = "rfcl_snapshot_dataset_v1"
    (tmp_path / "rfcl_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fresh v2"):
        resolve_snapshot_identity(tmp_path)
from scripts.rfcl.collect_rollouts import (
    build_worker_command,
    consolidate_trajectories,
    resolve_worker_device,
)


def test_balanced_quotas_are_exact_and_nonempty():
    assert balanced_quotas(10, 3) == [4, 3, 3]
    assert balanced_quotas(2, 4) == [1, 1]


def test_trajectory_uuid_is_stable_and_worker_specific():
    first = trajectory_uuid(
        checkpoint_digest="abc",
        adapter_id="insert_usb_v2",
        worker_seed=10,
        attempt=3,
    )
    assert first == trajectory_uuid(
        checkpoint_digest="abc",
        adapter_id="insert_usb_v2",
        worker_seed=10,
        attempt=3,
    )
    assert first != trajectory_uuid(
        checkpoint_digest="abc",
        adapter_id="insert_usb_v2",
        worker_seed=11,
        attempt=3,
    )


def test_sample_rollout_start_balances_demos_and_stays_near_frontier():
    rng = np.random.default_rng(5)
    samples = [
        sample_rollout_start(
            frontiers=[10, 20],
            state_counts=[50, 60],
            attempt=attempt,
            worker_seed=0,
            window=4,
            minimum_remaining_steps=20,
            rng=rng,
        )
        for attempt in range(4)
    ]
    assert [demo for demo, _ in samples] == [0, 1, 0, 1]
    assert all((10 <= state <= 14) if demo == 0 else (20 <= state <= 24) for demo, state in samples)


def test_parallel_rollout_command_and_consolidation(tmp_path):
    command = build_worker_command(
        python_executable="python",
        worker_script=tmp_path / "worker.py",
        checkpoint=tmp_path / "best.pt",
        snapshot_root=tmp_path / "snapshots",
        output=tmp_path / "worker",
        successes=5,
        max_attempts=50,
        minimum_steps=20,
        frontier_window=8,
        deterministic=True,
        exploration_noise=0.05,
        worker_id=1,
        worker_seed=10001,
        device="cuda:1",
        resume=True,
    )
    assert command[command.index("--device") + 1] == "cuda:1"
    assert "--headless" in command
    assert command[command.index("--exploration-noise") + 1] == "0.05"

    first = tmp_path / "workers/0/success_trajectories"
    second = tmp_path / "workers/1/success_trajectories"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "trajectory_a.npz").write_bytes(b"a")
    (second / "trajectory_b.npz").write_bytes(b"b")
    output = tmp_path / "merged"
    assert consolidate_trajectories([first.parent, second.parent], output) == 2
    assert (output / "success_trajectories/trajectory_b.npz").read_bytes() == b"b"


def test_parallel_rollout_worker_device_is_process_local():
    assert resolve_worker_device("cuda:5") == ("cuda:0", "5")
    assert resolve_worker_device("cpu") == ("cpu", None)
    with pytest.raises(ValueError, match="Invalid CUDA worker device"):
        resolve_worker_device("cuda:not-an-index")
