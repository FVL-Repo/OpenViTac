from pathlib import Path

import numpy as np
import pytest

from scripts.rfcl.rerecord import (
    build_worker_command,
    consolidate_hdf5,
    load_trajectory_directory,
    partition_entries,
    resolve_worker_device,
    trajectory_replay_sort_key,
)
from scripts.rfcl.internal.rerecord_worker import load_trajectory


def test_rerecord_loader_accepts_only_frozen_rollout_v2(tmp_path):
    trajectory = tmp_path / "trajectory.npz"
    payload = {
        "schema": np.asarray("rfcl_privileged_trajectory_v2"),
        "trajectory_uuid": np.asarray("trajectory-id"),
        "actions": np.zeros((2, 7), dtype=np.float32),
        "rewards": np.asarray([0.0, 1.0], dtype=np.float32),
        "terminated": np.asarray([False, True]),
        "demo_index": np.asarray(1),
        "state_index": np.asarray(2),
        "adapter_id": np.asarray("insert_usb_v2"),
        "checkpoint_sha256": np.asarray("digest"),
        "action_scale": np.ones(7, dtype=np.float32),
        "action_mode": np.asarray("qpos_delta"),
    }
    np.savez_compressed(trajectory, **payload)
    loaded = load_trajectory(trajectory)
    assert loaded["trajectory_uuid"] == "trajectory-id"

    payload["schema"] = np.asarray("rfcl_training_trajectory_v1")
    np.savez_compressed(trajectory, **payload)
    with pytest.raises(ValueError, match="Unsupported trajectory schema"):
        load_trajectory(trajectory)


def test_partition_entries_balances_round_robin():
    entries = [Path(f"trajectory_{index}.npz") for index in range(7)]
    shards = partition_entries(entries, 3)
    assert [len(shard) for shard in shards] == [3, 2, 2]
    assert shards[0] == [entries[0], entries[3], entries[6]]


def test_load_trajectory_directory_is_stable_and_npz_only(tmp_path):
    (tmp_path / "b.npz").write_bytes(b"b")
    (tmp_path / "a.npz").write_bytes(b"a")
    (tmp_path / "ignored.txt").write_text("ignored", encoding="utf-8")

    entries = load_trajectory_directory(tmp_path)

    assert [entry.name for entry in entries] == ["a.npz", "b.npz"]


def test_partition_entries_does_not_create_empty_workers():
    entries = [Path("a.npz"), Path("b.npz")]
    shards = partition_entries(entries, 8)
    assert shards == [[entries[0]], [entries[1]]]


def test_partition_entries_can_group_replays_without_changing_membership(tmp_path):
    entries = []
    for index, (demo_index, state_index) in enumerate(((2, 3), (1, 8), (1, 4))):
        path = tmp_path / f"trajectory_{index}.npz"
        np.savez_compressed(
            path,
            demo_index=np.asarray(demo_index),
            state_index=np.asarray(state_index),
            trajectory_uuid=np.asarray(f"uuid-{index}"),
        )
        entries.append(path)

    shards = partition_entries(
        entries,
        1,
        shard_sort_key=trajectory_replay_sort_key,
    )

    assert shards == [[entries[2], entries[1], entries[0]]]


def test_build_worker_command_preserves_device_and_resume(tmp_path):
    command = build_worker_command(
        python_executable="python",
        worker_script=tmp_path / "worker.py",
        snapshot_root=tmp_path / "snapshots",
        selection_file=tmp_path / "selection.txt",
        output=tmp_path / "output",
        task_name="insert_USB",
        task_config="gelsight",
        step_limit=200,
        max_retries=3,
        device="cuda:1",
        resume=False,
        worker_args=("--livestream", "0"),
    )
    assert command[0] == "python"
    assert command[command.index("--device") + 1] == "cuda:1"
    assert "--headless" in command
    assert "--no-resume" in command
    assert command[-2:] == ["--livestream", "0"]


def test_rerecord_worker_device_is_process_local():
    assert resolve_worker_device("cuda:5") == ("cuda:0", "5")
    assert resolve_worker_device("cpu") == ("cpu", None)
    with pytest.raises(ValueError, match="Invalid CUDA worker device"):
        resolve_worker_device("cuda:not-an-index")


def test_consolidate_hdf5_creates_relative_links(tmp_path):
    first = tmp_path / "parallel_workers/worker_000/hdf5"
    second = tmp_path / "parallel_workers/worker_001/hdf5"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "episode_1.hdf5").write_bytes(b"one")
    (second / "episode_2.hdf5").write_bytes(b"two")

    count = consolidate_hdf5([first.parent, second.parent], tmp_path)

    assert count == 2
    assert (tmp_path / "hdf5/episode_1.hdf5").is_symlink()
    assert (tmp_path / "hdf5/episode_2.hdf5").read_bytes() == b"two"
    assert consolidate_hdf5([first.parent, second.parent], tmp_path) == 2
