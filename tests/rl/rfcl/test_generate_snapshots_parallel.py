from __future__ import annotations

from types import SimpleNamespace

from scripts.rfcl.generate_snapshots import _snapshot_jobs
from scripts.rfcl.generate_snapshots_parallel import (
    build_worker_command,
    partition_profile_indices,
)


def test_profile_shards_keep_global_demo_indices() -> None:
    args = SimpleNamespace(
        demo_plan="insert_usb_balanced40",
        task_name="insert_USB",
        seeds=None,
        max_attempts_per_profile=2,
        profile_indices=[18, 24, 40],
        seed_base=40_000,
    )

    jobs = _snapshot_jobs(args)

    assert [job["demo_index"] for job in jobs] == [17, 23, 39]
    assert [job["demo_id"] for job in jobs] == [
        "usb40_18",
        "usb40_24",
        "usb40_40",
    ]
    assert jobs[0]["attempt_seeds"] == [41_800, 41_801]


def test_partition_profile_indices_balances_workers() -> None:
    assert partition_profile_indices(range(18, 24), 4) == [
        [18, 22],
        [19, 23],
        [20],
        [21],
    ]


def test_parallel_snapshot_worker_uses_process_local_gpu(tmp_path) -> None:
    command = build_worker_command(
        python_executable="python",
        generator=tmp_path / "generate.py",
        output=tmp_path / "shard",
        profile_indices=[18, 24],
        seed_base=40_000,
        max_attempts_per_profile=12,
        stride=1,
        action_mode="target_pos_vel_force",
        step_limit=800,
        task_config="gelsight",
    )

    assert command[command.index("--profile-indices") + 1 :][:2] == ["18", "24"]
    assert command[command.index("--device") + 1] == "cuda:0"
    assert "--headless" in command
