from pathlib import Path

import numpy as np

from policy.RL.rfcl_distributed import (
    DistributedDemoScheduler,
    expand_worker_devices,
    export_actor_policy,
    load_worker_result,
    save_worker_result,
)
from policy.RL.rfcl_sac import RFCLSACTrainer
from scripts.rfcl.train import (
    build_worker_command,
    resolve_worker_device,
)


def test_expand_worker_devices_assigns_each_device_exactly():
    assert expand_worker_devices(["cuda:0", "cuda:1"], 2) == [
        "cuda:0",
        "cuda:0",
        "cuda:1",
        "cuda:1",
    ]


def test_distributed_scheduler_keeps_successful_demo_block():
    scheduler = DistributedDemoScheduler(3, 2, block_size=2)
    unavailable = np.zeros(3, dtype=bool)

    first_demo, new_block, needs_handoff = scheduler.select_demo(0, unavailable)
    second_demo, _, _ = scheduler.select_demo(1, unavailable)
    assert first_demo == 0
    assert second_demo == 1
    assert new_block
    assert needs_handoff

    scheduler.record_episode(0, first_demo, success=True)
    repeated_demo, new_block, needs_handoff = scheduler.select_demo(0, unavailable)
    assert repeated_demo == first_demo
    assert not new_block
    assert not needs_handoff

    scheduler.record_episode(0, repeated_demo, success=False)
    next_demo, new_block, needs_handoff = scheduler.select_demo(0, unavailable)
    assert next_demo == 2
    assert new_block
    assert needs_handoff


def test_distributed_scheduler_resume_invalidates_live_contact():
    scheduler = DistributedDemoScheduler(2, 1, block_size=3)
    demo_index, _, _ = scheduler.select_demo(0, [False, False])
    scheduler.record_episode(0, demo_index, success=True)

    restored = DistributedDemoScheduler(2, 1, block_size=3)
    restored.load_state_dict(scheduler.state_dict())
    _, new_block, needs_handoff = restored.select_demo(0, [False, False])
    assert not new_block
    assert needs_handoff


def test_worker_result_round_trip(tmp_path):
    transition = (
        np.arange(3, dtype=np.float32),
        np.arange(2, dtype=np.float32),
        1.0,
        np.arange(3, dtype=np.float32) + 1,
        True,
    )
    path = tmp_path / "result.npz"
    save_worker_result(
        path,
        metadata={"job_id": 5, "error": "", "state_dim": 3, "action_dim": 2},
        transitions=[transition],
        replay_eligible=[True],
    )
    result = load_worker_result(path)
    assert int(result["job_id"]) == 5
    assert result["states"].shape == (1, 3)
    assert result["actions"].shape == (1, 2)
    assert result["replay_eligible"].tolist() == [True]


def test_actor_export_contains_one_policy(tmp_path):
    trainer = RFCLSACTrainer(state_dim=3, action_dim=2, device="cpu")
    destination = tmp_path / "actor.pt"
    export_actor_policy(trainer, destination, version=7)

    import torch

    payload = torch.load(destination, map_location="cpu", weights_only=False)
    assert payload["schema"] == "rfcl_distributed_actor_v1"
    assert payload["version"] == 7
    assert "critic" not in payload


def test_worker_command_selects_gpu_without_cuda_visible_devices(tmp_path):
    command = build_worker_command(
        python_executable="python",
        worker_script=Path("worker.py"),
        run_config=tmp_path / "run.json",
        ipc_root=tmp_path / "ipc",
        worker_id=3,
        worker_seed=10003,
        device="cuda:2",
    )
    assert command[command.index("--device") + 1] == "cuda:2"
    assert "--headless" in command
    assert all("CUDA_VISIBLE_DEVICES" not in argument for argument in command)


def test_worker_device_maps_physical_gpu_to_process_local_zero():
    assert resolve_worker_device("cuda:5") == ("cuda:0", "5")
    assert resolve_worker_device("cpu") == ("cpu", None)
