import h5py
import json
import numpy as np
import pytest
from policy.RL.rfcl_snapshot import (
    RFCLSnapshot,
    RFCLSnapshotDataset,
    archive_uipc_frame,
    write_snapshot_sidecar,
)

from policy.RL.rfcl import (
    PRIVILEGED_STATE_LAYOUT,
    DemoTrajectory,
    MixedReplayBuffer,
    RFCLTransition,
    ReverseCurriculum,
    RoundRobinDemoScheduler,
    build_pair_privileged_state,
    handoff_xy_error,
    load_insert_usb_demo,
    relative_pose,
)
from policy.RL.rfcl_sac import RFCLSACTrainer
from policy.RL.rfcl_task_adapter import InsertUSBTaskAdapter


def test_relative_pose_identity_and_translation():
    pose = np.array([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    reference = np.array([[0.5, 1.0, 1.5, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    result = relative_pose(pose, reference)
    np.testing.assert_allclose(result[0, :3], [0.5, 1.0, 1.5])
    np.testing.assert_allclose(result[0, 3:], [1.0, 0.0, 0.0, 0.0])


def test_build_pair_privileged_state_uses_true_joint_velocity():
    state = build_pair_privileged_state(
        joint=np.arange(9, dtype=np.float32),
        joint_velocity=np.arange(9, dtype=np.float32) + 10.0,
        ee_pose=np.asarray([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
        controlled_pose=np.asarray([1, 2, 3, 1, 0, 0, 0], dtype=np.float32),
        target_pose=np.asarray([0.5, 1, 1.5, 1, 0, 0, 0], dtype=np.float32),
    )
    assert state.shape == (PRIVILEGED_STATE_LAYOUT.dim,)
    np.testing.assert_allclose(
        state[PRIVILEGED_STATE_LAYOUT.joint_delta],
        np.arange(9, dtype=np.float32) + 10.0,
    )
    np.testing.assert_allclose(
        state[PRIVILEGED_STATE_LAYOUT.controlled_in_target][:3],
        [0.5, 1.0, 1.5],
    )


def test_handoff_xy_error_uses_gripper_center_offset():
    ee = np.asarray([0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0])
    usb = np.asarray([0.0, 0.0, 0.03, 1.0, 0.0, 0.0, 0.0])
    assert handoff_xy_error(ee, usb) == 0.0
    usb[0] = 0.011
    assert handoff_xy_error(ee, usb) == pytest.approx(0.011)


def test_load_demo_builds_privileged_transitions(tmp_path):
    path = tmp_path / "7.hdf5"
    count = 5
    joint = np.zeros((count, 9), dtype=np.float32)
    joint[:, 0] = np.arange(count, dtype=np.float32) * 0.1
    ee = np.tile([0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0], (count, 1)).astype(np.float32)
    usb = np.tile([0.1, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0], (count, 1)).astype(np.float32)
    slot = np.tile([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], (count, 1)).astype(np.float32)
    tags = np.asarray(
        ["move_usb_to_pre_insert", "move_usb_to_play_pre_insert", "insert_usb_into_slot", "insert_usb_into_slot", ""],
        dtype="S27",
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("atom/tag", data=tags)
        handle.create_dataset("embodiment/joint", data=joint)
        handle.create_dataset("embodiment/ee", data=ee)
        handle.create_dataset("actor/prism", data=usb)
        handle.create_dataset("actor/slot", data=slot)
        handle.create_group("phase").attrs["save_frequency"] = 1

    demo = load_insert_usb_demo(path, action_scale=np.full(7, 0.2, dtype=np.float32))
    assert demo.states.shape == (count, PRIVILEGED_STATE_LAYOUT.dim)
    assert demo.actions.shape == (count - 1, 7)
    assert demo.rewards[-1] == 1.0
    assert demo.terminated[-1]
    assert demo.tags[0] == "move_usb_to_pre_insert"


def _demo(length=5):
    states = np.zeros((length, 2), dtype=np.float32)
    actions = np.zeros((length - 1, 1), dtype=np.float32)
    return DemoTrajectory(
        demo_id="demo",
        path=__import__("pathlib").Path("demo.hdf5"),
        frame_indices=np.arange(length),
        states=states,
        actions=actions,
        rewards=np.zeros(length - 1, dtype=np.float32),
        terminated=np.zeros(length - 1, dtype=bool),
        tags=("x",) * length,
        velocity_source="test",
    )


def test_reverse_curriculum_matches_rfcl_frontier_rules():
    curriculum = ReverseCurriculum([_demo(5)], reverse_step_size=2, seed=0)
    assert curriculum.state()["frontiers"].tolist() == [4]

    # Easier geometric samples train the learner but do not count toward the
    # three-success frontier window.
    assert curriculum.record_result(0, 4, success=True) == 4
    assert curriculum.record_result(0, 3, success=True) == 4
    assert curriculum.record_result(0, 4, success=False) == 4
    assert curriculum.record_result(0, 4, success=True) == 4
    assert curriculum.record_result(0, 4, success=True) == 4
    assert curriculum.record_result(0, 4, success=True) == 2

    # Reaching state zero and solving it are separate curriculum events.
    for _ in range(3):
        assert curriculum.record_result(0, 2, success=True) in (2, 0)
    assert curriculum.state()["frontiers"].tolist() == [0]
    assert not curriculum.state()["solved"].item()
    for _ in range(3):
        curriculum.record_result(0, 0, success=True)
    assert curriculum.state()["solved"].item()


def test_reverse_curriculum_geometric_distribution_and_dynamic_horizon():
    curriculum = ReverseCurriculum([_demo(10)], reverse_step_size=2, seed=0)
    states, probabilities = curriculum.checkpoint_distribution(0)
    np.testing.assert_array_equal(states, [9, 9, 9, 9, 9])
    np.testing.assert_allclose(probabilities, [0.5, 0.25, 0.125, 0.0625, 0.0625])
    assert curriculum.episode_horizon(0, 9) == 16

    for _ in range(3):
        curriculum.record_result(0, 9, success=True)
    states, probabilities = curriculum.checkpoint_distribution(0)
    np.testing.assert_array_equal(states, [7, 8, 9, 9, 9])
    np.testing.assert_allclose(probabilities, [0.5, 0.25, 0.125, 0.0625, 0.0625])
    assert curriculum.episode_horizon(0, 7) == 18


def test_reverse_curriculum_reports_partial_progress_targets():
    curriculum = ReverseCurriculum(
        [_demo(9), _demo(5)],
        reverse_step_size=2,
        seed=0,
    )
    np.testing.assert_allclose(curriculum.progress(), [0.0, 0.0])

    for _ in range(3):
        curriculum.record_result(0, 8, success=True)
        curriculum.record_result(1, 4, success=True)
    np.testing.assert_allclose(curriculum.progress(), [0.25, 0.5])

    partial = curriculum.progress_status(
        target_progress=0.5,
        target_demo_fraction=0.5,
    )
    assert partial["complete"]
    assert partial["reached_demos"] == 1
    np.testing.assert_array_equal(partial["reached"], [False, True])

    all_demos = curriculum.progress_status(
        target_progress=0.5,
        target_demo_fraction=1.0,
    )
    assert not all_demos["complete"]


def test_reverse_curriculum_resume_restores_progress_and_rng():
    original = ReverseCurriculum([_demo(8), _demo(9)], reverse_step_size=2, seed=7)
    for _ in range(3):
        original.record_result(0, 7, success=True)
    original.sample_checkpoint(demo_id=1)
    state = original.state()

    restored = ReverseCurriculum([_demo(8), _demo(9)], reverse_step_size=2, seed=99)
    restored.load_state_dict(state)
    np.testing.assert_array_equal(restored.frontiers, original.frontiers)
    np.testing.assert_array_equal(restored.success_counts, original.success_counts)
    assert restored.sample_checkpoint() == original.sample_checkpoint()


def test_round_robin_scheduler_covers_and_resumes_every_demo():
    scheduler = RoundRobinDemoScheduler(3, block_size=2)
    selected = []
    for _ in range(3):
        demo_index, _ = scheduler.select_demo([False, False, False])
        selected.append(demo_index)
        scheduler.record_episode(demo_index)
    assert selected == [0, 0, 1]

    restored = RoundRobinDemoScheduler(3, block_size=2)
    restored.load_state_dict(scheduler.state_dict())
    for _ in range(3):
        demo_index, _ = restored.select_demo([False, False, False])
        selected.append(demo_index)
        restored.record_episode(demo_index)
    assert selected == [0, 0, 1, 1, 2, 2]
    np.testing.assert_array_equal(restored.visit_counts, [2, 2, 2])

    demo_index, new_block = restored.select_demo([False, True, False])
    assert (demo_index, new_block) == (0, True)


def test_mixed_replay_preserves_requested_sources():
    buffer = MixedReplayBuffer(capacity=10, seed=0)
    for source in ("demo", "online"):
        buffer.add(
            RFCLTransition(
                state=np.zeros(2),
                action=np.zeros(1),
                reward=0.0,
                next_state=np.ones(2),
                terminated=False,
                demo_id=None,
                timestep=0,
                source=source,
            )
        )
    batch = buffer.sample(10, demo_fraction=0.5)
    assert len(batch) == 10
    assert {transition.source for transition in batch} == {"demo", "online"}
    assert sum(item.source == "demo" for item in batch) == 5
    assert sum(item.source == "online" for item in batch) == 5


def test_mixed_replay_keeps_demos_and_restores_online_rng():
    replay = MixedReplayBuffer(capacity=2, seed=11)
    replay.add(
        RFCLTransition(
            state=np.zeros(2), action=np.zeros(1), reward=1.0,
            next_state=np.ones(2), terminated=True, demo_id="demo",
            timestep=0, source="demo",
        )
    )
    for index in range(3):
        replay.add(
            RFCLTransition(
                state=np.full(2, index), action=np.zeros(1), reward=0.0,
                next_state=np.full(2, index + 1), terminated=False, demo_id=None,
                timestep=index, source="online",
            )
        )
    assert replay.source_counts() == {"demo": 1, "online": 2}
    state = replay.state_dict()

    restored = MixedReplayBuffer(capacity=2, seed=99)
    restored.add(
        RFCLTransition(
            state=np.zeros(2), action=np.zeros(1), reward=1.0,
            next_state=np.ones(2), terminated=True, demo_id="demo",
            timestep=0, source="demo",
        )
    )
    restored.load_state_dict(state)
    assert restored.source_counts() == replay.source_counts()
    assert [item.timestep for item in restored.sample(8, demo_fraction=0.5)] == [
        item.timestep for item in replay.sample(8, demo_fraction=0.5)
    ]


def test_rfcl_sac_supports_q_subsampling_and_temperature_learning():
    replay = MixedReplayBuffer(capacity=64, seed=3)
    for source in ("demo", "online"):
        for index in range(16):
            replay.add(
                RFCLTransition(
                    state=np.zeros(4, dtype=np.float32),
                    action=np.zeros(2, dtype=np.float32),
                    reward=float(index == 15),
                    next_state=np.ones(4, dtype=np.float32),
                    terminated=index == 15,
                    demo_id=None,
                    timestep=index,
                    source=source,
                )
            )
    trainer = RFCLSACTrainer(
        state_dim=4,
        action_dim=2,
        device="cpu",
        num_qs=4,
        num_min_qs=2,
        auto_alpha=True,
        backup_entropy=False,
    )
    metrics = trainer.update(replay, batch_size=16, demo_fraction=0.5)
    assert trainer.num_qs == 4
    assert trainer.num_min_qs == 2
    assert trainer.backup_entropy is False
    assert trainer.last_batch_source_counts == {"demo": 8, "online": 8}
    assert np.isfinite(metrics.critic_loss)
    assert np.isfinite(metrics.actor_loss)
    assert metrics.alpha > 0.0


def test_rfcl_sac_checkpoint_restores_training_state(tmp_path):
    replay = MixedReplayBuffer(capacity=16, seed=4)
    for source in ("demo", "online"):
        for index in range(4):
            replay.add(
                RFCLTransition(
                    state=np.full(3, index, dtype=np.float32),
                    action=np.zeros(2, dtype=np.float32),
                    reward=float(index == 3),
                    next_state=np.full(3, index + 1, dtype=np.float32),
                    terminated=index == 3,
                    demo_id="d" if source == "demo" else None,
                    timestep=index,
                    source=source,
                )
            )
    trainer = RFCLSACTrainer(
        state_dim=3, action_dim=2, device="cpu", auto_alpha=True,
        initial_log_std=-2.5,
    )
    trainer.update(replay, batch_size=8, demo_fraction=0.5)
    path = tmp_path / "checkpoint.pt"
    trainer.save(path, extra={"runner_schema": "test"})

    restored = RFCLSACTrainer(
        state_dim=3, action_dim=2, device="cpu", auto_alpha=True,
        initial_log_std=-2.5,
    )
    payload = restored.load(path)
    assert payload["extra"] == {"runner_schema": "test"}
    assert restored.update_count == trainer.update_count
    np.testing.assert_allclose(
        restored.act(np.ones(3), deterministic=True),
        trainer.act(np.ones(3), deterministic=True),
    )
    assert not (tmp_path / ".checkpoint.pt.tmp").exists()


def test_snapshot_sidecar_round_trip_and_demo_conversion(tmp_path):
    snapshots = {}
    metadata = {}
    for index in range(3):
        joint = np.zeros(9, dtype=np.float32)
        joint[0] = index * 0.1
        snapshot = RFCLSnapshot(
            snapshot_id=f"demo_000000_state_{index:04d}",
            demo_id="0",
            state_index=index,
            uipc_frame=index + 10,
            sim_step=index * 2,
            atom_id=1,
            atom_tag="insert_usb_into_slot",
            success=index == 2,
            task_state={"step_count": index * 2, "phase_id": 1},
            robot_state={
                "joint_pos": joint,
                "joint_vel": np.zeros(9, dtype=np.float32),
                "joint_pos_target": joint.copy(),
                "joint_vel_target": np.zeros(9, dtype=np.float32),
            },
            actor_control_state={
                "prism": {
                    "next_status": "set",
                    "next_pts": np.zeros((2, 3), dtype=np.float32),
                    "next_mat": None,
                    "next_mask": None,
                }
            },
            tactile_attachment_state={},
            poses={
                "joint_pos": joint,
                "joint_vel": np.zeros(9, dtype=np.float32),
                "ee": np.asarray([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
                "controlled": np.asarray([1, 0, 0, 1, 0, 0, 0], dtype=np.float32),
                "target": np.asarray([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
                "slot": np.asarray([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
            },
            privileged_state=np.full(46, index, dtype=np.float32),
        )
        entry = write_snapshot_sidecar(tmp_path, snapshot)
        snapshots[entry["snapshot_id"]] = entry

    manifest = {
        "schema": "rfcl_snapshot_dataset_v2",
        "task": "insert_USB",
        "adapter": InsertUSBTaskAdapter().manifest_metadata(),
        "demos": [{
            "demo_id": "0",
            "snapshot_ids": list(snapshots),
            "policy_start_state_index": 0,
        }],
        "snapshots": snapshots,
    }
    (tmp_path / "rfcl_manifest.json").write_text(
        __import__("json").dumps(manifest), encoding="utf-8"
    )
    dataset = RFCLSnapshotDataset(tmp_path)
    trajectories = dataset.to_demo_trajectories(np.ones(7, dtype=np.float32))
    assert len(trajectories) == 1
    np.testing.assert_allclose(trajectories[0].states[:, 0], [0, 1, 2])
    np.testing.assert_allclose(trajectories[0].actions[:, 0], [0.1, 0.1])
    np.testing.assert_array_equal(trajectories[0].rewards, [0.0, 1.0])


def test_archive_uipc_frame_copies_binary_state_and_rewrites_engine_frame(tmp_path):
    dump_root = tmp_path / "scene" / "dump"
    engine_dir = dump_root / "common" / "sim_engine.cpp"
    data_dir = dump_root / "cuda" / "system.cu"
    engine_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    (engine_dir / "state.42.json").write_text(
        json.dumps({"backend": "cuda", "frame": 42}),
        encoding="utf-8",
    )
    (data_dir / "q.42").write_bytes(b"checkpoint")

    archive_uipc_frame(dump_root, 42, 1_000_007)

    archived_state = json.loads(
        (engine_dir / "state.1000007.json").read_text(encoding="utf-8")
    )
    assert archived_state == {"backend": "cuda", "frame": 1_000_007}
    assert (data_dir / "q.1000007").read_bytes() == b"checkpoint"
    assert (data_dir / "q.42").read_bytes() == b"checkpoint"


def test_snapshot_dataset_rejects_duplicate_uipc_frames(tmp_path):
    manifest = {
        "schema": "rfcl_snapshot_dataset_v2",
        "task": "insert_USB",
        "adapter": InsertUSBTaskAdapter().manifest_metadata(),
        "demos": [{
            "demo_id": "0",
            "snapshot_ids": ["a", "b"],
            "policy_start_state_index": 0,
        }],
        "snapshots": {
            "a": {"uipc_frame": 7},
            "b": {"uipc_frame": 7},
        },
    }
    (tmp_path / "rfcl_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="reuses UIPC frame 7"):
        RFCLSnapshotDataset(tmp_path)
