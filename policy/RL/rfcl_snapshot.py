"""Persistent simulator snapshots for privileged-state RFCL training.

UIPC's ``world.dump()`` stores deformable and affine-body simulation state,
but several controller and attachment values live only in Python.  This module
stores those values in a portable NumPy sidecar and restores both halves of a
checkpoint before an RFCL episode starts.
"""

from __future__ import annotations

import copy
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from policy.RL.action import (
    TARGET_QPOS_VELOCITY_ACTION_DIM,
    TARGET_QPOS_VELOCITY_FORCE_ACTION_DIM,
    target_qpos_to_action,
    target_qpos_velocity_force_to_action,
    target_qpos_velocity_to_action,
)
from policy.RL.rfcl import DemoTrajectory
from policy.RL.rfcl_task_adapter import (
    RFCLTaskAdapter,
    create_rfcl_task_adapter,
)


SNAPSHOT_DATASET_SCHEMA = "rfcl_snapshot_dataset_v2"
SNAPSHOT_MANIFEST_NAME = "rfcl_manifest.json"


def archive_uipc_frame(
    dump_root: str | Path,
    source_frame: int,
    target_frame: int,
) -> None:
    """Copy one UIPC dump to a collision-free logical frame number.

    UIPC names dump files only by ``world.frame()``.  Task resets can reuse
    those numbers, so recording several demos into one workspace otherwise
    overwrites earlier checkpoints.  The JSON engine state also embeds the
    frame and must be rewritten to match the copied filenames.
    """

    dump_root = Path(dump_root)
    source_frame = int(source_frame)
    target_frame = int(target_frame)
    if source_frame < 0 or target_frame < 0:
        raise ValueError("UIPC frame numbers must be non-negative")
    if source_frame == target_frame:
        return

    source_suffix = f".{source_frame}"
    source_json_suffix = f".{source_frame}.json"
    matches: list[tuple[Path, Path, bool]] = []
    for source in dump_root.rglob("*"):
        if not source.is_file():
            continue
        if source.name.endswith(source_json_suffix):
            stem = source.name[: -len(source_json_suffix)]
            target = source.with_name(f"{stem}.{target_frame}.json")
            matches.append((source, target, True))
        elif source.name.endswith(source_suffix):
            stem = source.name[: -len(source_suffix)]
            target = source.with_name(f"{stem}.{target_frame}")
            matches.append((source, target, False))

    if not matches:
        raise FileNotFoundError(
            f"No UIPC dump files for frame {source_frame} under {dump_root}"
        )

    engine_states = 0
    for source, target, is_json in matches:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not is_json:
            shutil.copy2(source, target)
            continue
        payload = json.loads(source.read_text(encoding="utf-8"))
        if "frame" not in payload:
            raise ValueError(f"UIPC state file has no frame field: {source}")
        if int(payload["frame"]) != source_frame:
            raise ValueError(
                f"UIPC state frame mismatch in {source}: "
                f"expected {source_frame}, got {payload['frame']}"
            )
        payload["frame"] = target_frame
        target.write_text(
            json.dumps(payload, indent=4) + "\n",
            encoding="utf-8",
        )
        engine_states += 1

    if engine_states != 1:
        raise RuntimeError(
            f"Expected one UIPC engine state for frame {source_frame}, "
            f"found {engine_states}"
        )


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def _optional_robot_value(robot: Any, name: str, fallback: np.ndarray) -> np.ndarray:
    value = getattr(robot.data, name, None)
    return np.asarray(fallback).copy() if value is None else _to_numpy(value)


def read_robot_physics_state(task: Any) -> dict[str, np.ndarray]:
    """Read qpos, qvel, and EE pose from PhysX instead of stale Lab caches."""

    manager = task._robot_manager
    robot = manager.robot
    robot._physics_sim_view.update_articulations_kinematic()
    # RobotManager force-writes Motion Plan qpos through ``root_physx_view``.
    # That bypasses Articulation's normal cache invalidation, while a task step
    # with dt=0 can leave every TimestampedBuffer at the current timestamp.
    # Invalidating the three buffers makes the public data properties fetch the
    # actual PhysX values and preserves RobotManager's existing frame transform.
    for buffer_name in ("_joint_pos", "_joint_vel", "_body_link_pose_w"):
        buffer = getattr(robot.data, buffer_name, None)
        if buffer is not None:
            buffer.timestamp = -1.0
    return {
        "joint_pos": _to_numpy(robot.data.joint_pos).reshape(-1),
        "joint_vel": _to_numpy(robot.data.joint_vel).reshape(-1),
        "ee": np.asarray(manager.get_ee_pose().tolist(), dtype=np.float64),
    }


def _pose_snapshot(
    task: Any,
    adapter: RFCLTaskAdapter,
) -> dict[str, np.ndarray]:
    physics_state = read_robot_physics_state(task)
    return {
        **physics_state,
        **adapter.capture_entities(task),
    }


@dataclass
class RFCLSnapshot:
    snapshot_id: str
    demo_id: str
    state_index: int
    uipc_frame: int
    sim_step: int
    atom_id: int
    atom_tag: str
    success: bool
    task_state: dict[str, Any]
    robot_state: dict[str, np.ndarray]
    actor_control_state: dict[str, dict[str, Any]]
    tactile_attachment_state: dict[str, dict[str, Any]]
    poses: dict[str, np.ndarray]
    privileged_state: np.ndarray
    force_position_write: bool = True


def capture_snapshot(
    task: Any,
    adapter: RFCLTaskAdapter,
    *,
    snapshot_id: str,
    demo_id: str,
    state_index: int,
    success: bool,
    force_position_write: bool = True,
) -> RFCLSnapshot:
    """Dump the current UIPC frame and capture all Python-side state."""

    # ``ArticulationData`` is lazily refreshed.  Motion Plan writes the arm
    # directly through the PhysX view, so reading ``robot.data.joint_pos``
    # before this refresh can return one atom's qpos for dozens of simulator
    # steps.  Snapshot actions are derived from consecutive joint positions;
    # keep those values coherent with the EE pose captured in the same frame.
    poses = _pose_snapshot(task, adapter)
    world = task.uipc_sim.world
    if not bool(world.dump()):
        raise RuntimeError(f"UIPC world.dump() failed for {snapshot_id!r}")

    robot = task._robot_manager.robot
    joint_pos = poses["joint_pos"]
    joint_vel = poses["joint_vel"]
    actor_control_state: dict[str, dict[str, Any]] = {}
    for name, actor in task._actor_manager.actors.items():
        actor_control_state[name] = {
            "next_status": actor.next_status,
            "next_pts": (
                None if actor.next_pts is None else _to_numpy(actor.next_pts)
            ),
            "next_mat": (
                None if actor.next_mat is None else _to_numpy(actor.next_mat)
            ),
            "next_mask": (
                None if actor.next_mask is None else _to_numpy(actor.next_mask)
            ),
        }

    tactile_attachment_state: dict[str, dict[str, Any]] = {}
    for name, tactile in task._tactile_manager.tactiles.items():
        attachment = tactile.attachment
        tactile_attachment_state[name] = {
            "aim_positions": _to_numpy(attachment.aim_positions),
            "attachment_offsets": _to_numpy(attachment.attachment_offsets),
            "obj_pose": (
                None
                if getattr(attachment, "obj_pose", None) is None
                else _to_numpy(attachment.obj_pose)
            ),
        }

    privileged_state = adapter.build_privileged_state(
        physics_state={
            "joint_pos": poses["joint_pos"],
            "joint_vel": poses["joint_vel"],
            "ee": poses["ee"],
        },
        entities=poses,
    )
    task_state = {
        "step_count": int(task.step_count),
        "take_action_cnt": int(task.take_action_cnt),
        "policy_step_count": int(task.policy_step_count),
        "phase_id": int(task.phase_id),
        "atom_id": int(task.atom_id),
        "atom_tag": str(task.atom_tag),
        "eval_success": bool(task.eval_success),
        "plan_success": bool(task.plan_success),
        "terminal_reason": task.terminal_reason,
        "last_render": int(task.last_render),
    }
    return RFCLSnapshot(
        snapshot_id=str(snapshot_id),
        demo_id=str(demo_id),
        state_index=int(state_index),
        uipc_frame=int(world.frame()),
        sim_step=int(task.step_count),
        atom_id=int(task.atom_id),
        atom_tag=str(task.atom_tag),
        success=bool(success),
        force_position_write=bool(force_position_write),
        task_state=task_state,
        robot_state={
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "joint_pos_target": _optional_robot_value(
                robot, "joint_pos_target", joint_pos
            ),
            "joint_vel_target": _optional_robot_value(
                robot, "joint_vel_target", np.zeros_like(joint_vel)
            ),
        },
        actor_control_state=actor_control_state,
        tactile_attachment_state=tactile_attachment_state,
        poses=poses,
        privileged_state=privileged_state,
    )


def write_snapshot_sidecar(
    root: str | Path,
    snapshot: RFCLSnapshot,
) -> dict[str, Any]:
    """Write one snapshot's Python state and return its JSON manifest entry."""

    root = Path(root)
    state_dir = root / "python_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    relative_path = Path("python_state") / f"{snapshot.snapshot_id}.npz"
    arrays: dict[str, np.ndarray] = {
        "privileged_state": snapshot.privileged_state,
    }
    metadata: dict[str, Any] = {
        "snapshot_id": snapshot.snapshot_id,
        "demo_id": snapshot.demo_id,
        "state_index": snapshot.state_index,
        "uipc_frame": snapshot.uipc_frame,
        "sim_step": snapshot.sim_step,
        "atom_id": snapshot.atom_id,
        "atom_tag": snapshot.atom_tag,
        "success": snapshot.success,
        "force_position_write": snapshot.force_position_write,
        "task_state": snapshot.task_state,
        "state_file": str(relative_path),
        "robot_arrays": {},
        "pose_arrays": {},
        "actors": [],
        "tactiles": [],
    }

    for index, (name, value) in enumerate(snapshot.robot_state.items()):
        key = f"robot_{index}"
        arrays[key] = np.asarray(value)
        metadata["robot_arrays"][name] = key
    for index, (name, value) in enumerate(snapshot.poses.items()):
        key = f"pose_{index}"
        arrays[key] = np.asarray(value)
        metadata["pose_arrays"][name] = key
    for actor_index, (name, values) in enumerate(
        snapshot.actor_control_state.items()
    ):
        entry = {
            "name": name,
            "next_status": values["next_status"],
            "arrays": {},
        }
        for value_index, field in enumerate(("next_pts", "next_mat", "next_mask")):
            if values[field] is None:
                continue
            key = f"actor_{actor_index}_{value_index}"
            arrays[key] = np.asarray(values[field])
            entry["arrays"][field] = key
        metadata["actors"].append(entry)
    for tactile_index, (name, values) in enumerate(
        snapshot.tactile_attachment_state.items()
    ):
        entry = {"name": name, "arrays": {}}
        for value_index, field in enumerate(
            ("aim_positions", "attachment_offsets", "obj_pose")
        ):
            if values[field] is None:
                continue
            key = f"tactile_{tactile_index}_{value_index}"
            arrays[key] = np.asarray(values[field])
            entry["arrays"][field] = key
        metadata["tactiles"].append(entry)

    np.savez_compressed(root / relative_path, **arrays)
    return metadata


def read_snapshot_sidecar(
    root: str | Path,
    metadata: Mapping[str, Any],
) -> RFCLSnapshot:
    root = Path(root)
    with np.load(root / metadata["state_file"], allow_pickle=False) as archive:
        robot_state = {
            name: archive[key].copy()
            for name, key in metadata["robot_arrays"].items()
        }
        poses = {
            name: archive[key].copy()
            for name, key in metadata["pose_arrays"].items()
        }
        actor_control_state = {}
        for entry in metadata["actors"]:
            actor_control_state[entry["name"]] = {
                "next_status": entry["next_status"],
                **{
                    field: (
                        archive[entry["arrays"][field]].copy()
                        if field in entry["arrays"]
                        else None
                    )
                    for field in ("next_pts", "next_mat", "next_mask")
                },
            }
        tactile_attachment_state = {}
        for entry in metadata["tactiles"]:
            tactile_attachment_state[entry["name"]] = {
                field: (
                    archive[entry["arrays"][field]].copy()
                    if field in entry["arrays"]
                    else None
                )
                for field in ("aim_positions", "attachment_offsets", "obj_pose")
            }
        privileged_state = archive["privileged_state"].copy()

    return RFCLSnapshot(
        snapshot_id=str(metadata["snapshot_id"]),
        demo_id=str(metadata["demo_id"]),
        state_index=int(metadata["state_index"]),
        uipc_frame=int(metadata["uipc_frame"]),
        sim_step=int(metadata["sim_step"]),
        atom_id=int(metadata["atom_id"]),
        atom_tag=str(metadata["atom_tag"]),
        success=bool(metadata["success"]),
        force_position_write=bool(metadata.get("force_position_write", True)),
        task_state=dict(metadata["task_state"]),
        robot_state=robot_state,
        actor_control_state=actor_control_state,
        tactile_attachment_state=tactile_attachment_state,
        poses=poses,
        privileged_state=privileged_state,
    )


def _write_robot_state(task: Any, state: Mapping[str, np.ndarray]) -> None:
    robot = task._robot_manager.robot
    device = robot.device

    def tensor(name: str) -> torch.Tensor:
        return torch.as_tensor(state[name], dtype=torch.float32, device=device)

    joint_pos = tensor("joint_pos")
    joint_vel = tensor("joint_vel")
    robot.set_joint_position_target(tensor("joint_pos_target"))
    robot.set_joint_velocity_target(tensor("joint_vel_target"))
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot._physics_sim_view.update_articulations_kinematic()


def restore_snapshot(
    task: Any,
    adapter: RFCLTaskAdapter,
    snapshot: RFCLSnapshot,
) -> None:
    """Restore UIPC, articulation, actor-controller, and gelpad state."""

    world = task.uipc_sim.world
    if not bool(world.recover(snapshot.uipc_frame)):
        raise RuntimeError(
            f"UIPC world.recover({snapshot.uipc_frame}) failed for "
            f"{snapshot.snapshot_id!r}"
        )
    world.retrieve()

    # Affine actor poses are already part of the UIPC frame.  Do not call
    # ``write_vertex_positions_to_sim`` here: the current affine-body backend
    # interprets that API as a body-state reset and discards the requested
    # transform.
    world.retrieve()

    for name, values in snapshot.actor_control_state.items():
        actor = task._actor_manager.actors[name]
        actor.next_status = values["next_status"]
        actor.next_pts = copy.deepcopy(values["next_pts"])
        actor.next_mat = copy.deepcopy(values["next_mat"])
        actor.next_mask = copy.deepcopy(values["next_mask"])
    _write_robot_state(task, snapshot.robot_state)
    for name, values in snapshot.tactile_attachment_state.items():
        attachment = task._tactile_manager.tactiles[name].attachment
        attachment.aim_positions = np.asarray(values["aim_positions"]).copy()
        attachment.attachment_offsets = np.asarray(
            values["attachment_offsets"]
        ).copy()
        if values["obj_pose"] is not None:
            attachment.obj_pose = torch.as_tensor(
                values["obj_pose"],
                dtype=torch.float32,
                device=attachment.device,
            )
        # The attachment callback normally refreshes this one frame after the
        # robot moves.  Refresh it immediately after restoring robot qpos so
        # the first RFCL no-op step cannot pull the USB toward the reset pose.
        refresh_targets = getattr(attachment, "_compute_aim_positions", None)
        if callable(refresh_targets):
            refresh_targets()

    for name, value in snapshot.task_state.items():
        setattr(task, name, value)
    task.render_outdated = True
    task.uipc_sim._contact_grad_cache = None
    task.scene.update(dt=0.0)
    task._actor_manager.update(dt=0.0)
    task.uipc_sim.update_render_meshes()
    task._actor_manager.sync_visuals()
    adapter.after_restore(task)


def prepare_snapshot_for_policy(
    task: Any,
    adapter: RFCLTaskAdapter,
    snapshot: RFCLSnapshot,
) -> None:
    """Restore a snapshot and reset episode-local counters for RFCL control."""

    restore_snapshot(task, adapter, snapshot)
    task.take_action_cnt = 0
    task.policy_step_count = 0
    task.eval_success = False
    task.plan_success = True
    task.terminal_reason = None
    task.last_render = int(task.step_count)
    task.metadata = {
        "rfcl_demo_id": snapshot.demo_id,
        "rfcl_state_index": snapshot.state_index,
        "rfcl_snapshot_id": snapshot.snapshot_id,
    }
    if hasattr(task, "_set_phase") and hasattr(task, "PHASE_POLICY"):
        task._set_phase(task.PHASE_POLICY)


def runtime_snapshot_diagnostics(
    task: Any,
    adapter: RFCLTaskAdapter,
    snapshot: RFCLSnapshot | None = None,
) -> dict[str, Any]:
    """Collect the simulator-side quantities relevant to checkpoint fidelity.

    This intentionally stays independent of the learner.  A restored frame can
    have matching qpos while still carrying stale actor constraints or tactile
    targets, so the diagnostics include both relative poses and controller
    state.  Values are JSON-serializable for long-running probes.
    """

    manager = task._robot_manager
    robot = manager.robot
    physics_state = read_robot_physics_state(task)
    joint_pos = physics_state["joint_pos"]
    joint_vel = physics_state["joint_vel"]
    ee_pose = physics_state["ee"]
    gripper_pose = np.asarray(
        manager.get_gripper_center_pose().tolist(), dtype=np.float64
    )
    entities = adapter.capture_entities(task)

    def pose_list(value: Any) -> list[float]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        return np.asarray(value, dtype=np.float64).reshape(-1).tolist()

    def pose_error(actual: np.ndarray, expected: Any) -> dict[str, float]:
        expected_array = np.asarray(expected, dtype=np.float64).reshape(-1)
        return {
            "position_l2_m": float(np.linalg.norm(actual[:3] - expected_array[:3])),
            "quaternion_l2": float(np.linalg.norm(actual[3:7] - expected_array[3:7])),
        }

    diagnostics: dict[str, Any] = {
        "uipc_frame": int(task.uipc_sim.world.frame()),
        "sim_step": int(task.step_count),
        "phase_id": int(task.phase_id),
        "atom_id": int(task.atom_id),
        "atom_tag": str(task.atom_tag),
        "poses": {
            "ee": pose_list(ee_pose),
            "gripper_center": pose_list(gripper_pose),
            **{name: pose_list(value) for name, value in entities.items()},
        },
        "robot": {
            "joint_pos": joint_pos.tolist(),
            "joint_vel": joint_vel.tolist(),
            "gripper_qpos": float(joint_pos[int(manager._gripper_ids[0])]),
        },
        "actors": {
            name: {
                "actor_type": str(getattr(actor, "actor_type", "")),
                "next_status": actor.next_status,
                "next_pts_shape": None if actor.next_pts is None else list(np.asarray(actor.next_pts).shape),
                "next_mat_shape": None if actor.next_mat is None else list(np.asarray(actor.next_mat).shape),
                "next_mask_shape": None if actor.next_mask is None else list(np.asarray(actor.next_mask).shape),
            }
            for name, actor in task._actor_manager.actors.items()
        },
        "tactiles": {},
    }
    for name, tactile in task._tactile_manager.tactiles.items():
        attachment = tactile.attachment
        obj_pose = getattr(attachment, "obj_pose", None)
        diagnostics["tactiles"][name] = {
            "attachment_points": int(np.asarray(attachment.attachment_points_idx).size),
            "aim_positions_shape": list(np.asarray(attachment.aim_positions).shape),
            "aim_positions_mean": np.asarray(attachment.aim_positions, dtype=np.float64).reshape(-1, 3).mean(axis=0).tolist(),
            "obj_pose": None if obj_pose is None else pose_list(obj_pose),
        }

    if snapshot is not None:
        diagnostics["expected_pose_errors"] = {
            name: pose_error(
                np.asarray(diagnostics["poses"][name], dtype=np.float64),
                snapshot.poses[name],
            )
            for name in ("ee", *entities.keys())
            if name in snapshot.poses
        }
        diagnostics["expected_joint_pos_max_abs"] = float(
            np.max(
                np.abs(
                    np.asarray(diagnostics["robot"]["joint_pos"], dtype=np.float64)
                    - np.asarray(snapshot.robot_state["joint_pos"], dtype=np.float64).reshape(-1)
                )
            )
        )
    return diagnostics


class RFCLSnapshotDataset:
    """Lazy reader for an RFCL snapshot manifest and NumPy sidecars."""

    def __init__(
        self,
        root: str | Path,
    ) -> None:
        self.root = Path(root)
        self.manifest_path = self.root / SNAPSHOT_MANIFEST_NAME
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if self.manifest.get("schema") != SNAPSHOT_DATASET_SCHEMA:
            raise ValueError(
                f"Unsupported snapshot schema {self.manifest.get('schema')!r}"
            )
        adapter_metadata = self.manifest.get("adapter")
        if not isinstance(adapter_metadata, dict):
            raise ValueError("Snapshot manifest is missing adapter metadata")
        self.adapter = create_rfcl_task_adapter(
            adapter_id=adapter_metadata.get("adapter_id"),
            task_name=self.manifest.get("task"),
        )
        if int(adapter_metadata.get("state_dim", -1)) != self.adapter.state_dim:
            raise ValueError("Snapshot adapter state dimension mismatch")
        expected_adapter_metadata = self.adapter.manifest_metadata()
        if adapter_metadata != expected_adapter_metadata:
            raise ValueError("Snapshot adapter metadata does not match the registered adapter")
        self.demos = tuple(self.manifest.get("demos", ()))
        self.snapshot_metadata = dict(self.manifest.get("snapshots", {}))
        if not self.demos:
            raise ValueError(f"Snapshot dataset contains no demos: {self.root}")
        frame_owners: dict[int, str] = {}
        for snapshot_id, metadata in self.snapshot_metadata.items():
            frame = int(metadata["uipc_frame"])
            previous = frame_owners.get(frame)
            if previous is not None:
                raise ValueError(
                    "Snapshot dataset reuses UIPC frame "
                    f"{frame} for {previous!r} and {snapshot_id!r}; regenerate "
                    "with collision-free archived frames"
                )
            frame_owners[frame] = str(snapshot_id)
        self._cache: dict[str, RFCLSnapshot] = {}
        self._policy_starts: list[int] = []
        for demo in self.demos:
            snapshot_ids = demo.get("snapshot_ids", ())
            if not snapshot_ids:
                raise ValueError(f"Demo {demo.get('demo_id')} contains no snapshots")
            for state_index, snapshot_id in enumerate(snapshot_ids):
                metadata = self.snapshot_metadata.get(snapshot_id)
                if metadata is None:
                    raise KeyError(f"Missing snapshot metadata for {snapshot_id!r}")
                if int(metadata["state_index"]) != state_index:
                    raise ValueError(
                        f"Non-contiguous state index in demo {demo.get('demo_id')}: "
                        f"expected {state_index}, got {metadata['state_index']}"
                    )
            configured_start = demo.get("policy_start_state_index")
            if configured_start is None:
                raise ValueError(
                    f"Demo {demo.get('demo_id')} is missing policy_start_state_index"
                )
            configured_start = int(configured_start)
            if not 0 <= configured_start < len(snapshot_ids) - 1:
                raise ValueError(
                    "policy_start_state_index must leave at least one "
                    f"transition, got {configured_start} for demo "
                    f"{demo.get('demo_id')}"
                )
            self._policy_starts.append(configured_start)

    def _raw_snapshot(self, demo_index: int, state_index: int) -> RFCLSnapshot:
        demo_index = int(demo_index)
        if not 0 <= demo_index < len(self.demos):
            raise IndexError(f"demo_index out of range: {demo_index}")
        snapshot_ids = self.demos[demo_index]["snapshot_ids"]
        state_index = int(state_index)
        if not 0 <= state_index < len(snapshot_ids):
            raise IndexError(f"state_index out of range: {state_index}")
        snapshot_id = str(snapshot_ids[state_index])
        if snapshot_id not in self._cache:
            self._cache[snapshot_id] = read_snapshot_sidecar(
                self.root,
                self.snapshot_metadata[snapshot_id],
            )
        return self._cache[snapshot_id]

    def raw_snapshot(self, demo_index: int, state_index: int) -> RFCLSnapshot:
        """Read a snapshot using its original Motion Plan state index."""

        return self._raw_snapshot(demo_index, state_index)

    def policy_start_state_index(self, demo_index: int) -> int:
        """Return the raw snapshot index used as local state zero."""

        demo_index = int(demo_index)
        if not 0 <= demo_index < len(self.demos):
            raise IndexError(f"demo_index out of range: {demo_index}")
        return int(self._policy_starts[demo_index])

    def state_count(self, demo_index: int) -> int:
        demo_index = int(demo_index)
        if not 0 <= demo_index < len(self.demos):
            raise IndexError(f"demo_index out of range: {demo_index}")
        return len(self.demos[demo_index]["snapshot_ids"]) - self.policy_start_state_index(
            demo_index
        )

    def local_to_raw_state_index(self, demo_index: int, state_index: int) -> int:
        demo_index = int(demo_index)
        state_index = int(state_index)
        count = self.state_count(demo_index)
        if not 0 <= state_index < count:
            raise IndexError(f"local state_index out of range: {state_index}")
        return self.policy_start_state_index(demo_index) + state_index

    def snapshot(self, demo_index: int, state_index: int) -> RFCLSnapshot:
        """Read a snapshot using the RFCL-local index (zero is handoff)."""

        return self._raw_snapshot(
            int(demo_index), self.local_to_raw_state_index(demo_index, state_index)
        )

    def handoff_diagnostics(self, demo_index: int) -> dict[str, float | int]:
        raw_index = self.policy_start_state_index(demo_index)
        snapshot = self._raw_snapshot(demo_index, raw_index)
        return {
            "policy_start_state_index": raw_index,
            "handoff_error_m": self.adapter.handoff_error(snapshot.poses),
            "handoff_tolerance_m": self.adapter.handoff_tolerance_m,
        }

    def infer_action_scale(
        self,
        *,
        margin: float = 1.05,
        minimum: float = 1e-4,
    ) -> np.ndarray:
        """Infer the RFCL qpos-delta scale from all saved demo transitions.

        BC checkpoints use a scale tuned for their small image-policy steps.
        RFCL instead follows the upstream convention of normalizing each demo
        action dimension by its observed magnitude.  This matters at Motion
        Plan atom boundaries, where one saved transition can be much larger
        than a normal servo step.  The margin leaves a small amount of room
        for exploration while keeping every demo action representable.
        """

        if not np.isfinite(float(margin)) or float(margin) < 1.0:
            raise ValueError("margin must be finite and at least 1")
        if not np.isfinite(float(minimum)) or float(minimum) <= 0.0:
            raise ValueError("minimum must be finite and positive")
        maxima = np.zeros(7, dtype=np.float64)
        for demo_index, demo in enumerate(self.demos):
            snapshot_ids = demo["snapshot_ids"]
            previous = self.snapshot(demo_index, 0).robot_state["joint_pos"]
            previous = np.asarray(previous).reshape(-1)[:7]
            for state_index in range(1, self.state_count(demo_index)):
                current = self.snapshot(
                    demo_index, state_index
                ).robot_state["joint_pos"]
                current = np.asarray(current).reshape(-1)[:7]
                maxima = np.maximum(maxima, np.abs(current - previous))
                previous = current
        return np.maximum(maxima * float(margin), float(minimum)).astype(np.float32)

    def infer_target_pos_vel_action_scale(
        self,
        *,
        margin: float = 1.05,
        minimum: float = 1e-4,
        velocity_minimum: float = 1e-3,
    ) -> np.ndarray:
        """Infer scales for Motion Plan position and velocity targets."""
        if not np.isfinite(float(margin)) or float(margin) < 1.0:
            raise ValueError("margin must be finite and at least 1")
        if not np.isfinite(float(minimum)) or float(minimum) <= 0.0:
            raise ValueError("minimum must be finite and positive")
        if not np.isfinite(float(velocity_minimum)) or float(velocity_minimum) <= 0.0:
            raise ValueError("velocity_minimum must be finite and positive")
        qpos_maxima = np.zeros(7, dtype=np.float64)
        velocity_maxima = np.zeros(7, dtype=np.float64)
        for demo_index, _demo in enumerate(self.demos):
            for state_index in range(self.state_count(demo_index) - 1):
                current = np.asarray(
                    self.snapshot(demo_index, state_index).robot_state["joint_pos"],
                    dtype=np.float64,
                ).reshape(-1)[:7]
                command = self.snapshot(
                    demo_index, state_index + 1
                ).robot_state
                target = np.asarray(command["joint_pos_target"], dtype=np.float64).reshape(-1)[:7]
                velocity = np.asarray(command["joint_vel_target"], dtype=np.float64).reshape(-1)[:7]
                qpos_maxima = np.maximum(qpos_maxima, np.abs(target - current))
                velocity_maxima = np.maximum(velocity_maxima, np.abs(velocity))
        return np.concatenate(
            (
                np.maximum(qpos_maxima * float(margin), float(minimum)),
                np.maximum(velocity_maxima * float(margin), float(velocity_minimum)),
            )
        ).astype(np.float32)

    def to_demo_trajectories(
        self,
        action_scale: Sequence[float] | None = None,
        action_mode: str | None = None,
    ) -> list[DemoTrajectory]:
        action_mode = str(
            self.manifest.get("action_mode", "qpos_delta")
            if action_mode is None
            else action_mode
        )
        if action_scale is None:
            action_scale = self.manifest.get("action_scale")
        if action_mode in ("target_pos_vel", "target_pos_vel_force"):
            required_scale_dim = (
                TARGET_QPOS_VELOCITY_FORCE_ACTION_DIM
                if action_mode == "target_pos_vel_force"
                else TARGET_QPOS_VELOCITY_ACTION_DIM
            )
            if action_scale is None or len(action_scale) != required_scale_dim:
                action_scale = self.infer_target_pos_vel_action_scale()
                if action_mode == "target_pos_vel_force":
                    action_scale = np.concatenate(
                        (action_scale, np.ones(1, dtype=np.float32))
                    )
        elif action_mode == "qpos_delta":
            if action_scale is None:
                action_scale = self.infer_action_scale()
        else:
            raise ValueError(f"Unsupported RFCL action_mode: {action_mode!r}")
        trajectories = []
        for demo_index, demo_metadata in enumerate(self.demos):
            snapshots = [
                self.snapshot(demo_index, state_index)
                for state_index in range(self.state_count(demo_index))
            ]
            states = np.stack(
                [snapshot.privileged_state for snapshot in snapshots], axis=0
            ).astype(np.float32)
            if action_mode == "qpos_delta":
                actions = np.stack(
                    [
                        target_qpos_to_action(
                            snapshots[index].robot_state["joint_pos"].reshape(-1)[:7],
                            snapshots[index + 1]
                            .robot_state["joint_pos"]
                            .reshape(-1)[:7],
                            action_scale,
                        )
                        for index in range(max(0, len(snapshots) - 1))
                    ],
                    axis=0,
                ) if len(snapshots) > 1 else np.empty((0, 7), dtype=np.float32)
            else:
                qpos_scale = np.asarray(action_scale, dtype=np.float32)[:7]
                velocity_scale = np.asarray(action_scale, dtype=np.float32)[7:]
                if action_mode == "target_pos_vel_force":
                    velocity_scale = velocity_scale[:7]
                    actions = np.stack(
                        [
                            target_qpos_velocity_force_to_action(
                                snapshots[index].robot_state["joint_pos"].reshape(-1)[:7],
                                snapshots[index + 1]
                                .robot_state["joint_pos_target"]
                                .reshape(-1)[:7],
                                snapshots[index + 1]
                                .robot_state["joint_vel_target"]
                                .reshape(-1)[:7],
                                bool(snapshots[index + 1].force_position_write),
                                qpos_scale,
                                velocity_scale,
                            )
                            for index in range(max(0, len(snapshots) - 1))
                        ],
                        axis=0,
                    ) if len(snapshots) > 1 else np.empty((0, TARGET_QPOS_VELOCITY_FORCE_ACTION_DIM), dtype=np.float32)
                else:
                    actions = np.stack(
                        [
                            target_qpos_velocity_to_action(
                                snapshots[index].robot_state["joint_pos"].reshape(-1)[:7],
                                snapshots[index + 1]
                                .robot_state["joint_pos_target"]
                                .reshape(-1)[:7],
                                snapshots[index + 1]
                                .robot_state["joint_vel_target"]
                                .reshape(-1)[:7],
                                qpos_scale,
                                velocity_scale,
                            )
                            for index in range(max(0, len(snapshots) - 1))
                        ],
                        axis=0,
                    ) if len(snapshots) > 1 else np.empty((0, TARGET_QPOS_VELOCITY_ACTION_DIM), dtype=np.float32)
            rewards = np.zeros(len(actions), dtype=np.float32)
            terminated = np.zeros(len(actions), dtype=bool)
            if len(actions):
                rewards[-1] = 1.0
                terminated[-1] = True
            trajectories.append(
                DemoTrajectory(
                    demo_id=str(demo_metadata["demo_id"]),
                    path=self.manifest_path,
                    frame_indices=np.asarray(
                        [snapshot.sim_step for snapshot in snapshots],
                        dtype=np.int64,
                    ),
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    terminated=terminated,
                    tags=tuple(snapshot.atom_tag for snapshot in snapshots),
                    velocity_source="simulator_joint_vel",
                )
            )
        return trajectories
