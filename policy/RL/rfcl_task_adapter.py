"""Task-specific semantics for RFCL snapshot training and data collection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from policy.RL.rfcl import (
    DEFAULT_GRIPPER_OFFSET,
    PRIVILEGED_STATE_LAYOUT,
    build_pair_privileged_state,
    handoff_xy_error,
)


@dataclass(frozen=True)
class SnapshotActorBinding:
    actor_name: str
    pose_key: str
    release_after_sync: bool = False


class RFCLTaskAdapter(ABC):
    adapter_id: str
    task_name: str
    gripper_mode: str
    start_tags: tuple[str, ...]
    allowed_tags: tuple[str, ...]
    required_hdf5_keys: tuple[str, ...]
    handoff_tolerance_m: float = 0.01
    gripper_offset_m: float = DEFAULT_GRIPPER_OFFSET

    @property
    def state_dim(self) -> int:
        return PRIVILEGED_STATE_LAYOUT.dim

    @abstractmethod
    def capture_entities(self, task: Any) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def build_privileged_state(
        self,
        *,
        physics_state: Mapping[str, np.ndarray],
        entities: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        return build_pair_privileged_state(
            joint=physics_state["joint_pos"],
            joint_velocity=physics_state["joint_vel"],
            ee_pose=physics_state["ee"],
            controlled_pose=entities["controlled"],
            target_pose=entities["target"],
        )

    @abstractmethod
    def prepare_handoff(self, task: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def snapshot_actor_bindings(self) -> tuple[SnapshotActorBinding, ...]:
        raise NotImplementedError

    def after_restore(self, task: Any) -> None:
        """Refresh task-local references derived from restored entities."""

    def is_snapshot_start(self, atom_tag: str) -> bool:
        return str(atom_tag).strip().lower() in self.start_tags

    def is_policy_entry(self, atom_tag: str) -> bool:
        """Return whether the first RFCL-controlled transition has started."""

        return not self.is_snapshot_start(atom_tag)

    def is_snapshot_eligible(self, atom_tag: str) -> bool:
        return str(atom_tag).strip().lower() in self.allowed_tags

    def check_success(self, task: Any) -> bool:
        return bool(task.check_success())

    def success_diagnostics(self, task: Any) -> dict[str, Any]:
        diagnostics = getattr(task, "_get_success_diagnostics", None)
        return {} if not callable(diagnostics) else dict(diagnostics())

    def handoff_error(self, poses: Mapping[str, np.ndarray]) -> float:
        return handoff_xy_error(
            poses["ee"],
            poses["controlled"],
            gripper_offset=self.gripper_offset_m,
        )

    def irrecoverable_failure(self, task: Any) -> str | None:
        entities = self.capture_entities(task)
        controlled = np.asarray(entities["controlled"], dtype=np.float64)
        target = np.asarray(entities["target"], dtype=np.float64)
        if not np.isfinite(controlled).all() or not np.isfinite(target).all():
            return "non_finite_entity_pose"
        if float(np.linalg.norm(controlled[:3] - target[:3])) > 0.35:
            return "controlled_entity_out_of_workspace"
        if float(controlled[2]) < -0.05 or float(controlled[2]) > 0.50:
            return "controlled_entity_invalid_height"
        return None

    def diversity_features(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        return np.concatenate(
            (
                state[PRIVILEGED_STATE_LAYOUT.controlled_in_target],
                state[PRIVILEGED_STATE_LAYOUT.joint],
            )
        ).astype(np.float32)

    def manifest_metadata(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "task_name": self.task_name,
            "gripper_mode": self.gripper_mode,
            "state_dim": self.state_dim,
            "start_tags": list(self.start_tags),
            "allowed_tags": list(self.allowed_tags),
            "handoff_tolerance_m": float(self.handoff_tolerance_m),
            "required_hdf5_keys": list(self.required_hdf5_keys),
        }


class InsertUSBTaskAdapter(RFCLTaskAdapter):
    adapter_id = "insert_usb_v2"
    task_name = "insert_USB"
    gripper_mode = "held_closed"
    start_tags = ("move_usb_to_pre_insert",)
    allowed_tags = (
        "move_usb_to_pre_insert",
        "move_usb_to_play_pre_insert",
        "rfcl_free_align",
        "rfcl_outer_rim_touch",
        "rfcl_outer_rim_retract",
        "rfcl_recovery_align",
        "rfcl_inner_wall_shallow_insert",
        "rfcl_inner_wall_retract",
        "rfcl_inner_wall_correction",
        "rfcl_contact_rich_outer_touch",
        "rfcl_contact_rich_retract",
        "rfcl_contact_rich_cross_align",
        "rfcl_contact_rich_inner_touch",
        "rfcl_contact_rich_inner_retract",
        "rfcl_contact_rich_final_align",
        "insert_usb_into_slot",
        "delay",
        "",
    )
    required_hdf5_keys = (
        "observation/head/rgb",
        "observation/wrist/rgb",
        "tactile/left_tactile/rgb_marker",
        "tactile/right_tactile/rgb_marker",
        "embodiment/joint",
    )

    def is_policy_entry(self, atom_tag: str) -> bool:
        tag = str(atom_tag).strip().lower()
        return tag.startswith("rfcl_") or tag == "move_usb_to_play_pre_insert"

    def capture_entities(self, task: Any) -> dict[str, np.ndarray]:
        slot_pose = task.slot.get_pose()
        return {
            "controlled": np.asarray(
                task.prism.get_pose().tolist(), dtype=np.float64
            ),
            "target": np.asarray(
                task._usb_pose_in_slot(slot_pose).tolist(), dtype=np.float64
            ),
            "slot": np.asarray(slot_pose.tolist(), dtype=np.float64),
        }

    def prepare_handoff(self, task: Any) -> None:
        task._prepare_usb_standard()

    def snapshot_actor_bindings(self) -> tuple[SnapshotActorBinding, ...]:
        return (
            SnapshotActorBinding("slot", "slot"),
            SnapshotActorBinding("prism", "controlled", release_after_sync=True),
        )

    def after_restore(self, task: Any) -> None:
        task._update_insert_reference_poses()


_ADAPTERS = {
    InsertUSBTaskAdapter.adapter_id: InsertUSBTaskAdapter,
}
_TASK_DEFAULTS = {
    InsertUSBTaskAdapter.task_name: InsertUSBTaskAdapter.adapter_id,
}


def create_rfcl_task_adapter(
    *,
    adapter_id: str | None = None,
    task_name: str | None = None,
) -> RFCLTaskAdapter:
    if adapter_id is None:
        if task_name is None:
            raise ValueError("Either adapter_id or task_name must be provided")
        adapter_id = _TASK_DEFAULTS.get(str(task_name))
        if adapter_id is None:
            raise KeyError(f"No RFCL adapter is registered for task {task_name!r}")
    adapter_type = _ADAPTERS.get(str(adapter_id))
    if adapter_type is None:
        raise KeyError(f"Unknown RFCL task adapter: {adapter_id!r}")
    adapter = adapter_type()
    if task_name is not None and adapter.task_name != str(task_name):
        raise ValueError(
            f"Adapter {adapter.adapter_id!r} belongs to {adapter.task_name!r}, "
            f"not {task_name!r}"
        )
    return adapter
