"""Pure typing for scripted experts, success checks, and the world handles.

This module is import-safe without isaaclab: numpy appears only under
``TYPE_CHECKING``.  The sim ``Scene`` implements :class:`ExpertWorld` structurally;
the cli holds the conformance assertion, so sim never imports tasks.

Gripper actions are capability-gated rather than sensor-variant-typed: a task
declares ``required_capabilities`` and the world exposes a :class:`GripperHandle`
whose methods set the common binary target.  Xense-specific adaptive methods
remain a later runtime concern; declaring the capability does not fabricate them.

Arm motion/state lives on an explicit :class:`ArmHandle` reached through
``ExpertWorld.arm``; the current worlds are single-arm, so that default arm is
the only handle until a multi-arm world adds an ``arms`` mapping.

The scripted policy is an explicit per-tick :class:`ExpertStateMachine`; a
success check is an iterator advanced after every physics step.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from openvitac_spec import GripperCapability

if TYPE_CHECKING:
    import numpy.typing as npt

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # XYZW

type TaskLogger = logging.LoggerAdapter[logging.Logger]


class GripperHandle(Protocol):
    """Capability-annotated gripper handle shared by all current variants."""

    @property
    def capabilities(self) -> frozenset[GripperCapability]: ...

    @property
    def open_qpos(self) -> float:
        """Joint position target for a fully open gripper."""
        ...

    @property
    def close_qpos(self) -> float:
        """Joint position target for a fully closed gripper."""
        ...

    def set_target(self, qpos: float) -> None:
        """Set the gripper joint target for the next control tick (no step)."""
        ...

    def joint_pos(self) -> float:
        """Current driven gripper position (mean over mirrored jaws)."""
        ...


class ArmHandle(Protocol):
    """One articulated arm and its gripper.

    World frame throughout; quaternions are XYZW.  ``ee_pose`` and
    ``set_ee_target`` address the embodiment's ``arm.ee_link`` body (the
    wrist/flange link the IK solver drives), not the tool center point between
    the fingers.
    """

    @property
    def gripper(self) -> GripperHandle: ...

    def ee_pose(self) -> tuple[Vec3, Quat]:
        """World pose ``(pos, quat_xyzw)`` of the arm's EE link."""
        ...

    def set_ee_target(self, pos: Vec3, quat: Quat) -> None:
        """Set the EE target for the next control tick (no step)."""
        ...

    def joint_pos(self) -> npt.NDArray[Any]:
        """Current joint positions in the articulation's joint order."""
        ...

    def gripper_contact(self, *, force_threshold: float = 1.0) -> tuple[bool, bool]:
        """``(left, right)`` finger-pad contact above ``force_threshold`` N."""
        ...


class ExpertWorld(Protocol):
    """Ground-truth world handle a scripted expert/success fn drives.

    World frame everywhere; quaternions XYZW.  The world layer exposes one
    :class:`ArmHandle` and the task/setup metadata; arm motion and sensing stay
    on the arm.
    """

    @property
    def arm(self) -> ArmHandle:
        """The default arm driven by scripted experts.

        Single-arm worlds expose only this handle; a multi-arm world will add an
        ``arms`` mapping keyed by arm id and keep ``arm`` as the default entry.
        """
        ...

    @property
    def logger(self) -> TaskLogger:
        """Run-scoped logger; task code must not create its own logger."""
        ...

    def object_pose(self, prim_name: str) -> tuple[Vec3, Quat]:
        """Ground-truth world pose of a dynamic actor, by prim path or leaf name."""
        ...

    def expert_param(self, name: str) -> str | float:
        """Task setup metadata available to the scripted expert/success check.

        Values come from ``EpisodeRandomization.metadata`` and ``expert_params``;
        tasks must not use process globals to recover a sampled label.
        """
        ...


# A success check is a per-tick verdict generator: the collector creates the
# iterator once per episode and advances it after each physics step.
type SuccessFn = Callable[[ExpertWorld], Iterator[bool]]


@dataclass(frozen=True)
class Action:
    """One control tick's explicit command set (world frame, quaternions XYZW).

    Both targets are always present for each control tick.
    """

    ee_pos: Vec3
    ee_quat: Quat
    gripper_qpos: float


class ExpertStateMachine[S](Protocol):
    """Per-tick scripted policy."""

    def init(self) -> S: ...

    def step(
        self, world: ExpertWorld, state: S
    ) -> tuple[S, Action] | None:
        """Return ``(state, action)``, or ``None`` to end the episode.

        ``world`` is the post-step world; raising is a policy error.
        """
        ...
