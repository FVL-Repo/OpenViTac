"""Minimal per-tick motion helpers shared by the task state machines.

The old blocking ``move_ee_to(pos, quat, steps=N)`` is re-expressed as N control
ticks: each tick commands the linear interpolation from the pose captured when
the move started.  Tasks run a second ``Move`` with the same goal and their
settle tick count to reproduce the legacy re-command that let the servo settle.
A ``Hold`` is a fixed-tick hold of the current command with a new gripper
target.  These are motion primitives only: each task keeps its own explicit
phase machine and no registry or policy engine lives here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .expert import Action, ExpertWorld, Quat, Vec3


def _lerp_pos(a: Vec3, b: Vec3, alpha: float) -> Vec3:
    return (
        a[0] + (b[0] - a[0]) * alpha,
        a[1] + (b[1] - a[1]) * alpha,
        a[2] + (b[2] - a[2]) * alpha,
    )


def _lerp_quat(a: Quat, b: Quat, alpha: float) -> Quat:
    """Component interpolation plus normalization, matching ``move_ee_to``."""
    q = (
        a[0] + (b[0] - a[0]) * alpha,
        a[1] + (b[1] - a[1]) * alpha,
        a[2] + (b[2] - a[2]) * alpha,
        a[3] + (b[3] - a[3]) * alpha,
    )
    norm = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    return (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm)


@dataclass(frozen=True)
class Move:
    """The ticks of one legacy move: ``drive_ticks`` then ``settle_ticks``."""

    goal_pos: Vec3
    goal_quat: Quat
    drive_ticks: int
    settle_ticks: int
    start_pos: Vec3
    start_quat: Quat
    tick: int = 1
    settling: bool = False

    def action(self, gripper_qpos: float) -> Action:
        ticks = self.settle_ticks if self.settling else self.drive_ticks
        alpha = self.tick / ticks
        return Action(
            ee_pos=_lerp_pos(self.start_pos, self.goal_pos, alpha),
            ee_quat=_lerp_quat(self.start_quat, self.goal_quat, alpha),
            gripper_qpos=gripper_qpos,
        )


def begin_move(
    world: ExpertWorld,
    pos: Vec3,
    quat: Quat,
    *,
    ticks: int,
    settle_ticks: int,
    gripper_qpos: float,
) -> tuple[Move, Action]:
    """Start a move from the live EE pose and emit its first action."""
    start_pos, start_quat = world.arm.ee_pose()
    move = Move(
        goal_pos=pos,
        goal_quat=quat,
        drive_ticks=ticks,
        settle_ticks=settle_ticks,
        start_pos=start_pos,
        start_quat=start_quat,
    )
    return move, move.action(gripper_qpos)


def advance_move(
    world: ExpertWorld, move: Move, gripper_qpos: float
) -> tuple[Move, Action] | None:
    """Emit the next tick of ``move``; ``None`` after its settle segment."""
    if not move.settling:
        if move.tick < move.drive_ticks:
            advanced = replace(move, tick=move.tick + 1)
            return advanced, advanced.action(gripper_qpos)
        start_pos, start_quat = world.arm.ee_pose()
        settling = replace(
            move,
            settling=True,
            tick=1,
            start_pos=start_pos,
            start_quat=start_quat,
        )
        return settling, settling.action(gripper_qpos)
    if move.tick < move.settle_ticks:
        advanced = replace(move, tick=move.tick + 1)
        return advanced, advanced.action(gripper_qpos)
    return None


@dataclass(frozen=True)
class Hold:
    """A fixed number of ticks holding the current command."""

    remaining: int

    def consume(self) -> Hold | None:
        """The hold after this tick, or ``None`` when this tick was the last."""
        if self.remaining <= 1:
            return None
        return replace(self, remaining=self.remaining - 1)
