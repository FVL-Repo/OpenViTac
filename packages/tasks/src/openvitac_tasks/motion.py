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

# The gel flat-face centre sits this far below the panda_hand origin along hand
# z: measured 118.0-118.6 mm over a recorded episode (forward kinematics on the
# recorded joint_pos).  0.131, the inherited value, sat the pads 12.2 mm high.
HAND_TO_GEL_FACE = 0.11834
# Gel flat-face normal in the panda_hand frame (mount kinematics, independent
# of the finger joint q); the right pad is the exact opposite.
PAD_NORMAL_IN_HAND = (0.0, -1.0, 0.0)
# Command-space hand origin -> gel pad midpoint along the hand's finger axis
# (+z): measured 116.67 mm at the settled close pose in insert_usb (118.02 mm
# geometric; the servo settles ~1.3 mm short, so targeting uses the command
# value).  The offset must be rotated with the hand - adding it to a world
# coordinate puts the pads centimetres off whenever the pad plane is aligned
# to an object face.
PAD_MID_TO_HAND = 0.11667


def pad_midpoint(hand: Vec3, quat: Quat) -> Vec3:
    """Gel pad midpoint for a commanded hand origin and orientation."""
    axis = quat_rotate(quat, (0.0, 0.0, 1.0))
    return (
        hand[0] + axis[0] * PAD_MID_TO_HAND,
        hand[1] + axis[1] * PAD_MID_TO_HAND,
        hand[2] + axis[2] * PAD_MID_TO_HAND,
    )


def hand_for_pad_midpoint(
    pad_mid: Vec3, quat: Quat, *, clearance: float = 0.0
) -> Vec3:
    """Hand origin putting the gel pad midpoint at ``pad_mid``.

    ``clearance`` backs the hand off further along the finger axis (pre-grasp).
    """
    axis = quat_rotate(quat, (0.0, 0.0, 1.0))
    back = PAD_MID_TO_HAND + clearance
    return (
        pad_mid[0] - axis[0] * back,
        pad_mid[1] - axis[1] * back,
        pad_mid[2] - axis[2] * back,
    )


# Close law.  On the UIPC path the gel is the contact surface, so close_qpos =
# 0.0 drives each pad 4.376 mm into a 4.00 mm gel.  The measured pad-face gap
# at this embodiment is PAD_GAP_AT_ZERO + PAD_GAP_PER_Q * q (q = finger
# displacement, metres), so closing by ``overtravel`` per side needs
# q = (width - 2 * overtravel - gap at q=0) / (2 mm per mm of q).  1 mm is the
# default and suits the blocks; pull_drawer passes 1.5 mm because its drag
# needs the larger normal force.
GRASP_OVERTRAVEL = 0.001
PAD_GAP_AT_ZERO = 0.0071276
PAD_GAP_PER_Q = 2.0


def pad_close_q(
    block_width: float, overtravel: float = GRASP_OVERTRAVEL
) -> float:
    """Gripper target that presses the gel ``overtravel`` into a block."""
    return (block_width - 2.0 * overtravel - PAD_GAP_AT_ZERO) / PAD_GAP_PER_Q


def quat_rotate(q: Quat, v: Vec3) -> Vec3:
    """Rotate ``v`` by the XYZW quaternion ``q``."""
    x, y, z, w = q
    m = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    return (
        sum(m[0][j] * v[j] for j in range(3)),
        sum(m[1][j] * v[j] for j in range(3)),
        sum(m[2][j] * v[j] for j in range(3)),
    )


def quat_conjugate(quat: Quat) -> Quat:
    """Inverse of a unit quaternion (XYZW)."""
    x, y, z, w = quat
    return (-x, -y, -z, w)


def quat_mul(a: Quat, b: Quat) -> Quat:
    """Hamilton product, XYZW."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def align_pad_to_face(quat: Quat, face_normal: Vec3) -> Quat:
    """Shortest-arc rotation putting the pad flat face on ``face_normal``."""
    n = quat_rotate(quat, PAD_NORMAL_IN_HAND)
    s_vec = (
        n[1] * face_normal[2] - n[2] * face_normal[1],
        n[2] * face_normal[0] - n[0] * face_normal[2],
        n[0] * face_normal[1] - n[1] * face_normal[0],
    )
    s = math.sqrt(sum(c * c for c in s_vec))
    c = sum(a * b for a, b in zip(n, face_normal, strict=True))
    if s < 1e-9:
        if c > 0.0:  # already parallel
            return quat
        # Antiparallel: the shortest arc is a 180 deg turn about any axis in the
        # pad plane (a degenerate cross product gives no axis).
        axis = (0.0, 0.0, 1.0) if abs(n[2]) < 0.9 else (1.0, 0.0, 0.0)
        d = (
            n[1] * axis[2] - n[2] * axis[1],
            n[2] * axis[0] - n[0] * axis[2],
            n[0] * axis[1] - n[1] * axis[0],
        )
        norm = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        half_turn = (d[0] / norm, d[1] / norm, d[2] / norm, 0.0)
        return quat_mul(half_turn, quat)
    h = 0.5 * math.atan2(s, c)
    d = (
        s_vec[0] / s * math.sin(h),
        s_vec[1] / s * math.sin(h),
        s_vec[2] / s * math.sin(h),
        math.cos(h),
    )
    return quat_mul(d, quat)  # left-multiplied: the error is in the world frame


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
