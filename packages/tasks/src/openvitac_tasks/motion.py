"""Minimal per-tick motion helpers shared by the task state machines.

The old blocking ``move_ee_to(pos, quat, steps=N)`` is re-expressed as N control
ticks: each tick commands the linear interpolation from the pose captured when
the move started.  Tasks run a second ``Move`` with the same goal and their
settle tick count to reproduce the legacy re-command that let the servo settle.
A ``Hold`` is a fixed-tick hold of the current command with a new gripper
target.  These are motion primitives only: each task keeps its own explicit
phase machine and no registry or policy engine lives here.

Every grasp point here is a *tool-center* point, and every gripper command is
derived from the tool contract the robot asset declares
(``openvitac_spec.ToolGeometry``), so no helper in this module carries a
gripper model's numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from openvitac_spec import ToolGeometry

from .expert import Action, ExpertWorld, Quat, Vec3

# Grasp policy, not gripper geometry: how far each pad face is commanded into the
# object it holds.  1 mm suits the blocks; pull_drawer passes 1.5 mm because its
# drag needs the larger normal force.  How much finger travel that takes is the
# asset's business (``ToolGeometry.pad_gap``).
GRASP_OVERTRAVEL = 0.001


def pad_span(tool: ToolGeometry, quat: Quat) -> Vec3:
    """World vector from the EE origin to the pad midpoint for hand pose ``quat``.

    The offset comes from the asset, so this is the grasp point for any hand
    orientation; the offset must be rotated with the hand -- adding it to a world
    coordinate puts the pads centimetres off whenever the pad plane is tilted to
    an object face.
    """
    return quat_rotate(quat, tool.center_in_ee)


def tool_center(ee_pos: Vec3, ee_quat: Quat, tool: ToolGeometry) -> Vec3:
    """World point of the asset's ``tool_center`` frame (the pad midpoint)."""
    offset = pad_span(tool, ee_quat)
    return (
        ee_pos[0] + offset[0],
        ee_pos[1] + offset[1],
        ee_pos[2] + offset[2],
    )


def ee_for_tool_center(
    point: Vec3, quat: Quat, tool: ToolGeometry, *, clearance: float = 0.0
) -> Vec3:
    """EE (arm link) target putting ``tool_center`` exactly on ``point``.

    ``clearance`` backs the EE off further along the hand's own +z axis (the
    direction the pads point), which is how a pre-grasp pose clears the object.
    """
    offset = pad_span(tool, quat)
    back = quat_rotate(quat, (0.0, 0.0, 1.0))
    return (
        point[0] - offset[0] - back[0] * clearance,
        point[1] - offset[1] - back[1] * clearance,
        point[2] - offset[2] - back[2] * clearance,
    )


def grasp_depth(tool: ToolGeometry) -> float:
    """Pad midpoint's distance from the EE origin along the hand's +z axis.

    What a height gate needs when it has no hand orientation to hand: with the
    hand pointing straight down this is the pad midpoint's drop below the EE.
    """
    return tool.center_in_ee[2]


def pad_midpoint(hand: Vec3, quat: Quat, tool: ToolGeometry) -> Vec3:
    """Pre-tool-contract name of :func:`tool_center`."""
    return tool_center(hand, quat, tool)


def hand_for_pad_midpoint(
    pad_mid: Vec3, quat: Quat, tool: ToolGeometry, *, clearance: float = 0.0
) -> Vec3:
    """Pre-tool-contract name of :func:`ee_for_tool_center`."""
    return ee_for_tool_center(pad_mid, quat, tool, clearance=clearance)


def pad_close_q(
    tool: ToolGeometry,
    block_width: float,
    overtravel: float = GRASP_OVERTRAVEL,
) -> float:
    """Finger command that presses each pad face ``overtravel`` into a block.

    The commanded pad gap is ``width - 2 * overtravel``; how much finger travel
    that needs is the asset's close law (``ToolGeometry.finger_command_for_gap``),
    so a wider pad stack or a different finger mechanism needs no change here.
    """
    return tool.finger_command_for_gap(block_width - 2.0 * overtravel)


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


def align_pad_to_face(tool: ToolGeometry, quat: Quat, face_normal: Vec3) -> Quat:
    """Shortest-arc rotation putting the pad flat face on ``face_normal``.

    ``tool.pad_normal_in_ee`` is the asset's closing axis, so the pads line up
    with the object face whatever frame the embodiment drives its IK in.
    """
    n = quat_rotate(quat, tool.pad_normal_in_ee)
    s_vec = (
        n[1] * face_normal[2] - n[2] * face_normal[1],
        n[2] * face_normal[0] - n[0] * face_normal[2],
        n[0] * face_normal[1] - n[1] * face_normal[0],
    )
    s = math.sqrt(sum(c * c for c in s_vec))
    c = sum(a * b for a, b in zip(n, face_normal, strict=True))
    # ``n`` comes from a measured joint frame, so "parallel" is never exact: an
    # |n x f| of 1e-8 rad would otherwise be turned into a full half turn about
    # whatever axis the dust happens to point at, i.e. a 180 deg roll flip.
    if s < 1e-6:
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
