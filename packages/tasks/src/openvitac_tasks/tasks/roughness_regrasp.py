"""RR (paper abbreviation): find the rough block by touch and place it on the yellow
plate, regrasping when the first block picked is the smooth one.

Both blocks start on the table at x=0.40 (left y=+0.12, right y=-0.12).  ``seed``
samples which side holds the rough block and which side the expert grasps first;
the instruction is always to place the rough block on the yellow plate.
"""
from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import Enum, auto

from openvitac_spec import (
    FRANKA_PANDA_GSMINI,
    Actor,
    ActorPose,
    CameraSpec,
    PinholeIntrinsics,
    gsmini_pair,
)

from ..base import EpisodeRandomization, Task
from ..expert import Action, ExpertWorld, Quat, Vec3
from ..motion import (
    HAND_TO_GEL_FACE,
    Hold,
    Move,
    advance_move,
    align_pad_to_face,
    begin_move,
    pad_close_q,
    quat_rotate,
)

_HEAD_CAM_QUAT_XYZW = (0.5, 0.5, 0.5, 0.5)  # 2.x camera offset was WXYZ

# The table top is at 2.5 mm; spawn 1.5 mm above it (UIPC rejects an initial
# overlap with the fixed support body, and PhysX settles the block).
_LEFT_BLOCK_POS = (0.40, 0.12, 0.004)
_RIGHT_BLOCK_POS = (0.40, -0.12, 0.004)
# Park the wrongly grasped block away from the rough block and gripper sweep.
_PARK_BY_SIDE = {
    "left": (0.30, 0.30, 0.004),
    "right": (0.30, -0.30, 0.004),
}
_SIDES = ("left", "right")
_ROUGH_PLATE = "yellow_plate"
_GRASP_HEIGHT = 0.035
# UIPC fixed supports mirror the table and the yellow plate the rough block is
# released onto (box centres/half extents are actor-local metres; the tops sit
# on the authored world z = 2.5 mm table and 8 mm plate, like
# roughness_classify).  The table box is wider than that task's 0.3 m half
# extent because this task parks a block at y = +/-0.30, its edge.
_TABLE_SUPPORT_BOX = ((0.0, 0.0, 0.0013), (0.35, 0.35, 0.0012))
_PLATE_SUPPORT_BOX = ((0.0, 0.0, 0.0033), (0.04, 0.04, 0.0027))
# Both blocks are UIPC bodies, so the close comes from the grasped block's own
# width through the shared close law (motion.pad_close_q); widths are the asset
# x,y bounds (smooth +/-0.0075 m, rough +/-0.00794 m).
_BLOCK_WIDTH_BY_NAME = {"smooth_block": 0.01500, "rough_block": 0.01588}
_CLOSE_Q_BY_BLOCK = {
    name: pad_close_q(width) for name, width in _BLOCK_WIDTH_BY_NAME.items()
}
_APPROACH_CLEARANCE = 0.05
# The grasp approach and the park retract travel horizontally over a standing
# 80 mm block (the target or the just-parked one).  The pad face is 25.3 mm
# tall, so an approach at _APPROACH_CLEARANCE keeps the pad's lower edge
# 7.6 mm inside the block and sweeps it over; PhysX had no pad-block collision
# there.  Approach high enough that the whole pad clears the block top.
_TRANSIT_CLEARANCE = 0.09
_LIFT_HEIGHT = 0.05
_TARGET_BLOCK_Z = 0.01
_SUCCESS_XY_TOLERANCE = 0.04
_SUCCESS_Z_TOLERANCE = 0.01
_SUCCESS_Z_AXIS_DOT = 0.965


def _side_param(world: ExpertWorld, name: str) -> str:
    """Read a sampled block/grasp side from the episode metadata."""
    value = world.expert_param(name)
    if not isinstance(value, str) or value not in _SIDES:
        raise ValueError(f"unexpected {name}: {value!r}")
    return value


def _other(side: str) -> str:
    return "right" if side == "left" else "left"


def _randomize(task: Task, seed: int) -> EpisodeRandomization:
    """Sample rough side and first grasp side; keep both blocks on the table."""
    rng = random.Random(seed)
    rough_side = rng.choice(_SIDES)
    initial_side = rng.choice(_SIDES)
    smooth_side = _other(rough_side)
    poses = {
        "/World/actors/smooth_block": ActorPose(
            _LEFT_BLOCK_POS if smooth_side == "left" else _RIGHT_BLOCK_POS,
            (0.0, 0.0, 0.0, 1.0),
        ),
        "/World/actors/rough_block": ActorPose(
            _LEFT_BLOCK_POS if rough_side == "left" else _RIGHT_BLOCK_POS,
            (0.0, 0.0, 0.0, 1.0),
        ),
    }
    return EpisodeRandomization(
        seed=seed,
        instruction="Find the rough block by touch and place it on the yellow plate.",
        actor_poses=poses,
        # The exploratory grasp of the other block has no contact metrics; only
        # the target (success object) needs them.
        contact_object="/World/actors/rough_block",
        metadata={
            "rough_block_side": rough_side,
            "smooth_block_side": smooth_side,
            "initial_grasp_side": initial_side,
            "target_block": "rough",
            "target_plate": _ROUGH_PLATE,
        },
    )


class _RegraspPhase(Enum):
    INIT = auto()
    GRASP_APPROACH = auto()
    GRASP_DESCEND = auto()
    GRASP_CLOSE = auto()
    GRASP_LIFT = auto()
    PARK_APPROACH = auto()
    PARK_DESCEND = auto()
    PARK_OPEN = auto()
    PARK_RETRACT = auto()
    PLACE_APPROACH = auto()
    PLACE_DESCEND = auto()
    PLACE_OPEN = auto()
    PLACE_RETRACT = auto()
    DONE = auto()


_NEXT = {
    _RegraspPhase.GRASP_APPROACH: _RegraspPhase.GRASP_DESCEND,
    _RegraspPhase.GRASP_DESCEND: _RegraspPhase.GRASP_CLOSE,
    _RegraspPhase.GRASP_CLOSE: _RegraspPhase.GRASP_LIFT,
    _RegraspPhase.PARK_APPROACH: _RegraspPhase.PARK_DESCEND,
    _RegraspPhase.PARK_DESCEND: _RegraspPhase.PARK_OPEN,
    _RegraspPhase.PARK_OPEN: _RegraspPhase.PARK_RETRACT,
    _RegraspPhase.PLACE_APPROACH: _RegraspPhase.PLACE_DESCEND,
    _RegraspPhase.PLACE_DESCEND: _RegraspPhase.PLACE_OPEN,
    _RegraspPhase.PLACE_OPEN: _RegraspPhase.PLACE_RETRACT,
    _RegraspPhase.PLACE_RETRACT: _RegraspPhase.DONE,
}

_MOVE_TICKS = 60
_SETTLE_TICKS = 60


@dataclass(frozen=True)
class _RegraspState:
    phase: _RegraspPhase
    grasp_block: str = ""
    park: Vec3 = (0.0, 0.0, 0.0)
    quat: Quat = (0.0, 0.0, 0.0, 1.0)
    command_pos: Vec3 = (0.0, 0.0, 0.0)
    command_quat: Quat = (0.0, 0.0, 0.0, 1.0)
    gripper_qpos: float = 0.0
    move: Move | None = None
    hold: Hold | None = None


def _action(state: _RegraspState) -> Action:
    return Action(state.command_pos, state.command_quat, state.gripper_qpos)


def _enter(
    phase: _RegraspPhase, world: ExpertWorld, state: _RegraspState
) -> tuple[_RegraspState, Action]:
    """Compute ``phase``'s targets from the live world and its first action."""
    match phase:
        case _RegraspPhase.GRASP_APPROACH:
            (bx, by, bz), _ = world.object_pose(state.grasp_block)
            grasp_z = bz + _GRASP_HEIGHT
            goal = (
                bx,
                by,
                grasp_z + HAND_TO_GEL_FACE + _TRANSIT_CLEARANCE,
            )
            ticks = 100
        case _RegraspPhase.GRASP_DESCEND:
            (bx, by, bz), _ = world.object_pose(state.grasp_block)
            grasp_z = bz + _GRASP_HEIGHT
            goal = (bx, by, grasp_z + HAND_TO_GEL_FACE)
            ticks = _MOVE_TICKS
        case _RegraspPhase.GRASP_CLOSE:
            # Gel face a GRASP_OVERTRAVEL into the grasped block per side.
            qpos = _CLOSE_Q_BY_BLOCK[state.grasp_block]
            state = replace(state, phase=phase, gripper_qpos=qpos, hold=Hold(60))
            return state, _action(state)
        case _RegraspPhase.GRASP_LIFT:
            (bx, by, bz), _ = world.object_pose(state.grasp_block)
            grasp_z = bz + _GRASP_HEIGHT
            goal = (bx, by, grasp_z + HAND_TO_GEL_FACE + _LIFT_HEIGHT)
            ticks = 90
        case _RegraspPhase.PARK_APPROACH:
            px, py, pz = state.park
            goal = (
                px,
                py,
                pz + _GRASP_HEIGHT + HAND_TO_GEL_FACE + _APPROACH_CLEARANCE,
            )
            ticks = 70
        case _RegraspPhase.PARK_DESCEND:
            px, py, pz = state.park
            goal = (px, py, pz + _GRASP_HEIGHT + HAND_TO_GEL_FACE)
            ticks = _MOVE_TICKS
        case _RegraspPhase.PARK_OPEN:
            qpos = world.arm.gripper.open_qpos
            state = replace(state, phase=phase, gripper_qpos=qpos, hold=Hold(60))
            return state, _action(state)
        case _RegraspPhase.PARK_RETRACT:
            px, py, pz = state.park
            goal = (
                px,
                py,
                pz + _GRASP_HEIGHT + HAND_TO_GEL_FACE + _TRANSIT_CLEARANCE,
            )
            ticks = _MOVE_TICKS
        case _RegraspPhase.PLACE_APPROACH:
            (tx, ty, tz), _ = world.object_pose(_ROUGH_PLATE)
            place_z = tz + _TARGET_BLOCK_Z + _GRASP_HEIGHT
            goal = (
                tx,
                ty,
                place_z + HAND_TO_GEL_FACE + _APPROACH_CLEARANCE,
            )
            ticks = 100
        case _RegraspPhase.PLACE_DESCEND:
            (tx, ty, tz), _ = world.object_pose(_ROUGH_PLATE)
            place_z = tz + _TARGET_BLOCK_Z + _GRASP_HEIGHT
            goal = (tx, ty, place_z + HAND_TO_GEL_FACE)
            ticks = 90
        case _RegraspPhase.PLACE_OPEN:
            qpos = world.arm.gripper.open_qpos
            state = replace(state, phase=phase, gripper_qpos=qpos, hold=Hold(60))
            return state, _action(state)
        case _RegraspPhase.PLACE_RETRACT:
            (tx, ty, tz), _ = world.object_pose(_ROUGH_PLATE)
            place_z = tz + _TARGET_BLOCK_Z + _GRASP_HEIGHT
            goal = (
                tx,
                ty,
                place_z + HAND_TO_GEL_FACE + _APPROACH_CLEARANCE,
            )
            ticks = 90
        case _RegraspPhase.INIT | _RegraspPhase.DONE:
            raise AssertionError(f"{phase} has no entry targets")
    move, action = begin_move(
        world,
        goal,
        state.quat,
        ticks=ticks,
        settle_ticks=_SETTLE_TICKS,
        gripper_qpos=state.gripper_qpos,
    )
    return (
        replace(
            state,
            phase=phase,
            command_pos=goal,
            command_quat=state.quat,
            move=move,
        ),
        action,
    )


def _advance(
    phase: _RegraspPhase, world: ExpertWorld, state: _RegraspState
) -> tuple[_RegraspState, Action] | None:
    """Follow ``phase``'s transition, including the regrasp branch."""
    if phase is _RegraspPhase.GRASP_LIFT:
        target = (
            _RegraspPhase.PLACE_APPROACH
            if state.grasp_block == "rough_block"
            else _RegraspPhase.PARK_APPROACH
        )
    elif phase is _RegraspPhase.PARK_RETRACT:
        target = _RegraspPhase.GRASP_APPROACH
        state = replace(state, grasp_block="rough_block")
    else:
        target = _NEXT[phase]
    if target is _RegraspPhase.DONE:
        return None
    return _enter(target, world, state)


class _RegraspExpert:
    """Per-tick scripted policy: grasp a block, regrasp if it was smooth.

    Phase order: GRASP_* (approach, descend, close, lift), then either
    PLACE_* when the rough block was grasped first or PARK_* followed by
    GRASP_* for the rough block and PLACE_*.  Each move keeps the legacy tick
    budget and 60-tick re-command.
    """

    def init(self) -> _RegraspState:
        return _RegraspState(phase=_RegraspPhase.INIT)

    def step(
        self, world: ExpertWorld, state: _RegraspState
    ) -> tuple[_RegraspState, Action] | None:
        if state.phase is _RegraspPhase.INIT:
            pos, quat = world.arm.ee_pose()
            rough_side = _side_param(world, "rough_block_side")
            initial_side = _side_param(world, "initial_grasp_side")
            first_block = (
                "rough_block" if initial_side == rough_side else "smooth_block"
            )
            # Parallel the pad plane to the grasped block's +/-y face; see
            # roughness_classify for the lever-arm rationale.  Both blocks stay
            # axis-aligned (upright on the table), so the second grasp reuses it.
            _, bquat = world.object_pose(first_block)
            quat = align_pad_to_face(quat, quat_rotate(bquat, (0.0, 1.0, 0.0)))
            state = replace(
                state,
                grasp_block=first_block,
                park=_PARK_BY_SIDE[initial_side],
                quat=quat,
                command_pos=pos,
                command_quat=quat,
                gripper_qpos=world.arm.gripper.joint_pos(),
            )
            return _enter(_RegraspPhase.GRASP_APPROACH, world, state)
        if state.phase is _RegraspPhase.DONE:
            return None
        if state.move is not None:
            advanced = advance_move(world, state.move, state.gripper_qpos)
            if advanced is not None:
                move, action = advanced
                return (
                    replace(
                        state,
                        move=move,
                        command_pos=action.ee_pos,
                        command_quat=action.ee_quat,
                    ),
                    action,
                )
            state = replace(state, move=None)
            return _advance(state.phase, world, state)
        if state.hold is not None:
            hold = state.hold.consume()
            if hold is not None:
                return replace(state, hold=hold), _action(state)
            state = replace(state, hold=None)
            return _advance(state.phase, world, state)
        raise AssertionError(f"phase {state.phase} has no active command")


def _success(world: ExpertWorld) -> Iterator[bool]:
    """Rough block on the yellow plate, upright, after release and retract."""
    while True:
        (bx, by, bz), quat = world.object_pose("rough_block")
        (tx, ty, tz), _ = world.object_pose(_ROUGH_PLATE)
        dx = bx - tx
        dy = by - ty
        dz = bz - (tz + _TARGET_BLOCK_Z)
        qx, qy = quat[0], quat[1]
        upright = 1.0 - 2.0 * (qx * qx + qy * qy)
        seated = (
            abs(dx) < _SUCCESS_XY_TOLERANCE
            and abs(dy) < _SUCCESS_XY_TOLERANCE
            and abs(dz) < _SUCCESS_Z_TOLERANCE
            and upright > _SUCCESS_Z_AXIS_DOT
        )
        # Release/retract gate: the fingers have opened and the hand reached the
        # retract height above the plate, so the block is no longer carried.
        released = (
            world.arm.gripper.joint_pos()
            >= world.arm.gripper.open_qpos - _SUCCESS_Z_TOLERANCE
        )
        (ex, ey, ez), _ = world.arm.ee_pose()
        retract_z = (
            tz
            + _TARGET_BLOCK_Z
            + _GRASP_HEIGHT
            + HAND_TO_GEL_FACE
            + _APPROACH_CLEARANCE
        )
        retracted = (
            abs(ex - tx) < _SUCCESS_XY_TOLERANCE
            and abs(ey - ty) < _SUCCESS_XY_TOLERANCE
            and ez >= retract_z - _SUCCESS_Z_TOLERANCE
        )
        yield seated and released and retracted


RoughnessRegrasp = Task(
    name="roughness_regrasp",
    embodiment=FRANKA_PANDA_GSMINI,
    sensors=gsmini_pair(streams=("rgb", "depth", "marker_rgb")),
    cameras=(
        CameraSpec(
            prim_path="/World/Camera",
            role="head",
            width=480,
            height=270,
            pos=(0.9, 0.0, 0.15),
            rot=_HEAD_CAM_QUAT_XYZW,
            intrinsics=PinholeIntrinsics(
                focal_length=2.5, focus_distance=1.0, horizontal_aperture=3.6
            ),
        ),
        # Authored in the robot USD; rides the wrist mount.
        CameraSpec(
            prim_path="/World/Robot/WristCamera/Camera",
            role="wrist",
            width=480,
            height=270,
            mount="/World/Robot/WristCamera",
        ),
    ),
    actors=(
        Actor(
            "plate.usda",
            "/World/plate",
            pos=(0.5, 0.0, 0.0),
            dynamic=False,
            uipc_fixed_boxes=(_TABLE_SUPPORT_BOX,),
            authority="physx",
        ),
        # Both plates are static so the UIPC world owns them as fixed bodies
        # (a dynamic actor cannot declare a fixed UIPC collision copy); identical
        # plates must also share one rest height.
        Actor(
            "roughness_task/blue_square_plate.usd",
            "/World/actors/blue_plate",
            pos=(0.48, 0.08, 0.002),
            dynamic=False,
            uipc_fixed_boxes=(_PLATE_SUPPORT_BOX,),
            authority="physx",
        ),
        Actor(
            "roughness_task/yellow_square_plate.usd",
            "/World/actors/yellow_plate",
            pos=(0.48, -0.08, 0.002),
            dynamic=False,
            uipc_fixed_boxes=(_PLATE_SUPPORT_BOX,),
            authority="physx",
        ),
        Actor(
            "roughness_task/black_smooth_cuboid.usd",
            "/World/actors/smooth_block",
            pos=_LEFT_BLOCK_POS,
            authority="uipc",
            tactile_mesh="roughness_task/black_smooth_cuboid_tet.npz",
        ),
        Actor(
            "roughness_task/black_rough_cuboid.usd",
            "/World/actors/rough_block",
            pos=_RIGHT_BLOCK_POS,
            authority="uipc",
            tactile_mesh="roughness_task/black_rough_cuboid_tet.npz",
        ),
    ),
    instruction="Find the rough block by touch and place it on the yellow plate.",
    expert=_RegraspExpert(),
    success=_success,
    randomizer=_randomize,
    required_capabilities=frozenset({"binary_close"}),
)
