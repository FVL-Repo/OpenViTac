"""RC (paper abbreviation): classify the block surface by touch, then place it on the
matching plate.  The smooth and rough variants are distinct usd files.

Per episode, ``seed`` samples one label.  That block starts at ``(0.38, 0, 0.002)``;
the other block rests on the table at ``(0.30, 0.25, 0.002)``.  Legacy mapping:
smooth -> blue plate, rough -> yellow plate.
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
from ..motion import Hold, Move, advance_move, begin_move

_HEAD_CAM_QUAT_XYZW = (0.5, 0.5, 0.5, 0.5)  # 2.x camera offset was WXYZ

_LABELS = ("smooth", "rough")
_BLOCK_BY_LABEL = {"smooth": "smooth_block", "rough": "rough_block"}
_PLATE_BY_LABEL = {"smooth": "blue_plate", "rough": "yellow_plate"}
_BLOCK_START_POS = (0.38, 0.0, 0.002)
_BLOCK_STANDBY_POS = (0.30, 0.25, 0.002)
_HAND_TO_GRASP = 0.131
_GRASP_HEIGHT = 0.035
_APPROACH_CLEARANCE = 0.05
_LIFT_HEIGHT = 0.05
_TARGET_BLOCK_Z = 0.01
_SUCCESS_XY_TOLERANCE = 0.04
_SUCCESS_Z_TOLERANCE = 0.01
_SUCCESS_Z_AXIS_DOT = 0.965


def _label(world: ExpertWorld) -> str:
    """Read the sampled classification label from the task setup metadata."""
    value = world.expert_param("roughness_label")
    if not isinstance(value, str) or value not in _LABELS:
        raise ValueError(f"unexpected roughness_label: {value!r}")
    return value


def _randomize(task: Task, seed: int) -> EpisodeRandomization:
    """Sample the label and place the selected block at the shared start pose."""
    label = random.Random(seed).choice(_LABELS)
    selected = _BLOCK_BY_LABEL[label]
    other = _BLOCK_BY_LABEL["rough" if label == "smooth" else "smooth"]
    target = _PLATE_BY_LABEL[label]
    poses = {
        f"/World/actors/{selected}": ActorPose(_BLOCK_START_POS, (0.0, 0.0, 0.0, 1.0)),
        f"/World/actors/{other}": ActorPose(_BLOCK_STANDBY_POS, (0.0, 0.0, 0.0, 1.0)),
    }
    return EpisodeRandomization(
        seed=seed,
        instruction=f"Place the {label} block on the {target.replace('_', ' ')}.",
        actor_poses=poses,
        metadata={"roughness_label": label, "target_plate": target},
    )


class _RoughnessPhase(Enum):
    INIT = auto()
    PREGRASP = auto()
    DESCEND = auto()
    CLOSE = auto()
    LIFT = auto()
    CARRY = auto()
    PLACE = auto()
    OPEN = auto()
    RETRACT = auto()
    DONE = auto()


_NEXT = {
    _RoughnessPhase.PREGRASP: _RoughnessPhase.DESCEND,
    _RoughnessPhase.DESCEND: _RoughnessPhase.CLOSE,
    _RoughnessPhase.CLOSE: _RoughnessPhase.LIFT,
    _RoughnessPhase.LIFT: _RoughnessPhase.CARRY,
    _RoughnessPhase.CARRY: _RoughnessPhase.PLACE,
    _RoughnessPhase.PLACE: _RoughnessPhase.OPEN,
    _RoughnessPhase.OPEN: _RoughnessPhase.RETRACT,
    _RoughnessPhase.RETRACT: _RoughnessPhase.DONE,
}

_MOVE_TICKS = 60
_SETTLE_TICKS = 60


@dataclass(frozen=True)
class _RoughnessState:
    phase: _RoughnessPhase
    label: str = ""
    quat: Quat = (0.0, 0.0, 0.0, 1.0)
    command_pos: Vec3 = (0.0, 0.0, 0.0)
    command_quat: Quat = (0.0, 0.0, 0.0, 1.0)
    gripper_qpos: float = 0.0
    move: Move | None = None
    hold: Hold | None = None


def _action(state: _RoughnessState) -> Action:
    return Action(state.command_pos, state.command_quat, state.gripper_qpos)


def _enter(
    phase: _RoughnessPhase, world: ExpertWorld, state: _RoughnessState
) -> tuple[_RoughnessState, Action]:
    """Compute ``phase``'s targets from the live world and its first action."""
    block = _BLOCK_BY_LABEL[state.label]
    plate = _PLATE_BY_LABEL[state.label]
    match phase:
        case _RoughnessPhase.PREGRASP:
            (bx, by, bz), _ = world.object_pose(block)
            grasp_z = bz + _GRASP_HEIGHT
            goal = (
                bx,
                by,
                grasp_z + _HAND_TO_GRASP + _APPROACH_CLEARANCE,
            )
            ticks = 120
        case _RoughnessPhase.DESCEND:
            (bx, by, bz), _ = world.object_pose(block)
            grasp_z = bz + _GRASP_HEIGHT
            goal = (bx, by, grasp_z + _HAND_TO_GRASP)
            ticks = _MOVE_TICKS
        case _RoughnessPhase.CLOSE:
            qpos = world.arm.gripper.close_qpos
            state = replace(state, phase=phase, gripper_qpos=qpos, hold=Hold(60))
            return state, _action(state)
        case _RoughnessPhase.LIFT:
            (bx, by, bz), _ = world.object_pose(block)
            grasp_z = bz + _GRASP_HEIGHT
            goal = (bx, by, grasp_z + _HAND_TO_GRASP + _LIFT_HEIGHT)
            ticks = 90
        case _RoughnessPhase.CARRY:
            (tx, ty, tz), _ = world.object_pose(plate)
            place_z = tz + _TARGET_BLOCK_Z + _GRASP_HEIGHT
            goal = (
                tx,
                ty,
                place_z + _HAND_TO_GRASP + _APPROACH_CLEARANCE,
            )
            ticks = 120
        case _RoughnessPhase.PLACE:
            (tx, ty, tz), _ = world.object_pose(plate)
            place_z = tz + _TARGET_BLOCK_Z + _GRASP_HEIGHT
            goal = (tx, ty, place_z + _HAND_TO_GRASP)
            ticks = 90
        case _RoughnessPhase.OPEN:
            qpos = world.arm.gripper.open_qpos
            state = replace(state, phase=phase, gripper_qpos=qpos, hold=Hold(60))
            return state, _action(state)
        case _RoughnessPhase.RETRACT:
            (tx, ty, tz), _ = world.object_pose(plate)
            place_z = tz + _TARGET_BLOCK_Z + _GRASP_HEIGHT
            goal = (
                tx,
                ty,
                place_z + _HAND_TO_GRASP + _APPROACH_CLEARANCE,
            )
            ticks = 90
        case _RoughnessPhase.INIT | _RoughnessPhase.DONE:
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


class _RoughnessExpert:
    """Per-tick scripted policy: grasp the selected block, place it, retract.

    Phase order: PREGRASP -> DESCEND -> CLOSE -> LIFT -> CARRY -> PLACE -> OPEN
    -> RETRACT -> DONE, with the legacy tick budgets and 60-tick re-command.
    """

    def init(self) -> _RoughnessState:
        return _RoughnessState(phase=_RoughnessPhase.INIT)

    def step(
        self, world: ExpertWorld, state: _RoughnessState
    ) -> tuple[_RoughnessState, Action] | None:
        if state.phase is _RoughnessPhase.INIT:
            pos, quat = world.arm.ee_pose()
            state = replace(
                state,
                label=_label(world),
                quat=quat,
                command_pos=pos,
                command_quat=quat,
                gripper_qpos=world.arm.gripper.joint_pos(),
            )
            return _enter(_RoughnessPhase.PREGRASP, world, state)
        if state.phase is _RoughnessPhase.DONE:
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
            next_phase = _NEXT[state.phase]
            if next_phase is _RoughnessPhase.DONE:
                return None
            return _enter(next_phase, world, state)
        if state.hold is not None:
            hold = state.hold.consume()
            if hold is not None:
                return replace(state, hold=hold), _action(state)
            state = replace(state, hold=None)
            next_phase = _NEXT[state.phase]
            if next_phase is _RoughnessPhase.DONE:
                return None
            return _enter(next_phase, world, state)
        raise AssertionError(f"phase {state.phase} has no active command")


def _success(world: ExpertWorld) -> Iterator[bool]:
    """Selected block on its plate, upright, after gripper release and retract."""
    while True:
        label = _label(world)
        (bx, by, bz), quat = world.object_pose(_BLOCK_BY_LABEL[label])
        (tx, ty, tz), _ = world.object_pose(_PLATE_BY_LABEL[label])
        # Plates are axis-aligned.
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
            + _HAND_TO_GRASP
            + _APPROACH_CLEARANCE
        )
        retracted = (
            abs(ex - tx) < _SUCCESS_XY_TOLERANCE
            and abs(ey - ty) < _SUCCESS_XY_TOLERANCE
            and ez >= retract_z - _SUCCESS_Z_TOLERANCE
        )
        yield seated and released and retracted


RoughnessClassify = Task(
    name="roughness_classify",
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
        Actor("plate.usda", "/World/plate", pos=(0.5, 0.0, 0.0), dynamic=False),
        Actor(
            "roughness_task/blue_square_plate.usd",
            "/World/actors/blue_plate",
            pos=(0.45, 0.08, 0.002),
        ),
        Actor(
            "roughness_task/yellow_square_plate.usd",
            "/World/actors/yellow_plate",
            pos=(0.45, -0.08, 0.002),
        ),
        Actor(
            "roughness_task/black_smooth_cuboid.usd",
            "/World/actors/smooth_block",
            pos=_BLOCK_STANDBY_POS,
        ),
        Actor(
            "roughness_task/black_rough_cuboid.usd",
            "/World/actors/rough_block",
            pos=_BLOCK_START_POS,
        ),
    ),
    instruction=(
        "Classify the block surface by touch: place the smooth block on the blue "
        "plate and the rough block on the yellow plate."
    ),
    expert=_RoughnessExpert(),
    success=_success,
    randomizer=_randomize,
    required_capabilities=frozenset({"binary_close"}),
)
