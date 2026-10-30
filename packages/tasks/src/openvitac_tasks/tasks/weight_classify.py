"""WC (paper abbreviation): classify the block weight by touch, then place it on the
matching plate.  Both blocks share one usd; only their PhysX mass density differs.

Per episode, ``seed`` samples one label.  That block starts at ``(0.38, 0, 0.002)``;
the other block rests on the table at ``(0.30, +/-0.25, 0.002)`` (the legacy y=+/-1.0
standby lies off the new table).  Legacy mapping: light -> blue plate, heavy ->
yellow plate.
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

_LABELS = ("light", "heavy")
_BLOCK_BY_LABEL = {"light": "light_block", "heavy": "heavy_block"}
_PLATE_BY_LABEL = {"light": "blue_plate", "heavy": "yellow_plate"}
# The old stack used 2e5, but that 3.6 kg block slips in the current
# binary-close PhysX gripper; 3e4 (0.54 kg) preserves a 300x contrast.
_DENSITY_BY_LABEL = {"light": 1.0e2, "heavy": 3.0e4}
# Both blocks are UIPC bodies: spawn 1.5 mm above the 2.5 mm table top (the
# solver rejects an initial overlap with the fixed support body).
_BLOCK_START_POS = (0.38, 0.0, 0.004)
_BLOCK_STANDBY_BY_LABEL = {
    "light": (0.30, 0.25, 0.004),
    "heavy": (0.30, -0.25, 0.004),
}
# UIPC fixed supports mirror the table and both plates (box centres/half
# extents are actor-local metres; the tops sit on the authored 2.5 mm table
# and 8 mm plate, as in roughness_classify).
_TABLE_SUPPORT_BOX = ((0.0, 0.0, 0.0013), (0.3, 0.3, 0.0012))
_PLATE_SUPPORT_BOX = ((0.0, 0.0, 0.0033), (0.04, 0.04, 0.0027))
_GRASP_HEIGHT = 0.035
# The gel is the contact surface on the UIPC path, so a blind close crushes it;
# the commanded close comes from the block width (15.00 mm, black_smooth_cuboid
# bounds +/-0.0075 m) through the shared close law.
_BLOCK_WIDTH = 0.01500
_CLOSE_Q = pad_close_q(_BLOCK_WIDTH)
_APPROACH_CLEARANCE = 0.05
# The approach arrives horizontally over a standing 80 mm block.  The pad face
# is 25.3 mm tall, so at _APPROACH_CLEARANCE the pad's lower edge is inside the
# block (see roughness_regrasp); clear the block top instead.
_TRANSIT_CLEARANCE = 0.09
_LIFT_HEIGHT = 0.06
_TARGET_BLOCK_Z = 0.01
_SUCCESS_XY_TOLERANCE = 0.04
_SUCCESS_Z_TOLERANCE = 0.01
_SUCCESS_Z_AXIS_DOT = 0.965


def _label(world: ExpertWorld) -> str:
    """Read the sampled weight label from the task setup metadata."""
    value = world.expert_param("weight_label")
    if not isinstance(value, str) or value not in _LABELS:
        raise ValueError(f"unexpected weight_label: {value!r}")
    return value


def _randomize(task: Task, seed: int) -> EpisodeRandomization:
    """Sample the weight label and move the selected block to the shared start pose."""
    label = random.Random(seed).choice(_LABELS)
    selected = _BLOCK_BY_LABEL[label]
    other_label = "heavy" if label == "light" else "light"
    other = _BLOCK_BY_LABEL[other_label]
    target = _PLATE_BY_LABEL[label]
    poses = {
        f"/World/actors/{selected}": ActorPose(
            _BLOCK_START_POS, (0.0, 0.0, 0.0, 1.0)
        ),
        f"/World/actors/{other}": ActorPose(
            _BLOCK_STANDBY_BY_LABEL[other_label], (0.0, 0.0, 0.0, 1.0)
        ),
    }
    plate = target.removesuffix("_plate")
    return EpisodeRandomization(
        seed=seed,
        instruction=(
            f"Classify the block weight by touch: place the {label} block on "
            f"the {plate} plate."
        ),
        actor_poses=poses,
        contact_object=f"/World/actors/{selected}",
        metadata={
            "weight_label": label,
            "block_density": _DENSITY_BY_LABEL[label],
            "target_plate": target,
        },
    )


class _WeightPhase(Enum):
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
    _WeightPhase.PREGRASP: _WeightPhase.DESCEND,
    _WeightPhase.DESCEND: _WeightPhase.CLOSE,
    _WeightPhase.CLOSE: _WeightPhase.LIFT,
    _WeightPhase.LIFT: _WeightPhase.CARRY,
    _WeightPhase.CARRY: _WeightPhase.PLACE,
    _WeightPhase.PLACE: _WeightPhase.OPEN,
    _WeightPhase.OPEN: _WeightPhase.RETRACT,
    _WeightPhase.RETRACT: _WeightPhase.DONE,
}

_MOVE_TICKS = 60
_SETTLE_TICKS = 60


@dataclass(frozen=True)
class _WeightState:
    phase: _WeightPhase
    label: str = ""
    quat: Quat = (0.0, 0.0, 0.0, 1.0)
    command_pos: Vec3 = (0.0, 0.0, 0.0)
    command_quat: Quat = (0.0, 0.0, 0.0, 1.0)
    gripper_qpos: float = 0.0
    move: Move | None = None
    hold: Hold | None = None


def _action(state: _WeightState) -> Action:
    return Action(state.command_pos, state.command_quat, state.gripper_qpos)


def _enter(
    phase: _WeightPhase, world: ExpertWorld, state: _WeightState
) -> tuple[_WeightState, Action]:
    """Compute ``phase``'s targets from the live world and its first action."""
    block = _BLOCK_BY_LABEL[state.label]
    plate = _PLATE_BY_LABEL[state.label]
    match phase:
        case _WeightPhase.PREGRASP:
            (bx, by, bz), _ = world.object_pose(block)
            hand_z = bz + _GRASP_HEIGHT + HAND_TO_GEL_FACE
            goal = (bx, by, hand_z + _TRANSIT_CLEARANCE)
            ticks = 120
        case _WeightPhase.DESCEND:
            (bx, by, bz), _ = world.object_pose(block)
            hand_z = bz + _GRASP_HEIGHT + HAND_TO_GEL_FACE
            goal = (bx, by, hand_z)
            ticks = _MOVE_TICKS
        case _WeightPhase.CLOSE:
            # Gel face _GRASP_OVERTRAVEL into the block per side.
            qpos = _CLOSE_Q
            state = replace(state, phase=phase, gripper_qpos=qpos, hold=Hold(60))
            return state, _action(state)
        case _WeightPhase.LIFT:
            (bx, by, bz), _ = world.object_pose(block)
            hand_z = bz + _GRASP_HEIGHT + HAND_TO_GEL_FACE
            goal = (bx, by, hand_z + _LIFT_HEIGHT)
            ticks = 90
        case _WeightPhase.CARRY:
            (tx, ty, tz), _ = world.object_pose(plate)
            place_z = tz + _TARGET_BLOCK_Z + _GRASP_HEIGHT + HAND_TO_GEL_FACE
            goal = (tx, ty, place_z + _APPROACH_CLEARANCE)
            ticks = 120
        case _WeightPhase.PLACE:
            (tx, ty, tz), _ = world.object_pose(plate)
            place_z = tz + _TARGET_BLOCK_Z + _GRASP_HEIGHT + HAND_TO_GEL_FACE
            goal = (tx, ty, place_z)
            ticks = 90
        case _WeightPhase.OPEN:
            qpos = world.arm.gripper.open_qpos
            state = replace(state, phase=phase, gripper_qpos=qpos, hold=Hold(60))
            return state, _action(state)
        case _WeightPhase.RETRACT:
            (tx, ty, tz), _ = world.object_pose(plate)
            place_z = tz + _TARGET_BLOCK_Z + _GRASP_HEIGHT + HAND_TO_GEL_FACE
            goal = (tx, ty, place_z + _APPROACH_CLEARANCE)
            ticks = 90
        case _WeightPhase.INIT | _WeightPhase.DONE:
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


class _WeightExpert:
    """Per-tick scripted policy: grasp the selected block, place it, retract.

    Phase order: PREGRASP -> DESCEND -> CLOSE -> LIFT -> CARRY -> PLACE -> OPEN
    -> RETRACT -> DONE, with the legacy tick budgets and 60-tick re-command.
    """

    def init(self) -> _WeightState:
        return _WeightState(phase=_WeightPhase.INIT)

    def step(
        self, world: ExpertWorld, state: _WeightState
    ) -> tuple[_WeightState, Action] | None:
        if state.phase is _WeightPhase.INIT:
            pos, quat = world.arm.ee_pose()
            # Parallel the pad plane to the selected block's +/-y face; see
            # roughness_classify for the lever-arm rationale.
            _, bquat = world.object_pose(_BLOCK_BY_LABEL[_label(world)])
            quat = align_pad_to_face(quat, quat_rotate(bquat, (0.0, 1.0, 0.0)))
            state = replace(
                state,
                label=_label(world),
                quat=quat,
                command_pos=pos,
                command_quat=quat,
                gripper_qpos=world.arm.gripper.joint_pos(),
            )
            return _enter(_WeightPhase.PREGRASP, world, state)
        if state.phase is _WeightPhase.DONE:
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
            if next_phase is _WeightPhase.DONE:
                return None
            return _enter(next_phase, world, state)
        if state.hold is not None:
            hold = state.hold.consume()
            if hold is not None:
                return replace(state, hold=hold), _action(state)
            state = replace(state, hold=None)
            next_phase = _NEXT[state.phase]
            if next_phase is _WeightPhase.DONE:
                return None
            return _enter(next_phase, world, state)
        raise AssertionError(f"phase {state.phase} has no active command")


def _success(world: ExpertWorld) -> Iterator[bool]:
    """Selected block on its plate, upright, after gripper release and retract."""
    while True:
        label = _label(world)
        (bx, by, bz), quat = world.object_pose(_BLOCK_BY_LABEL[label])
        (tx, ty, tz), _ = world.object_pose(_PLATE_BY_LABEL[label])
        qx, qy = quat[0], quat[1]
        upright = 1.0 - 2.0 * (qx * qx + qy * qy)
        seated = (
            abs(bx - tx) < _SUCCESS_XY_TOLERANCE
            and abs(by - ty) < _SUCCESS_XY_TOLERANCE
            and abs(bz - (tz + _TARGET_BLOCK_Z)) < _SUCCESS_Z_TOLERANCE
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


WeightClassify = Task(
    name="weight_classify",
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
        # Both plates are fixed supports: the label picks which one receives the
        # block.  A dynamic actor cannot declare a fixed UIPC collision copy.
        Actor(
            "roughness_task/blue_square_plate.usd",
            "/World/actors/blue_plate",
            pos=(0.45, 0.08, 0.002),
            dynamic=False,
            uipc_fixed_boxes=(_PLATE_SUPPORT_BOX,),
            authority="physx",
        ),
        Actor(
            "roughness_task/yellow_square_plate.usd",
            "/World/actors/yellow_plate",
            pos=(0.45, -0.08, 0.002),
            dynamic=False,
            uipc_fixed_boxes=(_PLATE_SUPPORT_BOX,),
            authority="physx",
        ),
        Actor(
            "roughness_task/black_smooth_cuboid.usd",
            "/World/actors/light_block",
            pos=_BLOCK_STANDBY_BY_LABEL["light"],
            authority="uipc",
            tactile_mesh="roughness_task/black_smooth_cuboid_tet.npz",
            density=_DENSITY_BY_LABEL["light"],
        ),
        Actor(
            "roughness_task/black_smooth_cuboid.usd",
            "/World/actors/heavy_block",
            pos=_BLOCK_STANDBY_BY_LABEL["heavy"],
            authority="uipc",
            tactile_mesh="roughness_task/black_smooth_cuboid_tet.npz",
            density=_DENSITY_BY_LABEL["heavy"],
        ),
    ),
    instruction=(
        "Classify the block weight by touch: place the light block on the blue "
        "plate and the heavy block on the yellow plate."
    ),
    expert=_WeightExpert(),
    success=_success,
    randomizer=_randomize,
    required_capabilities=frozenset({"binary_close"}),
)
