"""In-B v1 (paper abbreviation): insert a block into its matching hole in the
yellow box.

Three blocks share the scene; each episode samples one target and the expert
inserts it into the corresponding hole.  ``insert_block_v2`` remains a registry
alias without language-conditioned selection.  The box is static and hollow:
the block is carried over the hole, lowered through the top plate, released,
and falls to the box floor.
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
    hand_for_pad_midpoint,
    pad_close_q,
    quat_rotate,
)

_HEAD_CAM_QUAT_XYZW = (0.362287, 0.344597, 0.596861, 0.627501)

# Geometry from the retired task (asset origins sit at the block/box bottom).
# The box USD uses a triangle-mesh collider (physics:approximation="none"):
# PhysX convex decomposition fills the three through-holes so the block cannot
# enter; the box is static, where a triangle mesh is supported.
_BOX_PRIM = "box_with_holes_yellow"
# 2.6 mm: the tet-mesh bottom sits 0.1 mm above the UIPC table support top
# (2.5 mm), so the two fixed bodies do not overlap at init; the authored 2 mm
# sank the box 0.5 mm into the table.
_BOX_POS = (0.48, -0.12, 0.0026)
_BOX_SIZE = 0.1500
_BOX_WALL_THICKNESS = 0.0060
_BLOCK_HEIGHT = 0.0400
# +/-15 mm keeps every block pair at least 5 mm apart on each axis in the worst
# draw (the tightest pair, cube/half cylinder, is 80 x 70 mm apart and their
# extents sum to 45 x 34.4 mm); a UIPC body cannot start overlapping another.
_BLOCK_XY_NOISE = 0.015
_GRASP_HEIGHT = _BLOCK_HEIGHT * 0.5
_APPROACH_CLEARANCE = 0.050
# The approach arrives horizontally over a standing 40 mm block; clear its top
# (the pad face is 25.3 mm tall, see roughness_regrasp) before descending.
_TRANSIT_CLEARANCE = 0.09
# Pad-face widths at the grasp height: cube 40.0 mm; half cylinder 25.0 mm
# (flat face at y=-10.608 mm to the arc apex at +14.392 mm); hexagonal prism
# 31.176 mm across the two grasped flats.  The close law presses 1 mm of gel
# per side, so a blind close would crush the gel on every shape.
_BLOCK_WIDTH_BY_ACTOR = {
    "block_blue_cube": 0.040000,
    "block_blue_half_cylinder": 0.025000,
    "block_red_hexagonal_prism": 0.031176,
}
_CLOSE_Q_BY_BLOCK = {
    name: pad_close_q(width) for name, width in _BLOCK_WIDTH_BY_ACTOR.items()
}
# UIPC fixed support for the table (actor-local box; top on the 2.5 mm table).
_TABLE_SUPPORT_BOX = ((0.0, 0.0, 0.0013), (0.3, 0.3, 0.0012))
_CARRY_BLOCK_Z = 0.200  # block origin above the 152 mm box top
_PRE_INSERT_CLEARANCE = 0.002
_INSERT_DEPTH = 0.010
_RELEASE_RETRACT = 0.060
_SETTLE_STEPS = 90
_INNER_X_MIN = -_BOX_SIZE * 0.5 + _BOX_WALL_THICKNESS
_INNER_X_MAX = _BOX_SIZE * 0.5 - _BOX_WALL_THICKNESS
_INNER_Y_MIN = -_BOX_SIZE * 0.5 + _BOX_WALL_THICKNESS
_INNER_Y_MAX = _BOX_SIZE * 0.5 - _BOX_WALL_THICKNESS
_INNER_Z_MIN = max(0.0, _BOX_WALL_THICKNESS - 0.006)
_INNER_Z_MAX = 0.124

# Sim-time mechanism value (not a spec field).
_BLOCK_DENSITY = 1e3

# Nominal arm pose; joint2 carries the initial height lift (-0.104).
_TASK_INITIAL_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.104,
    "panda_joint3": 0.0,
    "panda_joint4": -2.46,
    "panda_joint5": 0.0,
    "panda_joint6": 2.5,
    "panda_joint7": 0.741,
    "panda_finger.*": 0.02,
}


@dataclass(frozen=True)
class _BlockSpec:
    actor: str
    description: str
    base_pos: tuple[float, float, float]
    hole_xy: tuple[float, float]


_BLOCK_SPECS: dict[str, _BlockSpec] = {
    "cube": _BlockSpec(
        "block_blue_cube",
        "blue cube",
        (0.45, 0.15, 0.004),
        (-0.035000, 0.030000),
    ),
    "half_cylinder": _BlockSpec(
        "block_blue_half_cylinder",
        "blue half cylinder",
        (0.37, 0.08, 0.004),
        (0.034000, -0.018117),
    ),
    "hexagon": _BlockSpec(
        "block_red_hexagonal_prism",
        "red hexagonal prism",
        (0.36, 0.24, 0.004),
        (-0.026000, -0.036000),
    ),
}
_TARGET_BLOCKS = tuple(_BLOCK_SPECS)


def _target(world: ExpertWorld) -> _BlockSpec:
    """Read the sampled target block from the episode setup metadata."""
    key = world.expert_param("target_block")
    if not isinstance(key, str) or key not in _BLOCK_SPECS:
        raise ValueError(f"unexpected target_block: {key!r}")
    return _BLOCK_SPECS[key]


def _box_to_world(
    xy: tuple[float, float],
    box_pos: tuple[float, float, float],
    box_quat: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Box-local xy (at local z=0) in the world frame."""
    dx, dy, _ = quat_rotate(box_quat, (xy[0], xy[1], 0.0))
    return box_pos[0] + dx, box_pos[1] + dy


def _world_to_box(
    pos: tuple[float, float, float],
    box_pos: tuple[float, float, float],
    box_quat: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """World position in the box frame (inverse of the static box pose)."""
    x, y, z, w = box_quat
    return quat_rotate(
        (-x, -y, -z, w),
        (pos[0] - box_pos[0], pos[1] - box_pos[1], pos[2] - box_pos[2]),
    )


class _BlockPhase(Enum):
    INIT = auto()
    PREGRASP = auto()
    DESCEND = auto()
    CLOSE = auto()
    LIFT = auto()
    CARRY = auto()
    ALIGN = auto()
    PRE_INSERT = auto()
    INSERT = auto()
    OPEN = auto()
    RETRACT = auto()
    SETTLE = auto()
    DONE = auto()


_NEXT = {
    _BlockPhase.PREGRASP: _BlockPhase.DESCEND,
    _BlockPhase.DESCEND: _BlockPhase.CLOSE,
    _BlockPhase.CLOSE: _BlockPhase.LIFT,
    _BlockPhase.LIFT: _BlockPhase.CARRY,
    _BlockPhase.CARRY: _BlockPhase.ALIGN,
    _BlockPhase.ALIGN: _BlockPhase.PRE_INSERT,
    _BlockPhase.PRE_INSERT: _BlockPhase.INSERT,
    _BlockPhase.INSERT: _BlockPhase.OPEN,
    _BlockPhase.OPEN: _BlockPhase.RETRACT,
    _BlockPhase.RETRACT: _BlockPhase.SETTLE,
    _BlockPhase.SETTLE: _BlockPhase.DONE,
}

# Legacy per-move tick budgets.  Each move also gets the 60-tick re-command.
_MOVE_TICKS = 60
_SETTLE_TICKS = 60


@dataclass(frozen=True)
class _BlockState:
    phase: _BlockPhase
    quat: Quat = (0.0, 0.0, 0.0, 1.0)
    command_pos: Vec3 = (0.0, 0.0, 0.0)
    command_quat: Quat = (0.0, 0.0, 0.0, 1.0)
    gripper_qpos: float = 0.0
    spec: _BlockSpec | None = None
    move: Move | None = None
    hold: Hold | None = None
    retract_pos: Vec3 = (0.0, 0.0, 0.0)


def _action(state: _BlockState) -> Action:
    return Action(state.command_pos, state.command_quat, state.gripper_qpos)


def _align_hand(
    world: ExpertWorld,
    block: str,
    hole_xy: tuple[float, float],
    box_pos: tuple[float, float, float],
    box_quat: tuple[float, float, float, float],
    block_z: float,
) -> tuple[float, float, float]:
    """Hand target that puts the held block's origin at ``(hole, block_z)``.

    The grasp can centre the block a few millimetres off the hand axis, so
    null the measured block-to-hand offset against the live hole position.
    """
    (gx, gy, gz), _ = world.object_pose(block)
    (ex, ey, ez), _ = world.arm.ee_pose()
    hole_x, hole_y = _box_to_world(hole_xy, box_pos, box_quat)
    return hole_x + (ex - gx), hole_y + (ey - gy), ez + (block_z - gz)


def _enter(
    phase: _BlockPhase, world: ExpertWorld, state: _BlockState
) -> tuple[_BlockState, Action]:
    """Compute ``phase``'s targets from the live world and its first action."""
    spec = state.spec
    assert spec is not None
    match phase:
        case _BlockPhase.PREGRASP:
            # The pad plane is aligned to the block face, so the hand's finger
            # axis is tilted; the pad midpoint (not the hand origin) must sit on
            # the block's mid-plane at _GRASP_HEIGHT.
            (bx, by, bz), _ = world.object_pose(spec.actor)
            goal = hand_for_pad_midpoint(
                (bx, by, bz + _GRASP_HEIGHT),
                state.quat,
                clearance=_TRANSIT_CLEARANCE,
            )
            ticks = 120
        case _BlockPhase.DESCEND:
            (bx, by, bz), _ = world.object_pose(spec.actor)
            goal = hand_for_pad_midpoint(
                (bx, by, bz + _GRASP_HEIGHT), state.quat
            )
            ticks = _MOVE_TICKS
        case _BlockPhase.CLOSE:
            # Gel face _GRASP_OVERTRAVEL into this block per side.
            qpos = _CLOSE_Q_BY_BLOCK[spec.actor]
            state = replace(state, phase=phase, gripper_qpos=qpos, hold=Hold(40))
            return state, _action(state)
        case _BlockPhase.LIFT:
            hx, hy, _ = world.arm.ee_pose()[0]
            box_pos, box_quat = world.object_pose(_BOX_PRIM)
            lift_z = _align_hand(
                world, spec.actor, spec.hole_xy, box_pos, box_quat, _CARRY_BLOCK_Z
            )[2]
            goal = (hx, hy, lift_z)
            ticks = 90
        case _BlockPhase.CARRY:
            box_pos, box_quat = world.object_pose(_BOX_PRIM)
            goal = _align_hand(
                world, spec.actor, spec.hole_xy, box_pos, box_quat, _CARRY_BLOCK_Z
            )
            ticks = 150
        case _BlockPhase.ALIGN:
            box_pos, box_quat = world.object_pose(_BOX_PRIM)
            goal = _align_hand(
                world,
                spec.actor,
                spec.hole_xy,
                box_pos,
                box_quat,
                world.object_pose(spec.actor)[0][2],
            )
            ticks = _MOVE_TICKS
        case _BlockPhase.PRE_INSERT:
            box_pos, box_quat = world.object_pose(_BOX_PRIM)
            box_top = box_pos[2] + _BOX_SIZE
            goal = _align_hand(
                world,
                spec.actor,
                spec.hole_xy,
                box_pos,
                box_quat,
                box_top + _PRE_INSERT_CLEARANCE,
            )
            ticks = 120
        case _BlockPhase.INSERT:
            box_pos, box_quat = world.object_pose(_BOX_PRIM)
            box_top = box_pos[2] + _BOX_SIZE
            goal = _align_hand(
                world,
                spec.actor,
                spec.hole_xy,
                box_pos,
                box_quat,
                box_top + _PRE_INSERT_CLEARANCE - _INSERT_DEPTH,
            )
            ticks = 180
        case _BlockPhase.OPEN:
            qpos = world.arm.gripper.open_qpos
            state = replace(state, phase=phase, gripper_qpos=qpos, hold=Hold(60))
            return state, _action(state)
        case _BlockPhase.RETRACT:
            rx, ry, rz = world.arm.ee_pose()[0]
            goal = (rx, ry, rz + _RELEASE_RETRACT)
            ticks = _MOVE_TICKS
            state = replace(state, retract_pos=goal)
        case _BlockPhase.SETTLE:
            goal = state.retract_pos
            ticks = _SETTLE_STEPS
        case _BlockPhase.INIT | _BlockPhase.DONE:
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


class _BlockExpert:
    """Per-tick scripted policy: pick the block, insert, release, settle.

    Phase order: PREGRASP -> DESCEND -> CLOSE -> LIFT -> CARRY -> ALIGN ->
    PRE_INSERT -> INSERT -> OPEN -> RETRACT -> SETTLE -> DONE.  Each move keeps
    the legacy tick budget and 60-tick re-command; the collector applies every
    Action and owns the only ``scene.step()``.
    """

    def init(self) -> _BlockState:
        return _BlockState(phase=_BlockPhase.INIT)

    def step(
        self, world: ExpertWorld, state: _BlockState
    ) -> tuple[_BlockState, Action] | None:
        if state.phase is _BlockPhase.INIT:
            pos, quat = world.arm.ee_pose()
            spec = _target(world)
            # Parallel the pad plane to the target block's +/-y face; see
            # roughness_classify for the lever-arm rationale.
            _, bquat = world.object_pose(spec.actor)
            quat = align_pad_to_face(quat, quat_rotate(bquat, (0.0, 1.0, 0.0)))
            state = replace(
                state,
                quat=quat,
                command_pos=pos,
                command_quat=quat,
                gripper_qpos=world.arm.gripper.joint_pos(),
                spec=spec,
            )
            return _enter(_BlockPhase.PREGRASP, world, state)
        if state.phase is _BlockPhase.DONE:
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
            if next_phase is _BlockPhase.DONE:
                return None
            return _enter(next_phase, world, state)
        if state.hold is not None:
            hold = state.hold.consume()
            if hold is not None:
                return replace(state, hold=hold), _action(state)
            state = replace(state, hold=None)
            next_phase = _NEXT[state.phase]
            if next_phase is _BlockPhase.DONE:
                return None
            return _enter(next_phase, world, state)
        raise AssertionError(f"phase {state.phase} has no active command")


def _success(world: ExpertWorld) -> Iterator[bool]:
    """Target block seated after release, retract, and a settle window."""
    stable = 0
    while True:
        spec = _target(world)
        block_pos, _ = world.object_pose(spec.actor)
        box_pos, box_quat = world.object_pose(_BOX_PRIM)
        x, y, z = _world_to_box(block_pos, box_pos, box_quat)
        # Legacy checks only y/z and leaves x unbounded.  Requiring x inside the
        # cavity as well rejects a block that missed the hole and came to rest on
        # the table beside the box (same y, z ~ 0 in the box frame).
        seated = (
            _INNER_X_MIN <= x <= _INNER_X_MAX
            and _INNER_Y_MIN <= y <= _INNER_Y_MAX
            and _INNER_Z_MIN <= z <= _INNER_Z_MAX
        )
        # The hand must be released and at the retract height above the box.
        (_, _, ez), _ = world.arm.ee_pose()
        retract_z = (
            box_pos[2]
            + _BOX_SIZE
            + HAND_TO_GEL_FACE
            + _RELEASE_RETRACT
            - _INSERT_DEPTH
        )
        released = (
            world.arm.gripper.joint_pos()
            >= world.arm.gripper.open_qpos - _PRE_INSERT_CLEARANCE
        )
        stable = (
            stable + 1
            if seated and released and ez >= retract_z
            else 0
        )
        # First true only after the settled hold has lasted the legacy settle
        # drive, so the episode ends in SETTLE rather than during RETRACT.
        yield stable >= _SETTLE_STEPS


def _randomize(task: Task, seed: int) -> EpisodeRandomization:
    """Sample the target block and per-block xy offsets.

    The box stays at its nominal pose: static PhysX actors cannot accept
    reset-time pose overrides, and the box is the scene's fixed fixture.
    """
    rng = random.Random(seed)
    target = rng.choice(_TARGET_BLOCKS)
    spec = _BLOCK_SPECS[target]
    poses: dict[str, ActorPose] = {}
    actors = {actor.prim_path: actor for actor in task.actors}
    for block_spec in _BLOCK_SPECS.values():
        actor = actors[f"/World/actors/{block_spec.actor}"]
        poses[actor.prim_path] = ActorPose(
            (
                actor.pos[0] + rng.uniform(-_BLOCK_XY_NOISE, _BLOCK_XY_NOISE),
                actor.pos[1] + rng.uniform(-_BLOCK_XY_NOISE, _BLOCK_XY_NOISE),
                actor.pos[2],
            ),
            actor.rot,
        )
    return EpisodeRandomization(
        seed=seed,
        instruction=(
            f"Insert the {spec.description} into the matching hole in the "
            "yellow box."
        ),
        actor_poses=poses,
        contact_object=f"/World/actors/{spec.actor}",
        metadata={"target_block": target},
    )


InsertBlock = Task(
    name="insert_block",
    embodiment=FRANKA_PANDA_GSMINI,
    sensors=gsmini_pair(streams=("rgb", "depth", "marker_rgb")),
    cameras=(
        CameraSpec(
            prim_path="/World/Camera",
            role="head",
            width=480,
            height=270,
            pos=(0.8, -0.02, 0.26),
            rot=_HEAD_CAM_QUAT_XYZW,
            intrinsics=PinholeIntrinsics(
                focal_length=1.6, focus_distance=1.0, horizontal_aperture=2.4
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
        # The hollow box needs its through-holes in the collision body, which
        # boxes cannot express; the fixed mesh carries them as tets.
        Actor(
            "insert_block/box_with_holes_yellow.usd",
            f"/World/actors/{_BOX_PRIM}",
            pos=_BOX_POS,
            dynamic=False,
            uipc_fixed_mesh="insert_block/box_with_holes_yellow_tet.npz",
            authority="physx",
        ),
        Actor(
            "insert_block/block_blue_cube.usd",
            "/World/actors/block_blue_cube",
            pos=_BLOCK_SPECS["cube"].base_pos,
            authority="uipc",
            tactile_mesh="insert_block/block_blue_cube_tet.npz",
            density=_BLOCK_DENSITY,
        ),
        Actor(
            "insert_block/block_blue_half_cylinder.usd",
            "/World/actors/block_blue_half_cylinder",
            pos=_BLOCK_SPECS["half_cylinder"].base_pos,
            authority="uipc",
            tactile_mesh="insert_block/block_blue_half_cylinder_tet.npz",
            density=_BLOCK_DENSITY,
        ),
        Actor(
            "insert_block/block_red_hexagonal_prism.usd",
            "/World/actors/block_red_hexagonal_prism",
            pos=_BLOCK_SPECS["hexagon"].base_pos,
            authority="uipc",
            tactile_mesh="insert_block/block_red_hexagonal_prism_tet.npz",
            density=_BLOCK_DENSITY,
        ),
    ),
    instruction=(
        "Insert the blue half cylinder into the matching hole in the yellow box."
    ),
    expert=_BlockExpert(),
    success=_success,
    randomizer=_randomize,
    joint_pos=_TASK_INITIAL_JOINT_POS,
    required_capabilities=frozenset({"binary_close"}),
)

# v2 is a registry alias; language-conditioned target selection is not implemented.
InsertBlockV2 = replace(InsertBlock, name="insert_block_v2")
