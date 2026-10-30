"""PD (paper abbreviation): grasp the upper drawer handle and pull the drawer open.

No joints: the cabinet and both drawers are ordinary rigid props in the UIPC
world.  The cabinet is rotated 90 deg about z so its front faces the robot, and
is a fixed body whose collision copy reproduces the cavity (two sides, back,
bottom, top and the two shelves), not the hull: the drawers must slide inside
it and ride their shelves.  The upper drawer is a dynamic ABD body dragged out
by its 20 mm nub against the shelf friction; the lower drawer is never grasped
or touched, so it is fixed in place with a box collision copy.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import Enum, auto

from openvitac_spec import (
    FRANKA_PANDA_GSMINI,
    Actor,
    CameraSpec,
    PinholeIntrinsics,
    gsmini_pair,
)

from ..base import Task
from ..expert import Action, ExpertWorld
from ..motion import (
    Hold,
    Move,
    advance_move,
    begin_move,
    hand_for_pad_midpoint,
    pad_close_q,
    quat_conjugate,
    quat_mul,
    quat_rotate,
)

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


_HEAD_CAM_QUAT_XYZW = (-0.061628, 0.704416, 0.704416, -0.061628)

_CABINET_POS = (0.75, 0.0, 0.002)
_CABINET_ROT_XYZW = (0.0, 0.0, 0.7071067811865476, 0.7071067811865476)  # 90 deg about z
# Drawer front (0.105) is 10 mm behind the cabinet front (0.115) in local y.
_DRAWER_Y_OFFSET = 0.010
_LOWER_DRAWER_OFFSET_Z = 0.0085 + 0.0010  # shelf height + assembly clearance
_UPPER_DRAWER_OFFSET_Z = 0.0700 + 0.0010

# Closed drawer poses in the cabinet frame: the drawer origin (its bottom centre)
# sits 10 mm behind the cabinet front, 1 mm above its shelf.
_LOWER_DRAWER_CLOSED_POS = (0.0, _DRAWER_Y_OFFSET, _LOWER_DRAWER_OFFSET_Z)
_UPPER_DRAWER_CLOSED_POS = (0.0, _DRAWER_Y_OFFSET, _UPPER_DRAWER_OFFSET_Z)
# Drawer poses are those anchors transformed by the cabinet's stage pose. In the
# cabinet's 90 deg frame, local +y points along world -x.
_LOWER_DRAWER_POS = _add(
    _CABINET_POS, quat_rotate(_CABINET_ROT_XYZW, _LOWER_DRAWER_CLOSED_POS)
)
_UPPER_DRAWER_POS = _add(
    _CABINET_POS, quat_rotate(_CABINET_ROT_XYZW, _UPPER_DRAWER_CLOSED_POS)
)
# Grasp geometry for the upper drawer's front post (the handle is a 20 mm wide,
# 20 mm tall rectangular nub at the drawer front, protruding along local +y).
_HANDLE_POS = (0.0, 0.124, 0.025)  # m, upper drawer local frame
# The gripping pad midpoint is 5 mm behind the handle's world-x centre; this
# clears the GsMini housing from the drawer face while keeping the pads on the
# 25 mm-deep nub.
_GRASP_BACKOFF_X = 0.005  # m
_APPROACH_DISTANCE = 0.08  # m, pre-grasp back-off opposite the finger axis
_PULL_SEGMENT = 0.02  # m along world -x per segmented stroke
_PULL_STEPS = 6  # 0.12 m commanded, enough for the 0.08 m success rail stroke
_SUCCESS_PULL_DISTANCE = 0.08  # m, displacement along the rail
_SUCCESS_Z_THRESHOLD = 0.01  # m, avoid counting a lifted drawer as pulled out
# Without a joint the drawer can be dragged out obliquely; gate the yaw and the
# lateral drift as well. The cabinet leaves 1 mm per side (drawer x +-75 vs wall
# inner faces at +-76, back at -106 vs -105) and the pull is 0.10 m, so a clean
# pull drifts far less than the gates; they only reject a gross oblique drag.
_SUCCESS_YAW_LIMIT = math.radians(10.0)
_SUCCESS_X_DRIFT = 0.01  # m

# Cabinet-local fixed collision copy: 7 boxes (two sides, back, floor, two
# shelves, top) enclose the cavity; a solid hull would fill it and eject the
# drawers.  The upper drawer's tet hull is solid, so its 150 x 210 mm body
# clears the 9 mm walls by 1 mm per side and its shelf by 1 mm.  The boxes are
# disjoint on purpose: UIPC rejects a fixed body whose own meshes intersect or
# nearly touch, hence the 0.2 mm gaps at the junctions (far smaller than the
# drawer clearance).
_CABINET_CAVITY_BOXES = (
    ((-0.0805, 0.0, 0.072), (0.0045, 0.115, 0.072)),  # left wall: x -85..-76
    ((0.0805, 0.0, 0.072), (0.0045, 0.115, 0.072)),  # right wall: x 76..85
    ((0.0, -0.1105, 0.072), (0.0758, 0.0045, 0.072)),  # back: y -115..-106
    ((0.0, 0.0046, 0.00365), (0.0758, 0.1104, 0.00365)),  # floor: z 0..7.3
    ((0.0, 0.0046, 0.008), (0.0758, 0.1104, 0.0005)),  # lower shelf: 7.5..8.5
    ((0.0, 0.0046, 0.0695), (0.0758, 0.1104, 0.0005)),  # upper shelf: 69..70
    ((0.0, 0.0046, 0.133), (0.0758, 0.1104, 0.011)),  # top: z 122..144 mm
)
# The lower drawer is never touched: its body box is enough, the nub is omitted.
_LOWER_DRAWER_BOXES = (((0.0, 0.0, 0.025), (0.075, 0.105, 0.025)),)
# The tet hull is 1.585e-3 m^3; 205.7 kg/m^3 reproduces the 0.326 kg that the
# shipped hollow shell masses at PhysX's 1e3.
_UPPER_DRAWER_DENSITY = 205.7

# The front panel (2.5 mm) is below the pad gap; the expert grips the 20 mm nub.
# 1.5 mm of overtravel per pad gives 2 mu F_n ~ 11.7 N, above the 8.6 N the
# shelf drag demands (mu m g = 2.5 * 3.2 N); the block tasks keep the 1 mm
# default (7.85 N).
_GRASP_OVERTRAVEL = 0.0015
_HANDLE_WIDTH = 0.020
_CLOSE_Q = pad_close_q(_HANDLE_WIDTH, _GRASP_OVERTRAVEL)

# Nominal arm pose: GsMini default plus joint7 = 0.741, pinned explicitly.
_TASK_INITIAL_JOINT_POS = {
    "panda_joint1": -0.010809095,
    "panda_joint2": 0.096037410,
    "panda_joint3": 0.000734462,
    "panda_joint4": -2.433035851,
    "panda_joint5": 0.035354517,
    "panda_joint6": 2.500859022,
    "panda_joint7": 0.741,
    "panda_finger.*": 0.02,
}


def _handle_world(world: ExpertWorld) -> Vec3:
    """World position of the upper drawer's grasp post (drawer-local point)."""
    pos, quat = world.object_pose("upper_drawer")
    return _add(pos, quat_rotate(quat, _HANDLE_POS))


def _hand_target(handle: Vec3, quat: Quat, *, clearance: float) -> Vec3:
    """Hand-origin target whose pad midpoint sits at ``handle``.

    ``motion.hand_for_pad_midpoint`` places the pads on the handle along the
    finger axis; the extra back-off clears the GsMini housing from the drawer.
    """
    hx, hy, hz = hand_for_pad_midpoint(handle, quat, clearance=clearance)
    return (hx - _GRASP_BACKOFF_X, hy, hz)


class _PullPhase(Enum):
    INIT = auto()
    OPEN = auto()
    APPROACH = auto()
    DESCEND = auto()
    CLOSE = auto()
    PULL = auto()
    DONE = auto()


@dataclass(frozen=True)
class _PullState:
    phase: _PullPhase
    quat: Quat = (0.0, 0.0, 0.0, 1.0)
    command_pos: Vec3 = (0.0, 0.0, 0.0)
    command_quat: Quat = (0.0, 0.0, 0.0, 1.0)
    gripper_qpos: float = 0.0
    move: Move | None = None
    hold: Hold | None = None
    pull_index: int = 0


def _action(state: _PullState) -> Action:
    return Action(state.command_pos, state.command_quat, state.gripper_qpos)


def _start_pull(
    world: ExpertWorld, state: _PullState
) -> tuple[_PullState, Action]:
    ex, ey, ez = world.arm.ee_pose()[0]
    goal = (ex - _PULL_SEGMENT, ey, ez)
    move, action = begin_move(
        world,
        goal,
        state.quat,
        ticks=60,
        settle_ticks=60,
        gripper_qpos=state.gripper_qpos,
    )
    return (
        replace(
            state,
            phase=_PullPhase.PULL,
            command_pos=goal,
            command_quat=state.quat,
            move=move,
        ),
        action,
    )


def _enter(
    phase: _PullPhase, world: ExpertWorld, state: _PullState
) -> tuple[_PullState, Action]:
    """Compute ``phase``'s targets from the live world and its first action."""
    match phase:
        case _PullPhase.OPEN:
            qpos = world.arm.gripper.open_qpos
            state = replace(state, phase=phase, gripper_qpos=qpos, hold=Hold(200))
            return state, _action(state)
        case _PullPhase.APPROACH:
            handle = _handle_world(world)
            goal = _hand_target(handle, state.quat, clearance=_APPROACH_DISTANCE)
            ticks = 200
        case _PullPhase.DESCEND:
            handle = _handle_world(world)
            goal = _hand_target(handle, state.quat, clearance=0.0)
            ticks = 250
        case _PullPhase.CLOSE:
            qpos = _CLOSE_Q
            state = replace(state, phase=phase, gripper_qpos=qpos, hold=Hold(300))
            return state, _action(state)
        case _PullPhase.PULL:
            return _start_pull(world, state)
        case _PullPhase.INIT | _PullPhase.DONE:
            raise AssertionError(f"{phase} has no entry targets")
    move, action = begin_move(
        world,
        goal,
        state.quat,
        ticks=ticks,
        settle_ticks=60,
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
    phase: _PullPhase, world: ExpertWorld, state: _PullState
) -> tuple[_PullState, Action] | None:
    """Follow ``phase``'s transition, including the pull-segment loop."""
    if phase is _PullPhase.PULL:
        if next(_success(world), False):
            return None
        if state.pull_index + 1 >= _PULL_STEPS:
            return None
        state = replace(state, move=None, pull_index=state.pull_index + 1)
        return _start_pull(world, state)
    if phase is _PullPhase.CLOSE:
        return _start_pull(world, replace(state, move=None, hold=None))
    next_phase = {
        _PullPhase.OPEN: _PullPhase.APPROACH,
        _PullPhase.APPROACH: _PullPhase.DESCEND,
        _PullPhase.DESCEND: _PullPhase.CLOSE,
        _PullPhase.PULL: _PullPhase.DONE,
    }[phase]
    return _enter(next_phase, world, state)


class _PullExpert:
    """Per-tick scripted policy: open, approach the handle, close, pull.

    Phase order: OPEN -> APPROACH -> DESCEND -> CLOSE -> PULL (up to
    ``_PULL_STEPS`` segments, each stopping on the success gate) -> DONE.
    Each move keeps the legacy tick budget and 60-tick re-command.
    """

    def init(self) -> _PullState:
        return _PullState(phase=_PullPhase.INIT)

    def step(
        self, world: ExpertWorld, state: _PullState
    ) -> tuple[_PullState, Action] | None:
        if state.phase is _PullPhase.INIT:
            pos, quat = world.arm.ee_pose()
            state = replace(
                state,
                quat=quat,
                command_pos=pos,
                command_quat=quat,
                gripper_qpos=world.arm.gripper.joint_pos(),
            )
            return _enter(_PullPhase.OPEN, world, state)
        if state.phase is _PullPhase.DONE:
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
    """Upper drawer pulled out >8 cm along its rail without being dragged askew.

    Displacement, yaw and drift are measured in the cabinet frame against the
    closed drawer pose, so they describe the drawer rather than the arm. With
    no joint an oblique drag would still move the drawer +y, hence the yaw and
    lateral gates.
    """
    while True:
        (dx, dy, dz), drawer_quat = world.object_pose("upper_drawer")
        (cx, cy, cz), cabinet_quat = world.object_pose("cabinet")
        local = quat_rotate(quat_conjugate(cabinet_quat), (dx - cx, dy - cy, dz - cz))
        pull = local[1] - _UPPER_DRAWER_CLOSED_POS[1]
        z_drift = abs(local[2] - _UPPER_DRAWER_CLOSED_POS[2])
        x_drift = abs(local[0] - _UPPER_DRAWER_CLOSED_POS[0])
        rx, ry, rz, rw = quat_mul(quat_conjugate(cabinet_quat), drawer_quat)
        sin_yaw = 2.0 * (rw * rz + rx * ry)
        cos_yaw = 1.0 - 2.0 * (ry * ry + rz * rz)
        yaw = abs(math.atan2(sin_yaw, cos_yaw))
        yield (
            pull > _SUCCESS_PULL_DISTANCE
            and z_drift < _SUCCESS_Z_THRESHOLD
            and x_drift <= _SUCCESS_X_DRIFT
            and yaw <= _SUCCESS_YAW_LIMIT
        )


PullDrawer = Task(
    name="pull_drawer",
    embodiment=FRANKA_PANDA_GSMINI,
    sensors=gsmini_pair(streams=("rgb", "depth", "marker_rgb")),
    cameras=(
        CameraSpec(
            prim_path="/World/Camera",
            role="head",
            width=480,
            height=270,
            pos=(0.5, 0.5, 0.16),
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
            authority="physx",
        ),
        # A fixed UIPC body; its cavity-shaped box copy (not the hull) lets the
        # drawers slide inside and ride their shelves.
        Actor(
            "pull_drawer/cabinet_body.usd",
            "/World/actors/cabinet",
            pos=_CABINET_POS,
            rot=_CABINET_ROT_XYZW,
            dynamic=False,
            uipc_fixed_boxes=_CABINET_CAVITY_BOXES,
            authority="physx",
        ),
        # Never grasped or touched: a fixed body box keeps it in place without a
        # tet asset; only the visual usd keeps the hollow shape.
        Actor(
            "pull_drawer/lower_drawer.usd",
            "/World/actors/lower_drawer",
            pos=_LOWER_DRAWER_POS,
            rot=_CABINET_ROT_XYZW,
            dynamic=False,
            uipc_fixed_boxes=_LOWER_DRAWER_BOXES,
            authority="physx",
        ),
        # Solid-hull tet proxy. The shipped usd is a hollow shell (outer faces at
        # x +-75, y -105/105, z 0/50 mm; wall 5.5 mm at the sides, 4.2 mm floor -
        # the 2.46 mm figure is V/A). Closing the cavity faithfully needs
        # epsilon <= 3e-4 and 1.2 mm edges (1.96 M tets); nothing enters the
        # drawer in this task, so the hull - the exact outer silhouette
        # including the 20 mm nub - is a deliberate simplification.
        Actor(
            "pull_drawer/upper_drawer.usd",
            "/World/actors/upper_drawer",
            pos=_UPPER_DRAWER_POS,
            rot=_CABINET_ROT_XYZW,
            authority="uipc",
            tactile_mesh="pull_drawer/upper_drawer_tet.npz",
            contact_object=True,
            density=_UPPER_DRAWER_DENSITY,
        ),
    ),
    instruction="Grasp the upper drawer handle and pull the upper drawer open.",
    expert=_PullExpert(),
    success=_success,
    joint_pos=_TASK_INITIAL_JOINT_POS,
    required_capabilities=frozenset({"binary_close"}),
)
