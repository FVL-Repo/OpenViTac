"""PD (paper abbreviation): grasp the upper drawer handle and pull the drawer open.

The cabinet and both drawers are separate rigid props; the cabinet is rotated 90 deg
about z so its front faces the robot. Each drawer rides a prismatic joint on the
cabinet, offset 10 mm along the cabinet's local +y and lifted to its shelf.

The cabinet is spawned dynamic with density 1e5 so each prismatic joint has a real
body0. PhysX does not simulate a world-anchored prismatic joint on a bare rigid
object: with a static (collision-only) cabinet the authored joint freezes the
drawer, so a 100 N pull moves it <0.4 mm. The high density keeps the cabinet
effectively fixed against the drawer's reaction.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import Enum, auto

from openvitac_spec import (
    FRANKA_PANDA_GSMINI,
    Actor,
    CameraSpec,
    PinholeIntrinsics,
    PrismaticJoint,
    gsmini_pair,
)

from ..base import Task
from ..expert import Action, ExpertWorld
from ..motion import Hold, Move, advance_move, begin_move

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _rotate(quat: Quat, vec: Vec3) -> Vec3:
    """Rotate ``vec`` by the XYZW quaternion ``quat``."""
    x, y, z, w = quat
    tx = 2.0 * (y * vec[2] - z * vec[1])
    ty = 2.0 * (z * vec[0] - x * vec[2])
    tz = 2.0 * (x * vec[1] - y * vec[0])
    return (
        vec[0] + w * tx + (y * tz - z * ty),
        vec[1] + w * ty + (z * tx - x * tz),
        vec[2] + w * tz + (x * ty - y * tx),
    )


def _conjugate(quat: Quat) -> Quat:
    x, y, z, w = quat
    return (-x, -y, -z, w)


_HEAD_CAM_QUAT_XYZW = (-0.061628, 0.704416, 0.704416, -0.061628)

_CABINET_POS = (0.75, 0.0, 0.002)
_CABINET_ROT_XYZW = (0.0, 0.0, 0.7071067811865476, 0.7071067811865476)  # 90 deg about z
# Drawer front (0.105) is 10 mm behind the cabinet front (0.115) in local y.
_DRAWER_Y_OFFSET = 0.010
_LOWER_DRAWER_OFFSET_Z = 0.0085 + 0.0010  # shelf height + assembly clearance
_UPPER_DRAWER_OFFSET_Z = 0.0700 + 0.0010

# Joint anchors are the drawer centres in the cabinet frame; joint value 0 is the
# spawned (closed) pose.
_LOWER_DRAWER_JOINT_POS = (0.0, _DRAWER_Y_OFFSET, _LOWER_DRAWER_OFFSET_Z)
_UPPER_DRAWER_JOINT_POS = (0.0, _DRAWER_Y_OFFSET, _UPPER_DRAWER_OFFSET_Z)
# Drawer poses are those anchors transformed by the cabinet's stage pose. In the
# cabinet's 90 deg frame, local +y points along world -x.
_LOWER_DRAWER_POS = _add(
    _CABINET_POS, _rotate(_CABINET_ROT_XYZW, _LOWER_DRAWER_JOINT_POS)
)
_UPPER_DRAWER_POS = _add(
    _CABINET_POS, _rotate(_CABINET_ROT_XYZW, _UPPER_DRAWER_JOINT_POS)
)
# Local +y is the pull-out direction; the pull stroke is 0.10 m, so 0.15 leaves
# headroom without letting the drawer leave the rail.
_DRAWER_PULL_LIMITS = (0.0, 0.15)

# Grasp geometry for the upper drawer's front post (the handle is a 20 mm wide,
# 20 mm tall rectangular nub at the drawer front, protruding along local +y).
_HANDLE_POS = (0.0, 0.124, 0.025)  # m, upper drawer local frame
# Commanded hand origin -> pad grasp midpoint along the hand's finger axis (+z),
# measured for this GsMini hand at the settled close pose in insert_usb
# (118.02 mm geometric, 116.67 mm in command space). The vertical grasp puts the
# pad midpoint on the handle, so the hand sits this far above it.
_HAND_TCP_OFFSET = 0.11667  # m
# The gripping pad midpoint is 5 mm behind the handle's world-x centre; this
# clears the GsMini housing from the drawer face while keeping the pads on the
# 25 mm-deep nub.
_GRASP_BACKOFF_X = 0.005  # m
_APPROACH_DISTANCE = 0.08  # m, pre-grasp back-off opposite the finger axis
_PULL_SEGMENT = 0.02  # m along world -x per segmented stroke
_PULL_STEPS = 6  # 0.12 m commanded, enough for the 0.08 m success rail stroke
_SUCCESS_PULL_DISTANCE = 0.08  # m, joint displacement along the rail
_SUCCESS_Z_THRESHOLD = 0.01  # m, avoid counting a lifted drawer as pulled out

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
    return _add(pos, _rotate(quat, _HANDLE_POS))


def _hand_target(handle: Vec3, quat: Quat, *, clearance: float) -> Vec3:
    """Hand-origin target whose pad midpoint sits at ``handle``.

    ``quat`` is the hand orientation used for the approach; its local +z axis is
    the finger direction, so the hand origin is ``HAND_TCP_OFFSET`` back along
    that axis (above the handle for the vertical grasp). ``clearance`` backs the
    hand off further in the same direction for the pre-grasp.
    """
    finger_axis = _rotate(quat, (0.0, 0.0, 1.0))
    back = _HAND_TCP_OFFSET + clearance
    return (
        handle[0] - finger_axis[0] * back - _GRASP_BACKOFF_X,
        handle[1] - finger_axis[1] * back,
        handle[2] - finger_axis[2] * back,
    )


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
            qpos = world.arm.gripper.close_qpos
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
    """Upper drawer pulled out >8 cm along its rail with <1 cm z drift.

    Displacement is measured in the cabinet frame against the closed joint
    anchor, so it is a real rail displacement rather than an arm/command proxy.
    """
    while True:
        (dx, dy, dz), _ = world.object_pose("upper_drawer")
        (cx, cy, cz), cabinet_quat = world.object_pose("cabinet")
        local = _rotate(_conjugate(cabinet_quat), (dx - cx, dy - cy, dz - cz))
        pull = local[1] - _UPPER_DRAWER_JOINT_POS[1]
        z_drift = abs(local[2] - _UPPER_DRAWER_JOINT_POS[2])
        yield pull > _SUCCESS_PULL_DISTANCE and z_drift < _SUCCESS_Z_THRESHOLD


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
        Actor("plate.usda", "/World/plate", pos=(0.5, 0.0, 0.0), dynamic=False),
        # Dynamic so each prismatic joint has a body0; 1e5 density keeps the
        # cabinet effectively fixed against the drawer reaction.
        Actor(
            "pull_drawer/cabinet_body.usd",
            "/World/actors/cabinet",
            pos=_CABINET_POS,
            rot=_CABINET_ROT_XYZW,
            dynamic=True,
            density=1e5,
        ),
        Actor(
            "pull_drawer/lower_drawer.usd",
            "/World/actors/lower_drawer",
            pos=_LOWER_DRAWER_POS,
            rot=_CABINET_ROT_XYZW,
        ),
        Actor(
            "pull_drawer/upper_drawer.usd",
            "/World/actors/upper_drawer",
            pos=_UPPER_DRAWER_POS,
            rot=_CABINET_ROT_XYZW,
        ),
    ),
    # The cabinet is dynamic, so each joint's body0 is a real rigid body; the
    # anchor resolves through the cabinet's stage pose (cabinet-frame spec.pos).
    joints=(
        PrismaticJoint(
            prim_path="/World/joints/lower_drawer",
            parent="/World/actors/cabinet",
            child="/World/actors/lower_drawer",
            pos=_LOWER_DRAWER_JOINT_POS,
            axis="Y",
            limits=_DRAWER_PULL_LIMITS,
        ),
        PrismaticJoint(
            prim_path="/World/joints/upper_drawer",
            parent="/World/actors/cabinet",
            child="/World/actors/upper_drawer",
            pos=_UPPER_DRAWER_JOINT_POS,
            axis="Y",
            limits=_DRAWER_PULL_LIMITS,
        ),
    ),
    instruction="Grasp the upper drawer handle and pull the upper drawer open.",
    expert=_PullExpert(),
    success=_success,
    joint_pos=_TASK_INITIAL_JOINT_POS,
    required_capabilities=frozenset({"binary_close"}),
)
