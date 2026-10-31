"""In-USB: insert the USB into the slot."""

from __future__ import annotations

import math
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

_HEAD_CAM_QUAT_XYZW = (0.310977, 0.562556, 0.670428, 0.370608)

# The USB USD origin is the plug tip; the plug points down into the slot and the body
# stands above it.
_PLUG_HEIGHT = 0.0124
_BODY_HEIGHT = 0.0450
# Mid-body grasp point, biased slightly up so the fingers clear the start-slot
# rim and table. The +7 mm is inherited from ViTaForge's USB_GRASP_HEIGHT and
# is intentional, not part of the pad TCP calibration.
_GRASP_HEIGHT = _PLUG_HEIGHT + _BODY_HEIGHT * 0.5 + 0.007
_LIFT_HEIGHT = 0.0300  # clears the start-slot rim
_SLOT_HEIGHT = 0.0200  # slot body top above its origin
_SLOT_HOLE_BOTTOM = 0.0050  # hole floor above the slot origin
# Plug-tip seated float: above the hole floor, not jammed against it.
_INSERT_Z = max(_SLOT_HOLE_BOTTOM, _SLOT_HEIGHT - _PLUG_HEIGHT)
# Hover above the target opening before descending.
_APPROACH_CLEARANCE = 0.010
# Commanded hand origin -> pad grasp midpoint. Measured at the settled close
# pose (insert_usb seed 0, 120 Hz): hand (0.400640, -0.000036, 0.182223), pad
# midpoint (0.397410, 0.002730, 0.064201). The actual hand->pad distance is
# 118.02 mm, but the command interface settles ~1.35 mm high and ~0.64 mm in
# +x, so the command-space offset is 116.67 mm. The 0.0584 m bare finger
# offset drives the GsMini housings into the table; UIPC gelpads are
# attachment meshes, so the authored USD pose cannot re-derive this.
_GRASP_TCP_OFFSET = (0.00259, -0.00273, 0.11667)
# Retract clear of the released USB so the success gate sees an open gripper.
_EE_RETRACT_Z = 0.16

# Success gates: xy 2 mm, z 3 mm, tilt ~15 deg, EE retracted.
_SUCCESS_XY_TOLERANCE = 0.002
_SUCCESS_Z_TOLERANCE = 0.003
_SUCCESS_Z_AXIS_DOT = 0.965  # ~15 deg from vertical
_SUCCESS_EE_Z = 0.145
# P0 scene-pose DR: start/target slot x/y each uniform in +/-30 mm.  This
# matches ViTaForge envs/insert_USB.py:33 (SLOT_XY_NOISE = (0.030, 0.030, 0)).
_DR_SLOT_XY_HALF_M = 0.030
# Lower start-slot support (bottom + four walls, open top).  The USB body starts
# above this height, so the fixed support holds the plug without colliding with
# the wider body.  The target slot uses the full tet mesh.
_SUPPORT_HX = 0.0132 / 2
_SUPPORT_HY = 0.0057 / 2
_SUPPORT_WALL = 0.0015
_SUPPORT_GAP = 0.00002
_SUPPORT_H = 0.0160
_START_SUPPORT_BOXES = (
    (
        (_SUPPORT_HX + _SUPPORT_WALL / 2, 0.0, _SUPPORT_H / 2),
        (_SUPPORT_WALL / 2, _SUPPORT_HY + _SUPPORT_WALL, _SUPPORT_H / 2),
    ),
    (
        (-_SUPPORT_HX - _SUPPORT_WALL / 2, 0.0, _SUPPORT_H / 2),
        (_SUPPORT_WALL / 2, _SUPPORT_HY + _SUPPORT_WALL, _SUPPORT_H / 2),
    ),
    (
        (0.0, _SUPPORT_HY + _SUPPORT_WALL / 2, _SUPPORT_H / 2),
        (_SUPPORT_HX - _SUPPORT_GAP, _SUPPORT_WALL / 2, _SUPPORT_H / 2),
    ),
    (
        (0.0, -_SUPPORT_HY - _SUPPORT_WALL / 2, _SUPPORT_H / 2),
        (_SUPPORT_HX - _SUPPORT_GAP, _SUPPORT_WALL / 2, _SUPPORT_H / 2),
    ),
    (
        (0.0, 0.0, 0.005 - _SUPPORT_WALL / 2),
        (_SUPPORT_HX - _SUPPORT_GAP, _SUPPORT_HY - _SUPPORT_GAP, _SUPPORT_WALL / 2),
    ),
)


def _grasp_target(
    grasp_x: float,
    grasp_y: float,
    grasp_z: float,
    *,
    clearance: float = 0.0,
) -> tuple[float, float, float]:
    """Commanded hand pose whose settled pad midpoint is at the grasp point."""
    offset_x, offset_y, offset_z = _GRASP_TCP_OFFSET
    return (
        grasp_x + offset_x,
        grasp_y + offset_y,
        grasp_z + offset_z + clearance,
    )


_MOVE_TOLERANCE_M = 1e-3
# After the interpolated segment, hold the exact goal until it has been within
# tolerance for five ticks (or a hard 60-tick cap, never raising).
_MOVE_SETTLE_TICKS = 5
_MOVE_SETTLE_MAX_TICKS = 60
_GRIP_TOLERANCE = 2e-3
# A jaw blocked by a grasped object stops short of its target; a short
# stationary window means the squeeze is complete.
_GRIP_STALL_EPS = 5e-5
_GRIP_STALL_TICKS = 8


class _UsbPhase(Enum):
    INIT = auto()
    PREGRASP = auto()
    DESCEND = auto()
    CLOSE = auto()
    LIFT = auto()
    CARRY = auto()
    ALIGN = auto()
    CORRECT_1 = auto()
    CORRECT_2 = auto()
    INSERT = auto()
    OPEN = auto()
    RETRACT = auto()
    DONE = auto()


_MOVE_PHASES = frozenset(
    {
        _UsbPhase.PREGRASP,
        _UsbPhase.DESCEND,
        _UsbPhase.LIFT,
        _UsbPhase.CARRY,
        _UsbPhase.ALIGN,
        _UsbPhase.CORRECT_1,
        _UsbPhase.CORRECT_2,
        _UsbPhase.INSERT,
        _UsbPhase.RETRACT,
    }
)

_NEXT: dict[_UsbPhase, _UsbPhase] = {
    _UsbPhase.PREGRASP: _UsbPhase.DESCEND,
    _UsbPhase.DESCEND: _UsbPhase.CLOSE,
    _UsbPhase.CLOSE: _UsbPhase.LIFT,
    _UsbPhase.LIFT: _UsbPhase.CARRY,
    _UsbPhase.CARRY: _UsbPhase.ALIGN,
    _UsbPhase.ALIGN: _UsbPhase.CORRECT_1,
    _UsbPhase.CORRECT_1: _UsbPhase.CORRECT_2,
    _UsbPhase.CORRECT_2: _UsbPhase.INSERT,
    _UsbPhase.INSERT: _UsbPhase.OPEN,
    _UsbPhase.OPEN: _UsbPhase.RETRACT,
    _UsbPhase.RETRACT: _UsbPhase.DONE,
}

# Cartesian speed per tick.  Free-space transport stays fast; the final
# align/null/insert sequence keeps the old slow contact approach.
_PHASE_MAX_STEP_M: dict[_UsbPhase, float] = {
    _UsbPhase.PREGRASP: 1e-3,
    _UsbPhase.DESCEND: 1e-3,
    _UsbPhase.LIFT: 1e-3,
    _UsbPhase.CARRY: 1e-3,
    _UsbPhase.ALIGN: 1e-4,
    _UsbPhase.CORRECT_1: 1e-4,
    _UsbPhase.CORRECT_2: 1e-4,
    _UsbPhase.INSERT: 1e-4,
    _UsbPhase.RETRACT: 1e-3,
}


@dataclass(frozen=True)
class _UsbState:
    """Explicit machine state; the two command targets are always current."""

    phase: _UsbPhase
    ee_pos: Vec3 = (0.0, 0.0, 0.0)
    ee_quat: Quat = (0.0, 0.0, 0.0, 1.0)
    gripper_qpos: float = 0.0
    grasp_x: float = 0.0
    grasp_y: float = 0.0
    grasp_z: float = 0.0
    slot_x: float = 0.0
    slot_y: float = 0.0
    slot_z: float = 0.0
    insert_x: float = 0.0
    insert_y: float = 0.0
    grip_prev: float = 0.0
    grip_stall: int = 0
    start_pos: Vec3 = (0.0, 0.0, 0.0)
    move_ticks: int = 0
    move_tick: int = 0
    settle: int = 0
    settle_ticks: int = 0


def _action(state: _UsbState, ee_pos: Vec3 | None = None) -> Action:
    return Action(
        ee_pos=state.ee_pos if ee_pos is None else ee_pos,
        ee_quat=state.ee_quat,
        gripper_qpos=state.gripper_qpos,
    )


def _lerp_pos(a: Vec3, b: Vec3, alpha: float) -> Vec3:
    return (
        a[0] + (b[0] - a[0]) * alpha,
        a[1] + (b[1] - a[1]) * alpha,
        a[2] + (b[2] - a[2]) * alpha,
    )


def _enter(
    phase: _UsbPhase, world: ExpertWorld, state: _UsbState
) -> tuple[_UsbState, Action] | None:
    """Compute the entry targets for ``phase`` and its first action."""
    match phase:
        case _UsbPhase.PREGRASP:
            (ux, uy, uz), _ = world.object_pose("usb")
            grasp_z = uz + _GRASP_HEIGHT
            state = replace(
                state,
                phase=phase,
                grasp_x=ux,
                grasp_y=uy,
                grasp_z=grasp_z,
                ee_pos=_grasp_target(
                    ux, uy, grasp_z, clearance=_APPROACH_CLEARANCE
                ),
            )
        case _UsbPhase.DESCEND:
            state = replace(
                state,
                phase=phase,
                ee_pos=_grasp_target(state.grasp_x, state.grasp_y, state.grasp_z),
            )
        case _UsbPhase.CLOSE:
            state = replace(
                state,
                phase=phase,
                gripper_qpos=world.arm.gripper.close_qpos,
                grip_prev=world.arm.gripper.joint_pos(),
                grip_stall=0,
            )
        case _UsbPhase.LIFT:
            state = replace(
                state,
                phase=phase,
                ee_pos=_grasp_target(
                    state.grasp_x, state.grasp_y, state.grasp_z + _LIFT_HEIGHT
                ),
            )
        case _UsbPhase.CARRY:
            (sx, sy, sz), _ = world.object_pose("slot")
            state = replace(
                state,
                phase=phase,
                slot_x=sx,
                slot_y=sy,
                slot_z=sz,
                ee_pos=_grasp_target(
                    sx,
                    sy,
                    sz + _SLOT_HEIGHT + _GRASP_HEIGHT,
                    clearance=_APPROACH_CLEARANCE,
                ),
            )
        case _UsbPhase.ALIGN:
            state = replace(
                state,
                phase=phase,
                ee_pos=_grasp_target(
                    state.slot_x,
                    state.slot_y,
                    state.slot_z + _SLOT_HEIGHT + _GRASP_HEIGHT,
                ),
            )
        case _UsbPhase.CORRECT_1 | _UsbPhase.CORRECT_2:
            # Null the measured USB-to-hand offset against the live opening.
            (gx, gy, _), _ = world.object_pose("usb")
            (ex, ey, ez), _ = world.arm.ee_pose()
            insert_x, insert_y = ex + (state.slot_x - gx), ey + (state.slot_y - gy)
            state = replace(
                state,
                phase=phase,
                insert_x=insert_x,
                insert_y=insert_y,
                ee_pos=(insert_x, insert_y, ez),
            )
        case _UsbPhase.INSERT:
            insert_distance = world.object_pose("usb")[0][2] - (
                state.slot_z + _INSERT_Z
            )
            ee_z = world.arm.ee_pose()[0][2]
            state = replace(
                state,
                phase=phase,
                ee_pos=(state.insert_x, state.insert_y, ee_z - insert_distance),
            )
        case _UsbPhase.OPEN:
            state = replace(
                state,
                phase=phase,
                gripper_qpos=world.arm.gripper.open_qpos,
                grip_prev=world.arm.gripper.joint_pos(),
                grip_stall=0,
            )
        case _UsbPhase.RETRACT:
            state = replace(
                state,
                phase=phase,
                ee_pos=(state.insert_x, state.insert_y, _EE_RETRACT_Z),
            )
        case _UsbPhase.DONE:
            return None
        case _UsbPhase.INIT:
            raise AssertionError("INIT enters through step(), not _enter()")
    if phase in _MOVE_PHASES:
        (cx, cy, cz), _ = world.arm.ee_pose()
        start = (cx, cy, cz)
        distance = math.dist(start, state.ee_pos)
        ticks = max(1, math.ceil(distance / _PHASE_MAX_STEP_M[phase]))
        state = replace(
            state,
            start_pos=start,
            move_ticks=ticks,
            move_tick=1,
            settle=0,
            settle_ticks=0,
        )
        return state, _action(
            state, _lerp_pos(start, state.ee_pos, 1.0 / ticks)
        )
    return state, _action(state)


class _UsbExpert:
    """Per-tick USB phase machine.

    Phase order: PREGRASP -> DESCEND -> CLOSE -> LIFT -> CARRY -> ALIGN ->
    CORRECT_1 -> CORRECT_2 -> INSERT -> OPEN -> RETRACT -> DONE.  Move phases
    carry a distance-derived interpolation and settle window; CLOSE/OPEN wait
    for the jaw to reach its target or stall.  The collector applies each
    Action and owns the only ``scene.step()``.
    """

    def init(self) -> _UsbState:
        return _UsbState(phase=_UsbPhase.INIT)

    def step(
        self, world: ExpertWorld, state: _UsbState
    ) -> tuple[_UsbState, Action] | None:
        if state.phase is _UsbPhase.INIT:
            pos, quat = world.arm.ee_pose()
            state = replace(
                state,
                ee_pos=pos,
                ee_quat=quat,
                gripper_qpos=world.arm.gripper.joint_pos(),
            )
            return _enter(_UsbPhase.PREGRASP, world, state)
        if state.phase is _UsbPhase.DONE:
            return None
        if state.phase in _MOVE_PHASES:
            if state.move_tick < state.move_ticks:
                state = replace(state, move_tick=state.move_tick + 1)
                alpha = state.move_tick / state.move_ticks
                return state, _action(
                    state, _lerp_pos(state.start_pos, state.ee_pos, alpha)
                )
            (ex, ey, ez), _ = world.arm.ee_pose()
            converged = (
                math.dist((ex, ey, ez), state.ee_pos) <= _MOVE_TOLERANCE_M
            )
            settled = state.settle + 1 if converged else 0
            state = replace(
                state,
                settle=settled,
                settle_ticks=state.settle_ticks + 1,
            )
            if (
                settled >= _MOVE_SETTLE_TICKS
                or state.settle_ticks >= _MOVE_SETTLE_MAX_TICKS
            ):
                return _enter(_NEXT[state.phase], world, state)
            return state, _action(state)
        qpos = world.arm.gripper.joint_pos()
        if abs(qpos - state.gripper_qpos) <= _GRIP_TOLERANCE:
            return _enter(_NEXT[state.phase], world, state)
        stalled = (
            state.grip_stall + 1
            if abs(qpos - state.grip_prev) <= _GRIP_STALL_EPS
            else 0
        )
        state = replace(state, grip_prev=qpos, grip_stall=stalled)
        if stalled >= _GRIP_STALL_TICKS:
            return _enter(_NEXT[state.phase], world, state)
        return state, _action(state)


def _success(world: ExpertWorld) -> Iterator[bool]:
    """Per-tick USB success generator.

    The collector creates this iterator once per episode and advances it after
    every physics step.  Each yield reports the current verdict; the rollout
    stops on the first True.  A local slot can hold simple state here if the
    gate ever needs it (this gate does not).
    """
    while True:
        (ux, uy, uz), usb_quat = world.object_pose("usb")
        (sx, sy, sz), _ = world.object_pose("slot")
        dx, dy, dz = ux - sx, uy - sy, uz - (sz + _INSERT_Z)
        x, y = usb_quat[0], usb_quat[1]
        # USB local +z (the plug/body axis) in world coordinates, XYZW quaternion.
        axis_z_world = 1 - 2 * (x * x + y * y)
        yield (
            abs(dx) < _SUCCESS_XY_TOLERANCE
            and abs(dy) < _SUCCESS_XY_TOLERANCE
            and abs(dz) < _SUCCESS_Z_TOLERANCE
            and axis_z_world > _SUCCESS_Z_AXIS_DOT
            and world.arm.ee_pose()[0][2] > _SUCCESS_EE_Z
        )


def _randomize(task: Task, seed: int) -> EpisodeRandomization:
    """Sample only scene pose P0: slot x/y, with USB rigidly following start_slot.

    z and quaternions stay nominal.  The USB offset is measured from the task's
    nominal actors, so this does not encode the legacy ``USB_INSERT_Z`` constant
    independently of the current asset layout.
    """
    actors = {actor.prim_path: actor for actor in task.actors}
    start = actors["/World/actors/start_slot"]
    target = actors["/World/actors/slot"]
    usb = actors["/World/actors/usb"]

    rng = random.Random(seed)
    half = _DR_SLOT_XY_HALF_M
    start_dx = rng.uniform(-half, half)
    start_dy = rng.uniform(-half, half)
    target_dx = rng.uniform(-half, half)
    target_dy = rng.uniform(-half, half)

    usb_dx = usb.pos[0] - start.pos[0]
    usb_dy = usb.pos[1] - start.pos[1]
    overrides = {
        start.prim_path: ActorPose(
            (start.pos[0] + start_dx, start.pos[1] + start_dy, start.pos[2]),
            start.rot,
        ),
        target.prim_path: ActorPose(
            (target.pos[0] + target_dx, target.pos[1] + target_dy, target.pos[2]),
            target.rot,
        ),
        usb.prim_path: ActorPose(
            (
                start.pos[0] + start_dx + usb_dx,
                start.pos[1] + start_dy + usb_dy,
                usb.pos[2],
            ),
            usb.rot,
        ),
    }
    return EpisodeRandomization(
        seed=seed,
        instruction=task.instruction,
        actor_poses=overrides,
        expert_params={},
        metadata={
            "dr.version": "insert_usb.scene_pose.v1",
            "dr.slot_xy_half_m": half,
        },
    )


InsertUsb = Task(
    name="insert_usb",
    embodiment=FRANKA_PANDA_GSMINI,
    sensors=gsmini_pair(streams=("rgb", "depth", "marker_rgb")),
    cameras=(
        CameraSpec(
            prim_path="/World/Camera",
            role="head",
            width=480,
            height=270,
            pos=(0.66, 0.14, 0.11),
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
        Actor(
            "insert_usb/slot_start.usd",
            "/World/actors/start_slot",
            pos=(0.4, 0.0, 0.002),
            dynamic=False,
            uipc_fixed_boxes=_START_SUPPORT_BOXES,
            authority="physx",
        ),
        Actor(
            "insert_usb/slot_target.usd",
            "/World/actors/slot",
            pos=(0.52, 0.0, 0.002),
            dynamic=False,
            uipc_fixed_mesh="insert_usb/slot_target_tet.npz",
            authority="physx",
        ),
        # Keep the plug tip >1 mm above the fixed support floor: UIPC's init
        # sanity check rejects any initial overlap.  The expert reads the live
        # USB pose, so this is not a hard-coded grasp height.
        Actor(
            "insert_usb/usb.usd",
            "/World/actors/usb",
            pos=(0.4, 0.0, 0.0085),
            authority="uipc",
            tactile_mesh="insert_usb/usb_tet.npz",
            contact_object=True,
        ),
    ),
    instruction="insert the usb into the slot",
    success=_success,
    # Per-tick scripted policy.  Phase order: PREGRASP -> DESCEND -> CLOSE ->
    # LIFT -> CARRY -> ALIGN -> CORRECT_1 -> CORRECT_2 -> INSERT -> OPEN ->
    # RETRACT -> DONE.  Each Action carries the EE pose and gripper target;
    # success is a per-tick generator and the first True stops the rollout.
    expert=_UsbExpert(),
    required_capabilities=frozenset({"binary_close"}),
    randomizer=_randomize,
)
