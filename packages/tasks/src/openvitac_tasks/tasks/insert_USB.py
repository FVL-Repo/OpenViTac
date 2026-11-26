"""In-USB: insert the USB into the slot."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterator

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
from ..motion import ee_for_tool_center
from ..policy import Command, Policy, Sequence

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
    world: ExpertWorld,
    quat: Quat,
    grasp_x: float,
    grasp_y: float,
    grasp_z: float,
    *,
    clearance: float = 0.0,
) -> Vec3:
    """Commanded hand pose whose pad midpoint is at the grasp point.

    The hand-to-pad offset is the asset's (``ToolGeometry.center_in_ee``), rotated
    with the hand; nothing here is a per-gripper calibration.
    """
    return ee_for_tool_center(
        (grasp_x, grasp_y, grasp_z), quat, world.arm.tool, clearance=clearance
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

# Cartesian speed per tick.  Free-space transport stays fast; the final
# align/null/insert sequence keeps the old slow contact approach.
_PHASE_MAX_STEP_M: dict[str, float] = {
    "PREGRASP": 1e-3,
    "DESCEND": 1e-3,
    "LIFT": 1e-3,
    "CARRY": 1e-3,
    "ALIGN": 1e-4,
    "CORRECT_1": 1e-4,
    "CORRECT_2": 1e-4,
    "INSERT": 1e-4,
    "RETRACT": 1e-3,
}


def _lerp_pos(a: Vec3, b: Vec3, alpha: float) -> Vec3:
    return (
        a[0] + (b[0] - a[0]) * alpha,
        a[1] + (b[1] - a[1]) * alpha,
        a[2] + (b[2] - a[2]) * alpha,
    )


class _UsbPoints:
    """The coordinates each phase samples off the world for the later ones."""

    def __init__(self) -> None:
        self.grasp: Vec3 = (0.0, 0.0, 0.0)
        self.slot: Vec3 = (0.0, 0.0, 0.0)
        self.insert_xy: tuple[float, float] = (0.0, 0.0)

    def pregrasp(self, world: ExpertWorld, cmd: Command) -> Vec3:
        (ux, uy, uz), _ = world.object_pose("usb")
        self.grasp = (ux, uy, uz + _GRASP_HEIGHT)
        gx, gy, gz = self.grasp
        return _grasp_target(
            world, cmd.ee_quat, gx, gy, gz, clearance=_APPROACH_CLEARANCE
        )

    def descend(self, world: ExpertWorld, cmd: Command) -> Vec3:
        gx, gy, gz = self.grasp
        return _grasp_target(world, cmd.ee_quat, gx, gy, gz)

    def lift(self, world: ExpertWorld, cmd: Command) -> Vec3:
        gx, gy, gz = self.grasp
        return _grasp_target(world, cmd.ee_quat, gx, gy, gz + _LIFT_HEIGHT)

    def carry(self, world: ExpertWorld, cmd: Command) -> Vec3:
        (sx, sy, sz), _ = world.object_pose("slot")
        self.slot = (sx, sy, sz)
        return _grasp_target(
            world,
            cmd.ee_quat,
            sx,
            sy,
            sz + _SLOT_HEIGHT + _GRASP_HEIGHT,
            clearance=_APPROACH_CLEARANCE,
        )

    def align(self, world: ExpertWorld, cmd: Command) -> Vec3:
        sx, sy, sz = self.slot
        return _grasp_target(
            world, cmd.ee_quat, sx, sy, sz + _SLOT_HEIGHT + _GRASP_HEIGHT
        )

    def correct(self, world: ExpertWorld, cmd: Command) -> Vec3:
        # Null the measured USB-to-hand offset against the live opening.
        (gx, gy, _), _ = world.object_pose("usb")
        (ex, ey, ez), _ = world.arm.ee_pose()
        self.insert_xy = (ex + (self.slot[0] - gx), ey + (self.slot[1] - gy))
        return (self.insert_xy[0], self.insert_xy[1], ez)

    def insert(self, world: ExpertWorld, cmd: Command) -> Vec3:
        insert_distance = world.object_pose("usb")[0][2] - (
            self.slot[2] + _INSERT_Z
        )
        ee_z = world.arm.ee_pose()[0][2]
        return (self.insert_xy[0], self.insert_xy[1], ee_z - insert_distance)

    def retract(self, world: ExpertWorld, cmd: Command) -> Vec3:
        return (self.insert_xy[0], self.insert_xy[1], _EE_RETRACT_Z)


class _UsbMove:
    """One move phase: distance-derived interpolation, then a settle window.

    ``target`` is read on the phase's first tick; its tick count comes from the
    measured distance and the phase's own per-tick Cartesian speed.  After the
    interpolation the goal is re-commanded until the hand has been within
    tolerance for ``_MOVE_SETTLE_TICKS`` ticks, capped at
    ``_MOVE_SETTLE_MAX_TICKS``.
    """

    def __init__(
        self,
        target: Callable[[ExpertWorld, Command], Vec3],
        *,
        max_step: float,
    ) -> None:
        self.target = target
        self.max_step = max_step
        self.finished = False
        self._goal: Vec3 | None = None
        self._start: Vec3 = (0.0, 0.0, 0.0)
        self._ticks = 1
        self._tick = 0
        self._settle = 0
        self._settle_ticks = 0

    def step(self, world: ExpertWorld, cmd: Command) -> Action | None:
        if self.finished:
            raise RuntimeError("usb move already finished")
        goal = self._goal
        if goal is None:
            goal = self.target(world, cmd)
            self._goal = goal
            self._start = world.arm.ee_pose()[0]
            distance = math.dist(self._start, goal)
            self._ticks = max(1, math.ceil(distance / self.max_step))
            self._tick = 1
            return Action(
                _lerp_pos(self._start, goal, 1.0 / self._ticks),
                cmd.ee_quat,
                cmd.gripper_qpos,
            )
        if self._tick < self._ticks:
            self._tick += 1
            return Action(
                _lerp_pos(self._start, goal, self._tick / self._ticks),
                cmd.ee_quat,
                cmd.gripper_qpos,
            )
        (ex, ey, ez), _ = world.arm.ee_pose()
        converged = math.dist((ex, ey, ez), goal) <= _MOVE_TOLERANCE_M
        self._settle = self._settle + 1 if converged else 0
        self._settle_ticks += 1
        if (
            self._settle >= _MOVE_SETTLE_TICKS
            or self._settle_ticks >= _MOVE_SETTLE_MAX_TICKS
        ):
            self.finished = True
            return None
        return Action(goal, cmd.ee_quat, cmd.gripper_qpos)


class _UsbGrip:
    """Hold the command until the jaw reaches its target or stops moving."""

    def __init__(self) -> None:
        self.finished = False
        self._qpos: float | None = None
        self._prev = 0.0
        self._stall = 0

    def qpos(self, world: ExpertWorld) -> float:
        """The gripper target this phase commands; read once, on its first tick."""
        raise NotImplementedError

    def step(self, world: ExpertWorld, cmd: Command) -> Action | None:
        if self.finished:
            raise RuntimeError("usb grip already finished")
        target = self._qpos
        if target is None:
            target = self.qpos(world)
            self._qpos = target
            self._prev = world.arm.gripper.joint_pos()
            return Action(cmd.ee_pos, cmd.ee_quat, target)
        qpos = world.arm.gripper.joint_pos()
        if abs(qpos - target) <= _GRIP_TOLERANCE:
            self.finished = True
            return None
        self._stall = (
            self._stall + 1 if abs(qpos - self._prev) <= _GRIP_STALL_EPS else 0
        )
        self._prev = qpos
        if self._stall >= _GRIP_STALL_TICKS:
            self.finished = True
            return None
        return Action(cmd.ee_pos, cmd.ee_quat, target)


class _GripClose(_UsbGrip):
    """Command the embodiment's fully closed gripper target."""

    def qpos(self, world: ExpertWorld) -> float:
        return world.arm.gripper.close_qpos


class _GripOpen(_UsbGrip):
    """Command the embodiment's fully open gripper target."""

    def qpos(self, world: ExpertWorld) -> float:
        return world.arm.gripper.open_qpos


class _UsbPolicy(Policy):
    """Insert the USB: grasp, lift, align over the slot, insert, release.

    The step sequence spells out the legacy phase order -- PREGRASP, DESCEND,
    CLOSE, LIFT, CARRY, ALIGN, CORRECT_1, CORRECT_2, INSERT, OPEN, RETRACT --
    with each move phase keeping its own per-tick Cartesian speed.
    """

    def plan(self, world: ExpertWorld) -> Sequence:
        points = _UsbPoints()
        return Sequence(
            _UsbMove(points.pregrasp, max_step=_PHASE_MAX_STEP_M["PREGRASP"]),
            _UsbMove(points.descend, max_step=_PHASE_MAX_STEP_M["DESCEND"]),
            _GripClose(),
            _UsbMove(points.lift, max_step=_PHASE_MAX_STEP_M["LIFT"]),
            _UsbMove(points.carry, max_step=_PHASE_MAX_STEP_M["CARRY"]),
            _UsbMove(points.align, max_step=_PHASE_MAX_STEP_M["ALIGN"]),
            _UsbMove(points.correct, max_step=_PHASE_MAX_STEP_M["CORRECT_1"]),
            _UsbMove(points.correct, max_step=_PHASE_MAX_STEP_M["CORRECT_2"]),
            _UsbMove(points.insert, max_step=_PHASE_MAX_STEP_M["INSERT"]),
            _GripOpen(),
            _UsbMove(points.retract, max_step=_PHASE_MAX_STEP_M["RETRACT"]),
        )


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
            "dr.version": "insert_USB.scene_pose.v1",
            "dr.slot_xy_half_m": half,
        },
    )


InsertUsb = Task(
    name="insert_USB",
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
            "insert_USB/slot_start.usd",
            "/World/actors/start_slot",
            pos=(0.4, 0.0, 0.002),
            dynamic=False,
            uipc_fixed_boxes=_START_SUPPORT_BOXES,
            authority="physx",
        ),
        Actor(
            "insert_USB/slot_target.usd",
            "/World/actors/slot",
            pos=(0.52, 0.0, 0.002),
            dynamic=False,
            uipc_fixed_mesh="insert_USB/slot_target_tet.npz",
            authority="physx",
        ),
        # Keep the plug tip >1 mm above the fixed support floor: UIPC's init
        # sanity check rejects any initial overlap.  The expert reads the live
        # USB pose, so this is not a hard-coded grasp height.
        Actor(
            "insert_USB/usb.usd",
            "/World/actors/usb",
            pos=(0.4, 0.0, 0.0085),
            authority="uipc",
            tactile_mesh="insert_USB/usb_tet.npz",
            contact_object=True,
        ),
    ),
    instruction="insert the usb into the slot",
    success=_success,
    # Per-tick scripted policy.  Phase order: PREGRASP -> DESCEND -> CLOSE ->
    # LIFT -> CARRY -> ALIGN -> CORRECT_1 -> CORRECT_2 -> INSERT -> OPEN ->
    # RETRACT -> DONE.  Each Action carries the EE pose and gripper target;
    # success is a per-tick generator and the first True stops the rollout.
    expert=_UsbPolicy,
    required_capabilities=frozenset({"binary_close"}),
    randomizer=_randomize,
)
