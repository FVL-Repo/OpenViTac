"""RC (paper abbreviation): classify the block surface by touch, then place it on the
matching plate.  The smooth and rough variants are distinct usd files.

Per episode, ``seed`` samples one label.  That block starts at ``(0.38, 0, 0.002)``;
the other block rests on the table at ``(0.30, 0.25, 0.002)``.  Legacy mapping:
smooth -> blue plate, rough -> yellow plate.
"""
from __future__ import annotations

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
from ..expert import ExpertWorld, Quat, Vec3
from ..motion import align_pad_to_face, grasp_depth, quat_rotate
from ..policy import CloseTo, MoveToolTo, OpenGripper, Policy, Sequence

_HEAD_CAM_QUAT_XYZW = (0.5, 0.5, 0.5, 0.5)  # 2.x camera offset was WXYZ

_LABELS = ("smooth", "rough")
_BLOCK_BY_LABEL = {"smooth": "smooth_block", "rough": "rough_block"}
_PLATE_BY_LABEL = {"smooth": "blue_plate", "rough": "yellow_plate"}
# The table top is at 2.5 mm (a 1 m cube scaled to 5 mm in the authored USD).
# Spawn 1.5 mm above it: UIPC rejects any initial overlap with the fixed
# support body, so the block must start clear and settle onto it.
_BLOCK_START_POS = (0.38, 0.0, 0.004)
_BLOCK_STANDBY_POS = (0.30, 0.25, 0.004)
# UIPC fixed supports mirror the PhysX table and the matching plate: box
# centers/half extents are actor-local metres and their top faces sit exactly
# on the authored surfaces (world z = 2.5 mm table, 8 mm plate).  Fixed bodies
# must not overlap each other at init, so the table box starts just above the
# UIPC floor (z = 0) and the plate box just above the table top.
_TABLE_SUPPORT_BOX = ((0.0, 0.0, 0.0013), (0.3, 0.3, 0.0012))
_PLATE_SUPPORT_BOX = ((0.0, 0.0, 0.0033), (0.04, 0.04, 0.0027))
_GRASP_HEIGHT = 0.035
# Both blocks are UIPC bodies, so the commanded close comes from the grasped
# block's own width through the shared close law (motion.pad_close_q).  The
# widths are the asset bounds (black_smooth_cuboid.usd +/-0.00750 m,
# black_rough_cuboid.usd +/-0.00794 m).  The gel is the contact surface on the
# UIPC path, so a blind close would crush it.
_BLOCK_WIDTH_BY_NAME = {"smooth_block": 0.01500, "rough_block": 0.01588}
# Neither cuboid USD authors a mass density; 1e3 kg/m^3 is the PhysX default
# and the UIPC bridge default, so the explicit value only pins the path.
_BLOCK_DENSITY = 1.0e3
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
        contact_object=f"/World/actors/{selected}",
        metadata={"roughness_label": label, "target_plate": target},
    )


_MOVE_TICKS = 60
_SETTLE_TICKS = 60


def _move(
    target: Vec3 | Callable[[], Vec3],
    quat: Quat,
    *,
    clearance: float = 0.0,
    ticks: int = _MOVE_TICKS,
) -> MoveToolTo:
    """One legacy move: ``ticks`` to the goal, then the task's settle re-command."""
    return MoveToolTo(
        target,
        quat,
        clearance=clearance,
        ticks=ticks,
        settle_ticks=_SETTLE_TICKS,
    )


class _RoughnessPolicy(Policy):
    """Grasp the sampled block, carry it to its plate, release, retract.

    The step sequence spells out the legacy phase order -- PREGRASP, DESCEND,
    CLOSE, LIFT, CARRY, PLACE, OPEN, RETRACT -- with the legacy tick budgets and
    a 60-tick hold on each grasp and release.
    """

    def plan(self, world: ExpertWorld) -> Sequence:
        """Read the label, square the pads to the block face, chain the moves."""
        label = _label(world)
        block = _BLOCK_BY_LABEL[label]
        plate = _PLATE_BY_LABEL[label]
        _, quat = world.arm.ee_pose()
        # Parallel the pad plane to the block's +/-y face instead of inheriting
        # the home pose: the 1.28 deg error there, times the hand-to-pad lever
        # the asset declares, is millimetres of lateral grasp error at the pads.
        _, bquat = world.object_pose(block)
        quat = align_pad_to_face(
            world.arm.tool, quat, quat_rotate(bquat, (0.0, 1.0, 0.0))
        )

        # Every goal below is a *tool-center* point: the asset's pad midpoint.
        # ``_move`` turns it into the EE pose the IK solver wants.  The goals are
        # read on the tick their move starts, like the phase machine did -- the
        # block moves under the closing pads and the carry follows it.
        def grasp() -> Vec3:
            (bx, by, bz), _ = world.object_pose(block)
            return (bx, by, bz + _GRASP_HEIGHT)

        def lift() -> Vec3:
            (bx, by, bz), _ = world.object_pose(block)
            return (bx, by, bz + _GRASP_HEIGHT + _LIFT_HEIGHT)

        def place() -> Vec3:
            (tx, ty, tz), _ = world.object_pose(plate)
            return (tx, ty, tz + _TARGET_BLOCK_Z + _GRASP_HEIGHT)

        return Sequence(
            _move(grasp, quat, clearance=_APPROACH_CLEARANCE, ticks=120),  # PREGRASP
            _move(grasp, quat),  # DESCEND
            CloseTo(_BLOCK_WIDTH_BY_NAME[block]),  # CLOSE
            _move(lift, quat, ticks=90),  # LIFT
            _move(place, quat, clearance=_APPROACH_CLEARANCE, ticks=120),  # CARRY
            _move(place, quat, ticks=90),  # PLACE
            OpenGripper(),  # OPEN
            _move(place, quat, clearance=_APPROACH_CLEARANCE, ticks=90),  # RETRACT
        )


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
            + grasp_depth(world.arm.tool)
            + _APPROACH_CLEARANCE
        )
        retracted = (
            abs(ex - tx) < _SUCCESS_XY_TOLERANCE
            and abs(ey - ty) < _SUCCESS_XY_TOLERANCE
            and ez >= retract_z - _SUCCESS_Z_TOLERANCE
        )
        yield seated and released and retracted


RoughnessClassification = Task(
    name="roughness_classification",
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
        # Both blocks are UIPC bodies; the episode's contact object is the
        # selected one (see _randomize), so neither is flagged here.
        Actor(
            "roughness_task/black_smooth_cuboid.usd",
            "/World/actors/smooth_block",
            pos=_BLOCK_STANDBY_POS,
            authority="uipc",
            tactile_mesh="roughness_task/black_smooth_cuboid_tet.npz",
            density=_BLOCK_DENSITY,
        ),
        Actor(
            "roughness_task/black_rough_cuboid.usd",
            "/World/actors/rough_block",
            pos=_BLOCK_START_POS,
            authority="uipc",
            tactile_mesh="roughness_task/black_rough_cuboid_tet.npz",
            density=_BLOCK_DENSITY,
        ),
    ),
    instruction=(
        "Classify the block surface by touch: place the smooth block on the blue "
        "plate and the rough block on the yellow plate."
    ),
    expert=_RoughnessPolicy,
    success=_success,
    randomizer=_randomize,
    required_capabilities=frozenset({"binary_close"}),
)
