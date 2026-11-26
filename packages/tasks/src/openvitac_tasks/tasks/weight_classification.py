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
from ..policy import CloseTo, MoveToolTo, OpenGripper, Point, Policy, Sequence

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
# and 8 mm plate, as in roughness_classification).
_TABLE_SUPPORT_BOX = ((0.0, 0.0, 0.0013), (0.3, 0.3, 0.0012))
_PLATE_SUPPORT_BOX = ((0.0, 0.0, 0.0033), (0.04, 0.04, 0.0027))
_GRASP_HEIGHT = 0.035
# The gel is the contact surface on the UIPC path, so a blind close crushes it;
# the commanded close comes from the block width (15.00 mm, black_smooth_cuboid
# bounds +/-0.0075 m) through the shared close law.
_BLOCK_WIDTH = 0.01500
_APPROACH_CLEARANCE = 0.05
# The approach arrives horizontally over a standing 80 mm block.  The pads are
# taller than _APPROACH_CLEARANCE leaves room for, so their lower edge would be
# inside the block (see roughness_guided_regrasp); clear the block top instead.
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


_MOVE_TICKS = 60
_SETTLE_TICKS = 60


def _move(
    target: Point,
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


class _WeightPolicy(Policy):
    """Grasp the selected block, place it on its plate, release, retract.

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
        # Parallel the pad plane to the selected block's +/-y face; see
        # roughness_classification for the lever-arm rationale.
        _, bquat = world.object_pose(block)
        quat = align_pad_to_face(
            world.arm.tool, quat, quat_rotate(bquat, (0.0, 1.0, 0.0))
        )

        # Every goal below is a *tool-center* point: the asset's pad midpoint,
        # which ``_move`` turns into the EE pose the IK solver wants.  The goals
        # are read on the tick their move starts, like the phase machine did.
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
            _move(grasp, quat, clearance=_TRANSIT_CLEARANCE, ticks=120),  # PREGRASP
            _move(grasp, quat),  # DESCEND
            CloseTo(_BLOCK_WIDTH, hold_ticks=60),  # CLOSE
            _move(lift, quat, ticks=90),  # LIFT
            _move(place, quat, clearance=_APPROACH_CLEARANCE, ticks=120),  # CARRY
            _move(place, quat, ticks=90),  # PLACE
            OpenGripper(hold_ticks=60),  # OPEN
            _move(place, quat, clearance=_APPROACH_CLEARANCE, ticks=90),  # RETRACT
        )


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
            + grasp_depth(world.arm.tool)
            + _APPROACH_CLEARANCE
        )
        retracted = (
            abs(ex - tx) < _SUCCESS_XY_TOLERANCE
            and abs(ey - ty) < _SUCCESS_XY_TOLERANCE
            and ez >= retract_z - _SUCCESS_Z_TOLERANCE
        )
        yield seated and released and retracted


WeightClassification = Task(
    name="weight_classification",
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
    expert=_WeightPolicy,
    success=_success,
    randomizer=_randomize,
    required_capabilities=frozenset({"binary_close"}),
)
