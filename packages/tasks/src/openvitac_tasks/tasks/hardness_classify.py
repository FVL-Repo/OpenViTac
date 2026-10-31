"""HC (paper abbreviation): classify the block hardness by touch, then place it on
the matching plate. Both blocks share one usd (``red_smooth_cuboid``).
TODO(task-setup): the old-stack UIPC compliance override (youngs 0.1, poisson 0.45,
density 300 vs the hard block's density 1e3) is not migrated yet. Standby poses are
hard y=+1.0, soft y=-1.0; the per-episode reset that moves the chosen block to
(0.38, 0, 0.002) is also not implemented yet (``Task.setup`` currently only returns
the instruction).
"""
from __future__ import annotations

from collections.abc import Iterator

from openvitac_spec import (
    FRANKA_PANDA_GSMINI,
    Actor,
    CameraSpec,
    PinholeIntrinsics,
    gsmini_pair,
)

from ..base import Task
from ..expert import Action, ExpertWorld

_HEAD_CAM_QUAT_XYZW = (0.5, 0.5, 0.5, 0.5)  # 2.x camera offset was WXYZ


class _TodoExpert:
    """Explicit stub: the scripted policy for this task is not yet written."""

    def init(self) -> None:
        raise NotImplementedError("expert not yet written for hardness_classify")

    def step(
        self, world: ExpertWorld, state: None
    ) -> tuple[None, Action] | None:
        raise NotImplementedError("expert not yet written for hardness_classify")


def _todo_success(world: ExpertWorld) -> Iterator[bool]:
    """Explicit stub: the success check for this task is not yet written."""
    raise NotImplementedError("success check not yet written for hardness_classify")
    yield False


HardnessClassify = Task(
    name="hardness_classify",
    embodiment=FRANKA_PANDA_GSMINI,
    sensors=gsmini_pair(),
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
            authority="physx",
        ),
        Actor(
            "roughness_task/blue_square_plate.usd",
            "/World/actors/blue_plate",
            pos=(0.45, 0.08, 0.002),
            authority="physx",
        ),
        Actor(
            "roughness_task/yellow_square_plate.usd",
            "/World/actors/yellow_plate",
            pos=(0.45, -0.08, 0.002),
            authority="physx",
        ),
        Actor(
            "roughness_task/red_smooth_cuboid.usd",
            "/World/actors/hard_red_block",
            pos=(0.38, 1.0, 0.002),
            authority="physx",
        ),
        Actor(
            "roughness_task/red_smooth_cuboid.usd",
            "/World/actors/soft_red_block",
            pos=(0.38, -1.0, 0.002),
            authority="physx",
        ),
    ),
    instruction=(
        "Classify the block hardness by touch: place the soft red block on the blue "
        "plate and the hard red block on the yellow plate."
    ),
    expert=_TodoExpert(),
    success=_todo_success,
    required_capabilities=frozenset({"binary_close"}),
)
