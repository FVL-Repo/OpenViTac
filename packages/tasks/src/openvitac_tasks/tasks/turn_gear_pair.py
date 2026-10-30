"""GA (paper abbreviation): pick up the red gear, assemble it on the base, and turn
the pair. The base is a fixed fixture and the blue gear rides a revolute joint; the
red gear stays a free rigid body until assembly. Gears mesh at a 0.074 m centre
distance with the blue gear authored 6 deg out of phase.

Gear geometry stays in module constants because ``Actor`` has no fields for
derived positions.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

from openvitac_spec import (
    FRANKA_ROBOTIQ,
    Actor,
    CameraSpec,
    PinholeIntrinsics,
    RevoluteJoint,
    xense_pair,
)

from ..base import Task
from ..expert import Action, ExpertWorld

_HEAD_CAM_QUAT_XYZW = (0.310977, 0.562556, 0.670428, 0.370608)

# Assembly geometry: base centre x=0.45, gears centred +/-0.037 about it.
_GEAR_CENTER_DISTANCE = 0.074
_GEAR_PHASE_OFFSET_DEG = 6.0
_BLUE_GEAR_ROT_XYZW = (
    0.0,
    0.0,
    math.sin(math.radians(_GEAR_PHASE_OFFSET_DEG) / 2),
    math.cos(math.radians(_GEAR_PHASE_OFFSET_DEG) / 2),
)
_BASE_POS = (0.45, 0.0, 0.002)
_RED_GEAR_PICK_POS = (0.45, -0.13, 0.002)
_BLUE_GEAR_POS = (0.45 + _GEAR_CENTER_DISTANCE * 0.5, 0.0, 0.010)
# Blue gear centre in the base frame; the joint frame is its shaft.
_BLUE_GEAR_JOINT_POS = (0.037, 0.0, 0.008)
_GEAR_JOINT_DAMPING = 0.01  # quells free-spin jitter without resisting the turn


class _TodoExpert:
    """Explicit stub: the scripted policy for this task is not yet written."""

    def init(self) -> None:
        raise NotImplementedError("expert not yet written for turn_gear_pair")

    def step(
        self, world: ExpertWorld, state: None
    ) -> tuple[None, Action] | None:
        raise NotImplementedError("expert not yet written for turn_gear_pair")


def _todo_success(world: ExpertWorld) -> Iterator[bool]:
    """Explicit stub: the success check for this task is not yet written."""
    raise NotImplementedError("success check not yet written for turn_gear_pair")
    yield False


TurnGearPair = Task(
    name="turn_gear_pair",
    embodiment=FRANKA_ROBOTIQ,
    sensors=xense_pair(),
    cameras=(
        CameraSpec(
            prim_path="/World/Camera",
            role="head",
            width=480,
            height=270,
            pos=(0.74, 0.13, 0.15),
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
        # Fixture, like the PD cabinet: the joint makes the base immovable without a
        # density proxy.
        Actor(
            "turn_gear_pair/gear_pair_base.usd",
            "/World/actors/base",
            pos=_BASE_POS,
            dynamic=False,
            authority="physx",
        ),
        Actor(
            "turn_gear_pair/gear_pair_red.usd",
            "/World/actors/red_gear",
            pos=_RED_GEAR_PICK_POS,
            authority="physx",
        ),
        Actor(
            "turn_gear_pair/gear_pair_blue.usd",
            "/World/actors/blue_gear",
            pos=_BLUE_GEAR_POS,
            rot=_BLUE_GEAR_ROT_XYZW,
            authority="physx",
        ),
    ),
    # Static base -> body0 falls to the world; pos is the blue gear's shaft in the
    # base frame. Free spin (no limits), only light damping.
    joints=(
        RevoluteJoint(
            prim_path="/World/joints/blue_gear",
            parent="/World/actors/base",
            child="/World/actors/blue_gear",
            pos=_BLUE_GEAR_JOINT_POS,
            axis="Z",
            limits=None,
            damping=_GEAR_JOINT_DAMPING,
        ),
    ),
    instruction=(
        "Pick up the red gear, assemble it onto the base, and rotate it to turn the "
        "gear pair."
    ),
    expert=_TodoExpert(),
    success=_todo_success,
    required_capabilities=frozenset({"adaptive_close", "percent_open"}),
)
