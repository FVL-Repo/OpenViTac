"""Robot embodiment presets: arm + gripper + USD assembly, with no sensor choice.

``FRANKA_PANDA_GSMINI`` is a whole-machine USD with the GsMini finger pads already
authored.  ``FRANKA_ROBOTIQ`` is the whole-machine Robotiq asset.  The ``asset`` field
can also describe a split ``arm_plus_tool`` assembly, but the current runtime only
implements ``monolithic`` and fails loud otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .types import IDENTITY_TRANSFORM, Transform

type ArmKind = Literal["franka_panda"]
type GripperKind = Literal["panda_parallel", "robotiq_85"]
type GripperCapability = Literal[
    "binary_close", "adaptive_close", "percent_open", "contact_stop"
]


@dataclass(frozen=True)
class ArmSpec:
    """The arm's semantic joints and end-effector link."""

    kind: ArmKind
    ee_link: str
    arm_joints: tuple[str, ...]
    actuator_profile: str


@dataclass(frozen=True)
class GripperSpec:
    """The gripper's commandable joints, finger bodies, and declared capabilities."""

    kind: GripperKind
    joints: tuple[str, ...]
    open_value: float
    close_value: float
    contact_bodies: tuple[str, ...]  # (left, right)
    capabilities: frozenset[GripperCapability]
    tool_offset: Transform = IDENTITY_TRANSFORM


@dataclass(frozen=True)
class RobotAssetRef:
    """USD asset reference; ``arm_plus_tool`` is reserved but not implemented."""

    usd: str
    assembly: Literal["monolithic", "arm_plus_tool"] = "monolithic"
    tool_usd: str | None = None
    attach_prim: str | None = None

    def __post_init__(self) -> None:
        if not self.usd:
            raise ValueError("RobotAssetRef.usd must be non-empty")
        if self.assembly == "arm_plus_tool" and not (
            self.tool_usd and self.attach_prim
        ):
            raise ValueError(
                "arm_plus_tool requires tool_usd and attach_prim; "
                "monolithic assets leave both empty"
            )


@dataclass(frozen=True)
class EmbodimentSpec:
    """One robot + gripper preset; a task may override the nominal joint pose."""

    name: str
    arm: ArmSpec
    gripper: GripperSpec
    asset: RobotAssetRef
    default_joint_pos: dict[str, float] = field(
        default_factory=dict[str, float]
    )
    camera_mounts: dict[str, str] = field(default_factory=dict[str, str])

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("EmbodimentSpec.name must be non-empty")

    def validate_supported(self) -> None:
        """Raise for authored-but-not-implemented assembly paths."""
        if self.asset.assembly != "monolithic":
            raise NotImplementedError(
                f"embodiment {self.name!r}: {self.asset.assembly!r} assembly is not "
                "implemented; only monolithic whole-machine USDs are supported"
            )


@dataclass(frozen=True)
class RobotAssetManifest:
    """Expected body/joint/camera names for a monolithic asset (build-time check)."""

    usd: str
    bodies: tuple[str, ...]
    joints: tuple[str, ...]
    cameras: tuple[str, ...] = ()


_PANDA_ARM = ArmSpec(
    kind="franka_panda",
    ee_link="panda_hand",
    arm_joints=tuple(f"panda_joint{i}" for i in range(1, 8)),
    actuator_profile="franka_high_pd",
)

_PANDA_PARALLEL = GripperSpec(
    kind="panda_parallel",
    joints=("panda_finger_joint1", "panda_finger_joint2"),
    open_value=0.039,
    close_value=0.0,
    contact_bodies=("panda_leftfinger", "panda_rightfinger"),
    capabilities=frozenset({"binary_close"}),
)

# 2F-85 finger_joint spans 0..45 deg: at 0 rad the linkages are folded and the pads
# meet (closed), at 45 deg the pads are ~79 mm apart (open). Measured on the loaded
# articulation: inner-finger origins 0.2 mm apart at 0 rad, 78.8 mm at 0.785 rad.
_ROBOTIQ_85 = GripperSpec(
    kind="robotiq_85",
    joints=("finger_joint",),
    open_value=0.7853981633974483,
    close_value=0.0,
    contact_bodies=("left_inner_finger", "right_inner_finger"),
    capabilities=frozenset({"binary_close", "adaptive_close", "percent_open"}),
)

FRANKA_PANDA_GSMINI = EmbodimentSpec(
    name="franka_panda_gsmini",
    arm=_PANDA_ARM,
    gripper=_PANDA_PARALLEL,
    asset=RobotAssetRef("embodiment/franka_gsmini.usda"),
    default_joint_pos={
        "panda_joint1": -0.010809095,
        "panda_joint2": 0.096037410,
        "panda_joint3": 0.000734462,
        "panda_joint4": -2.433035851,
        "panda_joint5": 0.035354517,
        "panda_joint6": 2.500859022,
        "panda_joint7": 0.741,
        "panda_finger.*": 0.02,
    },
    camera_mounts={"wrist": "/World/Robot/WristCamera"},
)

_RPANDA_ARM = ArmSpec(
    kind="franka_panda",
    ee_link="panda_link8",
    arm_joints=tuple(f"panda_joint{i}" for i in range(1, 8)),
    actuator_profile="franka_high_pd",
)

FRANKA_ROBOTIQ = EmbodimentSpec(
    name="franka_robotiq",
    arm=_RPANDA_ARM,
    gripper=_ROBOTIQ_85,
    asset=RobotAssetRef("embodiment/franka_robotiq.usda"),
    default_joint_pos={
        "panda_joint1": 0.0,
        "panda_joint2": 0.0,
        "panda_joint3": 0.0,
        "panda_joint4": -2.46,
        "panda_joint5": 0.0,
        "panda_joint6": 2.5,
        "panda_joint7": 0.741,
        "finger_joint": 0.0,
    },
    camera_mounts={"wrist": "/World/Robot/WristCamera"},
)
