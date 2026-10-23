"""Pure scene content: camera/actor/joint/light specs and the top-level scene.

Task modules author these values; ``openvitac_sim.Scene`` translates them into
isaaclab cfgs only after Kit is up.  This module must remain import-safe without
isaaclab, pxr, pyuipc, numpy, or any openvitac runtime package.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from .embodiment import EmbodimentSpec
from .sensor import SensorBinding
from .types import IDENTITY_XYZW, Quat, Vec3


@dataclass(frozen=True)
class PinholeIntrinsics:
    """Pinhole intrinsics for a sim-spawned camera."""

    focal_length: float
    focus_distance: float
    horizontal_aperture: float
    clipping_range: tuple[float, float] = (0.1, 100.0)


@dataclass(frozen=True)
class CameraSpec:
    """One camera's role, resolution, extrinsics, and optional intrinsics."""

    prim_path: str
    role: str
    width: int
    height: int
    pos: Vec3 = (0.0, 0.0, 0.0)
    rot: Quat = IDENTITY_XYZW
    intrinsics: PinholeIntrinsics | None = None  # None on USD-authored cameras
    mount: str | None = None
    data_types: tuple[str, ...] = ("rgb",)


@dataclass(frozen=True)
class DepthCameraCfg:
    """An authored tactile depth camera; the sim reuses its prim (``spawn=None``)."""

    prim_path: str
    width: int
    height: int
    focal_length: float
    horizontal_aperture: float
    vertical_aperture: float
    clipping_range: tuple[float, float]
    data_type: Literal[
        "distance_to_image_plane", "distance_to_camera"
    ] = "distance_to_image_plane"


@dataclass(frozen=True)
class Actor:
    """A USD prop's spawn description.

    ``authority`` selects the solver that owns a dynamic actor: PhysX simulates the
    USD body directly, while ``uipc`` keeps a kinematic non-colliding PhysX proxy
    whose pose follows the UIPC object.  ``density`` is passed to PhysX as the
    spawned body's mass density; static and kinematic actors ignore it.

    ``uipc_fixed_boxes``/``uipc_fixed_mesh`` build a fixed UIPC collision copy for a
    static fixture; they are not a way to pin a dynamic actor.

    ``constitution`` selects the tactile material model of a dynamic UIPC object:
    ``abd`` (the default) or ``fem``.  FEM requires ``youngs_mpa`` and ``poisson``
    and is only valid on dynamic ``uipc`` actors; PhysX and static actors have no
    tactile material of their own.
    """

    usd: str
    prim_path: str
    pos: Vec3 = (0.0, 0.0, 0.0)
    rot: Quat = IDENTITY_XYZW
    dynamic: bool = True
    authority: Literal["uipc", "physx"] = "physx"
    tactile_mesh: str | None = None
    uipc_fixed_boxes: tuple[tuple[Vec3, Vec3], ...] = ()
    uipc_fixed_mesh: str | None = None
    contact_object: bool = False
    density: float | None = None  # PhysX rigid-body mass density (kg/m^3)
    constitution: Literal["abd", "fem"] = "abd"
    youngs_mpa: float | None = None  # FEM Young's modulus
    poisson: float | None = None  # FEM Poisson's ratio

    def __post_init__(self) -> None:
        if self.authority not in ("uipc", "physx"):
            raise ValueError(
                f"actor authority must be 'uipc' or 'physx', got {self.authority!r}"
            )
        if not self.dynamic and self.authority != "physx":
            raise ValueError(
                f"static actor {self.prim_path!r} must use authority='physx', "
                f"got {self.authority!r}"
            )
        if self.dynamic and (
            self.uipc_fixed_boxes or self.uipc_fixed_mesh is not None
        ):
            raise ValueError(
                f"dynamic actor {self.prim_path!r} cannot declare "
                "uipc_fixed_boxes/uipc_fixed_mesh; fixed UIPC bodies are only "
                "built for static actors"
            )
        if self.contact_object and (not self.dynamic or self.authority != "uipc"):
            raise ValueError(
                "contact_object requires a dynamic UIPC actor "
                f"({self.prim_path!r})"
            )
        if self.density is not None and self.density <= 0.0:
            raise ValueError(
                f"actor density must be positive, got {self.density!r}"
            )
        if self.constitution not in ("abd", "fem"):
            raise ValueError(
                f"actor constitution must be 'abd' or 'fem', got {self.constitution!r}"
            )
        if self.constitution == "fem":
            if not self.dynamic or self.authority != "uipc":
                raise ValueError(
                    f"actor {self.prim_path!r} with constitution='fem' must be a "
                    f"dynamic UIPC actor, got dynamic={self.dynamic!r}, "
                    f"authority={self.authority!r}"
                )
            if self.youngs_mpa is None:
                raise ValueError(
                    f"actor {self.prim_path!r} with constitution='fem' requires "
                    "youngs_mpa"
                )
            if self.poisson is None:
                raise ValueError(
                    f"actor {self.prim_path!r} with constitution='fem' requires poisson"
                )
            if self.youngs_mpa <= 0.0:
                raise ValueError("actor youngs_mpa must be positive")
            if not 0.0 <= self.poisson < 0.5:
                raise ValueError("actor poisson must be in [0, 0.5)")
        elif self.youngs_mpa is not None or self.poisson is not None:
            raise ValueError(
                "actor youngs_mpa/poisson only apply to constitution='fem'"
            )


@dataclass(frozen=True)
class Joint:
    """Base USD physics-joint spec; ``pos``/``rot`` are in the parent frame."""

    prim_path: str
    parent: str | None
    child: str
    pos: Vec3 = (0.0, 0.0, 0.0)
    rot: Quat = IDENTITY_XYZW


@dataclass(frozen=True)
class FixedJoint(Joint):
    """Weld ``child`` to ``parent``."""


@dataclass(frozen=True)
class RevoluteJoint(Joint):
    """Revolute joint; ``limits`` are radians and ``None`` spins free."""

    axis: Literal["X", "Y", "Z"] = "Z"
    limits: tuple[float, float] | None = None
    damping: float = 0.0


@dataclass(frozen=True)
class PrismaticJoint(Joint):
    """Prismatic joint; ``limits`` are meters and ``None`` slides free."""

    axis: Literal["X", "Y", "Z"] = "X"
    limits: tuple[float, float] | None = None
    damping: float = 0.0


@dataclass(frozen=True)
class LightSpec:
    """Dome-light defaults; a task overrides individual fields."""

    color: Vec3 = (0.75, 0.75, 0.75)
    intensity: float = 1500.0
    texture_file: str | None = None


@dataclass(frozen=True)
class ActorPose:
    """A world-frame actor pose override; quaternion is XYZW and must be non-zero."""

    position: Vec3
    quaternion_xyzw: Quat

    def __post_init__(self) -> None:
        values = (*self.position, *self.quaternion_xyzw)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("ActorPose values must be finite")
        norm = math.sqrt(sum(value * value for value in self.quaternion_xyzw))
        if norm <= 1e-12:
            raise ValueError("ActorPose quaternion must be non-zero")


def _duplicates(values: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


@dataclass(frozen=True)
class SceneSpec:
    """Everything needed to build a scene; run parameters (dt/steps/device) are CLI's.

    There is exactly one embodiment and an ordered set of sensor bindings.  A binding
    owns the pad key and mount; the embodiment owns the robot asset/joints/gripper.
    """

    embodiment: EmbodimentSpec
    sensors: tuple[SensorBinding, ...]
    cameras: tuple[CameraSpec, ...]
    actors: tuple[Actor, ...]
    light: LightSpec
    joints: tuple[Joint, ...] = ()
    joint_pos: dict[str, float] | None = None
    robot_prim_path: str = "/World/Robot"

    def __post_init__(self) -> None:
        keys = [binding.pad_key for binding in self.sensors]
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate sensor pad keys: {keys}")
        roles = [camera.role for camera in self.cameras]
        if len(set(roles)) != len(roles):
            raise ValueError(f"duplicate camera roles: {roles}")
        actor_paths = tuple(actor.prim_path for actor in self.actors)
        duplicate_actors = _duplicates(actor_paths)
        if duplicate_actors:
            raise ValueError(
                "SceneSpec.actors prim_path must be unique; duplicated: "
                f"{duplicate_actors}"
            )
        camera_paths = tuple(camera.prim_path for camera in self.cameras)
        duplicate_cameras = _duplicates(camera_paths)
        if duplicate_cameras:
            raise ValueError(
                "SceneSpec.cameras prim_path must be unique; duplicated: "
                f"{duplicate_cameras}"
            )
        actor_path_set = set(actor_paths)
        for joint in self.joints:
            if joint.parent is not None and joint.parent not in actor_path_set:
                raise ValueError(
                    f"SceneSpec.joints parent {joint.parent!r} of joint "
                    f"{joint.prim_path!r} matches no SceneSpec.actors prim_path"
                )
            if joint.child not in actor_path_set:
                raise ValueError(
                    f"SceneSpec.joints child {joint.child!r} of joint "
                    f"{joint.prim_path!r} matches no SceneSpec.actors prim_path"
                )
        if self.joint_pos is not None:
            self._validate_joint_pos()

    def _validate_joint_pos(self) -> None:
        """Check override keys against the embodiment's declared joints.

        IsaacLab treats ``initial_state.joint_pos`` keys as full-match regular
        expressions against the articulation's joint names and raises on a key that
        matches none, so catch the typo here while the failing field is still known.
        """
        assert self.joint_pos is not None
        joint_names = (
            *self.embodiment.arm.arm_joints,
            *self.embodiment.gripper.joints,
        )
        for pattern in self.joint_pos:
            try:
                matched = any(
                    re.fullmatch(pattern, name) for name in joint_names
                )
            except re.error as exc:
                raise ValueError(
                    f"SceneSpec.joint_pos key {pattern!r} is not a valid regex: "
                    f"{exc}"
                ) from exc
            if not matched:
                raise ValueError(
                    f"SceneSpec.joint_pos key {pattern!r} matches no joint of "
                    f"embodiment {self.embodiment.name!r}; available joints: "
                    f"{list(joint_names)}"
                )
