"""openvitac_sim: sim primitives. The only package allowed to import isaaclab/omni/pxr.

isaaclab-dependent submodules (``spawn``/``robot``/``scene``) are lazy-loaded via
PEP 562 ``__getattr__``: importing this package never touches isaaclab before the app
is up.  ``spec`` and ``sensors`` are isaaclab-free and import eagerly.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .app import launch
from .assets import asset, assets_root
from .sensors import (
    AttachmentBody,
    GripperKind,
    GsMiniCfg,
    NeoteCfg,
    SensorCfg,
    SensorVariant,
    SettleProtocol,
    XenseWSCfg,
)
from .spec import (
    FRANKA_PANDA_GSMINI,
    FRANKA_ROBOTIQ,
    Actor,
    CameraSpec,
    DepthCameraCfg,
    EmbodimentSpec,
    FixedJoint,
    GripperCapability,
    GsMiniSensor,
    Joint,
    LightSpec,
    MountSpec,
    PinholeIntrinsics,
    PrismaticJoint,
    RevoluteJoint,
    RobotAssetRef,
    SceneSpec,
    SensorBinding,
    XenseWSSensor,
)

# submodule -> names it exports, imported on first access (post-launch).
_LAZY: dict[str, tuple[str, ...]] = {
    "spawn": (
        "create_joint",
        "spawn_articulation",
        "spawn_camera",
        "spawn_light",
        "spawn_rigid",
        "spawn_static",
    ),
    "robot": ("RobotArm", "assemble_robot"),
    "scene": ("Scene", "SceneObservation", "StepLimitExceeded"),
}


def __getattr__(name: str) -> object:
    for mod_name, names in _LAZY.items():
        if name in names:
            module = importlib.import_module(f".{mod_name}", __name__)
            value = getattr(module, name)
            globals()[name] = value  # cache so __getattr__ fires once per name
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from .robot import RobotArm, assemble_robot
    from .scene import Scene, SceneObservation, StepLimitExceeded
    from .spawn import (
        create_joint,
        spawn_articulation,
        spawn_camera,
        spawn_light,
        spawn_rigid,
        spawn_static,
    )

__all__ = [
    "FRANKA_PANDA_GSMINI",
    "FRANKA_ROBOTIQ",
    "Actor",
    "AttachmentBody",
    "CameraSpec",
    "DepthCameraCfg",
    "EmbodimentSpec",
    "FixedJoint",
    "GripperCapability",
    "GripperKind",
    "GsMiniCfg",
    "GsMiniSensor",
    "Joint",
    "LightSpec",
    "MountSpec",
    "NeoteCfg",
    "PinholeIntrinsics",
    "PrismaticJoint",
    "RevoluteJoint",
    "RobotArm",
    "RobotAssetRef",
    "Scene",
    "SceneObservation",
    "SceneSpec",
    "SensorBinding",
    "SensorCfg",
    "SensorVariant",
    "SettleProtocol",
    "StepLimitExceeded",
    "XenseWSCfg",
    "XenseWSSensor",
    "assemble_robot",
    "asset",
    "assets_root",
    "create_joint",
    "launch",
    "spawn_articulation",
    "spawn_camera",
    "spawn_light",
    "spawn_rigid",
    "spawn_static",
]
