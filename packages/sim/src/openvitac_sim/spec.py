"""Compatibility re-exports of the pure specs now owned by :mod:`openvitac_spec`.

The sim package depends on ``openvitac-spec``; task code should import the spec
package directly.  This module remains for one migration release so existing sim
call sites and external notebooks keep importing the same names.
"""

from __future__ import annotations

from openvitac_spec.embodiment import (
    FRANKA_PANDA_GSMINI,
    FRANKA_ROBOTIQ,
    ArmSpec,
    EmbodimentSpec,
    GripperCapability,
    GripperSpec,
    RobotAssetManifest,
    RobotAssetRef,
)
from openvitac_spec.scene import (
    Actor,
    CameraSpec,
    DepthCameraCfg,
    FixedJoint,
    Joint,
    LightSpec,
    PinholeIntrinsics,
    PrismaticJoint,
    RevoluteJoint,
    SceneSpec,
)
from openvitac_spec.sensor import (
    GsMiniSensor,
    MountSpec,
    NeoteSensor,
    SensorBinding,
    SensorCfg,
    SensorKind,
    SettleProtocol,
    StreamKind,
    XenseWSSensor,
)
from openvitac_spec.types import PadKey, Quat, Transform, Vec3

__all__ = [
    "FRANKA_PANDA_GSMINI",
    "FRANKA_ROBOTIQ",
    "Actor",
    "ArmSpec",
    "CameraSpec",
    "DepthCameraCfg",
    "EmbodimentSpec",
    "FixedJoint",
    "GripperCapability",
    "GripperSpec",
    "GsMiniSensor",
    "Joint",
    "LightSpec",
    "MountSpec",
    "NeoteSensor",
    "PadKey",
    "PinholeIntrinsics",
    "PrismaticJoint",
    "Quat",
    "RevoluteJoint",
    "RobotAssetManifest",
    "RobotAssetRef",
    "SceneSpec",
    "SensorBinding",
    "SensorCfg",
    "SensorKind",
    "SettleProtocol",
    "StreamKind",
    "Transform",
    "Vec3",
    "XenseWSSensor",
]
