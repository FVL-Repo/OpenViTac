"""Sensor hardware and mounting specs, independent of a robot asset.

``SensorCfg`` describes the transducer only.  ``SensorBinding`` is the task-side
placement: which pad key it exposes, which body it rides, and where its gelpad/case
prims live.  A robot preset never selects a sensor, and a sensor never selects a
robot (the old ``gripper_kind`` coupling is gone).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from .types import IDENTITY_TRANSFORM, PadKey, Transform

type SensorKind = Literal["gsmini", "xense_ws", "neote"]
type StreamKind = Literal["rgb", "depth", "marker_rgb"]

# Runtime consumers match on these kinds and silently drop anything else, so an
# unknown token (e.g. a ``"rgba"`` typo) would otherwise render nothing.
_STREAM_KINDS: frozenset[StreamKind] = frozenset({"rgb", "depth", "marker_rgb"})
_DEFAULT_STREAMS: tuple[StreamKind, ...] = ("rgb", "depth", "marker_rgb")


@dataclass(frozen=True)
class SettleProtocol:
    """Optional per-sensor settle schedule used by the tactile runtime."""

    stable_steps: int
    max_steps: int


@dataclass(frozen=True)
class SensorCfg:
    """Physical description of one tactile sensor (no robot/pad identity)."""

    kind: SensorKind
    width: int
    height: int
    marker_rows: int
    marker_cols: int
    youngs_mpa: float
    strength: float
    settle: SettleProtocol | None = None


@dataclass(frozen=True)
class GsMiniSensor(SensorCfg):
    kind: SensorKind = "gsmini"
    width: int = 320
    height: int = 240
    marker_rows: int = 7
    marker_cols: int = 9
    youngs_mpa: float = 0.1
    strength: float = 1e4


@dataclass(frozen=True)
class NeoteSensor(GsMiniSensor):
    kind: SensorKind = "neote"


@dataclass(frozen=True)
class XenseWSSensor(SensorCfg):
    kind: SensorKind = "xense_ws"
    width: int = 400
    height: int = 700
    marker_rows: int = 20
    marker_cols: int = 11
    youngs_mpa: float = 15.0
    strength: float = 2e5
    settle: SettleProtocol | None = SettleProtocol(
        stable_steps=8, max_steps=180
    )


@dataclass(frozen=True)
class MountSpec:
    """Mechanical/calibration data for one binding.

    ``joint_offsets`` are additive nominal joint offsets (e.g. the Xense wrist
    alignment); ``grasp_rotate`` is the object-frame rotation used before a grasp;
    ``camera_prim`` is the authored optical camera for depth/marker streams;
    ``attach_radius_m`` is the radial pad attachment used by UIPC frames.
    """

    mount_id: str
    joint_offsets: dict[str, float] = field(
        default_factory=dict[str, float]
    )
    camera_prim: str | None = None
    grasp_rotate: float = 0.0
    attach_radius_m: float | None = None
    tool_offset: Transform = IDENTITY_TRANSFORM


@dataclass(frozen=True)
class SensorBinding:
    """One sensor placed on a named robot body and exposed under a stable pad key."""

    pad_key: PadKey
    sensor: SensorCfg
    body_name: str
    body_prim_path: str
    case_prim_path: str
    gelpad_prim_path: str
    mount: MountSpec
    streams: tuple[StreamKind, ...] = _DEFAULT_STREAMS

    def __post_init__(self) -> None:
        for name in ("pad_key", "body_name", "body_prim_path", "gelpad_prim_path"):
            if not getattr(self, name):
                raise ValueError(f"SensorBinding.{name} must be non-empty")
        if not self.mount.mount_id:
            raise ValueError("SensorBinding.mount.mount_id must be non-empty")
        if not self.streams:
            raise ValueError(
                f"sensor {self.pad_key!r} must request at least one stream"
            )
        unknown = tuple(
            kind for kind in self.streams if kind not in _STREAM_KINDS
        )
        if unknown:
            raise ValueError(
                f"SensorBinding.streams for sensor {self.pad_key!r} contains "
                f"unknown stream kind(s) {unknown!r}; allowed: "
                f"{sorted(_STREAM_KINDS)}"
            )


def _binding(
    *,
    pad_key: str,
    sensor: SensorCfg,
    case_prim: str,
    gelpad_prim: str,
    mount: MountSpec,
    streams: tuple[StreamKind, ...],
) -> SensorBinding:
    body = case_prim.rsplit("/", 1)[-1]
    return SensorBinding(
        pad_key=pad_key,
        sensor=sensor,
        body_name=body,
        body_prim_path=case_prim,
        case_prim_path=case_prim,
        gelpad_prim_path=gelpad_prim,
        mount=mount,
        streams=streams,
    )


def gsmini_pair(
    *, streams: tuple[StreamKind, ...] = _DEFAULT_STREAMS
) -> tuple[SensorBinding, ...]:
    """The standard GsMini left/right pair on the Panda parallel gripper."""

    return tuple(
        _binding(
            pad_key=side,
            sensor=GsMiniSensor(),
            case_prim=f"/World/Robot/gelsight_mini_case_{side}",
            gelpad_prim=f"/World/Robot/gelpad_{side}",
            mount=MountSpec(
                mount_id=f"gsmini_{side}",
                camera_prim=f"/World/Robot/gelsight_mini_case_{side}/Camera",
            ),
            streams=streams,
        )
        for side in ("left", "right")
    )


def neote_pair(
    *, streams: tuple[StreamKind, ...] = _DEFAULT_STREAMS
) -> tuple[SensorBinding, ...]:
    """The standard Neote left/right pair (same case/gelpad layout as GsMini)."""

    return tuple(
        _binding(
            pad_key=side,
            sensor=NeoteSensor(),
            case_prim=f"/World/Robot/gelsight_mini_case_{side}",
            gelpad_prim=f"/World/Robot/gelpad_{side}",
            mount=MountSpec(
                mount_id=f"neote_{side}",
                camera_prim=f"/World/Robot/gelsight_mini_case_{side}/Camera",
            ),
            streams=streams,
        )
        for side in ("left", "right")
    )


def xense_pair(
    *, streams: tuple[StreamKind, ...] = _DEFAULT_STREAMS
) -> tuple[SensorBinding, ...]:
    """The standard XenseWS left/right pair on the Robotiq mount.

    The old ``align_wrist`` offset (-45 deg on panda_joint7) is data here, not a
    sensor-side robot switch.
    """

    return tuple(
        _binding(
            pad_key=side,
            sensor=XenseWSSensor(),
            case_prim=f"/World/Robot/root_joint/XenseWS_{side}",
            gelpad_prim=f"/World/Robot/XenseWS_gelpad_{side}",
            mount=MountSpec(
                mount_id=f"xense_{side}",
                joint_offsets={"panda_joint7": -math.pi / 4.0},
                attach_radius_m=0.0008,
            ),
            streams=streams,
        )
        for side in ("left", "right")
    )
