"""Tactile sensor variants as an ADT: one frozen dataclass per sensor, no enum dispatch.

Physical-protocol numbers (resolution, marker grid, ``youngs``, ``strength``,
``attach_radius``) are consumed by the tactile-simulation stack; the sim core reads
only ``gripper_kind`` (embodiment selection), ``align_wrist`` (joint-init correction),
and ``settle`` (reset protocol).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

GripperKind = Literal["franka_panda", "franka_robotiq"]


@dataclass(frozen=True)
class AttachmentBody:
    """Robot body a tactile gelpad is fixed to."""

    name: str
    prim_path: str


@dataclass(frozen=True)
class SettleProtocol:
    """Marker-reference settle: ``stable_steps`` consecutive stable readings settle the
    reference, giving up after ``max_steps``."""

    stable_steps: int
    max_steps: int


@dataclass(frozen=True)
class SensorCfg:
    """Fields every tactile variant carries."""

    name: str
    prim_path: str
    gelpad_prim_path: str
    attachment_body: AttachmentBody
    gripper_kind: GripperKind
    data_types: tuple[str, ...] = ("rgb",)
    settle: SettleProtocol | None = None

    def align_wrist(self, joint_pos: dict[str, float]) -> dict[str, float]:
        """Joint-init correction the hardware imposes; identity unless overridden."""
        return joint_pos


@dataclass(frozen=True)
class GsMiniCfg(SensorCfg):
    """GelSight Mini, the baseline variant."""

    width: int = 320
    height: int = 240
    marker_rows: int = 7
    marker_cols: int = 9
    youngs: float = 0.1
    strength: float = 1e4


@dataclass(frozen=True)
class NeoteCfg(GsMiniCfg):
    """Neote: a GsMini decoration variant (same physics protocol, different casing)."""


@dataclass(frozen=True)
class XenseWSCfg(SensorCfg):
    """Xense WS: independent physics protocol and Robotiq gripper hardware."""

    width: int = 400
    height: int = 700
    marker_rows: int = 20
    marker_cols: int = 11
    youngs: float = 15.0
    strength: float = 2e5
    attach_radius: float = 0.0008
    settle: SettleProtocol | None = SettleProtocol(stable_steps=8, max_steps=180)
    wrist_alignment_offset: float = -math.pi / 4
    # Robotiq grasp rotation pinned to zero for the Xense wrist mount.
    grasp_rotate: float = 0.0

    def align_wrist(self, joint_pos: dict[str, float]) -> dict[str, float]:
        # The Xense WS mounts rotated about the wrist; back panda_joint7 off by pi/4 so
        # the gelpads face forward. Applied here so task joint values stay nominal.
        aligned = dict(joint_pos)
        aligned["panda_joint7"] = aligned["panda_joint7"] + self.wrist_alignment_offset
        return aligned


type SensorVariant = GsMiniCfg | XenseWSCfg | NeoteCfg
