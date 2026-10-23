"""Pure-numpy observation types exposed to the simulation layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

type PadKey = str
"""Stable task-facing pad identifier; ``left``/``right`` are the current helpers."""

type Side = PadKey
"""Deprecated alias kept for one migration release."""

type ForceKind = Literal["net_force", "contact_gradient", "none"]


def _empty_objects() -> dict[str, ObjectObservation]:
    return {}


def _empty_pads() -> dict[PadKey, PadObservation]:
    return {}


def _empty_rgb() -> dict[PadKey, NDArray[np.uint8]]:
    return {}


def _empty_depth() -> dict[PadKey, NDArray[np.float32]]:
    return {}


@dataclass(frozen=True)
class ObjectObservation:
    name: str
    pose_xyzw: NDArray[np.float32]
    valid: bool = True


@dataclass(frozen=True)
class PadObservation:
    side: PadKey
    vertices: NDArray[np.float32] | None
    n_vertices: int
    contact_gradient: NDArray[np.float32]
    force: NDArray[np.float32] | None
    force_kind: ForceKind
    contact_count: int
    min_sd_proxy_mm: float
    min_sd_full_mm: float | None
    tracking_error_mm: float
    seq: int
    age_ms: float
    stale: bool
    valid: bool = True
    pad_usb_min_sd_mm: float | None = None
    pad_usb_contact_vertices: int = 0

    @property
    def pad_key(self) -> PadKey:
        """The generic task-facing key; ``side`` remains the compatibility name."""
        return self.side


@dataclass(frozen=True)
class TactileObservation:
    seq: int
    finger_seq: int
    physx_step: int
    objects: Mapping[str, ObjectObservation] = field(default_factory=_empty_objects)
    pads: Mapping[PadKey, PadObservation] = field(default_factory=_empty_pads)
    age_ms: float = 0.0
    stale: bool = False
    seq_gap: int = 0
    finger_age_ms: float = 0.0
    finger_stale: bool = False
    tactile_rgb: Mapping[PadKey, NDArray[np.uint8]] = field(
        default_factory=_empty_rgb
    )
    tactile_marker_rgb: Mapping[PadKey, NDArray[np.uint8]] = field(
        default_factory=_empty_rgb
    )
    tactile_depth: Mapping[PadKey, NDArray[np.float32]] = field(
        default_factory=_empty_depth
    )
