"""Per-tick world state handed to bound sensors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .ids import BodyId, CameraId, Pose


@dataclass(frozen=True)
class RigidState:
    """World pose of one rigid body."""

    pose: Pose


@dataclass(frozen=True)
class DeformableState:
    """One soft body: its calibration frame plus the live world-space surface."""

    frame: Pose
    vertices: NDArray[np.floating[Any]]
    triangles: NDArray[np.integer[Any]]


@dataclass(frozen=True)
class Requires:
    """The world handles one sensor needs; resolved once at build time."""

    bodies: frozenset[BodyId] = frozenset()
    deformables: frozenset[BodyId] = frozenset()
    cameras: frozenset[CameraId] = frozenset()
    objects: frozenset[BodyId] = frozenset()

    def __or__(self, other: Requires) -> Requires:
        return Requires(
            bodies=self.bodies | other.bodies,
            deformables=self.deformables | other.deformables,
            cameras=self.cameras | other.cameras,
            objects=self.objects | other.objects,
        )


@dataclass(frozen=True)
class TickSnapshot:
    """Immutable world state for one tick, memoised per required handle."""

    step_index: int
    timestamp_us: int
    bodies: Mapping[BodyId, RigidState]
    deformables: Mapping[BodyId, DeformableState]
    cameras: Mapping[CameraId, Pose]

    def __post_init__(self) -> None:
        object.__setattr__(self, "bodies", MappingProxyType(dict(self.bodies)))
        object.__setattr__(
            self, "deformables", MappingProxyType(dict(self.deformables))
        )
        object.__setattr__(self, "cameras", MappingProxyType(dict(self.cameras)))

    def pose(self, body: BodyId) -> Pose:
        """World pose of ``body``; raises when it was not requested."""
        state = self.bodies.get(body)
        if state is None:
            raise KeyError(f"snapshot was not built for body {body!r}")
        return state.pose

    def deformable(self, body: BodyId) -> DeformableState:
        """Live surface of ``body``; raises when it was not requested."""
        state = self.deformables.get(body)
        if state is None:
            raise KeyError(f"snapshot was not built for deformable {body!r}")
        return state

    def camera(self, camera: CameraId) -> Pose:
        """World pose of ``camera``; raises when it was not requested."""
        pose = self.cameras.get(camera)
        if pose is None:
            raise KeyError(f"snapshot was not built for camera {camera!r}")
        return pose

    def relative(self, ref: BodyId, body: BodyId) -> Pose:
        """``ref <- body`` composed from the two world poses."""
        return self.pose(ref).invert() @ self.pose(body)
