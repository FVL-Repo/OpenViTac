"""Typed identifiers and poses shared by sensors and the framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

import numpy as np
from numpy.typing import NDArray

BodyId = NewType("BodyId", str)
"""Any physical body: a PhysX link, a rigid actor, or a soft body."""

CameraId = NewType("CameraId", str)
"""A camera calibrated by the framework."""

PrimId = NewType("PrimId", str)
"""A non-physical prim the sensor tracks (e.g. a visual proxy)."""


@dataclass(frozen=True)
class Pose:
    """A frozen 4x4 ``world <- X`` transform."""

    matrix: NDArray[np.float64]

    def __post_init__(self) -> None:
        matrix = np.array(self.matrix, dtype=np.float64, copy=True)
        if matrix.shape != (4, 4):
            raise ValueError(f"pose matrix must be (4, 4), got {matrix.shape}")
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)

    @classmethod
    def identity(cls) -> Pose:
        return cls(np.eye(4, dtype=np.float64))

    def invert(self) -> Pose:
        return Pose(np.asarray(np.linalg.inv(self.matrix), dtype=np.float64))

    def __matmul__(self, other: Pose) -> Pose:
        return Pose(self.matrix @ other.matrix)
