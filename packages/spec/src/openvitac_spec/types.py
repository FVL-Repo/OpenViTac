"""Small pure-data aliases shared by the spec dataclasses.

Nothing here may import isaaclab, pxr, pyuipc, numpy, the sim package, or the
tactile runtime: task modules import these types on machines without a simulator.
"""

from __future__ import annotations

from typing import Literal

type Vec3 = tuple[float, float, float]
type Quat = tuple[float, float, float, float]  # XYZW
type PadKey = str

# Terminal result of a task's success evaluation. Harness states (``timeout``,
# ``unknown``) are not produced by an evaluator; the episode writer owns them.
type Outcome = Literal["success", "fail"]

# Row-major 4x4 transform.  We keep tuples rather than numpy arrays so a scene spec
# stays hashable/serializable and cannot be mutated by a runtime consumer.
type Transform = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]

IDENTITY_XYZW: Quat = (0.0, 0.0, 0.0, 1.0)
IDENTITY_TRANSFORM: Transform = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
