"""Pure-numpy contact metrics for the UIPC tactile worker.

This module intentionally imports neither pyuipc nor Isaac so the selection
rule and the pad/object proxy can be unit-tested without launching a solver.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PadObjectMetrics:
    """Pad-object proximity measured by :func:`pad_object_metrics`."""

    min_sd_mm: float
    contact_vertices: int


def select_contact_index(objects: Sequence[tuple[str, bool, bool]]) -> int | None:
    """Select the UIPC object whose surface pad contact is measured.

    ``objects`` entries are ``(name, fixed, contact_object)`` in scene order.
    An explicit ``contact_object`` flag wins; otherwise exactly one non-fixed
    object is selected. Multiple candidates without a flag are ambiguous.
    """
    flagged = [i for i, (_, _, flag) in enumerate(objects) if flag]
    if len(flagged) > 1:
        names = ", ".join(objects[i][0] for i in flagged)
        raise ValueError(f"multiple ObjectCfg.contact_object entries: {names}")
    if flagged:
        index = flagged[0]
        if objects[index][1]:
            raise ValueError(
                f"fixed UIPC object {objects[index][0]!r} cannot be contact_object"
            )
        return index
    candidates = [i for i, (_, fixed, _) in enumerate(objects) if not fixed]
    if len(candidates) > 1:
        names = ", ".join(objects[i][0] for i in candidates)
        raise ValueError(
            "multiple non-fixed UIPC objects; set ObjectCfg.contact_object on "
            f"exactly one: {names}"
        )
    return candidates[0] if candidates else None


def pad_object_metrics(
    pad_vertices_m: NDArray[np.floating[Any]],
    object_T: NDArray[np.floating[Any]],
    local_min: NDArray[np.floating[Any]],
    local_max: NDArray[np.floating[Any]],
    d_hat_m: float,
) -> PadObjectMetrics:
    """Measure pad vertices against the object's local AABB.

    ``object_T`` maps object-local to world; ``local_min``/``local_max`` bound
    the object in that local frame. The signed distance is positive outside the
    AABB and negative inside. ``contact_vertices`` counts pad vertices closer
    than ``d_hat_m``. This is a cheap pose-based proxy, not a full
    mesh-to-mesh distance.
    """
    T = np.asarray(object_T, dtype=np.float64).reshape(4, 4)
    points = np.asarray(pad_vertices_m, dtype=np.float64).reshape(-1, 3)
    if points.size == 0:
        return PadObjectMetrics(min_sd_mm=float("nan"), contact_vertices=0)
    # Row-vector form of R^T (p - t).
    local = (points - T[:3, 3]) @ T[:3, :3]
    lo = np.asarray(local_min, dtype=np.float64).reshape(3)
    hi = np.asarray(local_max, dtype=np.float64).reshape(3)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    q = np.abs(local - center) - half
    sd = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(
        np.max(q, axis=1), 0.0
    )
    return PadObjectMetrics(
        min_sd_mm=float(np.min(sd) * 1000.0),
        contact_vertices=int(np.count_nonzero(sd < float(d_hat_m))),
    )
