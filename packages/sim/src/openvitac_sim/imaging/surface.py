"""Mesh-to-depth rasterization primitives for the tactile imaging pass.

The rasterizer is a nearest-surface z-buffer: every triangle writes the optical
axis depth of the surface it covers, and the smallest depth wins.  It is the
only place where mesh geometry becomes an image; contact height and normals are
derived from that depth image in :mod:`openvitac_sim.imaging.height`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _project(
    points: NDArray[np.floating[Any]],
    camera_pose: NDArray[np.floating[Any]],
    intrinsics: NDArray[np.floating[Any]],
) -> tuple[FloatArray, FloatArray]:
    """Project points to pixels and positive optical-axis depth."""
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    T = np.asarray(camera_pose, dtype=np.float64).reshape(4, 4)
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    inverse = np.linalg.inv(T)
    camera_points = xyz @ inverse[:3, :3].T + inverse[:3, 3]
    depth = -camera_points[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * camera_points[:, 0] / depth + K[0, 2]
        v = K[1, 1] * (-camera_points[:, 1]) / depth + K[1, 2]
    return np.stack((u, v), axis=1), depth


def _z_buffer(
    depth: FloatArray,
    vertices: NDArray[np.floating[Any]],
    triangles: NDArray[np.integer[Any]],
    camera_pose: NDArray[np.floating[Any]],
    intrinsics: NDArray[np.floating[Any]],
    *,
    camera_facing_only: bool = False,
    max_depth: float | None = None,
) -> None:
    """Write the nearest surface of ``triangles`` into ``depth`` in place.

    ``depth`` is ``(H, W)`` optical-axis depth in metres, ``+inf`` where no
    surface has been written; values only ever decrease.  ``camera_facing_only``
    drops triangles whose outward normal points away from the camera, which is
    exact for a closed, outward-oriented mesh (the visible surface of a closed
    solid is always front-facing) and halves the per-frame triangle count.
    ``max_depth`` drops triangles that lie wholly behind that depth: they can
    neither be the nearest surface of a pixel that matters nor occlude one.
    """
    height, width = depth.shape
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    camera = np.asarray(camera_pose, dtype=np.float64).reshape(4, 4)[:3, 3]
    if camera_facing_only:
        corners = points[tris]
        normals = np.cross(
            corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
        )
        centroids = corners.mean(axis=1)
        front = np.einsum("ij,ij->i", normals, camera - centroids) > 0.0
        tris = tris[front]
    if len(tris) == 0:
        return
    uv, depths = _project(points, camera_pose, intrinsics)
    if max_depth is not None:
        tris = tris[np.min(depths[tris], axis=1) <= max_depth]
        if len(tris) == 0:
            return
    for tri in tris:
        if np.any(depths[tri] <= 1e-9):
            continue
        x = uv[tri, 0]
        y = uv[tri, 1]
        x_min = max(0, int(np.floor(x.min())))
        x_max = min(width - 1, int(np.ceil(x.max())))
        y_min = max(0, int(np.floor(y.min())))
        y_max = min(height - 1, int(np.ceil(y.max())))
        if x_min > x_max or y_min > y_max:
            continue
        xs, ys = np.meshgrid(
            np.arange(x_min, x_max + 1, dtype=np.float64),
            np.arange(y_min, y_max + 1, dtype=np.float64),
        )
        denominator = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (
            y[0] - y[2]
        )
        if abs(float(denominator)) <= 1e-15:
            continue
        a = (
            (y[1] - y[2]) * (xs - x[2]) + (x[2] - x[1]) * (ys - y[2])
        ) / denominator
        b = (
            (y[2] - y[0]) * (xs - x[2]) + (x[0] - x[2]) * (ys - y[2])
        ) / denominator
        c = 1.0 - a - b
        inside = (a >= -1e-7) & (b >= -1e-7) & (c >= -1e-7)
        inverse_depth = (
            a / depths[tri[0]] + b / depths[tri[1]] + c / depths[tri[2]]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            z = 1.0 / inverse_depth
        inside &= np.isfinite(z) & (z > 0.0)
        if not bool(inside.any()):
            continue
        region_depth = depth[y_min : y_max + 1, x_min : x_max + 1]
        update = inside & (z < region_depth)
        if not bool(update.any()):
            continue
        region_depth[update] = z[update]
