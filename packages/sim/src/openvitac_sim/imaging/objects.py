"""The manipulated object set: cached triangle meshes and their z-buffer.

Imaging never looks at the sensor's own gel mesh.  The only geometry that can
produce contact height is the surface of a declared contact object, cached once
as an object-frame triangle mesh and rigidly transformed into the pad camera
every tick.  Nothing here owns or guesses object identity: the scene resolves
the object set from its actor declarations and hands it to the sensor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .surface import FloatArray, _z_buffer


def _is_closed(triangles: NDArray[np.int32]) -> bool:
    """Whether ``triangles`` form a closed mesh worth back-face culling.

    A closed solid's visible surface is always front-facing, so the rasterizer
    may cull back faces.  The check is exact: every undirected edge must be
    shared by exactly two triangles; an inverted winding is flipped in
    :meth:`ObjectSurface.from_mesh`, not tolerated.
    """
    edges: dict[tuple[int, int], int] = {}
    for a, b, c in triangles:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (min(u, v), max(u, v))
            edges[key] = edges.get(key, 0) + 1
    return all(count == 2 for count in edges.values())


def _signed_volume(vertices: FloatArray, triangles: NDArray[np.int32]) -> float:
    """Six times the enclosed volume; negative when the winding is inward."""
    corners = vertices[triangles]
    return float(
        np.einsum(
            "ij,ij->i",
            corners[:, 0],
            np.cross(corners[:, 1], corners[:, 2]),
        ).sum()
    )


@dataclass(frozen=True)
class ObjectSurface:
    """One contact object's triangle surface in its own rigid body frame."""

    body: str
    prim_path: str
    vertices: FloatArray
    triangles: NDArray[np.int32]
    camera_facing_only: bool

    @classmethod
    def from_mesh(
        cls,
        *,
        body: str,
        prim_path: str,
        vertices: NDArray[np.floating[Any]],
        triangles: NDArray[np.integer[Any]],
    ) -> ObjectSurface:
        """Validate and cache one object's triangles; reject unusable meshes."""
        points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        tris = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
        if len(points) < 3 or len(tris) == 0:
            raise ValueError(
                f"contact object {body!r} has no triangle surface "
                f"({len(points)} vertices, {len(tris)} triangles)"
            )
        if not np.isfinite(points).all():
            raise ValueError(f"contact object {body!r} mesh has non-finite points")
        if int(tris.min()) < 0 or int(tris.max()) >= len(points):
            raise ValueError(f"contact object {body!r} mesh indices are out of range")
        corners = points[tris]
        area = np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
            axis=1,
        )
        # A zero-area triangle covers no pixel; dropping it is exact.
        keep = area > 0.0
        if not bool(keep.any()):
            raise ValueError(f"contact object {body!r} mesh is fully degenerate")
        tris = tris[keep]
        if _is_closed(tris):
            if _signed_volume(points, tris) < 0.0:
                tris = np.ascontiguousarray(tris[:, ::-1])
            camera_facing_only = True
        else:
            camera_facing_only = False
        return cls(
            body=body,
            prim_path=prim_path,
            vertices=points,
            triangles=tris,
            camera_facing_only=camera_facing_only,
        )

    def world_vertices(self, pose: NDArray[np.floating[Any]]) -> FloatArray:
        """Object-frame vertices in world space under ``pose`` (world <- body)."""
        matrix = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        return self.vertices @ matrix[:3, :3].T + matrix[:3, 3]


def object_depth(
    objects: Sequence[ObjectSurface],
    poses: Mapping[str, NDArray[np.floating[Any]]],
    camera_pose: NDArray[np.floating[Any]],
    intrinsics: NDArray[np.floating[Any]],
    image_size: tuple[int, int],
    *,
    max_depth: float | None = None,
) -> FloatArray:
    """Nearest declared-object surface depth through the pad camera.

    Returns ``(H, W)`` optical-axis depth in metres, ``+inf`` where no declared
    object covers the pixel.  Every object is z-buffered into one depth image,
    so a nearer object wins and a farther one cannot create height.
    ``max_depth`` drops surfaces that cannot be nearer than the rest plane, so
    they cannot produce height.
    """
    width, height = image_size
    depth = np.full((height, width), np.inf, dtype=np.float64)
    for obj in objects:
        pose = poses.get(obj.body)
        if pose is None:
            raise KeyError(
                f"no object pose for {obj.body!r}; the tick snapshot must carry "
                "every declared contact object"
            )
        _z_buffer(
            depth,
            obj.world_vertices(pose),
            obj.triangles,
            camera_pose,
            intrinsics,
            camera_facing_only=obj.camera_facing_only,
            max_depth=max_depth,
        )
    return depth
