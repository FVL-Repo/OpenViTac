"""USD surface meshes for the contact object set.

The object set is the scene's dynamic actors.  Their imaging geometry is the
authored USD surface, read once at bind (in the sim) or once per replay and
cached as object-frame triangles.  This module is the only place that opens a
USD stage outside the app; ``pxr`` is imported lazily so the pure geometry and
sensor modules stay importable without it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _triangulate(
    counts: NDArray[np.int64], indices: NDArray[np.int64]
) -> NDArray[np.int32]:
    """Fan-triangulate every face; triangle meshes pass through unchanged."""
    triangles: list[tuple[int, int, int]] = []
    offset = 0
    for count in counts:
        face = indices[offset : offset + int(count)]
        offset += int(count)
        for k in range(1, int(count) - 1):
            triangles.append((int(face[0]), int(face[k]), int(face[k + 1])))
    return np.asarray(triangles, dtype=np.int32).reshape(-1, 3)


def load_usd_surface(path: str) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Triangulated ``(vertices, triangles)`` of every mesh in a USD asset.

    Vertices are in the asset root frame (the frame a rigid actor's world pose
    maps to world), not in each mesh prim's local frame.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(path)
    if stage is None:
        raise RuntimeError(f"cannot open USD surface {path!r}")
    root = stage.GetDefaultPrim()
    if not root.IsValid():
        root = stage.GetPseudoRoot()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    root_to_world = cache.GetLocalToWorldTransform(root)
    vertices: list[NDArray[np.float64]] = []
    triangles: list[NDArray[np.int32]] = []
    offset = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if points is None or counts is None or indices is None:
            raise RuntimeError(f"mesh {prim.GetPath()} in {path!r} is incomplete")
        local = root_to_world.GetInverse() * cache.GetLocalToWorldTransform(prim)
        # Gf.Matrix4d converts to a row-major array; transposing gives the
        # column-vector transform used everywhere else in the repo.
        matrix = np.array(local, dtype=np.float64).T
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        pts = pts @ matrix[:3, :3].T + matrix[:3, 3]
        tris = _triangulate(
            np.asarray(counts, dtype=np.int64),
            np.asarray(indices, dtype=np.int64),
        )
        if len(tris) == 0:
            continue
        vertices.append(pts)
        triangles.append(tris + offset)
        offset += len(pts)
    if not vertices:
        raise RuntimeError(f"USD asset {path!r} contains no triangle surface")
    return np.vstack(vertices), np.vstack(triangles)
