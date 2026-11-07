"""Pure-numpy tetrahedral mesh helpers and pad surface truncation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TetMesh:
    vertices: NDArray[np.float64]
    tets: NDArray[np.int32]

    def __post_init__(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("TetMesh.vertices must be (V,3)")
        if self.tets.ndim != 2 or self.tets.shape[1] != 4:
            raise ValueError("TetMesh.tets must be (T,4)")
        if np.any(self.tets < 0) or np.any(self.tets >= len(self.vertices)):
            raise ValueError("TetMesh.tets index out of range")


@dataclass(frozen=True)
class SurfaceMesh:
    vertices: NDArray[np.float32]
    triangles: NDArray[np.int32]

    def __post_init__(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("SurfaceMesh.vertices must be (V,3)")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError("SurfaceMesh.triangles must be (F,3)")


def _positive_tets(
    vertices: NDArray[np.float64], tets: NDArray[np.int64]
) -> NDArray[np.int32]:
    out = np.array(tets, dtype=np.int32, copy=True)
    for i, tet in enumerate(out):
        a, b, c, d = vertices[tet]
        if float(np.dot(np.cross(b - a, c - a), d - a)) < 0.0:
            out[i] = [tet[0], tet[1], tet[3], tet[2]]
    return out


def box_tet_mesh(
    size_m: tuple[float, float, float],
    resolution: tuple[int, int, int] = (8, 1, 5),
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> TetMesh:
    """Structured box tet mesh, six tets per hex, consistently oriented."""
    lx, ly, lz = size_m
    nx, ny, nz = resolution
    if min(lx, ly, lz) <= 0.0 or min(nx, ny, nz) <= 0:
        raise ValueError("box_tet_mesh requires positive size/resolution")
    xs = np.linspace(-lx / 2.0, lx / 2.0, nx + 1) + center[0]
    ys = np.linspace(-ly / 2.0, ly / 2.0, ny + 1) + center[1]
    zs = np.linspace(-lz / 2.0, lz / 2.0, nz + 1) + center[2]
    vertices = np.asarray(
        [[x, y, z] for z in zs for y in ys for x in xs], dtype=np.float64
    )

    def idx(i: int, j: int, k: int) -> int:
        return (k * (ny + 1) + j) * (nx + 1) + i

    tets: list[list[int]] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                v000 = idx(i, j, k)
                v100 = idx(i + 1, j, k)
                v110 = idx(i + 1, j + 1, k)
                v010 = idx(i, j + 1, k)
                v001 = idx(i, j, k + 1)
                v101 = idx(i + 1, j, k + 1)
                v111 = idx(i + 1, j + 1, k + 1)
                v011 = idx(i, j + 1, k + 1)
                tets.extend(
                    [
                        [v000, v100, v110, v111],
                        [v000, v110, v010, v111],
                        [v000, v010, v011, v111],
                        [v000, v011, v001, v111],
                        [v000, v001, v101, v111],
                        [v000, v101, v100, v111],
                    ]
                )
    t = np.asarray(tets, dtype=np.int64)
    return TetMesh(vertices=vertices, tets=_positive_tets(vertices, t))


def load_tet_mesh(path: str | Path) -> TetMesh:
    p = Path(path)
    if p.suffix != ".npz":
        raise ValueError(f"unsupported tet mesh format {p.suffix!r}; expected .npz")
    with np.load(p, allow_pickle=False) as data:
        v_key = "verts" if "verts" in data else "vertices"
        if v_key not in data or "tets" not in data:
            raise ValueError(f"{p} must contain {v_key!r} and 'tets'")
        vertices = np.ascontiguousarray(data[v_key], dtype=np.float64)
        tets = np.ascontiguousarray(data["tets"], dtype=np.int32)
    return TetMesh(vertices=vertices, tets=tets)


def boundary_triangles(tets: NDArray[np.int32]) -> NDArray[np.int32]:
    """Return consistently-oriented triangles on the tet-mesh boundary."""
    local = np.asarray([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    faces = tets[:, local].reshape(-1, 3)
    keys = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    boundary = faces[counts[inverse] == 1]
    return boundary.astype(np.int32)


def _select_surface(
    vertices: NDArray[np.floating[Any]],
    triangles: NDArray[np.integer[Any]],
    *,
    max_vertices: int,
    decimation: int,
) -> SurfaceMesh:
    """Truncate/decimate one already-extracted surface and remap its faces."""
    if decimation < 1:
        raise ValueError("decimation must be >= 1")
    if max_vertices < 0:
        raise ValueError("max_vertices must be >= 0")
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if points.size == 0 or tris.size == 0:
        return SurfaceMesh(
            vertices=np.empty((0, 3), dtype=np.float32),
            triangles=np.empty((0, 3), dtype=np.int32),
        )
    used = np.unique(tris.reshape(-1))
    if decimation > 1:
        used = used[::decimation]
    used = used[:max_vertices]
    remap = -np.ones(len(points), dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    keep = np.all(remap[tris] >= 0, axis=1)
    tris = remap[tris[keep]]
    if len(used) == 0 or len(tris) == 0:
        return SurfaceMesh(
            vertices=np.empty((0, 3), dtype=np.float32),
            triangles=np.empty((0, 3), dtype=np.int32),
        )
    return SurfaceMesh(
        vertices=points[used].astype(np.float32),
        triangles=tris.astype(np.int32),
    )


def tetmesh_surface(
    mesh: TetMesh,
    *,
    max_vertices: int | None = None,
    decimation: int = 1,
) -> SurfaceMesh:
    """Extract a boundary surface and optionally truncate/decimate vertices.

    ``max_vertices`` is a hard SHM-slot cap; faces referencing dropped vertices
    are dropped and indices remapped.  This is intentionally a prototype
    operation: for a high-resolution gel pad, decimate/remesh offline before
    increasing the protocol slot.
    """
    return _select_surface(
        mesh.vertices,
        boundary_triangles(mesh.tets),
        max_vertices=len(mesh.vertices) if max_vertices is None else max_vertices,
        decimation=decimation,
    )


def published_surface(
    vertices: NDArray[np.floating[Any]],
    triangles: NDArray[np.integer[Any]],
    *,
    max_vertices: int,
    decimation: int = 1,
) -> SurfaceMesh:
    """Apply the worker's published cap/decimation to an extracted surface.

    UIPC returns a triangle surface in its own vertex order; this keeps that
    order (and remaps faces) so the rest mesh and every per-frame vertex stream
    are the same sequence.  ``max_vertices`` is the SHM slot cap.
    """
    return _select_surface(
        vertices,
        triangles,
        max_vertices=max_vertices,
        decimation=decimation,
    )
