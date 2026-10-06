#!/usr/bin/env python3
"""Generate a simple white cable-routing clip OBJ/MTL asset."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np


ASSET_NAME = "cable_clip_white"
OUT_DIR = Path("assets/objects/task_assets/cable_routing")
OBJ_PATH = OUT_DIR / f"{ASSET_NAME}.obj"
MTL_PATH = OUT_DIR / f"{ASSET_NAME}.mtl"

# Axis-aligned boxes: (x0, x1, y0, y1, z0, z1), in meters.
# The cable channel runs along X. The origin is at the bottom-face center.
BOXES = (
    (-0.030, 0.030, -0.022, 0.022, 0.000, 0.005),  # base plate
    (-0.018, 0.018, -0.016, -0.010, 0.005, 0.034),  # near rail
    (-0.018, 0.018, 0.010, 0.016, 0.005, 0.034),  # far rail
    (-0.018, 0.018, -0.016, -0.007, 0.029, 0.034),  # near snap lip
    (-0.018, 0.018, 0.007, 0.016, 0.029, 0.034),  # far snap lip
)

# Simple square mounting holes through the base plate. These are intentionally
# low-poly to keep the asset close to the simple SILO-style fixtures.
HOLES = (
    (-0.025, -0.019, -0.003, 0.003, 0.000, 0.005),
    (0.019, 0.025, -0.003, 0.003, 0.000, 0.005),
)


FACE_DEFS = (
    ((-1, 0, 0), lambda x0, x1, y0, y1, z0, z1: ((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0))),
    ((1, 0, 0), lambda x0, x1, y0, y1, z0, z1: ((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1))),
    ((0, -1, 0), lambda x0, x1, y0, y1, z0, z1: ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1))),
    ((0, 1, 0), lambda x0, x1, y0, y1, z0, z1: ((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0))),
    ((0, 0, -1), lambda x0, x1, y0, y1, z0, z1: ((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0))),
    ((0, 0, 1), lambda x0, x1, y0, y1, z0, z1: ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))),
)


def in_box(point: tuple[float, float, float], box: tuple[float, float, float, float, float, float]) -> bool:
    x, y, z = point
    x0, x1, y0, y1, z0, z1 = box
    return x0 < x < x1 and y0 < y < y1 and z0 < z < z1


def make_grid() -> tuple[list[float], list[float], list[float], np.ndarray]:
    xs = sorted({coord for box in BOXES + HOLES for coord in box[:2]})
    ys = sorted({coord for box in BOXES + HOLES for coord in box[2:4]})
    zs = sorted({coord for box in BOXES + HOLES for coord in box[4:]})

    occupied = np.zeros((len(xs) - 1, len(ys) - 1, len(zs) - 1), dtype=bool)
    for ix in range(len(xs) - 1):
        x = 0.5 * (xs[ix] + xs[ix + 1])
        for iy in range(len(ys) - 1):
            y = 0.5 * (ys[iy] + ys[iy + 1])
            for iz in range(len(zs) - 1):
                z = 0.5 * (zs[iz] + zs[iz + 1])
                point = (x, y, z)
                solid = any(in_box(point, box) for box in BOXES)
                cut = any(in_box(point, hole) for hole in HOLES)
                occupied[ix, iy, iz] = solid and not cut
    return xs, ys, zs, occupied


def boundary_mesh() -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int, int]]]:
    xs, ys, zs, occupied = make_grid()
    vertices: list[tuple[float, float, float]] = []
    vertex_ids: dict[tuple[float, float, float], int] = {}
    faces: list[tuple[int, int, int, int]] = []
    nx, ny, nz = occupied.shape

    def vertex_index(vertex: tuple[float, float, float]) -> int:
        key = tuple(round(coord, 9) for coord in vertex)
        idx = vertex_ids.get(key)
        if idx is None:
            idx = len(vertices)
            vertex_ids[key] = idx
            vertices.append(key)
        return idx

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if not occupied[ix, iy, iz]:
                    continue
                bounds = (xs[ix], xs[ix + 1], ys[iy], ys[iy + 1], zs[iz], zs[iz + 1])
                for (dx, dy, dz), make_face in FACE_DEFS:
                    jx, jy, jz = ix + dx, iy + dy, iz + dz
                    outside = jx < 0 or jy < 0 or jz < 0 or jx >= nx or jy >= ny or jz >= nz
                    if outside or not occupied[jx, jy, jz]:
                        faces.append(tuple(vertex_index(vertex) for vertex in make_face(*bounds)))
    return vertices, faces


def validate(vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int, int]]) -> dict[str, float | int]:
    edge_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    directed_edges: defaultdict[tuple[int, int], int] = defaultdict(int)
    vertex_faces: defaultdict[int, list[int]] = defaultdict(list)

    for face_index, face in enumerate(faces):
        for vertex in face:
            vertex_faces[vertex].append(face_index)
        for a, b in zip(face, face[1:] + face[:1]):
            edge_counts[tuple(sorted((a, b)))] += 1
            directed_edges[(a, b)] += 1

    bad_edges = [edge for edge, count in edge_counts.items() if count != 2]
    if bad_edges:
        raise RuntimeError(f"Mesh is not closed/manifold at edges; bad edge count: {len(bad_edges)}")

    bad_orientation = [
        edge for edge, count in directed_edges.items() if count != 1 or directed_edges.get((edge[1], edge[0]), 0) != 1
    ]
    if bad_orientation:
        raise RuntimeError(f"Mesh has inconsistent face orientation at {len(bad_orientation)} directed edges")

    for vertex, incident_faces in vertex_faces.items():
        face_neighbors = {face_index: set() for face_index in incident_faces}
        incident_set = set(incident_faces)
        for face_index in incident_faces:
            face = faces[face_index]
            vertex_pos = face.index(vertex)
            adjacent_vertices = (face[vertex_pos - 1], face[(vertex_pos + 1) % len(face)])
            for other_index in incident_faces:
                if other_index == face_index:
                    continue
                other = faces[other_index]
                if any(adjacent_vertex in other for adjacent_vertex in adjacent_vertices):
                    face_neighbors[face_index].add(other_index)
        queue = deque([incident_faces[0]])
        seen = set()
        while queue:
            face_index = queue.popleft()
            if face_index in seen:
                continue
            seen.add(face_index)
            queue.extend(face_neighbors[face_index] - seen)
        if seen != incident_set:
            raise RuntimeError(f"Non-manifold vertex neighborhood at vertex {vertex}")

    verts = np.asarray(vertices, dtype=float)
    volume = 0.0
    area = 0.0
    for face in faces:
        points = verts[list(face)]
        for a, b, c in ((points[0], points[1], points[2]), (points[0], points[2], points[3])):
            volume += float(np.dot(a, np.cross(b, c))) / 6.0
            area += float(np.linalg.norm(np.cross(b - a, c - a))) * 0.5
    if volume <= 0.0:
        raise RuntimeError(f"Expected positive outward-oriented volume, got {volume}")

    return {
        "vertices": len(vertices),
        "faces": len(faces),
        "edges": len(edge_counts),
        "surface_area_m2": area,
        "volume_m3": volume,
        "bbox_min_x": float(verts[:, 0].min()),
        "bbox_max_x": float(verts[:, 0].max()),
        "bbox_min_y": float(verts[:, 1].min()),
        "bbox_max_y": float(verts[:, 1].max()),
        "bbox_min_z": float(verts[:, 2].min()),
        "bbox_max_z": float(verts[:, 2].max()),
    }


def write_obj(vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int, int]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OBJ_PATH.open("w", encoding="ascii") as obj:
        obj.write("# Simple white cable-routing snap clip\n")
        obj.write("# Units: meters\n")
        obj.write("# Coordinate system: Z up\n")
        obj.write("# Origin: bottom-face center\n")
        obj.write("# Bounding box: 0.060 m x 0.044 m x 0.034 m\n")
        obj.write("# Cable channel: along X axis; centered on Y=0\n")
        obj.write("# Mounting holes: two square through-holes in the base plate\n\n")
        obj.write(f"mtllib {MTL_PATH.name}\n")
        obj.write(f"o {ASSET_NAME}\n\n")
        for x, y, z in vertices:
            obj.write(f"v {x: .9f} {y: .9f} {z: .9f}\n")
        obj.write(f"\nusemtl {ASSET_NAME}\n")
        obj.write("s off\n\n")
        for face in faces:
            obj.write("f " + " ".join(str(index + 1) for index in face) + "\n")


def write_mtl() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with MTL_PATH.open("w", encoding="ascii") as mtl:
        mtl.write("# Simple white cable-routing snap clip material\n")
        mtl.write("# Units: meters\n\n")
        mtl.write(f"newmtl {ASSET_NAME}\n")
        mtl.write("Ka 0.960000 0.960000 0.930000\n")
        mtl.write("Kd 0.960000 0.960000 0.930000\n")
        mtl.write("Ks 0.040000 0.040000 0.040000\n")
        mtl.write("Ns 20.000000\n")
        mtl.write("d 1.000000\n")
        mtl.write("illum 2\n")


def main() -> None:
    vertices, faces = boundary_mesh()
    stats = validate(vertices, faces)
    write_mtl()
    write_obj(vertices, faces)
    print(f"Wrote {OBJ_PATH}")
    print(f"Wrote {MTL_PATH}")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
