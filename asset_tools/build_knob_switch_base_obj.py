#!/usr/bin/env python3
"""Build the fixed blue base for the knob-switch task.

Run:
    cd /path/to/OpenViTac
    python asset_tools/build_knob_switch_base_obj.py

Convert to OpenViTac USD with tet data:
    python scripts/convert.py \
      -i assets/objects/knob_switch_base.obj \
      -o assets/objects/knob_switch_base.usd
"""

from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "assets" / "objects" / "knob_switch_base.obj"

BASE_X = 0.080
BASE_Y = 0.080
BASE_Z = 0.010
BASE_CORNER_R = 0.004

POST_R = 0.009
POST_Z = 0.017
POST_OVERLAP = 0.0003

CORNER_SECTIONS = 8
POST_SECTIONS = 64

BLUE_KD = (0.02, 0.22, 0.85)


def rounded_rect_points(width: float, depth: float, radius: float, sections: int):
    hx = width * 0.5
    hy = depth * 0.5
    centers = [
        (hx - radius, hy - radius),
        (-hx + radius, hy - radius),
        (-hx + radius, -hy + radius),
        (hx - radius, -hy + radius),
    ]
    ranges = [
        (0.0, np.pi * 0.5),
        (np.pi * 0.5, np.pi),
        (np.pi, np.pi * 1.5),
        (np.pi * 1.5, np.pi * 2.0),
    ]

    points = []
    for center, angle_range in zip(centers, ranges):
        start, end = angle_range
        for angle in np.linspace(start, end, sections + 1)[:-1]:
            points.append((center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)))
    return points


def extrude_convex_polygon(points, height: float):
    vertices = []
    faces = []
    n = len(points)

    for x, y in points:
        vertices.append((x, y, 0.0))
    for x, y in points:
        vertices.append((x, y, height))

    bottom_center_id = len(vertices)
    vertices.append((0.0, 0.0, 0.0))
    top_center_id = len(vertices)
    vertices.append((0.0, 0.0, height))

    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j))
        faces.append((i, n + j, n + i))
        faces.append((bottom_center_id, j, i))
        faces.append((top_center_id, n + i, n + j))

    return trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=True)


def write_mtl(path: Path, material_name: str, kd):
    with path.open("w") as f:
        f.write(f"newmtl {material_name}\n")
        f.write("illum 2\n")
        f.write(f"Kd {kd[0]} {kd[1]} {kd[2]}\n")
        f.write("Ks 0.05 0.05 0.05\n")
        f.write("Ns 30\n")


def write_obj(path: Path, mesh: trimesh.Trimesh, object_name: str, material_name: str, kd):
    path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = path.with_suffix(".mtl")
    with path.open("w") as f:
        f.write(f"mtllib {mtl_path.name}\n")
        f.write(f"o {object_name}\n")
        for x, y, z in mesh.vertices:
            f.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")
        f.write(f"usemtl {material_name}\n")
        for face in mesh.faces:
            f.write("f " + " ".join(str(int(index) + 1) for index in face) + "\n")
    write_mtl(mtl_path, material_name, kd)


def clean_mesh(mesh: trimesh.Trimesh):
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    mesh.apply_translation([0.0, 0.0, -mesh.bounds[0, 2]])
    return mesh


def build_base_mesh():
    base_outline = rounded_rect_points(BASE_X, BASE_Y, BASE_CORNER_R, CORNER_SECTIONS)
    base = extrude_convex_polygon(base_outline, BASE_Z)

    post = trimesh.creation.cylinder(radius=POST_R, height=POST_Z, sections=POST_SECTIONS)
    post.apply_translation([0.0, 0.0, BASE_Z + POST_Z * 0.5 - POST_OVERLAP])

    mesh = trimesh.boolean.union([base, post], engine="manifold")
    if isinstance(mesh, list):
        mesh = trimesh.util.concatenate(mesh)

    return clean_mesh(mesh)


def export_mesh(path: Path, mesh: trimesh.Trimesh, object_name: str, material_name: str, kd):
    components = mesh.split(only_watertight=False)
    write_obj(path, mesh, object_name, material_name, kd)
    print(f"wrote {path}")
    print(f"extents(mm)={np.round(mesh.extents * 1000, 3).tolist()}")
    print(f"bounds(mm)={np.round(mesh.bounds * 1000, 3).tolist()}")
    print(f"watertight={mesh.is_watertight}")
    print(f"components={len(components)}")
    print(f"volume={mesh.volume:.9g}")


def main():
    export_mesh(
        OUT_PATH,
        build_base_mesh(),
        "knob_switch_base",
        "knob_switch_blue",
        BLUE_KD,
    )


if __name__ == "__main__":
    main()
