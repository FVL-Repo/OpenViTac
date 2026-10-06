#!/usr/bin/env python3
"""Build the movable black cap for the knob-switch task.

Run:
    cd /path/to/OpenViTac
    python asset_tools/build_knob_switch_cap_obj.py

Convert to OpenViTac USD with tet data:
    python scripts/convert.py \
      -i assets/objects/knob_switch_cap.obj \
      -o assets/objects/knob_switch_cap.usd
"""

from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "assets" / "objects" / "knob_switch_cap.obj"

CAP_OUTER_R = 0.0105
CAP_Z = 0.020
CAP_HOLE_R = 0.0095
CAP_HOLE_DEPTH = 0.019
CAP_SECTIONS = 96

INDICATOR_BASE_X = 0.0095
INDICATOR_TIP_X = 0.019
INDICATOR_BASE_Y = 0.005

BLACK_KD = (0.02, 0.02, 0.02)


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


def build_indicator_mesh():
    half_y = INDICATOR_BASE_Y * 0.5
    vertices = np.array(
        [
            [INDICATOR_BASE_X, -half_y, 0.0],
            [INDICATOR_BASE_X, half_y, 0.0],
            [INDICATOR_TIP_X, 0.0, 0.0],
            [INDICATOR_BASE_X, -half_y, CAP_Z],
            [INDICATOR_BASE_X, half_y, CAP_Z],
            [INDICATOR_TIP_X, 0.0, CAP_Z],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [3, 4, 5],
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 5],
            [1, 5, 4],
            [2, 0, 3],
            [2, 3, 5],
        ],
        dtype=int,
    )
    indicator = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    indicator.fix_normals()
    return indicator


def build_cap_mesh():
    outer = trimesh.creation.cylinder(radius=CAP_OUTER_R, height=CAP_Z, sections=CAP_SECTIONS)
    outer.apply_translation([0.0, 0.0, CAP_Z * 0.5])

    hole = trimesh.creation.cylinder(radius=CAP_HOLE_R, height=CAP_HOLE_DEPTH + 0.001, sections=CAP_SECTIONS)
    hole.apply_translation([0.0, 0.0, (CAP_HOLE_DEPTH + 0.001) * 0.5 - 0.0005])

    cap = trimesh.boolean.difference([outer, hole], engine="manifold")
    if isinstance(cap, list):
        cap = trimesh.util.concatenate(cap)

    indicator = build_indicator_mesh()

    mesh = trimesh.boolean.union([cap, indicator], engine="manifold")
    if isinstance(mesh, list):
        mesh = trimesh.util.concatenate(mesh)

    return clean_mesh(mesh)


def main():
    mesh = build_cap_mesh()
    components = mesh.split(only_watertight=False)
    write_obj(OUT_PATH, mesh, "knob_switch_cap", "knob_switch_black", BLACK_KD)
    print(f"wrote {OUT_PATH}")
    print(f"extents(mm)={np.round(mesh.extents * 1000, 3).tolist()}")
    print(f"bounds(mm)={np.round(mesh.bounds * 1000, 3).tolist()}")
    print(f"watertight={mesh.is_watertight}")
    print(f"components={len(components)}")
    print(f"volume={mesh.volume:.9g}")


if __name__ == "__main__":
    main()
