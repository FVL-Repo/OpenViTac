#!/usr/bin/env python3
"""Build the black base with two posts for the gear-pair visual check.

Run:
    cd /path/to/OpenViTac
    python asset_tools/build_gear_pair_base_obj.py

Convert to OpenViTac USD with tet data:
    python scripts/convert.py \
      -i assets/objects/gear_pair_base.obj \
      -o assets/objects/gear_pair_base.usd
"""

from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "assets" / "objects" / "gear_pair_base.obj"

BASE_X = 0.160
BASE_Y = 0.100
BASE_Z = 0.005

PAD_X = 0.020
PAD_Y = 0.020
PAD_Z = 0.002
PAD_OVERLAP = 0.0003

POST_CENTER_DISTANCE = 0.072
# gear_pair.obj 的中心孔直径约 14.15mm；柱子设成 12mm，
# 单边间隙约 (14.15-12)/2 = 1.075mm，能导向，也不容易卡死。
POST_DIAMETER = 0.012
POST_R = POST_DIAMETER * 0.5
POST_Z = 0.020
POST_OVERLAP = 0.0003
POST_SECTIONS = 64

BLACK_KD = (0.005, 0.005, 0.005)


def write_mtl(path: Path, material_name: str, kd):
    with path.open("w") as f:
        f.write(f"newmtl {material_name}\n")
        f.write("Ka 0.000000 0.000000 0.000000\n")
        f.write(f"Kd {kd[0]:.6f} {kd[1]:.6f} {kd[2]:.6f}\n")
        f.write("Ks 0.000000 0.000000 0.000000\n")
        f.write("Ns 32.000000\n")
        f.write("illum 2\n")


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
    base = trimesh.creation.box(extents=[BASE_X, BASE_Y, BASE_Z])
    base.apply_translation([0.0, 0.0, BASE_Z * 0.5])

    parts = [base]
    for x in (-POST_CENTER_DISTANCE * 0.5, POST_CENTER_DISTANCE * 0.5):
        pad = trimesh.creation.box(extents=[PAD_X, PAD_Y, PAD_Z])
        pad.apply_translation([x, 0.0, BASE_Z + PAD_Z * 0.5 - PAD_OVERLAP])
        parts.append(pad)

        post = trimesh.creation.cylinder(radius=POST_R, height=POST_Z, sections=POST_SECTIONS)
        post.apply_translation([x, 0.0, BASE_Z + PAD_Z + POST_Z * 0.5 - POST_OVERLAP])
        parts.append(post)

    mesh = trimesh.boolean.union(parts, engine="manifold")
    if isinstance(mesh, list):
        mesh = trimesh.util.concatenate(mesh)
    return clean_mesh(mesh)


def main():
    mesh = build_base_mesh()
    components = mesh.split(only_watertight=False)
    write_obj(OUT_PATH, mesh, "gear_pair_base", "gear_pair_base_black", BLACK_KD)
    print(f"wrote {OUT_PATH}")
    print(f"base_size(mm)={[BASE_X * 1000, BASE_Y * 1000, BASE_Z * 1000]}")
    print(f"pad_size(mm)={[PAD_X * 1000, PAD_Y * 1000, PAD_Z * 1000]}")
    print(f"post_center_distance(mm)={POST_CENTER_DISTANCE * 1000:.3f}")
    print(f"post_diameter(mm)={POST_DIAMETER * 1000:.3f}")
    print(f"post_height_above_base(mm)={POST_Z * 1000:.3f}")
    print(f"extents(mm)={np.round(mesh.extents * 1000, 3).tolist()}")
    print(f"bounds(mm)={np.round(mesh.bounds * 1000, 3).tolist()}")
    print(f"watertight={mesh.is_watertight}")
    print(f"components={len(components)}")
    print(f"volume={mesh.volume:.9g}")


if __name__ == "__main__":
    main()
