#!/usr/bin/env python3
"""Build a simplified parametric gear asset.

The source gear.obj in assets/objects uses Y as the gear axis.  This generated
version is Z-up: the gear hole runs along Z, so it can sit on vertical posts.

Run:
    cd /path/to/OpenViTac
    python asset_tools/build_gear_obj.py

Convert to OpenViTac USD with tet data:
    python scripts/convert.py \
      -i assets/objects/gear_parametric.obj \
      -o assets/objects/gear_parametric.usd
"""

from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "assets" / "objects" / "gear_parametric.obj"

NUM_TEETH = 30
GEAR_ROOT_R = 0.0333
GEAR_OUTER_R = 0.0384
GEAR_THICKNESS = 0.010

BOSS_R = 0.0150
BOSS_TOTAL_Z = 0.030
HOLE_R = 0.003075

TOOTH_TIP_FRAC = 0.32
TOOTH_ROOT_FRAC = 0.92

GEAR_KD = (0.92, 0.62, 0.12)


def gear_profile_points():
    pitch = 2.0 * np.pi / NUM_TEETH
    tip_half = pitch * TOOTH_TIP_FRAC * 0.5
    root_half = pitch * TOOTH_ROOT_FRAC * 0.5
    points = []
    angles = []

    for tooth_id in range(NUM_TEETH):
        center = tooth_id * pitch
        for angle, radius in (
            (center - root_half, GEAR_ROOT_R),
            (center - tip_half, GEAR_OUTER_R),
            (center + tip_half, GEAR_OUTER_R),
            (center + root_half, GEAR_ROOT_R),
        ):
            points.append((radius * np.cos(angle), radius * np.sin(angle)))
            angles.append(angle)
    return np.asarray(points, dtype=float), np.asarray(angles, dtype=float)


def circle_points(radius: float, angles):
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def add_ring(vertices, points, z):
    start = len(vertices)
    for x, y in points:
        vertices.append((x, y, z))
    return np.arange(start, start + len(points), dtype=int)


def add_quad_strip(faces, a, b, flip=False):
    n = len(a)
    for i in range(n):
        j = (i + 1) % n
        if flip:
            faces.append((a[i], b[j], b[i]))
            faces.append((a[i], a[j], b[j]))
        else:
            faces.append((a[i], b[i], b[j]))
            faces.append((a[i], b[j], a[j]))


def build_gear_mesh():
    outer_points, angles = gear_profile_points()
    hub_points = circle_points(BOSS_R, angles)
    hole_points = circle_points(HOLE_R, angles)

    vertices = []
    faces = []

    outer_bottom = add_ring(vertices, outer_points, 0.0)
    outer_top = add_ring(vertices, outer_points, GEAR_THICKNESS)
    hub_bottom = add_ring(vertices, hub_points, GEAR_THICKNESS)
    hub_top = add_ring(vertices, hub_points, BOSS_TOTAL_Z)
    hole_bottom = add_ring(vertices, hole_points, 0.0)
    hole_top = add_ring(vertices, hole_points, BOSS_TOTAL_Z)

    # Boundary surfaces of the stepped gear volume.
    add_quad_strip(faces, outer_bottom, outer_top)
    add_quad_strip(faces, hole_bottom, outer_bottom, flip=True)
    add_quad_strip(faces, outer_top, hub_bottom)
    add_quad_strip(faces, hub_bottom, hub_top)
    add_quad_strip(faces, hub_top, hole_top)
    add_quad_strip(faces, hole_top, hole_bottom, flip=True)

    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=True)
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    mesh.apply_translation([0.0, 0.0, -mesh.bounds[0, 2]])
    return mesh


def write_mtl(path: Path, material_name: str, kd):
    with path.open("w") as f:
        f.write(f"newmtl {material_name}\n")
        f.write("illum 2\n")
        f.write(f"Kd {kd[0]} {kd[1]} {kd[2]}\n")
        f.write("Ks 0.08 0.08 0.08\n")
        f.write("Ns 40\n")


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


def main():
    mesh = build_gear_mesh()
    components = mesh.split(only_watertight=False)
    write_obj(OUT_PATH, mesh, "gear_parametric", "gear_yellow", GEAR_KD)
    print(f"wrote {OUT_PATH}")
    print(f"teeth={NUM_TEETH}")
    print(f"extents(mm)={np.round(mesh.extents * 1000, 3).tolist()}")
    print(f"bounds(mm)={np.round(mesh.bounds * 1000, 3).tolist()}")
    print(f"outer_diameter(mm)={GEAR_OUTER_R * 2000:.3f}")
    print(f"root_diameter(mm)={GEAR_ROOT_R * 2000:.3f}")
    print(f"boss_diameter(mm)={BOSS_R * 2000:.3f}")
    print(f"hole_diameter(mm)={HOLE_R * 2000:.3f}")
    print(f"watertight={mesh.is_watertight}")
    print(f"components={len(components)}")
    print(f"volume={mesh.volume:.9g}")


if __name__ == "__main__":
    main()
