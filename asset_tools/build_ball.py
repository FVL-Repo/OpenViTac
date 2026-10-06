#!/usr/bin/env python3
"""生成一个红色塑料球 OBJ 网格。

球使用 icosphere 生成, 三角面分布比较均匀, 适合后续转 USD 和烤 tet。
当前球直径为 30 mm, 单位写成米, 球心在原点。

SUBDIV 控制 icosphere 细分等级:
  - SUBDIV=3: 642 顶点 / 1280 面
  - SUBDIV=4: 2562 顶点 / 5120 面, 更圆, 但后续 tet/仿真更重

运行:
    cd /path/to/OpenViTac
    python asset_tools/build_ball.py

将生成的 OBJ 转成 OpenViTac USD:
    python scripts/convert.py \
      -i assets/objects/ball_red.obj \
      -o assets/objects/ball_red.usd
"""

from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "assets" / "objects" / "ball_red.obj"

DIAMETER = 0.030
RADIUS = DIAMETER * 0.5
SUBDIV = 4

RED_KD = (0.92, 0.02, 0.02)


def write_mtl(path: Path, material_name: str, kd):
    with path.open("w") as f:
        f.write(f"newmtl {material_name}\n")
        f.write(f"Ka {kd[0]:.6f} {kd[1]:.6f} {kd[2]:.6f}\n")
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
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    return mesh


def build_ball_mesh():
    return clean_mesh(trimesh.creation.icosphere(subdivisions=SUBDIV, radius=RADIUS))


def main():
    mesh = build_ball_mesh()
    components = mesh.split(only_watertight=False)
    write_obj(OUT_PATH, mesh, "ball_red", "ball_red", RED_KD)

    print(f"wrote {OUT_PATH}")
    print(f"diameter(mm)={DIAMETER * 1000:.3f}")
    print(f"subdiv={SUBDIV}")
    print(f"extents(mm)={np.round(mesh.extents * 1000, 3).tolist()}")
    print(f"bounds(mm)={np.round(mesh.bounds * 1000, 3).tolist()}")
    print(f"verts={len(mesh.vertices)} faces={len(mesh.faces)}")
    print(f"watertight={mesh.is_watertight}")
    print(f"components={len(components)}")
    print(f"volume={mesh.volume:.9g}")


if __name__ == "__main__":
    main()
