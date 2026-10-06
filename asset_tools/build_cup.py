#!/usr/bin/env python3
"""生成一个塑料杯 OBJ 网格。

这个杯子是轴对称的薄壁截锥杯: 杯口朝 local +Z, 杯底朝 local -Z, 底部封闭,
上口外翻形成一圈卷边。网格单位是米, 最低面落在 z=0, 方便直接放到桌面或作为
OpenViTac/UIPC 资产的表面网格输入。

当前尺寸来自实物近似测量:
  - 杯底外径: 46.5 mm
  - 杯口最大外径: 68.0 mm
  - 杯高: 92.5 mm
  - 卷边高度: 11.0 mm
  - 壁厚/底厚暂用经验值: 1.6 mm / 2.5 mm

脚本会一次生成三种颜色的同尺寸杯子:
  - assets/objects/cup_green.obj + cup_green.mtl
  - assets/objects/cup_blue.obj + cup_blue.mtl
  - assets/objects/cup_yellow.obj + cup_yellow.mtl

运行:
    cd /path/to/OpenViTac
    python asset_tools/build_cup.py

将某个生成的 OBJ 转成 OpenViTac USD:
    python scripts/convert.py \
      -i assets/objects/cup_green.obj \
      -o assets/objects/cup_green.usd
"""

from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "assets" / "objects"

# Geometry, in meters. Diameters are measured on the outside of the cup.
BASE_OUTER_DIAMETER = 0.0465
RIM_OUTER_DIAMETER = 0.0680
HEIGHT = 0.0925
WALL = 0.0016
BOTTOM = 0.0025
LIP_OUT = 0.0022
LIP_HEIGHT = 0.0110
SECTIONS = 96

R_BASE = BASE_OUTER_DIAMETER * 0.5
# R_TOP is the cup body radius just below the rolled rim. The maximum rim radius
# is R_TOP + LIP_OUT, so keep the visible mouth diameter at RIM_OUTER_DIAMETER.
R_TOP = RIM_OUTER_DIAMETER * 0.5 - LIP_OUT

CUP_VARIANTS = {
    "cup_green": (0.43, 0.82, 0.02),
    "cup_blue": (0.02, 0.28, 0.95),
    "cup_yellow": (1.00, 0.82, 0.02),
}


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
    mesh.apply_translation([0.0, 0.0, -mesh.bounds[0, 2]])
    return mesh


def build_cup_mesh():
    r_top_in = R_TOP - WALL
    r_base_in = R_BASE - WALL

    # Closed half-profile in (radius, z). Revolving it around Z creates the cup.
    profile = np.array([
        [0.0, 0.0],
        [R_BASE, 0.0],
        [R_TOP, HEIGHT - LIP_HEIGHT],
        [R_TOP + LIP_OUT, HEIGHT - LIP_HEIGHT],
        [R_TOP + LIP_OUT, HEIGHT],
        [r_top_in, HEIGHT],
        [r_base_in, BOTTOM],
        [0.0, BOTTOM],
    ])
    return clean_mesh(trimesh.creation.revolve(profile, sections=SECTIONS))


def main():
    mesh = build_cup_mesh()
    components = mesh.split(only_watertight=False)
    for name, kd in CUP_VARIANTS.items():
        out_path = OUT_DIR / f"{name}.obj"
        write_obj(out_path, mesh, name, name, kd)
        print(f"wrote {out_path}")

    print(f"rim_outer_diameter(mm)={(R_TOP + LIP_OUT) * 2 * 1000:.3f}")
    print(f"body_top_outer_diameter(mm)={R_TOP * 2 * 1000:.3f}")
    print(f"base_outer_diameter(mm)={R_BASE * 2 * 1000:.3f}")
    print(f"height(mm)={HEIGHT * 1000:.3f}")
    print(f"wall(mm)={WALL * 1000:.3f}")
    print(f"bottom(mm)={BOTTOM * 1000:.3f}")
    print(f"lip_out(mm)={LIP_OUT * 1000:.3f}")
    print(f"lip_height(mm)={LIP_HEIGHT * 1000:.3f}")
    print(f"extents(mm)={np.round(mesh.extents * 1000, 3).tolist()}")
    print(f"bounds(mm)={np.round(mesh.bounds * 1000, 3).tolist()}")
    print(f"verts={len(mesh.vertices)} faces={len(mesh.faces)}")
    print(f"watertight={mesh.is_watertight}")
    print(f"components={len(components)}")
    print(f"volume={mesh.volume:.9g}")


if __name__ == "__main__":
    main()
