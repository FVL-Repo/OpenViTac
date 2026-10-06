#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "assets" / "objects"

PLUG_Y = 0.0124
PLUG_Z = 0.0044

SLOT_X = 0.05
SLOT_Y = 0.05
SLOT_Z = 0.02
HOLE_DEPTH = 0.015

# HOLE_CLEARANCES = {
#     "usb_slot_0_1.obj": 0.0005,
#     "usb_slot_0_2.obj": 0.001,
#     "usb_slot_0_3.obj": 0.0015,
# }

HOLE_CLEARANCES = {
    "usb_slot_target.obj": 0.001,
}


def write_obj(path: Path, vertices, faces, material_name: str, kd):
    mtl_path = path.with_suffix(".mtl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"mtllib {mtl_path.name}\n")
        f.write(f"o {path.stem}\n")
        for x, y, z in vertices:
            f.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")
        f.write(f"usemtl {material_name}\n")
        for face in faces:
            # OBJ uses 1-based indexing.
            f.write("f " + " ".join(str(i + 1) for i in face) + "\n")

    with mtl_path.open("w") as f:
        f.write(f"newmtl {material_name}\n")
        f.write("illum 2\n")
        f.write(f"Kd {kd[0]} {kd[1]} {kd[2]}\n")
        f.write("Ks 0.05 0.05 0.05\n")
        f.write("Ns 20\n")


def boundary_mesh(xs, ys, zs, occupied):
    vertices = []
    faces = []
    vertex_ids = {}

    def vertex_id(point):
        if point not in vertex_ids:
            vertex_ids[point] = len(vertices)
            vertices.append(point)
        return vertex_ids[point]

    def add_face(points):
        faces.append(tuple(vertex_id(point) for point in points))

    nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not occupied(i, j, k):
                    continue
                x0, x1 = xs[i], xs[i + 1]
                y0, y1 = ys[j], ys[j + 1]
                z0, z1 = zs[k], zs[k + 1]
                if i == 0 or not occupied(i - 1, j, k):
                    add_face([(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)])
                if i == nx - 1 or not occupied(i + 1, j, k):
                    add_face([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)])
                if j == 0 or not occupied(i, j - 1, k):
                    add_face([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)])
                if j == ny - 1 or not occupied(i, j + 1, k):
                    add_face([(x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)])
                if k == 0 or not occupied(i, j, k - 1):
                    add_face([(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)])
                if k == nz - 1 or not occupied(i, j, k + 1):
                    add_face([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)])
    return vertices, faces


def slot_mesh(size_x: float, size_y: float, size_z: float, hole_x: float, hole_y: float, hole_depth: float):
    ox = size_x * 0.5
    oy = size_y * 0.5
    hx = hole_x * 0.5
    hy = hole_y * 0.5
    z0, z1 = 0.0, size_z
    hz = size_z - hole_depth

    if hole_x >= size_x or hole_y >= size_y:
        raise ValueError("Hole is larger than slot base.")
    if not 0.0 < hole_depth < size_z:
        raise ValueError("Hole depth must be between 0 and slot thickness.")

    xs = sorted({-ox, -hx, hx, ox})
    ys = sorted({-oy, -hy, hy, oy})
    zs = [z0, hz, z1]

    def occupied(i, j, k):
        cx = (xs[i] + xs[i + 1]) * 0.5
        cy = (ys[j] + ys[j + 1]) * 0.5
        cz = (zs[k] + zs[k + 1]) * 0.5
        in_hole_void = abs(cx) < hx and abs(cy) < hy and hz < cz <= z1
        return not in_hole_void

    return boundary_mesh(xs, ys, zs, occupied)


def main():
    for filename, clearance in HOLE_CLEARANCES.items():
        hole_x = PLUG_Y + clearance
        hole_y = PLUG_Z + clearance
        vertices, faces = slot_mesh(SLOT_X, SLOT_Y, SLOT_Z, hole_x, hole_y, HOLE_DEPTH)
        write_obj(OUT_DIR / filename, vertices, faces, "usb_slot", kd=(0.9, 0.05, 0.04))
        print(
            f"{filename}: base=({SLOT_X}, {SLOT_Y}, {SLOT_Z}), "
            f"hole=({hole_x}, {hole_y}, depth={HOLE_DEPTH}), clearance={clearance}, min_z=0"
        )


if __name__ == "__main__":
    main()
