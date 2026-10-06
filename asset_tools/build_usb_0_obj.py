#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "assets" / "objects"

BODY_X = 0.020
BODY_Y = 0.010
BODY_Z = 0.050

PLUG_X = 0.0124
PLUG_Y = 0.0044
PLUG_Z = 0.0124


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


def usb_mesh():
    xs = sorted({-BODY_X * 0.5, -PLUG_X * 0.5, PLUG_X * 0.5, BODY_X * 0.5})
    ys = sorted({-BODY_Y * 0.5, -PLUG_Y * 0.5, PLUG_Y * 0.5, BODY_Y * 0.5})
    zs = [0.0, PLUG_Z, PLUG_Z + BODY_Z]

    def occupied(i, j, k):
        cx = (xs[i] + xs[i + 1]) * 0.5
        cy = (ys[j] + ys[j + 1]) * 0.5
        cz = (zs[k] + zs[k + 1]) * 0.5
        in_plug = abs(cx) <= PLUG_X * 0.5 and abs(cy) <= PLUG_Y * 0.5 and 0.0 <= cz <= PLUG_Z
        in_body = abs(cx) <= BODY_X * 0.5 and abs(cy) <= BODY_Y * 0.5 and PLUG_Z <= cz <= PLUG_Z + BODY_Z
        return in_plug or in_body

    return boundary_mesh(xs, ys, zs, occupied)


def write_obj(path, vertices, faces):
    path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = path.with_suffix(".mtl")
    material_name = "usb_black"
    black_kd = (0.02, 0.02, 0.02)

    with path.open("w") as f:
        f.write(f"mtllib {mtl_path.name}\n")
        f.write("o usb_0\n")
        for x, y, z in vertices:
            f.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")
        f.write(f"usemtl {material_name}\n")
        for face in faces:
            f.write("f " + " ".join(str(index + 1) for index in face) + "\n")

    with mtl_path.open("w") as f:
        f.write(f"newmtl {material_name}\n")
        f.write("illum 2\n")
        f.write(f"Kd {black_kd[0]} {black_kd[1]} {black_kd[2]}\n")
        f.write("Ks 0.05 0.05 0.05\n")
        f.write("Ns 20\n")


def main():
    vertices, faces = usb_mesh()
    write_obj(OUT_DIR / "usb_0.obj", vertices, faces)
    print(f"wrote {OUT_DIR / 'usb_0.obj'}")
    print(f"plug: x={PLUG_X}, y={PLUG_Y}, z={PLUG_Z}")
    print(f"body: x={BODY_X}, y={BODY_Y}, z={BODY_Z}")
    print(f"total z={PLUG_Z + BODY_Z}, min_z=0")


if __name__ == "__main__":
    main()
