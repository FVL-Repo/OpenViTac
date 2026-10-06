"""[PATCH-E] Offline: rebuild the dense tactile force field from the stored per-vertex force.

During collection we store only `vertex_force` (per-vertex contact force in the sensor frame,
lossless) plus a small per-gel binding sidecar `ff_meta_<gel>.npz`. This script reconstructs the
dense `force_field` (H, W, 3) = barycentric interpolation of the per-vertex force onto the regular
grid -- exactly what the in-sim `get_force_field` produces, but computed offline (CPU, no Isaac/GPU)
so the expensive simulation runs only once and the grid can be regenerated any time.

Usage (from the repo root or anywhere):
  python scripts/sensors/gelsight/contact_force_to_field.py <run_dir_or_hdf5> [--out-key force_field]
    <run_dir_or_hdf5>  a collected run dir (…/<save_dir>/<task>/<config>) or a single .hdf5
    --out-key          dataset name to write under tactile/<gel>/ (default: force_field)
    --meta-dir         where ff_meta_<gel>.npz live (default: found next to / above the hdf5)

Needs only numpy + h5py.
"""
import argparse
import glob
import os

import h5py
import numpy as np


def find_meta(hdf5_path, gel, meta_dir=None):
    names = [f"ff_meta_{gel}.npz"]
    dirs = [meta_dir] if meta_dir else []
    d = os.path.dirname(os.path.abspath(hdf5_path))
    for _ in range(4):  # search the hdf5 dir and a few parents
        dirs.append(d)
        d = os.path.dirname(d)
    for dd in dirs:
        for n in names:
            p = os.path.join(dd, n)
            if os.path.exists(p):
                return p
    return None


def build_binding(surf_ref_xy, W, H):
    """Barycentric binding of a regular WxH grid to the surface mesh (mirrors the in-sim
    _precompute_force_field_map). Returns ff_verts (H*W,3), ff_bary (H*W,3), ff_valid (H*W,1)."""
    from scipy.spatial import Delaunay
    (xmin, ymin), (xmax, ymax) = surf_ref_xy.min(0), surf_ref_xy.max(0)
    GX, GY = np.meshgrid(np.linspace(xmin, xmax, W), np.linspace(ymin, ymax, H))
    pts = np.stack([GX.ravel(), GY.ravel()], axis=1)
    tri = Delaunay(surf_ref_xy)
    s = tri.find_simplex(pts)
    T = tri.transform[s]
    bc = np.einsum("nij,nj->ni", T[:, :2, :], pts - T[:, 2, :])
    bary = np.concatenate([bc, 1.0 - bc.sum(1, keepdims=True)], axis=1)
    verts = tri.simplices[s]
    valid = (s >= 0)
    verts[~valid] = 0
    bary[~valid] = 0.0
    return verts, bary.astype(np.float32), valid.astype(np.float32)[:, None]


def reconstruct(vf, meta, grid=None):
    """vf: (T, N_v, 3) per-vertex sensor-frame force. meta: ff_meta npz. -> (T, H, W, 3).
    grid=(W,H) rebuilds the binding at that resolution from the stored reference surface;
    None uses the collection-time binding in the meta."""
    surf_global = meta["surf_global"]              # nodal -> surface subset
    if grid is None:
        ff_verts, ff_bary, ff_valid = meta["ff_verts"], meta["ff_bary"].astype(np.float32), meta["ff_valid"].astype(np.float32)
        W, H = int(meta["grid"][0]), int(meta["grid"][1])
    else:
        W, H = int(grid[0]), int(grid[1])
        ff_verts, ff_bary, ff_valid = build_binding(meta["surf_ref_xy"], W, H)
    vf_surf = vf[:, surf_global, :]                                    # (T, N_surf, 3)
    fld = (vf_surf[:, ff_verts, :] * ff_bary[None, :, :, None]).sum(2)  # (T, H*W, 3)
    fld = fld * ff_valid[None]                                          # zero outside the hull
    return fld.reshape(vf.shape[0], H, W, 3).astype(np.float32)


def process_hdf5(path, out_key, meta_dir, grid=None):
    with h5py.File(path, "a") as f:
        if "tactile" not in f:
            print(f"  {path}: no tactile group, skip"); return
        for gel in f["tactile"]:
            g = f[f"tactile/{gel}"]
            if "vertex_force" not in g:
                continue
            meta_path = find_meta(path, gel, meta_dir)
            if meta_path is None:
                print(f"  {path} [{gel}]: ff_meta_{gel}.npz not found, skip"); continue
            meta = np.load(meta_path)
            vf = g["vertex_force"][:]
            field = reconstruct(vf, meta, grid=grid)
            if out_key in g:
                del g[out_key]
            g.create_dataset(out_key, data=field, compression="gzip")
            print(f"  {path} [{gel}]: {vf.shape} -> {out_key} {field.shape}")


def main():
    ap = argparse.ArgumentParser(description="Rebuild dense force_field from stored vertex_force")
    ap.add_argument("target", help="run dir or a single .hdf5")
    ap.add_argument("--out-key", default="force_field")
    ap.add_argument("--meta-dir", default=None)
    ap.add_argument("--grid", nargs=2, type=int, default=None, metavar=("W", "H"),
                    help="rebuild the force_field at this grid (default: the collection grid in the meta)")
    a = ap.parse_args()
    grid = tuple(a.grid) if a.grid else None
    if a.target.endswith(".hdf5") or a.target.endswith(".h5"):
        files = [a.target]
    else:
        files = sorted(glob.glob(os.path.join(a.target, "**", "*.hdf5"), recursive=True))
    if not files:
        print("no hdf5 found under", a.target); return
    print(f"reconstructing '{a.out_key}' for {len(files)} file(s)" + (f" at grid {grid}" if grid else ""))
    for p in files:
        process_hdf5(p, a.out_key, a.meta_dir, grid=grid)
    print("done")


if __name__ == "__main__":
    main()
