"""Add the markerless "gel + speckle" tactile image (`gel_particle`) to an ALREADY-COLLECTED
hdf5 — no re-simulation. Reproduces the flow-coupled gel_particle representation:

  - BRIGHT SILVER base from the clean optical `rgb` luminance;
  - a fine, vivid colored speckle coating (the "gel coating");
  - the coating FLOWS with the contact: TANGENTIAL flow from the stored `marker` motion +
    NORMAL flow from the `depth` indentation gradient (the press pushes particles outward, like
    the gel bulging around the indenter) -> particle motion is coupled to the contact, not static;
  - a subtle per-frame contact shade from `depth`.

Faithful offline reconstruction of the in-sim `get_gel_particle_image` look (envs/sensors/
tactile.py, PATCH-G). Requires per gel: `rgb` (clean Taxim gel), `marker` (T,2,64,2), `depth`
(T,H,W). See docs/GelParticle.md.

Usage:
  python scripts/sensors/gelsight/add_gel_particle_offline.py IN.hdf5 --preview /tmp/gp.png
  python scripts/sensors/gelsight/add_gel_particle_offline.py IN.hdf5 --inplace
  python scripts/sensors/gelsight/add_gel_particle_offline.py IN.hdf5 --out OUT.hdf5
  # --gels auto (default) processes every gel incl. dual-arm _b; --bulge tunes the normal-press flow
"""
import argparse, shutil, numpy as np, cv2, h5py

ap = argparse.ArgumentParser()
ap.add_argument("hdf5")
ap.add_argument("--gels", default="auto", help="'auto' = every gel under tactile/ (incl. dual-arm _b); or comma list")
ap.add_argument("--preview", default=None, help="write a 6-phase sheet PNG and exit (no hdf5 write)")
ap.add_argument("--out", default=None, help="write result to this NEW hdf5 (copy of input + gel_particle)")
ap.add_argument("--inplace", action="store_true", help="add gel_particle into the input hdf5 in place")
ap.add_argument("--bulge", type=float, default=0.5, help="normal-press flow strength (depth-gradient); 0 = tangential only")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()


def dec(b):
    a = cv2.imdecode(np.frombuffer(bytes(b), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def build_coating(H, W, seed):
    """Fine, vivid, static colored speckle coating + neutral grain."""
    rs = np.random.RandomState(seed)
    n = int(0.09 * H * W)
    xs = rs.uniform(0, W, n); ys = rs.uniform(0, H, n)
    hue = rs.uniform(0, 180, n).astype(np.uint8); sat = rs.uniform(120, 210, n).astype(np.uint8)
    val = rs.uniform(200, 255, n).astype(np.uint8)
    cols = cv2.cvtColor(np.stack([hue, sat, val], 1)[None], cv2.COLOR_HSV2RGB)[0].astype(np.float32)
    coat = np.zeros((H, W, 3), np.float32); alpha = np.zeros((H, W), np.float32)
    for x, y, c, a in zip(xs, ys, cols, rs.uniform(0.6, 1.0, n)):
        cv2.circle(coat, (int(x), int(y)), 1, c.tolist(), -1, cv2.LINE_AA)
        cv2.circle(alpha, (int(x), int(y)), 1, float(a), -1, cv2.LINE_AA)
    coat = cv2.GaussianBlur(coat, (0, 0), 0.5); alpha = cv2.GaussianBlur(alpha, (0, 0), 0.5)
    grain = cv2.GaussianBlur(np.random.RandomState(seed + 1).randn(H, W).astype(np.float32), (0, 0), 0.7)
    grain /= grain.std() + 1e-6
    return coat, alpha, grain


class GelRenderer:
    """Silver rgb base + speckle coating advected by (tangential marker flow + normal depth flow)."""
    def __init__(self, marker, depth, H, W, seed, bulge):
        self.H, self.W, self.bulge = H, W, bulge
        self.coat, self.alpha, self.grain = build_coating(H, W, seed)
        self.MX, self.MY = np.meshgrid(np.arange(W), np.arange(H))
        var = np.array([marker[:, s].std(0).mean() for s in range(marker.shape[1])])
        self.cur = int(var.argmax())                                   # moving slot; frame-0 = reference
        self.ref_uv = marker[0, self.cur].astype(np.float32)
        self.marker = marker
        T = len(depth) if depth is not None else 0
        # per-pixel no-contact baseline (contact only lowers height) -> absolute indentation
        self.D0 = depth[::max(1, T // 20)].astype(np.float32).max(0) if depth is not None else None

    def disp_field(self, t, depth_t):
        # tangential: scatter marker flow -> normalized-convolution dense field
        d = self.marker[t, self.cur].astype(np.float32) - self.ref_uv
        d = d - d.mean(0, keepdims=True)                               # drop rigid motion
        s = 8; dh, dw = self.H // s, self.W // s
        ax = np.zeros((dh, dw), np.float32); ay = np.zeros((dh, dw), np.float32); cn = np.zeros((dh, dw), np.float32)
        xi = np.clip((self.ref_uv[:, 0] / s).astype(int), 0, dw - 1)
        yi = np.clip((self.ref_uv[:, 1] / s).astype(int), 0, dh - 1)
        np.add.at(ax, (yi, xi), d[:, 0]); np.add.at(ay, (yi, xi), d[:, 1]); np.add.at(cn, (yi, xi), 1.0)
        wc = cv2.GaussianBlur(cn, (0, 0), 2.0) + 1e-3
        DX = cv2.resize(cv2.GaussianBlur(ax, (0, 0), 2.0) / wc, (self.W, self.H))
        DY = cv2.resize(cv2.GaussianBlur(ay, (0, 0), 2.0) / wc, (self.W, self.H))
        # normal: the press pushes particles down the indentation gradient (bulge outward)
        if self.D0 is not None and self.bulge > 0 and depth_t is not None:
            indent = cv2.GaussianBlur(np.clip(self.D0 - depth_t.astype(np.float32), 0, None), (0, 0), 4.0)
            DX = DX - cv2.Sobel(indent, cv2.CV_32F, 1, 0, ksize=5) * self.bulge
            DY = DY - cv2.Sobel(indent, cv2.CV_32F, 0, 1, ksize=5) * self.bulge
        return DX.astype(np.float32), DY.astype(np.float32)

    def frame(self, rgb, depth_t, t):
        H, W = self.H, self.W
        lum = cv2.GaussianBlur(rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32), (0, 0), 1.5)
        g0 = np.clip(172.0 + (lum - lum.mean()) * 1.1, 128, 214)       # bright silver base
        gel = np.stack([g0 * 0.985, g0 * 1.0, g0 * 1.03], -1) + self.grain[..., None] * 3.0
        DX, DY = self.disp_field(t, depth_t)
        mapx = (self.MX - DX).astype(np.float32); mapy = (self.MY - DY).astype(np.float32)
        sp = cv2.remap(self.coat, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        al = cv2.remap(self.alpha, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        a = np.clip(al * 0.45, 0, 1)[..., None]
        out = gel * (1.0 - a) + sp * a
        if depth_t is not None:                                        # subtle per-frame contact shade
            dd = depth_t.astype(np.float32); dd = np.clip((dd - dd.min()) / (np.ptp(dd) + 1e-6), 0, 1)
            out = out * (1.0 - 0.10 * dd[..., None])
        return np.clip(out, 0, 255).astype(np.uint8)


def render_gel(f, gel, seed, bulge, preview_only):
    g = f["tactile/" + gel]
    rgb_raw = g["rgb"][:]; marker = g["marker"][:]
    depth = g["depth"] if "depth" in g else None
    T = len(rgb_raw); H, W = dec(rgb_raw[0]).shape[:2]
    R = GelRenderer(marker, depth, H, W, seed, bulge)
    dt = lambda t: (depth[t] if depth is not None else None)
    if preview_only:
        if depth is not None:                                          # phases by contact strength
            mag = np.array([np.clip(R.D0 - depth[t].astype(np.float32), 0, None).mean() for t in range(T)])
        else:
            mag = np.array([np.abs(marker[t, R.cur] - R.ref_uv).mean() for t in range(T)])
        pk = int(mag.argmax())
        idx = [0, pk // 2, pk, min(T - 1, pk + (T - pk) // 3), min(T - 1, pk + 2 * (T - pk) // 3), T - 1]
        labs = ["approach", "contact", "PEAK", "hold", "ease", "release"]
        tiles = []
        for i, lab in zip(idx, labs):
            im = R.frame(dec(rgb_raw[i]), dt(i), i)
            cv2.putText(im, f"{lab} f{i}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 2, cv2.LINE_AA)
            cv2.putText(im, f"{lab} f{i}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
            tiles.append(im)
        return np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:], 1)], 0), None
    out = np.empty((T, H, W, 3), np.uint8)
    for t in range(T):
        out[t] = R.frame(dec(rgb_raw[t]), dt(t), t)
    return None, out


if args.gels in ("auto", "all", ""):                        # auto-detect every gel (incl. dual-arm _b)
    with h5py.File(args.hdf5, "r") as _f:
        gels = list(_f["tactile"].keys()) if "tactile" in _f else []
else:
    gels = [x for x in args.gels.split(",") if x]

if args.preview:
    with h5py.File(args.hdf5, "r") as f:
        gel = gels[0] if ("tactile/" + gels[0]) in f else list(f["tactile"].keys())[0]
        sheet, _ = render_gel(f, gel, args.seed, args.bulge, True)
    cv2.imwrite(args.preview, sheet[:, :, ::-1])
    print("preview saved:", args.preview, "gel:", gel)
else:
    dst = args.hdf5 if args.inplace else args.out
    assert dst, "give --out OUT.hdf5 or --inplace"
    if not args.inplace:
        shutil.copyfile(args.hdf5, dst)
    with h5py.File(dst, "a") as f:
        for gel in gels:
            node = "tactile/" + gel
            if node not in f:
                continue
            if "gel_particle" in f[node]:
                del f[node]["gel_particle"]
            _, arr = render_gel(f, gel, args.seed, args.bulge, False)
            f[node].create_dataset("gel_particle", data=arr, compression="gzip", compression_opts=4)
            print(f"  wrote {node}/gel_particle {arr.shape}")
    print("done:", dst)
