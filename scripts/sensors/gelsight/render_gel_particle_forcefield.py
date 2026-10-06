"""[v2 video] Natural silver-gray markerless 'gel + force' video.
Left  = the tactile representation (silver gel + colorful speckle coating that advects with the
        real force_field + contact deformation blob from the Taxim luminance).
Right = the force_field normal-force heatmap + shear-vector arrows, so motion<->force is explicit.
"""
import sys, h5py, numpy as np, cv2

H5  = sys.argv[1] if len(sys.argv) > 1 else "data_force/lift_can/lift_can_force169/hdf5/0.hdf5"
GEL = sys.argv[2] if len(sys.argv) > 2 else "left_tactile"
OUT = sys.argv[3] if len(sys.argv) > 3 else "/tmp/gel_particle_v2.mp4"

f = h5py.File(H5, "r")
g = f[f"tactile/{GEL}"]
rgb_raw = g["rgb"][:]
ff = g["force_field"][:].astype(np.float32)
T = len(rgb_raw)

def dec(b):
    a = cv2.imdecode(np.frombuffer(bytes(b), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

H, W = dec(rgb_raw[0]).shape[:2]
RS = np.random.RandomState(0)

n = 4200
xs = RS.uniform(0, W, n); ys = RS.uniform(0, H, n)
hue = RS.uniform(0, 180, n).astype(np.uint8)
sat = RS.uniform(120, 210, n).astype(np.uint8)
val = RS.uniform(200, 255, n).astype(np.uint8)
cols = cv2.cvtColor(np.stack([hue, sat, val], 1)[None], cv2.COLOR_HSV2RGB)[0].astype(np.float32)
SPECK = np.zeros((H, W, 3), np.float32); ALPHA = np.zeros((H, W), np.float32)
for x, y, c, a in zip(xs, ys, cols, RS.uniform(0.6, 1.0, n)):
    cv2.circle(SPECK, (int(x), int(y)), 1, c.tolist(), -1, cv2.LINE_AA)
    cv2.circle(ALPHA, (int(x), int(y)), 1, float(a), -1, cv2.LINE_AA)
SPECK = cv2.GaussianBlur(SPECK, (0, 0), 0.5); ALPHA = cv2.GaussianBlur(ALPHA, (0, 0), 0.5)
GRAIN = cv2.GaussianBlur(RS.randn(H, W).astype(np.float32), (0, 0), 0.7); GRAIN /= GRAIN.std() + 1e-6

mxsh = np.linalg.norm(ff[..., :2], axis=-1).max() + 1e-9
mxn  = np.abs(ff[..., 2]).max() + 1e-9
SHEAR_SC = 12.0 / mxsh; NORM_SC = 90.0 / mxn
MX, MY = np.meshgrid(np.arange(W), np.arange(H))

def disp(fft):
    sh = cv2.resize(fft[..., :2], (W, H), cv2.INTER_LINEAR)
    N  = cv2.GaussianBlur(np.abs(cv2.resize(fft[..., 2], (W, H), cv2.INTER_LINEAR)), (0, 0), 6)
    gx = cv2.Sobel(N, cv2.CV_32F, 1, 0, ksize=5); gy = cv2.Sobel(N, cv2.CV_32F, 0, 1, ksize=5)
    dx = sh[..., 0] * SHEAR_SC - gx * NORM_SC
    dy = sh[..., 1] * SHEAR_SC - gy * NORM_SC
    return dx.astype(np.float32), dy.astype(np.float32), N

def base_silver(t):
    rgb = dec(rgb_raw[t]).astype(np.float32)
    lum = cv2.GaussianBlur(rgb @ np.array([0.299, 0.587, 0.114], np.float32), (0, 0), 1.5)
    g0 = np.clip(172.0 + (lum - lum.mean()) * 1.1, 128, 214)
    return np.stack([g0 * 0.985, g0 * 1.0, g0 * 1.03], -1) + GRAIN[..., None] * 3.0

def gel_panel(t):
    gel = base_silver(t)
    dx, dy, N = disp(ff[t])
    mapx = (MX - dx).astype(np.float32); mapy = (MY - dy).astype(np.float32)
    sp = cv2.remap(SPECK, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    al = cv2.remap(ALPHA, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    a = np.clip(al * 0.45, 0, 1)[..., None]
    out = gel * (1.0 - a) + sp * a
    shade = np.clip(N / (mxn * 0.9), 0, 1)[..., None]
    return np.clip(out * (1.0 - 0.12 * shade), 0, 255).astype(np.uint8)

def force_panel(t):
    N = np.abs(cv2.resize(ff[t][..., 2], (W, H), cv2.INTER_LINEAR))
    tt = np.clip(N / (mxn * 0.9), 0, 1)
    heat = cv2.applyColorMap((tt * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[:, :, ::-1].copy()
    # shear arrows on a coarse grid
    sh = ff[t][..., :2]; gh, gw = sh.shape[:2]
    for j in range(0, gh, 4):
        for i in range(0, gw, 4):
            x = int((i + 0.5) / gw * W); y = int((j + 0.5) / gh * H)
            vx = sh[j, i, 0] * SHEAR_SC * 1.4; vy = sh[j, i, 1] * SHEAR_SC * 1.4
            if abs(vx) + abs(vy) > 1.2:
                cv2.arrowedLine(heat, (x, y), (int(x + vx), int(y + vy)), (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.35)
    return heat

fz = np.abs(ff[..., 2]).reshape(T, -1).sum(1)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vw = cv2.VideoWriter(OUT, fourcc, 15.0, (W * 2 + 6, H))
for t in range(T):
    L = gel_panel(t); R = force_panel(t)
    cv2.putText(L, "markerless gel + force", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(L, "markerless gel + force", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(R, f"force_field  |fz|={fz[t]:.2f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    frame = np.concatenate([L, np.full((H, 6, 3), 50, np.uint8), R], axis=1)
    vw.write(frame[:, :, ::-1])
vw.release()
print("saved", OUT, "T", T)
