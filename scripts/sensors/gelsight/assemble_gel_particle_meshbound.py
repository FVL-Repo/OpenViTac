"""Assemble the mesh-bound gel_particle PNG side-channel dump into an mp4.
Left  = raw Taxim gel (rgb/)   Right = mesh-bound markerless representation (gp/), whose particle
motion is the GENUINE FEM surface displacement. One panel pair per saved frame.
"""
import sys, glob, os, numpy as np, cv2

DUMP = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gpdump"
GEL  = sys.argv[2] if len(sys.argv) > 2 else "left_tactile"
OUT  = sys.argv[3] if len(sys.argv) > 3 else "/tmp/gel_particle_meshbound.mp4"

gp  = sorted(glob.glob(os.path.join(DUMP, "gp",  f"{GEL}_*.png")))
rgb = sorted(glob.glob(os.path.join(DUMP, "rgb", f"{GEL}_*.png")))
assert gp, f"no gp frames for {GEL} in {DUMP}"
H, W = cv2.imread(gp[0]).shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vw = cv2.VideoWriter(OUT, fourcc, 15.0, (W * 2 + 6, H))
for i, gpf in enumerate(gp):
    g = cv2.imread(gpf)
    r = cv2.imread(rgb[i]) if i < len(rgb) else np.full_like(g, 128)
    cv2.putText(r, "raw Taxim gel", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(g, "mesh-bound gel+force", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(g, "mesh-bound gel+force", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
    vw.write(np.concatenate([r, np.full((H, 6, 3), 50, np.uint8), g], axis=1))
vw.release()
# also a 6-phase contact sheet from evenly spaced frames
idx = [int(k) for k in np.linspace(0, len(gp) - 1, 6)]
tiles = [cv2.imread(gp[k]) for k in idx]
sheet = np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:], 1)], 0)
cv2.imwrite(OUT.replace(".mp4", "_sheet.png"), sheet)
print("saved", OUT, "frames", len(gp), "sheet_idx", idx)
