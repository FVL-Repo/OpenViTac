"""Offline replay of the tactile imaging pass from a recorded episode.

This is the same renderer the simulator runs, driven by the recorded state
instead of the live world: pad calibration, per-frame ``world <- camera``,
per-frame object poses and the object USD surfaces.  No Isaac app, no GPU, no
simulator: a frame renders in tens of milliseconds on the CPU, so appearance
parameters are tuned by looking at images rather than by re-running episodes.

    python -m openvitac_sim.tactile_replay EPISODE.h5 --frame 180 --out DIR

Every appearance/geometry knob lives in :class:`GsMiniOptics`; ``--params``
overrides it by name (see :data:`PARAMETERS` for the table and defaults).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .assets import asset
from .calibration import pinhole_matrix
from .imaging.objects import ObjectSurface
from .imaging.surface import BoolArray, FloatArray
from .mesh_io import load_usd_surface
from .optical import OpticalBackend, ProceduralOptical, build_optical_backend
from .sensor.gsmini import GSMINI_OPTICS, GsMiniOptics, TactileSensorVisual

#: ``(name, default, meaning)`` for every override accepted by ``--params``.
#: ``*_mm`` names are converted to the dataclass's metre fields.
PARAMETERS: tuple[tuple[str, str, str], ...] = (
    ("plane_height_mm", "4.0", "gel contact-face plane height, pad coordinates"),
    ("plane_offset_mm", "0.0", "signed offset added to the contact-face plane"),
    ("face_half_x_mm", "12.625", "optical-face half width, pad coordinates"),
    ("face_half_y_mm", "10.375", "optical-face half height, pad coordinates"),
    ("height_gain", "1.0", "height handed to the optical model, x physical"),
    ("marker_gain", "1.0", "marker move along the optical axis, x h"),
    ("aperture_inset_px", "0", "erosion of the face rectangle, pixels"),
    ("background_gain", "1.0", "procedural gel background brightness"),
    ("shade_blur_sigma", "2.0", "shading blur sigma, pixels"),
    ("marker_radius_px", "6.2", "printed-dot radius at the 240 px reference"),
    ("contact_gain", "0.5", "shading darkening per unit contact"),
    ("normal_gain", "0.08", "shading darkening per unit tilt"),
    ("contact_ramp_mm", "0.8", "height that reaches full contact shading"),
)


def optics_from_params(values: Mapping[str, float | str]) -> GsMiniOptics:
    """``GsMiniOptics`` with the named overrides applied; unknown names fail."""
    known = {name for name, _default, _help in PARAMETERS}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(
            f"unknown replay parameter(s) {unknown}; known: {sorted(known)}"
        )
    fields: dict[str, Any] = {}
    for name, value in values.items():
        number = float(value)
        if name in ("plane_height_mm", "plane_offset_mm", "contact_ramp_mm"):
            fields[name.removesuffix("_mm") + "_m"] = number * 1e-3
        elif name == "aperture_inset_px":
            fields[name] = int(number)
        else:
            fields[name] = number
    return dataclasses.replace(GSMINI_OPTICS, **fields)


def parse_params(raw: Sequence[str]) -> dict[str, float]:
    """Parse ``key=value[,key=value]`` arguments into a parameter mapping."""
    values: dict[str, float] = {}
    for item in raw:
        for pair in item.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(f"replay parameter {pair!r} is not key=value")
            key, value = pair.split("=", 1)
            values[key.strip()] = float(value)
    return values


@dataclass(frozen=True)
class ReplayPad:
    """One pad's static calibration from the recording."""

    side: str
    intrinsics: FloatArray
    camera_from_pad: FloatArray
    rest_vertices: FloatArray
    triangles: NDArray[np.int32]
    width: int
    height: int
    marker_rows: int
    marker_cols: int


@dataclass(frozen=True)
class ReplayObject:
    """One imaging object: its USD surface and per-frame render pose."""

    name: str
    surface: ObjectSurface
    poses: NDArray[np.float64]  # (T, 4, 4) world <- object


@dataclass(frozen=True)
class ReplayFrame:
    """One replayed pad frame: the measurement and the two images."""

    side: str
    height_mm: FloatArray
    object_depth_m: FloatArray
    tactile_rgb: NDArray[np.uint8]
    marker_rgb: NDArray[np.uint8]
    aperture: BoolArray


@dataclass(frozen=True)
class ReplaySource:
    """A whole recording, loaded once; the renderer is a pure function of it."""

    path: Path
    pads: Mapping[str, ReplayPad]
    objects: tuple[ReplayObject, ...]
    camera_poses: Mapping[str, NDArray[np.float64]]  # side -> (T, 4, 4)
    frame_count: int

    @classmethod
    def load(cls, path: str | Path) -> ReplaySource:
        """Read the calibration and per-frame poses a replay needs."""
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "offline replay needs h5py to read episode files"
            ) from exc

        episode = Path(path)
        # h5py's stubs do not model Group key iteration; the access layer is
        # thin and every array is converted explicitly below.
        handle: Any = h5py.File(episode, "r")
        try:
            frames: Any = handle["frames"]
            frame_count = int(frames["joint_pos"].shape[0])
            pads: dict[str, ReplayPad] = {}
            camera_poses: dict[str, NDArray[np.float64]] = {}
            meta_pads: Any = handle["meta/tactile"]
            for side in list(meta_pads.keys()):
                group: Any = meta_pads[side]
                if "camera_from_pad" not in group:
                    raise RuntimeError(
                        f"episode lacks meta/tactile/{side}/camera_from_pad; "
                        "re-record with the current writer"
                    )
                attrs = group.attrs
                width = int(attrs["width"])
                height = int(attrs["height"])
                intrinsics = pinhole_matrix(
                    focal_length=float(attrs["depth_focal_length"]),
                    horizontal_aperture=float(attrs["depth_horizontal_aperture"]),
                    vertical_aperture=float(attrs["depth_vertical_aperture"]),
                    width=width,
                    height=height,
                )
                tactile: Any = frames["tactile"][side]
                if "camera_pose" not in tactile:
                    raise RuntimeError(
                        f"episode lacks frames/tactile/{side}/camera_pose; "
                        "re-record with the current writer"
                    )
                camera_poses[side] = np.asarray(tactile["camera_pose"], np.float64)
                pads[side] = ReplayPad(
                    side=side,
                    intrinsics=intrinsics,
                    camera_from_pad=np.asarray(group["camera_from_pad"], np.float64),
                    rest_vertices=np.asarray(group["rest_vertices"], np.float64),
                    triangles=np.asarray(group["surface_triangles"], np.int32),
                    width=width,
                    height=height,
                    marker_rows=int(attrs.get("marker_rows", 7)),
                    marker_cols=int(attrs.get("marker_cols", 9)),
                )
            objects: list[ReplayObject] = []
            objects_group: Any = handle["meta/objects"]
            object_frames: Any = frames["objects"]
            for name in list(objects_group.keys()):
                group = objects_group[name]
                if not bool(group.attrs.get("imaging", False)):
                    continue
                usd = group.attrs.get("usd")
                if usd is None:
                    raise RuntimeError(
                        f"imaging object {name!r} has no recorded USD path"
                    )
                vertices, triangles = load_usd_surface(asset(str(usd)))
                surface = ObjectSurface.from_mesh(
                    body=str(name),
                    prim_path=str(group.attrs.get("prim_path", name)),
                    vertices=vertices,
                    triangles=triangles,
                )
                objects.append(
                    ReplayObject(
                        name=str(name),
                        surface=surface,
                        poses=np.asarray(object_frames[name]["T"], np.float64),
                    )
                )
        finally:
            handle.close()
        return cls(
            path=episode,
            pads=pads,
            objects=tuple(objects),
            camera_poses=camera_poses,
            frame_count=frame_count,
        )

    def visuals(
        self, optics: GsMiniOptics = GSMINI_OPTICS
    ) -> dict[str, TactileSensorVisual]:
        """Build one calibrated visual per pad; caches nothing across calls."""
        visuals: dict[str, TactileSensorVisual] = {}
        for side, pad in self.pads.items():
            visuals[side] = TactileSensorVisual.bind(
                side=side,
                rest_vertices=pad.rest_vertices,
                camera_from_pad=pad.camera_from_pad,
                camera_intrinsics=pad.intrinsics,
                image_size=(pad.width, pad.height),
                objects=tuple(obj.surface for obj in self.objects),
                marker_rows=pad.marker_rows,
                marker_cols=pad.marker_cols,
                optics=optics,
            )
        return visuals

    def render(
        self,
        frame: int,
        *,
        optics: GsMiniOptics = GSMINI_OPTICS,
        backend: str = "procedural",
        device: str = "cpu",
        visuals: Mapping[str, TactileSensorVisual] | None = None,
    ) -> dict[str, ReplayFrame]:
        """Render every pad of one recorded frame from recorded state."""
        if not 0 <= frame < self.frame_count:
            raise IndexError(
                f"frame {frame} outside the recording (0..{self.frame_count - 1})"
            )
        visuals = self.visuals(optics) if visuals is None else visuals
        poses = {obj.name: obj.poses[frame] for obj in self.objects}
        out: dict[str, ReplayFrame] = {}
        for side, visual in visuals.items():
            depth = visual.object_depth(poses, self.camera_poses[side][frame])
            height = visual.height_map_mm(depth)
            optical = _optical(backend, visual, optics, device)
            images = optical.render(height_mm=height)
            out[side] = ReplayFrame(
                side=side,
                height_mm=height,
                object_depth_m=depth,
                tactile_rgb=images.tactile_rgb,
                marker_rgb=images.marker_rgb,
                aperture=images.aperture_mask,
            )
        return out


def _optical(
    backend: str, visual: TactileSensorVisual, optics: GsMiniOptics, device: str
) -> OpticalBackend:
    if backend == "procedural":
        return ProceduralOptical(visual)
    if backend == "taxim":
        calib_dir = optics.taxim_calib_dir()
        if calib_dir is None:
            raise FileNotFoundError("Taxim calibration is unavailable")
        return build_optical_backend(
            backend="taxim",
            visual=visual,
            calib_dir=calib_dir,
            device=device,
            height_gain=optics.height_gain,
        )
    raise ValueError(f"unknown replay backend {backend!r}")


def _write_png(path: Path, image: NDArray[np.uint8]) -> None:
    from PIL import Image

    Image.fromarray(image).save(path)


def _height_png(height_mm: FloatArray) -> NDArray[np.uint8]:
    """Grayscale height image, white = no contact, black = deepest contact."""
    peak = float(height_mm.max()) if height_mm.size else 0.0
    scale = 255.0 / peak if peak > 0.0 else 0.0
    return np.clip(255.0 - height_mm * scale, 0.0, 255.0).astype(np.uint8)


def main(argv: Sequence[str] | None = None) -> int:
    """Render recorded frames offline and write images plus a JSON summary."""
    parser = argparse.ArgumentParser(
        prog="python -m openvitac_sim.tactile_replay",
        description=(
            "Render the tactile imaging pass from a recorded episode without "
            "Isaac or a GPU.  The renderer is a pure function of recorded state."
        ),
    )
    parser.add_argument(
        "episode", type=Path, nargs="?", help="recorded episode.h5"
    )
    parser.add_argument(
        "--frame",
        type=int,
        action="append",
        default=[],
        help="frame index to render (repeatable)",
    )
    parser.add_argument("--out", type=Path, default=Path("."), help="output dir")
    parser.add_argument(
        "--params",
        action="append",
        default=[],
        metavar="KEY=VALUE[,KEY=VALUE]",
        help="tune GsMiniOptics; see PARAMETERS for names and defaults",
    )
    parser.add_argument(
        "--backend", choices=("procedural", "taxim"), default="procedural"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--list-params", action="store_true")
    args = parser.parse_args(argv)

    if args.list_params:
        for name, default, help_text in PARAMETERS:
            print(f"{name:22s} default={default:6s} {help_text}")
        return 0
    if args.episode is None or not args.frame:
        parser.error(
            "episode and --frame are required unless --list-params is given"
        )

    optics = optics_from_params(parse_params(args.params))
    source = ReplaySource.load(args.episode)
    visuals = source.visuals(optics)
    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "episode": str(args.episode),
        "params": {name: getattr(optics, _field(name)) for name, _d, _h in PARAMETERS},
        "frames": {},
    }
    for frame in args.frame:
        rendered = source.render(
            frame,
            optics=optics,
            backend=args.backend,
            device=args.device,
            visuals=visuals,
        )
        frame_summary: dict[str, Any] = {}
        for side, replay in rendered.items():
            stem = args.out / f"{side}_frame{frame:04d}"
            _write_png(stem.with_suffix(".tactile.png"), replay.tactile_rgb)
            _write_png(stem.with_suffix(".marker.png"), replay.marker_rgb)
            _write_png(stem.with_suffix(".height.png"), _height_png(replay.height_mm))
            height = replay.height_mm
            frame_summary[side] = {
                "height_max_mm": float(height.max()),
                "height_p99_mm": float(np.percentile(height, 99)),
                "area_above_0_2mm_px": int((height > 0.2).sum()),
                "contact_area_mm2": float(
                    (height > 0.2).sum() * _pixel_area_mm2(visuals[side])
                ),
            }
        summary["frames"][str(frame)] = frame_summary
        print(json.dumps({"frame": frame, **frame_summary}, indent=2))
    (args.out / "replay_summary.json").write_text(json.dumps(summary, indent=2))
    return 0


def _field(name: str) -> str:
    return name.removesuffix("_mm") + "_m" if name.endswith("_mm") else name


def _pixel_area_mm2(visual: TactileSensorVisual) -> float:
    """Metric area of one pixel at the rest plane, in square millimetres."""
    depth = visual.rest_plane_depth
    fx = visual.intrinsics[0, 0]
    fy = visual.intrinsics[1, 1]
    return float(np.median(depth**2) / (fx * fy) * 1e6)


if __name__ == "__main__":
    raise SystemExit(main())
