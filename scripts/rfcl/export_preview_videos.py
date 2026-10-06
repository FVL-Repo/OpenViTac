"""Export four-panel MP4 previews from re-recorded RFCL HDF5 trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import numpy as np


PANELS = (
    ("observation/head/rgb", "Head RGB"),
    ("observation/wrist/rgb", "Wrist RGB"),
    ("tactile/left_tactile/rgb_marker", "Left tactile"),
    ("tactile/right_tactile/rgb_marker", "Right tactile"),
)
NATIVE_WIDTH = 960
NATIVE_HEIGHT = 320
NATIVE_TACTILE_SIZE = 160


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=NATIVE_WIDTH)
    parser.add_argument("--height", type=int, default=NATIVE_HEIGHT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def decode_image(value: object, *, path: Path, dataset: str, frame: int) -> np.ndarray:
    if isinstance(value, np.ndarray):
        encoded = np.asarray(value, dtype=np.uint8).reshape(-1)
    else:
        encoded = np.frombuffer(value, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode {path}:{dataset}[{frame}]")
    return image


def build_frame(
    handle: h5py.File,
    frame_index: int,
    *,
    width: int,
    height: int,
    source: Path,
) -> np.ndarray:
    images = {
        dataset: decode_image(
            handle[dataset][frame_index],
            path=source,
            dataset=dataset,
            frame=frame_index,
        )
        for dataset, _ in PANELS
    }
    head = cv2.resize(images[PANELS[0][0]], (480, 320))
    wrist = cv2.resize(images[PANELS[1][0]], (480, 320))
    left_tactile = cv2.resize(
        images[PANELS[2][0]],
        (NATIVE_TACTILE_SIZE, NATIVE_TACTILE_SIZE),
    )
    right_tactile = cv2.resize(
        images[PANELS[3][0]],
        (NATIVE_TACTILE_SIZE, NATIVE_TACTILE_SIZE),
    )
    native_canvas = np.zeros((320, 1120, 3), dtype=np.uint8)
    native_canvas[:, :480] = head
    native_canvas[:, 480:960] = wrist
    native_canvas[:NATIVE_TACTILE_SIZE, 960:] = left_tactile
    native_canvas[NATIVE_TACTILE_SIZE:, 960:] = right_tactile
    return cv2.resize(native_canvas, (width, height))


def video_info(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        return {
            "frames": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
            "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        }
    finally:
        capture.release()


def export_video(
    source: Path,
    destination: Path,
    *,
    fps: float,
    width: int,
    height: int,
) -> dict[str, float | int | str]:
    partial = destination.with_suffix(".partial.mp4")
    partial.unlink(missing_ok=True)
    start = time.perf_counter()
    with h5py.File(source, "r") as handle:
        missing = [dataset for dataset, _ in PANELS if dataset not in handle]
        if missing:
            raise ValueError(f"{source} is missing datasets: {missing}")
        lengths = [len(handle[dataset]) for dataset, _ in PANELS]
        if len(set(lengths)) != 1:
            raise ValueError(f"Panel lengths disagree in {source}: {lengths}")
        frame_count = lengths[0]
        process = subprocess.Popen(
            (
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                "-",
                "-pix_fmt",
                "yuv420p",
                "-vcodec",
                "libx264",
                "-crf",
                "23",
                "-movflags",
                "+faststart",
                str(partial),
            ),
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            if process.stdin is None:
                raise RuntimeError("ffmpeg stdin is unavailable")
            for frame_index in range(frame_count):
                frame = build_frame(
                    handle,
                    frame_index,
                    width=width,
                    height=height,
                    source=source,
                )
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            stderr = process.stderr.read() if process.stderr is not None else b""
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(
                    f"ffmpeg failed for {source.name}: {stderr.decode(errors='replace')}"
                )
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            partial.unlink(missing_ok=True)
            raise
    info = video_info(partial)
    expected = {
        "frames": frame_count,
        "width": width,
        "height": height,
    }
    for key, value in expected.items():
        if info[key] != value:
            partial.unlink(missing_ok=True)
            raise ValueError(
                f"Video validation failed for {source.name}: {key}={info[key]} != {value}"
            )
    os.replace(partial, destination)
    return {
        "source": str(source.resolve()),
        "video": str(destination.resolve()),
        **info,
        "size_bytes": destination.stat().st_size,
        "elapsed_s": time.perf_counter() - start,
    }


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if (args.width, args.height) != (NATIVE_WIDTH, NATIVE_HEIGHT):
        raise ValueError(
            f"OpenViTac GelSight previews require {NATIVE_WIDTH}x{NATIVE_HEIGHT}"
        )
    sources = sorted(args.input.glob("*.hdf5"))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        sources = sources[: args.limit]
    if not sources:
        raise ValueError(f"No HDF5 files found in {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = 0
    for index, source in enumerate(sources, start=1):
        destination = args.output / f"{source.stem}.mp4"
        if args.resume and destination.is_file():
            with h5py.File(source, "r") as handle:
                expected_frames = len(handle[PANELS[0][0]])
            try:
                info = video_info(destination)
            except Exception:
                pass
            else:
                if (
                    info["frames"] == expected_frames
                    and info["width"] == args.width
                    and info["height"] == args.height
                ):
                    skipped += 1
                    print(
                        f"[preview-export] skip {index}/{len(sources)} "
                        f"frames={expected_frames} {destination.name}",
                        flush=True,
                    )
                    continue
        row = export_video(
            source,
            destination,
            fps=args.fps,
            width=args.width,
            height=args.height,
        )
        rows.append(row)
        print(
            f"[preview-export] item={index}/{len(sources)} "
            f"frames={row['frames']} elapsed_s={row['elapsed_s']:.2f} "
            f"{destination.name}",
            flush=True,
        )
    manifest_path = args.output / "manifest.csv"
    all_videos = sorted(args.output.glob("*.mp4"))
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("video", "frames", "width", "height", "fps", "size_bytes"))
        for video in all_videos:
            info = video_info(video)
            writer.writerow(
                (
                    video.name,
                    info["frames"],
                    info["width"],
                    info["height"],
                    f"{info['fps']:.3f}",
                    video.stat().st_size,
                )
            )
    summary = {
        "schema": "rfcl_preview_video_export_v1",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "selected": len(sources),
        "videos": len(all_videos),
        "exported_this_run": len(rows),
        "skipped_this_run": skipped,
        "fps": args.fps,
        "resolution": [args.width, args.height],
        "codec": "h264",
        "pixel_format": "yuv420p",
        "layout": "head | wrist | left_tactile/right_tactile",
        "color_order": "rgb24",
        "panels": [dataset for dataset, _ in PANELS],
        "size_bytes": sum(path.stat().st_size for path in all_videos),
        "finished_at": datetime.now().astimezone().isoformat(),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
