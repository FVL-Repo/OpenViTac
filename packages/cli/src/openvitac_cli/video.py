"""Render a recorded episode to video.

Video is always post-processing: the collector closes the HDF5 episode first,
then this module re-reads the finished file and pipes the recorded camera
frames to ffmpeg. Nothing here touches the simulation.
"""

from __future__ import annotations

import contextlib
import itertools
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import numpy.typing as npt

from openvitac_data import Frame, Hdf5EpisodeReader


def _even(value: int) -> int:
    """Round up to an even size; yuv420p requires even dimensions."""
    return value + value % 2


def _resize(
    image: npt.NDArray[np.uint8], height: int, width: int
) -> npt.NDArray[np.uint8]:
    """Nearest-neighbour resize; only used to align cameras of different sizes."""
    source_height, source_width = image.shape[:2]
    rows = np.minimum(np.arange(height) * source_height // height, source_height - 1)
    cols = np.minimum(np.arange(width) * source_width // width, source_width - 1)
    return image[rows[:, None], cols]


def _layout(
    images: dict[str, npt.NDArray[np.uint8]], roles: list[str]
) -> npt.NDArray[np.uint8]:
    """Place all camera streams left to right, aligned to a common height."""
    target_height = _even(max(images[role].shape[0] for role in roles))
    panels: list[npt.NDArray[np.uint8]] = []
    for role in roles:
        image = images[role]
        width = max(2, _even(round(image.shape[1] * target_height / image.shape[0])))
        panels.append(_resize(image, target_height, width))
    return np.concatenate(panels, axis=1)


_BENCHMARK_WIDTH = 1140
_BENCHMARK_HEIGHT = 270
_BENCHMARK_MARKER_WIDTH = 180
_BENCHMARK_MARKER_HEIGHT = 135


def _marker_panel(
    frame: Frame, side: str, width: int, height: int
) -> npt.NDArray[np.uint8] | None:
    """Return one pad's recorded marker image, if the episode has one."""
    if not frame.tactile:
        return None
    pad = frame.tactile.get(side)
    if pad is None or pad.marker_rgb is None:
        return None
    image = np.asarray(pad.marker_rgb)
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        return None
    return _resize(image.astype(np.uint8, copy=False), height, width)


def _layout_frame(
    frame: Frame, roles: list[str]
) -> npt.NDArray[np.uint8]:
    """Build the benchmark grid when head, wrist, and both markers exist.

    Layout is fixed for these evidence videos: head and wrist at 480x270,
    then left/right marker panels at 180x135 stacked in a 180x270 column.
    Episodes without both marker streams keep the generic side-by-side layout.
    """
    images = frame.images
    marker_left = _marker_panel(
        frame, "left", _BENCHMARK_MARKER_WIDTH, _BENCHMARK_MARKER_HEIGHT
    )
    marker_right = _marker_panel(
        frame, "right", _BENCHMARK_MARKER_WIDTH, _BENCHMARK_MARKER_HEIGHT
    )
    if (
        "head" not in images
        or "wrist" not in images
        or marker_left is None
        or marker_right is None
    ):
        return _layout(images, roles)
    canvas = np.zeros(
        (_BENCHMARK_HEIGHT, _BENCHMARK_WIDTH, 3), dtype=np.uint8
    )
    canvas[:, :480] = _resize(images["head"], 270, 480)
    canvas[:, 480:960] = _resize(images["wrist"], 270, 480)
    canvas[:135, 960:1140] = marker_left
    canvas[135:270, 960:1140] = marker_right
    return canvas


def _recorded_fps(episode: Path, timestamps_us: npt.NDArray[np.int64] | None) -> int:
    """Playback fps from recorded frame timestamps.

    Files without timestamps have no safe default: ``sim_dt`` describes the
    physics step, not a throttled recording cadence.
    """
    if timestamps_us is None:
        raise ValueError(
            f"{episode} has no recorded frame timestamps; pass fps explicitly"
        )
    if timestamps_us.size < 2:
        raise ValueError(f"{episode} has fewer than two frames; pass fps explicitly")
    intervals = np.diff(timestamps_us)
    if np.any(intervals <= 0):
        raise ValueError(
            f"{episode} has non-increasing frame timestamps; pass fps explicitly"
        )
    return max(1, round(1_000_000 / float(np.median(intervals))))


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    """Terminate a render process that failed mid-stream, then reap it."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def render_video(episode: Path, video: Path, fps: int | None = None) -> int:
    """Render ``episode`` into ``video`` and return the frame count.

    Cameras come from the episode metadata. Benchmark recordings with head,
    wrist, and both gelpad marker streams use a fixed 1140x270 grid; other
    episodes stitch their cameras side by side. No labels are drawn. ``fps``
    defaults to the rate implied by the recorded ``timestamp_us`` intervals so
    playback runs at wall-clock speed; pass a value to override it. Files
    without timestamps require an explicit ``fps``.
    """
    with Hdf5EpisodeReader(episode) as reader:
        if fps is None:
            fps = _recorded_fps(episode, reader.timestamps_us)
        elif fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        roles = [camera.role for camera in reader.meta.cameras]
        if not roles:
            raise ValueError(f"{episode} has no camera streams")
        frames = iter(reader)
        first = next(frames)
        first_frame = _layout_frame(first, roles)
        height, width = first_frame.shape[:2]
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
            str(video),
        ]
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdin = process.stdin
        stderr = process.stderr
        assert stdin is not None and stderr is not None
        count = 0
        try:
            for frame in itertools.chain((first,), frames):
                stdin.write(_layout_frame(frame, roles).tobytes())
                count += 1
        except BrokenPipeError:
            # ffmpeg died; its stderr below explains why.
            pass
        except BaseException as exc:
            _stop_process(process)
            stderr_bytes = stderr.read()
            exc.add_note(f"ffmpeg stderr: {stderr_bytes.decode(errors='replace')}")
            raise
        finally:
            with contextlib.suppress(OSError, ValueError):
                stdin.close()
        error = stderr.read()
        returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed with exit code {returncode}: "
            f"{error.decode(errors='replace')}"
        )
    return count
