"""Video rendering: default fps comes from recorded timestamps, not sim_dt."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from openvitac_cli import video
from openvitac_data import (
    ActionSpec,
    CameraMeta,
    EpisodeExtra,
    EpisodeMeta,
    Frame,
    Hdf5EpisodeWriter,
)

_IMAGE = np.zeros((3, 4, 3), dtype=np.uint8)


def _meta() -> EpisodeMeta[EpisodeExtra]:
    return EpisodeMeta(
        schema_version=3,
        task_id="test",
        sim_dt=8333,
        joint_names=("j0",),
        cameras=(
            CameraMeta(
                role="head",
                prim_path="/World/Camera",
                width=4,
                height=3,
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
        ),
        action=ActionSpec("abs_joint_pos", ("j0",)),
        extra=EpisodeExtra(seed=0, instruction="", outcome="success"),
    )


def _write_episode(path: Path, timestamps: list[int]) -> None:
    with Hdf5EpisodeWriter(path, _meta()) as writer:
        for timestamp in timestamps:
            writer.append(
                Frame(
                    images={"head": _IMAGE},
                    joint_pos=np.zeros(1, dtype=np.float32),
                    joint_vel=np.zeros(1, dtype=np.float32),
                    action=np.zeros(1, dtype=np.float32),
                    timestamp_us=timestamp,
                )
            )


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = io.BytesIO()
        self.stderr = io.BytesIO()

    def wait(self) -> int:
        return 0


class _FailingStdin:
    def __init__(self) -> None:
        self.closed = False

    def write(self, data: bytes) -> None:
        raise OSError("broken pipe simulation")

    def close(self) -> None:
        self.closed = True


class _FaultyProcess:
    def __init__(self) -> None:
        self.stdin = _FailingStdin()
        self.stderr = io.BytesIO(b"ffmpeg exploded")
        self.terminated = False
        self.killed = False
        self.waits = 0

    def poll(self) -> int | None:
        return -15 if self.terminated or self.killed else None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        self.waits += 1
        return -15


class RenderVideoFpsTests(unittest.TestCase):
    def _render(self, path: Path, fps: int | None = None) -> tuple[int, list[str]]:
        process = _FakeProcess()
        with (
            mock.patch.object(
                video.imageio_ffmpeg, "get_ffmpeg_exe", return_value="ffmpeg"
            ),
            mock.patch.object(video.subprocess, "Popen", return_value=process) as popen,
        ):
            count = video.render_video(path, path.with_suffix(".mp4"), fps)
        command = popen.call_args.args[0]
        rates = [
            command[index + 1]
            for index, argument in enumerate(command)
            if argument == "-r"
        ]
        return count, [str(rate) for rate in rates]

    def test_default_fps_uses_recorded_frame_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "throttled.h5"
            # 120 Hz physics recorded every 2nd step: 16666 us per frame.
            _write_episode(path, [0, 16666, 33332])
            count, rates = self._render(path)
        # sim_dt is 8333 us and would imply 120 fps; recorded frames are 60 fps.
        self.assertEqual(count, 3)
        self.assertEqual(rates, ["60", "60"])

    def test_explicit_fps_overrides_recorded_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "override.h5"
            _write_episode(path, [0, 16666, 33332])
            _, rates = self._render(path, fps=24)
        self.assertEqual(rates, ["24", "24"])

    def test_write_failure_terminates_ffmpeg_and_reraises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fail.h5"
            _write_episode(path, [0, 16666])
            process = _FaultyProcess()
            with (
                mock.patch.object(
                    video.imageio_ffmpeg, "get_ffmpeg_exe", return_value="ffmpeg"
                ),
                mock.patch.object(video.subprocess, "Popen", return_value=process),
                self.assertRaisesRegex(OSError, "broken pipe simulation"),
            ):
                video.render_video(path, path.with_suffix(".mp4"), fps=30)
        self.assertTrue(process.terminated)
        self.assertGreaterEqual(process.waits, 1)
        self.assertTrue(process.stdin.closed)


if __name__ == "__main__":
    unittest.main()
