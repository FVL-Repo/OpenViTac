"""Collect preflight validation and Scene close ownership."""

from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest import mock

from openvitac_cli.commands import collect as collect_module
from openvitac_cli.commands.collect import (
    _DEFAULT_EXPERT_STEPS,
    _DEFAULT_STEPS,
    _open_scene,
    _validate_run_parameters,
    collect,
)


class _FakeScene:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events if events is not None else []

    def close(self) -> None:
        self.events.append("close")


class _FakeGripper:
    capabilities: frozenset[str] = frozenset()


class _FakeArm:
    gripper = _FakeGripper()


class _FakeUnsupportedScene(_FakeScene):
    arm = _FakeArm()


@contextlib.contextmanager
def _fake_launch(**_kwargs):
    yield object()


class ValidateRunParametersTests(unittest.TestCase):
    def test_default_budget(self) -> None:
        self.assertEqual(
            _validate_run_parameters(
                steps=None,
                seed=0,
                no_video=False,
                expert=False,
                record_every=2,
            ),
            _DEFAULT_STEPS,
        )
        self.assertEqual(
            _validate_run_parameters(
                steps=None,
                seed=0,
                no_video=False,
                expert=True,
                record_every=2,
            ),
            _DEFAULT_EXPERT_STEPS,
        )

    def test_zero_steps_is_rejected(self) -> None:
        for steps in (0, -1):
            with self.assertRaisesRegex(ValueError, "--steps"):
                _validate_run_parameters(
                    steps=steps,
                    seed=0,
                    no_video=True,
                    expert=False,
                    record_every=2,
                )

    def test_one_step_requires_no_video(self) -> None:
        with self.assertRaisesRegex(ValueError, "video needs"):
            _validate_run_parameters(
                steps=1,
                seed=0,
                no_video=False,
                expert=False,
                record_every=2,
            )
        self.assertEqual(
            _validate_run_parameters(
                steps=1,
                seed=0,
                no_video=True,
                expert=False,
                record_every=2,
            ),
            1,
        )

    def test_video_needs_two_recorded_frames(self) -> None:
        # Two steps at the default cadence record only step 0.
        with self.assertRaisesRegex(ValueError, "video needs"):
            _validate_run_parameters(
                steps=2,
                seed=0,
                no_video=False,
                expert=False,
                record_every=2,
            )

    def test_negative_seed_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "--seed"):
            _validate_run_parameters(
                steps=10,
                seed=-1,
                no_video=False,
                expert=False,
                record_every=2,
            )

    def test_validation_happens_before_launch(self) -> None:
        with mock.patch.object(
            collect_module.sim, "launch", side_effect=AssertionError("launched")
        ):
            with self.assertRaisesRegex(ValueError, "--seed"):
                collect("insert_usb", seed=-1)
            with self.assertRaisesRegex(ValueError, "--steps"):
                collect("insert_usb", steps=0, no_video=True)
            with self.assertRaisesRegex(ValueError, "video needs"):
                collect("insert_usb", steps=1)


class OpenSceneTests(unittest.TestCase):
    def test_capability_failure_closes_the_scene(self) -> None:
        scene = _FakeUnsupportedScene()
        with (
            mock.patch.object(collect_module.sim, "Scene", return_value=scene),
            self.assertRaisesRegex(RuntimeError, "requires gripper"),
        ):
            _open_scene(
                "insert_usb",
                cast(Any, object()),
                tactile=False,
                optical_backend="taxim",
                taxim_device="cuda:0",
            )
        self.assertEqual(scene.events, ["close"])


class CollectCloseTests(unittest.TestCase):
    def _run(
        self,
        record,
        render,
        *,
        steps: int,
        no_video: bool,
    ) -> tuple[list[str], Exception | None]:
        events: list[str] = []
        scene = _FakeScene(events)

        def fake_record(
            _task,
            _scene,
            _cfg,
            episode,
            _steps,
            _seed,
            _expert,
            **_kwargs,
        ):
            events.append("record")
            Path(episode).write_bytes(b"episode")
            return record()

        def fake_render(*_args):
            events.append("render")
            render()

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(collect_module, "tee_output", contextlib.nullcontext),
            mock.patch.object(collect_module.sim, "launch", _fake_launch),
            mock.patch.object(collect_module, "_seed_run", lambda seed: seed),
            mock.patch.object(
                collect_module, "_open_scene", return_value=(scene, None)
            ),
            mock.patch.object(collect_module, "_record", fake_record),
            mock.patch.object(collect_module, "render_video", fake_render),
        ):
            error: Exception | None = None
            try:
                collect(
                    "insert_usb",
                    out=Path(directory),
                    steps=steps,
                    no_video=no_video,
                )
            except Exception as exc:
                error = exc
        return events, error

    def test_close_runs_after_video_rendering(self) -> None:
        events, error = self._run(
            lambda: (3, 1),
            lambda: None,
            steps=4,
            no_video=False,
        )
        self.assertIsNone(error)
        self.assertEqual(events, ["record", "render", "close"])

    def test_close_runs_when_recording_fails(self) -> None:
        def fail() -> tuple[int, int]:
            raise RuntimeError("writer failed")

        events, error = self._run(fail, lambda: None, steps=4, no_video=True)
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(events, ["record", "close"])

    def test_close_runs_when_video_rendering_fails(self) -> None:
        def fail() -> None:
            raise RuntimeError("ffmpeg failed")

        events, error = self._run(
            lambda: (3, 1),
            fail,
            steps=4,
            no_video=False,
        )
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(events, ["record", "render", "close"])


if __name__ == "__main__":
    unittest.main()
