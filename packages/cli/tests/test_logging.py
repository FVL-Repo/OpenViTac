"""Run-scoped logger lifecycle and task-boundary logging tests."""

from __future__ import annotations

import logging
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

from openvitac_cli.commands.collect import _run_state_rollout
from openvitac_cli.run_log import run_logger
from openvitac_cli.tee import tee_output
from openvitac_sim.scene import Scene, StepLimitExceeded
from openvitac_tasks import Action


class _FakeGripper:
    def set_target(self, qpos: float) -> None:
        pass


class _FakeArm:
    def __init__(self) -> None:
        self.gripper = _FakeGripper()

    def set_ee_target(self, pos: object, quat: object) -> None:
        pass


class _FakeScene:
    def __init__(self) -> None:
        self.sink = None
        self.limit = None
        self.steps = 0
        self.arm = _FakeArm()

    def set_step_sink(self, sink, *, every: int = 1) -> None:
        self.sink = sink

    def set_step_limit(self, limit: int | None) -> None:
        self.limit = limit

    def step(self) -> None:
        if self.limit is not None and self.steps >= self.limit:
            raise StepLimitExceeded(f"step limit {self.limit} reached")
        self.steps += 1


class _RaisingMachine:
    def init(self) -> int:
        return 0

    def step(self, world: object, state: int):
        raise RuntimeError("expert boom")


class _ForeverMachine:
    def init(self) -> int:
        return 0

    def step(self, world: object, state: int):
        return state + 1, Action(
            ee_pos=(0.4, 0.0, 0.1),
            ee_quat=(0.0, 0.0, 0.0, 1.0),
            gripper_qpos=0.0,
        )


class _OneActionMachine:
    def init(self) -> int:
        return 0

    def step(self, world: object, state: int):
        if state >= 1:
            return None
        return state + 1, Action(
            ee_pos=(0.4, 0.0, 0.1),
            ee_quat=(0.0, 0.0, 0.0, 1.0),
            gripper_qpos=0.0,
        )


def _never(world: object):
    while True:
        yield False


def _raising_success(world: object):
    raise RuntimeError("success boom")
    yield False


class LoggerTeeTests(unittest.TestCase):
    def test_scene_exposes_setter_and_property(self) -> None:
        self.assertTrue(callable(Scene.set_logger))
        self.assertIsInstance(Scene.__dict__["logger"], property)

    def test_handler_writes_through_fd2_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collect.log"
            with (
                tee_output(path),
                run_logger(task="fd2", seed=0, run_id="fd") as log,
            ):
                handler = cast(Any, log.task.logger.handlers[-1])
                fd = handler.stream.fileno()
                self.assertNotEqual(fd, 2)
                duplicate = os.fstat(fd)
                redirected = os.fstat(2)
                self.assertEqual(
                    (duplicate.st_dev, duplicate.st_ino),
                    (redirected.st_dev, redirected.st_ino),
                )
                os.write(fd, b"direct fd marker\n")
            # The handler duplicated fd 2; closing it must not close fd 2.
            with self.assertRaises(OSError):
                os.fstat(fd)
            self.assertIn("direct fd marker", path.read_text())
        os.fstat(2)

    def test_task_and_run_loggers_reach_collect_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collect.log"
            with (
                tee_output(path),
                run_logger(task="demo", seed=7, run_id="run_1") as log,
            ):
                log.task.info("hello task")
                log.run.warning("run warning")
            text = path.read_text()
        self.assertIn("hello task", text)
        self.assertIn("run warning", text)
        self.assertIn("openvitac.task.demo", text)
        self.assertIn("task=demo seed=7 run=run_1", text)

    def test_handlers_are_removed_between_runs(self) -> None:
        task = "handler_test"
        logger = logging.getLogger(f"openvitac.task.{task}")
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / "first.log", Path(directory) / "second.log"]
            for index, path in enumerate(paths):
                with (
                    tee_output(path),
                    run_logger(task=task, seed=index, run_id=str(index)) as log,
                ):
                    log.task.info("once")
            for path in paths:
                self.assertEqual(path.read_text().count("once"), 1)
        self.assertEqual(logger.handlers, [])

    def test_debug_is_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            info_path = Path(directory) / "info.log"
            debug_path = Path(directory) / "debug.log"
            with (
                tee_output(info_path),
                run_logger(task="level_test", seed=0, run_id="info") as log,
            ):
                log.task.debug("hidden")
            with (
                tee_output(debug_path),
                run_logger(
                    task="level_test", seed=0, run_id="debug", level=logging.DEBUG
                ) as log,
            ):
                log.task.debug("visible")
            self.assertNotIn("hidden", info_path.read_text())
            self.assertIn("visible", debug_path.read_text())


class BoundaryLoggingTests(unittest.TestCase):
    def test_expert_exception_is_logged_as_policy_error(self) -> None:
        scene = cast(Any, _FakeScene())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collect.log"
            with (
                tee_output(path),
                run_logger(task="failing", seed=3, run_id="run") as log,
            ):
                outcome = _run_state_rollout(
                    scene,
                    _RaisingMachine(),
                    _never,
                    steps=10,
                    sink=lambda obs: None,
                    record_every=1,
                    logger=log.task,
                )
            text = path.read_text()
        self.assertEqual(outcome, "policy_error")
        self.assertIn("policy error in step", text)
        self.assertIn("RuntimeError: expert boom", text)
        self.assertIn("outcome=policy_error reason=expert-exception", text)

    def test_success_exception_is_logged_and_reraised(self) -> None:
        scene = cast(Any, _FakeScene())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collect.log"
            with (
                tee_output(path),
                run_logger(task="failing", seed=3, run_id="run") as log,
                self.assertRaisesRegex(RuntimeError, "success boom"),
            ):
                _run_state_rollout(
                    scene,
                    _OneActionMachine(),
                    _raising_success,
                    steps=10,
                    sink=lambda obs: None,
                    record_every=1,
                    logger=log.task,
                )
            text = path.read_text()
        self.assertIn("success evaluator failed", text)
        self.assertIn("RuntimeError: success boom", text)

    def test_step_limit_logs_warning_and_timeout(self) -> None:
        scene = cast(Any, _FakeScene())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collect.log"
            with (
                tee_output(path),
                run_logger(task="failing", seed=3, run_id="run") as log,
            ):
                outcome = _run_state_rollout(
                    scene,
                    _ForeverMachine(),
                    _never,
                    steps=3,
                    sink=lambda obs: None,
                    record_every=1,
                    logger=log.task,
                )
            text = path.read_text()
        self.assertEqual(outcome, "timeout")
        self.assertIn("expert hit step limit", text)
        self.assertIn("outcome=timeout reason=step-limit", text)

    def test_fail_logs_reason(self) -> None:
        scene = cast(Any, _FakeScene())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collect.log"
            with (
                tee_output(path),
                run_logger(task="failing", seed=3, run_id="run") as log,
            ):
                outcome = _run_state_rollout(
                    scene,
                    _OneActionMachine(),
                    _never,
                    steps=10,
                    sink=lambda obs: None,
                    record_every=1,
                    logger=log.task,
                )
            text = path.read_text()
        self.assertEqual(outcome, "fail")
        self.assertIn("outcome=fail reason=per-step-success-never-true", text)


if __name__ == "__main__":
    unittest.main()
