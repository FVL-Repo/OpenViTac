"""State-machine rollout: one Action, one step, one success read per tick."""

from __future__ import annotations

import unittest
from collections.abc import Iterator
from typing import Any, cast

from openvitac_cli.commands.collect import _run_state_rollout
from openvitac_sim.scene import StepLimitExceeded
from openvitac_tasks import Action


class _FakeRolloutScene:
    """Records sink installation and drives it like `Scene.step`."""

    def __init__(self) -> None:
        self.sink = None
        self.every = 1
        self.limit = None
        self.steps = 0

    def set_step_sink(self, sink, *, every: int = 1) -> None:
        self.sink = sink
        self.every = every

    def set_step_limit(self, limit: int | None) -> None:
        self.limit = limit

    def step(self) -> None:
        if self.limit is not None and self.steps >= self.limit:
            raise StepLimitExceeded(f"step limit {self.limit} reached")
        self.steps += 1
        if self.sink is not None and (self.steps - 1) % self.every == 0:
            self.sink("obs")


class _FakeGripper:
    def __init__(self) -> None:
        self.targets: list[float] = []

    def set_target(self, qpos: float) -> None:
        self.targets.append(qpos)


class _FakeArm:
    def __init__(self) -> None:
        self.gripper = _FakeGripper()
        self.ee_targets: list[object] = []

    def set_ee_target(self, pos: object, quat: object) -> None:
        self.ee_targets.append((pos, quat))


class _FakeStateScene(_FakeRolloutScene):
    def __init__(self) -> None:
        super().__init__()
        self.arm = _FakeArm()


class _TickMachine:
    def __init__(self, ticks: int) -> None:
        self.ticks = ticks
        self.emitted = 0

    def init(self) -> int:
        return 0

    def step(self, world: object, state: int):
        if state >= self.ticks:
            return None
        self.emitted += 1
        return (
            state + 1,
            Action(
                ee_pos=(0.4, 0.0, 0.1),
                ee_quat=(0.0, 0.0, 0.0, 1.0),
                gripper_qpos=0.0,
            ),
        )


class _RecordingLogger:
    """Minimal stand-in for the rollout's LoggerAdapter."""

    def __init__(self) -> None:
        self.exceptions: list[str] = []

    def info(self, message: str, *args: object) -> None:
        pass

    def warning(self, message: str, *args: object) -> None:
        pass

    def exception(self, message: str, *args: object) -> None:
        self.exceptions.append(message % args)


def _never_success(world: object) -> Iterator[bool]:
    while True:
        yield False


class RunStateRolloutTests(unittest.TestCase):
    def test_actions_are_applied_then_one_step_per_tick(self) -> None:
        scene = cast(Any, _FakeStateScene())
        machine = _TickMachine(5)
        calls = 0

        def success(world: object) -> Iterator[bool]:
            nonlocal calls
            while True:
                calls += 1
                yield calls == 3

        outcome = _run_state_rollout(
            scene,
            machine,
            success,
            steps=100,
            sink=lambda obs: None,
            record_every=1,
        )

        self.assertEqual(outcome, "success")
        self.assertEqual(scene.steps, 3)
        self.assertEqual(machine.emitted, 3)
        self.assertEqual(len(scene.arm.ee_targets), 3)
        self.assertEqual(scene.arm.gripper.targets, [0.0, 0.0, 0.0])
        self.assertEqual(calls, 3)
        self.assertIsNone(scene.sink)

    def test_success_is_read_after_the_step(self) -> None:
        scene = cast(Any, _FakeStateScene())
        seen: list[int] = []

        def success(world: object) -> Iterator[bool]:
            while True:
                seen.append(scene.steps)
                yield scene.steps >= 1

        outcome = _run_state_rollout(
            scene,
            _TickMachine(5),
            success,
            steps=100,
            sink=lambda obs: None,
            record_every=1,
        )

        self.assertEqual(outcome, "success")
        self.assertEqual(scene.steps, 1)
        self.assertEqual(seen, [1])
        self.assertIsNone(scene.sink)

    def test_success_exception_is_logged_inside_the_rollout(self) -> None:
        logger = cast(Any, _RecordingLogger())

        def success(world: object) -> Iterator[bool]:
            raise RuntimeError("success boom")
            yield False

        scene = cast(Any, _FakeStateScene())
        with self.assertRaisesRegex(RuntimeError, "success boom"):
            _run_state_rollout(
                scene,
                _TickMachine(4),
                success,
                steps=10,
                sink=lambda obs: None,
                record_every=1,
                logger=logger,
            )
        self.assertEqual(
            logger.exceptions,
            [
                "success evaluator failed; re-raising",
                "rollout failed; re-raising",
            ],
        )
        self.assertIsNone(scene.sink)

    def test_success_must_return_an_iterator(self) -> None:
        logger = cast(Any, _RecordingLogger())

        def success(world: object) -> bool:
            return True

        scene = cast(Any, _FakeStateScene())
        with self.assertRaisesRegex(TypeError, "must return an Iterator"):
            _run_state_rollout(
                scene,
                _TickMachine(4),
                cast(Any, success),
                steps=10,
                sink=lambda obs: None,
                record_every=1,
                logger=logger,
            )
        self.assertEqual(
            logger.exceptions,
            [
                "success evaluator failed; re-raising",
                "rollout failed; re-raising",
            ],
        )
        self.assertIsNone(scene.sink)

    def test_machine_init_failure_is_a_policy_error(self) -> None:
        logger = cast(Any, _RecordingLogger())

        class _RaisingInit:
            def init(self) -> int:
                raise RuntimeError("init boom")

            def step(self, world: object, state: int):
                return None

        scene = cast(Any, _FakeStateScene())
        outcome = _run_state_rollout(
            scene,
            _RaisingInit(),
            _never_success,
            steps=10,
            sink=lambda obs: None,
            record_every=1,
            logger=logger,
        )
        self.assertEqual(outcome, "policy_error")
        self.assertEqual(scene.steps, 0)
        self.assertIsNone(scene.sink)
        self.assertEqual(logger.exceptions, ["policy error in init"])

    def test_none_transition_ends_with_fail(self) -> None:
        scene = cast(Any, _FakeStateScene())
        outcome = _run_state_rollout(
            scene,
            _TickMachine(4),
            _never_success,
            steps=100,
            sink=lambda obs: None,
            record_every=1,
        )

        self.assertEqual(outcome, "fail")
        self.assertEqual(scene.steps, 4)
        self.assertIsNone(scene.sink)

    def test_action_none_is_not_termination(self) -> None:
        """A transition always carries an Action; ``(state, None)`` is an error."""

        class _OptionalActionMachine:
            def init(self) -> int:
                return 0

            def step(self, world: object, state: int):
                return state, None

        scene = cast(Any, _FakeStateScene())
        with self.assertRaises(AttributeError):
            _run_state_rollout(
                scene,
                _OptionalActionMachine(),
                _never_success,
                steps=10,
                sink=lambda obs: None,
                record_every=1,
            )
        self.assertEqual(scene.steps, 0)
        self.assertIsNone(scene.sink)

    def test_step_limit_is_timeout_and_cleans_the_sink(self) -> None:
        scene = cast(Any, _FakeStateScene())
        machine = _TickMachine(10)
        calls = 0

        def success(world: object) -> Iterator[bool]:
            nonlocal calls
            while True:
                calls += 1
                yield False

        outcome = _run_state_rollout(
            scene,
            machine,
            success,
            steps=3,
            sink=lambda obs: None,
            record_every=1,
        )

        self.assertEqual(outcome, "timeout")
        self.assertEqual(scene.steps, 3)
        # The machine is polled before each step, so the refused fourth step
        # still gets one action staged; no physics step or success check runs.
        self.assertEqual(machine.emitted, 4)
        self.assertEqual(calls, 3)
        self.assertIsNone(scene.sink)

    def test_machine_step_failure_keeps_frames_and_is_policy_error(self) -> None:
        logger = cast(Any, _RecordingLogger())

        class _RaisingMachine:
            def __init__(self) -> None:
                self.calls = 0

            def init(self) -> int:
                return 0

            def step(self, world: object, state: int):
                self.calls += 1
                if self.calls > 2:
                    raise RuntimeError("machine boom")
                return state + 1, Action(
                    ee_pos=(0.4, 0.0, 0.1),
                    ee_quat=(0.0, 0.0, 0.0, 1.0),
                    gripper_qpos=0.0,
                )

        frames: list[object] = []
        scene = cast(Any, _FakeStateScene())
        outcome = _run_state_rollout(
            scene,
            _RaisingMachine(),
            _never_success,
            steps=10,
            sink=frames.append,
            record_every=1,
            logger=logger,
        )
        self.assertEqual(outcome, "policy_error")
        self.assertEqual(scene.steps, 2)
        self.assertEqual(frames, ["obs", "obs"])
        self.assertIsNone(scene.sink)
        self.assertEqual(logger.exceptions, ["policy error in step"])

    def test_success_iterator_is_created_once_per_episode(self) -> None:
        created = 0

        def success(world: object) -> Iterator[bool]:
            nonlocal created
            created += 1

            def verdicts() -> Iterator[bool]:
                count = 0
                while True:
                    count += 1
                    yield count >= 3

            return verdicts()

        scene = cast(Any, _FakeStateScene())
        outcome = _run_state_rollout(
            scene,
            _TickMachine(5),
            success,
            steps=100,
            sink=lambda obs: None,
            record_every=1,
        )

        self.assertEqual(outcome, "success")
        self.assertEqual(created, 1)
        self.assertEqual(scene.steps, 3)
        self.assertIsNone(scene.sink)

    def test_generator_function_success_keeps_its_local_counter(self) -> None:
        def success(world: object) -> Iterator[bool]:
            count = 0
            while True:
                count += 1
                yield count >= 3

        scene = cast(Any, _FakeStateScene())
        outcome = _run_state_rollout(
            scene,
            _TickMachine(5),
            success,
            steps=100,
            sink=lambda obs: None,
            record_every=1,
        )

        self.assertEqual(outcome, "success")
        self.assertEqual(scene.steps, 3)
        self.assertIsNone(scene.sink)


if __name__ == "__main__":
    unittest.main()
