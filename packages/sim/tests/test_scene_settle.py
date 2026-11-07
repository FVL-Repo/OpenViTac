"""Scene.settle: fixed vs pose-delta convergence, clean clock, monotonic seq."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch

from openvitac_sim.scene import Scene, SettleResult


class _SettleScene:
    def __init__(self, snapshots: list[dict[str, np.ndarray]] | None = None) -> None:
        self.calls: list[object] = []
        self._step_sink: object | None = None
        self._step_limit: int | None = None
        self._reset_done = True
        self._steps_taken = 0
        self._last_step_index = 0
        self._physics_steps_total = 0
        self._snapshots = iter(snapshots or [])

    def step(self) -> None:
        self.calls.append(self._step_sink)
        self._steps_taken += 1
        self._last_step_index = self._steps_taken
        self._physics_steps_total += 1

    def _actor_pose_snapshot(self) -> dict[str, np.ndarray]:
        return next(self._snapshots)

    def settle(self, *args: Any, **kwargs: Any) -> SettleResult:
        return cast(SettleResult, cast(Any, Scene).settle(self, *args, **kwargs))


def _pose(x: float = 0.0) -> dict[str, np.ndarray]:
    return {"/World/block": np.array([x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])}


class SceneSettleTests(unittest.TestCase):
    def test_fixed_mode_runs_real_steps_and_resets_episode_clock(self) -> None:
        scene = _SettleScene()
        result = scene.settle(3)
        self.assertEqual(result.requested_steps, 3)
        self.assertEqual(result.executed_steps, 3)
        self.assertIsNone(result.converged)
        self.assertIsNone(result.max_pos_delta_m)
        self.assertEqual(scene.calls, [None, None, None])
        self.assertEqual(scene._steps_taken, 0)
        self.assertEqual(scene._last_step_index, 0)
        self.assertEqual(scene._physics_steps_total, 3)

    def test_zero_steps_is_the_legacy_no_settle_path(self) -> None:
        scene = _SettleScene()
        scene._steps_taken = cast(Any, 7)
        scene._last_step_index = cast(Any, 6)
        result = scene.settle(0, window=5)
        self.assertEqual(result.requested_steps, 0)
        self.assertEqual(result.executed_steps, 0)
        self.assertIsNone(result.converged)
        self.assertEqual(scene.calls, [])
        self.assertEqual(scene._steps_taken, 7)
        self.assertEqual(scene._last_step_index, 6)

    def test_convergence_stops_after_a_stable_window(self) -> None:
        scene = _SettleScene([_pose() for _ in range(10)])
        result = scene.settle(10, window=5, pos_eps=1e-4, rot_eps=1e-3)
        self.assertTrue(result.converged)
        self.assertEqual(result.executed_steps, 6)
        self.assertEqual(len(scene.calls), 6)
        self.assertEqual(result.max_pos_delta_m, 0.0)
        self.assertEqual(result.max_rot_delta_rad, 0.0)
        self.assertEqual(scene._physics_steps_total, 6)
        self.assertEqual(scene._steps_taken, 0)

    def test_non_convergence_is_explicit_not_silent(self) -> None:
        scene = _SettleScene([_pose(0.001 * index) for index in range(10)])
        result = scene.settle(10, window=5, pos_eps=1e-4, rot_eps=1e-3)
        self.assertFalse(result.converged)
        self.assertEqual(result.executed_steps, 10)
        self.assertGreater(result.max_pos_delta_m or 0.0, 1e-4)

    def test_settle_restores_an_existing_sink(self) -> None:
        scene = _SettleScene()
        sink = object()
        scene._step_sink = cast(Any, sink)
        scene.settle(2)
        self.assertEqual(scene.calls, [None, None])
        self.assertIs(scene._step_sink, sink)

    def test_negative_steps_and_bad_window_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _SettleScene().settle(-1)
        with self.assertRaises(ValueError):
            _SettleScene().settle(5, window=0)

    def test_settle_before_step_limit_is_required(self) -> None:
        scene = _SettleScene()
        scene._step_limit = cast(Any, 10)
        with self.assertRaises(RuntimeError):
            scene.settle(1)
        self.assertEqual(scene.calls, [])

    def test_settle_requires_reset(self) -> None:
        scene = _SettleScene()
        scene._reset_done = cast(Any, False)
        with self.assertRaises(RuntimeError):
            scene.settle(1)
        self.assertEqual(scene.calls, [])

    def test_finger_sample_uses_monotonic_physics_counter(self) -> None:
        scene = Scene.__new__(Scene)
        scene._steps_taken = cast(Any, 0)
        scene._physics_steps_total = cast(Any, 20)
        scene._body_index_cache = cast(Any, {})
        pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        bodies = torch.tensor([[pose, pose]], dtype=torch.float32)
        scene._arm = cast(
            Any,
            SimpleNamespace(
                embodiment=SimpleNamespace(
                    gripper=SimpleNamespace(contact_bodies=("left", "right"))
                ),
                robot=SimpleNamespace(
                    data=SimpleNamespace(body_pose_w=SimpleNamespace(torch=bodies))
                ),
            ),
        )
        scene._body_ids = cast(Any, lambda names: [0, 1])
        sample = scene._make_finger_sample()
        # Sample follows the monotonic counter, not the reset episode clock.
        self.assertEqual((sample.seq, sample.physx_step), (20, 20))


class _FakeBridge:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.reset_kwargs: dict[str, object] = {}

    def start(self) -> None:
        self.events.append("start")

    def reset(self, **kwargs: object) -> dict[str, object]:
        self.reset_kwargs = dict(kwargs)
        self.events.append("reset")
        return {}


class TactileStartTests(unittest.TestCase):
    def _scene(
        self, events: list[str], *, reset_done: bool = True, settled: bool = True
    ) -> Scene:
        scene = Scene.__new__(Scene)
        scene._tactile_bridge = cast(Any, _FakeBridge(events))
        scene._tactile_started = cast(Any, False)
        scene._reset_done = cast(Any, reset_done)
        scene._settled = cast(Any, settled)
        scene._episode_seed = cast(Any, 7)
        scene._uipc_proxies = cast(Any, {})
        scene._physics_steps_total = cast(Any, 20)
        scene._tactile_attachment_frames = cast(Any, lambda: {})
        scene._tactile_body_pad_frames = cast(Any, lambda frames: {})
        scene._tactile_object_poses = cast(Any, lambda: {})
        scene._bind_sensors = cast(Any, lambda: events.append("visuals"))
        return scene

    def test_start_tactile_is_explicit_and_idempotent(self) -> None:
        events: list[str] = []
        scene = self._scene(events)
        scene.start_tactile()
        self.assertEqual(events, ["start", "reset", "visuals"])
        self.assertTrue(scene._tactile_started)
        self.assertEqual(scene._physics_steps_total, 0)
        self.assertEqual(
            cast(Any, scene._tactile_bridge).reset_kwargs["seed"],
            7,
        )
        scene.start_tactile()
        self.assertEqual(events, ["start", "reset", "visuals"])

    def test_start_tactile_requires_reset(self) -> None:
        with self.assertRaises(RuntimeError):
            self._scene([], reset_done=False).start_tactile()

    def test_start_tactile_requires_settle(self) -> None:
        with self.assertRaises(RuntimeError):
            self._scene([], settled=False).start_tactile()


class SceneStepGuardTests(unittest.TestCase):
    def test_step_requires_reset(self) -> None:
        scene = Scene.__new__(Scene)
        scene._reset_done = cast(Any, False)
        with self.assertRaises(RuntimeError):
            scene.step()


if __name__ == "__main__":
    unittest.main()
