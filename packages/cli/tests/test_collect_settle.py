"""CLI settle wiring: fixed/convergence metadata and reset -> settle ordering."""

from __future__ import annotations

import contextlib
import io
import unittest
from dataclasses import replace
from typing import Any, cast

from openvitac_cli.commands.collect import (
    _DEFAULT_SETTLE_STEPS,
    _effective_settle_steps,
    _prepare_episode,
    collect,
)
from openvitac_sim.scene import SettleResult
from openvitac_spec import ActorPose
from openvitac_tasks import TASKS, EpisodeRandomization


class _FakeScene:
    def __init__(self, result: SettleResult | None = None) -> None:
        self.events: list[str] = []
        self.reset_poses: dict[str, ActorPose] = {}
        self.settle_args: tuple[int, dict[str, Any]] | None = None
        self.params: dict[str, float | str] = {}
        self._result = result

    def reset(self, *, actor_poses: dict[str, ActorPose]) -> None:
        self.events.append("reset")
        self.reset_poses = dict(actor_poses)

    def settle(self, steps: int, **kwargs: Any) -> SettleResult:
        self.events.append("settle")
        self.settle_args = (steps, kwargs)
        return self._result or SettleResult(requested_steps=steps, executed_steps=steps)

    def start_tactile(self, *, contact_object: str | None = None) -> None:
        self.events.append("start_tactile")
        self.contact_object = contact_object

    def set_expert_params(self, params: dict[str, float | str]) -> None:
        self.events.append("expert_params")
        self.params = dict(params)

    def object_pose(self, prim_name: str) -> tuple[Any, Any]:
        self.events.append("object_pose")
        return (0.41, -0.02, 0.003), (0.0, 0.0, 0.0, 1.0)

    def warmup(self, steps: int) -> None:
        self.events.append(f"warmup:{steps}")


def _randomization() -> EpisodeRandomization:
    pose = ActorPose((0.4, 0.0, 0.002), (0.0, 0.0, 0.0, 1.0))
    return EpisodeRandomization(
        seed=3,
        instruction="test",
        actor_poses={"/World/actors/block": pose},
        contact_object="/World/actors/block",
        metadata={"label": "rough"},
    )


class CollectSettleTests(unittest.TestCase):
    def test_default_is_a_small_fixed_budget(self) -> None:
        self.assertEqual(_DEFAULT_SETTLE_STEPS, 20)

    def test_fixed_mode_order_and_metadata(self) -> None:
        scene = cast(Any, _FakeScene())
        meta = _prepare_episode(
            scene, _randomization(), 7, None
        )
        self.assertEqual(
            scene.events,
            [
                "reset",
                "settle",
                "start_tactile",
                "expert_params",
                "object_pose",
                "warmup:5",
            ],
        )
        self.assertEqual(
            scene.settle_args,
            (7, {"window": None, "pos_eps": 1e-4, "rot_eps": 1e-3}),
        )
        self.assertEqual(scene.reset_poses, _randomization().actor_poses)
        self.assertEqual(scene.params, {"label": "rough"})
        self.assertEqual(scene.contact_object, "/World/actors/block")
        self.assertEqual(meta["settle.mode"], "fixed")
        self.assertEqual(meta["settle.steps"], 7.0)
        self.assertEqual(meta["settle.steps_requested"], 7.0)
        self.assertEqual(meta["settle.converged"], "n/a")
        self.assertEqual(meta["actual.block.pos_x"], 0.41)
        self.assertEqual(meta["actual.block.quat_w"], 1.0)

    def test_object_pose_error_is_not_silently_skipped(self) -> None:
        class _BrokenPoseScene(_FakeScene):
            def object_pose(self, prim_name: str) -> tuple[Any, Any]:
                raise KeyError(prim_name)

        with self.assertRaises(KeyError):
            _prepare_episode(
                cast(Any, _BrokenPoseScene()), _randomization(), 7, None
            )

    def test_expert_params_are_merged_with_metadata(self) -> None:
        scene = cast(Any, _FakeScene())
        randomization = replace(
            _randomization(), expert_params={"approach_z": 0.12}
        )
        meta = _prepare_episode(
            scene, randomization, 7, None
        )
        self.assertEqual(
            scene.params, {"label": "rough", "approach_z": 0.12}
        )
        self.assertEqual(meta["expert.approach_z"], 0.12)

    def test_duplicate_expert_params_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            replace(_randomization(), expert_params={"label": 1.0})

    def test_convergence_metadata_records_executed_steps(self) -> None:
        result = SettleResult(20, 9, True, 1e-5, 2e-4)
        scene = cast(Any, _FakeScene(result))
        meta = _prepare_episode(
            scene, _randomization(), 20, 5
        )
        self.assertEqual(
            scene.settle_args,
            (20, {"window": 5, "pos_eps": 1e-4, "rot_eps": 1e-3}),
        )
        self.assertEqual(meta["settle.mode"], "converged")
        self.assertEqual(meta["settle.converged"], "true")
        self.assertEqual(meta["settle.steps"], 9.0)
        self.assertEqual(meta["settle.steps_requested"], 20.0)
        self.assertEqual(meta["settle.max_pos_delta_m"], 1e-5)
        self.assertEqual(meta["settle.max_rot_delta_rad"], 2e-4)

    def test_non_convergence_warns_and_is_recorded(self) -> None:
        result = SettleResult(20, 20, False, 3e-4, 2e-3)
        scene = cast(Any, _FakeScene(result))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            meta = _prepare_episode(
                scene, _randomization(), 20, 5
            )
        self.assertIn("did not converge", output.getvalue())
        self.assertEqual(meta["settle.converged"], "false")
        self.assertEqual(meta["settle.steps"], 20.0)

    def test_task_default_and_cli_override(self) -> None:
        task = TASKS["roughness_regrasp"]
        self.assertEqual(task.settle_steps, _DEFAULT_SETTLE_STEPS)
        self.assertEqual(_effective_settle_steps(task, None), task.settle_steps)
        self.assertEqual(_effective_settle_steps(task, 0), 0)
        self.assertEqual(
            _effective_settle_steps(replace(task, settle_steps=5), None), 5
        )
        with self.assertRaises(ValueError):
            _effective_settle_steps(task, -1)

    def test_negative_settle_is_rejected_before_launch(self) -> None:
        with self.assertRaises(ValueError):
            collect("insert_usb", settle_steps=-1)


if __name__ == "__main__":
    unittest.main()
