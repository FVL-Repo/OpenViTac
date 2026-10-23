"""Weight-classify randomization/expert/success tests without Isaac."""

from __future__ import annotations

import logging
import unittest
from typing import Any

from openvitac_spec import GripperCapability
from openvitac_tasks import Action
from openvitac_tasks.tasks.weight_classify import (
    _APPROACH_CLEARANCE,
    _BLOCK_STANDBY_BY_LABEL,
    _BLOCK_START_POS,
    _DENSITY_BY_LABEL,
    _GRASP_HEIGHT,
    _HAND_TO_GRASP,
    _SUCCESS_Z_TOLERANCE,
    _TARGET_BLOCK_Z,
    WeightClassify,
)

_PLATES = {
    "blue_plate": (0.45, 0.08, 0.002),
    "yellow_plate": (0.45, -0.08, 0.002),
}
_BLOCKS = {"light": "light_block", "heavy": "heavy_block"}
_TARGETS = {"light": "blue_plate", "heavy": "yellow_plate"}


class _Gripper:
    open_qpos = 0.04
    close_qpos = 0.0

    @property
    def capabilities(self) -> frozenset[GripperCapability]:
        return frozenset({"binary_close"})

    def __init__(self, world: _World) -> None:
        self._world = world
        self._qpos = 0.02

    def set_target(self, qpos: float) -> None:
        world = self._world
        if qpos == self.close_qpos and self._qpos != self.close_qpos:
            world.events.append(("close", 1))
            world.holding = True
        elif qpos == self.open_qpos and self._qpos != self.open_qpos:
            world.events.append(("open", 1))
            world.holding = False
        self._qpos = qpos

    def joint_pos(self) -> float:
        return self._qpos


class _Arm:

    def joint_pos(self) -> Any:
        return None

    def gripper_contact(self, *, force_threshold: float = 1.0) -> tuple[bool, bool]:
        return (False, False)

    def __init__(self, world: _World) -> None:
        self._world = world

    @property
    def gripper(self) -> _Gripper:
        return self._world._gripper

    def ee_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        return self._world.ee, self._world.quat

    def set_ee_target(
        self,
        pos: tuple[float, float, float],
        quat: tuple[float, float, float, float],
    ) -> None:
        world = self._world
        world.ee = pos
        world.quat = quat
        world.events.append(("move", pos))
        if world.holding:
            world.blocks[world.label] = (
                pos[0],
                pos[1],
                pos[2] - _HAND_TO_GRASP - _GRASP_HEIGHT,
            )


class _World:

    @property
    def logger(self) -> logging.LoggerAdapter[logging.Logger]:
        return logging.LoggerAdapter(logging.getLogger(__name__))

    def __init__(self, label: str) -> None:
        self.label = label
        self.ee = (0.38, 0.0, 0.30)
        self.quat = (0.0, 0.0, 0.0, 1.0)
        self.blocks = {
            name: _BLOCK_STANDBY_BY_LABEL[name] for name in _BLOCKS
        }
        self.blocks[label] = _BLOCK_START_POS
        self.holding = False
        self.steps = 0
        self.events: list[tuple[str, Any]] = []
        self._gripper = _Gripper(self)
        self._arm = _Arm(self)

    def apply(self, action: Action) -> None:
        self.arm.set_ee_target(action.ee_pos, action.ee_quat)
        self._gripper.set_target(action.gripper_qpos)
        self.steps += 1

    @property
    def arm(self) -> _Arm:
        return self._arm

    def expert_param(self, name: str) -> str:
        if name == "weight_label":
            return self.label
        raise KeyError(name)

    def object_pose(
        self, prim_name: str
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        for label, block in _BLOCKS.items():
            if prim_name == block:
                return self.blocks[label], (0.0, 0.0, 0.0, 1.0)
        if prim_name in _PLATES:
            return _PLATES[prim_name], (0.0, 0.0, 0.0, 1.0)
        raise KeyError(prim_name)


def _run(world: _World) -> list[str]:
    """Drive the registered state machine and record its phase sequence."""
    machine = WeightClassify.expert
    state = machine.init()
    phases: list[str] = []
    while True:
        result = machine.step(world, state)
        if result is None:
            return phases
        state, action = result
        phases.append(state.phase.name)
        world.apply(action)


def _run_until_success(world: _World) -> tuple[str, float]:
    """Drive the machine like the collector; return the first-true phase."""
    machine = WeightClassify.expert
    state = machine.init()
    verdicts = WeightClassify.success(world)
    while True:
        result = machine.step(world, state)
        if result is None:
            raise AssertionError("policy ended before the success gate")
        state, action = result
        world.apply(action)
        if next(verdicts):
            return str(state.phase.name), world.arm.gripper.joint_pos()


class WeightClassifyTests(unittest.TestCase):
    def test_randomizer_samples_both_labels_and_legacy_mapping(self) -> None:
        seen: set[str] = set()
        for seed in range(8):
            randomization = WeightClassify.setup(seed)
            label = randomization.metadata["weight_label"]
            target = randomization.metadata["target_plate"]
            density = randomization.metadata["block_density"]
            assert isinstance(label, str)
            assert isinstance(target, str)
            seen.add(label)
            self.assertIn(label, _BLOCKS)
            self.assertEqual(target, _TARGETS[label])
            self.assertEqual(density, _DENSITY_BY_LABEL[label])
            plate = target.removesuffix("_plate")
            self.assertEqual(
                randomization.instruction,
                "Classify the block weight by touch: "
                f"place the {label} block on the {plate} plate.",
            )
            selected = _BLOCKS[label]
            other = "heavy" if label == "light" else "light"
            self.assertEqual(
                randomization.actor_poses[f"/World/actors/{selected}"].position,
                _BLOCK_START_POS,
            )
            self.assertEqual(
                randomization.actor_poses[f"/World/actors/{_BLOCKS[other]}"].position,
                _BLOCK_STANDBY_BY_LABEL[other],
            )
        self.assertEqual(seen, {"light", "heavy"})

    def test_task_uses_weight_densities_and_one_usd(self) -> None:
        blocks = {
            actor.prim_path: actor
            for actor in WeightClassify.actors
            if actor.prim_path.endswith("_block")
        }
        self.assertEqual(set(blocks), {
            "/World/actors/light_block",
            "/World/actors/heavy_block",
        })
        self.assertEqual(blocks["/World/actors/light_block"].density, 1.0e2)
        self.assertEqual(blocks["/World/actors/heavy_block"].density, 3.0e4)
        self.assertEqual(
            blocks["/World/actors/light_block"].usd,
            blocks["/World/actors/heavy_block"].usd,
        )

    def test_success_uses_selected_label_and_plate(self) -> None:
        for label in _BLOCKS:
            target = _PLATES[_TARGETS[label]]
            world = _World(label)
            world.blocks[label] = (target[0], target[1], target[2] + 0.01)
            world.arm.gripper.set_target(world.arm.gripper.open_qpos)
            world.ee = (
                target[0],
                target[1],
                target[2]
                + _TARGET_BLOCK_Z
                + _GRASP_HEIGHT
                + _HAND_TO_GRASP
                + _APPROACH_CLEARANCE,
            )
            self.assertTrue(next(WeightClassify.success(world)))

            world.blocks[label] = (target[0] + 0.05, target[1], target[2] + 0.01)
            self.assertFalse(next(WeightClassify.success(world)))

    def test_first_true_is_after_open_and_retract(self) -> None:
        target = _PLATES[_TARGETS["light"]]
        world = _World("light")
        phase, gripper_qpos = _run_until_success(world)
        self.assertEqual(phase, "RETRACT")
        self.assertGreaterEqual(
            gripper_qpos, world.arm.gripper.open_qpos - _SUCCESS_Z_TOLERANCE
        )
        retract_z = (
            target[2]
            + _TARGET_BLOCK_Z
            + _GRASP_HEIGHT
            + _HAND_TO_GRASP
            + _APPROACH_CLEARANCE
        )
        self.assertGreaterEqual(world.ee[2], retract_z - _SUCCESS_Z_TOLERANCE)

    def test_expert_moves_selected_block_to_its_plate(self) -> None:
        for label in _BLOCKS:
            target = _PLATES[_TARGETS[label]]
            world = _World(label)
            phases = _run(world)

            close = next(
                i for i, event in enumerate(world.events) if event[0] == "close"
            )
            open_ = next(
                i for i, event in enumerate(world.events) if event[0] == "open"
            )
            target_move = next(
                i
                for i, event in enumerate(world.events)
                if event[0] == "move" and abs(event[1][1] - target[1]) < 0.001
            )
            self.assertLess(close, target_move)
            self.assertLess(target_move, open_)
            self.assertLess(phases.index("CLOSE"), phases.index("OPEN"))
            self.assertEqual(phases[-1], "RETRACT")
            placed = world.blocks[label]
            self.assertAlmostEqual(placed[0], target[0], places=6)
            self.assertAlmostEqual(placed[1], target[1], places=6)
            self.assertAlmostEqual(
                placed[2], target[2] + _TARGET_BLOCK_Z, places=6
            )
            other = "heavy" if label == "light" else "light"
            self.assertEqual(world.blocks[other], _BLOCK_STANDBY_BY_LABEL[other])
            self.assertTrue(next(WeightClassify.success(world)))


if __name__ == "__main__":
    unittest.main()
