"""insert_block selection/success/expert tests without Isaac."""

from __future__ import annotations

import itertools
import logging
import unittest
from collections.abc import Iterator
from typing import Any

from openvitac_spec import GripperCapability
from openvitac_tasks import Action
from openvitac_tasks.tasks.insert_block import (
    _BOX_SIZE,
    _HAND_TO_GRASP,
    _INSERT_DEPTH,
    _RELEASE_RETRACT,
    _SETTLE_STEPS,
    InsertBlock,
)

_BOX = (0.48, -0.12, 0.002)
_ACTOR = {
    "cube": "block_blue_cube",
    "half_cylinder": "block_blue_half_cylinder",
    "hexagon": "block_red_hexagonal_prism",
}
_BLOCKS = {
    "block_blue_cube": ((0.45, 0.15, 0.002), (0.0, 0.0, 0.0, 1.0)),
    "block_blue_half_cylinder": ((0.37, 0.08, 0.002), (0.0, 0.0, 0.0, 1.0)),
    "block_red_hexagonal_prism": ((0.36, 0.24, 0.002), (0.0, 0.0, 0.0, 1.0)),
}
_HOLES = {
    "cube": (-0.035, 0.03),
    "half_cylinder": (0.034, -0.018117),
    "hexagon": (-0.026, -0.036),
}
_DESCRIPTIONS = {
    "cube": "blue cube",
    "half_cylinder": "blue half cylinder",
    "hexagon": "red hexagonal prism",
}


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
            world.holding = world.actor
        elif qpos == self.open_qpos and self._qpos != self.open_qpos:
            world.events.append(("open", 1))
            # The real block falls through the top plate to the box floor.
            if world.holding is not None:
                x, y, _ = world.poses[world.holding][0]
                world.poses[world.holding] = (
                    (x, y, _BOX[2] + 0.006),
                    (0.0, 0.0, 0.0, 1.0),
                )
            world.holding = None
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
        if world.holding is not None:
            bx, by, bz = world.poses[world.holding][0]
            world.poses[world.holding] = (
                (
                    bx + pos[0] - world.ee[0],
                    by + pos[1] - world.ee[1],
                    bz + pos[2] - world.ee[2],
                ),
                world.poses[world.holding][1],
            )
        world.ee = pos
        world.quat = quat
        world.events.append(("move", pos))


class _World:

    @property
    def logger(self) -> logging.LoggerAdapter[logging.Logger]:
        return logging.LoggerAdapter(logging.getLogger(__name__))

    def __init__(self, target: str) -> None:
        self.target = target
        self.actor = _ACTOR.get(target, "block_blue_cube")
        self.poses = {name: pose for name, pose in _BLOCKS.items()}
        self.poses["box_with_holes_yellow"] = (_BOX, (0.0, 0.0, 0.0, 1.0))
        self.ee = (0.30, 0.0, 0.30)
        self.quat = (0.0, 0.0, 0.0, 1.0)
        self.holding: str | None = None
        self.steps = 0
        self.events: list[tuple[object, ...]] = []
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
        if name == "target_block":
            return self.target
        raise KeyError(name)

    def object_pose(
        self, prim_name: str
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        try:
            return self.poses[prim_name]
        except KeyError:
            raise KeyError(prim_name) from None


def _run(world: _World) -> list[str]:
    """Drive the registered state machine and record its phase sequence."""
    machine = InsertBlock.expert
    state = machine.init()
    phases: list[str] = []
    while True:
        result = machine.step(world, state)
        if result is None:
            return phases
        state, action = result
        phases.append(state.phase.name)
        world.apply(action)


def _run_until_success(world: _World) -> tuple[str, bool]:
    """Drive the machine like the collector; return the first-true phase."""
    machine = InsertBlock.expert
    state = machine.init()
    verdicts = InsertBlock.success(world)
    while True:
        result = machine.step(world, state)
        if result is None:
            raise AssertionError("policy ended before the success gate")
        state, action = result
        world.apply(action)
        if next(verdicts):
            return str(state.phase.name), world.holding is None


def _fires(verdicts: Iterator[bool], ticks: int = _SETTLE_STEPS) -> bool:
    """Advance at most ``ticks`` verdicts; whether the settle gate fires."""
    return any(itertools.islice(verdicts, ticks))


class InsertBlockTests(unittest.TestCase):
    def test_randomizer_samples_all_targets_and_keeps_blocks_on_table(self) -> None:
        seen: set[str] = set()
        base = {
            actor.prim_path.rsplit("/", 1)[-1]: actor
            for actor in InsertBlock.actors
        }
        for seed in range(10):
            randomization = InsertBlock.setup(seed)
            target = randomization.metadata["target_block"]
            assert isinstance(target, str)
            seen.add(target)
            self.assertEqual(
                randomization.instruction,
                f"Insert the {_DESCRIPTIONS[target]} into the matching hole in "
                "the yellow box.",
            )
            self.assertNotIn(
                "/World/actors/box_with_holes_yellow", randomization.actor_poses
            )
            for block in _BLOCKS:
                pose = randomization.actor_poses[f"/World/actors/{block}"]
                self.assertAlmostEqual(pose.position[2], _BLOCKS[block][0][2])
                self.assertLessEqual(
                    abs(pose.position[0] - base[block].pos[0]), 0.02
                )
                self.assertLessEqual(
                    abs(pose.position[1] - base[block].pos[1]), 0.02
                )
        self.assertEqual(seen, set(_ACTOR))

    def test_success_requires_cavity_in_box_frame(self) -> None:
        retract_z = (
            _BOX[2] + _BOX_SIZE + _HAND_TO_GRASP + _RELEASE_RETRACT - _INSERT_DEPTH
        )
        for target, actor in _ACTOR.items():
            world = _World(target)
            world.poses[actor] = (
                (_BOX[0], _BOX[1], _BOX[2] + 0.006),
                (0.0, 0.0, 0.0, 1.0),
            )
            world.ee = (_BOX[0], _BOX[1], retract_z)
            world.arm.gripper.set_target(world.arm.gripper.open_qpos)
            self.assertTrue(_fires(InsertBlock.success(world)))

            # Missed the box and rests on the table beside it.
            world.poses[actor] = (
                (_BOX[0] + 0.10, _BOX[1], _BOX[2]),
                (0.0, 0.0, 0.0, 1.0),
            )
            self.assertFalse(_fires(InsertBlock.success(world)))

            # Still sitting on the box top.
            world.poses[actor] = (
                (_BOX[0], _BOX[1], _BOX[2] + 0.15),
                (0.0, 0.0, 0.0, 1.0),
            )
            self.assertFalse(_fires(InsertBlock.success(world)))

    def test_success_uses_rotated_box_frame(self) -> None:
        world = _World("cube")
        # 90 deg about z: box-local (x, y) maps to world (-y, x).
        world.poses["box_with_holes_yellow"] = (
            _BOX,
            (0.0, 0.0, 0.7071067811865476, 0.7071067811865476),
        )
        local = (0.02, 0.01, 0.006)
        world.poses["block_blue_cube"] = (
            (_BOX[0] - local[1], _BOX[1] + local[0], _BOX[2] + local[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
        world.ee = (
            _BOX[0] - local[1],
            _BOX[1] + local[0],
            _BOX[2] + _BOX_SIZE + _HAND_TO_GRASP + _RELEASE_RETRACT - _INSERT_DEPTH,
        )
        world.arm.gripper.set_target(world.arm.gripper.open_qpos)
        self.assertTrue(_fires(InsertBlock.success(world)))

    def test_expert_inserts_each_target_into_its_hole(self) -> None:
        for target, actor in _ACTOR.items():
            world = _World(target)
            phases = _run(world)

            close = next(
                i for i, event in enumerate(world.events) if event[0] == "close"
            )
            open_ = next(
                i for i, event in enumerate(world.events) if event[0] == "open"
            )
            self.assertLess(close, open_)
            self.assertLess(phases.index("CLOSE"), phases.index("OPEN"))
            self.assertEqual(phases[-1], "SETTLE")
            bx, by, bz = world.poses[actor][0]
            self.assertAlmostEqual(bx, _BOX[0] + _HOLES[target][0], places=6)
            self.assertAlmostEqual(by, _BOX[1] + _HOLES[target][1], places=6)
            self.assertAlmostEqual(bz, _BOX[2] + 0.006, places=6)
            self.assertTrue(_fires(InsertBlock.success(world)))

    def test_unknown_target_fails_loud(self) -> None:
        world = _World("banana")
        with self.assertRaises(ValueError):
            _run(world)
        with self.assertRaises(ValueError):
            next(InsertBlock.success(world))

    def test_first_true_is_after_release_retract_and_settle(self) -> None:
        world = _World("cube")
        phase, released = _run_until_success(world)
        self.assertEqual(phase, "SETTLE")
        self.assertTrue(released)
        box_top = world.poses["box_with_holes_yellow"][0][2] + _BOX_SIZE
        self.assertGreaterEqual(
            world.ee[2],
            box_top + _HAND_TO_GRASP + _RELEASE_RETRACT - _INSERT_DEPTH,
        )

    def test_expert_respects_step_budget(self) -> None:
        world = _World("half_cylinder")
        _run(world)
        self.assertLess(world.steps, 2000)


if __name__ == "__main__":
    unittest.main()
