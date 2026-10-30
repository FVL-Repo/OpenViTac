"""Roughness-regrasp expert/success tests without Isaac."""

from __future__ import annotations

import logging
import unittest
from typing import Any

from openvitac_spec import GripperCapability
from openvitac_tasks import Action
from openvitac_tasks.motion import HAND_TO_GEL_FACE
from openvitac_tasks.tasks.roughness_regrasp import (
    _APPROACH_CLEARANCE,
    _CLOSE_Q_BY_BLOCK,
    _GRASP_HEIGHT,
    _ROUGH_PLATE,
    _SUCCESS_Z_TOLERANCE,
    _TARGET_BLOCK_Z,
    RoughnessRegrasp,
)

_LEFT = (0.40, 0.12, 0.004)
_RIGHT = (0.40, -0.12, 0.004)
_PARK_RIGHT = (0.30, -0.30, 0.004)
_YELLOW = (0.48, -0.08, 0.002)
_BLUE = (0.48, 0.08, 0.002)


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
        # The task commands the width-derived close q, not the hardware limit.
        if qpos == self.open_qpos:
            if self._qpos != self.open_qpos:
                world.events.append(("open", 1))
                world.holding = None
        elif qpos != self._qpos:
            world.events.append(("close", qpos))
            world.holding = world.nearest_block()
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
        if world.holding is not None:
            world.blocks[world.holding] = (
                pos[0],
                pos[1],
                pos[2] - HAND_TO_GEL_FACE - _GRASP_HEIGHT,
            )


class _World:

    @property
    def logger(self) -> logging.LoggerAdapter[logging.Logger]:
        return logging.LoggerAdapter(logging.getLogger(__name__))

    def __init__(self, params: dict[str, str]) -> None:
        self.params = params
        rough_side = params["rough_block_side"]
        smooth_side = params["smooth_block_side"]
        self.blocks = {
            "smooth": _LEFT if smooth_side == "left" else _RIGHT,
            "rough": _LEFT if rough_side == "left" else _RIGHT,
        }
        self.ee = (0.40, 0.0, 0.30)
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
        return self.params[name]

    def object_pose(
        self, prim_name: str
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        if prim_name == "smooth_block":
            return self.blocks["smooth"], (0.0, 0.0, 0.0, 1.0)
        if prim_name == "rough_block":
            return self.blocks["rough"], (0.0, 0.0, 0.0, 1.0)
        if prim_name == "yellow_plate":
            return _YELLOW, (0.0, 0.0, 0.0, 1.0)
        if prim_name == "blue_plate":
            return _BLUE, (0.0, 0.0, 0.0, 1.0)
        raise KeyError(prim_name)

    def nearest_block(self) -> str:
        return min(
            self.blocks,
            key=lambda name: (self.blocks[name][0] - self.ee[0]) ** 2
            + (self.blocks[name][1] - self.ee[1]) ** 2,
        )


def _world(seed: int) -> tuple[dict[str, str], _World]:
    randomization = RoughnessRegrasp.setup(seed)
    params = {k: str(v) for k, v in randomization.metadata.items()}
    return params, _World(params)


def _run(world: _World) -> list[str]:
    """Drive the registered state machine and record its phase sequence."""
    machine = RoughnessRegrasp.expert
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
    machine = RoughnessRegrasp.expert
    state = machine.init()
    verdicts = RoughnessRegrasp.success(world)
    while True:
        result = machine.step(world, state)
        if result is None:
            raise AssertionError("policy ended before the success gate")
        state, action = result
        world.apply(action)
        if next(verdicts):
            return str(state.phase.name), world.arm.gripper.joint_pos()


class RoughnessRegraspTests(unittest.TestCase):
    def test_randomizer_sides_are_seeded_and_on_table(self) -> None:
        randomization = RoughnessRegrasp.setup(0)
        self.assertEqual(randomization.metadata["rough_block_side"], "right")
        self.assertEqual(randomization.metadata["initial_grasp_side"], "right")
        self.assertEqual(randomization.metadata["smooth_block_side"], "left")

        params, _ = _world(4)
        self.assertEqual(params["rough_block_side"], "left")
        self.assertEqual(params["smooth_block_side"], "right")
        self.assertEqual(params["initial_grasp_side"], "right")

    def test_fixed_nominal_success_uses_rough_on_yellow(self) -> None:
        _, world = _world(0)
        world.blocks["rough"] = (_YELLOW[0], _YELLOW[1], _YELLOW[2] + 0.01)
        world.arm.gripper.set_target(world.arm.gripper.open_qpos)
        world.ee = (
            _YELLOW[0],
            _YELLOW[1],
            _YELLOW[2]
            + _TARGET_BLOCK_Z
            + _GRASP_HEIGHT
            + HAND_TO_GEL_FACE
            + _APPROACH_CLEARANCE,
        )
        self.assertTrue(next(RoughnessRegrasp.success(world)))
        world.blocks["rough"] = (_YELLOW[0] + 0.05, _YELLOW[1], _YELLOW[2] + 0.01)
        self.assertFalse(next(RoughnessRegrasp.success(world)))

    def test_close_position_per_block(self) -> None:
        # 15.00 mm -> 2.936 mm, 15.88 mm -> 3.376 mm (1 mm gel overtravel;
        # measured pad-face gap 7.1276 mm at q=0); the wider block closes more.
        self.assertAlmostEqual(
            _CLOSE_Q_BY_BLOCK["smooth_block"], 0.0029362, places=7
        )
        self.assertAlmostEqual(
            _CLOSE_Q_BY_BLOCK["rough_block"], 0.0033762, places=7
        )
        self.assertLess(
            _CLOSE_Q_BY_BLOCK["smooth_block"],
            _CLOSE_Q_BY_BLOCK["rough_block"],
        )

    def test_first_true_is_after_open_and_retract(self) -> None:
        _, world = _world(0)
        plate, _ = world.object_pose(_ROUGH_PLATE)
        phase, gripper_qpos = _run_until_success(world)
        self.assertEqual(phase, "PLACE_RETRACT")
        self.assertGreaterEqual(
            gripper_qpos, world.arm.gripper.open_qpos - _SUCCESS_Z_TOLERANCE
        )
        retract_z = (
            plate[2]
            + _TARGET_BLOCK_Z
            + _GRASP_HEIGHT
            + HAND_TO_GEL_FACE
            + _APPROACH_CLEARANCE
        )
        self.assertGreaterEqual(world.ee[2], retract_z - _SUCCESS_Z_TOLERANCE)

    def test_expert_places_rough_directly_when_first_grasp_is_rough(self) -> None:
        _, world = _world(0)
        phases = _run(world)
        self.assertEqual(sum(event[0] == "close" for event in world.events), 1)
        self.assertNotIn("PARK_APPROACH", phases)
        self.assertEqual(phases[-1], "PLACE_RETRACT")
        for got, expected in zip(world.blocks["smooth"], _LEFT, strict=True):
            self.assertAlmostEqual(got, expected, places=6)
        self.assertAlmostEqual(world.blocks["rough"][0], _YELLOW[0], places=6)
        self.assertAlmostEqual(world.blocks["rough"][1], _YELLOW[1], places=6)
        self.assertAlmostEqual(world.blocks["rough"][2], _YELLOW[2] + 0.01, places=6)
        self.assertTrue(next(RoughnessRegrasp.success(world)))

    def test_expert_parks_smooth_then_grasps_rough(self) -> None:
        params, world = _world(4)
        self.assertEqual(params["rough_block_side"], "left")
        self.assertEqual(params["initial_grasp_side"], "right")
        phases = _run(world)
        self.assertEqual(sum(event[0] == "close" for event in world.events), 2)
        self.assertIn("PARK_APPROACH", phases)
        self.assertLess(phases.index("PARK_RETRACT"), phases.index("PLACE_APPROACH"))
        self.assertEqual(phases[-1], "PLACE_RETRACT")
        for got, expected in zip(world.blocks["smooth"], _PARK_RIGHT, strict=True):
            self.assertAlmostEqual(got, expected, places=6)
        self.assertAlmostEqual(world.blocks["rough"][0], _YELLOW[0], places=6)
        self.assertAlmostEqual(world.blocks["rough"][1], _YELLOW[1], places=6)
        self.assertAlmostEqual(world.blocks["rough"][2], _YELLOW[2] + 0.01, places=6)
        self.assertTrue(next(RoughnessRegrasp.success(world)))


if __name__ == "__main__":
    unittest.main()
