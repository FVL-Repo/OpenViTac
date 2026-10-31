"""Roughness-classify randomization/expert/success tests without Isaac."""

from __future__ import annotations

import logging
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from openvitac_spec import GripperCapability
from openvitac_tasks import Action
from openvitac_tasks.motion import HAND_TO_GEL_FACE, PAD_NORMAL_IN_HAND
from openvitac_tasks.tasks.roughness_classify import (
    _APPROACH_CLEARANCE,
    _CLOSE_Q_BY_BLOCK,
    _GRASP_HEIGHT,
    _SUCCESS_Z_TOLERANCE,
    _TARGET_BLOCK_Z,
    RoughnessClassify,
)

_PLATES = {
    "blue_plate": (0.45, 0.08, 0.002),
    "yellow_plate": (0.45, -0.08, 0.002),
}
_BLOCKS = {"smooth": "smooth_block", "rough": "rough_block"}
_TARGETS = {"smooth": "blue_plate", "rough": "yellow_plate"}
_START = (0.38, 0.0, 0.004)
_STANDBY = (0.30, 0.25, 0.004)


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
        # The task commands its width-derived close, not the hardware limit.
        if qpos in _CLOSE_Q_BY_BLOCK.values() and self._qpos != qpos:
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
                pos[2] - HAND_TO_GEL_FACE - _GRASP_HEIGHT,
            )


class _World:

    @property
    def logger(self) -> logging.LoggerAdapter[logging.Logger]:
        return logging.LoggerAdapter(logging.getLogger(__name__))

    def __init__(self, label: str) -> None:
        self.label = label
        self.ee = (0.38, 0.0, 0.30)
        self.quat = (0.0, 0.0, 0.0, 1.0)
        self.blocks = {"smooth": _START, "rough": _START}
        self.blocks["rough" if label == "smooth" else "smooth"] = _STANDBY
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
        if name == "roughness_label":
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
    machine = RoughnessClassify.expert
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
    machine = RoughnessClassify.expert
    state = machine.init()
    verdicts = RoughnessClassify.success(world)
    while True:
        result = machine.step(world, state)
        if result is None:
            raise AssertionError("policy ended before the success gate")
        state, action = result
        world.apply(action)
        if next(verdicts):
            return str(state.phase.name), world.arm.gripper.joint_pos()


class RoughnessClassifyTests(unittest.TestCase):
    def test_pad_normal_matches_authored_mount(self) -> None:
        from pxr import Usd, UsdGeom

        asset = (
            Path(__file__).resolve().parents[3]
            / "assets/embodiment/franka_gsmini.usda"
        )
        stage = Usd.Stage.Open(str(asset))
        cache = UsdGeom.XformCache()
        hand = np.array(
            cache.GetLocalToWorldTransform(
                stage.GetPrimAtPath("/panda/panda_hand")
            )
        ).T
        gelpad = np.array(
            cache.GetLocalToWorldTransform(
                stage.GetPrimAtPath("/panda/gelpad_left")
            )
        ).T
        normal = (np.linalg.inv(hand) @ gelpad)[:3, 2]
        for got, want in zip(normal, PAD_NORMAL_IN_HAND, strict=True):
            self.assertAlmostEqual(float(got), want, places=4)

    def test_close_position_matches_the_recorded_command(self) -> None:
        # Both blocks are grasped on the gel, so each close comes from its own
        # width (1 mm gel overtravel; measured pad-face gap 7.1276 mm at q=0).
        self.assertAlmostEqual(_CLOSE_Q_BY_BLOCK["smooth_block"], 0.0029362, places=7)
        self.assertAlmostEqual(_CLOSE_Q_BY_BLOCK["rough_block"], 0.0033762, places=7)

    def test_randomizer_samples_both_labels_and_table_standby(self) -> None:
        seen: set[str] = set()
        for seed in range(4):
            randomization = RoughnessClassify.setup(seed)
            label = randomization.metadata["roughness_label"]
            target = randomization.metadata["target_plate"]
            assert isinstance(label, str)
            assert isinstance(target, str)
            seen.add(label)
            self.assertEqual(label, "smooth" if target == "blue_plate" else "rough")
            self.assertEqual(target, _TARGETS[label])
            self.assertEqual(
                randomization.instruction,
                f"Place the {label} block on the {target.replace('_', ' ')}.",
            )
            selected = _BLOCKS[label]
            other = _BLOCKS["rough" if label == "smooth" else "smooth"]
            self.assertEqual(
                randomization.contact_object, f"/World/actors/{selected}"
            )
            self.assertEqual(
                randomization.actor_poses[f"/World/actors/{selected}"].position,
                _START,
            )
            self.assertEqual(
                randomization.actor_poses[f"/World/actors/{other}"].position,
                _STANDBY,
            )
        self.assertEqual(seen, {"smooth", "rough"})

    def test_success_uses_selected_label(self) -> None:
        for label in ("smooth", "rough"):
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
                + HAND_TO_GEL_FACE
                + _APPROACH_CLEARANCE,
            )
            self.assertTrue(next(RoughnessClassify.success(world)))

            world.blocks[label] = (target[0] + 0.05, target[1], target[2] + 0.01)
            self.assertFalse(next(RoughnessClassify.success(world)))

    def test_first_true_is_after_open_and_retract(self) -> None:
        target = _PLATES[_TARGETS["smooth"]]
        world = _World("smooth")
        phase, gripper_qpos = _run_until_success(world)
        self.assertEqual(phase, "RETRACT")
        self.assertGreaterEqual(
            gripper_qpos, world.arm.gripper.open_qpos - _SUCCESS_Z_TOLERANCE
        )
        retract_z = (
            target[2]
            + _TARGET_BLOCK_Z
            + _GRASP_HEIGHT
            + HAND_TO_GEL_FACE
            + _APPROACH_CLEARANCE
        )
        self.assertGreaterEqual(world.ee[2], retract_z - _SUCCESS_Z_TOLERANCE)

    def test_expert_moves_selected_block_to_its_plate(self) -> None:
        for label in ("smooth", "rough"):
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
            self.assertAlmostEqual(placed[2], target[2] + 0.01, places=6)
            other = "rough" if label == "smooth" else "smooth"
            self.assertEqual(world.blocks[other], _STANDBY)
            self.assertTrue(next(RoughnessClassify.success(world)))


if __name__ == "__main__":
    unittest.main()
