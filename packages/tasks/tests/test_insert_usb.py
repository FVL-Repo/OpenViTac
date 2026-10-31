"""insert_usb calibration and state-machine tests (no Isaac)."""


from __future__ import annotations

import inspect
import logging
import unittest
from typing import Any

from openvitac_spec import GripperCapability
from openvitac_tasks import Action, InsertUsb
from openvitac_tasks.tasks.insert_usb import (
    _APPROACH_CLEARANCE,
    _EE_RETRACT_Z,
    _GRASP_HEIGHT,
    _GRASP_TCP_OFFSET,
    _INSERT_Z,
    _LIFT_HEIGHT,
    _SLOT_HEIGHT,
    _SUCCESS_EE_Z,
    _grasp_target,
)

_USB_START = (0.4, 0.0, 0.0085)
_SLOT = (0.52, 0.0, 0.002)
_PHASE_ORDER = (
    "PREGRASP",
    "DESCEND",
    "CLOSE",
    "LIFT",
    "CARRY",
    "ALIGN",
    "CORRECT_1",
    "CORRECT_2",
    "INSERT",
    "OPEN",
    "RETRACT",
)


class InsertUsbGraspTargetTests(unittest.TestCase):
    def test_target_puts_pad_midpoint_at_grasp_point(self) -> None:
        grasp = (0.4, -0.01, 0.05)
        target = _grasp_target(*grasp)
        pad_midpoint = tuple(target[i] - _GRASP_TCP_OFFSET[i] for i in range(3))
        for got, want in zip(pad_midpoint, grasp, strict=True):
            self.assertAlmostEqual(got, want, places=12)

    def test_clearance_only_lifts_the_target(self) -> None:
        base = _grasp_target(0.4, 0.0, 0.05)
        lifted = _grasp_target(0.4, 0.0, 0.05, clearance=0.01)
        self.assertAlmostEqual(lifted[2] - base[2], 0.01, places=12)
        self.assertEqual(lifted[:2], base[:2])


class _FakeGripper:
    open_qpos = 0.039
    close_qpos = 0.0

    @property
    def capabilities(self) -> frozenset[GripperCapability]:
        return frozenset[GripperCapability]({"binary_close"})

    def __init__(self, world: _FakeWorld) -> None:
        self._world = world
        self.qpos = 0.02

    def joint_pos(self) -> float:
        return self.qpos

    def set_target(self, qpos: float) -> None:
        self.qpos = qpos
        if qpos == self.close_qpos:
            self._world.holding = True
        elif qpos == self.open_qpos:
            self._world.holding = False


class _FakeArm:

    def joint_pos(self) -> Any:
        return None

    def gripper_contact(self, *, force_threshold: float = 1.0) -> tuple[bool, bool]:
        return (False, False)

    def __init__(self, world: _FakeWorld) -> None:
        self._world = world
        self.gripper = _FakeGripper(world)
        self.pos = (0.4, 0.0, 0.25)
        self.quat = (0.0, 0.0, 0.0, 1.0)

    def ee_pose(self):
        return self.pos, self.quat

    def set_ee_target(self, pos, quat) -> None:
        world = self._world
        if world.holding:
            (ux, uy, uz), uq = world.poses["usb"]
            world.poses["usb"] = (
                (
                    ux + pos[0] - self.pos[0],
                    uy + pos[1] - self.pos[1],
                    uz + pos[2] - self.pos[2],
                ),
                uq,
            )
        self.pos = pos
        self.quat = quat


class _FakeWorld:
    """Teleporting world; a grasped USB rigidly follows the hand."""

    def __init__(self) -> None:
        self.poses = {
            "usb": (_USB_START, (0.0, 0.0, 0.0, 1.0)),
            "slot": (_SLOT, (0.0, 0.0, 0.0, 1.0)),
        }
        self.holding = False
        self.arm = _FakeArm(self)

    @property
    def logger(self) -> logging.LoggerAdapter[logging.Logger]:
        return logging.LoggerAdapter(logging.getLogger(__name__))

    def object_pose(self, prim_name: str) -> Any:
        return self.poses[prim_name]

    def expert_param(self, name: str) -> str:
        raise KeyError(name)


class InsertUsbStateMachineTests(unittest.TestCase):
    def _run(
        self,
    ) -> tuple[_FakeWorld, list[str], dict[str, Action], object]:
        machine = InsertUsb.expert
        world = _FakeWorld()
        state = machine.init()
        phases: list[str] = []
        last_action: dict[str, Action] = {}
        phase = ""
        for _ in range(4096):
            result = machine.step(world, state)
            if result is None:
                break
            state, action = result
            phase = state.phase.name
            if not phases or phases[-1] != phase:
                phases.append(phase)
            last_action[phase] = action
            self.assertEqual(len(action.ee_pos), 3)
            self.assertEqual(len(action.ee_quat), 4)
            world.arm.set_ee_target(action.ee_pos, action.ee_quat)
            world.arm.gripper.set_target(action.gripper_qpos)
        else:
            self.fail("state machine did not terminate within 4096 ticks")
        self.assertEqual(phase, "RETRACT")
        return world, phases, last_action, state

    def test_phase_order_and_representative_targets(self) -> None:
        world, phases, last, _ = self._run()
        self.assertEqual(tuple(phases), _PHASE_ORDER)

        grasp_z = _USB_START[2] + _GRASP_HEIGHT
        descend = _grasp_target(_USB_START[0], _USB_START[1], grasp_z)
        hover = _grasp_target(
            _USB_START[0], _USB_START[1], grasp_z, clearance=_APPROACH_CLEARANCE
        )
        lift = _grasp_target(
            _USB_START[0], _USB_START[1], grasp_z + _LIFT_HEIGHT
        )
        carry = _grasp_target(
            _SLOT[0],
            _SLOT[1],
            _SLOT[2] + _SLOT_HEIGHT + _GRASP_HEIGHT,
            clearance=_APPROACH_CLEARANCE,
        )
        align = _grasp_target(
            _SLOT[0], _SLOT[1], _SLOT[2] + _SLOT_HEIGHT + _GRASP_HEIGHT
        )

        pregrasp_action = last["PREGRASP"]
        descend_action = last["DESCEND"]
        self.assertAlmostEqual(pregrasp_action.ee_pos[2], hover[2], places=9)
        self.assertAlmostEqual(descend_action.ee_pos[2], descend[2], places=9)
        self.assertAlmostEqual(last["CLOSE"].gripper_qpos, 0.0, places=12)
        self.assertAlmostEqual(last["LIFT"].ee_pos[2], lift[2], places=9)
        self.assertAlmostEqual(last["CARRY"].ee_pos[0], carry[0], places=9)
        self.assertAlmostEqual(last["ALIGN"].ee_pos[2], align[2], places=9)
        self.assertLess(last["INSERT"].ee_pos[2], last["ALIGN"].ee_pos[2])
        self.assertAlmostEqual(last["OPEN"].gripper_qpos, 0.039, places=12)
        self.assertAlmostEqual(last["RETRACT"].ee_pos[2], _EE_RETRACT_Z, places=9)

        (ux, uy, uz), _ = world.object_pose("usb")
        self.assertAlmostEqual(ux, _SLOT[0], places=9)
        self.assertAlmostEqual(uy, _SLOT[1], places=9)
        self.assertAlmostEqual(uz, _SLOT[2] + _INSERT_Z, places=9)

    def test_done_is_sticky_and_stops_emitting(self) -> None:
        machine = InsertUsb.expert
        world, _, _, state = self._run()
        self.assertIsNone(machine.step(world, state))
        self.assertIsNone(machine.step(world, state))


class InsertUsbSuccessGeneratorTests(unittest.TestCase):
    def test_success_iterator_reads_live_world_across_ticks(self) -> None:
        self.assertTrue(inspect.isgeneratorfunction(InsertUsb.success))
        world = _FakeWorld()
        verdicts = InsertUsb.success(world)
        self.assertFalse(next(verdicts))

        usb_quat = world.poses["usb"][1]
        world.poses["usb"] = (
            (_SLOT[0], _SLOT[1], _SLOT[2] + _INSERT_Z),
            usb_quat,
        )
        world.arm.pos = (_SLOT[0], _SLOT[1], _SUCCESS_EE_Z + 0.01)
        self.assertTrue(next(verdicts))

        world.poses["usb"] = (
            (_SLOT[0] + 0.01, _SLOT[1], _SLOT[2] + _INSERT_Z),
            usb_quat,
        )
        self.assertFalse(next(verdicts))


if __name__ == "__main__":
    unittest.main()
