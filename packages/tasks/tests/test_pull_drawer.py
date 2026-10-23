"""PD geometry tests: drawer poses stay derived from the cabinet and joint anchors."""

from __future__ import annotations

import logging
import unittest
from typing import Any

from openvitac_spec import GripperCapability
from openvitac_tasks import Action
from openvitac_tasks.tasks import pull_drawer


class PullDrawerGeometryTests(unittest.TestCase):
    def test_drawer_poses_match_cabinet_pose_and_joint_anchors(self) -> None:
        cases = (
            (
                pull_drawer._LOWER_DRAWER_POS,
                pull_drawer._LOWER_DRAWER_JOINT_POS,
                0.0115,
            ),
            (
                pull_drawer._UPPER_DRAWER_POS,
                pull_drawer._UPPER_DRAWER_JOINT_POS,
                0.0730,
            ),
        )
        for drawer_pos, anchor, expected_z in cases:
            derived = pull_drawer._add(
                pull_drawer._CABINET_POS,
                pull_drawer._rotate(pull_drawer._CABINET_ROT_XYZW, anchor),
            )
            for got, want in zip(drawer_pos, derived, strict=True):
                self.assertAlmostEqual(got, want, places=12)
            self.assertAlmostEqual(drawer_pos[0], 0.74, places=12)
            self.assertAlmostEqual(drawer_pos[1], 0.0, places=12)
            self.assertAlmostEqual(drawer_pos[2], expected_z, places=12)

    def test_joint_anchors_are_the_closed_drawer_centres(self) -> None:
        for anchor, expected_z in (
            (pull_drawer._LOWER_DRAWER_JOINT_POS, 0.0095),
            (pull_drawer._UPPER_DRAWER_JOINT_POS, 0.0710),
        ):
            self.assertAlmostEqual(anchor[0], 0.0, places=12)
            self.assertAlmostEqual(anchor[1], 0.010, places=12)
            self.assertAlmostEqual(anchor[2], expected_z, places=12)


class _Gripper:
    open_qpos = 0.04
    close_qpos = 0.0

    @property
    def capabilities(self) -> frozenset[GripperCapability]:
        return frozenset({"binary_close"})

    def __init__(self) -> None:
        self.qpos = 0.02
        self.holding = False

    def set_target(self, qpos: float) -> None:
        if qpos == self.close_qpos and self.qpos != self.close_qpos:
            self.holding = True
        elif qpos == self.open_qpos and self.qpos != self.open_qpos:
            self.holding = False
        self.qpos = qpos

    def joint_pos(self) -> float:
        return self.qpos


class _Arm:

    def joint_pos(self) -> Any:
        return None

    def gripper_contact(self, *, force_threshold: float = 1.0) -> tuple[bool, bool]:
        return (False, False)

    def __init__(self, world: _World) -> None:
        self._world = world

    @property
    def gripper(self) -> _Gripper:
        return self._world.gripper

    def ee_pose(self):
        return self._world.ee, self._world.quat

    def set_ee_target(self, pos, quat) -> None:
        world = self._world
        if world.gripper.holding:
            world.drawer = (
                world.drawer[0] + pos[0] - world.ee[0],
                world.drawer[1] + pos[1] - world.ee[1],
                world.drawer[2] + pos[2] - world.ee[2],
            )
        world.ee = pos
        world.quat = quat


class _World:

    @property
    def logger(self) -> logging.LoggerAdapter[logging.Logger]:
        return logging.LoggerAdapter(logging.getLogger(__name__))

    def __init__(self) -> None:
        self.cabinet = pull_drawer._CABINET_POS
        self.cabinet_quat = pull_drawer._CABINET_ROT_XYZW
        self.drawer = pull_drawer._UPPER_DRAWER_POS
        self.ee = (0.74, 0.0, 0.30)
        self.quat = (0.0, 0.0, 0.0, 1.0)
        self.gripper = _Gripper()
        self._arm = _Arm(self)

    @property
    def arm(self) -> _Arm:
        return self._arm

    def apply(self, action: Action) -> None:
        self.arm.set_ee_target(action.ee_pos, action.ee_quat)
        self.gripper.set_target(action.gripper_qpos)

    def object_pose(self, prim_name: str):
        if prim_name == "upper_drawer":
            return self.drawer, (0.0, 0.0, 0.0, 1.0)
        if prim_name == "cabinet":
            return self.cabinet, self.cabinet_quat
        raise KeyError(prim_name)

    def expert_param(self, name: str) -> str:
        raise KeyError(name)


def _run_until_success(world: _World) -> str:
    """Drive the machine like the collector; return the first-true phase."""
    machine = pull_drawer.PullDrawer.expert
    state = machine.init()
    verdicts = pull_drawer.PullDrawer.success(world)
    while True:
        result = machine.step(world, state)
        if result is None:
            raise AssertionError("policy ended before the success gate")
        state, action = result
        world.apply(action)
        if next(verdicts):
            return str(state.phase.name)


class PullDrawerExpertTests(unittest.TestCase):
    def test_first_true_is_during_pull_with_gripper_closed(self) -> None:
        world = _World()
        phase = _run_until_success(world)
        self.assertEqual(phase, "PULL")
        self.assertAlmostEqual(world.gripper.joint_pos(), world.gripper.close_qpos)
        local = pull_drawer._rotate(
            pull_drawer._conjugate(world.cabinet_quat),
            (
                world.drawer[0] - world.cabinet[0],
                world.drawer[1] - world.cabinet[1],
                world.drawer[2] - world.cabinet[2],
            ),
        )
        pull = local[1] - pull_drawer._UPPER_DRAWER_JOINT_POS[1]
        self.assertGreater(pull, pull_drawer._SUCCESS_PULL_DISTANCE)


if __name__ == "__main__":
    unittest.main()
