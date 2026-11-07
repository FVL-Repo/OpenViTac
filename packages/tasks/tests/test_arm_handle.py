"""ArmHandle surface and ``ExpertWorld.arm`` typing (no Isaac)."""

from __future__ import annotations

import unittest
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from openvitac_spec import GripperCapability
from openvitac_tasks.expert import ArmHandle, ExpertWorld, Quat, Vec3


class _Gripper:
    @property
    def capabilities(self) -> frozenset[GripperCapability]:
        return frozenset[GripperCapability]()

    @property
    def open_qpos(self) -> float:
        return 0.04

    @property
    def close_qpos(self) -> float:
        return 0.0

    def set_target(self, qpos: float) -> None: ...

    def joint_pos(self) -> float:
        return 0.0


class _Arm:
    @property
    def gripper(self) -> _Gripper:
        return _Gripper()

    def ee_pose(self) -> tuple[Vec3, Quat]:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)

    def set_ee_target(self, pos: Vec3, quat: Quat) -> None: ...

    def joint_pos(self) -> npt.NDArray[np.floating[Any]]:
        return np.zeros(0)

    def gripper_contact(self, *, force_threshold: float = 1.0) -> tuple[bool, bool]:
        return (False, False)


class ArmHandleSurfaceTests(unittest.TestCase):
    def test_expert_world_arm_is_a_property(self) -> None:
        self.assertIsInstance(ExpertWorld.__dict__["arm"], property)

    def test_arm_handle_owns_motion_gripper_and_sensing(self) -> None:
        self.assertIsInstance(ArmHandle.__dict__["gripper"], property)
        for name in ("ee_pose", "set_ee_target", "joint_pos", "gripper_contact"):
            self.assertIn(name, ArmHandle.__dict__)

    def test_fake_arm_is_usable_through_the_protocol(self) -> None:
        arm = cast(ArmHandle, _Arm())
        pos, quat = arm.ee_pose()
        arm.set_ee_target(pos, quat)
        arm.gripper.set_target(arm.gripper.close_qpos)
        arm.gripper.set_target(arm.gripper.open_qpos)
        self.assertEqual(arm.gripper_contact(), (False, False))


if __name__ == "__main__":
    unittest.main()
