"""PD geometry tests: drawer poses stay derived from the cabinet and joint anchors."""

from __future__ import annotations

import logging
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from openvitac_spec import GripperCapability
from openvitac_tasks import Action
from openvitac_tasks.motion import pad_close_q, quat_conjugate, quat_rotate
from openvitac_tasks.tasks import pull_drawer


class PullDrawerGeometryTests(unittest.TestCase):
    def test_drawer_poses_match_cabinet_pose_and_joint_anchors(self) -> None:
        cases = (
            (
                pull_drawer._LOWER_DRAWER_POS,
                pull_drawer._LOWER_DRAWER_CLOSED_POS,
                0.0115,
            ),
            (
                pull_drawer._UPPER_DRAWER_POS,
                pull_drawer._UPPER_DRAWER_CLOSED_POS,
                0.0730,
            ),
        )
        for drawer_pos, anchor, expected_z in cases:
            derived = pull_drawer._add(
                pull_drawer._CABINET_POS,
                quat_rotate(pull_drawer._CABINET_ROT_XYZW, anchor),
            )
            for got, want in zip(drawer_pos, derived, strict=True):
                self.assertAlmostEqual(got, want, places=12)
            self.assertAlmostEqual(drawer_pos[0], 0.74, places=12)
            self.assertAlmostEqual(drawer_pos[1], 0.0, places=12)
            self.assertAlmostEqual(drawer_pos[2], expected_z, places=12)

    def test_joint_anchors_are_the_closed_drawer_centres(self) -> None:
        for anchor, expected_z in (
            (pull_drawer._LOWER_DRAWER_CLOSED_POS, 0.0095),
            (pull_drawer._UPPER_DRAWER_CLOSED_POS, 0.0710),
        ):
            self.assertAlmostEqual(anchor[0], 0.0, places=12)
            self.assertAlmostEqual(anchor[1], 0.010, places=12)
            self.assertAlmostEqual(anchor[2], expected_z, places=12)

    def test_close_target_presses_the_gel_into_the_handle_nub(self) -> None:
        # 20 mm nub, 1.5 mm per pad: (0.020 - 0.003 - 0.0071276) / 2.
        self.assertAlmostEqual(pull_drawer._CLOSE_Q, 0.0049362, places=9)
        self.assertAlmostEqual(
            pull_drawer._CLOSE_Q, pad_close_q(0.020, 0.0015), places=12
        )
        # The 1 mm default that suits the blocks would only give 7.85 N.
        self.assertLess(pull_drawer._CLOSE_Q, pad_close_q(0.020))

    def test_drawer_hull_has_the_nub_and_the_outer_silhouette(self) -> None:
        asset = (
            Path(__file__).resolve().parents[3]
            / "assets/pull_drawer/upper_drawer_tet.npz"
        )
        with np.load(asset) as data:
            verts = np.asarray(data["verts"])
        lo = verts.min(0)
        hi = verts.max(0)
        # Outer silhouette: body 150 x 210 x 50 mm plus the 25 mm nub.
        for got, want in zip(lo, (-0.075, -0.105, 0.0), strict=True):
            self.assertAlmostEqual(got, want, delta=5e-4)
        for got, want in zip(hi, (0.075, 0.130, 0.050), strict=True):
            self.assertAlmostEqual(got, want, delta=5e-4)
        # Beyond the cabinet front the mesh exists only where the nub is.
        nub = verts[verts[:, 1] > 0.115]
        self.assertGreater(len(nub), 0, "no nub vertices in the hull")
        self.assertAlmostEqual(float(nub[:, 0].min()), -0.010, delta=5e-4)
        self.assertAlmostEqual(float(nub[:, 0].max()), 0.010, delta=5e-4)
        self.assertAlmostEqual(float(nub[:, 2].min()), 0.015, delta=5e-4)
        self.assertAlmostEqual(float(nub[:, 2].max()), 0.035, delta=5e-4)

    def test_cabinet_boxes_leave_the_drawers_clear(self) -> None:
        # Boxes are cabinet-local; drawer boxes are drawer-local AABBs at their
        # spawned poses.  Rotating into the cabinet frame is a translation
        # (both actors carry the cabinet rotation), so compare AABBs directly.
        # The joint anchor is the drawer origin (its bottom), not its centre:
        # x +-75 mm, y -105..105 mm, z 0..50 mm.
        for anchor in (
            pull_drawer._UPPER_DRAWER_CLOSED_POS,
            pull_drawer._LOWER_DRAWER_CLOSED_POS,
        ):
            lo = np.array(anchor) + np.array((-0.075, -0.105, 0.0))
            hi = np.array(anchor) + np.array((0.075, 0.105, 0.050))
            for centre, bhalf in pull_drawer._CABINET_CAVITY_BOXES:
                blo = np.array(centre) - np.array(bhalf)
                bhi = np.array(centre) + np.array(bhalf)
                overlap = np.minimum(hi, bhi) - np.maximum(lo, blo)
                self.assertTrue(
                    bool(np.any(overlap < 0.0)),
                    f"cabinet box at {centre} intersects the drawer at "
                    f"{anchor}",
                )


class _Gripper:
    open_qpos = 0.04
    close_qpos = pull_drawer._CLOSE_Q

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
            # The drawer actor carries the cabinet's 90 deg rotation.
            return self.drawer, self.cabinet_quat
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
        local = quat_rotate(
            quat_conjugate(world.cabinet_quat),
            (
                world.drawer[0] - world.cabinet[0],
                world.drawer[1] - world.cabinet[1],
                world.drawer[2] - world.cabinet[2],
            ),
        )
        pull = local[1] - pull_drawer._UPPER_DRAWER_CLOSED_POS[1]
        self.assertGreater(pull, pull_drawer._SUCCESS_PULL_DISTANCE)


if __name__ == "__main__":
    unittest.main()
