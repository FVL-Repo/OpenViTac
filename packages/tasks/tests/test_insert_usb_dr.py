"""insert_usb P0 scene-pose randomization tests (pure, no Isaac)."""

from __future__ import annotations

import unittest

from openvitac_tasks import TASKS


class InsertUsbRandomizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = TASKS["insert_usb"]
        self.actors = {actor.prim_path: actor for actor in self.task.actors}

    def test_start_and_target_slots_are_static_fixtures(self) -> None:
        self.assertFalse(self.actors["/World/actors/slot"].dynamic)
        self.assertFalse(self.actors["/World/actors/start_slot"].dynamic)

    def test_usb_starts_with_clearance_above_start_support(self) -> None:
        from openvitac_tasks.tasks.insert_usb import _START_SUPPORT_BOXES

        start_z = self.actors["/World/actors/start_slot"].pos[2]
        bottom_center, bottom_half = _START_SUPPORT_BOXES[-1]
        support_top = start_z + bottom_center[2] + bottom_half[2]
        usb_tip = self.actors["/World/actors/usb"].pos[2]
        self.assertGreaterEqual(usb_tip - support_top, 0.001)

    def test_seed_is_deterministic_and_instruction_is_preserved(self) -> None:
        first = self.task.setup(7)
        second = self.task.setup(7)
        self.assertEqual(first, second)
        self.assertEqual(first.seed, 7)
        self.assertEqual(first.instruction, self.task.instruction)
        self.assertNotEqual(first.actor_poses, self.task.setup(8).actor_poses)

    def test_slot_xy_bounds_and_nominal_z_quat(self) -> None:
        dr = self.task.setup(0)
        for leaf in ("start_slot", "slot"):
            path = f"/World/actors/{leaf}"
            pose = dr.actor_poses[path]
            nominal = self.actors[path]
            self.assertGreaterEqual(pose.position[0], nominal.pos[0] - 0.030)
            self.assertLessEqual(pose.position[0], nominal.pos[0] + 0.030)
            self.assertGreaterEqual(pose.position[1], nominal.pos[1] - 0.030)
            self.assertLessEqual(pose.position[1], nominal.pos[1] + 0.030)
            self.assertEqual(pose.position[2], nominal.pos[2])
            self.assertEqual(pose.quaternion_xyzw, nominal.rot)

    def test_usb_follows_start_slot_without_independent_sampling(self) -> None:
        dr = self.task.setup(3)
        start = dr.actor_poses["/World/actors/start_slot"]
        usb = dr.actor_poses["/World/actors/usb"]
        nominal_start = self.actors["/World/actors/start_slot"]
        nominal_usb = self.actors["/World/actors/usb"]
        self.assertAlmostEqual(
            usb.position[0] - start.position[0],
            nominal_usb.pos[0] - nominal_start.pos[0],
        )
        self.assertAlmostEqual(
            usb.position[1] - start.position[1],
            nominal_usb.pos[1] - nominal_start.pos[1],
        )
        self.assertEqual(usb.position[2], nominal_usb.pos[2])
        self.assertEqual(usb.quaternion_xyzw, nominal_usb.rot)

    def test_metadata_is_flat_and_contains_actor_pose(self) -> None:
        dr = self.task.setup(1)
        meta = dr.to_metadata()
        self.assertEqual(dr.expert_params, {})
        self.assertEqual(meta["dr.version"], "insert_usb.scene_pose.v1")
        self.assertAlmostEqual(float(meta["dr.slot_xy_half_m"]), 0.030)
        self.assertIn("actor.start_slot.pos_x", meta)
        self.assertIn("actor.start_slot.quat_w", meta)
        self.assertTrue(all(isinstance(value, (float, str)) for value in meta.values()))


if __name__ == "__main__":
    unittest.main()
