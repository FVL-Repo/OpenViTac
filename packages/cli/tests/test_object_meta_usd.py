"""Every task's v3 object meta keeps its source USD provenance."""

from __future__ import annotations

import unittest

import numpy as np

from openvitac_cli.commands.collect import _tactile_meta
from openvitac_sim.tactile_bridge import PadSurface
from openvitac_sim.tactile_plan import tactile_plan_from_scene
from openvitac_tasks import TASKS


def _fake_frames_and_surfaces(cfg):
    frames = {}
    surfaces = {}
    rest = np.array(
        [[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.0, 0.001, 0.0]],
        dtype=np.float64,
    )
    for index, pad in enumerate(cfg.pads):
        frame = np.eye(4, dtype=np.float64)
        frame[:3, 3] = (0.40 + 0.01 * index, 0.0, 0.20)
        frames[pad.name] = frame
        surfaces[pad.name] = PadSurface(
            name=pad.name,
            side=pad.side,
            rest_vertices=rest + frame[:3, 3],
            triangles=np.array([[0, 1, 2]], dtype=np.int32),
        )
    return frames, surfaces


class TaskObjectUsdMetaTests(unittest.TestCase):
    def test_every_task_object_keeps_usd_provenance(self) -> None:
        self.assertEqual(len(TASKS), 9)
        for name, task in TASKS.items():
            with self.subTest(task=name):
                cfg = tactile_plan_from_scene(task.scene_spec())
                frames, surfaces = _fake_frames_and_surfaces(cfg)
                _, objects = _tactile_meta(
                    cfg,
                    pad_world_frames=frames,
                    pad_surfaces=surfaces,
                )
                self.assertTrue(objects, name)
                for obj_meta in objects:
                    self.assertTrue(obj_meta.usd, f"{name}:{obj_meta.name}")
                for obj, _obj_meta in zip(cfg.objects, objects, strict=True):
                    if obj.authority == "uipc" and not obj.fixed:
                        self.assertIsNotNone(obj.mesh_path, f"{name}:{obj.name}")
                        self.assertNotEqual(
                            obj.mesh_path, obj.usd_path, f"{name}:{obj.name}"
                        )

    def test_physx_and_uipc_objects_both_carry_usd(self) -> None:
        cfg = tactile_plan_from_scene(TASKS["insert_usb"].scene_spec())
        by_name = {obj.name: obj for obj in cfg.objects}
        usb = by_name["usb"]
        self.assertTrue(str(usb.usd_path).endswith("usb.usd"))
        self.assertTrue(str(usb.mesh_path).endswith("usb_tet.npz"))
        self.assertNotEqual(usb.usd_path, usb.mesh_path)
        for obj in cfg.objects:
            self.assertTrue(obj.usd_path, obj.name)


if __name__ == "__main__":
    unittest.main()
