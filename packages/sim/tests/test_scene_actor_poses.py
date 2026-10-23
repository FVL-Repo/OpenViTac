"""Scene reset-time actor-pose application tests (no Isaac app)."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import torch
from pxr import Gf, Usd, UsdGeom

from openvitac_sim import scene as scene_module
from openvitac_sim.scene import Scene
from openvitac_spec import ActorPose


class _FakeRigid:
    def __init__(self) -> None:
        self.poses: list[list[float]] = []
        self.velocities: list[torch.Tensor] = []

    def write_root_pose_to_sim(self, pose: torch.Tensor) -> None:
        self.poses.append(pose[0].tolist())

    def write_root_velocity_to_sim(self, velocity: torch.Tensor) -> None:
        self.velocities.append(velocity)


def _scene(actors: list[SimpleNamespace], rigids: dict[str, _FakeRigid]) -> Scene:
    scene = Scene.__new__(Scene)
    scene._spec = cast(Any, SimpleNamespace(actors=tuple(actors), sensors=()))
    scene._rigids = cast(Any, rigids)
    scene._actor_pose_overrides = cast(Any, {})
    scene._arm = cast(
        Any, SimpleNamespace(robot=SimpleNamespace(device=torch.device("cpu")))
    )
    scene._body_index_cache = cast(Any, {})
    return scene


def _stage(
    *prim_paths: str,
    parent_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    parent_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World")
    actors = UsdGeom.Xform.Define(stage, "/World/actors")
    xform = UsdGeom.Xformable(actors.GetPrim())
    if parent_position != (0.0, 0.0, 0.0):
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(*parent_position)
        )
    if parent_quat_xyzw != (0.0, 0.0, 0.0, 1.0):
        qx, qy, qz, qw = parent_quat_xyzw
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz))
        )
    for path in prim_paths:
        prim = UsdGeom.Xform.Define(stage, path)
        xform = UsdGeom.Xformable(prim.GetPrim())
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    return stage


def _world_pose(stage: Usd.Stage, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
        stage.GetPrimAtPath(prim_path)
    )
    translation = matrix.ExtractTranslation()
    rotation = matrix.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()
    return (
        np.asarray(translation, dtype=np.float64),
        np.asarray(
            [imaginary[0], imaginary[1], imaginary[2], rotation.GetReal()],
            dtype=np.float64,
        ),
    )


class ActorPoseApplicationTests(unittest.TestCase):
    def test_dynamic_actor_pose_is_written_with_zero_velocity(self) -> None:
        actor = SimpleNamespace(
            prim_path="/World/actors/start_slot",
            dynamic=True,
            uipc_fixed_boxes=(),
            uipc_fixed_mesh=None,
        )
        rigid = _FakeRigid()
        scene = _scene([actor], {actor.prim_path: rigid})
        pose = ActorPose((0.41, -0.02, 0.002), (0.0, 0.0, 0.0, 1.0))
        scene._apply_actor_poses({"/World/actors/start_slot": pose})
        expected = [0.41, -0.02, 0.002, 0.0, 0.0, 0.0, 1.0]
        for actual, wanted in zip(rigid.poses[0], expected, strict=True):
            self.assertAlmostEqual(actual, wanted, places=6)
        self.assertEqual(len(rigid.velocities), 1)
        self.assertEqual(rigid.velocities[0].tolist(), [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertEqual(scene._actor_pose_overrides[actor.prim_path], pose)

    def test_leaf_name_is_resolved_and_static_plain_actor_fails_loud(self) -> None:
        dynamic = SimpleNamespace(
            prim_path="/World/actors/usb",
            dynamic=True,
            uipc_fixed_boxes=(),
            uipc_fixed_mesh=None,
        )
        plate = SimpleNamespace(
            prim_path="/World/plate",
            dynamic=False,
            uipc_fixed_boxes=(),
            uipc_fixed_mesh=None,
        )
        scene = _scene([dynamic, plate], {dynamic.prim_path: _FakeRigid()})
        pose = ActorPose((0.4, 0.0, 0.0075), (0.0, 0.0, 0.0, 1.0))
        scene._apply_actor_poses({"usb": pose})
        self.assertEqual(scene._actor_pose_overrides[dynamic.prim_path], pose)
        with self.assertRaises(ValueError):
            scene._apply_actor_poses({"/World/plate": pose})

    def test_static_uipc_fixed_actor_has_no_physx_body_but_is_retained(self) -> None:
        fixed = SimpleNamespace(
            prim_path="/World/actors/start_slot",
            dynamic=False,
            uipc_fixed_boxes=((((0.0, 0.0, 0.0)), (0.1, 0.1, 0.1)),),
            uipc_fixed_mesh=None,
        )
        scene = _scene([fixed], {})
        pose = ActorPose((0.41, 0.0, 0.002), (0.0, 0.0, 0.0, 1.0))
        with patch.object(
            scene_module.sim_utils,
            "get_current_stage",
            return_value=_stage(fixed.prim_path),
        ):
            scene._apply_actor_poses({fixed.prim_path: pose})
        self.assertEqual(scene._actor_pose_overrides[fixed.prim_path], pose)
        self.assertEqual(scene._rigids, {})

    def test_object_pose_reads_static_actor_override(self) -> None:
        actor = SimpleNamespace(
            prim_path="/World/actors/slot",
            pos=(0.52, 0.0, 0.002),
            rot=(0.0, 0.0, 0.0, 1.0),
            dynamic=False,
            uipc_fixed_boxes=(),
            uipc_fixed_mesh="slot_target_tet.npz",
        )
        scene = _scene([actor], {})
        pose = ActorPose((0.49, -0.02, 0.002), (0.0, 0.0, 0.0, 1.0))
        with patch.object(
            scene_module.sim_utils,
            "get_current_stage",
            return_value=_stage(actor.prim_path),
        ):
            scene._apply_actor_poses({"slot": pose})
        self.assertEqual(
            scene.object_pose("slot"), (pose.position, pose.quaternion_xyzw)
        )
        self.assertEqual(
            scene.object_pose("/World/actors/slot"),
            (pose.position, pose.quaternion_xyzw),
        )

    def test_tactile_object_pose_uses_static_override_for_fixed_fixture(
        self,
    ) -> None:
        actor = SimpleNamespace(
            prim_path="/World/actors/start_slot",
            pos=(0.4, 0.0, 0.002),
            rot=(0.0, 0.0, 0.0, 1.0),
            dynamic=False,
            uipc_fixed_boxes=((((0.0, 0.0, 0.0)), (0.1, 0.1, 0.1)),),
            uipc_fixed_mesh=None,
        )
        scene = _scene([actor], {})
        pose = ActorPose((0.41, -0.02, 0.002), (0.0, 0.0, 0.0, 1.0))
        with patch.object(
            scene_module.sim_utils,
            "get_current_stage",
            return_value=_stage(actor.prim_path),
        ):
            scene._apply_actor_poses({actor.prim_path: pose})
        self.assertEqual(scene._rigids, {})
        scene._tactile_bridge = cast(
            Any,
            SimpleNamespace(
                cfg=SimpleNamespace(
                    objects=(
                        SimpleNamespace(
                            name="start_slot_static",
                            prim_path=actor.prim_path,
                        ),
                    )
                )
            ),
        )
        out = scene._tactile_object_poses()
        np.testing.assert_allclose(
            out["start_slot_static"][:3, 3], pose.position
        )

    def test_reset_order_is_sim_poses_arm_settle_update(self) -> None:
        events: list[str] = []
        scene = Scene.__new__(Scene)
        scene._reset_done = cast(Any, False)
        scene._spec = cast(Any, SimpleNamespace(sensors=()))
        scene._sim = cast(Any, SimpleNamespace(reset=lambda: events.append("sim")))
        scene._apply_actor_poses = cast(Any, lambda poses: events.append("poses"))
        scene._arm = cast(
            Any, SimpleNamespace(reset_to_init=lambda: events.append("arm"))
        )
        scene._update_actors = cast(Any, lambda: events.append("update"))
        scene._tactile_bridge = cast(Any, None)
        scene.reset(actor_poses={})
        self.assertEqual(events, ["sim", "poses", "arm", "update"])


class StaticVisualPoseTests(unittest.TestCase):
    def _assert_visual_pose(
        self,
        pose: ActorPose,
        *,
        parent_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        parent_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    ) -> None:
        stage = _stage(
            "/World/actors/slot",
            parent_position=parent_position,
            parent_quat_xyzw=parent_quat_xyzw,
        )
        with patch.object(
            scene_module.sim_utils, "get_current_stage", return_value=stage
        ):
            scene_module._set_static_visual_pose("/World/actors/slot", pose)
        actual_position, actual_quat = _world_pose(stage, "/World/actors/slot")
        np.testing.assert_allclose(actual_position, pose.position, atol=1e-9)
        self.assertGreaterEqual(
            abs(float(np.dot(actual_quat, pose.quaternion_xyzw))), 1.0 - 1e-9
        )

    def test_identity_pose(self) -> None:
        self._assert_visual_pose(
            ActorPose((0.51523, -0.01446, 0.002), (0.0, 0.0, 0.0, 1.0))
        )

    def test_45_degree_pose(self) -> None:
        half = math.sqrt(0.5)
        self._assert_visual_pose(
            ActorPose((0.51523, -0.01446, 0.002), (0.0, 0.0, half, half))
        )

    def test_rotated_parent(self) -> None:
        half = math.sqrt(0.5)
        parent_angle = math.pi / 12.0
        self._assert_visual_pose(
            ActorPose((0.51523, -0.01446, 0.002), (0.0, 0.0, half, half)),
            parent_position=(0.1, 0.2, 0.3),
            parent_quat_xyzw=(
                0.0,
                0.0,
                math.sin(parent_angle),
                math.cos(parent_angle),
            ),
        )


if __name__ == "__main__":
    unittest.main()
