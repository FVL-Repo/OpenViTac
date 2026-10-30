"""Tactile bridge unit tests; the real UIPC worker is an opt-in integration test.

The default suite never needs Isaac. ``test_real_worker_startup`` requires the
Isaac/Kit runtime libraries plus pyuipc, so it skips unless
``OPENVITAC_RUN_UIPC_INTEGRATION=1`` is set inside that environment.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest import mock

import numpy as np

from openvitac_sim.scene import SceneObservation
from openvitac_sim.sensors import AttachmentBody, GsMiniCfg, XenseWSCfg
from openvitac_sim.spec import Actor
from openvitac_sim.tactile_bridge import (
    TactileBridge,
    body_pose_to_T,
    finger_sample_from_body_poses,
    objects_from_actors,
    tactile_cfg_from_sensor,
)
from openvitac_tactile.client import TactileRuntime, TactileUnavailable
from openvitac_tactile.config import ObjectCfg, PadFrameCfg, WorkerCfg
from openvitac_tactile.protocol import (
    MAX_OBJECTS,
    MAX_PAD_VERTS,
    MAX_PADS,
    FingerSample,
    TactileState,
)
from openvitac_tactile.softworld import ObjectHandle, PadHandle

# Real UIPC needs Isaac/Kit's runtime libraries on the loader path; the bare
# pyuipc wheel cannot resolve its native symbols outside Kit.
RUN_UIPC_INTEGRATION = os.environ.get("OPENVITAC_RUN_UIPC_INTEGRATION") == "1"
UIPC_INTEGRATION_SKIP = (
    "UIPC integration test: requires the Isaac/Kit runtime plus pyuipc; "
    "set OPENVITAC_RUN_UIPC_INTEGRATION=1 to run it there"
)


def _sensor(variant: str = "gsmini") -> GsMiniCfg | XenseWSCfg:
    if variant == "xense":
        return XenseWSCfg(
            name="tactile",
            prim_path="/World/Robot/XenseWS_left",
            gelpad_prim_path="/World/Robot/XenseWS_gelpad_left",
            attachment_body=AttachmentBody("XenseWS_left", "/World/Robot/XenseWS_left"),
            gripper_kind="franka_panda",
        )
    return GsMiniCfg(
        name="tactile",
        prim_path="/World/Robot/gelsight_mini_case_left",
        gelpad_prim_path="/World/Robot/gelpad_left",
        attachment_body=AttachmentBody(
            "gelsight_mini_case_left", "/World/Robot/gelsight_mini_case_left"
        ),
        gripper_kind="franka_panda",
    )


class SensorMappingTests(unittest.TestCase):
    def test_gsmini_mapping_derives_right_pad(self) -> None:
        cfg = tactile_cfg_from_sensor(_sensor())
        self.assertEqual(len(cfg.pads), 2)
        self.assertEqual([pad.side for pad in cfg.pads], ["left", "right"])
        left, right = cfg.pads
        self.assertEqual(left.gelpad_prim_path, "/World/Robot/gelpad_left")
        self.assertEqual(right.gelpad_prim_path, "/World/Robot/gelpad_right")
        self.assertEqual(left.frame.body_name, "gelsight_mini_case_left")
        self.assertEqual(right.frame.body_name, "gelsight_mini_case_right")
        self.assertEqual(
            right.frame.body_prim_path, "/World/Robot/gelsight_mini_case_right"
        )
        self.assertEqual(left.material.youngs_mpa, 0.1)
        self.assertEqual(right.material.youngs_mpa, 0.1)
        self.assertEqual(left.material.constraint_strength, 1e4)
        self.assertEqual(right.material.constraint_strength, 1e4)
        self.assertEqual((left.image_width, left.image_height), (320, 240))
        self.assertEqual((right.image_width, right.image_height), (320, 240))
        self.assertEqual(len({pad.side for pad in cfg.pads}), 2)

    def test_xense_mapping_derives_right_pad(self) -> None:
        cfg = tactile_cfg_from_sensor(_sensor("xense"))
        self.assertEqual(len(cfg.pads), 2)
        left, right = cfg.pads
        self.assertEqual(left.gelpad_prim_path, "/World/Robot/XenseWS_gelpad_left")
        self.assertEqual(right.gelpad_prim_path, "/World/Robot/XenseWS_gelpad_right")
        self.assertEqual(left.frame.body_name, "XenseWS_left")
        self.assertEqual(right.frame.body_name, "XenseWS_right")
        self.assertEqual(left.material.youngs_mpa, 15.0)
        self.assertEqual(right.material.constraint_strength, 2e5)
        self.assertEqual(left.frame.attach_radius_m, 0.0008)
        self.assertEqual(right.frame.attach_radius_m, 0.0008)
        self.assertEqual(cfg.reset.settle_max_steps, 180)
        self.assertEqual(len({pad.side for pad in cfg.pads}), 2)

    def test_right_declaration_is_not_duplicated(self) -> None:
        sensor = replace(
            _sensor(),
            prim_path="/World/Robot/gelsight_mini_case_right",
            gelpad_prim_path="/World/Robot/gelpad_right",
            attachment_body=AttachmentBody(
                "gelsight_mini_case_right",
                "/World/Robot/gelsight_mini_case_right",
            ),
        )
        cfg = tactile_cfg_from_sensor(sensor)
        self.assertEqual([pad.side for pad in cfg.pads], ["left", "right"])
        self.assertEqual(cfg.pads[0].gelpad_prim_path, "/World/Robot/gelpad_left")
        self.assertEqual(cfg.pads[1].gelpad_prim_path, "/World/Robot/gelpad_right")

    def test_actor_mapping(self) -> None:
        objs = objects_from_actors(
            (
                Actor("usb.usd", "/World/actors/usb", dynamic=True, authority="physx"),
                Actor("plate.usda", "/World/plate", dynamic=False, authority="physx"),
                Actor(
                    "soft.usd",
                    "/World/actors/soft",
                    dynamic=True,
                    authority="uipc",
                    tactile_mesh="soft_tet.npz",
                ),
            )
        )
        self.assertEqual([o.name for o in objs], ["usb", "soft"])
        self.assertEqual(objs[0].authority, "physx")
        self.assertIsNone(objs[0].mesh_path)
        self.assertEqual(objs[0].usd_path, "usb.usd")
        self.assertEqual(objs[1].authority, "uipc")
        self.assertTrue(str(objs[1].mesh_path).endswith("soft_tet.npz"))
        self.assertEqual(objs[1].usd_path, "soft.usd")
        self.assertNotEqual(objs[1].mesh_path, objs[1].usd_path)

    def test_uipc_actor_requires_a_tactile_mesh(self) -> None:
        with self.assertRaises(ValueError):
            objects_from_actors(
                (
                    Actor(
                        "soft.usd",
                        "/World/actors/soft",
                        dynamic=True,
                        authority="uipc",
                    ),
                )
            )

    def test_uipc_npz_actor_is_its_own_tet_mesh(self) -> None:
        (obj,) = objects_from_actors(
            (Actor("soft_tet.npz", "/World/actors/soft", authority="uipc"),)
        )
        self.assertTrue(str(obj.mesh_path).endswith("soft_tet.npz"))
        self.assertEqual(obj.usd_path, "soft_tet.npz")

    def test_uipc_fixed_actor_keeps_its_usd(self) -> None:
        (obj,) = objects_from_actors(
            (
                Actor(
                    "slot.usd",
                    "/World/actors/slot",
                    dynamic=False,
                    uipc_fixed_mesh="slot_tet.npz",
                    authority="physx",
                ),
            )
        )
        self.assertEqual(obj.usd_path, "slot.usd")
        self.assertTrue(str(obj.mesh_path).endswith("slot_tet.npz"))
        self.assertNotEqual(obj.mesh_path, obj.usd_path)

    def test_actor_without_usd_has_no_usd_path(self) -> None:
        (obj,) = objects_from_actors(
            (
                Actor(
                    "",
                    "/World/actors/soft",
                    authority="uipc",
                    tactile_mesh="soft_tet.npz",
                ),
            )
        )
        self.assertIsNone(obj.usd_path)

    def test_actor_authority_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            Actor(
                "usb.usd",
                "/World/actors/usb",
                authority=cast(Any, "bad"),
            )

    def test_fem_actor_forwards_constitution_and_material(self) -> None:
        (obj,) = objects_from_actors(
            (
                Actor(
                    "soft.usd",
                    "/World/actors/soft",
                    authority="uipc",
                    tactile_mesh="soft_tet.npz",
                    constitution="fem",
                    youngs_mpa=50.0,
                    poisson=0.45,
                ),
            )
        )
        self.assertEqual(obj.constitution, "fem")
        self.assertEqual(obj.youngs_mpa, 50.0)
        self.assertEqual(obj.poisson, 0.45)
        self.assertEqual(obj.authority, "uipc")
        self.assertTrue(str(obj.mesh_path).endswith("soft_tet.npz"))

    def test_abd_actor_keeps_default_material(self) -> None:
        physx, uipc = objects_from_actors(
            (
                Actor("usb.usd", "/World/actors/usb", authority="physx"),
                Actor(
                    "soft.usd",
                    "/World/actors/soft",
                    authority="uipc",
                    tactile_mesh="soft_tet.npz",
                ),
            )
        )
        for obj in (physx, uipc):
            self.assertEqual(obj.constitution, "abd")
            self.assertIsNone(obj.youngs_mpa)
            self.assertIsNone(obj.poisson)

    def test_weight_classify_density_reaches_object_cfg(self) -> None:
        from openvitac_tasks.tasks.weight_classify import WeightClassify

        objects = {
            obj.name: obj for obj in objects_from_actors(WeightClassify.actors)
        }
        self.assertEqual(objects["light_block"].density_kg_m3, 100.0)
        self.assertEqual(objects["heavy_block"].density_kg_m3, 30000.0)

    def test_unset_density_keeps_the_default(self) -> None:
        actor = Actor(
            "block.usd",
            "/World/actors/block",
            authority="uipc",
            tactile_mesh="block_tet.npz",
        )
        (obj,) = objects_from_actors((actor,))
        self.assertEqual(obj.density_kg_m3, 1.0e3)


class _CaptureRuntime:
    def __init__(
        self,
        ack: dict[str, Any] | None = None,
        state: TactileState | None = None,
    ) -> None:
        self.requests: list[dict[str, Any]] = []
        self.ack = ack if ack is not None else {"status": "ready"}
        self.state = state

    def reset(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        return self.ack

    def exchange(
        self, sample: FingerSample, *, wait: bool = False
    ) -> TactileState | None:
        return self.state


class BridgeTests(unittest.TestCase):
    def test_observation_default_is_none(self) -> None:
        obs = SceneObservation(
            rgb={},
            joint_pos=np.zeros(1),
            joint_vel=np.zeros(1),
            action=np.zeros(1),
        )
        self.assertIsNone(obs.tactile)

    def test_fake_worker_lifecycle_and_idempotent_close(self) -> None:
        cfg = tactile_cfg_from_sensor(_sensor())
        bridge = TactileBridge(cfg, run_id="sim_bridge_test", check_uipc=False)
        bridge.start()
        frames = {pad.name: np.eye(4) for pad in cfg.pads}
        try:
            ack = bridge.reset(pad_frames=frames)
            self.assertEqual(ack["status"], "ready")
            sample = finger_sample_from_body_poses(
                np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64),
                np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64),
                seq=1,
                physx_step=1,
            )
            self.assertIsNone(bridge.exchange(sample))
        finally:
            bridge.close()
            bridge.close()
        self.assertIsNone(bridge.runtime)

    def test_failed_start_leaves_bridge_retryable(self) -> None:
        cfg = tactile_cfg_from_sensor(_sensor())
        bridge = TactileBridge(cfg, run_id="sim_retry_test", check_uipc=False)
        with (
            mock.patch.object(
                TactileRuntime, "start", side_effect=TactileUnavailable("boom")
            ),
            self.assertRaises(TactileUnavailable),
        ):
            bridge.start()
        self.assertIsNone(bridge._runtime)
        bridge.start()
        try:
            self.assertIsNotNone(bridge._runtime)
            self.assertTrue(cast(Any, bridge.runtime).health.alive)
        finally:
            bridge.close()

    @unittest.skipUnless(RUN_UIPC_INTEGRATION, UIPC_INTEGRATION_SKIP)
    def test_real_worker_startup(self) -> None:
        cfg = tactile_cfg_from_sensor(_sensor())
        bridge = TactileBridge(cfg, run_id="sim_real_worker", check_uipc=True)
        bridge.start()
        left = np.eye(4, dtype=np.float64)
        left[:3, 3] = (0.4, -0.06, 0.20)
        right = np.eye(4, dtype=np.float64)
        right[:3, 3] = (0.4, 0.06, 0.20)
        frames = {cfg.pads[0].name: left, cfg.pads[1].name: right}
        try:
            self.assertEqual(bridge.reset(pad_frames=frames)["status"], "ready")
        finally:
            bridge.close()

    def test_reset_carries_attachment_and_object_frames(self) -> None:
        capture = _CaptureRuntime()
        bridge = TactileBridge(
            tactile_cfg_from_sensor(
                _sensor(),
                objects=(ObjectCfg("usb", "/World/actors/usb"),),
            ),
            run_id="capture",
            runtime=cast(TactileRuntime, capture),
        )
        bridge.start()
        left = np.eye(4, dtype=np.float64)
        left[:3, 3] = (0.10, -0.02, 0.30)
        right = np.eye(4, dtype=np.float64)
        right[:3, 3] = (0.10, 0.02, 0.30)
        obj = np.eye(4, dtype=np.float64)
        obj[:3, 3] = (0.12, 0.0, 0.035)
        left_pad = np.eye(4, dtype=np.float64)
        left_pad[:3, 3] = (0.093132, -0.005240, 0.807600)
        right_pad = np.eye(4, dtype=np.float64)
        right_pad[:3, 3] = (0.093132, 0.005240, 0.807600)
        ack = bridge.reset(
            pad_frames={"tactile_left": left, "tactile_right": right},
            pad_body_frames={"tactile_left": left_pad, "tactile_right": right_pad},
            object_poses={"usb": obj},
            seed=17,
        )
        self.assertEqual(ack["status"], "ready")
        self.assertEqual(len(capture.requests), 1)
        request = capture.requests[0]
        self.assertEqual(request["seed"], 17)
        pads = cast(list[dict[str, Any]], request["pads"])
        self.assertEqual([p["side"] for p in pads], ["left", "right"])
        left_request = np.asarray(pads[0]["T_world_body0"]).reshape(4, 4)
        right_request = np.asarray(pads[1]["T_world_body0"]).reshape(4, 4)
        left_local = np.asarray(pads[0]["T_body_pad0"]).reshape(4, 4)
        right_local = np.asarray(pads[1]["T_body_pad0"]).reshape(4, 4)
        np.testing.assert_allclose(left_request, left)
        np.testing.assert_allclose(right_request, right)
        np.testing.assert_allclose(left_local, left_pad)
        np.testing.assert_allclose(right_local, right_pad)
        self.assertFalse(np.allclose(left_request, np.eye(4)))
        self.assertFalse(np.allclose(right_request, np.eye(4)))
        self.assertFalse(np.allclose(left_request, right_request))
        np.testing.assert_allclose(
            bridge.pad_world_frames["tactile_left"], left @ left_pad
        )
        np.testing.assert_allclose(
            bridge.pad_world_frames["tactile_right"], right @ right_pad
        )
        objects = cast(list[dict[str, Any]], request["objects"])
        self.assertEqual(objects[0]["name"], "usb")
        np.testing.assert_allclose(
            np.asarray(objects[0]["T_world0"]).reshape(4, 4), obj
        )

    def test_reset_overrides_exactly_one_contact_object(self) -> None:
        capture = _CaptureRuntime()
        bridge = TactileBridge(
            tactile_cfg_from_sensor(
                _sensor(),
                objects=(
                    ObjectCfg("usb", "/World/actors/usb"),
                    ObjectCfg("slot", "/World/actors/slot"),
                ),
            ),
            run_id="contact_object",
            runtime=cast(TactileRuntime, capture),
        )
        bridge.start()
        frames = {pad.name: np.eye(4) for pad in bridge.cfg.pads}
        poses = {"usb": np.eye(4), "slot": np.eye(4)}
        bridge.reset(
            pad_frames=frames,
            object_poses=poses,
            contact_object="/World/actors/slot",
        )
        request = capture.requests[0]
        config = cast(dict[str, Any], request["config"])
        objects = cast(list[dict[str, Any]], config["objects"])
        self.assertEqual(
            {obj["name"]: obj["contact_object"] for obj in objects},
            {"usb": False, "slot": True},
        )
        self.assertEqual(
            [obj.contact_object for obj in bridge.cfg.objects], [False, False]
        )
        self.assertEqual(
            [obj.contact_object for obj in bridge.episode_cfg.objects],
            [False, True],
        )
        with self.assertRaises(ValueError):
            bridge.reset(
                pad_frames=frames,
                object_poses=poses,
                contact_object="/World/actors/missing",
            )
        self.assertEqual(len(capture.requests), 1)

    def test_reset_rejects_a_negative_seed(self) -> None:
        capture = _CaptureRuntime()
        cfg = tactile_cfg_from_sensor(_sensor())
        bridge = TactileBridge(
            cfg, run_id="negative_seed", runtime=cast(TactileRuntime, capture)
        )
        frames = {pad.name: np.eye(4) for pad in cfg.pads}
        with self.assertRaises(ValueError):
            bridge.reset(pad_frames=frames, seed=-1)
        self.assertEqual(capture.requests, [])

    def test_reset_parses_pad_rest_surfaces(self) -> None:
        vertices = np.array(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]]
        )
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        capture = _CaptureRuntime(
            {
                "status": "ready",
                "pads": [
                    {
                        "name": "tactile_left",
                        "side": "left",
                        "rest_vertices": vertices.tolist(),
                        "triangles": triangles.tolist(),
                    }
                ],
            }
        )
        cfg = tactile_cfg_from_sensor(_sensor())
        bridge = TactileBridge(
            cfg,
            run_id="capture_pads",
            runtime=cast(TactileRuntime, capture),
        )
        frames = {pad.name: np.eye(4) for pad in cfg.pads}
        bridge.reset(pad_frames=frames)
        handle = bridge.pad_handle("tactile_left")
        surfaces = bridge.pad_surfaces
        self.assertEqual(set(surfaces), {handle})
        np.testing.assert_allclose(surfaces[handle].rest_vertices, vertices)
        np.testing.assert_array_equal(surfaces[handle].triangles, triangles)
        self.assertEqual(
            bridge.pad_surfaces_by_name["tactile_left"].name, "tactile_left"
        )

    def test_handle_reads_after_exchange(self) -> None:
        object_T = np.eye(4, dtype=np.float64)
        object_T[:3, 3] = (0.40, 0.0, 0.035)
        state = TactileState(
            seq=7,
            finger_seq=3,
            physx_step=14,
            timestamp_us=123456,
            object_T=np.zeros((MAX_OBJECTS, 4, 4), dtype=np.float64),
            object_pose=np.zeros((MAX_OBJECTS, 7), dtype=np.float64),
            object_valid=np.zeros((MAX_OBJECTS,), dtype=np.bool_),
            n_objects=1,
            pad_vertices=np.zeros(
                (MAX_PADS, MAX_PAD_VERTS, 3), dtype=np.float32
            ),
            pad_n_vertices=np.array([3, 2, 0, 0], dtype=np.int32),
            pad_contact_gradient=np.zeros((MAX_PADS, 4), dtype=np.float64),
            pad_force=np.zeros((MAX_PADS, 3), dtype=np.float64),
            pad_force_kind=np.zeros((MAX_PADS,), dtype=np.uint8),
            pad_tracking_error_mm=np.zeros((MAX_PADS,), dtype=np.float64),
            pad_stale=np.zeros((MAX_PADS,), dtype=np.bool_),
            pad_valid=np.ones((MAX_PADS,), dtype=np.bool_),
        )
        state.pad_vertices[0, :3] = [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
        ]
        state.pad_vertices[1, :2] = [[0.1, 0.0, 0.0], [0.11, 0.0, 0.0]]
        state.object_T[0] = object_T
        state.object_pose[0] = [0.40, 0.0, 0.035, 0.0, 0.0, 0.0, 1.0]
        state.object_valid[0] = True
        cfg = tactile_cfg_from_sensor(
            _sensor(), objects=(ObjectCfg("usb", "/World/actors/usb"),)
        )
        capture = _CaptureRuntime(state=state)
        bridge = TactileBridge(
            cfg, run_id="capture_handles", runtime=cast(TactileRuntime, capture)
        )
        frames = {pad.name: np.eye(4) for pad in cfg.pads}
        bridge.reset(
            pad_frames=frames,
            object_poses={"usb": object_T},
        )
        sample = FingerSample(
            seq=3,
            physx_step=14,
            left_T=np.eye(4),
            right_T=np.eye(4),
            timestamp_ns=123456000,
        )
        bridge.exchange(sample)
        left_handle = bridge.pad_handle("tactile_left")
        right_handle = bridge.pad_handle("tactile_right")
        left = bridge.read_pad(left_handle)
        right = bridge.read_pad(right_handle)
        self.assertEqual(left.n_vertices, 3)
        self.assertEqual(right.n_vertices, 2)
        np.testing.assert_allclose(left.vertices, state.pad_vertices[0, :3])
        np.testing.assert_allclose(right.vertices, state.pad_vertices[1, :2])
        self.assertTrue(left.valid)
        usb = bridge.read_object(bridge.object_handle("usb"))
        np.testing.assert_allclose(usb.T, object_T)
        np.testing.assert_allclose(usb.pose_xyzw, state.object_pose[0])
        with self.assertRaises(ValueError):
            bridge.read_pad(
                PadHandle(uid=99, name="missing", side="left", index=0)
            )
        with self.assertRaises(ValueError):
            bridge.read_object(ObjectHandle(uid=99, name="missing", index=0))

    def test_observation_maps_pad_valid_independently_of_stale(self) -> None:
        state = TactileState(
            seq=7,
            finger_seq=3,
            physx_step=14,
            timestamp_us=123456,
            object_T=np.zeros((MAX_OBJECTS, 4, 4), dtype=np.float64),
            object_pose=np.zeros((MAX_OBJECTS, 7), dtype=np.float64),
            object_valid=np.zeros((MAX_OBJECTS,), dtype=np.bool_),
            n_objects=0,
            pad_vertices=np.zeros((MAX_PADS, MAX_PAD_VERTS, 3), dtype=np.float32),
            pad_n_vertices=np.array([0, 2, 0, 0], dtype=np.int32),
            pad_contact_gradient=np.zeros((MAX_PADS, 4), dtype=np.float64),
            pad_force=np.zeros((MAX_PADS, 3), dtype=np.float64),
            pad_force_kind=np.zeros((MAX_PADS,), dtype=np.uint8),
            pad_tracking_error_mm=np.zeros((MAX_PADS,), dtype=np.float64),
            # The right pad is stale yet geometrically valid; the left pad has
            # no vertices, so it is invalid even though it is not stale.
            pad_stale=np.array([False, True, False, False], dtype=np.bool_),
            pad_valid=np.array([False, True, False, False], dtype=np.bool_),
        )
        state.pad_vertices[1, :2] = [[0.1, 0.0, 0.0], [0.11, 0.0, 0.0]]
        cfg = tactile_cfg_from_sensor(_sensor())
        capture = _CaptureRuntime(state=state)
        bridge = TactileBridge(
            cfg, run_id="capture_valid", runtime=cast(TactileRuntime, capture)
        )
        sample = FingerSample(
            seq=3,
            physx_step=14,
            left_T=np.eye(4),
            right_T=np.eye(4),
            timestamp_ns=123456000,
        )
        observation = bridge.exchange(sample)
        assert observation is not None
        left = observation.pads["left"]
        right = observation.pads["right"]
        self.assertFalse(left.valid)
        self.assertEqual(left.n_vertices, 0)
        self.assertIsNone(left.vertices)
        self.assertFalse(left.stale)
        self.assertTrue(right.valid)
        self.assertEqual(right.n_vertices, 2)
        self.assertTrue(right.stale)

    def test_fake_worker_captures_calibrated_reset(self) -> None:
        tests_dir = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "reset.json"
            base_cfg = tactile_cfg_from_sensor(
                _sensor(),
                objects=(ObjectCfg("usb", "/World/actors/usb"),),
            )
            local = PadFrameCfg(
                body_name="panda_leftfinger",
                body_prim_path="/World/Robot/panda_leftfinger",
                pos=(0.01, 0.0, 0.02),
                rot_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
            cfg = replace(
                base_cfg,
                pads=tuple(replace(pad, frame=local) for pad in base_cfg.pads),
                worker=WorkerCfg(
                    python=sys.executable,
                    workspace=directory,
                    env={
                        "PYTHONPATH": str(tests_dir),
                        "FAKE_RESET_CAPTURE": str(capture),
                    },
                ),
            )
            runtime = TactileRuntime(
                cfg,
                run_id="fake_capture",
                check_uipc=False,
                worker_module="fake_worker",
            )
            bridge = TactileBridge(cfg, run_id="fake_capture", runtime=runtime)
            runtime.start()
            bridge.start()
            left = np.eye(4, dtype=np.float64)
            left[:3, 3] = (0.10, -0.02, 0.30)
            right = np.eye(4, dtype=np.float64)
            right[:3, 3] = (0.10, 0.02, 0.30)
            obj = np.eye(4, dtype=np.float64)
            obj[:3, 3] = (0.12, 0.0, 0.035)
            try:
                bridge.reset(
                    pad_frames={"tactile_left": left, "tactile_right": right},
                    object_poses={"usb": obj},
                )
            finally:
                bridge.close()
            local_T = body_pose_to_T(np.asarray([*local.pos, *local.rot_xyzw]))
            assert bridge.pad_world_frames
            np.testing.assert_allclose(
                bridge.pad_world_frames["tactile_left"], left @ local_T
            )
            np.testing.assert_allclose(
                bridge.pad_world_frames["tactile_right"], right @ local_T
            )
            request = json.loads(capture.read_text(encoding="utf-8"))
            pads = cast(list[dict[str, Any]], request["pads"])
            self.assertEqual([p["side"] for p in pads], ["left", "right"])
            np.testing.assert_allclose(
                np.asarray(pads[0]["T_world_body0"]).reshape(4, 4), left
            )
            np.testing.assert_allclose(
                np.asarray(pads[1]["T_world_body0"]).reshape(4, 4), right
            )
            objects = cast(list[dict[str, Any]], request["objects"])
            np.testing.assert_allclose(
                np.asarray(objects[0]["T_world0"]).reshape(4, 4), obj
            )

    def test_disabled_bridge_does_not_send_reset(self) -> None:
        capture = _CaptureRuntime()
        cfg = replace(tactile_cfg_from_sensor(_sensor()), enabled=False)
        bridge = TactileBridge(
            cfg,
            run_id="disabled_reset",
            runtime=cast(TactileRuntime, capture),
        )
        with self.assertRaises(RuntimeError):
            bridge.reset()
        self.assertEqual(capture.requests, [])

    def test_disabled_cfg_is_not_started(self) -> None:
        cfg = replace(tactile_cfg_from_sensor(_sensor()), enabled=False)
        bridge = TactileBridge(cfg, run_id="disabled", check_uipc=False)
        with self.assertRaises(RuntimeError):
            bridge.start()


if __name__ == "__main__":
    unittest.main()
