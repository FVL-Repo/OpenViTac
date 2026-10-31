"""Unit tests for the kinematic PhysX proxy used by UIPC-authority actors.

These construct cfgs and an in-memory USD stage only; no Isaac app is launched.
"""
from __future__ import annotations

import unittest
from typing import Any, cast

import isaaclab.sim as sim_utils
import numpy as np
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg
from pxr import Usd, UsdPhysics

from openvitac_sim.spawn import (
    disable_named_collisions,
    make_proxy_kinematic,
    rigid_object_cfg,
    sync_kinematic_proxy,
)


class ProxyCfgTests(unittest.TestCase):
    def test_proxy_body_is_gravity_free(self) -> None:
        cfg = rigid_object_cfg(
            "usb.usd", "/World/actors/usb", (0.4, 0.0, 0.0075), kinematic=True
        )
        spawn = cast(Any, cfg.spawn)
        self.assertIsInstance(spawn.rigid_props, PhysxRigidBodyCfg)
        self.assertTrue(spawn.rigid_props.disable_gravity)
        self.assertEqual(cfg.init_state.pos, (0.4, 0.0, 0.0075))

    def test_dynamic_body_cfg_keeps_gravity(self) -> None:
        cfg = rigid_object_cfg("usb.usd", "/World/actors/usb")
        spawn = cast(Any, cfg.spawn)
        self.assertIsInstance(spawn.rigid_props, PhysxRigidBodyCfg)
        self.assertFalse(spawn.rigid_props.disable_gravity)

    def test_dynamic_body_cfg_applies_density(self) -> None:
        cfg = rigid_object_cfg(
            "block.usd", "/World/actors/block", density=1.0e2
        )
        spawn = cast(Any, cfg.spawn)
        self.assertIsInstance(spawn.mass_props, sim_utils.MassCfg)
        self.assertEqual(spawn.mass_props.density, 1.0e2)
        self.assertTrue(spawn.mass_props_create_if_missing)


class ProxyUsdTests(unittest.TestCase):
    def test_proxy_is_kinematic_and_collision_free(self) -> None:
        stage = Usd.Stage.CreateInMemory()
        root = stage.DefinePrim("/World/usb")
        child = stage.DefinePrim("/World/usb/geom")
        UsdPhysics.CollisionAPI.Apply(root)
        UsdPhysics.CollisionAPI.Apply(child)
        make_proxy_kinematic("/World/usb", stage=stage)
        self.assertTrue(
            UsdPhysics.RigidBodyAPI(root).GetKinematicEnabledAttr().Get()
        )
        self.assertFalse(
            UsdPhysics.CollisionAPI(root).GetCollisionEnabledAttr().Get()
        )
        self.assertFalse(
            UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get()
        )


class ProxyCollisionTests(unittest.TestCase):
    def test_named_pad_proxy_collision_is_disabled(self) -> None:
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim("/World/Robot")
        pad = stage.DefinePrim("/World/Robot/case/pad_proxy")
        UsdPhysics.CollisionAPI.Apply(pad)
        other = stage.DefinePrim("/World/Robot/case/other")
        UsdPhysics.CollisionAPI.Apply(other).CreateCollisionEnabledAttr().Set(True)
        self.assertEqual(
            disable_named_collisions("/World/Robot", "pad_proxy", stage=stage), 1
        )
        self.assertFalse(
            UsdPhysics.CollisionAPI(pad).GetCollisionEnabledAttr().Get()
        )
        self.assertTrue(
            UsdPhysics.CollisionAPI(other).GetCollisionEnabledAttr().Get()
        )


class _FakeProxy:
    def __init__(self) -> None:
        self.pose: Any = None
        self.velocity: Any = None

    def write_root_pose_to_sim(self, pose: Any) -> None:
        self.pose = pose

    def write_root_velocity_to_sim(self, velocity: Any) -> None:
        self.velocity = velocity


class ProxySyncTests(unittest.TestCase):
    def test_pose_and_zero_velocity_are_written(self) -> None:
        proxy = _FakeProxy()
        pose = np.array([0.4, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        sync_kinematic_proxy(cast(Any, proxy), pose, device="cpu")
        self.assertEqual(proxy.pose.shape, (1, 7))
        np.testing.assert_allclose(proxy.pose[0].cpu().numpy(), pose)
        self.assertEqual(proxy.velocity.shape, (1, 6))
        np.testing.assert_allclose(proxy.velocity.cpu().numpy(), 0.0)

    def test_wrong_pose_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sync_kinematic_proxy(
                cast(Any, _FakeProxy()), np.zeros(6, dtype=np.float32), device="cpu"
            )
