"""Pure-spec tests: no isaaclab, pxr, pyuipc, or sim runtime."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any, cast

from openvitac_spec import (
    FRANKA_PANDA_GSMINI,
    FRANKA_ROBOTIQ,
    Actor,
    CameraSpec,
    EmbodimentSpec,
    GsMiniSensor,
    Joint,
    LightSpec,
    MountSpec,
    RobotAssetRef,
    SceneSpec,
    SensorBinding,
    XenseWSSensor,
    gsmini_pair,
    neote_pair,
    xense_pair,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SPEC_SRC = _REPO_ROOT / "packages" / "spec" / "src"

_BOUNDARY_PROBE = """
import json
import sys

import openvitac_spec  # noqa: F401

banned = (
    "isaaclab",
    "pxr",
    "pyuipc",
    "openvitac_sim",
    "openvitac_tactile",
)
print(json.dumps([name for name in sys.modules if name.startswith(banned)]))
"""


def _camera(role: str = "head", prim_path: str = "/World/Camera") -> CameraSpec:
    return CameraSpec(prim_path, role, 32, 24)


def _scene(
    *,
    sensors: tuple[SensorBinding, ...] = (),
    cameras: tuple[CameraSpec, ...] | None = None,
    actors: tuple[Actor, ...] = (),
    joints: tuple[Joint, ...] = (),
    joint_pos: dict[str, float] | None = None,
) -> SceneSpec:
    return SceneSpec(
        embodiment=FRANKA_PANDA_GSMINI,
        sensors=sensors,
        cameras=(_camera(),) if cameras is None else cameras,
        actors=actors,
        light=LightSpec(),
        joints=joints,
        joint_pos=joint_pos,
    )


def _front_binding(
    streams: tuple[str, ...] | None = None,
) -> SensorBinding:
    extra = {} if streams is None else {"streams": cast(Any, streams)}
    return SensorBinding(
        pad_key="front",
        sensor=GsMiniSensor(),
        body_name="case_front",
        body_prim_path="/World/Robot/case_front",
        case_prim_path="/World/Robot/case_front",
        gelpad_prim_path="/World/Robot/gelpad_front",
        mount=MountSpec(mount_id="front"),
        **extra,
    )


class SpecConstructionTests(unittest.TestCase):
    def test_embodiment_presets_are_monolithic(self) -> None:
        for embodiment in (FRANKA_PANDA_GSMINI, FRANKA_ROBOTIQ):
            self.assertEqual(embodiment.asset.assembly, "monolithic")
            self.assertEqual(len(embodiment.gripper.contact_bodies), 2)
            self.assertTrue(embodiment.arm.arm_joints)
            self.assertEqual(embodiment.arm.ee_link.startswith("panda_"), True)
            self.assertEqual(embodiment.gripper.capabilities, frozenset(
                embodiment.gripper.capabilities
            ))

    def test_arm_plus_tool_fails_loud_but_preserves_fields(self) -> None:
        ref = RobotAssetRef(
            usd="embodiment/arm.usda",
            assembly="arm_plus_tool",
            tool_usd="embodiment/tool.usda",
            attach_prim="/World/Robot/Tool",
        )
        embodiment = EmbodimentSpec(
            name="split",
            arm=FRANKA_PANDA_GSMINI.arm,
            gripper=FRANKA_PANDA_GSMINI.gripper,
            asset=ref,
        )
        with self.assertRaises(NotImplementedError):
            embodiment.validate_supported()
        with self.assertRaises(ValueError):
            RobotAssetRef(usd="embodiment/arm.usda", assembly="arm_plus_tool")

    def test_sensor_binding_has_no_robot_selector(self) -> None:
        pairs = gsmini_pair()
        self.assertEqual([binding.pad_key for binding in pairs], ["left", "right"])
        self.assertFalse(hasattr(pairs[0].sensor, "gripper_kind"))
        self.assertTrue(
            all(isinstance(binding.sensor, GsMiniSensor) for binding in pairs)
        )

    def test_gsmini_marker_lattice_is_seven_rows_by_nine_columns(self) -> None:
        sensor = GsMiniSensor()
        self.assertEqual(sensor.marker_rows, 7)
        self.assertEqual(sensor.marker_cols, 9)
        xense = XenseWSSensor()
        self.assertEqual(xense.marker_rows, 20)
        self.assertEqual(xense.marker_cols, 11)

    def test_scene_rejects_duplicate_bindings(self) -> None:
        binding = gsmini_pair()[0]
        with self.assertRaises(ValueError):
            SceneSpec(
                embodiment=FRANKA_PANDA_GSMINI,
                sensors=(binding, binding),
                cameras=(
                    CameraSpec("/World/Camera", "head", 32, 24),
                ),
                actors=(),
                light=LightSpec(),
            )

    def test_pad_key_round_trip_through_custom_binding(self) -> None:
        binding = _front_binding()
        self.assertEqual(binding.pad_key, "front")
        self.assertEqual(binding.mount.joint_offsets, {})
        self.assertEqual(
            [b.pad_key for b in xense_pair()], ["left", "right"]
        )

    def test_sensor_binding_and_helpers_default_to_all_streams(self) -> None:
        expected = ("rgb", "depth", "marker_rgb")
        self.assertEqual(_front_binding().streams, expected)
        for pairs in (gsmini_pair(), neote_pair(), xense_pair()):
            self.assertEqual(pairs[0].streams, expected)
            self.assertEqual(pairs[1].streams, expected)

    def test_sensor_binding_rejects_unknown_stream_kind(self) -> None:
        with self.assertRaisesRegex(ValueError, "streams"):
            _front_binding(streams=("rgba",))

    def test_pair_helpers_reject_unknown_stream_kind(self) -> None:
        for helper in (gsmini_pair, neote_pair, xense_pair):
            with self.assertRaisesRegex(ValueError, "streams"):
                helper(streams=cast(Any, ("rgba",)))

    def test_sensor_binding_still_rejects_empty_streams(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one stream"):
            _front_binding(streams=())

    def test_actor_validation_still_holds(self) -> None:
        with self.assertRaises(ValueError):
            Actor(
                "bad.usd",
                "/World/bad",
                dynamic=False,
                authority="uipc",
                contact_object=True,
            )

    def test_static_actor_rejects_uipc_authority(self) -> None:
        with self.assertRaisesRegex(ValueError, "authority"):
            Actor("fixture.usd", "/World/fixture", dynamic=False, authority="uipc")

    def test_dynamic_actor_rejects_uipc_fixed_bodies(self) -> None:
        with self.assertRaisesRegex(ValueError, "uipc_fixed_boxes"):
            Actor(
                "fixture.usd",
                "/World/fixture",
                uipc_fixed_boxes=(((0.0, 0.0, 0.0), (0.1, 0.1, 0.1)),),
            )
        with self.assertRaisesRegex(ValueError, "uipc_fixed_mesh"):
            Actor("fixture.usd", "/World/fixture", uipc_fixed_mesh="slot_tet.npz")

    def test_static_fixture_may_carry_uipc_fixed_bodies(self) -> None:
        boxes = Actor(
            "fixture.usd",
            "/World/fixture",
            dynamic=False,
            uipc_fixed_boxes=(((0.0, 0.0, 0.0), (0.1, 0.1, 0.1)),),
        )
        self.assertEqual(len(boxes.uipc_fixed_boxes), 1)
        mesh = Actor(
            "fixture.usd",
            "/World/fixture",
            dynamic=False,
            uipc_fixed_mesh="slot_tet.npz",
        )
        self.assertEqual(mesh.uipc_fixed_mesh, "slot_tet.npz")

    def test_actor_density_is_optional_and_positive(self) -> None:
        self.assertIsNone(Actor("block.usd", "/World/block").density)
        actor = Actor("block.usd", "/World/block", density=1.0e2)
        self.assertEqual(actor.density, 1.0e2)
        with self.assertRaises(ValueError):
            Actor("block.usd", "/World/block", density=0.0)
        with self.assertRaises(ValueError):
            Actor("block.usd", "/World/block", density=-1.0)

    def test_dynamic_uipc_actor_accepts_fem_material(self) -> None:
        actor = Actor(
            "soft.usd",
            "/World/actors/soft",
            authority="uipc",
            tactile_mesh="soft_tet.npz",
            constitution="fem",
            youngs_mpa=50.0,
            poisson=0.45,
        )
        self.assertEqual(actor.constitution, "fem")
        self.assertEqual(actor.youngs_mpa, 50.0)
        self.assertEqual(actor.poisson, 0.45)

    def test_fem_requires_youngs_and_poisson(self) -> None:
        kwargs: dict[str, Any] = {
            "authority": "uipc",
            "tactile_mesh": "soft_tet.npz",
            "constitution": "fem",
        }
        with self.assertRaisesRegex(ValueError, "youngs_mpa"):
            Actor("soft.usd", "/World/actors/soft", poisson=0.45, **kwargs)
        with self.assertRaisesRegex(ValueError, "poisson"):
            Actor("soft.usd", "/World/actors/soft", youngs_mpa=10.0, **kwargs)

    def test_fem_material_ranges_match_object_cfg(self) -> None:
        kwargs: dict[str, Any] = {
            "authority": "uipc",
            "tactile_mesh": "soft_tet.npz",
            "constitution": "fem",
            "poisson": 0.45,
        }
        with self.assertRaisesRegex(ValueError, "youngs_mpa"):
            Actor("soft.usd", "/World/actors/soft", youngs_mpa=0.0, **kwargs)
        with self.assertRaisesRegex(ValueError, "youngs_mpa"):
            Actor("soft.usd", "/World/actors/soft", youngs_mpa=-1.0, **kwargs)
        kwargs["youngs_mpa"] = 10.0
        for poisson in (-0.1, 0.5):
            with self.assertRaisesRegex(ValueError, r"poisson.*\[0, 0.5\)"):
                Actor(
                    "soft.usd",
                    "/World/actors/soft",
                    poisson=poisson,
                    **{k: v for k, v in kwargs.items() if k != "poisson"},
                )

    def test_abd_rejects_fem_only_material(self) -> None:
        for kwargs in ({"youngs_mpa": 10.0}, {"poisson": 0.45}):
            kwargs: dict[str, Any] = cast(dict[str, Any], kwargs)
            with self.assertRaisesRegex(ValueError, "constitution='fem'"):
                Actor(
                    "block.usd",
                    "/World/block",
                    authority="uipc",
                    tactile_mesh="block_tet.npz",
                    **kwargs,
                )

    def test_fem_requires_dynamic_uipc_actor(self) -> None:
        kwargs: dict[str, Any] = {
            "constitution": "fem",
            "youngs_mpa": 10.0,
            "poisson": 0.45,
        }
        with self.assertRaisesRegex(ValueError, "dynamic UIPC"):
            Actor("block.usd", "/World/block", **kwargs)  # PhysX by default
        with self.assertRaisesRegex(ValueError, "dynamic UIPC"):
            Actor(
                "fixture.usd",
                "/World/fixture",
                dynamic=False,
                authority="physx",
                **kwargs,
            )

    def test_unknown_actor_constitution_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "constitution"):
            Actor(
                "block.usd",
                "/World/block",
                constitution=cast(Any, "snh"),
            )


class SceneSpecIdentityTests(unittest.TestCase):
    def test_duplicate_actor_prim_paths_are_rejected(self) -> None:
        actor = Actor("block.usd", "/World/block")
        with self.assertRaisesRegex(ValueError, "actors prim_path"):
            _scene(actors=(actor, actor))

    def test_duplicate_camera_prim_paths_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cameras prim_path"):
            _scene(
                cameras=(
                    _camera("head", "/World/Camera"),
                    _camera("wrist", "/World/Camera"),
                )
            )

    def test_joint_parent_must_be_an_actor(self) -> None:
        with self.assertRaisesRegex(ValueError, "joints parent"):
            _scene(
                actors=(Actor("gear.usd", "/World/actors/gear"),),
                joints=(
                    Joint(
                        "/World/joints/gear",
                        "/World/actors/missing",
                        "/World/actors/gear",
                    ),
                ),
            )

    def test_joint_child_must_be_an_actor(self) -> None:
        with self.assertRaisesRegex(ValueError, "joints child"):
            _scene(
                actors=(Actor("base.usd", "/World/actors/base", dynamic=False),),
                joints=(
                    Joint(
                        "/World/joints/gear",
                        "/World/actors/base",
                        "/World/actors/missing",
                    ),
                ),
            )

    def test_joint_without_parent_is_accepted(self) -> None:
        scene = _scene(
            actors=(Actor("gear.usd", "/World/actors/gear"),),
            joints=(
                Joint("/World/joints/gear", None, "/World/actors/gear"),
            ),
        )
        self.assertEqual(len(scene.joints), 1)

    def test_joint_between_actors_is_accepted(self) -> None:
        scene = _scene(
            actors=(
                Actor("base.usd", "/World/actors/base", dynamic=False),
                Actor("gear.usd", "/World/actors/gear"),
            ),
            joints=(
                Joint(
                    "/World/joints/gear",
                    "/World/actors/base",
                    "/World/actors/gear",
                ),
            ),
        )
        self.assertEqual(len(scene.joints), 1)

    def test_joint_pos_keys_must_match_a_declared_joint(self) -> None:
        with self.assertRaisesRegex(ValueError, "joint_pos key"):
            _scene(joint_pos={"panda_jiont1": 0.0})

    def test_joint_pos_accepts_joint_regex_patterns(self) -> None:
        joint_pos = {"panda_joint[1-7]": 0.0, "panda_finger.*": 0.02}
        self.assertEqual(_scene(joint_pos=joint_pos).joint_pos, joint_pos)

    def test_joint_pos_invalid_regex_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a valid regex"):
            _scene(joint_pos={"panda_joint[": 0.0})


class ImportBoundaryTests(unittest.TestCase):
    def test_importing_spec_does_not_load_runtime_packages(self) -> None:
        # Probe a fresh interpreter so the result cannot depend on modules
        # imported earlier by other tests in the same pytest process.
        env = dict(os.environ)
        pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(_SPEC_SRC)
            if not pythonpath
            else os.pathsep.join((str(_SPEC_SRC), pythonpath))
        )
        result = subprocess.run(
            [sys.executable, "-c", _BOUNDARY_PROBE],
            cwd=_REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        loaded = json.loads(result.stdout)
        self.assertEqual(loaded, [])


if __name__ == "__main__":
    unittest.main()
