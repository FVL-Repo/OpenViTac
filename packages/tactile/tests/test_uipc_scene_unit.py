"""Unit tests for object constitution selection in ``UipcScene``.

The pyuipc modules are stubbed before importing ``uipc_scene`` so the FEM/ABD
branching can be checked without CUDA, the native runtime, or a solver step.
"""

from __future__ import annotations

import math
import sys
import types
import unittest
from typing import Any

import numpy as np

from openvitac_tactile.config import ObjectCfg, TactileCfg

_MPA = 1.0e6


class _Slot:
    def __init__(self, array: Any) -> None:
        self.array = array

    def view(self) -> Any:
        return self.array


class _Collection:
    def __init__(self, geometry: _Geom, kind: str) -> None:
        self._geometry = geometry
        self._kind = kind

    def find(self, name: str) -> _Slot | None:
        array = self._geometry.attrs.get((self._kind, name))
        return None if array is None else _Slot(array)

    def create(self, name: str, default: Any) -> _Slot:
        array = np.full(self._geometry.count, default, dtype=np.int64)
        self._geometry.attrs[self._kind, name] = array
        return _Slot(array)


class _Geom:
    def __init__(self, vertices: Any) -> None:
        self.vertices_array = np.ascontiguousarray(vertices, dtype=np.float64).reshape(
            -1, 3
        )
        self.attrs: dict[tuple[str, str], np.ndarray] = {
            ("instance", "transform"): np.eye(4, dtype=np.float64)[None].copy()
        }

    @property
    def count(self) -> int:
        return len(self.vertices_array)

    def vertices(self) -> _Collection:
        return _Collection(self, "vertex")

    def instances(self) -> _Collection:
        return _Collection(self, "instance")

    def transforms(self) -> _Slot:
        return _Slot(self.attrs["instance", "transform"])

    def positions(self) -> _Slot:
        return _Slot(self.vertices_array)


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def apply_to(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((*args, kwargs))


class _RecordingAbd(_Recorder):
    def apply_to(
        self, geometry: _Geom, kappa: float, mass_density: float = 1000.0
    ) -> None:
        self.calls.append((geometry, kappa, mass_density))
        geometry.attrs["instance", "is_fixed"] = np.zeros(1, dtype=np.int64)


class _RecordingSnh(_Recorder):
    def apply_to(
        self, geometry: _Geom, moduli: Any, mass_density: float = 1000.0
    ) -> None:
        self.calls.append((geometry, moduli, mass_density))
        geometry.attrs["vertex", "is_fixed"] = np.zeros(
            geometry.count, dtype=np.int64
        )


class _ElasticModuli:
    @staticmethod
    def youngs_poisson(youngs: float, poisson: float) -> tuple[float, float]:
        return youngs, poisson


class _GeomSlot:
    def __init__(self, geometry: _Geom) -> None:
        self._geometry = geometry

    def geometry(self) -> _Geom:
        return self._geometry


class _Geometries:
    def create(self, geometry: _Geom) -> tuple[_GeomSlot]:
        return (_GeomSlot(geometry),)


class _FakeObject:
    def __init__(self) -> None:
        self._geometries = _Geometries()

    def geometries(self) -> _Geometries:
        return self._geometries


class _FakeObjects:
    def create(self, name: str) -> _FakeObject:
        return _FakeObject()


class _FakeScene:
    def objects(self) -> _FakeObjects:
        return _FakeObjects()


def _install_uipc_stubs() -> dict[str, types.ModuleType | None]:
    saved: dict[str, types.ModuleType | None] = {}

    def save(name: str) -> None:
        saved[name] = sys.modules.get(name)

    for name in (
        "uipc",
        "uipc.constitution",
        "uipc.core",
        "uipc.geometry",
        "uipc.unit",
    ):
        save(name)

    def identity(geometry: _Geom) -> _Geom:
        return geometry

    uipc = types.ModuleType("uipc")
    setattr(uipc, "builtin", types.SimpleNamespace(is_fixed="is_fixed", is_constrained="is_constrained", aim_position="aim_position", aim_transform="aim_transform"))
    setattr(uipc, "view", lambda slot: slot.array)
    setattr(uipc, "Transform", object)
    setattr(uipc, "Vector3", object)
    setattr(uipc, "Logger", types.SimpleNamespace(set_level=lambda *_args: None, Level=types.SimpleNamespace(Warn=0)))

    constitution = types.ModuleType("uipc.constitution")
    setattr(constitution, "AffineBodyConstitution", _RecordingAbd)
    setattr(constitution, "ElasticModuli", _ElasticModuli)
    setattr(constitution, "SoftPositionConstraint", _Recorder)
    setattr(constitution, "SoftTransformConstraint", _Recorder)
    setattr(constitution, "StableNeoHookean", _RecordingSnh)

    core = types.ModuleType("uipc.core")
    setattr(core, "Engine", object)
    setattr(core, "World", object)
    setattr(core, "Scene", object)
    setattr(core, "ContactSystemFeature", object)

    geometry = types.ModuleType("uipc.geometry")
    setattr(geometry, "Geometry", _Geom)
    setattr(geometry, "extract_surface", lambda geom: _Geom(geom.vertices_array.copy()))
    setattr(geometry, "label_surface", identity)
    setattr(geometry, "label_triangle_orient", identity)
    setattr(geometry, "flip_inward_triangles", identity)
    setattr(geometry, "tetmesh", lambda vertices, tets: _Geom(vertices))

    unit = types.ModuleType("uipc.unit")
    setattr(unit, "GPa", 1000000000.0)
    setattr(unit, "MPa", _MPA)

    setattr(uipc, "constitution", constitution)
    setattr(uipc, "core", core)
    setattr(uipc, "geometry", geometry)
    setattr(uipc, "unit", unit)
    sys.modules.update(
        {
            "uipc": uipc,
            "uipc.constitution": constitution,
            "uipc.core": core,
            "uipc.geometry": geometry,
            "uipc.unit": unit,
        }
    )
    return saved


_SAVED = _install_uipc_stubs()
import openvitac_tactile  # ruff: ignore[module-import-not-at-top-of-file]
import openvitac_tactile.uipc_scene as _scene_module  # ruff: ignore[module-import-not-at-top-of-file]

UipcScene = _scene_module.UipcScene
_object_pose = _scene_module._object_pose
_rigid_transform = _scene_module._rigid_transform

# Leave no fake ``uipc`` modules or fake-backed scene module behind for tests
# that run a real integration scene later in the same process.
sys.modules.pop("openvitac_tactile.uipc_scene", None)
if getattr(openvitac_tactile, "uipc_scene", None) is _scene_module:
    del openvitac_tactile.uipc_scene
for _name, _module in _SAVED.items():
    if _module is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _module


class UipcObjectConstitutionTests(unittest.TestCase):
    def _scene(
        self,
        objects: tuple[ObjectCfg, ...],
        *,
        abd: _RecordingAbd | None = None,
        snh: _RecordingSnh | None = None,
    ) -> tuple[UipcScene, _RecordingAbd, _RecordingSnh, _Recorder]:
        abd = abd if abd is not None else _RecordingAbd()
        snh = snh if snh is not None else _RecordingSnh()
        element = _Recorder()
        scene = UipcScene(
            cfg=TactileCfg(objects=objects),
            world=object(),
            engine=object(),
            scene=_FakeScene(),
            _element=element,
            _abd=abd,
            _snh=snh,
        )
        return scene, abd, snh, element

    def test_fem_selects_stable_neo_hookean(self) -> None:
        obj = ObjectCfg(
            "soft",
            "/World/soft",
            constitution="fem",
            youngs_mpa=2.0,
            poisson=0.45,
            density_kg_m3=900.0,
        )
        scene, abd, snh, element = self._scene((obj,))
        scene.register_object(obj)

        self.assertEqual(abd.calls, [])
        self.assertEqual(len(snh.calls), 1)
        geometry, moduli, density = snh.calls[0]
        self.assertEqual(moduli, (2.0 * _MPA, 0.45))
        self.assertEqual(density, 900.0)
        self.assertEqual(len(element.calls), 1)
        # FEM placement lives in vertex positions, not the instance transform.
        np.testing.assert_allclose(
            geometry.vertices_array[:, 2].mean(), 0.035, atol=1e-12
        )
        np.testing.assert_allclose(
            geometry.transforms().array[0], np.eye(4, dtype=np.float64)
        )
        self.assertFalse(np.any(geometry.attrs["vertex", "is_fixed"]))
        # ObjectState.T stays a world pose for FEM, like the ABD path.
        np.testing.assert_allclose(
            _object_pose(scene._objects[0])[:3, 3], (0.0, 0.0, 0.035), atol=1e-12
        )

    def test_fixed_fem_marks_all_vertices_fixed(self) -> None:
        obj = ObjectCfg(
            "soft",
            "/World/soft",
            constitution="fem",
            youngs_mpa=2.0,
            poisson=0.45,
            fixed=True,
        )
        scene, _abd, snh, _element = self._scene((obj,))
        scene.register_object(obj)
        geometry = snh.calls[0][0]
        self.assertTrue(np.all(geometry.attrs["vertex", "is_fixed"] == 1))

    def test_abd_path_is_unchanged(self) -> None:
        obj = ObjectCfg(
            "box",
            "/World/box",
            constitution="abd",
            kappa_mpa=50.0,
            density_kg_m3=1.0e3,
        )
        scene, abd, snh, element = self._scene((obj,))
        scene.register_object(obj)

        self.assertEqual(snh.calls, [])
        self.assertEqual(len(abd.calls), 1)
        geometry, kappa, density = abd.calls[0]
        self.assertEqual(kappa, 50.0 * _MPA)
        self.assertEqual(density, 1.0e3)
        self.assertEqual(len(element.calls), 1)
        T = geometry.transforms().array[0]
        np.testing.assert_allclose(T[:3, 3], (0.0, 0.0, 0.035))
        np.testing.assert_allclose(T[:3, :3], np.eye(3))
        self.assertFalse(np.any(geometry.attrs["instance", "is_fixed"]))

    def test_unknown_constitution_is_rejected_at_register(self) -> None:
        obj = ObjectCfg("box", "/World/box", constitution="abd")
        object.__setattr__(obj, "constitution", "spring")
        scene, _abd, _snh, _element = self._scene((obj,))
        with self.assertRaisesRegex(ValueError, "unsupported constitution"):
            scene.register_object(obj)

    def test_rigid_transform_recovers_pose(self) -> None:
        angle = math.radians(30.0)
        R = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        t = np.array([0.01, -0.02, 0.3])
        source = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        target = source @ R.T + t
        T = _rigid_transform(source, target)
        np.testing.assert_allclose(T[:3, :3], R, atol=1e-12)
        np.testing.assert_allclose(T[:3, 3], t, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
