"""Static tactile-coverage gate for every registered task.

A task whose embodiment carries GsMini pads must move real UIPC bodies: every
dynamic UIPC actor owns a loadable tetrahedral mesh that matches its USD, the
episode's contact object is exactly the object the success test depends on, and
the expert's close command is the width-derived one.  Production code carries no
migration flag: ``NOT_READY`` below is the only status source, and the gate
enforces both directions - a listed task must really fail a check (otherwise
"remove it from NOT_READY"), and every unlisted task must pass them all.
Every actor must also spell its world (``Actor.authority`` has no default), so
a task cannot land tactile-dead by inheriting an implicit solver assignment.
"""

from __future__ import annotations

import importlib
import unittest
from pathlib import Path
from typing import Any, cast

import numpy as np
from pxr import Usd, UsdGeom

from openvitac_spec import Actor
from openvitac_tasks import TASKS, TaskName
from openvitac_tasks.motion import pad_close_q

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / "assets"
BBOX_TOL_M = 1e-6
VOLUME_TOL = 1e-3

# task -> why the tactile property checks cannot pass yet.  The only source of
# truth for migration state: a listed task is skipped by the property checks,
# but the gate verifies it really fails one, so a fixed task must be removed.
NOT_READY: dict[TaskName, str] = {
    "hardness_classify": "stub: no per-episode setup, expert or success yet",
    "insert_block": (
        "tet bbox tolerance: half_cylinder 9.31 um, hexagon 9.25 um vs the "
        "gate's 1 um"
    ),
    "insert_block_v2": (
        "registry alias of insert_block; inherits its tet bbox tolerance"
    ),
    "insert_usb": (
        "asset bbox 26 um vs the gate's 1 um; close is not width-derived "
        "(close_qpos=0) and the tactile crush check is pending"
    ),
    "roughness_classify": (
        "smooth-label episodes still grasp the PhysX smooth_block; only "
        "rough_block is UIPC"
    ),
    "turn_gear_pair": "stub; Xense/Robotiq embodiment, no GsMini pads",
}

CANON = {"insert_block_v2": "insert_block"}
INSERT_TARGETS = {
    "cube": "block_blue_cube",
    "half_cylinder": "block_blue_half_cylinder",
    "hexagon": "block_red_hexagonal_prism",
}

# task -> actor name -> (grasp width [m], overtravel [m], close command [m]).
# The width is the grasped feature: the USD y extent where the pads close on the
# block's +/-y faces, the 20 mm handle nub for pull_drawer (verified against the
# tet hull in test_pull_drawer).  The command is the value an episode actually
# sent.
CLOSE: dict[str, dict[str, tuple[float, float, float]]] = {
    "insert_block": {
        "block_blue_cube": (0.040000, 0.001, 0.0154362),
        "block_blue_half_cylinder": (0.025000, 0.001, 0.0079362),
        "block_red_hexagonal_prism": (0.031176, 0.001, 0.0110242),
    },
    "pull_drawer": {"upper_drawer": (0.020000, 0.0015, 0.0049362)},
    "roughness_classify": {"rough_block": (0.01588, 0.001, 0.0033762)},
    "roughness_regrasp": {
        "smooth_block": (0.01500, 0.001, 0.0029362),
        "rough_block": (0.01588, 0.001, 0.0033762),
    },
    "weight_classify": {
        "light_block": (0.01500, 0.001, 0.0029362),
        "heavy_block": (0.01500, 0.001, 0.0029362),
    },
}

# Which usd axis carries the grasp width.  pull_drawer's width is the handle
# nub, not a bbox extent, so it is checked against the tet hull separately.
CLOSE_EXTENT_AXIS = {
    "insert_block": 1,
    "pull_drawer": None,
    "roughness_classify": 1,
    "roughness_regrasp": 1,
    "weight_classify": 1,
}

# Solid-hull tet proxies: the usd is hollow, so the tet mesh is the filled
# outer silhouette and its volume is the hull volume, not the usd's material
# volume.  (min corner, max corner, volume) [m, m^3], recorded from the
# generated asset; fTetWild tracks the input surface within a few tenths of a
# millimetre.
HULLS: dict[
    str, tuple[tuple[float, float, float], tuple[float, float, float], float]
] = {
    "pull_drawer/upper_drawer_tet.npz": (
        (-0.075, -0.105, 0.0),
        (0.075, 0.130, 0.050),
        1.585e-3,
    ),
}
HULL_BBOX_TOL_M = 5e-4

CLOSE_CONST = {
    "insert_block": (
        "openvitac_tasks.tasks.insert_block",
        "_CLOSE_Q_BY_BLOCK",
    ),
    "pull_drawer": (
        "openvitac_tasks.tasks.pull_drawer",
        "_CLOSE_Q",
    ),
    "roughness_classify": (
        "openvitac_tasks.tasks.roughness_classify",
        "_CLOSE_Q",
    ),
    "roughness_regrasp": (
        "openvitac_tasks.tasks.roughness_regrasp",
        "_CLOSE_Q_BY_BLOCK",
    ),
    "weight_classify": (
        "openvitac_tasks.tasks.weight_classify",
        "_CLOSE_Q",
    ),
}

_cache: dict[str, Any] = {}


def _usd_triangles(rel: str) -> tuple[np.ndarray, np.ndarray]:
    if rel in _cache:
        return _cache[rel]
    stage = Usd.Stage.Open(str(ASSETS / rel))
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    assert len(meshes) == 1, (rel, [str(p.GetPath()) for p in meshes])
    mesh = UsdGeom.Mesh(meshes[0])
    points = np.array(
        [[q[0], q[1], q[2]] for q in mesh.GetPointsAttr().Get()],
        dtype=np.float64,
    )
    indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    faces: list[tuple[int, int, int]] = []
    k = 0
    for c in counts:
        f = indices[k : k + c]
        k += c
        for i in range(1, c - 1):
            faces.append((int(f[0]), int(f[i]), int(f[i + 1])))
    _cache[rel] = (points, np.array(faces, dtype=np.int64))
    return _cache[rel]


def _tet_mesh(rel: str) -> tuple[np.ndarray, np.ndarray]:
    if rel in _cache:
        return _cache[rel]
    with np.load(ASSETS / rel) as data:
        assert "verts" in data and "tets" in data, f"{rel}: need verts/tets"
        verts = np.asarray(data["verts"])
        tets = np.asarray(data["tets"])
    _cache[rel] = (verts, tets)
    return _cache[rel]


def _surface_volume(points: np.ndarray, faces: np.ndarray) -> float:
    a, b, c = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0))


class TactileCoverageTests(unittest.TestCase):
    def test_not_ready_tasks_carry_a_reason_and_really_fail(self) -> None:
        for name, reason in NOT_READY.items():
            with self.subTest(task=name):
                self.assertIn(name, TASKS, f"{name}: not a registered task")
                self.assertTrue(reason.strip(), f"{name}: no reason given")
                failures = self._failures(TASKS[name])
                self.assertTrue(
                    failures,
                    f"{name}: passes every check - remove it from NOT_READY",
                )

    def test_other_tasks_pass_every_check(self) -> None:
        for name, task in TASKS.items():
            if name in NOT_READY:
                continue
            with self.subTest(task=name):
                failures = self._failures(task)
                self.assertFalse(
                    failures,
                    f"{name} is not listed in NOT_READY but fails: "
                    + "; ".join(failures),
                )

    def test_actor_requires_an_explicit_authority(self) -> None:
        # An implicit solver assignment is how seven tasks became tactile-dead;
        # construction must fail instead of quietly picking PhysX.
        with self.assertRaises(TypeError):
            cast(Any, Actor)("prop.usd", "/World/prop")

    def _failures(self, task: Any) -> list[str]:
        """Every property check's failure message, in one list."""
        canon = CANON.get(task.name, task.name)
        failures: list[str] = []
        for label, check in (
            ("meshes", self._check_meshes),
            ("contact", self._check_contact),
        ):
            try:
                check(task)
            except AssertionError as exc:
                failures.append(f"{label}: {exc}")
        if canon not in CLOSE:
            failures.append("close: no width-derived close pinned")
        else:
            try:
                self._check_close(task, canon)
            except AssertionError as exc:
                failures.append(f"close: {exc}")
        return failures

    # ---- helpers -------------------------------------------------------
    def _check_meshes(self, task: Any) -> None:
        physx_dynamic = [
            actor
            for actor in task.actors
            if actor.dynamic and actor.authority != "uipc"
        ]
        if physx_dynamic:
            self.fail(
                f"{task.name}: dynamic actor "
                f"{physx_dynamic[0].prim_path!r} is PhysX in a "
                "tactile-ready task"
            )
        dynamic = [
            actor
            for actor in task.actors
            if actor.dynamic and actor.authority == "uipc"
        ]
        self.assertTrue(
            dynamic,
            f"{task.name}: no dynamic UIPC actor; the manipulated objects "
            "are PhysX (mark the task not tactile-ready instead)",
        )
        for actor in dynamic:
            rel = actor.tactile_mesh
            if rel is None and actor.usd.endswith(".npz"):
                rel = actor.usd
            self.assertIsNotNone(
                rel,
                f"{task.name}:{actor.prim_path}: dynamic UIPC actor without "
                "a tactile mesh",
            )
            assert rel is not None
            self.assertTrue(
                (ASSETS / rel).is_file(),
                f"{task.name}:{actor.prim_path}: missing mesh asset {rel}",
            )
            verts, tets = _tet_mesh(rel)
            self.assertEqual(verts.ndim, 2, f"{rel}: verts must be (V,3)")
            self.assertEqual(verts.shape[1], 3, f"{rel}: verts must be (V,3)")
            self.assertTrue(np.issubdtype(verts.dtype, np.floating),
                            f"{rel}: verts dtype {verts.dtype}")
            self.assertEqual(tets.ndim, 2, f"{rel}: tets must be (T,4)")
            self.assertEqual(tets.shape[1], 4, f"{rel}: tets must be (T,4)")
            self.assertTrue(np.issubdtype(tets.dtype, np.integer),
                            f"{rel}: tets dtype {tets.dtype}")
            self.assertGreaterEqual(int(tets.min()), 0, f"{rel}: tet index < 0")
            self.assertLess(int(tets.max()), len(verts), f"{rel}: tet index OOB")
            p0, p1, p2, p3 = (verts[tets[:, i]] for i in range(4))
            signed = np.einsum(
                "ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0)
            ) / 6.0
            self.assertGreater(
                float(signed.min()), 0.0, f"{rel}: non-positive tet"
            )
            if rel in HULLS:
                want_min, want_max, want_vol = HULLS[rel]
                bbox_err = float(
                    np.max(
                        np.maximum(
                            np.abs(verts.min(0) - np.array(want_min)),
                            np.abs(verts.max(0) - np.array(want_max)),
                        )
                    )
                )
                self.assertLessEqual(
                    bbox_err,
                    HULL_BBOX_TOL_M,
                    f"{task.name}:{actor.prim_path}: hull bbox differs from "
                    f"the silhouette by {bbox_err * 1e6:.1f} um",
                )
                self.assertLessEqual(
                    abs(float(signed.sum()) - want_vol) / want_vol,
                    VOLUME_TOL,
                    f"{task.name}:{actor.prim_path}: hull volume differs "
                    "from the recorded hull by more than 0.1 %",
                )
                continue
            points, faces = _usd_triangles(actor.usd)
            bbox_err = float(
                np.max(
                    np.maximum(
                        np.abs(verts.min(0) - points.min(0)),
                        np.abs(verts.max(0) - points.max(0)),
                    )
                )
            )
            self.assertLessEqual(
                bbox_err,
                BBOX_TOL_M,
                f"{task.name}:{actor.prim_path}: tet bbox differs from the "
                f"USD by {bbox_err * 1e6:.3f} um",
            )
            surface_volume = _surface_volume(points, faces)
            self.assertLessEqual(
                abs(float(signed.sum()) - surface_volume) / surface_volume,
                VOLUME_TOL,
                f"{task.name}:{actor.prim_path}: tet volume differs from "
                "the USD by more than 0.1 %",
            )
        # Fixed meshes (static supports): volume and orientation, no bbox
        # contract - a coarse support mesh may legitimately round a corner.
        for actor in task.actors:
            if not actor.uipc_fixed_mesh:
                continue
            verts, tets = _tet_mesh(actor.uipc_fixed_mesh)
            p0, p1, p2, p3 = (verts[tets[:, i]] for i in range(4))
            signed = np.einsum(
                "ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0)
            ) / 6.0
            self.assertGreater(
                float(signed.min()), 0.0,
                f"{actor.uipc_fixed_mesh}: non-positive tet",
            )
            points, faces = _usd_triangles(actor.usd)
            surface_volume = _surface_volume(points, faces)
            self.assertLessEqual(
                abs(float(signed.sum()) - surface_volume) / surface_volume,
                VOLUME_TOL,
                f"{actor.prim_path}: fixed mesh volume differs by >0.1 %",
            )

    def _check_contact(self, task: Any) -> None:
        for seed in range(10):
            randomization = task.setup(seed)
            expected = self._success_object(task.name, randomization)
            self.assertIsNotNone(
                expected, f"{task.name}: no success-object mapping"
            )
            if randomization.contact_object is not None:
                actual = randomization.contact_object
            else:
                flagged = [
                    actor.prim_path
                    for actor in task.actors
                    if actor.contact_object
                ]
                self.assertEqual(
                    len(flagged),
                    1,
                    f"{task.name}: expected exactly one Actor.contact_object",
                )
                actual = flagged[0]
            self.assertEqual(
                actual,
                expected,
                f"{task.name} seed {seed}: contact object {actual!r} is not "
                f"the success object {expected!r}",
            )
            actor = next(
                (a for a in task.actors if a.prim_path == actual), None
            )
            self.assertIsNotNone(
                actor, f"{task.name}: contact object {actual!r} has no actor"
            )
            assert actor is not None
            self.assertTrue(
                actor.dynamic and actor.authority == "uipc",
                f"{task.name} seed {seed}: contact object {actual!r} is not "
                "a dynamic UIPC actor",
            )

    def _check_close(self, task: Any, canon: str) -> None:
        table = CLOSE[canon]
        module_name, attr = CLOSE_CONST[canon]
        module = importlib.import_module(module_name)
        const = getattr(module, attr)
        if isinstance(const, dict):
            self.assertEqual(
                set(const), set(table), f"{task.name}: close-law actors drifted"
            )
            for actor, (_width, _overtravel, q) in table.items():
                self.assertAlmostEqual(
                    const[actor],
                    q,
                    places=9,
                    msg=f"{task.name}:{actor}: commanded close q drifted "
                    "from the recorded value",
                )
        else:
            # A single module constant may cover several actors that share
            # one grasp width (both weight blocks use the same 15 mm usd).
            commands = {q for _width, _overtravel, q in table.values()}
            self.assertEqual(
                len(commands),
                1,
                f"{task.name}: one constant cannot cover different widths",
            )
            self.assertAlmostEqual(
                const,
                commands.pop(),
                places=9,
                msg=f"{task.name}: commanded close q drifted",
            )
        axis = CLOSE_EXTENT_AXIS[canon]
        for actor_name, (width, overtravel, q) in table.items():
            prim = f"/World/actors/{actor_name}"
            actor = next(
                (a for a in task.actors if a.prim_path == prim), None
            )
            self.assertIsNotNone(actor, f"{task.name}: no actor {prim}")
            assert actor is not None
            if axis is not None:
                points, _faces = _usd_triangles(actor.usd)
                extent = float(points[:, axis].max() - points[:, axis].min())
                self.assertAlmostEqual(
                    extent,
                    width,
                    delta=1e-5,
                    msg=f"{task.name}:{actor_name}: grasp width {width} is "
                    "not the usd extent",
                )
            self.assertAlmostEqual(
                pad_close_q(width, overtravel),
                q,
                places=9,
                msg=f"{task.name}:{actor_name}: close q is not width-derived",
            )

    @staticmethod
    def _success_object(name: str, randomization: Any) -> str | None:
        if name == "roughness_regrasp":
            return "/World/actors/rough_block"
        if name == "pull_drawer":
            return "/World/actors/upper_drawer"
        if name == "weight_classify":
            label = randomization.metadata["weight_label"]
            return f"/World/actors/{label}_block"
        if name == "insert_block" or name == "insert_block_v2":
            target = randomization.metadata["target_block"]
            return f"/World/actors/{INSERT_TARGETS[target]}"
        if name == "roughness_classify":
            label = randomization.metadata["roughness_label"]
            return f"/World/actors/{label}_block"
        if name == "insert_usb":
            return "/World/actors/usb"
        return None


if __name__ == "__main__":
    unittest.main()
