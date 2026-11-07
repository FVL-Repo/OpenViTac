"""Tests for the complete visual-only UIPC gelpad proxy."""

from __future__ import annotations

import unittest

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

from openvitac_sim.gelpad_visual import (
    GelpadVisual,
    find_gelpad_mesh,
    mesh_world_transform,
    rewrite_mesh_topology,
    visual_update_due,
    world_to_local,
)
from openvitac_sim.tactile_bridge import PadSurface


def _stage_with_gelpad(
    translate: tuple[float, float, float] = (1.0, 2.0, 3.0),
) -> tuple[Usd.Stage, Usd.Prim]:
    stage = Usd.Stage.CreateInMemory()
    gelpad = UsdGeom.Xform.Define(stage, "/World/Robot/gelpad_left")
    gelpad.AddTranslateOp().Set(Gf.Vec3d(*translate))
    mesh = UsdGeom.Mesh.Define(stage, "/World/Robot/gelpad_left/mesh")
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0, 0, 0),
                Gf.Vec3f(1, 0, 0),
                Gf.Vec3f(0, 1, 0),
                Gf.Vec3f(0, 0, 1),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3, 3]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 0, 1, 3]))
    return stage, mesh.GetPrim()


class GelpadVisualTests(unittest.TestCase):
    def test_world_to_local_inverts_transform(self) -> None:
        angle = 0.3
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = (0.4, -0.2, 0.3)
        local = np.array([[0.0, 0.0, 0.0], [0.01, 0.02, -0.03]])
        world = local @ rotation.T + T[:3, 3]
        np.testing.assert_allclose(world_to_local(T, world), local, atol=1e-12)

    def test_visual_update_due_matches_recording_cadence(self) -> None:
        for step in range(6):
            self.assertEqual(
                visual_update_due(step, 2, recording=True), step % 2 == 0
            )
            self.assertTrue(visual_update_due(step, 2, recording=False))
        with self.assertRaises(ValueError):
            visual_update_due(0, 0, recording=True)

    def test_find_gelpad_mesh_requires_direct_child(self) -> None:
        stage, mesh = _stage_with_gelpad()
        found = find_gelpad_mesh(stage, "/World/Robot/gelpad_left")
        self.assertEqual(found.GetPath(), mesh.GetPath())
        with self.assertRaises(RuntimeError):
            find_gelpad_mesh(stage, "/World/Robot/gelpad_left/mesh")

    def test_rewrite_topology_replaces_authored_surface(self) -> None:
        _stage, mesh_prim = _stage_with_gelpad()
        mesh = UsdGeom.Mesh(mesh_prim)
        mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 1)] * 6))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateVelocitiesAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 0)] * 4))
        st = mesh.GetPrim().CreateAttribute(
            "primvars:st", Sdf.ValueTypeNames.TexCoord2fArray
        )
        st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0)] * 4))
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.0, 0.01, 0.0],
                [0.0, 0.0, 0.01],
                [0.01, 0.01, 0.0],
            ]
        )
        triangles = np.array([[0, 1, 2], [2, 1, 3], [1, 4, 3]], dtype=np.int32)
        rewrite_mesh_topology(mesh_prim, vertices, triangles)
        points = np.asarray(mesh.GetPointsAttr().Get()).reshape(-1, 3)
        np.testing.assert_allclose(points, vertices, atol=1e-6)
        self.assertEqual(list(mesh.GetFaceVertexCountsAttr().Get()), [3, 3, 3])
        self.assertEqual(
            list(mesh.GetFaceVertexIndicesAttr().Get()),
            triangles.reshape(-1).tolist(),
        )
        normals = mesh.GetNormalsAttr().Get()
        self.assertEqual(len(normals), 3 * len(triangles))
        self.assertEqual(
            mesh.GetNormalsInterpolation(), UsdGeom.Tokens.faceVarying
        )
        self.assertEqual(str(mesh.GetSubdivisionSchemeAttr().Get()), "none")
        self.assertTrue(mesh.GetDoubleSidedAttr().Get())
        for name in ("velocities", "primvars:st"):
            if mesh_prim.HasProperty(name):
                stale = mesh_prim.GetAttribute(name).Get()
                self.assertTrue(stale is None or len(stale) == 0)

    def test_bind_keeps_authored_material_and_all_faces(self) -> None:
        stage, mesh_prim = _stage_with_gelpad((1.0, 2.0, 3.0))
        black = UsdShade.Material.Define(
            stage, "/World/Robot/gelpad_left/Looks/Black"
        )
        UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(black)
        local_rest = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.0, 0.01, 0.0],
                [0.0, 0.0, -0.004],
            ]
        )
        triangles = np.array(
            [[0, 1, 2], [0, 2, 3], [2, 1, 3], [1, 0, 3]], dtype=np.int32
        )
        T = np.eye(4)
        T[:3, 3] = (1.0, 2.0, 3.0)
        visual = GelpadVisual.bind(
            side="left",
            gelpad_prim_path="/World/Robot/gelpad_left",
            rest_vertices=local_rest + T[:3, 3],
            triangles=triangles,
            stage=stage,
        )
        mesh = UsdGeom.Mesh(visual.mesh_prim)
        self.assertEqual(
            len(mesh.GetFaceVertexCountsAttr().Get()), len(triangles)
        )
        bound = UsdShade.MaterialBindingAPI(visual.mesh_prim).ComputeBoundMaterial()[0]
        self.assertEqual(bound.GetPath(), black.GetPath())
        self.assertFalse(
            stage.GetPrimAtPath("/World/Robot/gelpad_left/Looks/Marker").IsValid()
        )
        np.testing.assert_allclose(visual.T_world_mesh, T)
        moved = local_rest + T[:3, 3] + np.array([0.0, 0.0, -0.005])
        visual.update(moved)
        np.testing.assert_allclose(
            np.asarray(mesh.GetPointsAttr().Get()).reshape(-1, 3),
            moved - T[:3, 3],
            atol=1e-6,
        )
        with self.assertRaises(RuntimeError):
            visual.update(moved[:2])

    def test_mesh_world_transform_reads_authored_translate(self) -> None:
        _stage, mesh = _stage_with_gelpad((0.4, -0.2, 0.3))
        T = mesh_world_transform(mesh)
        np.testing.assert_allclose(T[:3, 3], [0.4, -0.2, 0.3])


class PadSurfaceTests(unittest.TestCase):
    def test_validates_topology(self) -> None:
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        surface = PadSurface(
            name="tactile_left",
            side="left",
            rest_vertices=vertices,
            triangles=triangles,
        )
        np.testing.assert_array_equal(surface.rest_vertices, vertices)
        with self.assertRaises(ValueError):
            PadSurface(
                name="broken",
                side="left",
                rest_vertices=vertices,
                triangles=np.array([[0, 1, 3]], dtype=np.int32),
            )
        with self.assertRaises(ValueError):
            PadSurface(
                name="broken",
                side="left",
                rest_vertices=np.array([[0, 0, np.nan]]),
                triangles=triangles,
            )


if __name__ == "__main__":
    unittest.main()
