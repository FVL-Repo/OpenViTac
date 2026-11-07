from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from openvitac_tactile.assets import (
    boundary_triangles,
    box_tet_mesh,
    load_tet_mesh,
    published_surface,
    tetmesh_surface,
)


class AssetTests(unittest.TestCase):
    def test_box_tet_mesh(self) -> None:
        mesh = box_tet_mesh((0.03, 0.004, 0.02), (8, 1, 5))
        self.assertEqual(mesh.vertices.shape, (9 * 2 * 6, 3))
        self.assertEqual(mesh.tets.shape, (8 * 1 * 5 * 6, 4))
        self.assertTrue(np.all(mesh.tets >= 0))
        faces = boundary_triangles(mesh.tets)
        self.assertEqual(faces.shape[1], 3)
        self.assertGreater(len(faces), 0)

    def test_surface_truncation(self) -> None:
        mesh = box_tet_mesh((0.03, 0.004, 0.02), (8, 1, 5))
        full = tetmesh_surface(mesh)
        self.assertEqual(full.vertices.shape[1], 3)
        self.assertGreater(full.vertices.shape[0], 0)
        small = tetmesh_surface(mesh, max_vertices=8)
        self.assertLessEqual(small.vertices.shape[0], 8)
        self.assertLessEqual(
            small.triangles.max(initial=-1), small.vertices.shape[0] - 1
        )

    def test_published_surface_preserves_order_and_remaps_faces(self) -> None:
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
        )
        triangles = np.array([[1, 2, 3]], dtype=np.int32)
        surface = published_surface(vertices, triangles, max_vertices=3, decimation=1)
        np.testing.assert_array_equal(surface.vertices, vertices[[1, 2, 3]])
        np.testing.assert_array_equal(surface.triangles, [[0, 1, 2]])

        capped = published_surface(
            vertices,
            np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32),
            max_vertices=3,
        )
        np.testing.assert_array_equal(capped.vertices, vertices[:3])
        np.testing.assert_array_equal(capped.triangles, [[0, 1, 2]])

        dropped = published_surface(
            vertices,
            np.array([[0, 2, 3]], dtype=np.int32),
            max_vertices=4,
            decimation=2,
        )
        self.assertEqual(dropped.vertices.shape, (0, 3))
        self.assertEqual(dropped.triangles.shape, (0, 3))

    def test_load_npz_roundtrip(self) -> None:
        mesh = box_tet_mesh((0.03, 0.004, 0.02), (2, 1, 1))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "pad.npz"
            np.savez(p, verts=mesh.vertices, tets=mesh.tets)
            loaded = load_tet_mesh(p)
        self.assertEqual(loaded.vertices.shape, mesh.vertices.shape)
        self.assertEqual(loaded.tets.shape, mesh.tets.shape)


if __name__ == "__main__":
    unittest.main()
