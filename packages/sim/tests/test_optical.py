from __future__ import annotations

import unittest

import numpy as np

from openvitac_sim.optical import (
    build_optical_backend,
    taxim_height_map_mm,
)


def _erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    eroded = mask.copy()
    for _ in range(iterations):
        previous = eroded.copy()
        eroded[1:, :] &= previous[:-1, :]
        eroded[:-1, :] &= previous[1:, :]
        eroded[:, 1:] &= previous[:, :-1]
        eroded[:, :-1] &= previous[:, 1:]
    return eroded


def _plane() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = 21
    xs = np.linspace(-0.02, 0.02, n)
    ys = np.linspace(-0.015, 0.015, n)
    vertices = np.array(
        [[x, y, -0.03] for y in ys for x in xs], dtype=np.float64
    )
    triangles: list[list[int]] = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            triangles.extend(([a, a + 1, a + n], [a + 1, a + n + 1, a + n]))
    intrinsics = np.array(
        [[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]]
    )
    return vertices, np.asarray(triangles, dtype=np.int32), intrinsics, np.eye(4)


def _rest_depth() -> np.ndarray:
    # Visible gel, near clip removes the near sheet: the depth camera sees the
    # far/contact sheet at roughly the authored 28.25 mm rest distance.
    return np.full((240, 320), 0.02825, dtype=np.float32)


class OpticalBackendTests(unittest.TestCase):
    def test_procedural_backend_renders_contact_and_marker(self) -> None:
        vertices, triangles, intrinsics, camera = _plane()
        backend = build_optical_backend(
            backend="procedural",
            side="left",
            kind="gsmini",
            rest_vertices=vertices,
            triangles=triangles,
            camera_pose=camera,
            camera_intrinsics=intrinsics,
            image_size=(320, 240),
            marker_rows=7,
            marker_cols=9,
        )
        frame = backend.render(vertices, camera)
        self.assertEqual(frame.tactile_rgb.shape, (240, 320, 3))
        self.assertEqual(frame.marker_rgb.shape, (240, 320, 3))
        self.assertEqual(frame.aperture_mask.shape, (240, 320))
        self.assertGreater(int(frame.aperture_mask.sum()), 0)

    def test_taxim_height_map_is_rest_minus_live(self) -> None:
        rest = np.full((2, 2), 0.02825, dtype=np.float32)
        live = np.array(
            [[0.02825, 0.02775], [0.02725, 0.03000]], dtype=np.float32
        )
        np.testing.assert_allclose(
            taxim_height_map_mm(rest, live),
            np.array([[0.0, 0.5], [1.0, -1.75]], dtype=np.float32),
            atol=1e-6,
        )
        with self.assertRaises(ValueError):
            taxim_height_map_mm(rest, live[:1])
        with self.assertRaises(ValueError):
            bad = live.copy()
            bad[0, 0] = np.nan
            taxim_height_map_mm(rest, bad)

    def test_taxim_backend_responds_to_deformation(self) -> None:
        try:
            from openvitac_sim.optical import TaximOptical
        except ImportError:
            self.skipTest("torch/torchvision unavailable")
        vertices, triangles, intrinsics, camera = _plane()
        try:
            backend = TaximOptical.bind(
                side="left",
                kind="gsmini",
                rest_vertices=vertices,
                triangles=triangles,
                camera_pose=camera,
                camera_intrinsics=intrinsics,
                image_size=(320, 240),
                marker_rows=7,
                marker_cols=9,
                device="cpu",
            )
        except (FileNotFoundError, RuntimeError) as exc:
            self.skipTest(str(exc))
        rest = _rest_depth()
        flat = backend.render(
            vertices, camera, camera_depth_m=rest, rest_depth_m=rest
        ).tactile_rgb
        pressed_depth = rest.copy()
        pressed_depth[100:140, 120:200] = 0.0275
        deformed = backend.render(
            vertices, camera, camera_depth_m=pressed_depth, rest_depth_m=rest
        ).tactile_rgb
        self.assertEqual(flat.shape, (240, 320, 3))
        self.assertFalse(np.array_equal(flat, deformed))
        with self.assertRaisesRegex(RuntimeError, "requires live gel-surface"):
            backend.render(vertices, camera)
        with self.assertRaisesRegex(RuntimeError, "requires the per-pad rest"):
            backend.render(vertices, camera, camera_depth_m=rest)

    def test_taxim_rest_has_a_flat_spatial_background(self) -> None:
        try:
            from openvitac_sim.optical import TaximOptical
        except ImportError:
            self.skipTest("torch/torchvision unavailable")
        vertices, triangles, intrinsics, camera = _plane()
        try:
            backend = TaximOptical.bind(
                side="left",
                kind="gsmini",
                rest_vertices=vertices,
                triangles=triangles,
                camera_pose=camera,
                camera_intrinsics=intrinsics,
                image_size=(320, 240),
                marker_rows=7,
                marker_cols=9,
                device="cpu",
            )
        except (FileNotFoundError, RuntimeError) as exc:
            self.skipTest(str(exc))
        rest = _rest_depth()
        frame = backend.render(
            vertices, camera, camera_depth_m=rest, rest_depth_m=rest
        )
        luminance = frame.tactile_rgb.astype(np.float64).mean(axis=-1)
        aperture = frame.aperture_mask
        edge = aperture & ~_erode(aperture, 2)
        center = _erode(aperture, 15)
        self.assertTrue(edge.any())
        self.assertTrue(center.any())
        expected_background = (
            backend.rest_background_image.cpu().numpy().transpose(1, 2, 0) * 255.0
        )
        np.testing.assert_allclose(
            frame.tactile_rgb.astype(np.float64), expected_background, atol=1.0
        )
        self.assertGreater(
            float(luminance[center].mean()), float(luminance[edge].mean())
        )
        expected_mean = backend.rest_background_rgb.cpu().numpy() * 255.0
        actual_mean = frame.tactile_rgb.reshape(-1, 3).astype(np.float64).mean(axis=0)
        np.testing.assert_allclose(actual_mean, expected_mean, atol=1.0)
        self.assertTrue(np.all(np.isfinite(frame.tactile_rgb)))
        self.assertEqual(frame.marker_rgb.shape, (240, 320, 3))


if __name__ == "__main__":
    unittest.main()
