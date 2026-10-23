"""Tests for the independent tactile/marker optical pass."""

from __future__ import annotations

import unittest

import numpy as np

from openvitac_sim.sensor_visual import TactileSensorVisual, select_far_sheet


def _square_sheet(
    *,
    with_back: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [-0.03, -0.03, 0.0],
            [0.0, -0.03, 0.0],
            [0.03, -0.03, 0.0],
            [-0.03, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.03, 0.0, 0.0],
            [-0.03, 0.03, 0.0],
            [0.0, 0.03, 0.0],
            [0.03, 0.03, 0.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 5],
            [1, 5, 4],
            [3, 4, 7],
            [3, 7, 6],
            [4, 5, 8],
            [4, 8, 7],
        ],
        dtype=np.int32,
    )
    if with_back:
        back = vertices.copy()
        back[:, 2] = -0.004
        offset = len(vertices)
        vertices = np.concatenate((vertices, back), axis=0)
        back_triangles = triangles + offset
        triangles = np.concatenate((triangles, back_triangles), axis=0)
    return vertices, triangles


def _camera() -> tuple[np.ndarray, np.ndarray]:
    camera_pose = np.eye(4)
    camera_pose[2, 3] = 0.06
    intrinsics = np.array(
        [[80.0, 0.0, 48.0], [0.0, 80.0, 36.0], [0.0, 0.0, 1.0]]
    )
    return camera_pose, intrinsics


class TactileSensorVisualTests(unittest.TestCase):
    def _bind(self) -> tuple[TactileSensorVisual, np.ndarray, np.ndarray]:
        vertices, triangles = _square_sheet()
        camera_pose, intrinsics = _camera()
        sensor = TactileSensorVisual.bind(
            side="left",
            rest_vertices=vertices,
            triangles=triangles,
            camera_pose=camera_pose,
            camera_intrinsics=intrinsics,
            image_size=(96, 72),
            marker_rows=3,
            marker_cols=3,
        )
        return sensor, vertices, triangles

    def test_far_sheet_is_farther_than_the_parallel_near_sheet(self) -> None:
        vertices, triangles = _square_sheet(with_back=True)
        camera_pose, _ = _camera()
        selected = select_far_sheet(vertices, triangles, camera_pose[:3, 3])
        self.assertEqual(len(selected), 8)
        self.assertTrue(
            np.all(vertices[triangles[selected]][..., 2] == -0.004)
        )

    def test_render_fills_the_image_and_attaches_nine_dots(self) -> None:
        sensor, vertices, _triangles = self._bind()
        camera_pose, _ = _camera()
        frame = sensor.render(vertices, camera_pose)
        self.assertEqual(frame.tactile_rgb.shape, (72, 96, 3))
        self.assertEqual(frame.marker_rgb.shape, (72, 96, 3))
        self.assertTrue(frame.aperture_mask[36, 48])
        self.assertFalse(frame.aperture_mask[0, 0])
        # No black aperture border: the gel background fills the whole frame.
        self.assertTrue(np.all(frame.tactile_rgb.sum(axis=2) > 0))
        dark = frame.marker_rgb.mean(axis=2) < 25.0
        self.assertGreater(int(dark.sum()), 40)
        # The 3x3 lattice is visible; the dots are dark on the gel gradient.
        ys, xs = np.nonzero(dark)
        self.assertGreater(int(np.ptp(ys)), 5)
        self.assertGreater(int(np.ptp(xs)), 5)

    def test_lattice_points_follow_far_sheet_deformation(self) -> None:
        sensor, vertices, _triangles = self._bind()
        camera_pose, _ = _camera()
        before = sensor.render(vertices, camera_pose).marker_rgb
        deformed = vertices.copy()
        deformed[4, 0] += 0.012
        deformed[4, 1] += 0.008
        deformed[4, 2] += 0.002
        after = sensor.render(deformed, camera_pose).marker_rgb
        dark_before = before.mean(axis=2) < 25.0
        dark_after = after.mean(axis=2) < 25.0
        self.assertFalse(np.array_equal(dark_before, dark_after))
        _ys_before, xs_before = np.nonzero(dark_before)
        _ys_after, xs_after = np.nonzero(dark_after)
        self.assertGreater(abs(float(xs_after.mean() - xs_before.mean())), 0.2)

    def test_contact_deformation_darkens_sensor_surface(self) -> None:
        sensor, vertices, _triangles = self._bind()
        camera_pose, _ = _camera()
        rest = sensor.render(vertices, camera_pose).tactile_rgb
        deformed = vertices.copy()
        deformed[4, 2] += 0.0007
        contact = sensor.render(deformed, camera_pose).tactile_rgb
        center = (slice(28, 44), slice(38, 58))
        self.assertLess(
            float(contact[center].mean()), float(rest[center].mean())
        )


if __name__ == "__main__":
    unittest.main()


def _plane(
    extent: tuple[float, float] = (0.02, 0.015),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = 21
    half_x, half_y = extent
    xs = np.linspace(-half_x, half_x, n)
    ys = np.linspace(-half_y, half_y, n)
    vertices = np.array(
        [[x, y, -0.03] for y in ys for x in xs], dtype=np.float64
    )
    triangles: list[tuple[int, int, int]] = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            triangles.extend(((a, a + 1, a + n), (a + 1, a + n + 1, a + n)))
    intrinsics = np.array(
        [[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]]
    )
    return vertices, np.asarray(triangles, dtype=np.int32), intrinsics, np.eye(4)


def _bands(values: np.ndarray) -> int:
    """Number of contiguous True runs in a 1-D boolean array."""
    starts = np.count_nonzero(values & ~np.concatenate(([False], values[:-1])))
    return int(starts)


class FixedApertureTests(unittest.TestCase):
    def _bind(self) -> tuple[TactileSensorVisual, np.ndarray, np.ndarray]:
        vertices, triangles, intrinsics, camera = _plane()
        sensor = TactileSensorVisual.bind(
            side="left",
            rest_vertices=vertices,
            triangles=triangles,
            camera_pose=camera,
            camera_intrinsics=intrinsics,
            image_size=(320, 240),
            marker_rows=7,
            marker_cols=9,
        )
        return sensor, vertices, camera

    def test_rest_frame_is_exactly_unshaded_and_height_is_zero(self) -> None:
        sensor, vertices, camera = self._bind()
        frame = sensor.render(vertices, camera)
        self.assertTrue(np.array_equal(frame.tactile_rgb, sensor.background))
        height = sensor.height_map_mm(vertices, camera)
        self.assertLess(float(np.max(np.abs(height))), 1e-6)

    def test_depth_aperture_does_not_follow_contact(self) -> None:
        sensor, vertices, camera = self._bind()
        rest_aperture = sensor.aperture_mask()
        pressed = vertices.copy()
        pressed[:, 2] += 0.001
        sensor.render(pressed, camera)
        self.assertTrue(np.array_equal(sensor.aperture_mask(), rest_aperture))
        frame = sensor.render(pressed, camera)
        self.assertTrue(np.array_equal(frame.aperture_mask, rest_aperture))

    def test_height_is_positive_in_contact_and_zero_outside_aperture(self) -> None:
        sensor, vertices, camera = self._bind()
        pressed = vertices.copy()
        pressed[:, 2] += 0.001
        height = sensor.height_map_mm(pressed, camera)
        self.assertGreater(float(height.max()), 0.5)
        outside = ~sensor.aperture_mask()
        self.assertTrue(np.all(height[outside] == 0.0))

    def test_small_rest_footprint_keeps_centered_attached_grid(self) -> None:
        vertices, triangles, intrinsics, camera = _plane(extent=(0.01, 0.0075))
        sensor = TactileSensorVisual.bind(
            side="left",
            rest_vertices=vertices,
            triangles=triangles,
            camera_pose=camera,
            camera_intrinsics=intrinsics,
            image_size=(320, 240),
            marker_rows=7,
            marker_cols=9,
        )
        self.assertEqual(int(sensor.marker_attached.sum()), 63)
        dark = sensor.marker_mask(vertices, camera) < 25
        self.assertEqual(_bands(np.any(dark, axis=0)), 9)
        self.assertEqual(_bands(np.any(dark, axis=1)), 7)
        ys, xs = np.nonzero(dark)
        self.assertAlmostEqual(float(xs.mean()), 160.0, delta=3.0)
        self.assertAlmostEqual(float(ys.mean()), 120.0, delta=3.0)

    def test_marker_lattice_is_fixed_9_by_7_and_anti_aliased(self) -> None:
        sensor, vertices, camera = self._bind()
        marker = sensor.marker_mask(vertices, camera)
        self.assertTrue(np.any((marker > 0) & (marker < 255)))
        dark = marker < 25
        self.assertEqual(_bands(np.any(dark, axis=0)), 9)
        self.assertEqual(_bands(np.any(dark, axis=1)), 7)
        ys, xs = np.nonzero(dark)
        self.assertAlmostEqual(float(xs.mean()), 160.0, delta=3.0)
        self.assertAlmostEqual(float(ys.mean()), 120.0, delta=3.0)
        pixels = sensor.marker_pixels
        self.assertAlmostEqual(float(pixels[:, 0].mean()), 160.0, delta=3.0)
        self.assertAlmostEqual(float(pixels[:, 1].mean()), 120.0, delta=3.0)
        self.assertAlmostEqual(
            float(np.ptp(xs)) / 320.0, 232.0 / 320.0, delta=0.02
        )
        self.assertAlmostEqual(
            float(np.ptp(ys)) / 240.0, 170.0 / 240.0, delta=0.02
        )
        self.assertAlmostEqual(sensor.marker_radius, 6.2, delta=0.05)

    def test_marker_grid_centers_on_the_image_roi(self) -> None:
        vertices, triangles, intrinsics, camera = _plane(extent=(0.012, 0.009))
        shifted = vertices.copy()
        shifted[:, 0] -= 0.002
        shifted[:, 1] += 0.0003
        sensor = TactileSensorVisual.bind(
            side="left",
            rest_vertices=shifted,
            triangles=triangles,
            camera_pose=camera,
            camera_intrinsics=intrinsics,
            image_size=(320, 240),
            marker_rows=7,
            marker_cols=9,
        )
        pixels = sensor.marker_pixels
        self.assertAlmostEqual(float(pixels[:, 0].mean()), 160.0, delta=3.0)
        self.assertAlmostEqual(float(pixels[:, 1].mean()), 120.0, delta=3.0)
        self.assertEqual(int(sensor.marker_attached.sum()), 63)
        dark = sensor.marker_mask(shifted, camera) < 25
        ys, xs = np.nonzero(dark)
        self.assertAlmostEqual(
            float(xs.mean()), float(pixels[:, 0].mean()), delta=3.0
        )
        self.assertAlmostEqual(
            float(ys.mean()), float(pixels[:, 1].mean()), delta=3.0
        )
