"""Tests for the object-height tactile imaging pass."""

from __future__ import annotations

import unittest

import numpy as np

from openvitac_sim.imaging.objects import ObjectSurface, object_depth
from openvitac_sim.sensor.gsmini import GsMiniOptics, TactileSensorVisual

# The authored pad camera sits 24.254 mm behind the gelpad origin looking at
# +z_pad; the image is mirrored in y, exactly as the robot USD authors it.
CAMERA_FROM_PAD = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -0.024254],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
PAD_TO_WORLD = np.linalg.inv(CAMERA_FROM_PAD)
INTRINSICS = np.array(
    [[320.0, 0.0, 160.0], [0.0, 20.0 / 18.0 * 240.0, 120.0], [0.0, 0.0, 1.0]]
)
IMAGE_SIZE = (320, 240)


def _pad_surface(
    z: float = 0.004, half_x: float = 0.0126, half_y: float = 0.0104, n: int = 9
) -> tuple[np.ndarray, np.ndarray]:
    """Gel rest surface at pad ``z`` (calibration input, never imaging input)."""
    xs = np.linspace(-half_x, half_x, n)
    ys = np.linspace(-half_y, half_y, n)
    vertices = np.array([[x, y, z] for y in ys for x in xs], dtype=np.float64)
    triangles: list[tuple[int, int, int]] = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            triangles.extend(((a, a + n, a + 1), (a + 1, a + n, a + n + 1)))
    return vertices, np.asarray(triangles, dtype=np.int32)


def _box(half_xy: float, z0: float, z1: float) -> tuple[np.ndarray, np.ndarray]:
    x, y = half_xy, half_xy
    vertices = np.array(
        [
            [-x, -y, z0],
            [x, -y, z0],
            [x, y, z0],
            [-x, y, z0],
            [-x, -y, z1],
            [x, -y, z1],
            [x, y, z1],
            [-x, y, z1],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int32,
    )
    return vertices, triangles


def _surface(name: str = "block") -> ObjectSurface:
    vertices, triangles = _box(0.004, 0.0, 0.016)
    return ObjectSurface.from_mesh(
        body=name, prim_path=f"/World/{name}", vertices=vertices, triangles=triangles
    )


def _visual(
    *,
    objects: tuple[ObjectSurface, ...] = (),
    optics: GsMiniOptics | None = None,
    rest_vertices: np.ndarray | None = None,
) -> TactileSensorVisual:
    vertices = _pad_surface()[0] if rest_vertices is None else rest_vertices
    return TactileSensorVisual.bind(
        side="left",
        rest_vertices=vertices,
        camera_from_pad=CAMERA_FROM_PAD,
        camera_intrinsics=INTRINSICS,
        image_size=IMAGE_SIZE,
        objects=objects,
        marker_rows=7,
        marker_cols=9,
        optics=optics or GsMiniOptics(),
    )


def _pose(plane_z: float) -> np.ndarray:
    pose = np.eye(4)
    pose[2, 3] = plane_z
    return pose


def _height(visual: TactileSensorVisual, plane_z: float) -> np.ndarray:
    depth = visual.object_depth({"block": _pose(plane_z)}, PAD_TO_WORLD)
    return visual.height_map_mm(depth)


def _face_corners_px(optics: GsMiniOptics) -> np.ndarray:
    corners = optics.face_corners_pad()
    camera = corners @ CAMERA_FROM_PAD[:3, :3].T + CAMERA_FROM_PAD[:3, 3]
    depth = -camera[:, 2]
    return np.stack(
        (
            INTRINSICS[0, 0] * camera[:, 0] / depth + INTRINSICS[0, 2],
            INTRINSICS[1, 1] * (-camera[:, 1]) / depth + INTRINSICS[1, 2],
        ),
        axis=1,
    )


class ApertureTests(unittest.TestCase):
    def test_aperture_is_the_projected_face_rectangle(self) -> None:
        optics = GsMiniOptics()
        visual = _visual(optics=optics)
        corners = _face_corners_px(optics)
        width = np.ptp(corners[:, 0])
        height = np.ptp(corners[:, 1])
        self.assertAlmostEqual(
            float(visual.aperture.sum()) / (width * height), 1.0, delta=0.03
        )
        centre = corners.mean(axis=0)
        for corner in corners:
            inside = corner + 0.05 * (centre - corner)
            row, column = round(inside[1]), round(inside[0])
            self.assertTrue(visual.aperture[row, column])
            outside = corner + 0.05 * (corner - centre)
            self.assertFalse(visual.aperture[round(outside[1]), round(outside[0])])
        # Nothing outside the face plane can be inside the aperture.
        self.assertFalse(visual.aperture[0, 0])
        self.assertFalse(visual.aperture[-1, -1])
        # The aperture has straight edges: a row's span is a single run.
        row = visual.aperture[120]
        columns = np.flatnonzero(row)
        self.assertEqual(int(columns[-1] - columns[0] + 1), int(columns.size))

    def test_marker_lattice_is_the_pre_refactor_layout(self) -> None:
        # cfc6da0: target 232x170 px minus 2r at the 240 px reference height,
        # ROI margin 2 px, 44/35 px fallback margins, inside the rest
        # footprint.  On the face rectangle that is the centred 220x158 box.
        optics = GsMiniOptics()
        radius = optics.marker_radius(IMAGE_SIZE)
        self.assertEqual(radius, 6.2)
        self.assertEqual(optics.marker_target(IMAGE_SIZE, radius), (219.6, 157.6))
        self.assertEqual(optics.marker_fallback_margins(IMAGE_SIZE), (44.0, 35.0))
        visual = _visual(optics=optics)
        self.assertEqual(len(visual.marker_pixels), 63)
        np.testing.assert_allclose(visual.marker_pixels[0], [50.0, 41.0], atol=1e-9)
        np.testing.assert_allclose(visual.marker_pixels[-1], [269.0, 198.0], atol=1e-9)
        np.testing.assert_allclose(np.ptp(visual.marker_pixels[:, 0]), 219.0, atol=1e-9)
        np.testing.assert_allclose(np.ptp(visual.marker_pixels[:, 1]), 157.0, atol=1e-9)
        self.assertAlmostEqual(219.0 / IMAGE_SIZE[0], 0.6844, places=4)
        self.assertAlmostEqual(157.0 / IMAGE_SIZE[1], 0.6542, places=4)

    def test_marker_rest_points_lie_on_the_plane_at_their_pixels(self) -> None:
        visual = _visual()
        camera = visual.marker_rest_points
        K = visual.intrinsics
        depth = -camera[:, 2]
        reprojected = np.stack(
            (
                K[0, 0] * camera[:, 0] / depth + K[0, 2],
                K[1, 1] * (-camera[:, 1]) / depth + K[1, 2],
            ),
            axis=1,
        )
        np.testing.assert_allclose(reprojected, visual.marker_pixels, atol=1e-9)
        np.testing.assert_allclose(
            depth, float(np.median(visual.rest_plane_depth)), atol=1e-9
        )

    def test_calibration_check_rejects_a_wrong_plane(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not the gel"):
            _visual(optics=GsMiniOptics(plane_offset_m=-0.002))


class RestFrameTests(unittest.TestCase):
    def test_rest_frame_has_zero_height_and_the_rest_appearance(self) -> None:
        visual = _visual()
        depth = visual.object_depth({}, PAD_TO_WORLD)
        height = visual.height_map_mm(depth)
        self.assertEqual(float(height.max()), 0.0)
        frame = visual.render(height)
        self.assertTrue(np.array_equal(frame.tactile_rgb, visual.background))
        np.testing.assert_array_equal(
            frame.marker_rgb, visual.render(np.zeros(IMAGE_SIZE[::-1])).marker_rgb
        )

    def test_plane_is_the_pad_contact_face(self) -> None:
        visual = _visual()
        self.assertAlmostEqual(float(np.median(visual.rest_plane_depth)), 0.028254, 6)
        np.testing.assert_allclose(
            np.abs(visual.rest_plane_normal), [0.0, 0.0, 1.0], atol=1e-9
        )
        self.assertLess(visual.rest_plane_offset, 0.0)

    def test_object_behind_the_plane_is_zero_height(self) -> None:
        optics = GsMiniOptics()
        visual = _visual(objects=(_surface(),), optics=optics)
        behind = _height(visual, optics.plane_height() + 0.005)
        self.assertEqual(float(behind.max()), 0.0)


class ContactHeightTests(unittest.TestCase):
    def test_penetration_equals_plane_minus_object_depth(self) -> None:
        optics = GsMiniOptics()
        visual = _visual(objects=(_surface(),), optics=optics)
        for penetration_mm in (0.25, 1.0, 2.5):
            height = _height(visual, optics.plane_height() - penetration_mm * 1e-3)
            self.assertAlmostEqual(float(height.max()), penetration_mm, 4)
            self.assertTrue((height > 0.5 * penetration_mm).any())
            np.testing.assert_array_equal(height[~visual.aperture], 0.0)

    def test_contact_patch_is_the_object_outline_clipped_by_the_face(self) -> None:
        optics = GsMiniOptics()
        visual = _visual(objects=(_surface(),), optics=optics)
        height = _height(visual, optics.plane_height() - 0.001)
        contact = height > 0.5
        # The box half-extent (4 mm) is smaller than the face, so the patch is
        # the box footprint with straight edges and no diagonal beams.
        rows = np.flatnonzero(contact.any(axis=1))
        band = contact[rows.min() : rows.max() + 1]
        counts = band.sum(axis=1)
        self.assertLessEqual(int(counts.max() - counts.min()), 2)
        for column in np.flatnonzero(contact.any(axis=0))[::10]:
            runs = np.diff(
                np.concatenate(([0], contact[:, column].astype(np.int8), [0]))
            )
            self.assertEqual(int((runs == 1).sum()), 1)

    def test_contact_darkens_the_rendered_frame(self) -> None:
        optics = GsMiniOptics()
        visual = _visual(objects=(_surface(),), optics=optics)
        rest = visual.render(np.zeros(IMAGE_SIZE[::-1]))
        contact = visual.render(_height(visual, optics.plane_height() - 0.001))
        self.assertLess(
            float(contact.tactile_rgb[100:140, 140:180].mean()),
            float(rest.tactile_rgb[100:140, 140:180].mean()),
        )


class MarkerTests(unittest.TestCase):
    def test_markers_move_with_the_height_field_only(self) -> None:
        visual = _visual()
        rest = visual.marker_mask(np.zeros(IMAGE_SIZE[::-1]))
        ys, xs = np.mgrid[0:240, 0:320]
        centre = visual.marker_pixels[3 * 9 + 7]
        bump = np.exp(
            -(((xs - centre[0]) ** 2 + (ys - centre[1]) ** 2) / (2 * 20.0**2))
        )
        moved = visual.marker_mask(bump * 1.0)
        self.assertTrue(np.array_equal(rest[:40, :40], moved[:40, :40]))
        self.assertFalse(np.array_equal(rest, moved))

        def centroid(mask: np.ndarray, cx: float, cy: float) -> np.ndarray:
            half = 9
            window = mask[
                int(cy) - half : int(cy) + half + 1,
                int(cx) - half : int(cx) + half + 1,
            ]
            yy, xx = np.nonzero(window < 25)
            return np.array([xx.mean() + int(cx) - half, yy.mean() + int(cy) - half])

        before = centroid(rest, centre[0], centre[1])
        after = centroid(moved, centre[0], centre[1])
        principal = np.array([INTRINSICS[0, 2], INTRINSICS[1, 2]])
        radial = before - principal
        shift = after - before
        self.assertGreater(float(np.dot(shift, radial / np.linalg.norm(radial))), 0.2)

    def test_marker_gain_scales_the_displacement(self) -> None:
        base = _visual()
        stronger = _visual(optics=GsMiniOptics(marker_gain=3.0))
        height = np.zeros(IMAGE_SIZE[::-1])
        height[80:160, 100:220] = 1.0
        np.testing.assert_array_equal(
            base.marker_mask(np.zeros(IMAGE_SIZE[::-1])),
            stronger.marker_mask(np.zeros(IMAGE_SIZE[::-1])),
        )
        self.assertFalse(
            np.array_equal(base.marker_mask(height), stronger.marker_mask(height))
        )


class ObjectSurfaceTests(unittest.TestCase):
    def test_closed_mesh_culls_back_faces_without_changing_depth(self) -> None:
        surface = _surface()
        self.assertTrue(surface.camera_facing_only)
        culled = object_depth(
            (surface,), {"block": _pose(0.002)}, PAD_TO_WORLD, INTRINSICS, IMAGE_SIZE
        )
        every_triangle = ObjectSurface(
            body=surface.body,
            prim_path=surface.prim_path,
            vertices=surface.vertices,
            triangles=surface.triangles,
            camera_facing_only=False,
        )
        full = object_depth(
            (every_triangle,),
            {"block": _pose(0.002)},
            PAD_TO_WORLD,
            INTRINSICS,
            IMAGE_SIZE,
        )
        np.testing.assert_array_equal(culled, full)

    def test_open_mesh_is_not_culled(self) -> None:
        vertices, triangles = _pad_surface()
        surface = ObjectSurface.from_mesh(
            body="open",
            prim_path="/World/open",
            vertices=vertices,
            triangles=triangles,
        )
        self.assertFalse(surface.camera_facing_only)

    def test_rejects_a_mesh_without_triangles(self) -> None:
        with self.assertRaises(ValueError):
            ObjectSurface.from_mesh(
                body="empty",
                prim_path="/World/empty",
                vertices=np.zeros((4, 3)),
                triangles=np.zeros((0, 3), dtype=np.int32),
            )

    def test_object_depth_needs_every_pose(self) -> None:
        with self.assertRaises(KeyError):
            object_depth((_surface(),), {}, PAD_TO_WORLD, INTRINSICS, IMAGE_SIZE)


if __name__ == "__main__":
    unittest.main()
