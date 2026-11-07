from __future__ import annotations

import unittest

import numpy as np

from openvitac_sim.optical import (
    ProceduralOptical,
    build_optical_backend,
)
from openvitac_sim.sensor.gsmini import GSMINI_OPTICS, TactileSensorVisual

CAMERA_FROM_PAD = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -0.024254],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
INTRINSICS = np.array(
    [[320.0, 0.0, 160.0], [0.0, 20.0 / 18.0 * 240.0, 120.0], [0.0, 0.0, 1.0]]
)
IMAGE_SIZE = (320, 240)


def _pad_face() -> tuple[np.ndarray, np.ndarray]:
    n = 9
    xs = np.linspace(-0.0126, 0.0126, n)
    ys = np.linspace(-0.0104, 0.0104, n)
    vertices = np.array([[x, y, 0.004] for y in ys for x in xs], dtype=np.float64)
    triangles: list[tuple[int, int, int]] = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            triangles.extend(((a, a + n, a + 1), (a + 1, a + n, a + n + 1)))
    return vertices, np.asarray(triangles, dtype=np.int32)


def _visual() -> TactileSensorVisual:
    vertices, _ = _pad_face()
    return TactileSensorVisual.bind(
        side="left",
        rest_vertices=vertices,
        camera_from_pad=CAMERA_FROM_PAD,
        camera_intrinsics=INTRINSICS,
        image_size=IMAGE_SIZE,
        marker_rows=7,
        marker_cols=9,
    )


def _bump(height_mm: float = 1.0) -> np.ndarray:
    height = np.zeros(IMAGE_SIZE[::-1])
    height[100:150, 120:200] = height_mm
    return height


class OpticalBackendTests(unittest.TestCase):
    def test_procedural_backend_renders_from_the_height_map(self) -> None:
        backend = build_optical_backend(
            backend="procedural", visual=_visual()
        )
        frame = backend.render(height_mm=np.zeros(IMAGE_SIZE[::-1]))
        self.assertEqual(frame.tactile_rgb.shape, (240, 320, 3))
        self.assertEqual(frame.marker_rgb.shape, (240, 320, 3))
        self.assertEqual(frame.aperture_mask.shape, (240, 320))
        self.assertGreater(int(frame.aperture_mask.sum()), 0)

    def test_procedural_backend_responds_to_height(self) -> None:
        visual = _visual()
        backend = ProceduralOptical(visual)
        flat = backend.render(height_mm=np.zeros(IMAGE_SIZE[::-1])).tactile_rgb
        pressed = backend.render(height_mm=_bump()).tactile_rgb
        self.assertFalse(np.array_equal(flat, pressed))
        self.assertLess(
            float(pressed[100:150, 120:200].mean()),
            float(flat[100:150, 120:200].mean()),
        )

    def test_taxim_backend_validates_the_height_map(self) -> None:
        try:
            from openvitac_sim.optical import TaximOptical
        except ImportError:
            self.skipTest("torch/torchvision unavailable")
        calib_dir = GSMINI_OPTICS.taxim_calib_dir()
        if calib_dir is None:
            self.skipTest("Taxim calibration unavailable")
        try:
            backend = TaximOptical.bind(
                calib_dir=calib_dir, visual=_visual(), device="cpu"
            )
        except (FileNotFoundError, RuntimeError) as exc:
            self.skipTest(str(exc))
        with self.assertRaisesRegex(ValueError, "expected"):
            backend.render(height_mm=np.zeros((10, 10)))
        bad = np.zeros(IMAGE_SIZE[::-1])
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            backend.render(height_mm=bad)

    def test_taxim_backend_responds_to_height(self) -> None:
        try:
            from openvitac_sim.optical import TaximOptical
        except ImportError:
            self.skipTest("torch/torchvision unavailable")
        calib_dir = GSMINI_OPTICS.taxim_calib_dir()
        if calib_dir is None:
            self.skipTest("Taxim calibration unavailable")
        try:
            backend = TaximOptical.bind(
                calib_dir=calib_dir, visual=_visual(), device="cpu"
            )
        except (FileNotFoundError, RuntimeError) as exc:
            self.skipTest(str(exc))
        flat = backend.render(height_mm=np.zeros(IMAGE_SIZE[::-1])).tactile_rgb
        pressed = backend.render(height_mm=_bump()).tactile_rgb
        self.assertEqual(flat.shape, (240, 320, 3))
        self.assertFalse(np.array_equal(flat, pressed))

    def test_taxim_rest_has_a_flat_spatial_background(self) -> None:
        try:
            from openvitac_sim.optical import TaximOptical
        except ImportError:
            self.skipTest("torch/torchvision unavailable")
        calib_dir = GSMINI_OPTICS.taxim_calib_dir()
        if calib_dir is None:
            self.skipTest("Taxim calibration unavailable")
        try:
            backend = TaximOptical.bind(
                calib_dir=calib_dir, visual=_visual(), device="cpu"
            )
        except (FileNotFoundError, RuntimeError) as exc:
            self.skipTest(str(exc))
        frame = backend.render(height_mm=np.zeros(IMAGE_SIZE[::-1]))
        expected_background = (
            backend.rest_background_image.cpu().numpy().transpose(1, 2, 0) * 255.0
        )
        np.testing.assert_allclose(
            frame.tactile_rgb.astype(np.float64), expected_background, atol=1.0
        )
        self.assertTrue(np.all(np.isfinite(frame.tactile_rgb)))
        self.assertEqual(frame.marker_rgb.shape, (240, 320, 3))


if __name__ == "__main__":
    unittest.main()
