"""Marker lattice rendering for the tactile imaging pass.

The lattice is printed on the flat gel contact face.  Each dot has one fixed
camera-frame rest point on that plane; a contact height field moves it along the
pad's optical axis (towards the camera) by ``marker_gain * h``, and the pinhole
projection turns that into image motion.  Dots therefore move because the
height field moves them, never because a simulated gel mesh wobbles.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from .surface import BoolArray, FloatArray


def marker_pixels_at(
    rest_points: NDArray[np.floating[Any]],
    heights_m: NDArray[np.floating[Any]],
    intrinsics: NDArray[np.floating[Any]],
) -> tuple[FloatArray, BoolArray]:
    """Project lattice points lifted along the optical axis by ``heights_m``.

    ``rest_points`` are camera-frame points on the gel contact plane; a positive
    height moves each towards the camera, which projects it radially outward.
    Returns the pixel coordinates and a validity mask.
    """
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    points = np.asarray(rest_points, dtype=np.float64).reshape(-1, 3)
    lift = np.asarray(heights_m, dtype=np.float64).reshape(-1)
    if len(points) != len(lift):
        raise ValueError("marker points and heights must have equal length")
    depth = -points[:, 2] - lift
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * points[:, 0] / depth + K[0, 2]
        v = K[1, 1] * (-points[:, 1]) / depth + K[1, 2]
    visible = np.isfinite(u) & np.isfinite(v) & (depth > 1e-9)
    return np.stack((u, v), axis=1), visible


def _marker_image(
    pixels: NDArray[np.floating[Any]],
    image_size: tuple[int, int],
    *,
    radius: float,
) -> NDArray[np.uint8]:
    """White keep-mask with anti-aliased black dots at ``pixels``."""
    width, height = image_size
    u = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)[:, 0]
    v = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)[:, 1]
    visible = np.isfinite(u) & np.isfinite(v)
    supersample = 4
    marker = Image.new("L", (width * supersample, height * supersample), 255)
    draw = ImageDraw.Draw(marker)
    scaled_radius = radius * supersample
    for uu, vv, ok in zip(u, v, visible, strict=True):
        if not ok:
            continue
        if uu < -radius or uu > width + radius or vv < -radius or vv > height + radius:
            continue
        cu, cv = uu * supersample, vv * supersample
        draw.ellipse(
            (
                cu - scaled_radius,
                cv - scaled_radius,
                cu + scaled_radius,
                cv + scaled_radius,
            ),
            fill=0,
        )
    return np.asarray(
        marker.resize((width, height), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
