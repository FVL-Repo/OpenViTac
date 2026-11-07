"""Pinhole calibration shared by the sim cameras and the offline replay."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def pinhole_matrix(
    *,
    focal_length: float,
    horizontal_aperture: float,
    vertical_aperture: float,
    width: int,
    height: int,
) -> NDArray[np.float64]:
    """``K`` for a USD pinhole camera covering the full sensor aperture."""
    fx = focal_length / horizontal_aperture * width
    fy = focal_length / vertical_aperture * height
    return np.array(
        [
            [fx, 0.0, width / 2.0],
            [0.0, fy, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
