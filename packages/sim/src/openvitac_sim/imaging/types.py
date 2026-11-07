"""Value types shared by the tactile imaging passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .surface import BoolArray, FloatArray


@dataclass(frozen=True)
class TactileSensorImages:
    """One independent sensor frame, plus the fixed rest aperture mask."""

    tactile_rgb: NDArray[np.uint8]
    marker_rgb: NDArray[np.uint8]
    aperture_mask: BoolArray


class MarkerSource(Protocol):
    """Marker and aperture planes an optical backend composes with its RGB."""

    @property
    def image_size(self) -> tuple[int, int]: ...

    def aperture_mask(self) -> BoolArray: ...

    def marker_mask(self, height_map_mm: FloatArray) -> NDArray[np.uint8]: ...
