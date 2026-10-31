"""Tactile optical backends: Taxim (default) and procedural.

Both backends consume the *height map* produced by the imaging pass
(``height = clip(rest_plane - object_depth, 0, inf)`` in the pad camera frame).
Neither backend sees a gel mesh, a captured rest image or a camera depth image:
the height map is the whole geometric input.  Backends only add appearance: the
Taxim gel optics or the procedural gel background plus the printed-dot mask,
composed exactly as before (``tactile_rgb * marker_mask / 255``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from .imaging.surface import FloatArray
from .imaging.types import MarkerSource, TactileSensorImages

BoolArray = npt.NDArray[np.bool_]
UInt8Array = npt.NDArray[np.uint8]


class VisualSource(MarkerSource, Protocol):
    """The visual pass a backend composes: marker plane plus RGB rendering."""

    def render(self, height_map_mm: FloatArray) -> TactileSensorImages: ...


class OpticalBackend(Protocol):
    """One pad's optical pass: height map in, RGB images out."""

    @property
    def image_size(self) -> tuple[int, int]: ...

    def render(self, *, height_mm: FloatArray) -> TactileSensorImages: ...

    def aperture_mask(self) -> BoolArray: ...


class ProceduralOptical:
    """Analytic gel appearance driven by the height field."""

    def __init__(self, visual: VisualSource) -> None:
        self._visual = visual

    @property
    def image_size(self) -> tuple[int, int]:
        return self._visual.image_size

    def render(self, *, height_mm: FloatArray) -> TactileSensorImages:
        return self._visual.render(height_mm)

    def aperture_mask(self) -> BoolArray:
        return self._visual.aperture_mask()


class TaximOptical:
    """Taxim RGB from the measured height map plus the marker lattice mask.

    Taxim's default height-map convention is depth *below* the gel surface
    (negative for indentation), so the physical height is negated before the
    render; ``height_gain`` scales that hand-off.  The sensor's own height map
    stays physical and is never scaled.
    """

    def __init__(
        self,
        visual: VisualSource,
        taxim: Any,
        torch: Any,
        *,
        height_gain: float = 1.0,
    ) -> None:
        self._visual = visual
        self._taxim = taxim
        self._torch = torch
        self._height_gain = float(height_gain)
        image_height, image_width = visual.image_size[1], visual.image_size[0]
        zeros = torch.zeros(
            (1, image_height, image_width),
            dtype=torch.float32,
            device=taxim.device,
        )
        self._zero_render = taxim.render_direct(
            zeros,
            with_shadow=False,
            press_depth=None,
            orig_hm_fmt=False,
        )[0].detach()
        # Keep the old sensor's low-frequency shading, but smooth away the
        # sharp calibration structure.  At rest raw == zero_render, so the
        # corrected image returns to this low-passed zero-height background.
        from torchvision.transforms.functional import (
            gaussian_blur,
        )

        zero_255 = self._zero_render * 255.0
        blurred_255 = gaussian_blur(
            zero_255.unsqueeze(0),
            kernel_size=[33, 33],
            sigma=[4.0, 4.0],
        )[0]
        self._flat_bg = (blurred_255 / 255.0).detach()

    @property
    def rest_background_image(self) -> Any:
        """Low-frequency rest background RGB image, in [0, 1]."""
        return self._flat_bg

    @property
    def rest_background_rgb(self) -> Any:
        """Mean RGB of the low-frequency rest background, in [0, 1]."""
        return self._flat_bg.mean(dim=(-2, -1))

    @classmethod
    def bind(
        cls,
        *,
        calib_dir: Path,
        visual: VisualSource,
        device: str = "cuda",
        height_gain: float = 1.0,
    ) -> TaximOptical:
        required = ("params.json", "polycalib.npz", "gelmap.npy", "dataPack.npz")
        missing = [name for name in required if not (calib_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Taxim calibration {calib_dir} is missing {missing}"
            )

        try:
            import torch

            from .taxim_core.taxim_torch import TaximTorch
        except ImportError as exc:
            raise RuntimeError(
                "Taxim optical backend requested but torch/torchvision or the "
                "vendored Taxim core is unavailable"
            ) from exc

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "Taxim optical backend requires CUDA; use "
                "optical_backend='procedural' explicitly for CPU testing"
            )

        try:
            taxim = TaximTorch(calib_folder=calib_dir, device=device)
        except Exception as exc:
            raise RuntimeError(f"failed to initialize Taxim from {calib_dir}") from exc

        return cls(visual=visual, taxim=taxim, torch=torch, height_gain=height_gain)

    @property
    def image_size(self) -> tuple[int, int]:
        return self._visual.image_size

    def aperture_mask(self) -> BoolArray:
        return self._visual.aperture_mask()

    def render(self, *, height_mm: FloatArray) -> TactileSensorImages:
        height = np.asarray(height_mm, dtype=np.float32)
        expected = (self.image_size[1], self.image_size[0])
        if height.shape != expected:
            raise ValueError(
                f"Taxim height map has shape {height.shape}, expected {expected}"
            )
        if not np.isfinite(height).all():
            raise ValueError("Taxim height map contains non-finite values")

        # Taxim default height-map format is distance from the far gel surface,
        # so positive indentation becomes a negative height.
        scaled = -height * self._height_gain
        height_t = self._torch.from_numpy(scaled[None]).to(self._taxim.device)
        rendered = self._taxim.render_direct(
            height_t,
            with_shadow=False,
            press_depth=None,
            orig_hm_fmt=False,
        )
        corrected = self._flat_bg + rendered[0] - self._zero_render
        rgb = (
            corrected.clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
            * 255.0
        )
        tactile: UInt8Array = np.rint(rgb).clip(0, 255).astype(np.uint8)
        marker = self._visual.marker_mask(np.asarray(height_mm, dtype=np.float64))
        marker_rgb = (
            tactile.astype(np.uint16) * marker.astype(np.uint16)[..., None] // 255
        ).astype(np.uint8)
        return TactileSensorImages(
            tactile_rgb=tactile,
            marker_rgb=marker_rgb,
            aperture_mask=self._visual.aperture_mask(),
        )


def build_optical_backend(
    *,
    backend: str,
    visual: VisualSource,
    calib_dir: Path | None = None,
    device: str = "cuda",
    height_gain: float = 1.0,
) -> OpticalBackend:
    if backend == "procedural":
        return ProceduralOptical(visual)
    if backend == "taxim":
        if calib_dir is None:
            raise FileNotFoundError(
                "Taxim calibration not found; set OPENVITAC_TAXIM_ASSET_DIR "
                "to a staged calib directory."
            )
        return TaximOptical.bind(
            calib_dir=calib_dir,
            visual=visual,
            device=device,
            height_gain=height_gain,
        )
    raise ValueError(f"unknown optical backend {backend!r}")
