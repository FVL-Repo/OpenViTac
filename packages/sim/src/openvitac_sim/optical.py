"""Tactile optical backends: Taxim (default) and procedural (explicit fallback).

Taxim consumes a camera-depth image rasterized from the live gel far/contact
surface, with the near sheet removed.  The image is turned into a rest-minus-live
height map (positive indentation) before Taxim is called; Taxim never falls back
to object depth or to the live UIPC mesh.  It only produces the optical RGB
image.  Marker positions and the printed dot mask stay in
:mod:`openvitac_sim.sensor_visual`; the adapter composes them exactly like the
old GelSightSensor did (``tactile_rgb * marker_mask / 255``).  The recorded
``contact_far`` depth stream itself is untouched.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from .assets import assets_root
from .sensor_visual import TactileSensorImages, TactileSensorVisual

BoolArray = npt.NDArray[np.bool_]
UInt8Array = npt.NDArray[np.uint8]


def taxim_height_map_mm(
    rest_depth_m: npt.NDArray[np.floating[Any]],
    live_depth_m: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.float32]:
    """Rest-to-live gel-surface height map in millimetres.

    ``rest_depth_m`` and ``live_depth_m`` are positive camera-to-gel-surface
    distances in metres.  Positive values mean the live surface moved toward
    the camera (indentation); Taxim receives ``-height_mm`` because its default
    height-map convention is distance behind/in front of the far gel surface.
    """
    rest = np.asarray(rest_depth_m, dtype=np.float32)
    live = np.asarray(live_depth_m, dtype=np.float32)
    if rest.shape != live.shape:
        raise ValueError(
            f"Taxim rest/live depth shapes differ: {rest.shape} vs {live.shape}"
        )
    if not np.isfinite(rest).all() or not np.isfinite(live).all():
        raise ValueError("Taxim rest/live depth contains non-finite values")
    return ((rest - live) * 1000.0).astype(np.float32)


class OpticalBackend(Protocol):
    """One pad's optical pass."""

    @property
    def image_size(self) -> tuple[int, int]: ...

    def render(
        self,
        world_vertices: npt.NDArray[np.floating[Any]],
        camera_pose: npt.NDArray[np.floating[Any]],
        *,
        camera_depth_m: npt.NDArray[np.floating[Any]] | None = None,
        rest_depth_m: npt.NDArray[np.floating[Any]] | None = None,
    ) -> TactileSensorImages: ...

    def aperture_mask(self) -> BoolArray: ...


class ProceduralOptical:
    """Existing analytic sensor pass, selected only by an explicit opt-in."""

    def __init__(self, visual: TactileSensorVisual) -> None:
        self._visual = visual

    @property
    def image_size(self) -> tuple[int, int]:
        return self._visual.image_size

    def render(
        self,
        world_vertices: npt.NDArray[np.floating[Any]],
        camera_pose: npt.NDArray[np.floating[Any]],
        *,
        camera_depth_m: npt.NDArray[np.floating[Any]] | None = None,
        rest_depth_m: npt.NDArray[np.floating[Any]] | None = None,
    ) -> TactileSensorImages:
        # The analytic pass derives geometry from the live mesh; camera/rest
        # depth are only meaningful to the Taxim backend.
        del camera_depth_m, rest_depth_m
        return self._visual.render(world_vertices, camera_pose)

    def aperture_mask(self) -> BoolArray:
        return self._visual.aperture_mask()


def _calib_dir_for_kind(kind: str) -> Path:
    """Resolve GsMini Taxim calibration from env or the repo asset tree.

    ``OPENVITAC_TAXIM_ASSET_DIR`` may point either at a directory containing
    ``gelsight_mini/640x480`` / ``gelsight_mini/calibs/640x480`` or directly at
    the calib directory itself (the staged tree uses ``calibs``).
    """
    if kind not in ("gsmini", "neote"):
        raise RuntimeError(
            f"Taxim optical backend is not implemented for sensor kind {kind!r}; "
            "use optical_backend='procedural' or add a calibrated Taxim adapter"
        )

    env_dir = os.environ.get("OPENVITAC_TAXIM_ASSET_DIR")
    env_root = Path(env_dir) if env_dir else None
    if env_root is not None:
        candidates = (
            env_root,
            env_root / "gelsight_mini" / "640x480",
            env_root / "gelsight_mini" / "calibs" / "640x480",
            env_root / "Sensors" / "GelSight_Mini" / "calibs" / "640x480",
        )
    else:
        candidates = (
            assets_root() / "tactile" / "taxim" / "gelsight_mini" / "640x480",
        )

    for candidate in candidates:
        if (candidate / "params.json").is_file():
            return candidate
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Taxim calibration not found; checked: "
        f"{checked}. Set OPENVITAC_TAXIM_ASSET_DIR to a staged calib directory."
    )


class TaximOptical:
    """Taxim RGB from the live gel-surface depth + marker lattice mask.

    The depth image is rasterized from the live far/contact gel surface through
    the authored camera; the same-pixel camera annotation is not used because an
    opaque object can occlude the gel.  A per-pad rest depth image is required;
    the adapter forms
    ``height_mm = (rest_depth_m - live_depth_m) * 1000`` (positive indentation)
    and passes ``-height_mm`` to Taxim with ``press_depth=None``.  This preserves
    spatial deformation without reconstructing an object surface.

    This is a low-compute approximation: the resulting RGB is visually close
    enough to real tactile deformation for the benchmark, but it is not
    physically correct.  A real gel low-passes and loses high-frequency contact
    detail.  The marker image still uses live vertices.

    TODO: for more realistic deformation, reduce gelpad thickness, tune material
    parameters, increase surface/tet resolution, or use an explicit
    gel-surface deformation model.
    """

    def __init__(
        self,
        visual: TactileSensorVisual,
        taxim: Any,
        torch: Any,
    ) -> None:
        self._visual = visual
        self._taxim = taxim
        self._torch = torch
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
        side: str,
        kind: str,
        rest_vertices: npt.NDArray[np.floating[Any]],
        triangles: npt.NDArray[np.integer[Any]],
        camera_pose: npt.NDArray[np.floating[Any]],
        camera_intrinsics: npt.NDArray[np.floating[Any]],
        image_size: tuple[int, int],
        marker_rows: int,
        marker_cols: int,
        device: str = "cuda",
        visual: TactileSensorVisual | None = None,
    ) -> TaximOptical:
        calib_dir = _calib_dir_for_kind(kind)
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

        if visual is None:
            visual = TactileSensorVisual.bind(
                side=side,
                rest_vertices=rest_vertices,
                triangles=triangles,
                camera_pose=camera_pose,
                camera_intrinsics=camera_intrinsics,
                image_size=image_size,
                marker_rows=marker_rows,
                marker_cols=marker_cols,
            )
        return cls(visual=visual, taxim=taxim, torch=torch)

    @property
    def image_size(self) -> tuple[int, int]:
        return self._visual.image_size

    def aperture_mask(self) -> BoolArray:
        return self._visual.aperture_mask()

    def render(
        self,
        world_vertices: npt.NDArray[np.floating[Any]],
        camera_pose: npt.NDArray[np.floating[Any]],
        *,
        camera_depth_m: npt.NDArray[np.floating[Any]] | None = None,
        rest_depth_m: npt.NDArray[np.floating[Any]] | None = None,
    ) -> TactileSensorImages:
        if camera_depth_m is None:
            raise RuntimeError(
                "Taxim optical backend requires live gel-surface camera depth; "
                "refusing to fall back to object or live-mesh height"
            )
        if rest_depth_m is None:
            raise RuntimeError(
                "Taxim optical backend requires the per-pad rest depth baseline; "
                "refusing to infer it from object depth"
            )
        depth_m = np.asarray(camera_depth_m, dtype=np.float32)
        rest_m = np.asarray(rest_depth_m, dtype=np.float32)
        expected = (self.image_size[1], self.image_size[0])
        for name, image in (("live", depth_m), ("rest", rest_m)):
            if image.shape != expected:
                raise ValueError(
                    f"Taxim {name} camera depth has shape {image.shape}, "
                    f"expected {expected}"
                )
            if not np.isfinite(image).all():
                raise ValueError(
                    f"Taxim {name} camera depth contains non-finite values"
                )

        height_mm = taxim_height_map_mm(rest_m, depth_m)
        # Taxim default height-map format is distance from the far gel surface,
        # so positive indentation becomes a negative height.
        height_t = self._torch.from_numpy(-height_mm[None]).to(self._taxim.device)
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
        marker = self._visual.marker_mask(world_vertices, camera_pose)
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
    side: str,
    kind: str,
    rest_vertices: npt.NDArray[np.floating[Any]],
    triangles: npt.NDArray[np.integer[Any]],
    camera_pose: npt.NDArray[np.floating[Any]],
    camera_intrinsics: npt.NDArray[np.floating[Any]],
    image_size: tuple[int, int],
    marker_rows: int,
    marker_cols: int,
    device: str = "cuda",
    visual: TactileSensorVisual | None = None,
) -> OpticalBackend:
    if visual is None:
        visual = TactileSensorVisual.bind(
            side=side,
            rest_vertices=rest_vertices,
            triangles=triangles,
            camera_pose=camera_pose,
            camera_intrinsics=camera_intrinsics,
            image_size=image_size,
            marker_rows=marker_rows,
            marker_cols=marker_cols,
        )
    if backend == "procedural":
        return ProceduralOptical(visual)
    if backend == "taxim":
        return TaximOptical.bind(
            side=side,
            kind=kind,
            rest_vertices=rest_vertices,
            triangles=triangles,
            camera_pose=camera_pose,
            camera_intrinsics=camera_intrinsics,
            image_size=image_size,
            marker_rows=marker_rows,
            marker_cols=marker_cols,
            device=device,
            visual=visual,
        )
    raise ValueError(f"unknown optical backend {backend!r}")
