"""GsMini/Neote tactile sensor: geometry, appearance policy, visual pass.

The sensor image is a function of the *contact object set* only.  Each declared
contact object's triangle mesh is z-buffered through the pad camera; the pad's
flat gel contact face is a fixed geometric plane in pad coordinates; and

    height = clip(rest_plane - object_depth, 0, +inf)

is the tactile measurement.  Normals are the gradient of that height field, the
optical backend renders from it, and the printed-dot lattice is displaced by it.
The simulated gel mesh still drives physics and the cosmetic display proxy, but
it is never an input to imaging: the gelpad is the sensor, not an object.

``GsMiniOptics`` owns every GsMini-specific number (plane, height and marker
gains, palette, marker layout, aperture and shading policy); the modules under
:mod:`openvitac_sim.imaging` only provide the mechanisms.  The same visual is
constructed by the offline replay from recorded calibration, so the renderer is
a pure function of recorded state.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFilter

from openvitac_spec.sensor import SensorBinding, StreamKind

from ..assets import assets_root
from ..imaging.height import (
    height_field,
    normals_from_height,
    plane_depth,
    plane_points,
    sample_bilinear,
)
from ..imaging.markers import _marker_image, marker_pixels_at
from ..imaging.objects import ObjectSurface, object_depth
from ..imaging.surface import BoolArray, FloatArray, _project
from ..imaging.types import TactileSensorImages
from ..mesh_io import load_usd_surface
from ..optical import OpticalBackend, build_optical_backend
from .base import (
    ContactObject,
    Handles,
    PadCameraRequest,
    ProxyHandle,
    SensorObservation,
    SoftBodyHandle,
    World,
    WorldBuilder,
)
from .ids import BodyId, CameraId
from .snapshot import Requires, TickSnapshot


@dataclass(frozen=True)
class GsMiniOptics:
    """Every GsMini-specific geometry and appearance number, in one place.

    Geometry (the tunable parameter surface):

    ``plane_height_m``
        Height of the flat gel contact face plane in pad coordinates, ``z =``
        this value.  Default 4.0 mm, the authored gelpad contact face.
    ``plane_offset_m``
        Signed offset added to ``plane_height_m``; positive pushes the plane
        away from the camera, so a contact images deeper (more height).
    ``height_gain``
        Multiplier from measured height to the height the optical model
        receives; 1.0 is physical.
    ``face_half_x_m`` / ``face_half_y_m``
        Half extents of the pad's full optical face in pad coordinates, taken
        from the pad asset (a constant, not a mesh silhouette); this rectangle
        *is* the imaged area.  The whole rectangle is the single
        height-reference plane, flat top and bevel alike.
    ``aperture_inset_px``
        Erosion of that rectangle in pixels; 0 images the whole face.
    ``marker_gain``
        Marker dots move along the pad optical axis by
        ``marker_gain * h``; 1.0 is physical, larger reads as a softer gel.

    Appearance:

    ``background_gain``
        Multiplier on the procedural gel background RGB.
    ``shade_blur_sigma``
        Gaussian sigma (pixels) applied to the shading plane.
    ``marker_radius_px``
        Printed dot radius at the 240 px reference height.
    ``marker_target_width_px`` / ``marker_target_height_px``
        Printed-dot extent on the 320x240 reference frame, before the two dot
        radii are removed; ``marker_margin_x_px`` / ``marker_margin_y_px`` are
        the pre-refactor fallback margins and ``marker_roi_margin_px`` the
        erosion that keeps the lattice box inside the imaged face.
    """

    # Rest plane and height mapping.
    plane_height_m: float = 0.004
    plane_offset_m: float = 0.0
    face_half_x_m: float = 0.012625
    face_half_y_m: float = 0.010375
    height_gain: float = 1.0
    aperture_inset_px: int = 0
    marker_gain: float = 1.0

    # Marker layout, authored in the reference 320x240 sensor frame (the
    # numbers are the pre-refactor layout, cfc6da0).
    marker_reference_width_px: float = 320.0
    marker_reference_height_px: float = 240.0
    marker_margin_x_px: float = 44.0
    marker_margin_y_px: float = 35.0
    marker_target_width_px: float = 232.0
    marker_target_height_px: float = 170.0
    marker_min_radius_px: float = 2.0
    marker_radius_px: float = 6.2
    marker_radius_reference_px: float = 240.0
    marker_roi_margin_px: int = 2

    # Aperture and shading policy.
    aperture_blur_sigma: float = 2.0
    shade_blur_sigma: float = 2.0
    contact_ramp_m: float = 0.0008
    contact_gain: float = 0.50
    normal_gain: float = 0.08

    # Gel appearance.
    background_gain: float = 1.0
    teal_rgb: tuple[float, float, float] = (24.0, 128.0, 132.0)
    purple_rgb: tuple[float, float, float] = (112.0, 58.0, 128.0)
    mix_u: float = 0.62
    mix_v: float = 0.38
    vignette_gain: float = 0.12
    vignette_scale: float = 4.0
    noise_sigma: float = 1.7
    left_seed: int = 0
    right_seed: int = 1

    def plane_height(self) -> float:
        """Contact-face plane height in pad coordinates, metres."""
        return self.plane_height_m + self.plane_offset_m

    def marker_radius(self, image_size: tuple[int, int]) -> float:
        width, height = image_size
        return max(
            self.marker_min_radius_px,
            self.marker_radius_px
            * min(width, height)
            / self.marker_radius_reference_px,
        )

    def marker_target(
        self, image_size: tuple[int, int], radius: float
    ) -> tuple[float, float]:
        """Printed-dot extent on this image, the two dot radii removed."""
        width, height = image_size
        return (
            self.marker_target_width_px * width / self.marker_reference_width_px
            - 2.0 * radius,
            self.marker_target_height_px * height / self.marker_reference_height_px
            - 2.0 * radius,
        )

    def marker_fallback_margins(
        self, image_size: tuple[int, int]
    ) -> tuple[float, float]:
        """Pixel margins of the pre-refactor layout, on this image."""
        width, height = image_size
        return (
            self.marker_margin_x_px * width / self.marker_reference_width_px,
            self.marker_margin_y_px * height / self.marker_reference_height_px,
        )

    def face_corners_pad(self) -> FloatArray:
        """Counter-clockwise contact-face rectangle in pad coordinates."""
        hx, hy, z = self.face_half_x_m, self.face_half_y_m, self.plane_height()
        return np.array(
            [[-hx, -hy, z], [hx, -hy, z], [hx, hy, z], [-hx, hy, z]],
            dtype=np.float64,
        )

    def background_seed(self, side: str) -> int:
        return self.left_seed if side == "left" else self.right_seed

    def taxim_calib_dir(self) -> Path | None:
        """Resolve the Taxim calibration directory, or ``None`` if absent.

        ``OPENVITAC_TAXIM_ASSET_DIR`` may point either at a directory
        containing ``gelsight_mini/640x480`` / ``gelsight_mini/calibs/640x480``
        or directly at the calib directory itself (the staged tree uses
        ``calibs``).
        """
        env_dir = os.environ.get("OPENVITAC_TAXIM_ASSET_DIR")
        if env_dir:
            env_root = Path(env_dir)
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
        return None

    def aperture(
        self,
        corners_px: NDArray[np.floating[Any]],
        image_size: tuple[int, int],
    ) -> tuple[BoolArray, FloatArray]:
        """The projected contact-face rectangle, plus feathered alpha.

        The quad comes from the pad's face-plane corners, never from a mesh
        silhouette, so no bevel or frustum edge can shape the imaged area.
        """
        width, height = image_size
        canvas = Image.new("L", (width, height), 0)
        ImageDraw.Draw(canvas).polygon(
            [(float(u), float(v)) for u, v in np.asarray(corners_px)], fill=255
        )
        aperture = np.asarray(canvas, dtype=np.uint8) > 0
        if not bool(aperture.any()):
            raise RuntimeError("pad face rectangle projects outside the image")
        if self.aperture_inset_px > 0:
            size = 2 * self.aperture_inset_px + 1
            eroded = Image.fromarray(
                np.where(aperture, 255, 0).astype(np.uint8)
            ).filter(ImageFilter.MinFilter(size))
            aperture = np.asarray(eroded, dtype=np.uint8) > 0
        alpha = np.asarray(
            Image.fromarray(np.where(aperture, 255, 0).astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(self.aperture_blur_sigma)
            ),
            dtype=np.float64,
        ) / 255.0
        return aperture, alpha

    def background(
        self, image_size: tuple[int, int], side: str
    ) -> NDArray[np.uint8]:
        """Procedural gel background; the seed is fixed per pad side."""
        width, height = image_size
        ys, xs = np.mgrid[0:height, 0:width]
        u = xs.astype(np.float64) / max(1, width - 1)
        v = ys.astype(np.float64) / max(1, height - 1)
        teal = np.array(self.teal_rgb)
        purple = np.array(self.purple_rgb)
        mix = np.clip(self.mix_u * u + self.mix_v * v, 0.0, 1.0)[..., None]
        base = teal * (1.0 - mix) + purple * mix
        radius = (u - 0.5) ** 2 + (v - 0.5) ** 2
        vignette = 1.0 - self.vignette_gain * np.clip(
            radius * self.vignette_scale, 0.0, 1.0
        )
        base *= vignette[..., None]
        base *= self.background_gain
        rng = np.random.default_rng(self.background_seed(side))
        noise = rng.normal(0.0, self.noise_sigma, size=(height, width, 1))
        return np.clip(base + noise, 0.0, 255.0).astype(np.uint8)

    def surface_shading(
        self,
        height_mm: FloatArray,
        rest_plane_depth: FloatArray,
        aperture: BoolArray,
        aperture_alpha: FloatArray,
        intrinsics: NDArray[np.floating[Any]],
    ) -> FloatArray:
        """Contact and tilt shading derived only from the height field.

        Contact darkens with the penetration depth; tilt darkens where the
        height-field normal leaves the optical axis.  A zero height field gives
        exactly unity, so a contact-free frame is the rest appearance.
        """
        height, width = height_mm.shape
        shade = np.ones((height, width), dtype=np.float64)
        if not bool(aperture.any()):
            return shade
        contact = np.clip(
            np.asarray(height_mm, dtype=np.float64) / (self.contact_ramp_m * 1e3),
            0.0,
            1.0,
        )
        normals = normals_from_height(
            np.asarray(height_mm, dtype=np.float64) * 1e-3,
            rest_plane_depth,
            intrinsics,
        )
        tilt = np.clip(1.0 - normals[..., 2], 0.0, 1.0)
        shade[aperture] = 1.0 - aperture_alpha[aperture] * (
            self.contact_gain * contact[aperture]
            + self.normal_gain * tilt[aperture]
        )
        return shade


GSMINI_OPTICS = GsMiniOptics()


@dataclass(frozen=True)
class TactileSensorVisual:
    """One pad's geometry and appearance: object meshes in, sensor images out."""

    side: str
    image_width: int
    image_height: int
    intrinsics: FloatArray
    camera_from_pad: FloatArray
    objects: tuple[ObjectSurface, ...]
    background: NDArray[np.uint8]
    aperture: BoolArray
    aperture_alpha: FloatArray
    rest_plane_normal: FloatArray
    rest_plane_offset: float
    rest_plane_depth: FloatArray
    marker_pixels: FloatArray
    marker_rest_points: FloatArray
    marker_radius: float
    marker_rows: int
    marker_cols: int
    optics: GsMiniOptics

    @classmethod
    def bind(
        cls,
        *,
        side: str,
        rest_vertices: NDArray[np.floating[Any]],
        camera_from_pad: NDArray[np.floating[Any]],
        camera_intrinsics: NDArray[np.floating[Any]],
        image_size: tuple[int, int],
        objects: Sequence[ObjectSurface] = (),
        marker_rows: int = 7,
        marker_cols: int = 9,
        optics: GsMiniOptics = GSMINI_OPTICS,
    ) -> TactileSensorVisual:
        """Calibrate the pass from the pad's face plane and its camera.

        The imaged area is the flat contact face itself: a rectangle at
        ``optics.plane_height()`` in pad coordinates, projected through the
        camera.  No mesh silhouette, bevel or rasterised gel footprint enters
        the aperture or the marker layout.  ``rest_vertices`` is used only to
        assert that the calibrated plane really is the gel contact face.
        """
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError("sensor image size must be positive")
        K = np.asarray(camera_intrinsics, dtype=np.float64).reshape(3, 3)
        C = np.asarray(camera_from_pad, dtype=np.float64).reshape(4, 4)
        identity = np.eye(4, dtype=np.float64)
        normal, offset = _rest_plane(optics, C)
        corners_px, _depth = _project(_face_corners_camera(optics, C), identity, K)
        aperture, aperture_alpha = optics.aperture(corners_px, image_size)
        rest_plane_depth = plane_depth(normal, offset, K, image_size)
        _assert_plane_matches_face(
            np.asarray(rest_vertices, dtype=np.float64),
            C,
            K,
            aperture,
            rest_plane_depth,
        )
        radius = optics.marker_radius(image_size)
        marker_px = _lattice_pixels(
            aperture,
            optics.marker_target(image_size, radius),
            optics.marker_roi_margin_px,
            marker_rows,
            marker_cols,
        )
        marker_camera = plane_points(marker_px, normal, offset, K)
        return cls(
            side=side,
            image_width=width,
            image_height=height,
            intrinsics=K,
            camera_from_pad=C,
            objects=tuple(objects),
            background=optics.background(image_size, side),
            aperture=aperture,
            aperture_alpha=aperture_alpha,
            rest_plane_normal=normal,
            rest_plane_offset=offset,
            rest_plane_depth=rest_plane_depth,
            marker_pixels=marker_px,
            marker_rest_points=marker_camera,
            marker_radius=radius,
            marker_rows=marker_rows,
            marker_cols=marker_cols,
            optics=optics,
        )

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.image_width, self.image_height)

    def aperture_mask(self) -> BoolArray:
        """Fixed rest aperture, eroded at bind; never follows live contact."""
        return self.aperture

    def object_depth(
        self,
        poses: Mapping[str, NDArray[np.floating[Any]]],
        camera_pose: NDArray[np.floating[Any]],
    ) -> FloatArray:
        """Nearest declared-object depth through this pad's camera, metres.

        Surfaces wholly behind the rest plane are culled: they can neither
        produce positive height nor occlude a surface that does.
        """
        return object_depth(
            self.objects,
            poses,
            np.asarray(camera_pose, dtype=np.float64).reshape(4, 4),
            self.intrinsics,
            self.image_size,
            max_depth=float(self.rest_plane_depth[self.aperture].max()),
        )

    def height_map_mm(self, depth: FloatArray) -> FloatArray:
        """Contact height in millimetres from an object depth image."""
        height_m = height_field(self.rest_plane_depth, depth, self.aperture)
        return height_m * 1e3

    def marker_positions(self, height_map_mm: FloatArray) -> FloatArray:
        """Printed-dot pixel positions after the height field moves them."""
        lift = (
            sample_bilinear(
                height_map_mm,
                self.marker_pixels[:, 0],
                self.marker_pixels[:, 1],
            )
            * 1e-3
            * self.optics.marker_gain
        )
        pixels, _visible = marker_pixels_at(
            self.marker_rest_points, lift, self.intrinsics
        )
        return pixels

    def marker_mask(self, height_map_mm: FloatArray) -> NDArray[np.uint8]:
        """Black printed-dot keep-mask, displaced by the height field."""
        return _marker_image(
            self.marker_positions(height_map_mm),
            self.image_size,
            radius=self.marker_radius,
        )

    def render(self, height_map_mm: FloatArray) -> TactileSensorImages:
        """Render one tactile/marker frame from the height field alone."""
        shade = self.optics.surface_shading(
            height_map_mm,
            self.rest_plane_depth,
            self.aperture,
            self.aperture_alpha,
            self.intrinsics,
        )
        shade_image = Image.fromarray(
            np.clip(shade * 255.0, 0.0, 255.0).astype(np.uint8)
        )
        blurred = shade_image.filter(
            ImageFilter.GaussianBlur(self.optics.shade_blur_sigma)
        )
        shade = np.asarray(blurred, dtype=np.float64) / 255.0
        tactile = np.clip(
            self.background.astype(np.float64) * shade[..., None],
            0.0,
            255.0,
        ).astype(np.uint8)
        marker = self.marker_mask(height_map_mm)
        marker_rgb = (
            tactile.astype(np.uint16) * marker.astype(np.uint16)[..., None] // 255
        ).astype(np.uint8)
        return TactileSensorImages(
            tactile_rgb=tactile,
            marker_rgb=marker_rgb,
            aperture_mask=self.aperture,
        )


def _face_corners_camera(
    optics: GsMiniOptics, camera_from_pad: NDArray[np.floating[Any]]
) -> FloatArray:
    """Contact-face rectangle corners in camera coordinates."""
    C = np.asarray(camera_from_pad, dtype=np.float64).reshape(4, 4)
    corners = optics.face_corners_pad()
    return corners @ C[:3, :3].T + C[:3, 3]


def _marker_roi_bounds(
    footprint: BoolArray,
    target: tuple[float, float],
    margin: int,
) -> tuple[float, float, float, float]:
    """Largest target-aspect rectangle centred on the image inside ``footprint``.

    Same search as the pre-refactor layout (cfc6da0 ``_roi_grid_bounds``), but
    the footprint is the pad face rectangle rather than a rasterised gel
    silhouette: the requested extent is used whenever it fits.
    """
    height, width = footprint.shape
    eroded = np.asarray(
        Image.fromarray(np.where(footprint, 255, 0).astype(np.uint8)).filter(
            ImageFilter.MinFilter(2 * margin + 1)
        ),
        dtype=np.uint8,
    ) > 0
    centre_x = (width - 1) / 2.0
    centre_y = (height - 1) / 2.0
    for scale in range(100, 0, -1):
        box_width = max(1, round(target[0] * scale / 100.0))
        box_height = max(1, round(target[1] * scale / 100.0))
        x0 = round(centre_x - (box_width - 1) / 2.0)
        y0 = round(centre_y - (box_height - 1) / 2.0)
        x1 = x0 + box_width - 1
        y1 = y0 + box_height - 1
        if x0 < 0 or y0 < 0 or x1 >= width or y1 >= height:
            continue
        if bool(eroded[y0 : y1 + 1, x0 : x1 + 1].all()):
            return float(x0), float(x1), float(y0), float(y1)
    raise RuntimeError(
        "the pad face rectangle cannot contain the marker lattice; check "
        "marker_target_width_px/marker_target_height_px and face_half_x_m"
    )


def _lattice_pixels(
    footprint: BoolArray,
    target: tuple[float, float],
    margin: int,
    rows: int,
    cols: int,
) -> FloatArray:
    """Printed-dot centres in image pixels, on the pre-refactor lattice."""
    if rows <= 0 or cols <= 0:
        raise ValueError("marker lattice dimensions must be positive")
    x0, x1, y0, y1 = _marker_roi_bounds(footprint, target, margin)
    xs = np.linspace(x0, x1, cols)
    ys = np.linspace(y0, y1, rows)
    return np.asarray([(x, y) for y in ys for x in xs], dtype=np.float64)


def _assert_plane_matches_face(
    rest_vertices: NDArray[np.floating[Any]],
    camera_from_pad: NDArray[np.floating[Any]],
    intrinsics: NDArray[np.floating[Any]],
    aperture: BoolArray,
    rest_plane_depth: FloatArray,
) -> None:
    """Check that the calibrated plane is the gel's actual contact face.

    Calibration only: the rest surface never contributes geometry.
    """
    C = np.asarray(camera_from_pad, dtype=np.float64).reshape(4, 4)
    points = rest_vertices @ C[:3, :3].T + C[:3, 3]
    uv, depth = _project(points, np.eye(4), intrinsics)
    height, width = aperture.shape
    valid = np.isfinite(uv).all(axis=1) & (depth > 0.0)
    u = np.clip(np.round(np.where(valid, uv[:, 0], 0.0)).astype(int), 0, width - 1)
    v = np.clip(np.round(np.where(valid, uv[:, 1], 0.0)).astype(int), 0, height - 1)
    valid &= aperture[v, u]
    if not bool(valid.any()):
        raise RuntimeError(
            "no gel rest vertex projects inside the contact-face rectangle; "
            "check face_half_x_m/face_half_y_m and camera_from_pad"
        )
    # The camera sits behind the pad, so the optical face is the farthest
    # rest surface inside the imaged rectangle (the body's bottom is nearer).
    residual = depth[valid] - rest_plane_depth[v[valid], u[valid]]
    face = float(residual.max())
    if abs(face) > 1.5e-3:
        raise RuntimeError(
            "calibrated pad plane is not the gel's contact face: the farthest "
            f"rest surface is {face * 1e3:+.2f} mm off it; check "
            "plane_height_m/plane_offset_m and face_half_x_m/face_half_y_m"
        )


def _rest_plane(
    optics: GsMiniOptics, camera_from_pad: NDArray[np.floating[Any]]
) -> tuple[FloatArray, float]:
    """The gel contact-face plane, in camera coordinates.

    The plane is ``z = plane_height()`` in pad coordinates with the pad ``+z``
    axis pointing at the contact surface.  Expressing it in the camera frame
    makes it constant for the pad's life; it is a geometric plane, never a
    captured rest image.
    """
    C = np.asarray(camera_from_pad, dtype=np.float64).reshape(4, 4)
    height = optics.plane_height()
    point = C @ np.array([0.0, 0.0, height, 1.0], dtype=np.float64)
    normal = C[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        raise ValueError("camera-to-pad calibration has a degenerate rotation")
    normal /= norm
    # Point the normal from the plane towards the camera at the origin.
    if float(normal @ point[:3]) > 0.0:
        normal = -normal
    offset = float(normal @ point[:3])
    if offset >= 0.0:
        raise ValueError(
            "pad rest plane lies behind the pad camera; check that pad +z "
            "points from the gel back to the contact face"
        )
    return normal, offset


@dataclass(frozen=True)
class GsMiniObservation:
    """One GsMini frame: only the streams this pad actually produced."""

    rgb: NDArray[np.uint8] | None = None
    marker_rgb: NDArray[np.uint8] | None = None
    depth: NDArray[np.float32] | None = None
    height_mm: NDArray[np.float64] | None = None

    def streams(self) -> Mapping[StreamKind, NDArray[Any]]:
        streams: dict[StreamKind, NDArray[Any]] = {}
        if self.rgb is not None:
            streams["rgb"] = self.rgb
        if self.marker_rgb is not None:
            streams["marker_rgb"] = self.marker_rgb
        if self.depth is not None:
            streams["depth"] = self.depth
        return streams

    def extras(self) -> Mapping[str, Any]:
        if self.height_mm is None:
            return {}
        return {"height_mm": self.height_mm}


@dataclass
class GsMiniSensor:
    """A GsMini/Neote pad: owns its camera, proxy, soft body and object set."""

    binding: SensorBinding
    pad_key: str
    optical_backend: str
    taxim_device: str
    optics: GsMiniOptics = GSMINI_OPTICS
    finger_body: str | None = None
    _handles: Handles | None = field(default=None, init=False, repr=False)

    pad_proxy: ClassVar[bool] = True

    @property
    def name(self) -> str:
        return self.pad_key

    @property
    def outputs(self) -> frozenset[StreamKind]:
        return frozenset(self.binding.streams)

    def _records_depth(self) -> bool:
        return "depth" in self.binding.streams

    def _camera_prim_path(self) -> str:
        camera = self.binding.mount.camera_prim
        if camera is not None:
            return camera
        return f"{self.binding.body_prim_path}/Camera"

    def requires(self) -> Requires:
        handles = self._handles
        if handles is None:
            raise RuntimeError(f"sensor {self.pad_key!r} was not attached")
        if handles.camera is None:
            raise RuntimeError(f"sensor {self.pad_key!r} has no camera handle")
        bodies = {BodyId(self.binding.body_name)}
        if self.finger_body is not None:
            bodies.add(BodyId(self.finger_body))
        return Requires(
            bodies=frozenset(bodies),
            deformables=frozenset({handles.soft_body.id}),
            cameras=frozenset({handles.camera}),
            objects=frozenset(obj.id for obj in handles.objects),
        )

    def attach(self, builder: WorldBuilder) -> Handles:
        """Create this pad's render gear and resolve the contact object set."""
        body = builder.body(self.binding.body_name)
        if body is None:
            raise RuntimeError(
                f"tactile pad body {self.binding.body_name!r} is not in the scene"
            )
        soft_body = builder.add_soft_body(self.pad_key)
        proxy = builder.add_visual_proxy(self.binding.gelpad_prim_path)
        camera = builder.add_pad_camera(
            self._camera_prim_path(),
            PadCameraRequest(
                side=self.pad_key,
                width=self.binding.sensor.width,
                height=self.binding.sensor.height,
            ),
        )
        builder.remove_pad_proxy(body)
        objects = builder.contact_objects()
        self._reject_sensor_geometry(objects)
        handles = Handles(
            body=body,
            soft_body=soft_body,
            camera=camera,
            proxy=proxy,
            objects=objects,
        )
        self._handles = handles
        return handles

    def _reject_sensor_geometry(self, objects: Sequence[ContactObject]) -> None:
        """Assert no declared contact object is owned by any sensor."""
        owned = (
            self.binding.body_prim_path,
            self.binding.case_prim_path,
            self.binding.gelpad_prim_path,
        )
        for obj in objects:
            for prim in owned:
                if obj.prim_path == prim or obj.prim_path.startswith(f"{prim}/"):
                    raise RuntimeError(
                        f"sensor {self.pad_key!r} resolved {obj.prim_path!r} as a "
                        f"contact object, but it is the sensor's own prim {prim!r}; "
                        "the gelpad/case is the sensor, never an object"
                    )

    def bind(self, world: World) -> BoundGsMiniSensor:
        """Calibrate the rest surface, plane, objects, optics and markers."""
        handles = self._handles
        if handles is None:
            raise RuntimeError(f"tactile pad {self.pad_key!r} was not attached")
        sensor = self.binding.sensor
        side = self.binding.pad_key
        rest_world, triangles = world.pad_surface(handles.soft_body.id)
        pad_frame = world.pad_frame(handles.soft_body.id)
        camera_pose = world.rest_camera_pose(handles.soft_body.id)
        camera_intrinsics = world.camera_intrinsics(handles.soft_body.id)
        camera_from_pad = np.linalg.inv(camera_pose) @ pad_frame
        pad_from_world = np.linalg.inv(pad_frame)
        rest_pad = (
            np.asarray(rest_world, dtype=np.float64) @ pad_from_world[:3, :3].T
            + pad_from_world[:3, 3]
        )
        objects = []
        for obj in handles.objects:
            vertices, object_triangles = load_usd_surface(obj.usd_path)
            objects.append(
                ObjectSurface.from_mesh(
                    body=str(obj.id),
                    prim_path=obj.prim_path,
                    vertices=vertices,
                    triangles=object_triangles,
                )
            )
        visual = TactileSensorVisual.bind(
            side=side,
            rest_vertices=rest_pad,
            camera_from_pad=camera_from_pad,
            camera_intrinsics=camera_intrinsics,
            image_size=(sensor.width, sensor.height),
            objects=objects,
            marker_rows=sensor.marker_rows or 7,
            marker_cols=sensor.marker_cols or 9,
            optics=self.optics,
        )
        calib_dir = (
            self.optics.taxim_calib_dir()
            if self.optical_backend == "taxim"
            else None
        )
        backend = build_optical_backend(
            backend=self.optical_backend,
            visual=visual,
            calib_dir=calib_dir,
            device=self.taxim_device,
            height_gain=self.optics.height_gain,
        )
        handles.proxy.update(rest_world, triangles)
        return BoundGsMiniSensor(
            pad_key=self.pad_key,
            streams=frozenset(self.binding.streams),
            visual=visual,
            backend=backend,
            records_depth=self._records_depth(),
            proxy=handles.proxy,
            camera=handles.camera,
            finger_body=BodyId(self.finger_body or self.binding.body_name),
            soft_body=handles.soft_body,
        )


@dataclass(frozen=True)
class BoundGsMiniSensor:
    """One calibrated pad: object poses in, sensor streams out."""

    pad_key: str
    streams: frozenset[StreamKind]
    visual: TactileSensorVisual
    backend: OpticalBackend
    records_depth: bool
    proxy: ProxyHandle
    camera: CameraId | None
    finger_body: BodyId
    soft_body: SoftBodyHandle

    def aperture_mask(self) -> NDArray[np.bool_]:
        return self.backend.aperture_mask()

    def sync(self, snap: TickSnapshot) -> None:
        """Stage this pad's constraint for the next exchange."""
        self.soft_body.stage(snap.pose(self.finger_body))

    def publish(self, snap: TickSnapshot) -> None:
        """Publish this tick's live surface into the display mesh."""
        deformable = snap.deformable(self.soft_body.id)
        self.proxy.update(deformable.vertices, deformable.triangles)

    def observe(self, snap: TickSnapshot) -> SensorObservation:
        """Image this tick's declared objects through this tick's camera."""
        if self.camera is None:
            raise RuntimeError(
                f"tactile pad {self.pad_key!r} has no camera handle"
            )
        camera_pose = snap.camera(self.camera).matrix
        poses = {
            obj.body: snap.pose(BodyId(obj.body)).matrix
            for obj in self.visual.objects
        }
        depth = self.visual.object_depth(poses, camera_pose)
        height_mm = self.visual.height_map_mm(depth)
        frame = self.backend.render(height_mm=height_mm)
        expected = (self.backend.image_size[1], self.backend.image_size[0], 3)
        rgb_out: NDArray[np.uint8] | None = None
        marker_out: NDArray[np.uint8] | None = None
        depth_out: NDArray[np.float32] | None = None
        if "rgb" in self.streams:
            if frame.tactile_rgb.shape != expected:
                raise RuntimeError(
                    f"optical sensor {self.pad_key!r} returned "
                    f"{frame.tactile_rgb.shape}, expected {expected}"
                )
            rgb_out = frame.tactile_rgb
        if "marker_rgb" in self.streams:
            if frame.marker_rgb.shape != expected:
                raise RuntimeError(
                    f"marker sensor {self.pad_key!r} returned "
                    f"{frame.marker_rgb.shape}, expected {expected}"
                )
            marker_out = frame.marker_rgb
        if self.records_depth:
            visible = self.visual.aperture & np.isfinite(depth)
            depth_out = np.where(visible, depth, 0.0).astype(np.float32)
        return GsMiniObservation(
            rgb=rgb_out,
            marker_rgb=marker_out,
            depth=depth_out,
            height_mm=height_mm,
        )
