"""Independent optical tactile sensor pass.

The UIPC gelpad proxy is a separate, closed black mesh.  This module turns the
live far/contact-sheet vertices into a synthetic 2-D sensor image from an authored
camera calibration, without rendering the Isaac scene:

* a teal/purple gel background with light fixed pattern noise and
  depth/normal-based contact shading;
* a fixed 9-column x 7-row printed-dot lattice attached to rest triangles by
  barycentric coordinates, so dots move with the simulated far surface;
* ``marker_rgb = tactile_rgb * marker_mask``.

The caller owns camera calibration and supplies the current world <- camera
pose every frame.  The gel background below is procedural, not a calibrated
sensor response.  Nothing here changes physics or the visual proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFilter

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class TactileSensorImages:
    """One independent sensor frame, plus the fixed rest aperture mask."""

    tactile_rgb: NDArray[np.uint8]
    marker_rgb: NDArray[np.uint8]
    aperture_mask: BoolArray


@dataclass(frozen=True)
class _Raster:
    mask: BoolArray
    depth: NDArray[np.float32]
    normals: FloatArray


def _face_adjacency(
    triangle_count: int, triangles: NDArray[np.integer[Any]]
) -> list[set[int]]:
    """Face adjacency by shared mesh edges."""
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face, tri in enumerate(triangles):
        a, b, c = (int(v) for v in tri)
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces.setdefault((min(u, v), max(u, v)), []).append(face)
    adjacency: list[set[int]] = [set() for _ in range(triangle_count)]
    for faces in edge_faces.values():
        for i, face in enumerate(faces):
            for other in faces[i + 1 :]:
                adjacency[face].add(other)
                adjacency[other].add(face)
    return adjacency


def select_far_sheet(
    vertices: NDArray[np.floating[Any]],
    triangles: NDArray[np.integer[Any]],
    camera_position: NDArray[np.floating[Any]],
) -> NDArray[np.int32]:
    """Return the connected camera-facing far/contact-sheet triangle indices.

    Contact pushes the far sheet inward, so this uses normals and connectivity
    rather than a fixed depth slab.  The seed is the camera-facing triangle
    farthest from the camera; its connected, roughly parallel sheet is kept.
    """
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    camera = np.asarray(camera_position, dtype=np.float64).reshape(3)
    corners = points[tris]
    centroids = corners.mean(axis=1)
    normals = np.cross(
        corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    )
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("gelpad surface has degenerate triangles")
    to_camera = camera - centroids
    distance = np.linalg.norm(to_camera, axis=1)
    if np.any(distance <= 0.0):
        raise ValueError("camera lies on the gelpad surface")
    unit_normals = normals / lengths[:, None]
    alignment = np.abs(
        np.einsum("ij,ij->i", unit_normals, to_camera / distance[:, None])
    )
    facing = alignment >= 0.5
    if not bool(facing.any()):
        raise ValueError("gelpad surface has no camera-facing triangles")
    seed = int(np.argmax(np.where(facing, distance, -np.inf)))
    seed_normal = unit_normals[seed]
    parallel = np.abs(unit_normals @ seed_normal) >= 0.9
    orient = np.where(normals @ seed_normal >= 0.0, 1.0, -1.0)
    offset = (
        np.einsum("ij,ij->i", unit_normals, centroids - centroids[seed])
        * orient
    )
    candidate = facing & parallel & (np.abs(offset) <= 0.0035)
    if not bool(candidate[seed]):
        raise ValueError("gelpad camera-facing seed is not a valid sheet")
    adjacency = _face_adjacency(len(tris), tris)
    mask = np.zeros(len(tris), dtype=np.bool_)
    stack = [seed]
    mask[seed] = True
    while stack:
        face = stack.pop()
        for other in adjacency[face]:
            if candidate[other] and not mask[other]:
                mask[other] = True
                stack.append(other)
    if not bool(mask.any()):
        raise ValueError("gelpad surface has no visible connected sheet")
    return np.flatnonzero(mask).astype(np.int32)


def _camera_facing_triangles(
    vertices: NDArray[np.floating[Any]],
    triangles: NDArray[np.integer[Any]],
    camera_position: NDArray[np.floating[Any]],
) -> NDArray[np.int32]:
    """Rest triangles whose outward normal faces the camera.

    This includes the shallow bevel around the gel top while excluding the
    back/bottom faces of the closed proxy.  Marker layout uses this surface,
    not the narrower parallel contact sheet.
    """
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    camera = np.asarray(camera_position, dtype=np.float64).reshape(3)
    corners = points[tris]
    centroids = corners.mean(axis=1)
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 0.0):
        return np.zeros((0, 3), dtype=np.int32)
    to_camera = camera - centroids
    distance = np.linalg.norm(to_camera, axis=1)
    valid = distance > 0.0
    alignment = np.zeros(len(tris), dtype=np.float64)
    alignment[valid] = np.einsum(
        "ij,ij->i",
        normals[valid] / lengths[valid, None],
        to_camera[valid] / distance[valid, None],
    )
    return tris[alignment > 0.1].astype(np.int32)


def _project(
    points: NDArray[np.floating[Any]],
    camera_pose: NDArray[np.floating[Any]],
    intrinsics: NDArray[np.floating[Any]],
) -> tuple[FloatArray, FloatArray]:
    """Project world points to pixels and positive optical-axis depth."""
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    T = np.asarray(camera_pose, dtype=np.float64).reshape(4, 4)
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    inverse = np.linalg.inv(T)
    camera_points = xyz @ inverse[:3, :3].T + inverse[:3, 3]
    depth = -camera_points[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * camera_points[:, 0] / depth + K[0, 2]
        v = K[1, 1] * (-camera_points[:, 1]) / depth + K[1, 2]
    return np.stack((u, v), axis=1), depth


def _rasterize(
    vertices: NDArray[np.floating[Any]],
    triangles: NDArray[np.integer[Any]],
    camera_pose: NDArray[np.floating[Any]],
    intrinsics: NDArray[np.floating[Any]],
    image_size: tuple[int, int],
) -> _Raster:
    """Rasterize a triangle subset to a nearest depth/normal map."""
    width, height = image_size
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    uv, depths = _project(points, camera_pose, intrinsics)
    mask = np.zeros((height, width), dtype=np.bool_)
    depth = np.full((height, width), np.inf, dtype=np.float64)
    normals = np.zeros((height, width, 3), dtype=np.float64)
    for tri in tris:
        if np.any(depths[tri] <= 1e-9):
            continue
        corners = points[tri]
        normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        norm = float(np.linalg.norm(normal))
        if norm <= 0.0:
            continue
        normal = normal / norm
        x = uv[tri, 0]
        y = uv[tri, 1]
        x_min = max(0, int(np.floor(x.min())))
        x_max = min(width - 1, int(np.ceil(x.max())))
        y_min = max(0, int(np.floor(y.min())))
        y_max = min(height - 1, int(np.ceil(y.max())))
        if x_min > x_max or y_min > y_max:
            continue
        xs, ys = np.meshgrid(
            np.arange(x_min, x_max + 1, dtype=np.float64),
            np.arange(y_min, y_max + 1, dtype=np.float64),
        )
        denominator = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (
            y[0] - y[2]
        )
        if abs(float(denominator)) <= 1e-15:
            continue
        a = (
            (y[1] - y[2]) * (xs - x[2]) + (x[2] - x[1]) * (ys - y[2])
        ) / denominator
        b = (
            (y[2] - y[0]) * (xs - x[2]) + (x[0] - x[2]) * (ys - y[2])
        ) / denominator
        c = 1.0 - a - b
        inside = (a >= -1e-7) & (b >= -1e-7) & (c >= -1e-7)
        inverse_depth = (
            a / depths[tri[0]] + b / depths[tri[1]] + c / depths[tri[2]]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            z = 1.0 / inverse_depth
        inside &= np.isfinite(z) & (z > 0.0)
        if not bool(inside.any()):
            continue
        region_depth = depth[y_min : y_max + 1, x_min : x_max + 1]
        region_mask = mask[y_min : y_max + 1, x_min : x_max + 1]
        update = inside & ((~region_mask) | (z < region_depth))
        if not bool(update.any()):
            continue
        region_depth[update] = z[update]
        region_mask[update] = True
        normals[y_min : y_max + 1, x_min : x_max + 1][update] = normal
    return _Raster(
        mask=mask,
        depth=np.where(mask, depth, 0.0).astype(np.float32),
        normals=normals,
    )


def _camera_rays(
    pixels: NDArray[np.floating[Any]],
    camera_pose: NDArray[np.floating[Any]],
    intrinsics: NDArray[np.floating[Any]],
) -> tuple[FloatArray, FloatArray]:
    """World origins/directions for pixel coordinates (u, v)."""
    T = np.asarray(camera_pose, dtype=np.float64).reshape(4, 4)
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    uv = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    camera_rays = np.stack(
        (
            (uv[:, 0] - K[0, 2]) / K[0, 0],
            -(uv[:, 1] - K[1, 2]) / K[1, 1],
            -np.ones(len(uv), dtype=np.float64),
        ),
        axis=1,
    )
    directions = camera_rays @ T[:3, :3].T
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    origins = np.broadcast_to(T[:3, 3], directions.shape).copy()
    return origins, directions


def _roi_grid_bounds(
    footprint: BoolArray,
    *,
    target_width: float,
    target_height: float,
    margin: int = 2,
) -> tuple[float, float, float, float] | None:
    """Largest target-aspect 9x7 rectangle centred on the sensor ROI.

    ``footprint`` is the complete camera-facing rest surface, including its
    bevel.  The requested old-stack extent is used exactly when it fits; only
    then is a uniform shrink applied.  No camera/crop is changed here.
    """
    height, width = footprint.shape
    eroded_image = Image.fromarray(
        np.where(footprint, 255, 0).astype(np.uint8)
    ).filter(ImageFilter.MinFilter(2 * margin + 1))
    available: BoolArray = np.asarray(eroded_image, dtype=np.uint8) > 0
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    for scale in range(100, 0, -1):
        box_width = max(1, round(target_width * scale / 100.0))
        box_height = max(1, round(target_height * scale / 100.0))
        x0 = round(center_x - (box_width - 1) / 2.0)
        y0 = round(center_y - (box_height - 1) / 2.0)
        x1 = x0 + box_width - 1
        y1 = y0 + box_height - 1
        if x0 < 0 or y0 < 0 or x1 >= width or y1 >= height:
            continue
        if bool(available[y0 : y1 + 1, x0 : x1 + 1].all()):
            return float(x0), float(x1), float(y0), float(y1)
    return None


def _fixed_marker_pixels(
    width: int,
    height: int,
    *,
    rows: int,
    cols: int,
    bounds: tuple[float, float, float, float] | None = None,
) -> FloatArray:
    """Fixed image-space GsMini lattice, independent of the live footprint.

    ``bounds`` is the rest-footprint-calibrated rectangle.  When the
    footprint cannot contain such a rectangle, fall back to the authored
    GsMini margins (~x 43..276, y 35..205) and leave uncovered dots at their
    fixed pixels.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("marker lattice dimensions must be positive")
    if bounds is None:
        margin_x = 44.0 * width / 320.0
        margin_y = 35.0 * height / 240.0
        x0, x1 = margin_x, width - margin_x
        y0, y1 = margin_y, height - margin_y
    else:
        x0, x1, y0, y1 = bounds
    x_values = np.linspace(x0, x1, cols)
    y_values = np.linspace(y0, y1, rows)
    return np.asarray(
        [(x, y) for y in y_values for x in x_values], dtype=np.float64
    )


def _snap_markers(
    vertices: NDArray[np.floating[Any]],
    triangles: NDArray[np.integer[Any]],
    camera_pose: NDArray[np.floating[Any]],
    intrinsics: NDArray[np.floating[Any]],
    pixels: FloatArray,
) -> tuple[NDArray[np.int32], FloatArray, BoolArray]:
    """Attach fixed image-space marker rays to rest triangles.

    Dots whose fixed image position lies just outside the rest footprint keep
    their fixed image position (marker rendering falls back to the pixel) while
    all interior dots remain barycentrically attached and move with the sheet.
    This keeps the 9x7 layout and centroid fixed without moving the whole grid.
    """
    if pixels.size == 0:
        raise ValueError("marker lattice must not be empty")
    origins, directions = _camera_rays(pixels, camera_pose, intrinsics)
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    v0 = points[tris[:, 0]]
    edge1 = points[tris[:, 1]] - v0
    edge2 = points[tris[:, 2]] - v0
    lattice_triangles = np.zeros(len(pixels), dtype=np.int32)
    lattice_weights = np.zeros((len(pixels), 3), dtype=np.float64)
    lattice_weights[:, 0] = 1.0
    attached = np.zeros(len(pixels), dtype=np.bool_)
    for index, (origin, direction) in enumerate(
        zip(origins, directions, strict=True)
    ):
        h: FloatArray = np.asarray(
            np.cross(direction, edge2), dtype=np.float64
        )
        a: FloatArray = np.asarray(
            np.einsum("ij,ij->i", edge1, h), dtype=np.float64
        )
        valid = np.abs(a) > 1e-12
        f: FloatArray = np.zeros_like(a)
        f[valid] = 1.0 / a[valid]
        s: FloatArray = origin - v0
        u: FloatArray = f * np.asarray(
            np.einsum("ij,ij->i", s, h), dtype=np.float64
        )
        q: FloatArray = np.asarray(np.cross(s, edge1), dtype=np.float64)
        v: FloatArray = f * np.asarray(
            np.einsum("ij,ij->i", direction[None, :], q), dtype=np.float64
        )
        t: FloatArray = f * np.asarray(
            np.einsum("ij,ij->i", edge2, q), dtype=np.float64
        )
        hit = (
            valid
            & (u >= -1e-7)
            & (v >= -1e-7)
            & (u + v <= 1.0 + 1e-7)
            & (t > 1e-9)
        )
        if not bool(hit.any()):
            continue
        candidates = np.flatnonzero(hit)
        best = int(candidates[np.argmin(t[candidates])])
        lattice_triangles[index] = best
        lattice_weights[index] = (1.0 - u[best] - v[best], u[best], v[best])
        attached[index] = True
    return lattice_triangles, lattice_weights, attached


def _lattice_points(
    vertices: NDArray[np.floating[Any]],
    triangle_vertices: NDArray[np.int32],
    weights: FloatArray,
) -> FloatArray:
    """Evaluate barycentrically attached marker points on live vertices."""
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    corners = points[np.asarray(triangle_vertices, dtype=np.int32)]
    return np.einsum("ni,nij->nj", weights, corners)


def _gel_background(
    image_size: tuple[int, int], *, seed: int
) -> NDArray[np.uint8]:
    """Procedural teal/purple gel background with fixed pattern noise."""
    width, height = image_size
    ys, xs = np.mgrid[0:height, 0:width]
    u = xs.astype(np.float64) / max(1, width - 1)
    v = ys.astype(np.float64) / max(1, height - 1)
    teal = np.array((24.0, 128.0, 132.0))
    purple = np.array((112.0, 58.0, 128.0))
    mix = np.clip(0.62 * u + 0.38 * v, 0.0, 1.0)[..., None]
    base = teal * (1.0 - mix) + purple * mix
    radius = (u - 0.5) ** 2 + (v - 0.5) ** 2
    base *= (1.0 - 0.12 * np.clip(radius * 4.0, 0.0, 1.0))[..., None]
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.7, size=(height, width, 1))
    return np.clip(base + noise, 0.0, 255.0).astype(np.uint8)


def _surface_shading(
    raster: _Raster,
    rest_depth: FloatArray,
    aperture: BoolArray,
    aperture_alpha: FloatArray,
) -> FloatArray:
    """Fixed-rest indentation and normal shading for covered pixels.

    Contact height is ``rest_depth - live_depth`` in the camera frame, so rigid
    finger motion and a whole-pad tilt do not create a dark region, and the
    resting sheet is exactly unshaded.  Only the fixed aperture is shaded.
    """
    height, width = raster.depth.shape
    shade = np.ones((height, width), dtype=np.float64)
    valid = aperture & raster.mask
    count = int(valid.sum())
    if count < 3:
        return shade
    ys, xs = np.nonzero(valid)
    live_depth = raster.depth[ys, xs].astype(np.float64)
    indentation = np.clip(
        rest_depth[ys, xs] - live_depth, 0.0, 0.0015
    )
    contact: FloatArray = np.clip(indentation / 0.0008, 0.0, 1.0)
    normals = raster.normals[ys, xs]
    median_normal: FloatArray = normals.mean(axis=0)
    length = float(np.linalg.norm(median_normal))
    if length > 0.0:
        median_normal /= length
        normal_delta: FloatArray = np.clip(
            1.0 - np.abs(normals @ median_normal), 0.0, 1.0
        )
        normal_delta -= float(np.median(normal_delta))
        normal_delta = np.clip(normal_delta, 0.0, 1.0)
    else:
        normal_delta = np.zeros(count, dtype=np.float64)
    shade[ys, xs] = 1.0 - aperture_alpha[ys, xs] * (
        0.50 * contact + 0.08 * normal_delta
    )
    return shade


def _marker_image(
    vertices: NDArray[np.floating[Any]],
    camera_pose: NDArray[np.floating[Any]],
    intrinsics: NDArray[np.floating[Any]],
    image_size: tuple[int, int],
    lattice_vertices: NDArray[np.int32],
    lattice_weights: FloatArray,
    marker_pixels: FloatArray,
    marker_attached: BoolArray,
    *,
    radius: float,
) -> NDArray[np.uint8]:
    """White keep-mask with anti-aliased black dots on fixed image positions.

    Attached markers move with far-sheet deformation.  A fixed-lattice dot
    whose ray misses the rest footprint stays at its fixed pixel, so the
    9-column x 7-row layout and centroid do not shift.
    """
    width, height = image_size
    points = _lattice_points(vertices, lattice_vertices, lattice_weights)
    uv, depth = _project(points, camera_pose, intrinsics)
    supersample = 4
    marker = Image.new(
        "L", (width * supersample, height * supersample), 255
    )
    draw = ImageDraw.Draw(marker)
    scaled_radius = radius * supersample
    for index, ((u, v), d) in enumerate(zip(uv, depth, strict=True)):
        if (
            not marker_attached[index]
            or not np.isfinite(u)
            or not np.isfinite(v)
            or d <= 1e-9
        ):
            u, v = marker_pixels[index]
        if u < -radius or u > width + radius or v < -radius or v > height + radius:
            continue
        cu, cv = u * supersample, v * supersample
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


@dataclass(frozen=True)
class TactileSensorVisual:
    """One pad's independent marker/depth sensor image generator."""

    side: str
    image_width: int
    image_height: int
    intrinsics: FloatArray
    far_triangles: NDArray[np.int32]
    lattice_vertices: NDArray[np.int32]
    lattice_weights: FloatArray
    marker_pixels: FloatArray
    marker_attached: BoolArray
    background: NDArray[np.uint8]
    rest_depth: FloatArray
    aperture: BoolArray
    aperture_alpha: FloatArray
    marker_radius: float
    marker_rows: int
    marker_cols: int

    @classmethod
    def bind(
        cls,
        *,
        side: str,
        rest_vertices: NDArray[np.floating[Any]],
        triangles: NDArray[np.integer[Any]],
        camera_pose: NDArray[np.floating[Any]],
        camera_intrinsics: NDArray[np.floating[Any]],
        image_size: tuple[int, int],
        marker_rows: int = 7,
        marker_cols: int = 9,
    ) -> TactileSensorVisual:
        """Calibrate the sensor pass from the rest far sheet and camera."""
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError("sensor image size must be positive")
        intrinsics = np.asarray(camera_intrinsics, dtype=np.float64).reshape(3, 3)
        vertices = np.asarray(rest_vertices, dtype=np.float64).reshape(-1, 3)
        all_triangles = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
        camera = np.asarray(camera_pose, dtype=np.float64).reshape(4, 4)
        far = select_far_sheet(vertices, all_triangles, camera[:3, 3])
        far_triangles = all_triangles[far]
        raster = _rasterize(
            vertices, far_triangles, camera, intrinsics, image_size
        )
        if not bool(raster.mask.any()):
            raise RuntimeError("sensor far sheet covers no pixels")
        # The aperture is fixed at bind from the rest footprint, eroded and
        # feathered.  It is shared by depth and optical so contact cannot flip
        # the depth valid mask or expose a dark plane-fit step at the border.
        footprint = raster.mask.copy()
        # MinFilter replicates the border; clear it first so erosion also
        # removes the two image-edge pixels when the rest sheet fills the frame.
        footprint[:2, :] = False
        footprint[-2:, :] = False
        footprint[:, :2] = False
        footprint[:, -2:] = False
        eroded = Image.fromarray(
            np.where(footprint, 255, 0).astype(np.uint8)
        ).filter(ImageFilter.MinFilter(5))
        aperture = np.asarray(eroded, dtype=np.uint8) > 0
        if not bool(aperture.any()):
            aperture = raster.mask.copy()
        aperture_alpha = np.asarray(
            Image.fromarray(np.where(aperture, 255, 0).astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(2)
            ),
            dtype=np.float64,
        ) / 255.0
        marker_triangles = _camera_facing_triangles(
            vertices, all_triangles, camera[:3, 3]
        )
        marker_raster = _rasterize(
            vertices, marker_triangles, camera, intrinsics, image_size
        )
        if not bool(marker_raster.mask.any()):
            marker_triangles = far_triangles
            marker_raster = raster
        radius = max(2.0, 6.2 * min(width, height) / 240.0)
        pixels = _fixed_marker_pixels(
            width,
            height,
            rows=marker_rows,
            cols=marker_cols,
            bounds=_roi_grid_bounds(
                marker_raster.mask,
                # The requested bbox is the printed-dot extent; dot centres
                # sit one radius inside it on every side.
                target_width=232.0 * width / 320.0 - 2.0 * radius,
                target_height=170.0 * height / 240.0 - 2.0 * radius,
            ),
        )
        lattice_triangles, lattice_weights, marker_attached = _snap_markers(
            vertices, marker_triangles, camera, intrinsics, pixels
        )
        return cls(
            side=side,
            image_width=width,
            image_height=height,
            intrinsics=intrinsics,
            far_triangles=far_triangles,
            lattice_vertices=marker_triangles[lattice_triangles],
            lattice_weights=lattice_weights,
            marker_pixels=pixels,
            marker_attached=marker_attached,
            background=_gel_background(
                image_size, seed=0 if side == "left" else 1
            ),
            rest_depth=raster.depth.astype(np.float64),
            aperture=aperture,
            aperture_alpha=aperture_alpha,
            marker_radius=radius,
            marker_rows=marker_rows,
            marker_cols=marker_cols,
        )

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.image_width, self.image_height)

    def geometry(
        self,
        world_vertices: NDArray[np.floating[Any]],
        camera_pose: NDArray[np.floating[Any]],
    ) -> _Raster:
        """Rasterize the live far sheet once for depth and normals."""
        vertices = np.asarray(world_vertices, dtype=np.float64).reshape(-1, 3)
        if len(vertices) < int(self.far_triangles.max()) + 1:
            raise RuntimeError("sensor vertex count is too small for its triangles")
        return _rasterize(
            vertices,
            self.far_triangles,
            np.asarray(camera_pose, dtype=np.float64).reshape(4, 4),
            self.intrinsics,
            self.image_size,
        )

    def aperture_mask(self) -> BoolArray:
        """Fixed rest aperture, eroded at bind; never follows live contact."""
        return self.aperture

    def height_map_mm(
        self,
        world_vertices: NDArray[np.floating[Any]],
        camera_pose: NDArray[np.floating[Any]],
    ) -> FloatArray:
        """Contact height in millimetres against the fixed rest depth."""
        raster = self.geometry(world_vertices, camera_pose)
        height = np.zeros(raster.depth.shape, dtype=np.float64)
        valid = self.aperture & raster.mask
        # Camera depth shrinks when the sheet moves towards the camera, so
        # rest - live is positive for indentation and negative for lift.
        height[valid] = (
            self.rest_depth[valid] - raster.depth[valid].astype(np.float64)
        ) * 1000.0
        height *= self.aperture_alpha
        return height

    def marker_mask(
        self,
        world_vertices: NDArray[np.floating[Any]],
        camera_pose: NDArray[np.floating[Any]],
    ) -> NDArray[np.uint8]:
        """Black printed-dot keep-mask; marker positions stay lattice-driven."""
        vertices = np.asarray(world_vertices, dtype=np.float64).reshape(-1, 3)
        camera = np.asarray(camera_pose, dtype=np.float64).reshape(4, 4)
        return _marker_image(
            vertices,
            camera,
            self.intrinsics,
            self.image_size,
            self.lattice_vertices,
            self.lattice_weights,
            self.marker_pixels,
            self.marker_attached,
            radius=self.marker_radius,
        )

    def render(
        self,
        world_vertices: NDArray[np.floating[Any]],
        camera_pose: NDArray[np.floating[Any]],
    ) -> TactileSensorImages:
        """Render one independent tactile/marker frame from live vertices."""
        raster = self.geometry(world_vertices, camera_pose)
        shade = _surface_shading(
            raster, self.rest_depth, self.aperture, self.aperture_alpha
        )
        # Only the fixed 2-3 px aperture edge is feathered; shading outside it
        # stays at unity, so a moving footprint cannot draw a dark border.
        shade_image = Image.fromarray(
            np.clip(shade * 255.0, 0.0, 255.0).astype(np.uint8)
        )
        blurred = shade_image.filter(ImageFilter.GaussianBlur(2))
        shade = np.asarray(blurred, dtype=np.float64) / 255.0
        tactile = np.clip(
            self.background.astype(np.float64) * shade[..., None],
            0.0,
            255.0,
        ).astype(np.uint8)
        marker = self.marker_mask(world_vertices, camera_pose)
        marker_rgb = (
            tactile.astype(np.uint16) * marker.astype(np.uint16)[..., None] // 255
        ).astype(np.uint8)
        return TactileSensorImages(
            tactile_rgb=tactile,
            marker_rgb=marker_rgb,
            aperture_mask=self.aperture,
        )


def make_tactile_visual(
    *,
    side: str,
    rest_vertices: NDArray[np.floating[Any]],
    triangles: NDArray[np.integer[Any]],
    camera_pose: NDArray[np.floating[Any]],
    camera_intrinsics: NDArray[np.floating[Any]],
    image_size: tuple[int, int],
    marker_rows: int,
    marker_cols: int,
) -> TactileSensorVisual:
    """Attach the far-sheet marker lattice and build the shared visual pass."""
    return TactileSensorVisual.bind(
        side=side,
        rest_vertices=rest_vertices,
        triangles=triangles,
        camera_pose=camera_pose,
        camera_intrinsics=camera_intrinsics,
        image_size=image_size,
        marker_rows=marker_rows,
        marker_cols=marker_cols,
    )
