"""Pure contact geometry: the pad rest plane, the height field and its normals.

Everything here is a closed-form function of pad calibration and object depth:
there is no captured rest image and no simulated gel vertex anywhere in the
chain.  ``height = clip(rest_plane - object_depth, 0, inf)``, and the surface
normals are the gradient of that height field.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _pixel_rays(
    intrinsics: NDArray[np.floating[Any]], image_size: tuple[int, int]
) -> tuple[FloatArray, FloatArray]:
    """Pixel-centre camera rays in the rasterizer's pixel convention.

    The rasterizer samples a triangle at integer pixel coordinates, so the
    plane is evaluated at the same coordinates; principal point and focal
    lengths come from the recorded intrinsic matrix.
    """
    width, height = image_size
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    us, vs = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    dx = (us - K[0, 2]) / K[0, 0]
    dy = -(vs - K[1, 2]) / K[1, 1]
    return dx, dy


def plane_depth(
    normal: NDArray[np.floating[Any]],
    offset: float,
    intrinsics: NDArray[np.floating[Any]],
    image_size: tuple[int, int],
) -> FloatArray:
    """Optical-axis depth of the plane ``normal . X = offset`` per pixel.

    ``normal`` is a unit vector in camera coordinates pointing from the plane
    toward the camera and ``offset`` is its signed plane constant, so every
    depth is positive when the plane lies in front of the camera.
    """
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    dx, dy = _pixel_rays(intrinsics, image_size)
    denominator = n[0] * dx + n[1] * dy - n[2]
    depth = float(offset) / denominator
    if not np.isfinite(depth).all() or not bool((depth > 0.0).all()):
        raise ValueError(
            "pad rest plane does not lie in front of every pad pixel; check the "
            "camera-to-pad calibration and the plane height"
        )
    return depth


def plane_points(
    pixels: NDArray[np.floating[Any]],
    normal: NDArray[np.floating[Any]],
    offset: float,
    intrinsics: NDArray[np.floating[Any]],
) -> FloatArray:
    """Camera-frame points where the pixel rays meet the rest plane."""
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    uv = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    dx = (uv[:, 0] - K[0, 2]) / K[0, 0]
    dy = -(uv[:, 1] - K[1, 2]) / K[1, 1]
    denominator = n[0] * dx + n[1] * dy - n[2]
    depth = float(offset) / denominator
    if not np.isfinite(depth).all() or not bool((depth > 0.0).all()):
        raise ValueError("marker lattice falls outside the pad rest plane")
    return np.stack((depth * dx, depth * dy, -depth), axis=1)


def height_field(
    rest_plane_depth: FloatArray,
    object_depth: FloatArray,
    aperture: NDArray[np.bool_],
) -> FloatArray:
    """Contact height in metres: ``clip(rest_plane - object_depth, 0, inf)``.

    ``object_depth`` is ``+inf`` where no declared object covers the pixel, so
    those pixels and every pixel outside the fixed aperture are exactly zero.
    """
    height = np.asarray(rest_plane_depth, dtype=np.float64) - np.asarray(
        object_depth, dtype=np.float64
    )
    np.maximum(height, 0.0, out=height)
    height[~np.asarray(aperture, dtype=bool)] = 0.0
    return height


def normals_from_height(
    height: FloatArray,
    rest_plane_depth: FloatArray,
    intrinsics: NDArray[np.floating[Any]],
) -> FloatArray:
    """Unit normals ``(-dh/dx, -dh/dy, 1)`` in camera coordinates.

    The gradient is taken in metres of gel-face distance per metre of plane
    coordinate, so a pixel's physical size at the rest plane scales it.  A flat
    (zero) height field gives exactly ``(0, 0, 1)``: no shading at rest.
    """
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    depth = np.asarray(rest_plane_depth, dtype=np.float64)
    safe_depth = np.where(depth > 0.0, depth, 1.0)
    dh_dv, dh_du = np.gradient(np.asarray(height, dtype=np.float64))
    dh_dx = dh_du * K[0, 0] / safe_depth
    dh_dy = -dh_dv * K[1, 1] / safe_depth
    normal = np.stack((-dh_dx, -dh_dy, np.ones_like(dh_dx)), axis=-1)
    return normal / np.linalg.norm(normal, axis=-1, keepdims=True)


def sample_bilinear(
    image: NDArray[np.floating[Any]],
    xs: NDArray[np.floating[Any]],
    ys: NDArray[np.floating[Any]],
) -> FloatArray:
    """Bilinear sample of ``image`` at continuous pixel coordinates."""
    height, width = image.shape
    x = np.clip(np.asarray(xs, dtype=np.float64), 0.0, width - 1.0)
    y = np.clip(np.asarray(ys, dtype=np.float64), 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = x - x0
    fy = y - y0
    values = np.asarray(image, dtype=np.float64)
    top = values[y0, x0] * (1.0 - fx) + values[y0, x1] * fx
    bottom = values[y1, x0] * (1.0 - fx) + values[y1, x1] * fx
    return top * (1.0 - fy) + bottom * fy
