from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self, cast

import h5py
import numpy as np
import numpy.typing as npt

from openvitac_data.episode import (
    SCHEMA_VERSION,
    ActionSpec,
    CameraMeta,
    DepthCameraMeta,
    EpisodeExtra,
    EpisodeMeta,
    ObjectFrame,
    ObjectMeta,
    PadTactileMeta,
    PinholeIntrinsics,
    TactileFrame,
)
from openvitac_data.stream import StreamReader, StreamWriter

Outcome = Literal["success", "fail", "timeout", "unknown", "policy_error"]
_TactileStatus = Literal["ok", "disabled", "failed"]
_TactileImageStream = Literal["rgb", "marker_rgb", "depth"]
_OUTCOMES: frozenset[str] = frozenset(
    ("success", "fail", "timeout", "unknown", "policy_error")
)
_TACTILE_STATUSES: frozenset[str] = frozenset(("ok", "disabled", "failed"))
_TACTILE_IMAGE_STREAMS: tuple[_TactileImageStream, ...] = (
    "rgb",
    "marker_rgb",
    "depth",
)


class Hdf5SchemaError(ValueError):
    """An episode file violates the HDF5 layout invariants.

    Raised by readers for structurally inconsistent files (for example frame
    datasets with different first-dimension lengths) rather than silently
    trusting one stream.
    """


@dataclass(frozen=True)
class Frame:
    """One observation; every stream may also hold a ``(T, ...)`` chunk."""

    images: dict[str, npt.NDArray[np.uint8]]  # role -> (H, W, 3)
    joint_pos: npt.NDArray[np.floating[Any]]
    joint_vel: npt.NDArray[np.floating[Any]]
    # Target in effect at this frame, column order = meta.action.joint_names.
    action: npt.NDArray[np.floating[Any]]
    tactile: dict[str, TactileFrame] | None = None
    timestamp_us: int | None = None
    objects: dict[str, ObjectFrame] | None = None


def _attr_str(attrs: h5py.AttributeManager, key: str) -> str:
    value = attrs[key]
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _keys(node: Any) -> list[str]:
    """Member names of an HDF5 group (h5py's mapping API is untyped)."""
    return [str(key) for key in node]


def _str_list(dataset: h5py.Dataset) -> list[str]:
    values = cast(Any, dataset[0 : dataset.shape[0]])
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def _float_rows(
    value: npt.NDArray[np.floating[Any]] | None, name: str, width: int
) -> tuple[npt.NDArray[np.float32], int]:
    """Normalize a ``(W,)`` or ``(T, W)`` floating stream to ``(T, W)`` rows."""
    if value is None:
        raise ValueError(f"{name} is required")
    if value.ndim == 1:
        rows = value[None, :]
    elif value.ndim == 2:
        rows = value
    else:
        raise ValueError(
            f"{name} must be ({width},) or (T, {width}), got {value.shape}"
        )
    if not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"{name} must be floating, got {value.dtype}")
    if rows.shape[1] != width:
        raise ValueError(f"{name} must have {width} columns, got {rows.shape[1]}")
    return rows.astype(np.float32), int(rows.shape[0])


def _depth_rows(
    value: npt.NDArray[np.floating[Any]] | None,
    name: str,
    *,
    count: int,
    height: int,
    width: int,
) -> npt.NDArray[np.float32]:
    """Normalize ``(H, W)`` or ``(T, H, W)`` depth to ``(T, H, W)`` float32."""
    if value is None:
        raise ValueError(f"{name} is required when the pad has a depth camera")
    rows = np.asarray(value)
    if rows.ndim == 2:
        rows = rows[None, ...]
    elif rows.ndim != 3:
        raise ValueError(
            f"{name} must be ({height}, {width}) or "
            f"(T, {height}, {width}), got {rows.shape}"
        )
    if rows.shape[1:] != (height, width):
        raise ValueError(
            f"{name} must have trailing shape ({height}, {width}), "
            f"got {rows.shape}"
        )
    if rows.shape[0] != count:
        raise ValueError(f"{name} has {rows.shape[0]} frames but expected {count}")
    if not np.issubdtype(rows.dtype, np.floating):
        raise ValueError(f"{name} must be floating, got {rows.dtype}")
    rows = rows.astype(np.float32)
    if not np.isfinite(rows).all():
        raise ValueError(f"{name} contains non-finite depths")
    return rows


def _tactile_image_rows(
    value: npt.NDArray[np.uint8] | None,
    name: str,
    *,
    count: int,
    height: int,
    width: int,
) -> npt.NDArray[np.uint8]:
    """Normalize ``(H, W, 3)`` or ``(T, H, W, 3)`` tactile images to rows."""
    if value is None:
        raise ValueError(f"{name} is required for an image-enabled pad")
    rows = np.asarray(value)
    if rows.ndim == 3:
        rows = rows[None, ...]
    elif rows.ndim != 4:
        raise ValueError(
            f"{name} must be ({height}, {width}, 3) or "
            f"(T, {height}, {width}, 3), got {rows.shape}"
        )
    if rows.shape[1:] != (height, width, 3):
        raise ValueError(
            f"{name} must have trailing shape ({height}, {width}, 3), "
            f"got {rows.shape}"
        )
    if rows.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8, got {rows.dtype}")
    if rows.shape[0] != count:
        raise ValueError(f"{name} has {rows.shape[0]} frames but expected {count}")
    return np.ascontiguousarray(rows)


def _scalar_rows(
    value: Any, name: str, count: int, dtype: npt.DTypeLike
) -> npt.NDArray[Any]:
    """Normalize a scalar or ``(T,)`` stream to ``(T,)`` of ``dtype``."""
    array = np.asarray(value)
    if array.ndim == 0:
        return np.full((count,), array.item(), dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be scalar or (T,), got {array.shape}")
    if array.shape[0] == count:
        return array.astype(dtype)
    if array.shape[0] == 1:
        return np.repeat(array.astype(dtype), count)
    raise ValueError(f"{name} has {array.shape[0]} rows but expected {count}")


def _float_scalar_rows(
    value: float | npt.NDArray[np.floating[Any]] | None,
    name: str,
    count: int,
) -> npt.NDArray[np.float32]:
    """Normalize a metric to ``(T,)`` float32; ``None`` becomes NaN."""
    if value is None:
        return np.full((count,), np.nan, dtype=np.float32)
    return _scalar_rows(value, name, count, np.float32)


def _int_scalar_rows(
    value: int | npt.NDArray[np.integer[Any]], name: str, count: int
) -> npt.NDArray[np.int64]:
    return _scalar_rows(value, name, count, np.int64)


def _bool_scalar_rows(
    value: bool | npt.NDArray[np.bool_], name: str, count: int
) -> npt.NDArray[np.bool_]:
    return _scalar_rows(value, name, count, np.bool_)


def _pose_to_T(pose_xyzw: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float64]:
    """Build a world <- object transform from ``(x,y,z,qx,qy,qz,qw)``."""
    pose = np.asarray(pose_xyzw, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"object pose must be (7,), got {pose.shape}")
    qx, qy, qz, qw = pose[3:]
    norm = float(np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("object pose quaternion must be finite and nonzero")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]
    T[:3, 3] = pose[:3]
    return T


def _camera_pose_rows(
    value: npt.NDArray[Any] | None, name: str, count: int
) -> npt.NDArray[np.float64] | None:
    """Normalize an optional ``(4,4)``/``(T,4,4)`` per-frame camera pose."""
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (4, 4):
        array = array[None]
    if array.shape != (count, 4, 4):
        raise ValueError(f"{name} must be (4,4) or (T,4,4), got {array.shape}")
    return array


def _transform_rows(
    value: npt.NDArray[np.floating[Any]] | None,
    poses: npt.NDArray[np.float32],
    name: str,
    count: int,
) -> npt.NDArray[np.float64]:
    """Normalize a ``(4,4)``/``(T,4,4)`` transform stream; derive when absent."""
    if value is None:
        return np.stack([_pose_to_T(pose) for pose in poses], axis=0)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2:
        if array.shape != (4, 4):
            raise ValueError(f"{name} must be (4,4) or (T,4,4), got {array.shape}")
        array = array[None, ...]
    elif array.ndim != 3 or array.shape[1:] != (4, 4):
        raise ValueError(f"{name} must be (4,4) or (T,4,4), got {array.shape}")
    if array.shape[0] == count:
        return array
    if array.shape[0] == 1:
        return np.repeat(array, count, axis=0)
    raise ValueError(f"{name} has {array.shape[0]} rows but expected {count}")


def _pad_vertices(
    vertices: npt.NDArray[np.floating[Any]] | None,
    n_vertices: int | npt.NDArray[np.integer[Any]],
    name: str,
    count: int,
    max_vertices: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Normalize vertices to ``(T, max_vertices, 3)`` with zero padding."""
    n = _int_scalar_rows(n_vertices, f"{name}.n_vertices", count)
    if np.any(n < 0) or np.any(n > max_vertices):
        raise ValueError(f"{name}.n_vertices must be within 0..{max_vertices}")
    if vertices is None:
        if np.any(n > 0):
            raise ValueError(f"{name}.vertices is required when n_vertices > 0")
        return np.zeros((count, max_vertices, 3), dtype=np.float32), n.astype(np.int32)
    array = np.asarray(vertices, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name}.vertices must be (V,3) or (T,V,3), got {array.shape}")
    if array.shape[0] == count:
        rows = array
    elif array.shape[0] == 1:
        rows = np.repeat(array, count, axis=0)
    else:
        raise ValueError(
            f"{name}.vertices has {array.shape[0]} rows but expected {count}"
        )
    if np.any(n > rows.shape[1]):
        raise ValueError(f"{name}.n_vertices exceeds vertices width {rows.shape[1]}")
    padded = np.zeros((count, max_vertices, 3), dtype=np.float32)
    for i, row_n in enumerate(n):
        padded[i, : int(row_n)] = rows[i, : int(row_n)]
    return padded, n.astype(np.int32)


def _image_rows(
    image: npt.NDArray[np.uint8], role: str, camera: CameraMeta
) -> tuple[npt.NDArray[np.uint8], int]:
    """Normalize an ``(H, W, 3)`` or ``(T, H, W, 3)`` stream to batched rows."""
    if image.ndim == 3:
        rows = image[None, ...]
    elif image.ndim == 4:
        rows = image
    else:
        raise ValueError(
            f"images[{role!r}] must be (H, W, 3) or (T, H, W, 3), got {image.shape}"
        )
    if image.dtype != np.uint8:
        raise ValueError(f"images[{role!r}] must be uint8, got {image.dtype}")
    if rows.shape[-1] != 3:
        raise ValueError(f"images[{role!r}] channels must be 3, got {image.shape}")
    if rows.shape[1] != camera.height or rows.shape[2] != camera.width:
        raise ValueError(
            f"images[{role!r}] must be {camera.height}x{camera.width} per "
            f"meta.cameras, got {rows.shape[1]}x{rows.shape[2]}"
        )
    return rows, int(rows.shape[0])


class Hdf5EpisodeWriter(StreamWriter[Frame]):
    """Write one episode file, creating datasets from the first appended frame."""

    def __init__(
        self,
        path: str | Path,
        meta: EpisodeMeta[EpisodeExtra],
    ) -> None:
        if not meta.cameras:
            raise ValueError("meta.cameras must declare at least one camera")
        roles = [camera.role for camera in meta.cameras]
        if len(set(roles)) != len(roles):
            raise ValueError(f"duplicate camera roles: {roles}")
        if not meta.joint_names:
            raise ValueError("meta.joint_names must not be empty")
        if not meta.action.joint_names:
            raise ValueError("meta.action with joint_names is required")
        sides = [pad.side for pad in meta.tactile]
        if len(set(sides)) != len(sides):
            raise ValueError(f"duplicate tactile sides: {sides}")
        names = [obj.name for obj in meta.objects]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate object names: {names}")
        if meta.tactile_status not in _TACTILE_STATUSES:
            raise ValueError(f"invalid tactile_status {meta.tactile_status!r}")
        for pad in meta.tactile:
            streams = tuple(pad.image_streams)
            if len(set(streams)) != len(streams):
                raise ValueError(
                    f"tactile {pad.side!r} duplicate image_streams: {streams}"
                )
            unknown = sorted(set(streams) - set(_TACTILE_IMAGE_STREAMS))
            if unknown:
                raise ValueError(
                    f"tactile {pad.side!r} unsupported image_streams: {unknown}"
                )
            if "depth" in streams and pad.depth_camera is None:
                raise ValueError(
                    f"tactile {pad.side!r} declares depth but has no depth_camera"
                )
            if "depth" not in streams and pad.depth_camera is not None:
                raise ValueError(
                    f"tactile {pad.side!r} has a depth_camera but does not "
                    'declare the "depth" image stream'
                )
            if ("rgb" in streams or "marker_rgb" in streams) and (
                pad.width is None or pad.height is None
            ):
                raise ValueError(
                    f"tactile {pad.side!r} declares image streams but has "
                    "no width/height"
                )

        self._path = Path(path)
        self._meta = meta
        self._cameras = {camera.role: camera for camera in meta.cameras}
        self._file = h5py.File(self._path, "w")
        try:
            self._file.attrs["schema_version"] = SCHEMA_VERSION
            self._file.attrs["task_id"] = meta.task_id
            self._file.attrs["sim_dt"] = meta.sim_dt
            self._file.attrs["seed"] = meta.extra.seed
            self._file.attrs["instruction"] = meta.extra.instruction
            self._file.attrs["outcome"] = meta.extra.outcome
            self._file.attrs["episode_id"] = meta.episode_id
            self._file.attrs["tactile_status"] = meta.tactile_status
            self._write_meta_tree()
            self._frames_group = self._file.create_group("frames")
            self._images_group = self._frames_group.create_group("images")
        except Exception:
            self._file.close()
            raise
        self._images: dict[str, h5py.Dataset] = {}
        self._datasets: dict[str, h5py.Dataset] = {}
        self._object_datasets: dict[str, dict[str, h5py.Dataset]] = {}
        self._tactile_datasets: dict[str, dict[str, h5py.Dataset]] = {}
        self._tactile_force_kinds: dict[str, str] = {}
        # None = undecided; the first append fixes whether timestamps exist.
        self._has_timestamps: bool | None = None
        self._count = 0
        self._closed = False

    def _write_meta_tree(self) -> None:
        meta = self._meta
        action = meta.action
        group = self._file.create_group("meta")
        group.create_dataset(
            "joint_names",
            data=np.asarray(list(meta.joint_names), dtype=object),
            maxshape=(len(meta.joint_names),),
        )
        action_group = group.create_group("action")
        action_group.attrs["representation"] = action.representation
        action_group.create_dataset(
            "joint_names",
            data=np.asarray(list(action.joint_names), dtype=object),
            maxshape=(len(action.joint_names),),
        )
        cameras_group = group.create_group("cameras")
        for camera in meta.cameras:
            camera_group = cameras_group.create_group(camera.role)
            camera_group.attrs["prim_path"] = camera.prim_path
            camera_group.attrs["width"] = camera.width
            camera_group.attrs["height"] = camera.height
            camera_group.attrs["mount"] = camera.mount or ""
            camera_group.create_dataset(
                "pos", data=np.asarray(camera.pos), maxshape=(3,)
            )
            camera_group.create_dataset(
                "quat_xyzw", data=np.asarray(camera.rot), maxshape=(4,)
            )
            if camera.intrinsics is not None:
                camera_group.attrs["focal_length"] = camera.intrinsics.focal_length
                camera_group.attrs["focus_distance"] = camera.intrinsics.focus_distance
                camera_group.attrs["horizontal_aperture"] = (
                    camera.intrinsics.horizontal_aperture
                )
                camera_group.attrs["clipping_range"] = np.asarray(
                    camera.intrinsics.clipping_range
                )
        randomization = group.create_group("randomization")
        for key, value in meta.extra.randomization.items():
            randomization.attrs[key] = value
        group.attrs["tactile_status"] = meta.tactile_status
        tactile_group = group.create_group("tactile")
        for pad in meta.tactile:
            self._write_pad_meta(tactile_group, pad)
        objects_group = group.create_group("objects")
        for obj in meta.objects:
            self._write_object_meta(objects_group, obj)

    @staticmethod
    def _write_pad_meta(parent: h5py.Group, pad: PadTactileMeta) -> None:
        rest = np.asarray(pad.rest_vertices, dtype=np.float32)
        triangles = np.asarray(pad.surface_triangles, dtype=np.int32)
        frame = np.asarray(pad.pad_frame, dtype=np.float64)
        if rest.ndim != 2 or rest.shape[1] != 3:
            raise ValueError(f"tactile {pad.side!r} rest_vertices must be (V,3)")
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError(f"tactile {pad.side!r} surface_triangles must be (F,3)")
        if frame.shape != (4, 4):
            raise ValueError(f"tactile {pad.side!r} pad_frame must be (4,4)")
        max_vertices = (
            int(pad.max_vertices)
            if pad.max_vertices is not None
            else int(rest.shape[0])
        )
        if max_vertices < 1:
            raise ValueError(f"tactile {pad.side!r} max_vertices must be positive")
        group = parent.create_group(pad.side)
        group.attrs["prim_path"] = pad.prim_path
        group.attrs["attachment_body"] = pad.attachment_body
        group.attrs["youngs_mpa"] = pad.youngs_mpa
        group.attrs["E_MPa"] = pad.youngs_mpa
        group.attrs["strength"] = pad.strength
        group.attrs["max_vertices"] = max_vertices
        group.attrs["contact_count_semantics"] = "global_candidate_primitive_count"
        if pad.contact_object is not None:
            group.attrs["contact_object"] = pad.contact_object
        for key, value in (
            ("width", pad.width),
            ("height", pad.height),
            ("marker_rows", pad.marker_rows),
            ("marker_cols", pad.marker_cols),
        ):
            if value is not None:
                group.attrs[key] = value
        # Comma-joined so an empty declaration is still an explicit sentinel.
        group.attrs["image_streams"] = ",".join(pad.image_streams)
        if pad.depth_camera is not None:
            camera = pad.depth_camera
            group.attrs["depth_width"] = camera.width
            group.attrs["depth_height"] = camera.height
            group.attrs["depth_focal_length"] = camera.focal_length
            group.attrs["depth_horizontal_aperture"] = camera.horizontal_aperture
            group.attrs["depth_vertical_aperture"] = camera.vertical_aperture
            group.attrs["depth_clipping_range"] = np.asarray(
                camera.clipping_range, dtype=np.float64
            )
            group.attrs["depth_semantics"] = camera.semantics
            group.attrs["depth_surface"] = camera.surface
            group.attrs["depth_outside_aperture"] = camera.outside_aperture
        group.create_dataset("rest_vertices", data=rest, maxshape=rest.shape)
        group.create_dataset(
            "surface_triangles", data=triangles, maxshape=triangles.shape
        )
        group.create_dataset("pad_frame", data=frame, maxshape=(4, 4))
        if pad.camera_from_pad is not None:
            camera_from_pad = np.asarray(pad.camera_from_pad, dtype=np.float64)
            if camera_from_pad.shape != (4, 4):
                raise ValueError(
                    f"tactile {pad.side!r} camera_from_pad must be (4,4)"
                )
            group.create_dataset(
                "camera_from_pad", data=camera_from_pad, maxshape=(4, 4)
            )

    @staticmethod
    def _write_object_meta(parent: h5py.Group, obj: ObjectMeta) -> None:
        group = parent.create_group(obj.name)
        group.attrs["prim_path"] = obj.prim_path
        group.attrs["authority"] = obj.authority
        group.attrs["constitution"] = obj.constitution
        if obj.mass_kg is not None:
            group.attrs["mass_kg"] = obj.mass_kg
        if obj.usd is not None:
            group.attrs["usd"] = obj.usd
        group.attrs["imaging"] = bool(obj.imaging)
        if obj.mesh_checksum is not None:
            group.attrs["mesh_checksum"] = obj.mesh_checksum
        if obj.inertia is not None:
            inertia = np.asarray(obj.inertia, dtype=np.float32)
            if inertia.shape == (3,):
                group.create_dataset("inertia_diag_kg_m2", data=inertia, maxshape=(3,))
            elif inertia.shape == (3, 3):
                group.create_dataset("inertia", data=inertia, maxshape=(3, 3))
            else:
                raise ValueError(f"object {obj.name!r} inertia must be (3,) or (3,3)")
        if obj.initial_pose_xyzw is not None:
            pose = np.asarray(obj.initial_pose_xyzw, dtype=np.float32)
            if pose.shape != (7,):
                raise ValueError(f"object {obj.name!r} initial_pose must be (7,)")
            group.create_dataset("initial_pose_xyzw", data=pose, maxshape=(7,))
            group.create_dataset("initial_pose", data=_pose_to_T(pose), maxshape=(4, 4))

    @property
    def path(self) -> Path:
        return self._path

    @property
    def meta(self) -> EpisodeMeta[EpisodeExtra]:
        return self._meta

    @property
    def frame_count(self) -> int:
        return self._count

    def set_outcome(self, outcome: Outcome) -> None:
        """Finalize the episode outcome (success/fail/timeout/policy_error).

        A scripted expert runs *inside* the writer context, so frames stream in
        before its success predicate or step budget decides how the episode
        ended. ``outcome`` is rewritten in place; the in-memory :attr:`meta`
        follows so readers of ``meta`` and of the file agree.
        """
        if outcome not in _OUTCOMES:
            raise ValueError(f"invalid outcome {outcome!r}")
        if self._closed:
            raise ValueError("writer is closed")
        self._file.attrs["outcome"] = outcome
        self._meta = replace(
            self._meta, extra=replace(self._meta.extra, outcome=outcome)
        )

    def append(self, frame: Frame) -> None:
        if self._closed:
            raise ValueError("writer is closed")
        if frame.timestamp_us is None:
            raise ValueError("frame.timestamp_us is required")

        roles = set(frame.images)
        expected = set(self._cameras)
        if roles != expected:
            raise ValueError(
                f"images roles {sorted(roles)} do not match meta.cameras "
                f"roles {sorted(expected)}"
            )
        action = self._meta.action

        count: int | None = None
        image_rows: dict[str, npt.NDArray[np.uint8]] = {}
        for role, image in frame.images.items():
            batched, n = _image_rows(image, role, self._cameras[role])
            if count is None:
                count = n
            elif n != count:
                raise ValueError(
                    f"images[{role!r}] has {n} frames but expected {count}"
                )
            image_rows[role] = batched
        if count is None:  # meta.cameras is non-empty, roles match
            raise RuntimeError("frame images are required")

        float_rows: dict[str, npt.NDArray[np.float32]] = {}
        for name, value, width in (
            ("joint_pos", frame.joint_pos, len(self._meta.joint_names)),
            ("joint_vel", frame.joint_vel, len(self._meta.joint_names)),
            ("action", frame.action, len(action.joint_names)),
        ):
            batched, n = _float_rows(value, name, width)
            if n != count:
                raise ValueError(f"{name} has {n} frames but expected {count}")
            float_rows[name] = batched

        object_rows: dict[str, dict[str, npt.NDArray[Any]]] = self._object_rows(
            frame, count
        )
        tactile_rows: dict[str, dict[str, npt.NDArray[Any]]] = self._tactile_rows(
            frame, count
        )

        has_timestamps = frame.timestamp_us is not None
        if self._has_timestamps is None:
            self._has_timestamps = has_timestamps
        elif has_timestamps != self._has_timestamps:
            raise ValueError("timestamp_us must be set on every frame or on none")
        if has_timestamps and count != 1:
            raise ValueError(
                "timestamp_us is per frame; chunk appends must leave it None"
            )

        if not self._datasets:
            for role, batched in image_rows.items():
                self._images[role] = self._images_group.create_dataset(
                    role,
                    data=batched,
                    maxshape=(None, *batched.shape[1:]),
                    chunks=(1, *batched.shape[1:]),
                )
            for name, batched in float_rows.items():
                self._datasets[name] = self._frames_group.create_dataset(
                    name,
                    data=batched,
                    maxshape=(None, batched.shape[1]),
                    chunks=(1, batched.shape[1]),
                )
            if self._has_timestamps:
                self._datasets["timestamp_us"] = self._frames_group.create_dataset(
                    "timestamp_us",
                    data=np.array([frame.timestamp_us], dtype=np.int64),
                    maxshape=(None,),
                    chunks=(1024,),
                )
            self._create_stream_datasets(object_rows, tactile_rows)
        else:
            for role, dataset in self._images.items():
                self._append_rows(dataset, image_rows[role])
            for name in ("joint_pos", "joint_vel", "action"):
                self._append_rows(self._datasets[name], float_rows[name])
            if self._has_timestamps:
                self._append_rows(
                    self._datasets["timestamp_us"],
                    np.array([frame.timestamp_us], dtype=np.int64),
                )
            self._append_stream_datasets(object_rows, tactile_rows)

        self._count += count

    def _object_rows(
        self, frame: Frame, count: int
    ) -> dict[str, dict[str, npt.NDArray[Any]]]:
        frames = frame.objects or {}
        expected = {obj.name for obj in self._meta.objects}
        if set(frames) != expected:
            raise ValueError(
                f"object keys {sorted(frames)} do not match meta.objects "
                f"{sorted(expected)}"
            )
        rows: dict[str, dict[str, npt.NDArray[Any]]] = {}
        for obj_meta in self._meta.objects:
            obj = frames[obj_meta.name]
            pose, n = _float_rows(obj.pose_xyzw, f"objects[{obj_meta.name}].pose", 7)
            if n != count:
                raise ValueError(
                    f"objects[{obj_meta.name}].pose has {n} frames but expected {count}"
                )
            rows[obj_meta.name] = {
                "pose": pose,
                "T": _transform_rows(obj.T, pose, f"objects[{obj_meta.name}].T", count),
                "seq": _int_scalar_rows(
                    obj.seq, f"objects[{obj_meta.name}].seq", count
                ),
                "valid": _bool_scalar_rows(
                    obj.valid, f"objects[{obj_meta.name}].valid", count
                ),
            }
        return rows

    def _tactile_rows(
        self, frame: Frame, count: int
    ) -> dict[str, dict[str, npt.NDArray[Any]]]:
        frames = frame.tactile or {}
        expected = {pad.side for pad in self._meta.tactile}
        if set(frames) != expected:
            raise ValueError(
                f"tactile keys {sorted(frames)} do not match meta.tactile "
                f"{sorted(expected)}"
            )
        rows: dict[str, dict[str, npt.NDArray[Any]]] = {}
        for pad in self._meta.tactile:
            tactile = frames[pad.side]
            if tactile.side != pad.side:
                raise ValueError(f"tactile[{pad.side!r}].side is {tactile.side!r}")
            if tactile.force_kind not in ("net_force", "contact_gradient", "none"):
                raise ValueError(
                    f"tactile[{pad.side!r}] invalid force_kind {tactile.force_kind!r}"
                )
            previous_kind = self._tactile_force_kinds.get(pad.side)
            if previous_kind is None:
                self._tactile_force_kinds[pad.side] = tactile.force_kind
            elif previous_kind != tactile.force_kind:
                raise ValueError(f"tactile[{pad.side!r}] force_kind must stay constant")
            max_vertices = (
                int(pad.max_vertices)
                if pad.max_vertices is not None
                else int(np.asarray(pad.rest_vertices).shape[0])
            )
            vertices, n_vertices = _pad_vertices(
                tactile.vertices,
                tactile.n_vertices,
                f"tactile[{pad.side!r}]",
                count,
                max_vertices,
            )
            gradient, gradient_n = (
                _float_rows(
                    tactile.contact_gradient,
                    f"tactile[{pad.side!r}].contact_gradient",
                    4,
                )
                if tactile.contact_gradient is not None
                else (np.full((count, 4), np.nan, dtype=np.float32), count)
            )
            if gradient_n != count:
                raise ValueError(
                    f"tactile[{pad.side!r}].contact_gradient has {gradient_n} "
                    f"frames but expected {count}"
                )
            force, force_n = (
                _float_rows(tactile.force, f"tactile[{pad.side!r}].force", 3)
                if tactile.force is not None
                else (np.full((count, 3), np.nan, dtype=np.float32), count)
            )
            if force_n != count:
                raise ValueError(
                    f"tactile[{pad.side!r}].force has {force_n} frames but "
                    f"expected {count}"
                )
            seq = _int_scalar_rows(tactile.seq, f"tactile[{pad.side!r}].seq", count)
            finger_seq = (
                _int_scalar_rows(
                    tactile.finger_seq,
                    f"tactile[{pad.side!r}].finger_seq",
                    count,
                )
                if tactile.finger_seq is not None
                else seq
            )
            physx_step = (
                _int_scalar_rows(
                    tactile.physx_step,
                    f"tactile[{pad.side!r}].physx_step",
                    count,
                )
                if tactile.physx_step is not None
                else seq
            )
            timestamp = (
                _int_scalar_rows(
                    tactile.timestamp_us,
                    f"tactile[{pad.side!r}].timestamp_us",
                    count,
                )
                if tactile.timestamp_us is not None
                else np.full((count,), int(frame.timestamp_us or 0), dtype=np.int64)
            )
            rows[pad.side] = {
                "vertices": vertices,
                "n_vertices": n_vertices,
                "contact_gradient": gradient,
                "force": force,
                "contact_count": _int_scalar_rows(
                    tactile.contact_count,
                    f"tactile[{pad.side!r}].contact_count",
                    count,
                ).astype(np.int32),
                "min_sd_proxy_mm": _float_scalar_rows(
                    tactile.min_sd_proxy_mm,
                    f"tactile[{pad.side!r}].min_sd_proxy_mm",
                    count,
                ),
                "min_sd_full_mm": _float_scalar_rows(
                    tactile.min_sd_full_mm,
                    f"tactile[{pad.side!r}].min_sd_full_mm",
                    count,
                ),
                "pad_usb_min_sd_mm": _float_scalar_rows(
                    tactile.pad_usb_min_sd_mm,
                    f"tactile[{pad.side!r}].pad_usb_min_sd_mm",
                    count,
                ),
                "pad_usb_contact_vertices": _int_scalar_rows(
                    0
                    if tactile.pad_usb_contact_vertices is None
                    else tactile.pad_usb_contact_vertices,
                    f"tactile[{pad.side!r}].pad_usb_contact_vertices",
                    count,
                ).astype(np.int32),
                "tracking_error_mm": _float_scalar_rows(
                    tactile.tracking_error_mm,
                    f"tactile[{pad.side!r}].tracking_error_mm",
                    count,
                ),
                "seq": seq,
                "finger_seq": finger_seq,
                "physx_step": physx_step,
                "age_ms": _float_scalar_rows(
                    tactile.age_ms, f"tactile[{pad.side!r}].age_ms", count
                ),
                "finger_age_ms": (
                    _float_scalar_rows(
                        tactile.finger_age_ms,
                        f"tactile[{pad.side!r}].finger_age_ms",
                        count,
                    )
                    if tactile.finger_age_ms is not None
                    else np.full((count,), np.nan, dtype=np.float32)
                ),
                "finger_stale": (
                    _bool_scalar_rows(
                        tactile.finger_stale,
                        f"tactile[{pad.side!r}].finger_stale",
                        count,
                    )
                    if tactile.finger_stale is not None
                    else np.zeros((count,), dtype=np.bool_)
                ),
                "stale": _bool_scalar_rows(
                    tactile.stale, f"tactile[{pad.side!r}].stale", count
                ),
                "timestamp_us": timestamp,
                "valid": _bool_scalar_rows(
                    tactile.valid, f"tactile[{pad.side!r}].valid", count
                ),
            }
            camera_pose = _camera_pose_rows(
                tactile.camera_pose, f"tactile[{pad.side!r}].camera_pose", count
            )
            if camera_pose is not None:
                rows[pad.side]["camera_pose"] = camera_pose
            declared = set(pad.image_streams)
            for stream in _TACTILE_IMAGE_STREAMS:
                value = getattr(tactile, stream)
                if stream not in declared:
                    if value is not None:
                        raise ValueError(
                            f"tactile[{pad.side!r}].{stream} was provided but "
                            "meta.image_streams does not declare it"
                        )
                    continue
                if value is None:
                    raise ValueError(
                        f"tactile[{pad.side!r}].{stream} is required because "
                        "meta.image_streams declares it"
                    )
                if stream == "depth":
                    depth_camera = pad.depth_camera
                    if depth_camera is None:
                        raise RuntimeError(
                            f"tactile[{pad.side!r}] declares depth without a "
                            "depth_camera"
                        )
                    rows[pad.side]["depth"] = _depth_rows(
                        tactile.depth,
                        f"tactile[{pad.side!r}].depth",
                        count=count,
                        height=depth_camera.height,
                        width=depth_camera.width,
                    )
                else:
                    if pad.width is None or pad.height is None:
                        raise RuntimeError(
                            f"tactile[{pad.side!r}] declares {stream!r} without "
                            "width/height"
                        )
                    rows[pad.side][stream] = _tactile_image_rows(
                        value,
                        f"tactile[{pad.side!r}].{stream}",
                        count=count,
                        height=pad.height,
                        width=pad.width,
                    )
        return rows

    def _create_stream_datasets(
        self,
        object_rows: dict[str, dict[str, npt.NDArray[Any]]],
        tactile_rows: dict[str, dict[str, npt.NDArray[Any]]],
    ) -> None:
        if object_rows:
            objects_group = self._frames_group.create_group("objects")
            for name, rows in object_rows.items():
                group = objects_group.create_group(name)
                group.create_dataset(
                    "pose",
                    data=rows["pose"],
                    maxshape=(None, 7),
                    chunks=(1, 7),
                )
                group.create_dataset(
                    "T",
                    data=rows["T"],
                    maxshape=(None, 4, 4),
                    chunks=(1, 4, 4),
                )
                for key in ("seq", "valid"):
                    group.create_dataset(
                        key,
                        data=rows[key],
                        maxshape=(None,),
                        chunks=(1024,),
                    )
                self._object_datasets[name] = {
                    key: cast(h5py.Dataset, group[key])
                    for key in ("pose", "T", "seq", "valid")
                }
        if tactile_rows:
            tactile_group = self._frames_group.create_group("tactile")
            for side, rows in tactile_rows.items():
                group = tactile_group.create_group(side)
                vertices = cast(npt.NDArray[np.float32], rows["vertices"])
                group.create_dataset(
                    "vertices",
                    data=vertices,
                    maxshape=(None, vertices.shape[1], 3),
                    chunks=(1, vertices.shape[1], 3),
                )
                for key in (
                    "n_vertices",
                    "contact_gradient",
                    "force",
                    "contact_count",
                    "min_sd_proxy_mm",
                    "min_sd_full_mm",
                    "pad_usb_min_sd_mm",
                    "pad_usb_contact_vertices",
                    "tracking_error_mm",
                    "seq",
                    "finger_seq",
                    "physx_step",
                    "age_ms",
                    "finger_age_ms",
                    "finger_stale",
                    "stale",
                    "timestamp_us",
                    "valid",
                ):
                    value = rows[key]
                    group.create_dataset(
                        key,
                        data=value,
                        maxshape=(None, *value.shape[1:]),
                        chunks=(1024, *value.shape[1:]),
                    )
                if "camera_pose" in rows:
                    camera_pose = cast(npt.NDArray[np.float64], rows["camera_pose"])
                    group.create_dataset(
                        "camera_pose",
                        data=camera_pose,
                        maxshape=(None, 4, 4),
                        chunks=(1, 4, 4),
                    )
                if "depth" in rows:
                    depth = cast(npt.NDArray[np.float32], rows["depth"])
                    group.create_dataset(
                        "depth",
                        data=depth,
                        maxshape=(None, *depth.shape[1:]),
                        chunks=(1, *depth.shape[1:]),
                    )
                if "rgb" in rows:
                    rgb = cast(npt.NDArray[np.uint8], rows["rgb"])
                    group.create_dataset(
                        "rgb",
                        data=rgb,
                        maxshape=(None, *rgb.shape[1:]),
                        chunks=(1, *rgb.shape[1:]),
                    )
                if "marker_rgb" in rows:
                    marker_rgb = cast(npt.NDArray[np.uint8], rows["marker_rgb"])
                    group.create_dataset(
                        "marker_rgb",
                        data=marker_rgb,
                        maxshape=(None, *marker_rgb.shape[1:]),
                        chunks=(1, *marker_rgb.shape[1:]),
                    )
                group.attrs["force_kind"] = self._tactile_force_kinds[side]
                keys = [
                    "vertices",
                    "n_vertices",
                    "contact_gradient",
                    "force",
                    "contact_count",
                    "min_sd_proxy_mm",
                    "min_sd_full_mm",
                    "pad_usb_min_sd_mm",
                    "pad_usb_contact_vertices",
                    "tracking_error_mm",
                    "seq",
                    "finger_seq",
                    "physx_step",
                    "age_ms",
                    "finger_age_ms",
                    "finger_stale",
                    "stale",
                    "timestamp_us",
                    "valid",
                ]
                if "camera_pose" in rows:
                    keys.append("camera_pose")
                if "depth" in rows:
                    keys.append("depth")
                if "rgb" in rows:
                    keys.append("rgb")
                if "marker_rgb" in rows:
                    keys.append("marker_rgb")
                self._tactile_datasets[side] = {
                    key: cast(h5py.Dataset, group[key]) for key in keys
                }

    def _append_stream_datasets(
        self,
        object_rows: dict[str, dict[str, npt.NDArray[Any]]],
        tactile_rows: dict[str, dict[str, npt.NDArray[Any]]],
    ) -> None:
        for name, rows in object_rows.items():
            for key, dataset in self._object_datasets[name].items():
                self._append_rows(dataset, rows[key])
        for side, rows in tactile_rows.items():
            for key, dataset in self._tactile_datasets[side].items():
                self._append_rows(dataset, rows[key])

    @staticmethod
    def _append_rows(dataset: h5py.Dataset, rows: npt.NDArray[Any]) -> None:
        start = dataset.shape[0]
        stop = start + rows.shape[0]
        dataset.resize(stop, axis=0)
        dataset[start:stop] = rows

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        if not self._closed:
            self._file.close()
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class Hdf5EpisodeReader(StreamReader[Frame]):
    """Stream frames from one HDF5 episode file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._file = h5py.File(self._path, "r")
        self._object_datasets: dict[str, dict[str, h5py.Dataset]] = {}
        self._tactile_datasets: dict[str, dict[str, h5py.Dataset]] = {}
        self._tactile_force_kinds: dict[str, str] = {}
        try:
            version = int(cast(int, self._file.attrs["schema_version"]))
            if version != SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported schema_version {version}, "
                    f"expected {SCHEMA_VERSION}"
                )
            self._meta = self._read_stream_meta(self._read_meta())
            images_group = cast(h5py.Group, self._file["frames/images"])
            self._images = {
                camera.role: cast(h5py.Dataset, images_group[camera.role])
                for camera in self._meta.cameras
            }
            self._joint_pos = cast(h5py.Dataset, self._file["frames/joint_pos"])
            self._joint_vel = cast(h5py.Dataset, self._file["frames/joint_vel"])
            self._action = cast(h5py.Dataset, self._file["frames/action"])
            try:
                self._timestamps = cast(
                    h5py.Dataset, self._file["frames/timestamp_us"]
                )
            except KeyError:
                self._timestamps = None
            self._open_stream_frames()
            self._validate_frame_lengths()
        except Exception:
            self._file.close()
            raise
        self._index = 0
        self._closed = False

    def _validate_frame_lengths(self) -> None:
        """Require every dataset under ``/frames`` to share one frame count.

        The episode length is a property of the file, not of whichever stream a
        reader happens to trust first. A disagreement means a truncated or
        corrupted episode, so opening the file fails instead of silently
        pairing frame ``i`` of one stream with frame ``i`` of another.
        """
        frames = cast(h5py.Group, self._file["frames"])
        lengths: dict[str, int] = {}

        def walk(group: h5py.Group, prefix: str = "") -> None:
            for key in _keys(group):
                name = f"{prefix}/{key}" if prefix else key
                node = group[key]
                if isinstance(node, h5py.Group):
                    walk(node, name)
                    continue
                if not node.shape:
                    raise Hdf5SchemaError(
                        f"frames/{name} is scalar; frame streams need a time axis"
                    )
                lengths[name] = int(node.shape[0])

        walk(frames)
        if not lengths:
            raise Hdf5SchemaError("episode has no per-frame datasets")
        counts = set(lengths.values())
        if len(counts) != 1:
            detail = ", ".join(
                f"frames/{name}={count}" for name, count in sorted(lengths.items())
            )
            raise Hdf5SchemaError(f"frame datasets disagree on length: {detail}")
        self._frame_count = counts.pop()

    def _open_stream_frames(self) -> None:
        try:
            objects_group = cast(h5py.Group, self._file["frames/objects"])
        except KeyError:
            objects_group = None
        if objects_group is not None:
            for name in _keys(objects_group):
                group = cast(h5py.Group, objects_group[name])
                self._object_datasets[name] = {
                    key: cast(h5py.Dataset, group[key])
                    for key in ("pose", "T", "seq", "valid")
                }
        try:
            tactile_group = cast(h5py.Group, self._file["frames/tactile"])
        except KeyError:
            tactile_group = None
        if tactile_group is not None:
            for side in _keys(tactile_group):
                group = cast(h5py.Group, tactile_group[side])
                keys = (
                    "vertices",
                    "n_vertices",
                    "contact_gradient",
                    "force",
                    "contact_count",
                    "min_sd_proxy_mm",
                    "min_sd_full_mm",
                    "pad_usb_min_sd_mm",
                    "pad_usb_contact_vertices",
                    "tracking_error_mm",
                    "seq",
                    "finger_seq",
                    "physx_step",
                    "age_ms",
                    "finger_age_ms",
                    "finger_stale",
                    "stale",
                    "timestamp_us",
                    "valid",
                    "camera_pose",
                    "depth",
                    "rgb",
                    "marker_rgb",
                )
                self._tactile_datasets[side] = {
                    key: cast(h5py.Dataset, group[key])
                    for key in keys
                    if key in _keys(group)
                }
                self._tactile_force_kinds[side] = _attr_str(group.attrs, "force_kind")

    def _read_extra(
        self, randomization: dict[str, float | str]
    ) -> EpisodeExtra:
        attrs = self._file.attrs
        outcome = _attr_str(attrs, "outcome")
        if outcome not in _OUTCOMES:
            raise ValueError(f"invalid outcome {outcome!r}")
        return EpisodeExtra(
            seed=int(cast(int, attrs["seed"])),
            instruction=_attr_str(attrs, "instruction"),
            outcome=cast(Outcome, outcome),
            randomization=randomization,
        )

    def _read_meta(self) -> EpisodeMeta[EpisodeExtra]:
        attrs = self._file.attrs
        meta_group = cast(h5py.Group, self._file["meta"])
        joint_names = tuple(_str_list(cast(h5py.Dataset, meta_group["joint_names"])))
        action_group = cast(h5py.Group, meta_group["action"])
        representation = _attr_str(action_group.attrs, "representation")
        if representation != "abs_joint_pos":
            raise ValueError(f"unsupported action representation {representation!r}")
        action = ActionSpec(
            representation="abs_joint_pos",
            joint_names=tuple(
                _str_list(cast(h5py.Dataset, action_group["joint_names"]))
            ),
        )
        cameras_group = cast(h5py.Group, meta_group["cameras"])
        cameras = [
            self._read_camera(role, cast(h5py.Group, cameras_group[role]))
            for role in _keys(cameras_group)
        ]
        randomization: dict[str, float | str] = {}
        try:
            randomization_group = cast(h5py.Group, meta_group["randomization"])
        except KeyError:
            randomization_group = None
        if randomization_group is not None:
            for key in _keys(randomization_group.attrs):
                value = randomization_group.attrs[key]
                if isinstance(value, bytes):
                    randomization[key] = value.decode()
                elif isinstance(value, str):
                    randomization[key] = value
                else:
                    randomization[key] = float(value)
        return EpisodeMeta(
            schema_version=SCHEMA_VERSION,
            task_id=_attr_str(attrs, "task_id"),
            sim_dt=int(cast(int, attrs["sim_dt"])),
            joint_names=joint_names,
            cameras=tuple(cameras),
            action=action,
            extra=self._read_extra(randomization),
        )

    def _read_image_streams(
        self, side: str, attrs: h5py.AttributeManager
    ) -> tuple[_TactileImageStream, ...]:
        """Read a pad's declared image streams and cross-check the datasets."""
        if "image_streams" in _keys(attrs):
            raw = _attr_str(attrs, "image_streams")
            declared = tuple(part for part in raw.split(",") if part)
            unknown = sorted(set(declared) - set(_TACTILE_IMAGE_STREAMS))
            if unknown:
                raise Hdf5SchemaError(
                    f"tactile {side!r} declares unsupported image streams {unknown}"
                )
            if len(set(declared)) != len(declared):
                raise Hdf5SchemaError(
                    f"tactile {side!r} repeats an image stream: {declared}"
                )
            try:
                frames = cast(h5py.Group, self._file[f"frames/tactile/{side}"])
            except KeyError:
                frames = None
            actual = ()
            if frames is not None:
                members = _keys(frames)
                actual = tuple(
                    stream for stream in _TACTILE_IMAGE_STREAMS if stream in members
                )
            for stream in declared:
                if stream not in actual:
                    raise Hdf5SchemaError(
                        f"tactile {side!r} declares {stream!r} but the dataset "
                        "is missing"
                    )
            undeclared = sorted(set(actual) - set(declared))
            if undeclared:
                raise Hdf5SchemaError(
                    f"tactile {side!r} has undeclared image datasets {undeclared}"
                )
        else:
            raise Hdf5SchemaError(
                f"tactile {side!r} declares no image_streams"
            )
        return cast(tuple[_TactileImageStream, ...], declared)

    def _read_stream_meta(
        self, base: EpisodeMeta[EpisodeExtra]
    ) -> EpisodeMeta[EpisodeExtra]:
        attrs = self._file.attrs
        tactile_status = _attr_str(attrs, "tactile_status")
        if tactile_status not in _TACTILE_STATUSES:
            raise ValueError(f"invalid tactile_status {tactile_status!r}")
        meta_group = cast(h5py.Group, self._file["meta"])
        tactile: list[PadTactileMeta] = []
        try:
            tactile_group = cast(h5py.Group, meta_group["tactile"])
        except KeyError:
            tactile_group = None
        if tactile_group is not None:
            for side in _keys(tactile_group):
                group = cast(h5py.Group, tactile_group[side])
                pad_attrs = group.attrs
                rest = cast(
                    npt.NDArray[np.float32],
                    np.asarray(group["rest_vertices"]),
                )
                triangles = cast(
                    npt.NDArray[np.int32],
                    np.asarray(group["surface_triangles"]),
                )
                pad_frame = cast(
                    npt.NDArray[np.float64],
                    np.asarray(group["pad_frame"]),
                )
                camera_from_pad = (
                    cast(
                        npt.NDArray[np.float64],
                        np.asarray(group["camera_from_pad"]),
                    )
                    if "camera_from_pad" in _keys(group)
                    else None
                )
                tactile.append(
                    PadTactileMeta(
                        side=side,
                        rest_vertices=rest,
                        surface_triangles=triangles,
                        pad_frame=pad_frame,
                        youngs_mpa=float(
                            cast(
                                float,
                                pad_attrs[
                                    "youngs_mpa"
                                    if "youngs_mpa" in _keys(pad_attrs)
                                    else "E_MPa"
                                ],
                            )
                        ),
                        strength=float(cast(float, pad_attrs["strength"])),
                        prim_path=_attr_str(pad_attrs, "prim_path"),
                        attachment_body=_attr_str(pad_attrs, "attachment_body"),
                        max_vertices=int(cast(int, pad_attrs["max_vertices"])),
                        width=(
                            int(cast(int, pad_attrs["width"]))
                            if "width" in _keys(pad_attrs)
                            else None
                        ),
                        height=(
                            int(cast(int, pad_attrs["height"]))
                            if "height" in _keys(pad_attrs)
                            else None
                        ),
                        marker_rows=(
                            int(cast(int, pad_attrs["marker_rows"]))
                            if "marker_rows" in _keys(pad_attrs)
                            else None
                        ),
                        marker_cols=(
                            int(cast(int, pad_attrs["marker_cols"]))
                            if "marker_cols" in _keys(pad_attrs)
                            else None
                        ),
                        contact_object=(
                            _attr_str(pad_attrs, "contact_object")
                            if "contact_object" in _keys(pad_attrs)
                            else None
                        ),
                        camera_from_pad=camera_from_pad,
                        image_streams=self._read_image_streams(side, pad_attrs),
                        depth_camera=(
                            DepthCameraMeta(
                                width=int(cast(int, pad_attrs["depth_width"])),
                                height=int(cast(int, pad_attrs["depth_height"])),
                                focal_length=float(
                                    cast(float, pad_attrs["depth_focal_length"])
                                ),
                                horizontal_aperture=float(
                                    cast(
                                        float,
                                        pad_attrs["depth_horizontal_aperture"],
                                    )
                                ),
                                vertical_aperture=float(
                                    cast(
                                        float,
                                        pad_attrs["depth_vertical_aperture"],
                                    )
                                ),
                                clipping_range=cast(
                                    tuple[float, float],
                                    tuple(
                                        float(v)
                                        for v in np.asarray(
                                            pad_attrs["depth_clipping_range"],
                                            dtype=np.float64,
                                        ).reshape(-1)
                                    ),
                                ),
                                semantics=cast(
                                    Literal[
                                        "distance_to_image_plane",
                                        "distance_to_camera",
                                    ],
                                    _attr_str(pad_attrs, "depth_semantics"),
                                ),
                                surface=cast(
                                    Literal["contact_far"],
                                    _attr_str(pad_attrs, "depth_surface")
                                    if "depth_surface" in _keys(pad_attrs)
                                    else "contact_far",
                                ),
                                outside_aperture=cast(
                                    Literal["zero", "nan"],
                                    _attr_str(pad_attrs, "depth_outside_aperture")
                                    if "depth_outside_aperture" in _keys(pad_attrs)
                                    else "zero",
                                ),
                            )
                            if "depth_width" in _keys(pad_attrs)
                            else None
                        ),
                    )
                )
        objects: list[ObjectMeta] = []
        try:
            objects_group = cast(h5py.Group, meta_group["objects"])
        except KeyError:
            objects_group = None
        if objects_group is not None:
            for name in _keys(objects_group):
                group = cast(h5py.Group, objects_group[name])
                obj_attrs = group.attrs
                objects.append(
                    ObjectMeta(
                        name=name,
                        prim_path=_attr_str(obj_attrs, "prim_path"),
                        authority=cast(
                            Literal["uipc", "physx"],
                            _attr_str(obj_attrs, "authority"),
                        ),
                        constitution=_attr_str(obj_attrs, "constitution"),
                        mass_kg=(
                            float(cast(float, obj_attrs["mass_kg"]))
                            if "mass_kg" in _keys(obj_attrs)
                            else None
                        ),
                        inertia=(
                            cast(
                                npt.NDArray[np.float32],
                                np.asarray(
                                    group[
                                        "inertia"
                                        if "inertia" in _keys(group)
                                        else "inertia_diag_kg_m2"
                                    ]
                                ),
                            )
                            if "inertia" in _keys(group)
                            or "inertia_diag_kg_m2" in _keys(group)
                            else None
                        ),
                        initial_pose_xyzw=(
                            cast(
                                npt.NDArray[np.float32],
                                np.asarray(group["initial_pose_xyzw"]),
                            )
                            if "initial_pose_xyzw" in _keys(group)
                            else None
                        ),
                        usd=(
                            _attr_str(obj_attrs, "usd")
                            if "usd" in _keys(obj_attrs)
                            else None
                        ),
                        mesh_checksum=(
                            _attr_str(obj_attrs, "mesh_checksum")
                            if "mesh_checksum" in _keys(obj_attrs)
                            else None
                        ),
                        imaging=(
                            bool(np.asarray(obj_attrs["imaging"]))
                            if "imaging" in _keys(obj_attrs)
                            else False
                        ),
                    )
                )
        return replace(
            base,
            episode_id=(
                _attr_str(attrs, "episode_id") if "episode_id" in _keys(attrs) else ""
            ),
            tactile_status=cast(_TactileStatus, tactile_status),
            tactile=tuple(tactile),
            objects=tuple(objects),
        )

    @staticmethod
    def _read_camera(role: str, camera_group: h5py.Group) -> CameraMeta:
        attrs = camera_group.attrs
        try:
            focal_length = float(cast(float, attrs["focal_length"]))
        except KeyError:
            intrinsics: PinholeIntrinsics | None = None
        else:
            clipping = np.asarray(attrs["clipping_range"])
            intrinsics = PinholeIntrinsics(
                focal_length=focal_length,
                focus_distance=float(cast(float, attrs["focus_distance"])),
                horizontal_aperture=float(cast(float, attrs["horizontal_aperture"])),
                clipping_range=(float(clipping[0]), float(clipping[1])),
            )
        # A missing pose is a schema violation. Substituting identity would
        # silently place every replay camera at the world origin.
        try:
            pos_ds = cast(h5py.Dataset, camera_group["pos"])
            pos = pos_ds[0 : pos_ds.shape[0]]
            quat_ds = cast(h5py.Dataset, camera_group["quat_xyzw"])
            quat = quat_ds[0 : quat_ds.shape[0]]
        except KeyError as exc:
            raise Hdf5SchemaError(
                f"camera {role!r} is missing required pose dataset {exc.args[0]!r}"
            ) from exc
        if pos.shape != (3,) or quat.shape != (4,):
            raise Hdf5SchemaError(
                f"camera {role!r} pose must be pos (3,) and quat_xyzw (4,), "
                f"got {pos.shape} and {quat.shape}"
            )
        mount = _attr_str(attrs, "mount") if "mount" in _keys(attrs) else ""
        return CameraMeta(
            role=role,
            prim_path=_attr_str(attrs, "prim_path"),
            width=int(cast(int, attrs["width"])),
            height=int(cast(int, attrs["height"])),
            pos=(float(pos[0]), float(pos[1]), float(pos[2])),
            rot=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
            intrinsics=intrinsics,
            mount=mount or None,
        )

    @property
    def path(self) -> Path:
        return self._path

    @property
    def meta(self) -> EpisodeMeta[EpisodeExtra]:
        return self._meta

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def timestamps_us(self) -> npt.NDArray[np.int64] | None:
        """Recorded frame timestamps in microseconds, if the file has them."""
        if self._timestamps is None:
            return None
        return np.asarray(self._timestamps, dtype=np.int64)

    def __len__(self) -> int:
        return self.frame_count

    def __iter__(self) -> Self:
        return self

    def _read_object_frame(self, name: str, index: int) -> ObjectFrame:
        datasets = self._object_datasets[name]
        return ObjectFrame(
            pose_xyzw=cast(
                npt.NDArray[np.float32], np.asarray(datasets["pose"][index])
            ),
            T=cast(npt.NDArray[np.float64], np.asarray(datasets["T"][index])),
            seq=int(cast(Any, datasets["seq"][index])),
            valid=bool(cast(Any, datasets["valid"][index])),
        )

    def _read_tactile_frame(self, side: str, index: int) -> TactileFrame:
        datasets = self._tactile_datasets[side]
        return TactileFrame(
            camera_pose=(
                None
                if datasets.get("camera_pose") is None
                else cast(
                    npt.NDArray[np.float64],
                    np.asarray(datasets["camera_pose"][index]),
                )
            ),
            side=side,
            vertices=cast(
                npt.NDArray[np.float32],
                np.asarray(datasets["vertices"][index]),
            ),
            n_vertices=int(cast(Any, datasets["n_vertices"][index])),
            contact_gradient=cast(
                npt.NDArray[np.float32],
                np.asarray(datasets["contact_gradient"][index]),
            ),
            force=cast(npt.NDArray[np.float32], np.asarray(datasets["force"][index])),
            force_kind=cast(
                Literal["net_force", "contact_gradient", "none"],
                self._tactile_force_kinds[side],
            ),
            contact_count=int(cast(Any, datasets["contact_count"][index])),
            min_sd_proxy_mm=float(cast(Any, datasets["min_sd_proxy_mm"][index])),
            min_sd_full_mm=float(cast(Any, datasets["min_sd_full_mm"][index])),
            tracking_error_mm=float(cast(Any, datasets["tracking_error_mm"][index])),
            seq=int(cast(Any, datasets["seq"][index])),
            age_ms=float(cast(Any, datasets["age_ms"][index])),
            stale=bool(cast(Any, datasets["stale"][index])),
            valid=bool(cast(Any, datasets["valid"][index])),
            finger_seq=int(cast(Any, datasets["finger_seq"][index])),
            physx_step=int(cast(Any, datasets["physx_step"][index])),
            finger_age_ms=float(cast(Any, datasets["finger_age_ms"][index])),
            finger_stale=bool(cast(Any, datasets["finger_stale"][index])),
            timestamp_us=int(cast(Any, datasets["timestamp_us"][index])),
            pad_usb_min_sd_mm=(
                None
                if datasets.get("pad_usb_min_sd_mm") is None
                else float(cast(Any, datasets["pad_usb_min_sd_mm"][index]))
            ),
            pad_usb_contact_vertices=(
                0
                if datasets.get("pad_usb_contact_vertices") is None
                else int(cast(Any, datasets["pad_usb_contact_vertices"][index]))
            ),
            depth=(
                None
                if datasets.get("depth") is None
                else cast(
                    npt.NDArray[np.float32],
                    np.asarray(datasets["depth"][index]),
                )
            ),
            rgb=(
                None
                if datasets.get("rgb") is None
                else cast(
                    npt.NDArray[np.uint8],
                    np.asarray(datasets["rgb"][index]),
                )
            ),
            marker_rgb=(
                None
                if datasets.get("marker_rgb") is None
                else cast(
                    npt.NDArray[np.uint8],
                    np.asarray(datasets["marker_rgb"][index]),
                )
            ),
        )

    def __next__(self) -> Frame:
        if self._index >= self.frame_count:
            raise StopIteration
        index = self._index
        self._index += 1
        images = {
            role: cast(npt.NDArray[np.uint8], np.asarray(dataset[index]))
            for role, dataset in self._images.items()
        }
        joint_pos = cast(
            npt.NDArray[np.floating[Any]], np.asarray(self._joint_pos[index])
        )
        joint_vel = (
            cast(npt.NDArray[np.floating[Any]], np.asarray(self._joint_vel[index]))
            if self._joint_vel is not None
            else None
        )
        action = (
            cast(npt.NDArray[np.floating[Any]], np.asarray(self._action[index]))
            if self._action is not None
            else None
        )
        timestamp = (
            int(cast(Any, self._timestamps[index]))
            if self._timestamps is not None
            else None
        )
        objects = (
            {
                name: self._read_object_frame(name, index)
                for name in self._object_datasets
            }
            if self._object_datasets
            else None
        )
        tactile = (
            {
                side: self._read_tactile_frame(side, index)
                for side in self._tactile_datasets
            }
            if self._tactile_datasets
            else None
        )
        return Frame(
            images=images,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            action=action,
            tactile=tactile,
            timestamp_us=timestamp,
            objects=objects,
        )

    def close(self) -> None:
        if not self._closed:
            self._file.close()
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
