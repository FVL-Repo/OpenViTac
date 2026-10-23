from dataclasses import dataclass, field
from typing import Generic, Literal

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeVar

SCHEMA_VERSION = 3

T = TypeVar("T", default="EpisodeExtra")


@dataclass(frozen=True)
class PinholeIntrinsics:
    """Pinhole intrinsics for a sim-spawned camera."""

    focal_length: float
    focus_distance: float
    horizontal_aperture: float
    clipping_range: tuple[float, float] = (0.1, 100.0)


@dataclass(frozen=True)
class DepthCameraMeta:
    """Static description of one tactile depth camera.

    The camera is authored by the robot asset, so its pose follows the mount;
    only the calibration constants needed to lift the depth image back into a
    metric point cloud are recorded. ``semantics`` names the Isaac annotation
    used for the array: ``distance_to_image_plane`` stores perpendicular
    distance along the camera's optical axis.

    ``surface`` records which gel sheet the authored camera sees after the
    tactile near-plane fix: ``"contact_far"`` means the near sheet is clipped and
    the depth/marker image comes from the live far/contact gel surface, not from
    the object.  ``outside_aperture`` says how pixels outside the far-sheet
    footprint are represented in the depth array; the scene writes zero.
    """

    width: int
    height: int
    focal_length: float
    horizontal_aperture: float
    vertical_aperture: float
    clipping_range: tuple[float, float]
    semantics: Literal[
        "distance_to_image_plane", "distance_to_camera"
    ] = "distance_to_image_plane"
    surface: Literal["contact_far"] = "contact_far"
    outside_aperture: Literal["zero", "nan"] = "zero"


@dataclass(frozen=True)
class CameraMeta:
    """One recorded camera stream; ``role`` keys ``Frame.images``."""

    role: str  # "head" | "wrist" | ...
    prim_path: str
    width: int
    height: int
    pos: tuple[float, float, float]
    rot: tuple[float, float, float, float]  # XYZW, OpenGL convention
    intrinsics: PinholeIntrinsics | None = None  # None on USD-authored cameras
    mount: str | None = None


@dataclass(frozen=True)
class ActionSpec:
    """What one row of ``/frames/action`` means.

    The canonical recorded form is absolute joint targets; richer
    representations (ee-pose, delta, rot6d) are derived at export time, not
    baked in at record time.
    """

    representation: Literal["abs_joint_pos"]
    joint_names: tuple[str, ...]  # action column order, gripper included


@dataclass(frozen=True)
class EpisodeExtra:
    seed: int
    instruction: str
    outcome: Literal["success", "fail", "timeout", "unknown", "policy_error"]
    # Per-episode sampled randomization values (object poses, friction, ...).
    randomization: dict[str, float | str] = field(default_factory=dict)


@dataclass(frozen=True)
class PadTactileMeta:
    """Static description of one tactile pad.

    ``rest_vertices``/``surface_triangles`` are the worker's published rest
    surface, in the pad frame defined by ``pad_frame`` (world <- pad).  Vertex
    order matches every ``TactileFrame.vertices`` row, which is world-space, so
    ``pad_frame @ rest_vertices`` recovers the worker rest pose without relying
    on a second surface extraction.
    """

    side: str
    rest_vertices: npt.NDArray[np.float32]  # (V, 3), metres
    surface_triangles: npt.NDArray[np.int32]  # (F, 3)
    pad_frame: npt.NDArray[np.float64]  # (4, 4), world <- pad
    youngs_mpa: float
    strength: float
    prim_path: str
    attachment_body: str
    max_vertices: int | None = None  # defaults to rest_vertices.shape[0]
    width: int | None = None
    height: int | None = None
    marker_rows: int | None = None
    marker_cols: int | None = None
    contact_object: str | None = None
    """UIPC object whose surface the per-pad contact proxy measures (e.g. usb)."""
    depth_camera: DepthCameraMeta | None = None
    """Authored depth camera calibrated for this pad, when one is recorded."""

    image_streams: tuple[Literal["rgb", "marker_rgb", "depth"], ...] = ()
    """Image streams recorded for this pad.

    This is the schema's declaration of which per-frame datasets exist under
    ``/frames/tactile/<side>``. A writer creates exactly these datasets on the
    first append, requires every append to carry them, and rejects streams that
    were not declared. ``"depth"`` also requires ``depth_camera``; ``"rgb"`` and
    ``"marker_rgb"`` require ``width``/``height``. An empty tuple means the pad
    records no images.
    """

    @property
    def pad_key(self) -> str:
        """Generic pad key; HDF5 group names still use this value."""
        return self.side


@dataclass(frozen=True)
class ObjectMeta:
    """Static description of a recorded object."""

    name: str
    prim_path: str
    authority: Literal["uipc", "physx"]
    constitution: str
    mass_kg: float | None = None
    inertia: npt.NDArray[np.float32] | None = None  # (3, 3) or (3,)
    initial_pose_xyzw: npt.NDArray[np.float32] | None = None  # (7,)
    usd: str | None = None
    mesh_checksum: str | None = None


@dataclass(frozen=True)
class ObjectFrame:
    """One object's kinematic state in a frame or temporal chunk."""

    pose_xyzw: npt.NDArray[np.float32]  # (7,) x, y, z, qx, qy, qz, qw
    T: npt.NDArray[np.float64] | None = None  # (4, 4), world <- object
    seq: int = 0
    valid: bool = True


@dataclass(frozen=True)
class TactileFrame:
    """One pad's tactile state in a frame or temporal chunk.

    Scalars may be arrays for chunk appends. ``vertices`` is ``(V, 3)`` or
    ``(T, V, 3)``; the writer zero-pads each row to the pad's ``max_vertices``.
    ``depth`` is ``(H, W)`` or ``(T, H, W)``, in metres, when the pad has a
    depth camera. ``rgb`` is the optical tactile image from the selected backend
    over the live far/contact sheet, ``(H, W, 3)`` or ``(T, H, W, 3)`` uint8.
    ``marker_rgb`` is the independent printed-dot image: the tactile RGB
    multiplied by the projected marker mask, with the same shapes. Neither is a
    render of the visual proxy.
    """

    side: str
    vertices: npt.NDArray[np.float32] | None
    n_vertices: int | npt.NDArray[np.int32]
    contact_gradient: npt.NDArray[np.float32] | None  # (4,) or (T, 4)
    force: npt.NDArray[np.float32] | None  # (3,) or (T, 3), NaN when unavailable
    force_kind: Literal["net_force", "contact_gradient", "none"]
    contact_count: int | npt.NDArray[np.int32]
    min_sd_proxy_mm: float | npt.NDArray[np.float32]
    min_sd_full_mm: float | npt.NDArray[np.float32] | None
    tracking_error_mm: float | npt.NDArray[np.float32]
    seq: int | npt.NDArray[np.int64]
    age_ms: float | npt.NDArray[np.float32]
    stale: bool | npt.NDArray[np.bool_]
    valid: bool | npt.NDArray[np.bool_]
    finger_seq: int | npt.NDArray[np.int64] | None = None
    physx_step: int | npt.NDArray[np.int64] | None = None
    finger_age_ms: float | npt.NDArray[np.float32] | None = None
    finger_stale: bool | npt.NDArray[np.bool_] | None = None
    timestamp_us: int | npt.NDArray[np.int64] | None = None
    pad_usb_min_sd_mm: float | npt.NDArray[np.float32] | None = None
    pad_usb_contact_vertices: int | npt.NDArray[np.int32] | None = None
    depth: npt.NDArray[np.float32] | None = None  # (H, W) or (T, H, W), metres
    rgb: npt.NDArray[np.uint8] | None = None  # (H, W, 3) or (T, H, W, 3)
    marker_rgb: npt.NDArray[np.uint8] | None = None  # (H, W, 3) or (T, H, W, 3)

    @property
    def pad_key(self) -> str:
        """Generic pad key."""
        return self.side


@dataclass(frozen=True)
class EpisodeMeta(Generic[T]):
    schema_version: int
    task_id: str
    sim_dt: int  # microseconds
    joint_names: tuple[str, ...]  # joint_pos/joint_vel column order
    cameras: tuple[CameraMeta, ...]
    action: ActionSpec
    extra: T
    episode_id: str = ""
    tactile_status: Literal["ok", "disabled", "failed"] = "ok"
    tactile: tuple[PadTactileMeta, ...] = ()
    objects: tuple[ObjectMeta, ...] = ()
