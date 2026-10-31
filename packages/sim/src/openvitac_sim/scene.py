"""Scene lifecycle: build -> cameras -> reset (once) -> warmup -> step -> read.

Cameras/assets must exist before reset; frame handoff is a caller-supplied sink, so sim
stays independent of the data package; quaternions are XYZW. The scene consumes pure
``SceneSpec`` and translates it to isaaclab cfgs at build time, after ``launch()``;
this module is lazy-loaded so its isaaclab imports wait for the app. isaaclab 3.0
freezes an asset's ``.data`` at first read: ``step()``/``reset()`` refresh stepped
assets before ``read()``, so readers never see a stale snapshot.
"""

# isaaclab's generated .pyi leave its member types partially Unknown (ProxyArray
# surfaces); the boundary reads below are runtime-verified.
# The arm/gripper handles are Scene's own adapters and intentionally read its
# internals; they do not widen Scene's public state.

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal, cast

import isaaclab.sim as sim_utils
import numpy as np
import numpy.typing as npt
import torch
from isaaclab.assets import RigidObject
from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.spawners.lights import LightCfg
from pxr import Gf, Usd, UsdGeom

from openvitac_spec import (
    ActorPose,
    EmbodimentSpec,
    GripperCapability,
    SceneSpec,
    SensorBinding,
    SettleProtocol,
)
from openvitac_tactile.config import TactileCfg
from openvitac_tactile.observation import TactileObservation
from openvitac_tactile.protocol import FingerSample
from openvitac_tactile.softworld import ObjectHandle

from .assets import asset
from .gelpad_visual import GelpadVisual, visual_update_due
from .robot import assemble_robot
from .sensor import Sensor, make_sensor, sensor_class, sensor_has_pad_proxy
from .sensor.base import ContactObject, PadCameraRequest, World, WorldBuilder
from .sensor.gsmini import BoundGsMiniSensor
from .sensor.ids import BodyId, CameraId, Pose
from .sensor.snapshot import DeformableState, Requires, RigidState, TickSnapshot
from .spawn import (
    create_joint,
    disable_named_collisions,
    spawn_camera,
    spawn_light,
    spawn_rigid,
    spawn_static,
    sync_kinematic_proxy,
)
from .spec import CameraSpec, DepthCameraCfg, LightSpec
from .tactile_bridge import (
    PadSurface,
    TactileBridge,
    body_pose_to_T,
    finger_sample_from_body_poses,
)
from .tactile_plan import tactile_plan_from_scene

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # XYZW


def _ordered_body_ids(robot: Any, names: Sequence[str]) -> list[int]:
    """Resolve body names in request order, failing loud on any mismatch.

    IsaacLab's ``find_bodies`` does not preserve request order by default, so
    callers must never index the result by the requested sequence.  We ask for
    preserved order and verify the returned names before using the indices.
    """
    requested = list(names)
    if len(set(requested)) != len(requested):
        raise ValueError(f"duplicate body names requested: {requested}")
    body_ids, matched = robot.find_bodies(requested, preserve_order=True)
    ids = [int(index) for index in cast(list[int], body_ids)]
    matched_names = [str(name) for name in matched]
    if matched_names != requested or len(ids) != len(requested):
        raise RuntimeError(
            f"find_bodies returned names {matched_names} / ids {ids} for "
            f"request {requested}"
        )
    return ids


def _unsupported_render_requests(spec: SceneSpec, tactile: TactileCfg) -> list[str]:
    """Declared tactile render streams that no sim optical backend can produce."""
    bindings = {binding.pad_key: binding for binding in spec.sensors}
    unsupported: list[str] = []
    for pad in tactile.pads:
        if not pad.render_kinds:
            continue
        binding = bindings.get(pad.name)
        if binding is not None and sensor_class(binding.sensor.kind) is not None:
            continue
        kind = "unbound" if binding is None else binding.sensor.kind
        unsupported.append(f"{pad.name} ({kind}): {list(pad.render_kinds)}")
    return unsupported


def _require_supported_render_streams(spec: SceneSpec, tactile: TactileCfg) -> None:
    """Refuse to run a scene whose declared tactile streams cannot be recorded.

    XenseWS pads have no sim-side optical pipeline, so a requested ``rgb``
    would otherwise be dropped while the schema still advertises it.
    """
    unsupported = _unsupported_render_requests(spec, tactile)
    if unsupported:
        raise NotImplementedError(
            "tactile render streams have no sim-side optical pipeline for this "
            "sensor kind; refusing to run with declared streams that cannot be "
            "recorded: " + ", ".join(unsupported)
        )


def _requires_pad_proxy_disabling(
    spec: SceneSpec, bridge: TactileBridge | None
) -> bool:
    """Whether the UIPC path must remove the sensor ``pad_proxy`` cubes.

    Only registered sensor kinds that declare ``pad_proxy`` carry those cubes in
    their case USD.  A UIPC scene with any other sensor mount (e.g. the XenseWS
    adapters) has no ``pad_proxy`` colliders to remove, so demanding one would
    reject a valid scene.
    """
    if bridge is None:
        return False
    if not any(sensor_has_pad_proxy(binding.sensor.kind) for binding in spec.sensors):
        return False
    return any(
        actor.dynamic and actor.authority == "uipc" for actor in spec.actors
    )


def _prim_world_transform(prim_path: str) -> npt.NDArray[np.float64]:
    """World transform of an authored prim (e.g. a gelpad) at the default time."""
    stage = sim_utils.get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"missing prim {prim_path!r} for pad calibration")
    matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    return np.array(
        [[matrix[row][col] for col in range(4)] for row in range(4)],
        dtype=np.float64,
    ).T


def _set_static_visual_pose(prim_path: str, pose: ActorPose) -> None:
    """Move a static UIPC-fixed fixture's USD visual root to its actor pose."""
    stage = sim_utils.get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"missing static actor prim {prim_path!r} for visual sync")
    xform = UsdGeom.Xformable(prim)

    qx, qy, qz, qw = pose.quaternion_xyzw
    desired = Gf.Matrix4d(1.0)
    desired.SetRotateOnly(Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz)))
    desired.SetTranslateOnly(Gf.Vec3d(*pose.position))
    parent = prim.GetParent()
    parent_world = UsdGeom.XformCache(
        Usd.TimeCode.Default()
    ).GetLocalToWorldTransform(parent)
    local = desired * parent_world.GetInverse()

    ops = {
        op.GetOpName(): op
        for op in cast(Sequence[UsdGeom.XformOp], xform.GetOrderedXformOps() or ())
    }
    translate_op = ops["xformOp:translate"]
    orient_op = ops["xformOp:orient"]
    translate_op.Set(local.ExtractTranslation())
    rotation = local.ExtractRotationQuat()
    orient_op.Set(Gf.Quatd(rotation.GetReal(), rotation.GetImaginary()))

    world = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
        prim
    )
    actual = world.ExtractTranslation()
    pos_error = float(
        np.linalg.norm(np.asarray(actual, dtype=np.float64) - np.asarray(pose.position))
    )
    actual_q = world.ExtractRotationQuat()
    actual_i = actual_q.GetImaginary()
    actual_xyzw = (
        float(actual_i[0]),
        float(actual_i[1]),
        float(actual_i[2]),
        float(actual_q.GetReal()),
    )
    dot = abs(sum(a * b for a, b in zip(actual_xyzw, (qx, qy, qz, qw), strict=True)))
    if pos_error >= 1e-6 or abs(dot - 1.0) >= 1e-7:
        raise RuntimeError(
            f"static visual pose mismatch for {prim_path!r}: "
            f"position error {pos_error:.3e} m, quaternion dot {dot:.9f}"
        )


def _camera_cfg(spec: CameraSpec, update_period: float) -> CameraCfg:
    """Translate a pure-data camera spec into a ``CameraCfg``.

    The update period tracks the sim dt (a run parameter, not task content). A spec
    without intrinsics is a USD-authored (wrist) camera: nothing is spawned, and the
    camera re-reads its pose from the mount prim every update.
    """
    if spec.intrinsics is None:
        return CameraCfg(
            prim_path=spec.prim_path,
            update_period=update_period,
            height=spec.height,
            width=spec.width,
            data_types=list(spec.data_types),
            spawn=None,
            update_latest_camera_pose=True,
        )
    return CameraCfg(
        prim_path=spec.prim_path,
        update_period=update_period,
        height=spec.height,
        width=spec.width,
        data_types=list(spec.data_types),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=spec.intrinsics.focal_length,
            focus_distance=spec.intrinsics.focus_distance,
            horizontal_aperture=spec.intrinsics.horizontal_aperture,
            clipping_range=spec.intrinsics.clipping_range,
        ),
        offset=CameraCfg.OffsetCfg(pos=spec.pos, rot=spec.rot, convention="opengl"),
    )


def _camera_matrix(spec: DepthCameraCfg) -> npt.NDArray[np.float64]:
    """Pinhole pixel intrinsics from the authored USD camera calibration."""
    fx = spec.focal_length / spec.horizontal_aperture * spec.width
    fy = spec.focal_length / spec.vertical_aperture * spec.height
    return np.array(
        [
            [fx, 0.0, spec.width / 2.0],
            [0.0, fy, spec.height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _light_cfg(spec: LightSpec) -> LightCfg:
    if spec.texture_file is None:
        return sim_utils.DomeLightCfg(color=spec.color, intensity=spec.intensity)
    return sim_utils.DomeLightCfg(
        color=spec.color,
        intensity=spec.intensity,
        texture_file=asset(spec.texture_file),
    )


class StepLimitExceeded(RuntimeError):
    """A scene step was attempted after the configured step cap was reached.

    Raised by :meth:`Scene.step` when :meth:`Scene.set_step_limit` has been armed;
    callers treat it as a rollout that ran out of budget (``outcome="timeout"``).
    """


def recorded_step_indices(steps: int, every: int) -> tuple[int, ...]:
    """0-based physics-step indices at which a throttled step sink emits.

    ``every=1`` records every step. The first step is always recorded, so
    ``steps=5, every=2`` yields ``(0, 2, 4)``: three frames whose timestamps are
    ``index * sim_dt_us`` (0, 16666, 33332 μs for the 120 Hz sim).
    """
    if every <= 0:
        raise ValueError(f"record cadence must be positive, got {every}")
    return tuple(index for index in range(steps) if index % every == 0)


@dataclass(frozen=True)
class SceneObservation:
    """One stepped frame: per-camera rgb, the arm's joint state, and its target."""

    rgb: dict[str, npt.NDArray[np.uint8]]  # camera prim_path -> (H, W, 3) uint8
    joint_pos: npt.NDArray[np.floating]
    joint_vel: npt.NDArray[np.floating]
    action: npt.NDArray[np.floating]  # joint-position target, joint_pos column order
    tactile: TactileObservation | None = None
    # Exactly the poses the imaging pass consumed this frame, so a recording
    # can replay the renderer offline bit-for-bit: pad side -> world <- camera,
    # contact object name -> world <- object.
    tactile_cameras: Mapping[str, npt.NDArray[np.float64]] = field(
        default_factory=dict
    )
    tactile_objects: Mapping[str, npt.NDArray[np.float64]] = field(
        default_factory=dict
    )
    # 0-based index of the physics step that produced this frame. Timestamps are
    # this index * sim_dt, not the recorded-frame number * sim_dt: at the default
    # every=2 the two differ by the recording cadence.
    physics_step: int = 0


@dataclass(frozen=True)
class SettleResult:
    """Outcome of :meth:`Scene.settle`.

    ``converged`` is ``None`` for the legacy fixed-step mode, where no pose
    criterion is evaluated.  Deltas are the largest window changes observed
    while checking convergence.
    """

    requested_steps: int
    executed_steps: int
    converged: bool | None = None
    max_pos_delta_m: float | None = None
    max_rot_delta_rad: float | None = None


def _max_pose_delta(
    before: Mapping[str, npt.NDArray[np.float64]],
    after: Mapping[str, npt.NDArray[np.float64]],
) -> tuple[float, float]:
    """Largest translation (m) and rotation (rad) change across actors."""
    max_pos = 0.0
    max_rot = 0.0
    for path, previous in before.items():
        current = after.get(path)
        if current is None:
            continue
        max_pos = max(max_pos, float(np.linalg.norm(current[:3] - previous[:3])))
        dot = abs(float(np.dot(previous[3:7], current[3:7])))
        angle = 2.0 * math.acos(min(1.0, max(-1.0, dot)))
        max_rot = max(max_rot, angle)
    return max_pos, max_rot


class _SensorWorld:
    """Bind-time world access for a scene's gelpad sensors."""

    def __init__(
        self,
        scene: Scene,
        bindings: Mapping[str, SensorBinding],
        body_frames: Mapping[str, npt.NDArray[np.float64]],
    ) -> None:
        self._scene = scene
        self._bindings = bindings
        self._body_frames = body_frames
        self._camera_local: dict[str, npt.NDArray[np.float64]] = {}

    def _bridge(self) -> TactileBridge:
        bridge = self._scene._tactile_bridge
        if bridge is None:
            raise RuntimeError("tactile bridge is not started")
        return bridge

    def _pad(self, body: BodyId) -> Any:
        bridge = self._bridge()
        return bridge.pad_config(bridge.pad_handles[str(body)])

    def _camera_prim_path(self, body: BodyId, pad: Any) -> str:
        camera_prim = self._bindings[str(body)].mount.camera_prim
        if camera_prim is not None:
            return camera_prim
        return f"{pad.frame.body_prim_path}/Camera"

    def pad_surface(
        self, body: BodyId
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
        bridge = self._bridge()
        surface = bridge.pad_surface(bridge.pad_handles[str(body)])
        return surface.rest_vertices, surface.triangles

    def rest_camera_pose(self, body: BodyId) -> npt.NDArray[np.float64]:
        camera_local = self._camera_local.get(body)
        if camera_local is None:
            pad = self._pad(body)
            # The USD body/camera pair is authored in the default robot pose;
            # compose its fixed relative transform with the actual PhysX body
            # pose.  The CameraData.pose is still the authored pose before the
            # first stepped render, so it cannot be used here.
            body_usd = _prim_world_transform(pad.frame.body_prim_path)
            camera_usd = _prim_world_transform(
                self._camera_prim_path(body, pad)
            )
            camera_local = np.linalg.inv(body_usd) @ camera_usd
            self._camera_local[body] = camera_local
            self._scene._aperture_camera_local[pad.side] = camera_local
            self._scene._aperture_pads[pad.side] = pad
        return self._body_frames[str(body)] @ camera_local

    def pad_frame(self, body: BodyId) -> npt.NDArray[np.float64]:
        """World <- pad calibration frame of the gelpad at the bind instant."""
        bridge = self._bridge()
        return bridge.pad_world_frames[str(body)]

    def camera_intrinsics(self, body: BodyId) -> npt.NDArray[np.float64]:
        pad = self._pad(body)
        width, height = pad.image_width, pad.image_height
        if width is None or height is None:
            raise RuntimeError(f"rendering pad {str(body)!r} has no image size")
        camera_spec = self._scene._authored_depth_cfg(
            self._camera_prim_path(body, pad), width, height
        )
        return _camera_matrix(camera_spec)


class _SceneProxy:
    """Cosmetic gelpad display mesh owned by one bound sensor."""

    def __init__(self, scene: Scene, prim_path: str, stage: Any) -> None:
        self._scene = scene
        self._prim_path = prim_path
        self._stage = stage
        self._visual: GelpadVisual | None = None

    def update(
        self, vertices: npt.NDArray[Any], triangles: npt.NDArray[Any]
    ) -> None:
        """Bind on the first call, then publish only on due render frames."""
        if self._visual is None:
            self._visual = GelpadVisual.bind(
                side=self._prim_path,
                gelpad_prim_path=self._prim_path,
                rest_vertices=vertices,
                triangles=triangles,
                stage=self._stage,
            )
            self._scene._visual_changed = True
            return
        if not self._scene._visual_due:
            return
        self._visual.update(vertices)
        self._scene._visual_changed = True


class _SceneSoftBody:
    """One bridge pad exposed as the sensor-owned soft body."""

    def __init__(self, bridge: TactileBridge, pad_key: str) -> None:
        self._bridge = bridge
        self._pad_key = pad_key

    @property
    def id(self) -> BodyId:
        return BodyId(self._pad_key)

    def stage(self, pose: Pose) -> None:
        self._bridge.stage_finger(self._pad_key, pose.matrix)


class _SceneBuilder:
    """The narrow build API the scene hands to each sensor's ``attach``."""

    def __init__(self, scene: Scene) -> None:
        self._scene = scene
        self._pad_proxy_disabled = False

    def body(self, name: str) -> BodyId | None:
        # Articulation data is not initialized before reset, so resolve against
        # the authored sensor mounts; PhysX resolves the index after reset.
        if any(
            binding.body_name == name for binding in self._scene._spec.sensors
        ):
            return BodyId(name)
        return None

    def add_soft_body(self, pad_key: str) -> _SceneSoftBody:
        bridge = self._scene._tactile_bridge
        if bridge is None:
            raise RuntimeError("tactile bridge is not configured")
        if pad_key not in bridge.pad_handles:
            raise RuntimeError(
                f"tactile pad {pad_key!r} is not in the bridge config"
            )
        return _SceneSoftBody(bridge, pad_key)

    def add_pad_camera(
        self, prim_path: str, spec: PadCameraRequest
    ) -> CameraId:
        """Register the authored pad camera's calibration; spawn nothing.

        The sensor consumes its intrinsics and its fixed pose relative to the
        pad.  There is no depth render product: the recorded depth stream is
        analytic object depth, and the rest reference is a geometric plane.
        """
        stage = sim_utils.get_current_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
            raise RuntimeError(f"missing authored pad camera {prim_path!r}")
        self._scene._tactile_depth_specs[spec.side] = (
            self._scene._authored_depth_cfg(prim_path, spec.width, spec.height)
        )
        return CameraId(spec.side)

    def contact_objects(self) -> tuple[ContactObject, ...]:
        """Dynamic scene actors, resolved structurally from the spec.

        ``Actor.dynamic`` is the declaration: a dynamic actor is a thing the
        gripper manipulates, so it may touch a pad; a static actor is a support
        or fixture and is never imaged.  No name or label is inspected.
        Sensor-owned prims are rejected structurally: the gelpad/case is the
        sensor, never an object.
        """
        spec = self._scene._spec
        owned = [
            prim
            for binding in spec.sensors
            for prim in (
                binding.body_prim_path,
                binding.case_prim_path,
                binding.gelpad_prim_path,
            )
        ]
        objects: list[ContactObject] = []
        seen: set[str] = set()
        for actor in spec.actors:
            if not actor.dynamic:
                continue
            if actor.constitution == "fem":
                raise NotImplementedError(
                    f"contact object {actor.prim_path!r} is FEM-deformable; the "
                    "tactile exchange carries no live object surface, so imaging "
                    "would silently use its rest mesh instead of its deformed one"
                )
            for prim in owned:
                if actor.prim_path == prim or actor.prim_path.startswith(f"{prim}/"):
                    raise RuntimeError(
                        f"actor {actor.prim_path!r} is part of sensor {prim!r}; a "
                        "sensor's own geometry is never a contact object"
                    )
            name = actor.prim_path.rsplit("/", 1)[-1]
            if name in seen:
                raise RuntimeError(f"duplicate contact object name {name!r}")
            seen.add(name)
            objects.append(
                ContactObject(
                    id=BodyId(name),
                    prim_path=actor.prim_path,
                    usd_path=asset(actor.usd),
                )
            )
        return tuple(objects)

    def add_visual_proxy(self, prim_path: str) -> _SceneProxy:
        return _SceneProxy(
            self._scene, prim_path, sim_utils.get_current_stage()
        )

    def remove_pad_proxy(self, body: BodyId) -> None:
        """Remove authored ``pad_proxy`` colliders on a UIPC-authored scene."""
        if self._pad_proxy_disabled:
            return
        spec = self._scene._spec
        if not _requires_pad_proxy_disabling(
            spec, self._scene._tactile_bridge
        ):
            return
        disabled = disable_named_collisions(spec.robot_prim_path, "pad_proxy")
        self._pad_proxy_disabled = True
        if disabled == 0:
            raise RuntimeError(
                "UIPC scene with GsMini/Neote pads found no pad_proxy "
                "colliders under the robot root"
            )

    def frames(self) -> Mapping[str, Pose]:
        frames: dict[str, Pose] = {}
        bridge = self._scene._tactile_bridge
        if bridge is None:
            return frames
        for name, frame in bridge.pad_world_frames.items():
            frames[name] = Pose(frame)
        return frames


class Scene:
    """Assembles a scene from a spec and drives its fixed lifecycle.

    Cameras/assets must exist before reset: a camera created after reset never
    renders because its render product dies when the timeline stops. ``Scene``
    also satisfies the tasks package's ``ExpertWorld`` protocol structurally, and
    its ``arm`` handle satisfies ``ArmHandle``; the cli holds the conformance
    assertions.

    Episode order: ``reset -> settle -> start_tactile -> step``.  Tactile t0 is
    the settle-end state, so pre-rollout steps do not send finger samples.
    """

    # Partially constructed scenes (tests built via __new__) have no requirements.
    _sensor_requires: Requires = Requires()

    def __init__(
        self,
        spec: SceneSpec,
        sim_cfg: SimulationCfg,
        *,
        tactile_enabled: bool = True,
        optical_backend: Literal["taxim", "procedural"] = "taxim",
        taxim_device: str = "cuda:0",
        seed: int = 0,
    ) -> None:
        self._spec = spec
        self._logger: logging.LoggerAdapter[logging.Logger] | None = None
        if seed < 0:
            raise ValueError(f"episode seed must be non-negative, got {seed}")
        self._episode_seed = int(seed)
        tactile = (
            tactile_plan_from_scene(spec)
            if tactile_enabled and spec.sensors
            else None
        )
        if tactile is not None and tactile.enabled:
            _require_supported_render_streams(spec, tactile)
        self._sim = SimulationContext(sim_cfg)
        self._dt = sim_cfg.dt
        self._cameras: dict[str, Camera] = {}
        self._rigids: dict[str, RigidObject] = {}
        self._actor_pose_overrides: dict[str, ActorPose] = {}
        self._expert_params: dict[str, str | float] = {}
        # object name -> kinematic PhysX proxy following a UIPC-authority actor
        self._uipc_proxies: dict[str, RigidObject] = {}
        self._uipc_object_handles: dict[str, ObjectHandle] = {}
        self._reset_done = False
        self._settled = False
        self._step_sink: Callable[[SceneObservation], None] | None = None
        self._step_every = 1
        # 0-based index of the last completed physics step; stamped into the
        # observation so frame timestamps track physics time, not frame number.
        self._last_step_index = 0
        self._step_limit: int | None = None
        self._steps_taken = 0
        # Monotonic across settle: tactile finger samples and worker state use
        # this counter, while _steps_taken restarts at 0 for episode timestamps.
        self._physics_steps_total = 0
        self._tactile_bridge: TactileBridge | None = None
        self._tactile_started = False
        self._tactile_state: TactileObservation | None = None
        self._tactile_depth_specs: dict[str, DepthCameraCfg] = {}
        self._tactile_depth_record_sides: set[str] = {
            binding.pad_key
            for binding in spec.sensors
            if "depth" in binding.streams
        }
        self._tactile_sensors: dict[str, BoundGsMiniSensor] = {}
        self._attached_sensors: dict[str, Sensor] = {}
        self._contact_object_paths: dict[str, str] = {}
        self._tactile_camera_frames: dict[str, npt.NDArray[np.float64]] = {}
        self._tactile_height_maps: dict[str, npt.NDArray[np.float64]] = {}
        self._tactile_frame_cameras: dict[str, npt.NDArray[np.float64]] = {}
        self._tactile_frame_objects: dict[str, npt.NDArray[np.float64]] = {}
        self._visual_due = False
        self._visual_changed = False
        self._optical_backend = optical_backend
        self._taxim_device = taxim_device
        self._aperture_camera_local: dict[str, npt.NDArray[np.float64]] = {}
        self._aperture_pads: dict[str, Any] = {}
        self._body_index_cache: dict[tuple[str, ...], list[int]] = {}
        if tactile is not None and tactile.enabled:
            self._tactile_bridge = TactileBridge(
                tactile, run_id=f"scene_{id(self):x}", group_id="0"
            )
        spawn_light(_light_cfg(spec.light))
        joint_pos = dict(
            spec.embodiment.default_joint_pos
            if spec.joint_pos is None
            else spec.joint_pos
        )
        for binding in spec.sensors:
            for joint_name, offset in binding.mount.joint_offsets.items():
                joint_pos[joint_name] = joint_pos.get(joint_name, 0.0) + offset
        self._arm = assemble_robot(spec.embodiment, spec.robot_prim_path, joint_pos)
        self._arm_handle = SceneArm(self)
        if spec.embodiment.gripper.kind == "robotiq_85":
            # The XenseWS tactile adapters ride the inner fingers ~85 mm off the
            # tool axis.  They are sensor mounts, not jaw geometry, and on a plain
            # grasp they sweep a standing object aside before the pads can close.
            disable_named_collisions(spec.robot_prim_path, prefix="xense")
        for actor in spec.actors:
            if not actor.dynamic:
                if actor.authority != "physx":
                    raise ValueError(
                        f"static actor {actor.prim_path!r} must use physx authority"
                    )
                spawn_static(asset(actor.usd), actor.prim_path, actor.pos, actor.rot)
                continue
            kinematic = actor.authority == "uipc"
            rigid = spawn_rigid(
                asset(actor.usd),
                actor.prim_path,
                actor.pos,
                actor.rot,
                kinematic=kinematic,
                density=actor.density,
            )
            self._rigids[actor.prim_path] = rigid
            if kinematic:
                self._uipc_proxies[actor.prim_path.rsplit("/", 1)[-1]] = rigid
        for joint in spec.joints:
            create_joint(joint)
        for cam_spec in spec.cameras:
            cam_cfg = _camera_cfg(cam_spec, self._dt)
            self._cameras[cam_cfg.prim_path] = spawn_camera(cam_cfg)
        if tactile is not None and tactile.enabled:
            self._build_tactile_sensors()
        # PhysX contact reporting on the gripper's finger pad bodies (the robot spawns
        # with activate_contact_sensors); feeds the shared gripper_contact query.
        # The stub marks the base class abstract; the factory __new__ dispatches to
        # the PhysX backend at runtime.
        self._contact = cast(Any, ContactSensor)(
            ContactSensorCfg(
                prim_path=(
                    f"{spec.robot_prim_path}"
                    f"/({'|'.join(self._arm.embodiment.gripper.contact_bodies)})"
                )
            )
        )

    def _build_tactile_sensors(self) -> None:
        """Construct and attach each registered sensor before the scene resets."""
        self._attached_sensors = {}
        self._sensor_requires = Requires()
        bridge = self._tactile_bridge
        if bridge is None:
            return
        registered = [
            binding
            for binding in self._spec.sensors
            if sensor_class(binding.sensor.kind) is not None
        ]
        if not registered:
            return
        contact = list(self._arm.embodiment.gripper.contact_bodies)
        if len(contact) != 2:
            raise RuntimeError(
                f"expected 2 tactile attachment bodies, got {contact}"
            )
        finger_by_side = {"left": contact[0], "right": contact[1]}
        builder: WorldBuilder = _SceneBuilder(self)
        objects = builder.contact_objects()
        self._contact_object_paths = {str(obj.id): obj.prim_path for obj in objects}
        self._tactile_camera_frames = {}
        for binding in registered:
            sensor = make_sensor(
                binding.sensor.kind,
                binding=binding,
                pad_key=binding.pad_key,
                optical_backend=self._optical_backend,
                taxim_device=self._taxim_device,
                finger_body=finger_by_side.get(binding.pad_key),
            )
            sensor.attach(builder)
            self._attached_sensors[binding.pad_key] = sensor
            self._sensor_requires = self._sensor_requires | sensor.requires()

    @staticmethod
    def _authored_depth_cfg(
        prim_path: str, width: int, height: int
    ) -> DepthCameraCfg:
        """Read the authored camera's calibration; never fabricate intrinsics."""
        stage = sim_utils.get_current_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"missing authored depth camera {prim_path!r}")

        def attr(name: str) -> Any:
            attribute = prim.GetAttribute(name)
            if not attribute:
                raise RuntimeError(
                    f"authored depth camera {prim_path!r} lacks {name!r}"
                )
            return attribute.Get()

        clipping = tuple(float(v) for v in attr("clippingRange"))
        if len(clipping) != 2:
            raise RuntimeError(
                f"authored depth camera {prim_path!r} has invalid clippingRange"
            )
        return DepthCameraCfg(
            prim_path=prim_path,
            width=width,
            height=height,
            focal_length=float(attr("focalLength")),
            horizontal_aperture=float(attr("horizontalAperture")),
            vertical_aperture=float(attr("verticalAperture")),
            clipping_range=clipping,
        )

    @property
    def tactile_episode_cfg(self) -> TactileCfg | None:
        """The tactile config of the latest reset (per-episode contact object)."""
        bridge = self._tactile_bridge
        return bridge.episode_cfg if bridge is not None else None

    @property
    def tactile_depth_specs(self) -> Mapping[str, DepthCameraCfg]:
        """Pad side -> authored depth calibration for pads recording depth.

        Taxim-only cameras are intentionally excluded: the HDF5 metadata
        contract pairs ``depth_camera`` with the declared ``depth`` stream.
        """
        return {
            side: spec
            for side, spec in self._tactile_depth_specs.items()
            if side in self._tactile_depth_record_sides
        }

    def tactile_pad_world_frames(self) -> Mapping[str, npt.NDArray[np.float64]]:
        """Pad name -> world <- pad calibration from the latest reset."""
        bridge = self._tactile_bridge
        if bridge is None:
            return {}
        return bridge.pad_world_frames

    def tactile_camera_frames(self) -> Mapping[str, npt.NDArray[np.float64]]:
        """Pad side -> camera <- pad calibration used by the imaging pass."""
        return MappingProxyType(self._tactile_camera_frames)

    def tactile_height_maps(self) -> Mapping[str, npt.NDArray[np.float64]]:
        """Pad side -> the latest tick's contact height map, in millimetres."""
        return MappingProxyType(self._tactile_height_maps)

    def tactile_pad_surfaces(self) -> Mapping[str, PadSurface]:
        """Pad name -> worker rest surface (world-space, frame-vertex order)."""
        bridge = self._tactile_bridge
        if bridge is None:
            return cast(dict[str, PadSurface], {})
        return bridge.pad_surfaces_by_name

    def _body_ids(self, names: Sequence[str]) -> list[int]:
        """Cached body indices resolved in ``names`` order."""
        key = tuple(names)
        cache: dict[tuple[str, ...], list[int]] = getattr(
            self, "_body_index_cache", {}
        )
        cached = cache.get(key)
        if cached is None:
            cached = _ordered_body_ids(self._arm.robot, names)
            cache[key] = cached
            self._body_index_cache = cache
        return cached

    def _apply_actor_poses(self, overrides: Mapping[str, ActorPose]) -> None:
        """Write reset-time PhysX poses and retain static UIPC-fixed overrides.

        Full prim paths are canonical; a unique leaf name is accepted as a
        convenience.  Dynamic actors must have a spawned RigidObject (including
        UIPC kinematic proxies).  Static actors are rejected unless they are
        UIPC fixed collision copies, which have no PhysX body and are applied by
        the worker's reset request instead.
        """
        actors = {actor.prim_path: actor for actor in self._spec.actors}
        by_leaf: dict[str, str] = {}
        for path in actors:
            leaf = path.rsplit("/", 1)[-1]
            if leaf in by_leaf:
                raise ValueError(f"ambiguous actor leaf name {leaf!r}")
            by_leaf[leaf] = path
        normalized: dict[str, ActorPose] = {}
        for key, pose in overrides.items():
            path = key if key in actors else by_leaf.get(key)
            if path is None:
                raise ValueError(f"actor_poses key {key!r} matches no scene actor")
            actor = actors[path]
            if actor.dynamic:
                rigid = self._rigids.get(path)
                if rigid is None:
                    raise RuntimeError(
                        f"dynamic actor {path!r} has no spawned RigidObject"
                    )
                sync_kinematic_proxy(
                    rigid,
                    [*pose.position, *pose.quaternion_xyzw],
                    device=self._arm.robot.device,
                )
            elif actor.uipc_fixed_boxes or actor.uipc_fixed_mesh is not None:
                # The worker owns this fixed collision copy; it is built from
                # _tactile_object_poses() during TactileBridge.reset().  Move
                # the USD visual root with the same override so the rendered
                # fixture and the UIPC fixed body share one world pose.
                _set_static_visual_pose(actor.prim_path, pose)
            else:
                raise ValueError(
                    f"actor {path!r} is static and has no UIPC fixed body; "
                    "actor pose randomization is unsupported"
                )
            normalized[path] = pose
        self._actor_pose_overrides = normalized

    def reset(self, *, actor_poses: Mapping[str, ActorPose] | None = None) -> None:
        if self._reset_done:
            raise RuntimeError("Scene.reset() must run exactly once per scene")
        self._sim.reset()
        # DR must land before settle, and before the tactile bridge builds the
        # worker's UIPC fixed objects (which cannot move after world.init()).
        self._apply_actor_poses(actor_poses or {})
        # isaaclab 3.0 only seeds default_joint_pos on reset; write the init poses.
        self._arm.reset_to_init()
        self._reset_done = True
        settles = [
            binding.sensor.settle
            for binding in self._spec.sensors
            if binding.sensor.settle is not None
        ]
        if settles:
            settle = SettleProtocol(
                stable_steps=max(item.stable_steps for item in settles),
                max_steps=max(item.max_steps for item in settles),
            )
            self._run_settle(settle)
        # Settle steps ran without a data refresh; publish the settled state so a
        # read before the first step() sees it, not the USD-authored defaults.
        self._update_actors()

    def start_tactile(self, *, contact_object: str | None = None) -> None:
        """Start the tactile worker and capture its t0 reference.

        ``contact_object`` is the actor prim path of the object whose pad contact
        metrics this episode reports; it overrides ``Actor.contact_object`` for
        this reset only, so a task with several UIPC objects can pick one per
        episode.

        Must run after :meth:`reset` and :meth:`settle`: DR and pre-rollout
        physics must finish first so the reference frames and object poses are
        the settle-end state.  The tactile sequence clock restarts here, so the
        first rollout sample is seq/physx_step 1.  No-op for scenes without a
        bridge.
        """
        if self._tactile_bridge is None or self._tactile_started:
            return
        if not self._reset_done:
            raise RuntimeError("Scene.start_tactile() requires Scene.reset() first")
        if not self._settled:
            raise RuntimeError(
                "Scene.start_tactile() requires Scene.settle() first: the "
                "tactile t0 reference must be the settle-end state"
            )
        self._tactile_bridge.start()
        body_frames = self._tactile_attachment_frames()
        self._tactile_bridge.reset(
            pad_frames=body_frames,
            pad_body_frames=self._tactile_body_pad_frames(body_frames),
            object_poses=self._tactile_object_poses(),
            seed=self._episode_seed,
            contact_object=contact_object,
        )
        self._uipc_object_handles = {
            name: self._tactile_bridge.object_handle(name)
            for name in self._uipc_proxies
        }
        self._bind_sensors()
        # Tactile t0 is this instant: start its own sequence at 0 so the first
        # rollout finger sample is seq/physx_step 1 rather than jumping ahead
        # by the pre-tactile settle steps.  There was no tactile state before,
        # so this cannot regress an observed sequence.
        self._physics_steps_total = 0
        self._tactile_started = True

    def _tactile_attachment_frames(self) -> dict[str, npt.NDArray[np.float64]]:
        """World <- body transforms for each configured pad attachment."""
        bridge = self._tactile_bridge
        if bridge is None:
            return {}
        names = [pad.frame.body_name for pad in bridge.cfg.pads]
        ids = self._body_ids(names)
        poses = cast(
            npt.NDArray[np.float64],
            self._arm.robot.data.body_pose_w.torch[0, ids].cpu().numpy(),
        )
        pose_by_name = {
            name: body_pose_to_T(poses[index]) for index, name in enumerate(names)
        }
        return {pad.name: pose_by_name[pad.frame.body_name] for pad in bridge.cfg.pads}

    def _tactile_body_pad_frames(
        self, body_frames: Mapping[str, npt.NDArray[np.float64]]
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Body <- gelpad transforms, measured from the authored gelpad prims."""
        bridge = self._tactile_bridge
        if bridge is None:
            return {}
        out: dict[str, npt.NDArray[np.float64]] = {}
        for pad in bridge.cfg.pads:
            # Compare both prims in the same USD frame: the authored robot root
            # applies to both, and PhysX body_pose_w is already in world space.
            usd_body_T = _prim_world_transform(pad.frame.body_prim_path)
            gelpad_T = _prim_world_transform(pad.gelpad_prim_path)
            out[pad.name] = np.linalg.inv(usd_body_T) @ gelpad_T
        return out

    def _tactile_object_poses(self) -> dict[str, npt.NDArray[np.float64]]:
        """World transforms of the tactile objects at the reset pose."""
        bridge = self._tactile_bridge
        if bridge is None:
            return {}
        out: dict[str, npt.NDArray[np.float64]] = {}
        for obj in bridge.cfg.objects:
            try:
                pos, quat = self.object_pose(obj.prim_path)
            except KeyError:
                # Static UIPC-fixed actors have no RigidObject.  Their DR pose
                # is applied when the worker builds the fixed body.
                override = self._actor_pose_overrides.get(obj.prim_path)
                if override is None:
                    raise
                pos, quat = override.position, override.quaternion_xyzw
            out[obj.name] = body_pose_to_T(np.asarray([*pos, *quat], dtype=np.float64))
        return out

    def _bind_sensors(self) -> None:
        """Bind every attached sensor against this episode's rest state."""
        self._tactile_sensors = {}
        bridge = self._tactile_bridge
        if bridge is None:
            return
        body_frames = self._tactile_attachment_frames()
        bindings = {binding.pad_key: binding for binding in self._spec.sensors}
        for side, sensor in sorted(self._attached_sensors.items()):
            world: World = _SensorWorld(self, bindings, body_frames)
            bound = cast(BoundGsMiniSensor, sensor.bind(world))
            self._tactile_sensors[side] = bound
            self._tactile_camera_frames[side] = bound.visual.camera_from_pad

    def _run_settle(self, settle: SettleProtocol) -> None:
        # TODO(tactile-stack): early-exit on marker-reference stability once the tactile
        # marker source exists; until then settle runs the protocol's full step budget.
        for _ in range(settle.max_steps):
            self._arm.robot.write_data_to_sim()
            self._sim.step(render=False)
        self._settled = True

    def settle(
        self,
        steps: int,
        *,
        window: int | None = None,
        pos_eps: float = 1e-4,
        rot_eps: float = 1e-3,
    ) -> SettleResult:
        """Advance real physics before rollout; optionally stop when stable.

        Must run after :meth:`reset` (DR poses are already applied) and before
        :meth:`start_tactile` / any rollout step or ``actual.*`` pose capture.
        Unlike :meth:`warmup`, this calls :meth:`step`, so PhysX advances and
        contacts settle.  The tactile bridge is intentionally not started yet;
        its t0 reference must be captured from the settle-end state, so these
        steps cannot advance UIPC.  ``start_tactile`` runs afterwards.

        ``window=None`` is the legacy fixed-step mode: run exactly ``steps``
        and do not evaluate a criterion.  With ``window`` set, step up to
        ``steps`` and stop once every rigid actor's pose changed by at most
        ``pos_eps`` metres and ``rot_eps`` radians over the last ``window``
        steps.  This is a pose-delta criterion; raw velocity is deliberately
        not used because PhysX contact jitter does not decay to zero.

        ``steps=0`` is the legacy no-settle path.  The sink is suppressed for
        settle steps and the episode clock is reset afterwards, so the first
        rollout frame is timestamped 0.  The monotonic physics counter is not
        reset, so tactile finger/state sequences never regress.
        """
        if not self._reset_done:
            raise RuntimeError("Scene.settle() requires Scene.reset() first")
        if steps < 0:
            raise ValueError(f"settle steps must be non-negative, got {steps}")
        if window is not None:
            if window <= 0:
                raise ValueError(f"settle window must be positive, got {window}")
            if pos_eps <= 0 or rot_eps <= 0:
                raise ValueError(
                    "settle convergence epsilon must be positive, got "
                    f"pos_eps={pos_eps}, rot_eps={rot_eps}"
                )
        if steps == 0:
            self._settled = True
            return SettleResult(requested_steps=0, executed_steps=0)
        if self._step_limit is not None:
            raise RuntimeError("Scene.settle() must run before set_step_limit()")
        previous_sink = self._step_sink
        self._step_sink = None
        executed = 0
        converged: bool | None = None if window is None else False
        max_pos_delta: float | None = None
        max_rot_delta: float | None = None
        history: list[dict[str, npt.NDArray[np.float64]]] = []
        try:
            for _ in range(steps):
                self.step()
                executed += 1
                if window is None:
                    continue
                history.append(self._actor_pose_snapshot())
                if len(history) > window + 1:
                    history.pop(0)
                if len(history) < window + 1:
                    continue
                pos_delta, rot_delta = _max_pose_delta(history[0], history[-1])
                max_pos_delta = (
                    pos_delta
                    if max_pos_delta is None
                    else max(max_pos_delta, pos_delta)
                )
                max_rot_delta = (
                    rot_delta
                    if max_rot_delta is None
                    else max(max_rot_delta, rot_delta)
                )
                if pos_delta <= pos_eps and rot_delta <= rot_eps:
                    converged = True
                    break
        finally:
            self._step_sink = previous_sink
            self._steps_taken = 0
            self._last_step_index = 0
        self._settled = True
        return SettleResult(
            requested_steps=steps,
            executed_steps=executed,
            converged=converged,
            max_pos_delta_m=max_pos_delta,
            max_rot_delta_rad=max_rot_delta,
        )

    def _actor_pose_snapshot(self) -> dict[str, npt.NDArray[np.float64]]:
        """World poses of every PhysX rigid actor, refreshed after a step."""
        return {
            path: np.asarray(
                rigid.data.root_pose_w.torch[0].cpu().numpy(), dtype=np.float64
            )
            for path, rigid in self._rigids.items()
        }

    def warmup(self, steps: int) -> None:
        for _ in range(steps):
            self._sim.render()
            self._update_cameras()

    def set_step_sink(
        self,
        sink: Callable[[SceneObservation], None] | None,
        *,
        every: int = 1,
    ) -> None:
        """Register ``sink`` to receive ``read()`` after every ``every``-th step.

        ``None`` removes it. ``every=1`` records every physics step; ``every=2``
        records physics steps 0, 2, 4, ... (0-based indices, i.e. the first step
        is always recorded). Physics still advances every step either way. This
        is how the collector records a scripted expert frame by frame; it
        installs its own sink for the duration of a run.
        """
        if every <= 0:
            raise ValueError(f"sink cadence must be positive, got {every}")
        self._step_sink = sink
        self._step_every = every

    def set_step_limit(self, limit: int | None) -> None:
        """Cap subsequent :meth:`step` calls and reset the step counter.

        The cap counts physics steps. Attempting step ``limit + 1`` raises
        :class:`StepLimitExceeded`, so a caller can end the rollout and record
        ``outcome="timeout"``. ``None`` removes the cap.
        """
        if limit is not None and limit < 0:
            raise ValueError(f"step limit must be non-negative, got {limit}")
        self._step_limit = limit
        self._steps_taken = 0
        self._last_step_index = 0

    def step(self) -> None:
        if not self._reset_done:
            raise RuntimeError("Scene.step() requires Scene.reset() first")
        if self._step_limit is not None and self._steps_taken >= self._step_limit:
            raise StepLimitExceeded(f"scene step limit of {self._step_limit} reached")
        recording = self._step_sink is not None
        record_frame = visual_update_due(
            self._steps_taken, self._step_every, recording=recording
        )
        # Joint targets reach PhysX only through write_data_to_sim (it runs the
        # actuator compute/submit); Articulation.update(dt) is read-only. Without this
        # call every staged target is dropped and the arm just settles under gravity.
        self._arm.robot.write_data_to_sim()
        self._sim.step(render=True)
        # Refresh the robot before sampling finger poses for the UIPC exchange; the
        # remaining assets refresh after the proxy sync below.
        self._arm.robot.update(self._dt)
        self._physics_steps_total += 1
        body_names, body_rows = self._read_body_rows()
        if self._sensor_requires.objects:
            self._refresh_physx_objects()
        self._visual_changed = False
        published: TickSnapshot | None = None
        if self._tactile_bridge is not None and self._tactile_started:
            # The pre-exchange snapshot only stages finger poses; object poses
            # do not exist until this tick's exchange lands.
            sync_snapshot = self._build_tick_snapshot(
                body_names,
                body_rows,
                step_index=self._steps_taken,
                include_objects=False,
            )
            for side in sorted(self._tactile_sensors):
                self._tactile_sensors[side].sync(sync_snapshot)
            # Pads without a sim-side sensor still need a finger transform so
            # the worker sees the complete pair.
            missing = {"left", "right"} - set(self._tactile_sensors)
            if missing:
                sample = self._make_finger_sample(body_names, body_rows)
                for side, transform in (
                    ("left", sample.left_T),
                    ("right", sample.right_T),
                ):
                    if side in missing:
                        self._tactile_bridge.stage_finger(side, transform)
            self._tactile_state = self._tactile_bridge.exchange_staged()
            if self._tactile_state is not None:
                self._sync_uipc_proxies()
                self._visual_due = record_frame
                published = self._build_tick_snapshot(
                    body_names, body_rows, step_index=self._steps_taken
                )
                for side in sorted(self._tactile_sensors):
                    self._tactile_sensors[side].publish(published)
            else:
                self._visual_due = False
            if (
                self._uipc_proxies and self._tactile_state is not None
            ) or self._visual_changed:
                # sim.step rendered the previous proxy pose; render the UIPC pose
                # (and the refreshed gelpad mesh) before the cameras read.
                self._sim.render()
        # isaaclab 3.0 freezes an asset's .data at first read until update(dt);
        # refresh every stepped asset here so read() and between-step IK never see
        # a stale snapshot.
        self._update_actors()
        self._update_cameras()
        if published is not None and self._tactile_sensors:
            self._observe_tactile_sensors(published)
        # Stamp the 0-based index of the step just completed; the sink receives
        # that index (via read()) so v3 timestamps use physics time.
        self._last_step_index = self._steps_taken
        self._steps_taken += 1
        if record_frame and self._step_sink is not None:
            self._step_sink(self.read())

    def _sync_uipc_proxies(self) -> None:
        """Publish UIPC object poses to their kinematic PhysX proxies."""
        bridge = self._tactile_bridge
        if bridge is None or not self._uipc_proxies:
            return
        for name, proxy in self._uipc_proxies.items():
            handle = self._uipc_object_handles.get(name)
            if handle is None:
                raise RuntimeError(f"UIPC object {name!r} has no manager handle")
            obj = bridge.read_object(handle)
            if not obj.valid:
                raise RuntimeError(
                    f"UIPC object {name!r} has no valid pose for its proxy"
                )
            sync_kinematic_proxy(proxy, obj.pose_xyzw, device=self._sim.device)

    def _required_body_names(self) -> list[str]:
        """Gripper contact bodies plus every body a bound sensor requires."""
        names = set(self._arm.embodiment.gripper.contact_bodies)
        names.update(str(body) for body in self._sensor_requires.bodies)
        return sorted(names)

    def _read_body_rows(self) -> tuple[list[str], npt.NDArray[np.float64]]:
        """One batched PhysX pose read for every body this tick needs."""
        names = self._required_body_names()
        ids = self._body_ids(names)
        rows = cast(
            npt.NDArray[np.float64],
            self._arm.robot.data.body_pose_w.torch[0, ids].cpu().numpy(),
        )
        return names, rows

    def _make_finger_sample(
        self,
        body_names: Sequence[str] | None = None,
        body_rows: npt.NDArray[np.float64] | None = None,
    ) -> FingerSample:
        """PhysX post-step finger transforms for the tactile worker."""
        if body_names is None or body_rows is None:
            body_names, body_rows = self._read_body_rows()
        contact = list(self._arm.embodiment.gripper.contact_bodies)
        if len(contact) != 2:
            raise RuntimeError(f"expected 2 tactile attachment bodies, got {contact}")
        rows_by_name = {
            name: body_rows[index] for index, name in enumerate(body_names)
        }
        step = self._physics_steps_total
        return finger_sample_from_body_poses(
            rows_by_name[contact[0]],
            rows_by_name[contact[1]],
            seq=step,
            physx_step=step,
        )

    def read(self) -> SceneObservation:
        rgb: dict[str, npt.NDArray[np.uint8]] = {}
        for path, camera in self._cameras.items():
            output = camera.data.output
            assert output is not None  # exists once the camera has rendered
            rgb[path] = cast(
                npt.NDArray[np.uint8], output["rgb"].torch[0].cpu().numpy()
            )
        joint_pos = cast(
            npt.NDArray[np.floating[Any]],
            self._arm.robot.data.joint_pos.torch[0].cpu().numpy(),
        )
        joint_vel = cast(
            npt.NDArray[np.floating[Any]],
            self._arm.robot.data.joint_vel.torch[0].cpu().numpy(),
        )
        action = cast(
            npt.NDArray[np.floating[Any]],
            self._arm.last_target[0].cpu().numpy(),
        )
        return SceneObservation(
            rgb=rgb,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            action=action,
            tactile=self._tactile_state,
            tactile_cameras=self._tactile_frame_cameras,
            tactile_objects=self._tactile_frame_objects,
            physics_step=self._last_step_index,
        )

    def collect(
        self,
        sink: Callable[[SceneObservation], None],
        *,
        steps: int,
        every: int = 1,
    ) -> int:
        """Step ``steps`` times, handing every ``every``-th observation to ``sink``.

        Physics still advances every step; ``every`` only throttles recording
        (default 1 = record every step). This is a thin driver over the same
        per-step sink used by scripted experts; an already-registered sink (and
        its cadence) is restored when the run ends.
        """
        previous_sink = self._step_sink
        previous_every = self._step_every
        self.set_step_sink(sink, every=every)
        try:
            for _ in range(steps):
                self.step()
        finally:
            self._step_sink = previous_sink
            self._step_every = previous_every
        return steps

    def close(self) -> None:
        """Stop the tactile worker (if any); idempotent and optional."""
        if self._tactile_bridge is not None:
            self._tactile_bridge.close()
            self._tactile_bridge = None
            self._tactile_started = False

    def set_logger(self, logger: logging.LoggerAdapter[logging.Logger]) -> None:
        """Bind the run-scoped logger exposed to task experts/success checks."""
        self._logger = logger

    # --- ExpertWorld (structural; the cli holds the conformance check) ---

    @property
    def logger(self) -> logging.LoggerAdapter[logging.Logger]:
        """Run-scoped logger; fail loud if the caller forgot ``set_logger``."""
        if self._logger is None:
            raise RuntimeError(
                "Scene logger is not bound; call set_logger() before expert/success"
            )
        return self._logger

    @property
    def embodiment(self) -> EmbodimentSpec:
        """Public read-only view for the gripper adapters (spec is pure data)."""
        return self._spec.embodiment

    @property
    def arm(self) -> SceneArm:
        """The single arm handle scripted experts drive.

        Structural conformance to the tasks-side ``ArmHandle`` is asserted by the
        cli, so sim still does not import tasks.
        """
        return self._arm_handle

    def set_expert_params(self, params: Mapping[str, str | float]) -> None:
        """Attach episode setup metadata consumed by scripted experts."""
        self._expert_params = dict(params)

    def expert_param(self, name: str) -> str | float:
        """Task setup metadata, e.g. the sampled RC classification label."""
        try:
            return self._expert_params[name]
        except KeyError:
            raise KeyError(f"expert parameter {name!r} was not set") from None

    def object_pose(self, prim_name: str) -> tuple[Vec3, Quat]:
        """Ground-truth world pose ``(pos, quat)`` of a task actor.

        Static fixture actors have no RigidObject; they return their nominal or
        DR override pose from the spec.  Dynamic actors read the buffers
        refreshed by the latest ``step()``/``reset()``.
        """
        matches = [
            path
            for path in self._rigids
            if path == prim_name or path.rsplit("/", 1)[-1] == prim_name
        ]
        if not matches:
            static = [
                actor
                for actor in self._spec.actors
                if not actor.dynamic
                and (
                    actor.prim_path == prim_name
                    or actor.prim_path.rsplit("/", 1)[-1] == prim_name
                )
            ]
            if len(static) == 1:
                actor = static[0]
                override = self._actor_pose_overrides.get(actor.prim_path)
                if override is not None:
                    return override.position, override.quaternion_xyzw
                return actor.pos, actor.rot
        if len(matches) != 1:
            raise KeyError(
                f"object_pose({prim_name!r}) matched {len(matches)} dynamic actors"
                f" (expected 1): {matches}"
            )
        pose = self._rigids[matches[0]].data.root_pose_w.torch[0].cpu()
        values = cast(list[float], pose.tolist())
        return (values[0], values[1], values[2]), (
            values[3],
            values[4],
            values[5],
            values[6],
        )

    def _update_actors(self) -> None:
        self._arm.robot.update(self._dt)
        for actor in self._rigids.values():
            actor.update(self._dt)
        self._contact.update(dt=self._dt)

    def _update_cameras(self) -> None:
        for camera in self._cameras.values():
            camera.update(dt=self._dt)

    def _build_tick_snapshot(
        self,
        body_names: Sequence[str],
        body_rows: npt.NDArray[np.float64],
        *,
        step_index: int,
        include_objects: bool = True,
    ) -> TickSnapshot:
        """Memoise every required handle once for this tick."""
        pose_by_name = {
            name: body_pose_to_T(body_rows[index])
            for index, name in enumerate(body_names)
        }
        bodies = {
            body: RigidState(Pose(pose_by_name[str(body)]))
            for body in self._sensor_requires.bodies
        }
        if include_objects:
            for body in self._sensor_requires.objects:
                bodies[body] = RigidState(self._object_pose(body))
        deformables: dict[BodyId, DeformableState] = {}
        bridge = self._tactile_bridge
        state = self._tactile_state
        if bridge is not None and state is not None:
            for body in self._sensor_requires.deformables:
                pad = state.pads.get(str(body))
                if pad is None or pad.vertices is None:
                    raise RuntimeError(
                        f"deformable {body!r} has no live vertices this tick"
                    )
                surface = bridge.pad_surfaces_by_name[str(body)]
                frame = bridge.pad_world_frames.get(str(body))
                deformables[body] = DeformableState(
                    frame=Pose(frame) if frame is not None else Pose.identity(),
                    vertices=pad.vertices,
                    triangles=surface.triangles,
                )
        cameras: dict[CameraId, Pose] = {}
        for camera in self._sensor_requires.cameras:
            pad = self._aperture_pads[str(camera)]
            local = self._aperture_camera_local[str(camera)]
            cameras[camera] = Pose(pose_by_name[pad.frame.body_name]) @ Pose(local)
        return TickSnapshot(
            step_index=step_index,
            timestamp_us=round(self._physics_steps_total * self._dt * 1e6),
            bodies=bodies,
            deformables=deformables,
            cameras=cameras,
        )

    def _refresh_physx_objects(self) -> None:
        """Refresh PhysX-owned contact objects so a snapshot sees this tick."""
        for body in self._sensor_requires.objects:
            name = str(body)
            if name in self._uipc_proxies:
                continue
            self._rigids[self._contact_object_paths[name]].update(self._dt)

    def _object_pose(self, body: BodyId) -> Pose:
        """World pose of one declared contact object for this tick."""
        name = str(body)
        prim_path = self._contact_object_paths.get(name)
        if prim_path is None:
            raise RuntimeError(
                f"contact object {name!r} was not resolved from the scene actors"
            )
        if name in self._uipc_proxies:
            bridge = self._tactile_bridge
            state = self._tactile_state
            if bridge is None or state is None:
                raise RuntimeError(
                    f"contact object {name!r} has no UIPC exchange state this tick"
                )
            handle = self._uipc_object_handles.get(name)
            if handle is None:
                raise RuntimeError(f"UIPC object {name!r} has no manager handle")
            obj = bridge.read_object(handle)
            if not obj.valid:
                raise RuntimeError(f"UIPC object {name!r} has no valid pose")
            return Pose(obj.T)
        if prim_path not in self._rigids:
            raise RuntimeError(
                f"contact object {name!r} has no spawnable PhysX body"
            )
        row = cast(
            npt.NDArray[np.float64],
            self._rigids[prim_path].data.root_pose_w.torch[0].cpu().numpy(),
        )
        return Pose(body_pose_to_T(row))

    def _observe_tactile_sensors(self, snap: TickSnapshot) -> None:
        """Drive every bound sensor for one frame and merge its streams."""
        state = self._tactile_state
        if state is None:
            return
        depth: dict[str, npt.NDArray[np.float32]] = {}
        rgb: dict[str, npt.NDArray[np.uint8]] = {}
        marker: dict[str, npt.NDArray[np.uint8]] = {}
        heights: dict[str, npt.NDArray[np.float64]] = {}
        for side, sensor in sorted(self._tactile_sensors.items()):
            observation = sensor.observe(snap)
            streams = observation.streams()
            depth_image = streams.get("depth")
            if depth_image is not None:
                depth[side] = cast(npt.NDArray[np.float32], depth_image)
            rgb_image = streams.get("rgb")
            if rgb_image is not None:
                rgb[side] = cast(npt.NDArray[np.uint8], rgb_image)
            marker_image = streams.get("marker_rgb")
            if marker_image is not None:
                marker[side] = cast(npt.NDArray[np.uint8], marker_image)
            height_image = observation.extras().get("height_mm")
            if height_image is not None:
                heights[side] = cast(npt.NDArray[np.float64], height_image)
        updates: dict[str, Any] = {}
        if self._tactile_depth_record_sides:
            updates["tactile_depth"] = {
                side: depth[side]
                for side in self._tactile_depth_record_sides
                if side in depth
            }
        if self._tactile_sensors:
            updates["tactile_rgb"] = rgb
            updates["tactile_marker_rgb"] = marker
        self._tactile_height_maps = heights
        self._tactile_frame_cameras = {
            str(camera): pose.matrix for camera, pose in snap.cameras.items()
        }
        self._tactile_frame_objects = {
            str(body): snap.pose(body).matrix
            for body in self._sensor_requires.objects
        }
        self._tactile_state = replace(state, **updates)


class SceneArm:
    """Single-arm handle: the tasks-side ``ArmHandle`` surface over ``Scene``.

    The sim never imports tasks, so this mirrors the protocol structurally.  All
    poses address the embodiment's ``arm.ee_link`` body (the wrist/flange link the
    IK solver drives), not the tool center point.  ``joint_names`` is sim-only
    dataset metadata; it is not part of ``ArmHandle``.
    """

    def __init__(self, scene: Scene) -> None:
        self._scene = scene

    @property
    def gripper(self) -> _GsMiniGripper | _XenseWSGripper:
        """The capability-annotated gripper handle for this embodiment.

        A Robotiq handle exists because the preset declares adaptive/percent
        capabilities, but its binary/adaptive methods still fail loud until the
        tactile grasp stack lands.
        """
        if self._scene._spec.embodiment.gripper.kind == "robotiq_85":
            return _XenseWSGripper(self._scene)
        return _GsMiniGripper(self._scene)

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Articulation joint order, matching :meth:`joint_pos` column order."""
        return tuple(self._scene._arm.robot.joint_names)

    def ee_pose(self) -> tuple[Vec3, Quat]:
        """World pose ``(pos, quat_xyzw)`` of the ``arm.ee_link`` body.

        The robot base sits at the world origin, so the IK solver's base frame is
        the world frame.
        """
        pos, quat = self._scene._arm.ee_pose_base()
        values = cast(list[float], pos[0].cpu().tolist())
        quat_values = cast(list[float], quat[0].cpu().tolist())
        return (values[0], values[1], values[2]), (
            quat_values[0],
            quat_values[1],
            quat_values[2],
            quat_values[3],
        )

    def set_ee_target(self, pos: Vec3, quat: Quat) -> None:
        """Set the ``arm.ee_link`` target for the next control tick (no step).

        One differential-IK solve is staged; the collector advances physics.
        """
        arm = self._scene._arm
        device = arm.robot.device
        pos_t = torch.tensor([pos], device=device)
        quat_t = torch.tensor([quat], device=device)
        quat_t = cast(
            torch.Tensor, quat_t / quat_t.norm(dim=-1, keepdim=True)
        )
        arm.set_arm_target(arm.ik_to(pos_t, quat_t))

    def joint_pos(self) -> npt.NDArray[np.floating[Any]]:
        """Current joint positions in the articulation's joint order."""
        return cast(
            npt.NDArray[np.floating[Any]],
            self._scene._arm.robot.data.joint_pos.torch[0].cpu().numpy(),
        )

    def gripper_contact(self, *, force_threshold: float = 1.0) -> tuple[bool, bool]:
        """``(left, right)`` pad contact flags: PhysX net contact force on each
        of the embodiment's finger pad bodies above ``force_threshold`` newtons."""
        scene = self._scene
        net = scene._contact.data.net_forces_w
        if net is None:
            raise RuntimeError("contact sensor has produced no data yet")
        magnitudes = cast(
            list[float], cast(torch.Tensor, net.torch[0].norm(dim=-1)).cpu().tolist()
        )
        names = scene._contact.body_names or []
        by_name = {name: magnitudes[i] for i, name in enumerate(names)}
        bodies = scene._arm.embodiment.gripper.contact_bodies
        if len(bodies) != 2:
            raise ValueError(f"expected 2 contact pad bodies, got {bodies}")
        pads = [by_name[name] > force_threshold for name in bodies]
        return (pads[0], pads[1])


class _GsMiniGripper:
    """Binary position-controlled open/close for the Panda parallel gripper."""

    def __init__(self, scene: Scene) -> None:
        self._scene = scene

    @property
    def capabilities(self) -> frozenset[GripperCapability]:
        return self._scene.embodiment.gripper.capabilities

    @property
    def open_qpos(self) -> float:
        return self._scene.embodiment.gripper.open_value

    @property
    def close_qpos(self) -> float:
        return self._scene.embodiment.gripper.close_value

    def set_target(self, qpos: float) -> None:
        """Stage the finger target for the next control tick (no step)."""
        self._scene._arm.set_gripper_target(qpos)

    def joint_pos(self) -> float:
        return self._scene._arm.gripper_joint_pos()


class _XenseWSGripper:
    """Robotiq 2F-85 handle: binary position target; adaptive surface pending.

    The mechanism is one driven ``finger_joint`` with a PhysX mimic on the outer
    knuckles, so a binary target is exactly what the articulation supports. The
    adaptive surface (``close_percent`` / ``depth_threshold`` /
    ``require_both_contacts``) needs pad-contact reasoning that only the Xense
    tactile stack can supply, so it stays fail-loud.
    """

    def __init__(self, scene: Scene) -> None:
        self._scene = scene

    @property
    def capabilities(self) -> frozenset[GripperCapability]:
        return self._scene.embodiment.gripper.capabilities

    @property
    def open_qpos(self) -> float:
        return self._scene.embodiment.gripper.open_value

    @property
    def close_qpos(self) -> float:
        return self._scene.embodiment.gripper.close_value

    def set_target(self, qpos: float) -> None:
        """Stage the finger target for the next control tick (no step)."""
        self._scene._arm.set_gripper_target(qpos)

    def joint_pos(self) -> float:
        return self._scene._arm.gripper_joint_pos()
