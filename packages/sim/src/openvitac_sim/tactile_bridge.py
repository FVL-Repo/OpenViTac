"""Sim-side adapter between sensors/scene and the isolated tactile runtime.

This module is import-safe before the Isaac app launches: it imports only pure
``openvitac_tactile`` data/client modules and the pure ``sensors``/``spec``
dataclasses.  The worker is the only process that may import pyuipc.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from openvitac_tactile.client import TactileRuntime
from openvitac_tactile.config import (
    ObjectCfg,
    PadCfg,
    PadFrameCfg,
    PadGeometryCfg,
    PadMaterialCfg,
    ResetCfg,
    Side,
    TactileCfg,
)
from openvitac_tactile.observation import (
    ObjectObservation,
    PadObservation,
    TactileObservation,
)
from openvitac_tactile.protocol import (
    FingerSample,
    TactileState,
    force_kind_name,
)
from openvitac_tactile.softworld import (
    ObjectHandle,
    ObjectState,
    PadHandle,
    PadState,
)

from .assets import asset
from .sensors import GsMiniCfg, SensorVariant, XenseWSCfg
from .spec import Actor


def _actor_name(prim_path: str) -> str:
    return prim_path.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class PadSurface:
    """Rest surface returned by the worker's ``extract_surface``.

    ``rest_vertices`` are world-space points in the same order as the state's
    ``pad_vertices``; ``triangles`` index that array.  The visual mesh uses this
    topology instead of guessing from the parent-side ``tetmesh_surface``.
    """

    name: str
    side: str
    rest_vertices: npt.NDArray[np.float64]
    triangles: npt.NDArray[np.int32]

    def __post_init__(self) -> None:
        vertices = np.asarray(self.rest_vertices, dtype=np.float64)
        triangles = np.asarray(self.triangles, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"pad surface {self.name!r} vertices must be (V,3)")
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError(f"pad surface {self.name!r} triangles must be (F,3)")
        if vertices.shape[0] == 0 or triangles.shape[0] == 0:
            raise ValueError(f"pad surface {self.name!r} is empty")
        if not np.isfinite(vertices).all():
            raise ValueError(f"pad surface {self.name!r} vertices are not finite")
        if int(triangles.min()) < 0 or int(triangles.max()) >= vertices.shape[0]:
            raise ValueError(f"pad surface {self.name!r} indices are out of range")
        object.__setattr__(self, "rest_vertices", vertices)
        object.__setattr__(self, "triangles", triangles)


def objects_from_actors(actors: Iterable[Actor]) -> tuple[ObjectCfg, ...]:
    """Map dynamic scene actors to tactile object declarations.

    The actor's authority is preserved. UIPC objects need a tetrahedral mesh:
    a missing ``tactile_mesh`` is a configuration error, never a silent
    default-box fallback.  An actor whose path is already ``.npz`` is used
    as-is.  PhysX-owned objects are not built by the worker, so their
    ``mesh_path`` stays ``None`` rather than claiming the actor USD is a
    tetrahedral mesh.  ``usd_path`` always carries the source actor USD for
    provenance (``None`` only when the spec declares no USD).  ``constitution``
    and the FEM-only ``youngs_mpa``/``poisson`` are forwarded unchanged; the spec
    has already restricted ``fem`` to dynamic UIPC actors.  ``density`` becomes
    the UIPC body's mass density; ``None`` keeps the 1e3 default, which equals
    PhysX's default density.
    """
    out: list[ObjectCfg] = []
    for actor in actors:
        name = _actor_name(actor.prim_path)
        if actor.uipc_fixed_boxes or actor.uipc_fixed_mesh is not None:
            out.append(
                ObjectCfg(
                    name=f"{name}_static",
                    prim_path=actor.prim_path,
                    authority="uipc",
                    constitution="abd",
                    fixed=True,
                    boxes=actor.uipc_fixed_boxes,
                    mesh_path=(
                        asset(actor.uipc_fixed_mesh)
                        if actor.uipc_fixed_mesh is not None
                        else None
                    ),
                    usd_path=actor.usd or None,
                )
            )
        if not actor.dynamic:
            continue
        if actor.authority == "physx":
            # PhysX owns the body; the worker only mirrors its pose, so this
            # field must not pretend the actor USD is a tetrahedral mesh.
            mesh_path = None
        elif actor.tactile_mesh is not None:
            mesh_path = asset(actor.tactile_mesh)
        elif actor.usd.endswith(".npz"):
            mesh_path = asset(actor.usd)
        else:
            raise ValueError(
                f"UIPC actor {name!r} has no tactile_mesh; refusing to let the "
                "worker fall back to its default box"
            )
        out.append(
            ObjectCfg(
                name=name,
                prim_path=actor.prim_path,
                authority=actor.authority,
                constitution=actor.constitution,
                youngs_mpa=actor.youngs_mpa,
                poisson=actor.poisson,
                density_kg_m3=(
                    actor.density if actor.density is not None else 1.0e3
                ),
                mesh_path=mesh_path,
                usd_path=actor.usd or None,
                contact_object=actor.contact_object,
            )
        )
    return tuple(out)


def _other_side(value: str, src: str, dst: str) -> str:
    if src not in value:
        raise ValueError(f"cannot derive {dst} from {value!r}")
    return value.replace(src, dst)


def _pad_for_side(
    sensor: GsMiniCfg | XenseWSCfg,
    *,
    side: Side,
    gelpad_prim_path: str,
    body_name: str,
    body_prim_path: str,
) -> PadCfg:
    base_name = sensor.name
    for suffix in ("_left", "_right"):
        if base_name.endswith(suffix):
            base_name = base_name[: -len(suffix)]
            break
    gsmini = isinstance(sensor, GsMiniCfg)
    return PadCfg(
        side=side,
        name=f"{base_name}_{side}",
        gelpad_prim_path=gelpad_prim_path,
        frame=PadFrameCfg(
            body_name=body_name,
            body_prim_path=body_prim_path,
            pos=(0.0, 0.0, 0.0),
            rot_xyzw=(0.0, 0.0, 0.0, 1.0),
            back_axis="z" if gsmini else "y",
            back_is_min=True if gsmini else None,
        ),
        geometry=(
            PadGeometryCfg(
                kind="tet_mesh",
                mesh_path=asset("embodiment/gsmini_gelpad_tet.npz"),
            )
            if gsmini
            else PadGeometryCfg()
        ),
        material=PadMaterialCfg(
            youngs_mpa=sensor.youngs,
            constraint_strength=sensor.strength,
        ),
        image_width=sensor.width,
        image_height=sensor.height,
        marker_rows=sensor.marker_rows,
        marker_cols=sensor.marker_cols,
        render_kinds=tuple(
            kind for kind in sensor.data_types if kind in ("rgb", "marker_rgb", "depth")
        ),
    )


def tactile_cfg_from_sensor(
    sensor: SensorVariant,
    *,
    objects: Iterable[ObjectCfg] = (),
    include_uipc_fingers: bool = False,
) -> TactileCfg:
    """Map a ``GsMiniCfg``/``XenseWSCfg``-style sensor into two pads.

    Task specs declare the left sensor only (``..._left``).  The right pad is
    derived by replacing the ``left`` path/body tokens with ``right``; both
    pads share E/strength/dimensions.  If a future spec already declares a
    right-side token, the missing side is derived instead of duplicating the
    existing right pad.  A declaration containing both left and right tokens
    is ambiguous and rejected.

    The pad rest frame position/rotation is deliberately identity: it is a
    stage-2 placeholder, real calibration from the robot body/USD is pending.
    """
    parts = (
        sensor.prim_path,
        sensor.gelpad_prim_path,
        sensor.attachment_body.name,
        sensor.attachment_body.prim_path,
    )
    has_left = any("left" in part for part in parts)
    has_right = any("right" in part for part in parts)
    if has_left and has_right:
        raise ValueError(
            "sensor declaration contains both left and right tokens; "
            "provide an explicit pads pair in a future extension"
        )
    if not has_left and not has_right:
        raise ValueError(
            "sensor declaration has no left/right token to derive the other pad"
        )
    src, dst = ("left", "right") if has_left else ("right", "left")
    declared_gelpad = sensor.gelpad_prim_path
    declared_body = sensor.attachment_body.name
    declared_body_path = sensor.attachment_body.prim_path
    derived_gelpad = _other_side(declared_gelpad, src, dst)
    derived_body = _other_side(declared_body, src, dst)
    derived_body_path = _other_side(declared_body_path, src, dst)
    left_gelpad, right_gelpad = (
        (declared_gelpad, derived_gelpad)
        if src == "left"
        else (derived_gelpad, declared_gelpad)
    )
    left_body, right_body = (
        (declared_body, derived_body)
        if src == "left"
        else (derived_body, declared_body)
    )
    left_body_path, right_body_path = (
        (declared_body_path, derived_body_path)
        if src == "left"
        else (derived_body_path, declared_body_path)
    )
    pads = (
        _pad_for_side(
            sensor,
            side="left",
            gelpad_prim_path=left_gelpad,
            body_name=left_body,
            body_prim_path=left_body_path,
        ),
        _pad_for_side(
            sensor,
            side="right",
            gelpad_prim_path=right_gelpad,
            body_name=right_body,
            body_prim_path=right_body_path,
        ),
    )
    settle = sensor.settle
    return TactileCfg(
        enabled=True,
        pads=pads,
        objects=tuple(objects),
        include_uipc_fingers=include_uipc_fingers,
        reset=ResetCfg(
            settle_max_steps=(settle.max_steps if settle is not None else 0),
        ),
    )


def _T_from_pose(
    pos: npt.NDArray[np.floating[Any]], quat_xyzw: npt.NDArray[np.floating[Any]]
) -> npt.NDArray[np.float64]:
    x, y, z, qx, qy, qz, qw = (*pos, *quat_xyzw)
    n = float(np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw))
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    R = np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = (x, y, z)
    return T


def body_pose_to_T(
    pose_xyzw: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.float64]:
    """Convert one isaaclab ``body_pose_w`` row (XYZW) to a 4x4 transform."""
    pose = np.asarray(pose_xyzw, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"body pose must be (7,), got {pose.shape}")
    return _T_from_pose(pose[:3], pose[3:7])


def _checked_transform(
    value: npt.NDArray[np.floating[Any]], name: str
) -> npt.NDArray[np.float64]:
    T = np.asarray(value, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{name} must be (4,4), got {T.shape}")
    if not np.all(np.isfinite(T)):
        raise ValueError(f"{name} contains non-finite values")
    return T


def finger_sample_from_body_poses(
    left_pose: npt.NDArray[np.floating[Any]],
    right_pose: npt.NDArray[np.floating[Any]],
    *,
    seq: int,
    physx_step: int,
    timestamp_ns: int = 0,
) -> FingerSample:
    """Build a finger sample from two isaaclab ``body_pose_w`` rows (XYZW)."""
    lp = np.asarray(left_pose, dtype=np.float64)
    rp = np.asarray(right_pose, dtype=np.float64)
    if lp.shape != (7,) or rp.shape != (7,):
        raise ValueError("body poses must be (7,) x,y,z,qx,qy,qz,qw")
    return FingerSample(
        seq=seq,
        physx_step=physx_step,
        left_T=_T_from_pose(lp[:3], lp[3:7]),
        right_T=_T_from_pose(rp[:3], rp[3:7]),
        timestamp_ns=timestamp_ns,
    )


class TactileBridge:
    """Own a ``TactileRuntime`` and translate sim values to observations."""

    def __init__(
        self,
        cfg: TactileCfg,
        *,
        run_id: str,
        group_id: str = "0",
        check_uipc: bool = True,
        runtime: TactileRuntime | None = None,
    ) -> None:
        self._cfg = cfg
        self._episode_cfg = cfg
        self._run_id = run_id
        self._group_id = group_id
        self._check_uipc = check_uipc
        self._runtime = runtime
        self._closed = False
        self._pad_world_frames: dict[str, npt.NDArray[np.float64]] = {}
        self._pad_surfaces: dict[str, PadSurface] = {}
        self._pad_surfaces_by_handle: dict[PadHandle, PadSurface] = {}
        self._pad_handles: dict[str, PadHandle] = {}
        self._object_handles: dict[str, ObjectHandle] = {}
        self._last_state: TactileState | None = None
        self._staged_fingers: dict[str, npt.NDArray[np.float64]] = {}
        self._staged_step = 0
        self._rebuild_handles()
        self._last_observation: TactileObservation | None = None

    def _rebuild_handles(self) -> None:
        self._pad_handles = {
            pad.name: PadHandle(uid=i, name=pad.name, side=pad.side, index=i)
            for i, pad in enumerate(self._cfg.pads)
        }
        self._object_handles = {
            obj.name: ObjectHandle(uid=i, name=obj.name, index=i)
            for i, obj in enumerate(self._cfg.objects)
        }

    @classmethod
    def from_sensor(
        cls,
        sensor: SensorVariant,
        *,
        objects: Iterable[ObjectCfg] = (),
        run_id: str,
        group_id: str = "0",
        check_uipc: bool = True,
        include_uipc_fingers: bool = False,
    ) -> TactileBridge:
        return cls(
            tactile_cfg_from_sensor(
                sensor,
                objects=objects,
                include_uipc_fingers=include_uipc_fingers,
            ),
            run_id=run_id,
            group_id=group_id,
            check_uipc=check_uipc,
        )

    @property
    def cfg(self) -> TactileCfg:
        return self._cfg

    @property
    def episode_cfg(self) -> TactileCfg:
        """The config this episode's reset used (``contact_object`` applied)."""
        return self._episode_cfg

    @property
    def runtime(self) -> TactileRuntime | None:
        return self._runtime

    @property
    def pad_world_frames(self) -> Mapping[str, npt.NDArray[np.float64]]:
        """World <- pad calibration frames computed by the latest reset."""
        return MappingProxyType(self._pad_world_frames)

    @property
    def pad_surfaces(self) -> Mapping[PadHandle, PadSurface]:
        """Worker rest surfaces from the latest reset, keyed by pad handle."""
        return MappingProxyType(self._pad_surfaces_by_handle)

    @property
    def pad_surfaces_by_name(self) -> Mapping[str, PadSurface]:
        """Compatibility view of rest surfaces keyed by pad name."""
        return MappingProxyType(self._pad_surfaces)

    @property
    def pad_handles(self) -> Mapping[str, PadHandle]:
        return MappingProxyType(self._pad_handles)

    @property
    def object_handles(self) -> Mapping[str, ObjectHandle]:
        return MappingProxyType(self._object_handles)

    def pad_handle(self, name: str) -> PadHandle:
        handle = self._pad_handles.get(name)
        if handle is None:
            raise ValueError(f"unknown pad name {name!r}")
        return handle

    def object_handle(self, name: str) -> ObjectHandle:
        handle = self._object_handles.get(name)
        if handle is None:
            raise ValueError(f"unknown object name {name!r}")
        return handle

    def pad_config(self, handle: PadHandle) -> PadCfg:
        if self._pad_handles.get(handle.name) != handle:
            raise ValueError(f"unknown pad handle {handle!r}")
        return self._cfg.pads[handle.index]

    def pad_surface(self, handle: PadHandle) -> PadSurface:
        surface = self._pad_surfaces_by_handle.get(handle)
        if surface is None:
            raise ValueError(f"unknown pad handle {handle!r}")
        return surface

    def _require_state(self) -> TactileState:
        if self._last_state is None:
            raise RuntimeError("TactileBridge has no state; call exchange() first")
        return self._last_state

    def read_pad(self, handle: PadHandle) -> PadState:
        state = self._require_state()
        if self._pad_handles.get(handle.name) != handle:
            raise ValueError(f"unknown pad handle {handle!r}")
        index = handle.index
        n = int(state.pad_n_vertices[index])
        return PadState(
            handle=handle,
            seq=state.seq,
            finger_seq=state.finger_seq,
            physx_step=state.physx_step,
            timestamp_us=state.timestamp_us,
            n_vertices=n,
            vertices=np.array(
                state.pad_vertices[index, :n], dtype=np.float32, copy=True
            ),
            contact_gradient=np.array(
                state.pad_contact_gradient[index], dtype=np.float64, copy=True
            ),
            force=np.array(state.pad_force[index], dtype=np.float64, copy=True),
            force_kind=int(state.pad_force_kind[index]),
            tracking_error_mm=float(state.pad_tracking_error_mm[index]),
            stale=bool(state.pad_stale[index]),
            valid=bool(state.pad_valid[index]),
            usb_min_sd_mm=float(state.pad_usb_min_sd_mm[index]),
            usb_contact_vertices=int(state.pad_usb_contact_vertices[index]),
        )

    def read_object(self, handle: ObjectHandle) -> ObjectState:
        state = self._require_state()
        if self._object_handles.get(handle.name) != handle:
            raise ValueError(f"unknown object handle {handle!r}")
        index = handle.index
        return ObjectState(
            handle=handle,
            seq=state.seq,
            finger_seq=state.finger_seq,
            physx_step=state.physx_step,
            timestamp_us=state.timestamp_us,
            T=np.array(state.object_T[index], dtype=np.float64, copy=True),
            pose_xyzw=np.array(
                state.object_pose[index], dtype=np.float64, copy=True
            ),
            valid=bool(state.object_valid[index]),
        )

    def start(self, cfg: TactileCfg | None = None) -> None:
        if cfg is not None:
            self._cfg = cfg
            self._episode_cfg = cfg
            self._rebuild_handles()
        if self._closed:
            raise RuntimeError("TactileBridge is closed")
        if self._runtime is not None:
            return
        if not self._cfg.enabled:
            raise RuntimeError("TactileBridge.start() with disabled TactileCfg")
        self._cfg.validate()
        runtime = TactileRuntime(
            self._cfg,
            run_id=self._run_id,
            group_id=self._group_id,
            check_uipc=self._check_uipc,
        )
        runtime.start()
        self._runtime = runtime

    def reset(
        self,
        *,
        pad_frames: Mapping[str, npt.NDArray[np.floating[Any]]] | None = None,
        pad_body_frames: Mapping[str, npt.NDArray[np.floating[Any]]] | None = None,
        object_poses: Mapping[str, npt.NDArray[np.floating[Any]]] | None = None,
        seed: int = 0,
        contact_object: str | None = None,
    ) -> dict[str, Any]:
        """Send the reset handshake with the scene's actual initial frames.

        ``pad_frames`` maps pad name (or side) to world <- case body.
        ``pad_body_frames`` maps pad name to body <- gelpad (runtime calibration);
        when omitted it falls back to the identity ``PadFrameCfg``.  The worker
        builds the pad as ``T_world_body0 @ T_body_pad0``.  ``object_poses`` maps
        object name to the actor's initial world transform.  Every configured
        pad/object must be present; there is no silent identity fallback.

        ``seed`` is the episode seed, forwarded to the worker reset request so
        stochastic worker setup replays from the same episode.  ``contact_object``
        is the actor prim path of the one object whose pad contact metrics this
        episode reports; it replaces the ``Actor.contact_object`` flags in this
        reset's worker config only (``None`` keeps the spec's flags).
        """
        if not self._cfg.enabled:
            raise RuntimeError("TactileBridge.reset() with disabled TactileCfg")
        runtime = self._require_runtime()
        frames = pad_frames or {}
        body_pad = pad_body_frames or {}
        poses = object_poses or {}
        if seed < 0:
            raise ValueError(f"reset seed must be non-negative, got {seed}")
        pads: list[dict[str, Any]] = []
        world_frames: dict[str, npt.NDArray[np.float64]] = {}
        for pad in self._cfg.pads:
            raw = frames.get(pad.name, frames.get(pad.side))
            if raw is None:
                raise ValueError(f"missing reset frame for pad {pad.name!r}")
            T = _checked_transform(raw, f"pad frame {pad.name!r}")
            raw_local = body_pad.get(pad.name)
            local = (
                _checked_transform(raw_local, f"body-to-pad frame {pad.name!r}")
                if raw_local is not None
                else body_pose_to_T(
                    np.asarray([*pad.frame.pos, *pad.frame.rot_xyzw], dtype=np.float64)
                )
            )
            world_frames[pad.name] = T @ local
            pads.append(
                {
                    "side": pad.side,
                    "name": pad.name,
                    "gelpad_prim_path": pad.gelpad_prim_path,
                    "T_world_body0": T.reshape(-1).tolist(),
                    "T_body_pad0": local.reshape(-1).tolist(),
                }
            )
        objects: list[dict[str, Any]] = []
        for obj in self._cfg.objects:
            raw = poses.get(obj.name)
            if raw is None:
                raise ValueError(f"missing reset pose for object {obj.name!r}")
            T_obj = _checked_transform(raw, f"object pose {obj.name!r}")
            objects.append(
                {
                    "name": obj.name,
                    "prim_path": obj.prim_path,
                    "T_world0": T_obj.reshape(-1).tolist(),
                }
            )
        cfg = self._cfg
        if contact_object is not None:
            matched = [
                obj
                for obj in cfg.objects
                if obj.prim_path == contact_object and not obj.fixed
            ]
            if len(matched) != 1:
                raise ValueError(
                    f"contact_object {contact_object!r} is not exactly one "
                    "non-fixed configured tactile object"
                )
            cfg = replace(
                cfg,
                objects=tuple(
                    replace(obj, contact_object=(obj is matched[0]))
                    for obj in cfg.objects
                ),
            )
        self._episode_cfg = cfg
        ack = runtime.reset(
            {
                "episode_id": self._run_id,
                "dt_s": cfg.dt_s,
                "seed": int(seed),
                "pads": pads,
                "objects": objects,
                "config": asdict(cfg),
            }
        )
        self._pad_world_frames = world_frames
        raw_surfaces: Any = ack.get("pads")
        surfaces: dict[str, PadSurface] = {}
        surfaces_by_handle: dict[PadHandle, PadSurface] = {}
        if raw_surfaces is not None:
            if not isinstance(raw_surfaces, list):
                raise ValueError("worker reset pads payload must be a list")
            for entry_raw in cast(list[Any], raw_surfaces):
                if not isinstance(entry_raw, dict):
                    raise ValueError("worker reset pad entry must be an object")
                entry = cast(dict[str, Any], entry_raw)
                surface = PadSurface(
                    name=str(entry["name"]),
                    side=str(entry["side"]),
                    rest_vertices=np.asarray(
                        entry["rest_vertices"], dtype=np.float64
                    ),
                    triangles=np.asarray(entry["triangles"], dtype=np.int32),
                )
                handle = self._pad_handles.get(surface.name)
                if handle is None:
                    raise ValueError(
                        f"worker returned unregistered pad surface {surface.name!r}"
                    )
                surfaces[surface.name] = surface
                surfaces_by_handle[handle] = surface
        self._pad_surfaces = surfaces
        self._pad_surfaces_by_handle = surfaces_by_handle
        self._last_state = None
        self._last_observation = None
        return ack

    def stage_finger(
        self, side: str, transform: npt.NDArray[np.floating[Any]]
    ) -> None:
        """Stage one side's post-step finger transform for the next exchange."""
        if side not in ("left", "right"):
            raise ValueError(
                f"finger side must be 'left' or 'right', got {side!r}"
            )
        T = np.asarray(transform, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"finger transform must be (4, 4), got {T.shape}")
        self._staged_fingers[side] = T.copy()

    def exchange_staged(
        self, *, timestamp_ns: int = 0
    ) -> TactileObservation | None:
        """Exchange the staged left/right finger transforms as one sample."""
        missing = {"left", "right"} - set(self._staged_fingers)
        if missing:
            raise RuntimeError(
                f"finger transforms not staged for {sorted(missing)}"
            )
        self._staged_step += 1
        sample = FingerSample(
            seq=self._staged_step,
            physx_step=self._staged_step,
            left_T=self._staged_fingers["left"],
            right_T=self._staged_fingers["right"],
            timestamp_ns=timestamp_ns,
        )
        self._staged_fingers = {}
        return self.exchange(sample, wait=True)

    def exchange(
        self, sample: FingerSample, *, wait: bool = False
    ) -> TactileObservation | None:
        state = self._require_runtime().exchange(sample, wait=wait)
        if state is None:
            return self._last_observation
        self._last_state = state
        self._last_observation = self._to_observation(state)
        return self._last_observation

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None

    def _require_runtime(self) -> TactileRuntime:
        if self._runtime is None:
            raise RuntimeError("TactileBridge.start() was not called")
        return self._runtime

    def _to_observation(self, state: TactileState) -> TactileObservation:
        pads: dict[str, PadObservation] = {}
        for handle in self._pad_handles.values():
            pad = self.read_pad(handle)
            pad_cfg = self.pad_config(handle)
            force_kind = force_kind_name(pad.force_kind)
            force = (
                pad.force.astype(np.float32) if force_kind == "net_force" else None
            )
            usb_sd = pad.usb_min_sd_mm if np.isfinite(pad.usb_min_sd_mm) else None
            pads[pad_cfg.side] = PadObservation(
                side=pad_cfg.side,
                vertices=pad.vertices if pad.n_vertices > 0 else None,
                n_vertices=pad.n_vertices,
                contact_gradient=pad.contact_gradient.astype(np.float32),
                force=force,
                force_kind=force_kind,
                contact_count=int(pad.contact_gradient[1]),
                min_sd_proxy_mm=float(pad.contact_gradient[0]),
                min_sd_full_mm=None,
                tracking_error_mm=pad.tracking_error_mm,
                seq=pad.seq,
                age_ms=0.0,
                stale=pad.stale,
                valid=pad.valid,
                pad_usb_min_sd_mm=usb_sd,
                pad_usb_contact_vertices=pad.usb_contact_vertices,
            )
        objects: dict[str, ObjectObservation] = {}
        for handle in self._object_handles.values():
            obj = self.read_object(handle)
            objects[handle.name] = ObjectObservation(
                name=handle.name,
                pose_xyzw=obj.pose_xyzw.astype(np.float32),
                valid=obj.valid,
            )
        return TactileObservation(
            seq=state.seq,
            finger_seq=state.finger_seq,
            physx_step=state.physx_step,
            objects=objects,
            pads=pads,
            age_ms=0.0,
            stale=any(p.stale for p in pads.values()),
        )


__all__ = [
    "TactileBridge",
    "body_pose_to_T",
    "finger_sample_from_body_poses",
    "objects_from_actors",
    "tactile_cfg_from_sensor",
]
