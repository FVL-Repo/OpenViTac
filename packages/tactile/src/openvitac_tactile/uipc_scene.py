"""UIPC soft-pad scene used by the isolated tactile worker.

This module is imported only by :mod:`openvitac_tactile.worker` after the
worker decides to build a real scene.  It deliberately contains no Isaac/pxr
imports and never runs in the parent process.
"""

from __future__ import annotations

import contextlib
import importlib
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .assets import (
    SurfaceMesh,
    TetMesh,
    box_tet_mesh,
    load_tet_mesh,
    published_surface,
)
from .config import ObjectCfg, PadCfg, TactileCfg
from .contact_metrics import pad_object_metrics, select_contact_index
from .protocol import (
    FORCE_KIND_CONTACT_GRADIENT,
    MAX_OBJECTS,
    MAX_PAD_VERTS,
    MAX_PADS,
    FingerSample,
    TactileState,
)
from .softworld import (
    ObjectHandle,
    ObjectState,
    PadHandle,
    PadState,
    PadSurface,
)

_UIPC_ANY: Any = importlib.import_module("uipc")
_UIPC_CONSTITUTION: Any = importlib.import_module("uipc.constitution")
_UIPC_CORE: Any = importlib.import_module("uipc.core")
_UIPC_GEOMETRY: Any = importlib.import_module("uipc.geometry")
_UIPC_UNIT: Any = importlib.import_module("uipc.unit")

_builtin: Any = _UIPC_ANY.builtin
_view: Any = _UIPC_ANY.view
_Transform: Any = _UIPC_ANY.Transform
_Vector3: Any = _UIPC_ANY.Vector3
_Engine: Any = _UIPC_CORE.Engine
_World: Any = _UIPC_CORE.World
_Scene: Any = _UIPC_CORE.Scene
_ContactSystemFeature: Any = _UIPC_CORE.ContactSystemFeature
_Geometry: Any = _UIPC_GEOMETRY.Geometry
_extract_surface: Any = _UIPC_GEOMETRY.extract_surface
_label_surface: Any = _UIPC_GEOMETRY.label_surface
_label_triangle_orient: Any = _UIPC_GEOMETRY.label_triangle_orient
_flip_inward_triangles: Any = _UIPC_GEOMETRY.flip_inward_triangles
_tetmesh: Any = _UIPC_GEOMETRY.tetmesh
_AffineBodyConstitution: Any = _UIPC_CONSTITUTION.AffineBodyConstitution
_ElasticModuli: Any = _UIPC_CONSTITUTION.ElasticModuli
_SoftPositionConstraint: Any = _UIPC_CONSTITUTION.SoftPositionConstraint
_SoftTransformConstraint: Any = _UIPC_CONSTITUTION.SoftTransformConstraint
_StableNeoHookean: Any = _UIPC_CONSTITUTION.StableNeoHookean
_GPa: Any = _UIPC_UNIT.GPa
_MPa: Any = _UIPC_UNIT.MPa

with contextlib.suppress(Exception):
    _UIPC_ANY.Logger.set_level(_UIPC_ANY.Logger.Level.Warn)

_DEFAULT_OBJECT_SIZE = (0.0124, 0.0044, 0.030)
_DEFAULT_FLOOR_SIZE = (1.0, 1.0, 0.004)


def _prep(geometry: Any) -> Any:
    _label_surface(geometry)
    _label_triangle_orient(geometry)
    return _flip_inward_triangles(geometry)


def _transform_points(
    T: NDArray[np.floating[Any]], points: NDArray[np.floating[Any]]
) -> NDArray[np.float64]:
    return (T[:3, :3] @ points.T).T + T[:3, 3]


def _published_pad_surface(
    geometry: Any, *, max_vertices: int, decimation: int
) -> SurfaceMesh:
    """Extract the pad surface exactly as ``pad_vertices`` publishes it.

    UIPC's ``extract_surface`` order is authoritative; applying the same cap and
    decimation here keeps the rest mesh and every per-frame vertex stream in one
    sequence.  Face indices are remapped after dropped vertices.
    """
    surface = _extract_surface(geometry)
    positions = np.asarray(surface.positions().view(), dtype=np.float64).reshape(-1, 3)
    triangles = np.asarray(surface.triangles().topo().view(), dtype=np.int64).reshape(
        -1, 3
    )
    return published_surface(
        positions,
        triangles,
        max_vertices=max_vertices,
        decimation=decimation,
    )


def _translate(x: float, y: float, z: float) -> NDArray[np.float64]:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = (x, y, z)
    return T


def _frame_to_T(pad: PadCfg) -> NDArray[np.float64]:
    x, y, z, qx, qy, qz, qw = (*pad.frame.pos, *pad.frame.rot_xyzw)
    norm = float(np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"pad {pad.name!r} frame rotation is invalid")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]
    T[:3, 3] = (x, y, z)
    return T


def _quat_from_T(T: NDArray[np.floating[Any]]) -> tuple[float, float, float, float]:
    """Quaternion ``(qx, qy, qz, qw)`` from a rotation matrix."""
    m = np.asarray(T, dtype=np.float64)[:3, :3]
    trace = float(np.trace(m))
    if trace > 0.0:
        s = float(np.sqrt(trace + 1.0)) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = float(np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = float(np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = float(np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    if qw < 0.0:
        qw, qx, qy, qz = -qw, -qx, -qy, -qz
    return float(qx), float(qy), float(qz), float(qw)


def _contact_metrics(csf: Any) -> tuple[int, float, float, float]:
    """Return global contact-candidate metrics for the whole scene.

    ``contact_primitive_types()`` is the supported-type list (EE/PE/PP/PT/PH
    x normal/friction); ``contact_gradient()`` fills one instance per candidate
    primitive for that type, including zero-gradient (inactive) instances. The
    summed ``count`` is therefore a global candidate count, not active contact
    and not attributable to a pad. ``normal``/``friction`` are the corresponding
    global gradient magnitudes.
    """
    if csf is None:
        return 0, 0.0, 0.0, 0.0
    count = 0
    normal = 0.0
    friction = 0.0
    for prim_type in csf.contact_primitive_types():
        geometry = _Geometry()
        csf.contact_gradient(prim_type, geometry)
        instances = geometry.instances()
        n = int(instances.size())
        if n == 0:
            continue
        grad_slot = instances.find("grad")
        if grad_slot is None:
            continue
        grad = np.asarray(grad_slot.view()).reshape(-1)
        magnitude = float(np.linalg.norm(grad))
        if str(prim_type).endswith("+N"):
            normal += magnitude
        elif str(prim_type).endswith("+F"):
            friction += magnitude
        count += n
    return count, normal, friction, normal + friction


@dataclass
class _Pad:
    cfg: PadCfg
    sign: float
    slot: Any
    rest: NDArray[np.float64]
    back: NDArray[np.int64]
    front: NDArray[np.int64]
    rest_back: NDArray[np.float64]
    first_link: NDArray[np.float64] | None = None

    @property
    def side(self) -> str:
        return self.cfg.side


@dataclass
class _Object:
    cfg: ObjectCfg
    slot: Any
    rest_surface: NDArray[np.float64]
    rest_T: NDArray[np.float64]
    local_min: NDArray[np.float64]
    local_max: NDArray[np.float64]


def _live_object_surface(obj: _Object) -> NDArray[np.float64]:
    """Current world-space surface vertices of a registered object.

    Affine bodies only move by their instance transform, but finite-element
    bodies deform in place, so their state must be read from vertex positions.
    """
    surface = _extract_surface(obj.slot.geometry())
    return np.asarray(surface.positions().view(), dtype=np.float64).reshape(-1, 3)


def _object_pose(obj: _Object) -> NDArray[np.float64]:
    """Best available world pose for an object.

    ABD pose is the instance transform. FEM has no meaningful instance
    transform, so report the best-fit rigid motion from the rest surface to the
    current surface, composed with the object's rest world transform. This keeps
    ``ObjectState.T`` in world coordinates for both constitutions.
    """
    if obj.cfg.constitution == "abd":
        return np.asarray(
            obj.slot.geometry().transforms().view(), dtype=np.float64
        ).reshape(-1, 4, 4)[0]
    return _rigid_transform(obj.rest_surface, _live_object_surface(obj)) @ obj.rest_T


def _rigid_transform(
    source: NDArray[np.float64], target: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Least-squares rigid transform mapping ``source`` onto ``target``."""
    a = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if a.shape != b.shape or len(a) == 0:
        raise RuntimeError(
            "FEM object surface changed shape between rest and live state"
        )
    ca = a.mean(axis=0)
    cb = b.mean(axis=0)
    decomposition = cast(
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        np.linalg.svd((a - ca).T @ (b - cb)),
    )
    u, _s, vt = decomposition
    correction: float = 1.0 if float(np.linalg.det(vt.T @ u.T)) >= 0.0 else -1.0
    R = vt.T @ np.diag(np.array([1.0, 1.0, correction])) @ u.T
    T: NDArray[np.float64] = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = cb - R @ ca
    return T


@dataclass
class _Finger:
    side: str
    slot: Any
    B0: NDArray[np.float64]
    first_link: NDArray[np.float64] | None = None


@dataclass
class UipcScene:
    """Live UIPC scene implementing the minimal :class:`SoftWorld` manager API.

    The constructor arguments are the already-built UIPC world plus the shared
    constitution/element handles; pads and objects are added through
    :meth:`register_pad`/:meth:`register_object` so callers never touch
    ``_Pad``/``_Object``.  The legacy ``TactileState`` is exposed through
    :meth:`legacy_state` only as a SHM adapter.
    """

    cfg: TactileCfg
    world: Any
    engine: Any
    scene: Any
    contact_feature: Any = None
    _pads: list[_Pad] = field(default_factory=list[_Pad], repr=False)
    _objects: list[_Object] = field(default_factory=list[_Object], repr=False)
    _contact_object: _Object | None = field(default=None, repr=False)
    _fingers: list[_Finger] = field(default_factory=list[_Finger], repr=False)
    seq: int = 0
    _request: dict[str, Any] = field(default_factory=dict[str, Any], repr=False)
    _element: Any = None
    _abd: Any = None
    _snh: Any = None
    _soft: Any = None
    _pad_by_handle: dict[PadHandle, _Pad] = field(
        default_factory=dict[PadHandle, _Pad], repr=False
    )
    _object_by_handle: dict[ObjectHandle, _Object] = field(
        default_factory=dict[ObjectHandle, _Object], repr=False
    )
    _next_pad_uid: int = 0
    _next_object_uid: int = 0
    _last_state: TactileState | None = field(default=None, repr=False)

    @property
    def pad_handles(self) -> tuple[PadHandle, ...]:
        return tuple(self._pad_by_handle)

    @property
    def object_handles(self) -> tuple[ObjectHandle, ...]:
        return tuple(self._object_by_handle)

    def pad_handle(self, name: str) -> PadHandle:
        for handle in self._pad_by_handle:
            if handle.name == name:
                return handle
        raise ValueError(f"unknown pad name {name!r}")

    def object_handle(self, name: str) -> ObjectHandle:
        for handle in self._object_by_handle:
            if handle.name == name:
                return handle
        raise ValueError(f"unknown object name {name!r}")

    def register_pad(self, cfg: PadCfg) -> PadHandle:
        """Create and register one soft pad; returns its stable handle."""
        if any(handle.name == cfg.name for handle in self._pad_by_handle):
            raise ValueError(f"pad {cfg.name!r} is already registered")
        if self._snh is None or self._soft is None or self._element is None:
            raise RuntimeError("UipcScene was not created by build_scene()")
        sign = -1.0 if cfg.side == "left" else 1.0
        local_mesh = _load_pad_mesh(cfg)
        local_vertices = np.asarray(local_mesh.vertices, dtype=np.float64)
        back, front = _pad_indices(local_vertices, cfg, sign)
        T_pad = _frame_to_T(cfg)
        request_T = _request_transform(self._request, "pads", cfg.name)
        body_pad_raw = _request_named_transform(
            self._request, "pads", cfg.name, "T_body_pad0"
        )
        body_pad = body_pad_raw if body_pad_raw is not None else _frame_to_T(cfg)
        if request_T is not None:
            T_pad = request_T @ body_pad
        rest = _transform_points(T_pad, local_vertices)
        geom = _prep(_tetmesh(np.ascontiguousarray(rest), local_mesh.tets))
        moduli = _ElasticModuli.youngs_poisson(
            float(cfg.material.youngs_mpa) * _MPa,
            float(cfg.material.poisson),
        )
        self._snh.apply_to(geom, moduli, mass_density=float(cfg.material.density_kg_m3))
        self._soft.apply_to(geom, float(cfg.material.constraint_strength))
        _view(geom.vertices().find(_builtin.is_constrained))[back] = 1
        aim = _view(geom.vertices().find(_builtin.aim_position))
        aim[back] = rest[back][:, :, None]
        self._element.apply_to(geom)
        slot = self.scene.objects().create(cfg.name).geometries().create(geom)[0]
        pad = _Pad(
            cfg=cfg,
            sign=sign,
            slot=slot,
            rest=rest,
            back=back,
            front=front,
            rest_back=rest[back].copy(),
        )
        handle = PadHandle(
            uid=self._next_pad_uid,
            name=cfg.name,
            side=cfg.side,
            index=len(self._pads),
        )
        self._next_pad_uid += 1
        self._pads.append(pad)
        self._pad_by_handle[handle] = pad
        return handle

    def register_object(self, cfg: ObjectCfg) -> ObjectHandle:
        """Create and register one UIPC-authority object; returns its handle."""
        if any(handle.name == cfg.name for handle in self._object_by_handle):
            raise ValueError(f"object {cfg.name!r} is already registered")
        if cfg.authority != "uipc":
            raise ValueError(f"object {cfg.name!r} is not UIPC-authority")
        if cfg.constitution not in ("abd", "fem"):
            raise ValueError(
                f"object {cfg.name!r} has unsupported constitution {cfg.constitution!r}"
            )
        if self._element is None:
            raise RuntimeError("UipcScene was not created by build_scene()")
        if cfg.constitution == "abd" and self._abd is None:
            raise RuntimeError("UipcScene has no ABD constitution")
        if cfg.constitution == "fem" and self._snh is None:
            raise RuntimeError("UipcScene has no FEM constitution")
        try:
            index = next(
                i for i, item in enumerate(self.cfg.objects) if item.name == cfg.name
            )
        except StopIteration as exc:
            raise ValueError(f"object {cfg.name!r} is not in the scene config") from exc
        T_obj = _request_transform(self._request, "objects", cfg.name)
        if T_obj is None:
            T_obj = _translate(0.0, 0.0, 0.035)

        def build_geometry(mesh: TetMesh) -> Any:
            if cfg.constitution == "fem":
                assert cfg.youngs_mpa is not None and cfg.poisson is not None
                # FEM vertices are the state, so placement must happen before
                # the geometry is created; instance transforms are ignored.
                world_vertices = _transform_points(T_obj, mesh.vertices)
                geometry = _prep(
                    _tetmesh(
                        np.ascontiguousarray(world_vertices, dtype=np.float64),
                        mesh.tets,
                    )
                )
                _add_fem(
                    self._element,
                    self._snh,
                    geometry,
                    density=float(cfg.density_kg_m3),
                    youngs_mpa=float(cfg.youngs_mpa),
                    poisson=float(cfg.poisson),
                    fixed=bool(cfg.fixed),
                )
                return geometry
            geometry = _prep(
                _tetmesh(
                    np.ascontiguousarray(mesh.vertices, dtype=np.float64),
                    mesh.tets,
                )
            )
            _add_abd(
                self._element,
                self._abd,
                geometry,
                T_obj,
                density=float(cfg.density_kg_m3),
                kappa_mpa=float(cfg.kappa_mpa),
                fixed=bool(cfg.fixed),
            )
            return geometry

        if cfg.fixed and cfg.boxes:
            obj_slot = self.scene.objects().create(cfg.name)
            first_slot: Any = None
            for center, half in cfg.boxes:
                box_mesh = box_tet_mesh(
                    (2.0 * half[0], 2.0 * half[1], 2.0 * half[2]),
                    (1, 1, 1),
                    center=center,
                )
                box_geom = build_geometry(box_mesh)
                box_slot = obj_slot.geometries().create(box_geom)[0]
                if first_slot is None:
                    first_slot = box_slot
            assert first_slot is not None
            surface = _extract_surface(first_slot.geometry())
            rest_surface = np.asarray(
                surface.positions().view(), dtype=np.float64
            ).reshape(-1, 3)
            obj = _Object(
                cfg=cfg,
                slot=first_slot,
                rest_surface=rest_surface.copy(),
                rest_T=T_obj.copy(),
                local_min=rest_surface.min(axis=0).copy(),
                local_max=rest_surface.max(axis=0).copy(),
            )
        else:
            if cfg.mesh_path is not None:
                mesh = load_tet_mesh(cfg.mesh_path)
            else:
                mesh = box_tet_mesh(_DEFAULT_OBJECT_SIZE, (3, 1, 6))
            geom = build_geometry(mesh)
            slot = self.scene.objects().create(cfg.name).geometries().create(geom)[0]
            surface = _extract_surface(slot.geometry())
            rest_surface = np.asarray(
                surface.positions().view(), dtype=np.float64
            ).reshape(-1, 3)
            obj = _Object(
                cfg=cfg,
                slot=slot,
                rest_surface=rest_surface.copy(),
                rest_T=T_obj.copy(),
                local_min=rest_surface.min(axis=0).copy(),
                local_max=rest_surface.max(axis=0).copy(),
            )
        handle = ObjectHandle(uid=self._next_object_uid, name=cfg.name, index=index)
        self._next_object_uid += 1
        self._objects.append(obj)
        self._object_by_handle[handle] = obj
        return handle

    def _add_floor(self) -> None:
        if self._element is None or self._abd is None:
            raise RuntimeError("UipcScene was not created by build_scene()")
        floor_mesh = box_tet_mesh(
            _DEFAULT_FLOOR_SIZE, (4, 4, 1), center=(0.0, 0.0, -0.002)
        )
        floor = _prep(
            _tetmesh(
                np.ascontiguousarray(floor_mesh.vertices, dtype=np.float64),
                floor_mesh.tets,
            )
        )
        _add_abd(
            self._element,
            self._abd,
            floor,
            np.eye(4),
            density=1.0e3,
            kappa_mpa=100.0,
            fixed=True,
        )
        self.scene.objects().create("static_slot").geometries().create(floor)

    def _register_fingers(self) -> None:
        if not self.cfg.include_uipc_fingers:
            return
        if self._element is None or self._abd is None:
            raise RuntimeError("UipcScene was not created by build_scene()")
        stc = _SoftTransformConstraint()
        for sign, side in ((-1.0, "left"), (1.0, "right")):
            mesh = box_tet_mesh((0.012, 0.004, 0.020), (3, 2, 4))
            geom = _prep(
                _tetmesh(
                    np.ascontiguousarray(mesh.vertices, dtype=np.float64),
                    mesh.tets,
                )
            )
            _add_abd(
                self._element,
                self._abd,
                geom,
                _translate(0.0, sign * 0.025, 0.035),
                density=1.0e3,
                kappa_mpa=100.0,
            )
            stc.apply_to(geom, [1.0e4, 1.0e4])
            _view(geom.instances().find(_builtin.is_constrained))[0] = 1
            B0 = _translate(0.0, sign * 0.025, 0.035)
            _view(geom.transforms())[:] = B0
            slot = (
                self.scene.objects()
                .create(f"finger_{side}")
                .geometries()
                .create(geom)[0]
            )
            self._fingers.append(_Finger(side=side, slot=slot, B0=B0))

    def _select_contact_object(self) -> None:
        contact_index = select_contact_index(
            [
                (obj.cfg.name, obj.cfg.fixed, obj.cfg.contact_object)
                for obj in self._objects
            ]
        )
        self._contact_object = (
            self._objects[contact_index] if contact_index is not None else None
        )

    def complete_registration(self) -> None:
        """Add the fixed floor/fingers and select the contact object."""
        self._add_floor()
        self._register_fingers()
        self._select_contact_object()

    def _require_pad(self, handle: PadHandle) -> _Pad:
        pad = self._pad_by_handle.get(handle)
        if pad is None:
            raise ValueError(f"unknown pad handle {handle!r}")
        return pad

    def _require_object(self, handle: ObjectHandle) -> _Object:
        obj = self._object_by_handle.get(handle)
        if obj is None:
            raise ValueError(f"unknown object handle {handle!r}")
        return obj

    def _require_state(self) -> TactileState:
        if self._last_state is None:
            raise RuntimeError("SoftWorld has no state yet; call step() first")
        return self._last_state

    def read_pad(self, handle: PadHandle) -> PadState:
        self._require_pad(handle)
        state = self._require_state()
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
        self._require_object(handle)
        state = self._require_state()
        index = handle.index
        return ObjectState(
            handle=handle,
            seq=state.seq,
            finger_seq=state.finger_seq,
            physx_step=state.physx_step,
            timestamp_us=state.timestamp_us,
            T=np.array(state.object_T[index], dtype=np.float64, copy=True),
            pose_xyzw=np.array(state.object_pose[index], dtype=np.float64, copy=True),
            valid=bool(state.object_valid[index]),
        )

    def pad_surfaces(self) -> dict[PadHandle, PadSurface]:
        """Rest surface exactly as published in ``pad_vertices``.

        The vertices are world-space and share order with every ``PadState``
        vertex stream; the same cap/decimation is applied to both.
        """
        out: dict[PadHandle, PadSurface] = {}
        max_vertices = min(self.cfg.sync.max_pad_vertices, MAX_PAD_VERTS)
        for handle, pad in self._pad_by_handle.items():
            surface = _published_pad_surface(
                pad.slot.geometry(),
                max_vertices=max_vertices,
                decimation=pad.cfg.geometry.surface_decimation,
            )
            if surface.vertices.shape[0] == 0 or surface.triangles.shape[0] == 0:
                continue
            out[handle] = PadSurface(
                handle=handle,
                rest_vertices=surface.vertices.astype(np.float64),
                triangles=surface.triangles,
            )
        return out

    def legacy_state(self) -> TactileState:
        """Adapter for the existing SHM ``TactileState`` byte layout."""
        return self._require_state()

    def initial_state(self) -> TactileState:
        state = self._state(
            FingerSample(
                seq=0,
                physx_step=0,
                left_T=np.eye(4),
                right_T=np.eye(4),
                timestamp_ns=0,
            ),
            advance=False,
        )
        self._last_state = state
        return state

    def pad_surface_data(self) -> list[dict[str, Any]]:
        """Legacy JSON adapter for the worker reset ack.

        The parent uses these vertices/triangles to rewrite the visual gelpad
        mesh; both come from the same published-surface path used by
        ``pad_vertices``, so their order cannot drift.
        """
        return [
            {
                "name": surface.name,
                "side": surface.side,
                "rest_vertices": surface.rest_vertices.tolist(),
                "triangles": surface.triangles.tolist(),
            }
            for surface in self.pad_surfaces().values()
        ]

    def step(self, sample: FingerSample) -> None:
        for pad in self._pads:
            link = sample.left_T if pad.side == "left" else sample.right_T
            if pad.first_link is None:
                pad.first_link = np.asarray(link, dtype=np.float64).copy()
            motion = link @ np.linalg.inv(pad.first_link)
            target = _transform_points(motion, pad.rest_back)
            aim = _view(pad.slot.geometry().vertices().find(_builtin.aim_position))
            aim[pad.back] = target[:, :, None]
        for finger in self._fingers:
            link = sample.left_T if finger.side == "left" else sample.right_T
            if finger.first_link is None:
                finger.first_link = np.asarray(link, dtype=np.float64).copy()
            motion = link @ np.linalg.inv(finger.first_link)
            aim = _view(finger.slot.geometry().instances().find(_builtin.aim_transform))
            aim[:] = motion @ finger.B0
        self.world.advance()
        self.world.retrieve()
        self._last_state = self._state(sample, advance=True)

    def _state(self, sample: FingerSample, *, advance: bool) -> TactileState:
        if advance:
            self.seq += 1
        object_T = np.zeros((MAX_OBJECTS, 4, 4), dtype=np.float64)
        object_pose = np.zeros((MAX_OBJECTS, 7), dtype=np.float64)
        object_valid = np.zeros((MAX_OBJECTS,), dtype=np.bool_)
        built = {obj.cfg.name: obj for obj in self._objects}
        for index, obj_cfg in enumerate(self.cfg.objects[:MAX_OBJECTS]):
            obj = built.get(obj_cfg.name)
            if obj is None:
                continue
            T = _object_pose(obj)
            qx, qy, qz, qw = _quat_from_T(T)
            object_T[index] = T
            object_pose[index] = [*T[:3, 3], qx, qy, qz, qw]
            object_valid[index] = bool(np.all(np.isfinite(T)))
        pad_vertices = np.zeros((MAX_PADS, MAX_PAD_VERTS, 3), dtype=np.float32)
        pad_n = np.zeros((MAX_PADS,), dtype=np.int32)
        pad_gradient = np.full((MAX_PADS, 4), np.nan, dtype=np.float64)
        pad_force = np.full((MAX_PADS, 3), np.nan, dtype=np.float64)
        pad_force_kind = np.full(
            (MAX_PADS,), FORCE_KIND_CONTACT_GRADIENT, dtype=np.uint8
        )
        pad_tracking = np.zeros((MAX_PADS,), dtype=np.float64)
        pad_stale = np.zeros((MAX_PADS,), dtype=np.bool_)
        pad_valid = np.zeros((MAX_PADS,), dtype=np.bool_)
        pad_usb_min_sd = np.full((MAX_PADS,), np.nan, dtype=np.float64)
        pad_usb_contact_vertices = np.zeros((MAX_PADS,), dtype=np.int32)
        contact_T: NDArray[np.float64] | None = None
        object_points: NDArray[np.float64] | None = None
        contact = self._contact_object
        if contact is not None:
            if contact.cfg.constitution == "fem":
                object_points = _live_object_surface(contact)
                contact_T = (
                    _rigid_transform(contact.rest_surface, object_points)
                    @ contact.rest_T
                )
            else:
                live_T = np.asarray(
                    contact.slot.geometry().transforms().view(),
                    dtype=np.float64,
                ).reshape(-1, 4, 4)[0]
                contact_T = live_T
                object_points = _transform_points(live_T, contact.rest_surface)
        count, grad_normal, _grad_friction, grad_total = _contact_metrics(
            self.contact_feature
        )
        max_vertices = min(self.cfg.sync.max_pad_vertices, MAX_PAD_VERTS)
        for index, pad in enumerate(self._pads[:MAX_PADS]):
            geometry = pad.slot.geometry()
            surface = _published_pad_surface(
                geometry,
                max_vertices=max_vertices,
                decimation=pad.cfg.geometry.surface_decimation,
            )
            positions = surface.vertices
            n = len(positions)
            if n > 0:
                pad_vertices[index, :n] = positions
            pad_n[index] = n
            if contact_T is not None and n > 0 and self._contact_object is not None:
                metrics = pad_object_metrics(
                    positions,
                    contact_T,
                    self._contact_object.local_min,
                    self._contact_object.local_max,
                    float(pad.cfg.material.contact_d_hat_m),
                )
                pad_usb_min_sd[index] = metrics.min_sd_mm
                pad_usb_contact_vertices[index] = metrics.contact_vertices
            live_positions = np.asarray(
                geometry.positions().view(), dtype=np.float64
            ).reshape(-1, 3)
            target = _transform_points(
                (
                    (sample.left_T if pad.side == "left" else sample.right_T)
                    @ np.linalg.inv(
                        pad.first_link if pad.first_link is not None else np.eye(4)
                    )
                ),
                pad.rest_back,
            )
            tracking = float(
                np.linalg.norm(live_positions[pad.back] - target, axis=1).mean()
                * 1000.0
            )
            pad_tracking[index] = tracking
            if object_points is not None:
                front = live_positions[pad.front]
                dx = object_points[:, None, 0] - front[None, :, 0]
                dz = object_points[:, None, 2] - front[None, :, 2]
                nearest = np.argmin(dx * dx + dz * dz, axis=1)
                fy = front[nearest, 1]
                signed = (
                    object_points[:, 1] - fy
                    if pad.sign < 0.0
                    else fy - object_points[:, 1]
                )
                min_sd_mm = float(np.min(signed) * 1000.0)
            else:
                min_sd_mm = float("nan")
            pad_gradient[index] = [min_sd_mm, count, grad_normal, grad_total]
            pad_valid[index] = bool(n > 0 and np.all(np.isfinite(pad_vertices[index])))
            pad_force_kind[index] = FORCE_KIND_CONTACT_GRADIENT
        return TactileState(
            seq=self.seq,
            finger_seq=sample.seq,
            physx_step=sample.physx_step,
            timestamp_us=(
                sample.timestamp_ns // 1000
                if sample.timestamp_ns
                else int(time.time_ns() // 1000)
            ),
            object_T=object_T,
            object_pose=object_pose,
            object_valid=object_valid,
            n_objects=min(len(self.cfg.objects), MAX_OBJECTS),
            pad_vertices=pad_vertices,
            pad_n_vertices=pad_n,
            pad_contact_gradient=pad_gradient,
            pad_force=pad_force,
            pad_force_kind=pad_force_kind,
            pad_tracking_error_mm=pad_tracking,
            pad_stale=pad_stale,
            pad_valid=pad_valid,
            pad_usb_min_sd_mm=pad_usb_min_sd,
            pad_usb_contact_vertices=pad_usb_contact_vertices,
        )

    def close(self) -> None:
        # Some UIPC builds can raise while tearing down contact-heavy worlds;
        # the worker already published its final bytes, so teardown is best effort.
        try:
            self.world = None
            self.engine = None
        except Exception:
            pass


def _load_pad_mesh(pad: PadCfg) -> TetMesh:
    geometry = pad.geometry
    if geometry.kind == "tet_mesh":
        assert geometry.mesh_path is not None
        return load_tet_mesh(geometry.mesh_path)
    return box_tet_mesh(geometry.size_m, geometry.tet_resolution)


def _pad_indices(
    local_vertices: NDArray[np.float64], pad: PadCfg, sign: float
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    axis = {"x": 0, "y": 1, "z": 2}[pad.frame.back_axis]
    values = local_vertices[:, axis]
    if np.allclose(values.max(), values.min()):
        raise ValueError(f"pad {pad.name!r} is flat on back axis {axis}")
    back_is_min = pad.frame.back_is_min
    if back_is_min is None:
        back_is_min = sign < 0.0
    if back_is_min:
        back = np.where(np.abs(values - values.min()) < 1e-9)[0]
        front = np.where(np.abs(values - values.max()) < 1e-9)[0]
    else:
        back = np.where(np.abs(values - values.max()) < 1e-9)[0]
        front = np.where(np.abs(values - values.min()) < 1e-9)[0]
    return back.astype(np.int64), front.astype(np.int64)


def _add_abd(
    element: Any,
    abd: Any,
    geometry: Any,
    T: NDArray[np.float64],
    *,
    density: float,
    kappa_mpa: float,
    fixed: bool = False,
) -> Any:
    abd.apply_to(geometry, kappa_mpa * _MPa, mass_density=density)
    if fixed:
        _view(geometry.instances().find(_builtin.is_fixed))[0] = 1
    _view(geometry.transforms())[:] = T
    element.apply_to(geometry)
    return geometry


def _add_fem(
    element: Any,
    snh: Any,
    geometry: Any,
    *,
    density: float,
    youngs_mpa: float,
    poisson: float,
    fixed: bool = False,
) -> Any:
    """Apply StableNeoHookean to a tetrahedral geometry.

    FEM kinematics live in vertex positions, so ``geometry`` must already carry
    world-space vertices (there is no instance transform to set here).
    """
    moduli = _ElasticModuli.youngs_poisson(
        float(youngs_mpa) * _MPa,
        float(poisson),
    )
    snh.apply_to(geometry, moduli, mass_density=float(density))
    if fixed:
        fixed_attr = geometry.vertices().find(_builtin.is_fixed)
        if fixed_attr is None:
            fixed_attr = geometry.vertices().create(_builtin.is_fixed, 0)
        _view(fixed_attr)[:] = 1
    element.apply_to(geometry)
    return geometry


def _request_transform(
    reset_request: dict[str, Any], key: str, name: str
) -> NDArray[np.float64] | None:
    return _request_named_transform(reset_request, key, name, "T_world0")


def _request_named_transform(
    reset_request: dict[str, Any], key: str, name: str, field_name: str
) -> NDArray[np.float64] | None:
    entries = reset_request.get(key)
    if not isinstance(entries, list):
        return None
    for entry_raw in cast(list[Any], entries):
        if not isinstance(entry_raw, dict):
            continue
        entry = cast(dict[str, Any], entry_raw)
        if entry.get("name") != name:
            continue
        raw = entry.get(field_name)
        if raw is None and field_name == "T_world0":
            raw = entry.get("T_world_body0")
        if raw is None:
            continue
        T = np.asarray(raw, dtype=np.float64).reshape(4, 4)
        if not np.all(np.isfinite(T)):
            raise ValueError(f"reset request transform for {name!r} is not finite")
        return T
    return None


def build_scene(
    cfg: TactileCfg,
    reset_request: dict[str, Any] | None = None,
    *,
    workspace: str | Path | None = None,
) -> UipcScene:
    """Build the validated UIPC scene from a tactile configuration."""
    cfg.validate()
    if not cfg.enabled or not cfg.pads:
        raise ValueError("UIPC scene requires enabled=True and at least one pad")
    request = dict(reset_request or {})
    root = Path(workspace or cfg.worker.workspace)
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    engine = _Engine("cuda", str(root))
    world = _World(engine)
    scene_cfg = _Scene.default_config()
    scene_cfg["dt"] = cfg.dt_s
    scene_cfg["gravity"] = [[float(v)] for v in cfg.gravity_mps2]
    scene_cfg["contact"]["enable"] = True
    scene_cfg["contact"]["constitution"] = "ipc"
    scene_cfg["contact"]["d_hat"] = float(cfg.pads[0].material.contact_d_hat_m)
    scene_cfg["contact"]["eps_velocity"] = 0.1
    scene_cfg["contact"]["friction"]["enable"] = True
    scene = _Scene(scene_cfg)
    scene.contact_tabular().default_model(
        friction_rate=float(cfg.pads[0].material.friction_rate),
        resistance=float(cfg.pads[0].material.resistance_pa),
        enable=True,
    )
    element = scene.contact_tabular().default_element()
    abd = _AffineBodyConstitution()

    snh = _StableNeoHookean()
    soft = _SoftPositionConstraint()
    manager = UipcScene(
        cfg=cfg,
        world=world,
        engine=engine,
        scene=scene,
        _request=request,
        _element=element,
        _abd=abd,
        _snh=snh,
        _soft=soft,
    )
    for pad_cfg in cfg.pads:
        manager.register_pad(pad_cfg)

    for obj_cfg in cfg.objects:
        if obj_cfg.authority != "uipc":
            continue
        manager.register_object(obj_cfg)

    # A fixed floor is the only static collision geometry in the minimal scene;
    # task-specific slots can be added by extending ObjectCfg later.
    manager.complete_registration()

    world.init(scene)
    if not world.is_valid():
        raise RuntimeError(f"UIPC world invalid; inspect {root}/sanity_check")
    world.retrieve()
    manager.contact_feature = world.features().find(_ContactSystemFeature)
    return manager
