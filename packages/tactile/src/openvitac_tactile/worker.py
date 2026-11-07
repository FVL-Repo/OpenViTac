"""UIPC worker: RESET builds a scene, READY/health/stop stay on stdin.

The parent process owns both shared-memory channels.  This module imports
``uipc`` only when a reset request actually builds a scene (and only when the
parent started the worker with ``--require-uipc``); the default no-pyuipc path
keeps the stage-1 skeleton acknowledgement so callers can opt into a fake
worker without installing pyuipc.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import selectors
import sys
import time
from collections.abc import Mapping
from multiprocessing import shared_memory
from typing import Any, cast

from . import protocol
from .config import (
    ObjectCfg,
    PadCfg,
    PadFrameCfg,
    PadGeometryCfg,
    PadMaterialCfg,
    ResetCfg,
    SyncCfg,
    TactileCfg,
)


def _import_uipc() -> Any:
    """Lazy import; called only inside a real reset/build."""
    return importlib.import_module("uipc")


_control_fd: int | None = None


def _redirect_control_stream() -> None:
    """Keep UIPC's fd-level logging off the JSON control pipe."""
    global _control_fd
    sys.stdout.flush()
    _control_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())


def _send(payload: dict[str, Any]) -> None:
    data = (json.dumps(payload) + "\n").encode("utf-8")
    if _control_fd is not None:
        os.write(_control_fd, data)
    else:
        sys.stdout.write(data.decode("utf-8"))
        sys.stdout.flush()


def _mapping(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{what} must be an object")
    return cast(dict[str, Any], value)


def _tuple3(value: Any, what: str) -> tuple[float, float, float]:
    items = tuple(float(v) for v in value)
    if len(items) != 3:
        raise ValueError(f"{what} must have 3 values")
    return items


def _tuple4(value: Any, what: str) -> tuple[float, float, float, float]:
    items = tuple(float(v) for v in value)
    if len(items) != 4:
        raise ValueError(f"{what} must have 4 values")
    return items


def _int_tuple3(value: Any, what: str) -> tuple[int, int, int]:
    items = tuple(int(v) for v in value)
    if len(items) != 3:
        raise ValueError(f"{what} must have 3 values")
    return items


def _pad_from_dict(raw: Mapping[str, Any]) -> PadCfg:
    frame = _mapping(raw["frame"], "pad.frame")
    geometry = _mapping(raw["geometry"], "pad.geometry")
    material = _mapping(raw["material"], "pad.material")
    return PadCfg(
        side=raw["side"],
        name=str(raw["name"]),
        gelpad_prim_path=str(raw["gelpad_prim_path"]),
        frame=PadFrameCfg(
            body_name=str(frame["body_name"]),
            body_prim_path=str(frame["body_prim_path"]),
            pos=_tuple3(frame["pos"], "pad.frame.pos"),
            rot_xyzw=_tuple4(frame["rot_xyzw"], "pad.frame.rot_xyzw"),
            back_axis=frame.get("back_axis", "y"),
            back_value_m=float(frame.get("back_value_m", 0.0)),
            back_is_min=(
                None if frame.get("back_is_min") is None else bool(frame["back_is_min"])
            ),
        ),
        geometry=PadGeometryCfg(
            kind=geometry.get("kind", "box"),
            size_m=_tuple3(geometry.get("size_m", (0.03, 0.004, 0.02)), "size_m"),
            mesh_path=(
                None
                if geometry.get("mesh_path") is None
                else str(geometry["mesh_path"])
            ),
            tet_resolution=_int_tuple3(
                geometry.get("tet_resolution", (8, 1, 5)), "tet_resolution"
            ),
            max_surface_vertices=int(geometry.get("max_surface_vertices", 1024)),
            surface_decimation=int(geometry.get("surface_decimation", 1)),
        ),
        material=PadMaterialCfg(
            youngs_mpa=float(material.get("youngs_mpa", 0.1)),
            poisson=float(material.get("poisson", 0.49)),
            density_kg_m3=float(material.get("density_kg_m3", 1.0e4)),
            constraint_strength=float(material.get("constraint_strength", 1.0e4)),
            contact_d_hat_m=float(material.get("contact_d_hat_m", 5.0e-4)),
            friction_rate=float(material.get("friction_rate", 2.5)),
            resistance_pa=float(material.get("resistance_pa", 1.0e9)),
        ),
        image_width=(
            None if raw.get("image_width") is None else int(raw["image_width"])
        ),
        image_height=(
            None if raw.get("image_height") is None else int(raw["image_height"])
        ),
        marker_rows=(
            None if raw.get("marker_rows") is None else int(raw["marker_rows"])
        ),
        marker_cols=(
            None if raw.get("marker_cols") is None else int(raw["marker_cols"])
        ),
        render_kinds=tuple(raw.get("render_kinds", ())),
    )


def _object_from_dict(raw: Mapping[str, Any]) -> ObjectCfg:
    return ObjectCfg(
        name=str(raw["name"]),
        prim_path=str(raw["prim_path"]),
        authority=raw.get("authority", "uipc"),
        constitution=raw.get("constitution", "abd"),
        kappa_mpa=float(raw.get("kappa_mpa", 100.0)),
        youngs_mpa=(
            None if raw.get("youngs_mpa") is None else float(raw["youngs_mpa"])
        ),
        poisson=(None if raw.get("poisson") is None else float(raw["poisson"])),
        density_kg_m3=float(raw.get("density_kg_m3", 1.0e3)),
        mesh_path=(None if raw.get("mesh_path") is None else str(raw["mesh_path"])),
        usd_path=(
            None if raw.get("usd_path") is None else str(raw["usd_path"])
        ),
        fixed=bool(raw.get("fixed", False)),
        contact_object=bool(raw.get("contact_object", False)),
        boxes=tuple(
            (
                (float(center[0]), float(center[1]), float(center[2])),
                (float(half[0]), float(half[1]), float(half[2])),
            )
            for center, half in raw.get("boxes", ())
        ),
        max_speed_mps=float(raw.get("max_speed_mps", 5.0)),
        max_penetration_mm=float(raw.get("max_penetration_mm", 0.25)),
        max_pose_jump_m=float(raw.get("max_pose_jump_m", 0.05)),
    )


def _cfg_from_request(request: Mapping[str, Any]) -> TactileCfg | None:
    raw = request.get("config")
    if raw is None:
        return None
    data = _mapping(raw, "config")
    pads = tuple(
        _pad_from_dict(_mapping(item, "config.pads[]")) for item in data.get("pads", ())
    )
    objects = tuple(
        _object_from_dict(_mapping(item, "config.objects[]"))
        for item in data.get("objects", ())
    )
    reset_raw = _mapping(data.get("reset", {}), "config.reset")
    sync_raw = _mapping(data.get("sync", {}), "config.sync")
    return TactileCfg(
        enabled=bool(data.get("enabled", True)),
        pads=pads,
        objects=objects,
        dt_s=float(data.get("dt_s", 1.0 / 120.0)),
        substeps=int(data.get("substeps", 1)),
        include_uipc_fingers=bool(data.get("include_uipc_fingers", False)),
        gravity_mps2=_tuple3(data.get("gravity_mps2", (0, 0, -9.8)), "gravity"),
        reset=ResetCfg(
            settle_max_steps=int(reset_raw.get("settle_max_steps", 0)),
            warmup_steps=int(reset_raw.get("warmup_steps", 0)),
        ),
        sync=SyncCfg(
            mode=sync_raw.get("mode", "bounded_lag"),
            max_lag_steps=int(sync_raw.get("max_lag_steps", 2)),
            state_timeout_ms=float(sync_raw.get("state_timeout_ms", 20.0)),
            max_stale_frames=int(sync_raw.get("max_stale_frames", 0)),
            max_finger_age_ms=float(sync_raw.get("max_finger_age_ms", 20.0)),
            publish_pad_vertices=bool(sync_raw.get("publish_pad_vertices", True)),
            max_pad_vertices=int(sync_raw.get("max_pad_vertices", 1024)),
            finger_history=int(sync_raw.get("finger_history", 4)),
        ),
    )


def _scene_hash(cfg: TactileCfg) -> str:
    return hashlib.sha1(
        json.dumps(cfg.__dict__, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


class Worker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.state_shm = shared_memory.SharedMemory(name=args.state_shm)
        self.finger_shm = shared_memory.SharedMemory(name=args.finger_shm)
        state_buf = self.state_shm.buf
        finger_buf = self.finger_shm.buf
        assert state_buf is not None and finger_buf is not None
        self.state_buf = state_buf
        self.finger_buf = finger_buf
        self.scene: Any = None
        self.last_finger_seq = 0
        self.last_state_seq = 0
        self.cfg: TactileCfg | None = None

    def _close_scene(self) -> None:
        if self.scene is not None:
            try:
                self.scene.close()
            finally:
                self.scene = None

    def _reset(self, msg: Mapping[str, Any]) -> None:
        request_raw = msg.get("request", {})
        request = _mapping(request_raw, "reset.request")
        try:
            cfg = _cfg_from_request(request)
        except Exception as exc:
            self._close_scene()
            self.cfg = None
            _send(
                {
                    "op": "reset_ack",
                    "ack": {
                        "status": "failed",
                        "message": f"invalid tactile config: {exc}",
                    },
                }
            )
            return
        if cfg is None:
            self._close_scene()
            self.cfg = None
            _send(
                {
                    "op": "reset_ack",
                    "ack": {
                        "status": "ready",
                        "message": "skeleton reset (no serialized TactileCfg)",
                        "uipc_dt_s": 0.0,
                        "scene_hash": "",
                        "n_pads": 0,
                        "n_objects": 0,
                        "pads": [],
                        "seq": 0,
                        "object_z0": {},
                        "build_ms": 0.0,
                        "settle_ms": 0.0,
                    },
                }
            )
            return
        self.cfg = cfg
        if not cfg.enabled or not cfg.pads:
            self._close_scene()
            _send(
                {
                    "op": "reset_ack",
                    "ack": {
                        "status": "ready",
                        "message": "tactile disabled (no world built)",
                        "uipc_dt_s": cfg.dt_s,
                        "scene_hash": _scene_hash(cfg),
                        "n_pads": 0,
                        "n_objects": 0,
                        "pads": [],
                        "seq": 0,
                        "object_z0": {},
                        "build_ms": 0.0,
                        "settle_ms": 0.0,
                    },
                }
            )
            return
        if not self.args.require_uipc:
            self._close_scene()
            _send(
                {
                    "op": "reset_ack",
                    "ack": {
                        "status": "ready",
                        "message": "skeleton reset (pyuipc disabled in worker)",
                        "uipc_dt_s": cfg.dt_s,
                        "scene_hash": _scene_hash(cfg),
                        "n_pads": 0,
                        "n_objects": 0,
                        "pads": [],
                        "seq": 0,
                        "object_z0": {},
                        "build_ms": 0.0,
                        "settle_ms": 0.0,
                    },
                }
            )
            return

        started = time.perf_counter()
        try:
            _import_uipc()
            from .uipc_scene import build_scene

            scene = build_scene(cfg, request)
            state = scene.initial_state()
            protocol.write_state(self.state_buf, state)
            self.scene = scene
            self.last_finger_seq = 0
            self.last_state_seq = 0
            z_values = {
                handle.name: float(scene.read_object(handle).T[2, 3])
                for handle in scene.object_handles
            }
            _send(
                {
                    "op": "reset_ack",
                    "ack": {
                        "status": "ready",
                        "message": "UIPC scene ready",
                        "uipc_dt_s": cfg.dt_s,
                        "scene_hash": _scene_hash(cfg),
                        "n_pads": len(scene.pad_handles),
                        "n_objects": min(len(cfg.objects), protocol.MAX_OBJECTS),
                        "pads": scene.pad_surface_data(),
                        "seq": state.seq,
                        "object_z0": z_values,
                        "build_ms": (time.perf_counter() - started) * 1000.0,
                        "settle_ms": 0.0,
                    },
                }
            )
        except Exception as exc:
            self._close_scene()
            _send(
                {
                    "op": "reset_ack",
                    "ack": {
                        "status": "failed",
                        "message": f"UIPC scene build failed: {exc}",
                    },
                }
            )

    def _health(self) -> None:
        _send(
            {
                "op": "health_ack",
                "health": {
                    "alive": True,
                    "rc": None,
                    "scene": self.scene is not None,
                    "seq": self.last_state_seq,
                },
            }
        )

    def handle(self, msg: Mapping[str, Any]) -> bool:
        op = msg.get("op")
        if op == "reset":
            self._reset(msg)
            return True
        if op == "health":
            self._health()
            return True
        if op == "stop":
            self._close_scene()
            _send({"op": "stop_ack"})
            return False
        _send({"op": "error", "message": f"unknown op {op!r}"})
        return True

    def run(self) -> int:
        _send(
            {
                "op": "ready",
                "worker": "uipc-v1",
                "state_shm": self.args.state_shm,
                "finger_shm": self.args.finger_shm,
                "workspace": self.args.workspace,
                "state_slot_bytes": protocol.STATE_SLOT_SIZE,
                "finger_slot_bytes": protocol.FINGER_SLOT_SIZE,
            }
        )
        selector = selectors.DefaultSelector()
        selector.register(sys.stdin, selectors.EVENT_READ)
        finger_buf = self.finger_buf
        state_buf = self.state_buf
        try:
            while True:
                if selector.select(0.001):
                    line = sys.stdin.readline()
                    if not line:
                        break
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        _send({"op": "error", "message": "invalid json"})
                        continue
                    if not isinstance(raw, dict):
                        _send({"op": "error", "message": "message must be an object"})
                        continue
                    if not self.handle(cast(dict[str, Any], raw)):
                        break
                    continue
                if self.scene is None:
                    continue
                sample = protocol.read_fingers(finger_buf, self.last_finger_seq)
                if sample is None:
                    continue
                self.last_finger_seq = sample.seq
                try:
                    self.scene.step(sample)
                    state = self.scene.legacy_state()
                    protocol.write_state(state_buf, state)
                    self.last_state_seq = state.seq
                except Exception as exc:
                    _send({"op": "error", "message": f"UIPC step failed: {exc}"})
                    self._close_scene()
        finally:
            selector.close()
            self._close_scene()
            self.finger_shm.close()
            self.state_shm.close()
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-shm", required=True)
    ap.add_argument("--finger-shm", required=True)
    ap.add_argument("--workspace", default="/tmp/openvitac_uipc")
    ap.add_argument("--require-uipc", action="store_true")
    args = ap.parse_args()
    _redirect_control_stream()
    worker = Worker(args)
    return worker.run()


if __name__ == "__main__":
    raise SystemExit(main())
