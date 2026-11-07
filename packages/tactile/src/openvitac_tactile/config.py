"""Configuration dataclasses for the OpenViTac tactile runtime.

The field names are the public contract.  This module deliberately
imports no UIPC/Isaac code so the parent process can import it anywhere.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

type Side = Literal["left", "right"]
type SyncMode = Literal["lockstep", "bounded_lag"]
type BoxSpec = tuple[tuple[float, float, float], tuple[float, float, float]]
type FallbackMode = Literal["fail", "disable_tactile", "physx_only"]
type ForceKind = Literal["net_force", "contact_gradient", "none"]
type PadGeometryKind = Literal["box", "tet_mesh"]
type PadRenderKind = Literal["rgb", "marker_rgb", "depth"]


def _empty_env() -> dict[str, str]:
    return {}


@dataclass(frozen=True)
class PadGeometryCfg:
    kind: PadGeometryKind = "box"
    size_m: tuple[float, float, float] = (0.030, 0.004, 0.020)
    mesh_path: str | None = None
    tet_resolution: tuple[int, int, int] = (8, 1, 5)
    max_surface_vertices: int = 1024
    surface_decimation: int = 1

    def __post_init__(self) -> None:
        if self.kind == "tet_mesh" and self.mesh_path is None:
            raise ValueError("PadGeometryCfg.mesh_path is required for kind='tet_mesh'")
        if any(v <= 0.0 for v in self.size_m):
            raise ValueError("PadGeometryCfg.size_m must be positive")
        if any(v <= 0 for v in self.tet_resolution):
            raise ValueError("PadGeometryCfg.tet_resolution must be positive")
        if self.max_surface_vertices < 0:
            raise ValueError("PadGeometryCfg.max_surface_vertices must be >= 0")
        if self.surface_decimation < 1:
            raise ValueError("PadGeometryCfg.surface_decimation must be >= 1")


@dataclass(frozen=True)
class PadMaterialCfg:
    youngs_mpa: float = 0.1
    poisson: float = 0.49
    density_kg_m3: float = 1.0e4
    constraint_strength: float = 1.0e4
    contact_d_hat_m: float = 5.0e-4
    friction_rate: float = 2.5
    resistance_pa: float = 1.0e9

    def __post_init__(self) -> None:
        if self.youngs_mpa <= 0.0:
            raise ValueError("PadMaterialCfg.youngs_mpa must be positive")
        if not 0.0 <= self.poisson < 0.5:
            raise ValueError("PadMaterialCfg.poisson must be in [0, 0.5)")
        if self.density_kg_m3 <= 0.0:
            raise ValueError("PadMaterialCfg.density_kg_m3 must be positive")
        if self.constraint_strength <= 0.0:
            raise ValueError("PadMaterialCfg.constraint_strength must be positive")


@dataclass(frozen=True)
class PadFrameCfg:
    body_name: str
    body_prim_path: str
    pos: tuple[float, float, float]
    rot_xyzw: tuple[float, float, float, float]
    back_axis: Literal["x", "y", "z"] = "y"
    back_value_m: float = 0.0
    back_is_min: bool | None = None

    def __post_init__(self) -> None:
        if not self.body_name:
            raise ValueError("PadFrameCfg.body_name must not be empty")
        if not self.body_prim_path:
            raise ValueError("PadFrameCfg.body_prim_path must not be empty")
        if len(self.pos) != 3 or len(self.rot_xyzw) != 4:
            raise ValueError("PadFrameCfg.pos/rot_xyzw have wrong length")


@dataclass(frozen=True)
class PadCfg:
    side: Side
    name: str
    gelpad_prim_path: str
    frame: PadFrameCfg
    geometry: PadGeometryCfg
    material: PadMaterialCfg
    image_width: int | None = None
    image_height: int | None = None
    marker_rows: int | None = None
    marker_cols: int | None = None
    render_kinds: tuple[PadRenderKind, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("PadCfg.name must not be empty")
        if not self.gelpad_prim_path:
            raise ValueError("PadCfg.gelpad_prim_path must not be empty")
        if self.image_width is not None and self.image_width <= 0:
            raise ValueError("PadCfg.image_width must be positive")
        if self.image_height is not None and self.image_height <= 0:
            raise ValueError("PadCfg.image_height must be positive")


@dataclass(frozen=True)
class ObjectCfg:
    name: str
    prim_path: str
    authority: Literal["uipc", "physx"] = "uipc"
    constitution: Literal["abd", "fem"] = "abd"
    kappa_mpa: float = 100.0
    """ABD bulk stiffness (``constitution="abd"`` only)."""
    youngs_mpa: float | None = None
    """FEM Young's modulus; required for ``constitution="fem"``."""
    poisson: float | None = None
    """FEM Poisson's ratio; required for ``constitution="fem"``."""
    density_kg_m3: float = 1.0e3
    mesh_path: str | None = None
    usd_path: str | None = None
    """Source actor USD, kept for provenance; never a tetrahedral mesh."""
    fixed: bool = False
    contact_object: bool = False
    """Measure per-pad contact against this object (exactly one non-fixed UIPC)."""
    boxes: tuple[BoxSpec, ...] = ()
    max_speed_mps: float = 5.0
    max_penetration_mm: float = 0.25
    max_pose_jump_m: float = 0.05

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ObjectCfg.name must not be empty")
        if not self.prim_path:
            raise ValueError("ObjectCfg.prim_path must not be empty")
        if self.constitution not in ("abd", "fem"):
            raise ValueError(
                f"ObjectCfg.constitution must be 'abd' or 'fem', got "
                f"{self.constitution!r}"
            )
        if self.kappa_mpa <= 0.0:
            raise ValueError("ObjectCfg.kappa_mpa must be positive")
        if self.density_kg_m3 <= 0.0:
            raise ValueError("ObjectCfg.density_kg_m3 must be positive")
        if self.usd_path is not None and not self.usd_path:
            raise ValueError("ObjectCfg.usd_path must not be empty when set")
        if self.constitution == "fem":
            if self.youngs_mpa is None:
                raise ValueError(
                    f"ObjectCfg {self.name!r} with constitution='fem' requires "
                    "youngs_mpa"
                )
            if self.poisson is None:
                raise ValueError(
                    f"ObjectCfg {self.name!r} with constitution='fem' requires poisson"
                )
            if self.youngs_mpa <= 0.0:
                raise ValueError("ObjectCfg.youngs_mpa must be positive")
            if not 0.0 <= self.poisson < 0.5:
                raise ValueError("ObjectCfg.poisson must be in [0, 0.5)")
        elif self.youngs_mpa is not None or self.poisson is not None:
            raise ValueError(
                "ObjectCfg.youngs_mpa/poisson only apply to constitution='fem'"
            )
        if self.contact_object and self.fixed:
            raise ValueError("ObjectCfg.contact_object must be a non-fixed object")
        for center, half in self.boxes:
            if len(center) != 3 or len(half) != 3:
                raise ValueError("ObjectCfg.boxes entries must be 3-vectors")
            if any(value <= 0.0 for value in half):
                raise ValueError("ObjectCfg.boxes half extents must be positive")


@dataclass(frozen=True)
class WorkerCfg:
    python: str = sys.executable
    gpu: int | str | None = None
    workspace: str = "/tmp/openvitac_uipc"
    startup_timeout_s: float = 120.0
    reset_timeout_s: float = 180.0
    stop_timeout_s: float = 10.0
    env: Mapping[str, str] = field(default_factory=_empty_env)
    log_path: str | None = None

    def __post_init__(self) -> None:
        if not self.python:
            raise ValueError("WorkerCfg.python must not be empty")
        if self.startup_timeout_s <= 0.0 or self.reset_timeout_s <= 0.0:
            raise ValueError("WorkerCfg timeouts must be positive")
        if self.stop_timeout_s <= 0.0:
            raise ValueError("WorkerCfg.stop_timeout_s must be positive")


@dataclass(frozen=True)
class SyncCfg:
    mode: SyncMode = "bounded_lag"
    max_lag_steps: int = 2
    state_timeout_ms: float = 20.0
    max_stale_frames: int = 0
    max_finger_age_ms: float = 20.0
    publish_pad_vertices: bool = True
    max_pad_vertices: int = 1024
    finger_history: int = 4

    def __post_init__(self) -> None:
        if self.max_lag_steps < 0:
            raise ValueError("SyncCfg.max_lag_steps must be >= 0")
        if self.state_timeout_ms <= 0.0:
            raise ValueError("SyncCfg.state_timeout_ms must be positive")
        if self.max_stale_frames < 0:
            raise ValueError("SyncCfg.max_stale_frames must be >= 0")
        if self.max_finger_age_ms <= 0.0:
            raise ValueError("SyncCfg.max_finger_age_ms must be positive")
        if self.max_pad_vertices < 0:
            raise ValueError("SyncCfg.max_pad_vertices must be >= 0")
        if self.finger_history < 1:
            raise ValueError("SyncCfg.finger_history must be >= 1")


@dataclass(frozen=True)
class FallbackCfg:
    on_worker_abort: FallbackMode = "fail"
    on_timeout: FallbackMode = "fail"
    on_seq_gap: Literal["stale", "fail"] = "fail"
    on_object_diverge: FallbackMode = "fail"
    max_seq_gap: int = 1

    def __post_init__(self) -> None:
        if self.max_seq_gap < 0:
            raise ValueError("FallbackCfg.max_seq_gap must be >= 0")


@dataclass(frozen=True)
class ResetCfg:
    settle_max_steps: int = 0
    warmup_steps: int = 0

    def __post_init__(self) -> None:
        if self.settle_max_steps < 0 or self.warmup_steps < 0:
            raise ValueError("ResetCfg step counts must be >= 0")


@dataclass(frozen=True)
class RecordCfg:
    record_vertices: bool = True
    record_force: bool = True
    record_contact: bool = True
    record_rgb: bool = False
    record_depth: bool = False
    record_marker_rgb: bool = False


@dataclass(frozen=True)
class TactileCfg:
    enabled: bool = True
    pads: tuple[PadCfg, ...] = ()
    objects: tuple[ObjectCfg, ...] = ()
    dt_s: float = 1.0 / 120.0
    substeps: int = 1
    include_uipc_fingers: bool = False
    gravity_mps2: tuple[float, float, float] = (0.0, 0.0, -9.8)
    channel_prefix: str = "openvitac_tactile"
    worker: WorkerCfg = field(default_factory=WorkerCfg)
    sync: SyncCfg = field(default_factory=SyncCfg)
    fallback: FallbackCfg = field(default_factory=FallbackCfg)
    record: RecordCfg = field(default_factory=RecordCfg)
    reset: ResetCfg = field(default_factory=ResetCfg)
    contact_source: Literal["tactile", "physx"] = "tactile"
    object_authority: Literal["uipc", "physx"] = "uipc"

    def __post_init__(self) -> None:
        if self.dt_s <= 0.0:
            raise ValueError("TactileCfg.dt_s must be positive")
        if self.substeps != 1:
            raise ValueError("TactileCfg.substeps must be 1 in v1")
        if not self.channel_prefix:
            raise ValueError("TactileCfg.channel_prefix must not be empty")
        for pad in self.pads:
            if self.sync.max_pad_vertices > pad.geometry.max_surface_vertices:
                raise ValueError(
                    "SyncCfg.max_pad_vertices exceeds "
                    f"{pad.name}.geometry.max_surface_vertices"
                )

    def validate(self) -> None:
        """Runtime-level validation (strict enabled/disabled semantics)."""
        if not self.enabled:
            return
        if not self.pads:
            raise ValueError("TactileCfg.enabled=True requires at least one pad")
        names = [p.name for p in self.pads]
        if len(names) != len(set(names)):
            raise ValueError("TactileCfg.pads names must be unique")
        sides = [p.side for p in self.pads]
        if len(sides) != len(set(sides)):
            raise ValueError("TactileCfg.pads sides must be unique")
